"""Convert MUD JSON ACEs into compact behavioral text.

Each output line is one ACE-like behavioral primitive. The format keeps the
fields used our the paper: direction, controller/local hints, IP version,
transport protocol, endpoint, and port semantics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

PROTO_NAMES = {2: "igmp", 6: "tcp", 17: "udp", 58: "icmpv6"}


def proto_name(value: object) -> str:
    """Return a readable protocol label for an IP protocol number."""
    try:
        return PROTO_NAMES.get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value)


def iter_aces(doc: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield ``(acl_name, ace)`` pairs from common MUD ACL layouts."""
    root = (
        doc.get("ietf-access-control-list:access-lists")
        or doc.get("access-lists")
        or doc.get("ietf-access-control-list:acls")
        or doc.get("acls")
        or {}
    )
    for acl in root.get("acl") or []:
        acl_name = acl.get("name") or ""
        for ace in (acl.get("aces") or {}).get("ace") or []:
            if isinstance(ace, dict):
                yield acl_name, ace


def _append_ip_fields(
    *,
    parts: list[str],
    matches: dict[str, Any],
    ip_key: str,
    source_network_key: str,
    destination_network_key: str,
) -> bool:
    ip_block = matches.get(ip_key) or {}
    if not ip_block:
        return False

    parts.append(ip_key)
    if "protocol" in ip_block:
        parts.append(proto_name(ip_block["protocol"]))

    source_dns = ip_block.get("ietf-acldns:src-dnsname")
    destination_dns = ip_block.get("ietf-acldns:dst-dnsname")
    if source_dns:
        parts.append(f"src:{source_dns}")
    elif source_network_key in ip_block:
        parts.append(f"src:{ip_block[source_network_key]}")

    if destination_dns:
        parts.append(f"dst:{destination_dns}")
    elif destination_network_key in ip_block:
        parts.append(f"dst:{ip_block[destination_network_key]}")

    return True


def _append_transport_fields(parts: list[str], matches: dict[str, Any], name: str) -> None:
    layer = matches.get(name) or {}
    if not layer:
        return

    direction = layer.get("ietf-mud:direction-initiated")
    if direction:
        for idx, part in enumerate(parts):
            if part == name:
                parts[idx] = f"{name} (direction-initiated:{direction})"
                break

    source_port = layer.get("source-port")
    if isinstance(source_port, dict) and "port" in source_port:
        parts.append(f"src-port:{source_port['port']}")

    destination_port = layer.get("destination-port")
    if isinstance(destination_port, dict) and "port" in destination_port:
        parts.append(f"dst-port:{destination_port['port']}")


def summarize_ace(acl_name: str, ace: dict[str, Any]) -> str:
    """Return one compact text line for an ACE."""
    matches = ace.get("matches") or {}
    parts: list[str] = []

    if acl_name.startswith("from-"):
        parts.append("egress")
    elif acl_name.startswith("to-"):
        parts.append("ingress")

    mud_ext = matches.get("ietf-mud:mud") or {}
    if mud_ext.get("controller"):
        parts.append(f"controller:{str(mud_ext['controller']).split(':')[-1]}")
    if "local-networks" in mud_ext:
        parts.append("local")

    eth = matches.get("eth") or {}
    if "ethertype" in eth:
        parts.append(f"eth:{eth['ethertype']}")
    if "destination-mac-address" in eth:
        parts.append(f"mac:{eth['destination-mac-address']}")

    _append_ip_fields(
        parts=parts,
        matches=matches,
        ip_key="ipv4",
        source_network_key="source-ipv4-network",
        destination_network_key="destination-ipv4-network",
    )
    _append_ip_fields(
        parts=parts,
        matches=matches,
        ip_key="ipv6",
        source_network_key="source-ipv6-network",
        destination_network_key="destination-ipv6-network",
    )
    _append_transport_fields(parts, matches, "tcp")
    _append_transport_fields(parts, matches, "udp")

    return " ".join(parts)


def compact_file(path: Path) -> list[str]:
    """Read a MUD JSON file and return ordered, de-duplicated compact lines."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    lines = [summarize_ace(acl_name, ace) for acl_name, ace in iter_aces(doc)]
    return list(dict.fromkeys(line for line in lines if line))


def compact_directory(input_dir: Path, output_dir: Path) -> dict[str, object]:
    """Compact all JSON files in ``input_dir`` into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    total_original_size = 0
    total_flattened_size = 0

    for json_path in sorted(input_dir.rglob("*.json")):
        lines = compact_file(json_path)
        relative = json_path.relative_to(input_dir)
        output_path = output_dir / relative.parent / f"{json_path.stem}_compact.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")

        original_size = len(json_path.read_text(encoding="utf-8"))
        flattened_size = len(output_path.read_text(encoding="utf-8"))
        total_original_size += original_size
        total_flattened_size += flattened_size
        rows.append(
            {
                "filename": str(relative),
                "original_size": original_size,
                "flattened_size": flattened_size,
                "reduction_bytes": original_size - flattened_size,
                "reduction_percent": round((1 - flattened_size / original_size) * 100, 2)
                if original_size
                else 0.0,
            }
        )

    summary = {
        "devices": rows,
        "summary": {
            "file_count": len(rows),
            "total_original_size": total_original_size,
            "total_flattened_size": total_flattened_size,
            "total_reduction_bytes": total_original_size - total_flattened_size,
            "total_reduction_percent": round(
                (1 - total_flattened_size / total_original_size) * 100, 2
            )
            if total_original_size
            else 0.0,
        },
    }
    (output_dir / "reduction_stats.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/mud/mud_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mud/mud_compact"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = compact_directory(args.input_dir, args.output_dir)
    print(
        f"Compacted {summary['summary']['file_count']} files into {args.output_dir} "
        f"({summary['summary']['total_reduction_percent']}% byte reduction)."
    )


if __name__ == "__main__":
    main()
