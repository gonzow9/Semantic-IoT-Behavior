"""Load the real-traffic CSV files and extract exact-match features.

This module handles the inputs of the real traffic evaluation:

- the 26 runtime trace CSV files in ``data/runtime_aces/``, and
- the exact-match "service features" used by the exact and Jaccard methods.

A service feature is a comparable ``(protocol, endpoint, port)`` tuple.
Shared infrastructure behavior (DNS, DHCP, gateway, broadcast) also gives a
wildcard feature so that common flows create ties across devices instead of
arbitrary wins.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RUNTIME_DIR = REPO_ROOT / "data" / "runtime_aces"
DEFAULT_COMPACT_DIR = REPO_ROOT / "data" / "ref_mud" / "compact"

# Ground truth: trace device name (from the CSV filename) -> reference
# profile stem in the embedding bank and compact directory.
DEVICE_TO_MUD = {
    "AmazonEcho": "amazonEchoMud_compact",
    "AugustDoorBell": "augustdoorbellcamMud_compact",
    "AwairAirQuality": "awairAirQualityMud_compact",
    "BelkinCamera": "belkincameraMud_compact",
    "BelkinWemoMotionSensor": "wemomotionMud_compact",
    "BelkinWemoSwitch": "wemoswitchMud_compact",
    "BlipCareBPMeter": "blipcareBPmeterMud_compact",
    "CanaryCamera": "canaryCameraMud_compact",
    "HPPrinter": "hpprinterMud_compact",
    "HelloBarbie": "hellobarbieMud_compact",
    "LiFXBulb": "lifxbulbMud_compact",
    "NestDropCam": "dropcamMud_compact",
    "NestProtect": "nestsmokesensorMud_compact",
    "NetatmoWeatherStation": "NetatmoWeatherStationMud_compact",
    "NetatmoWelcome": "NetatmoCameraMud_compact",
    "PhilipsHue": "HueBulbMud_compact",
    "PixStarPhotoFrame": "pixstarphotoframeMud_compact",
    "RingDoorBell": "ringdoorbellMud_compact",
    "SamsungCamera": "samsungsmartcamMud_compact",
    "SamsungSmartThings": "SmartThingsMud_compact",
    "TPLinkCamera": "tplinkcameraMud_compact",
    "TPLinkSmartPlug": "tplinkplugMud_compact",
    "TribySpeaker": "tribyspeakerMud_compact",
    "WithingsBabyMonitor": "withingsbabymonitorMud_compact",
    "WithingsSleepSensor": "withingssleepsensorMud_compact",
    "iHome": "ihomepowerplugMud_compact",
}

# The original evaluation covered 27 traces (including WithingsSmartScale,
# which has no reference profile and is not shipped here) in sorted order to
# seed the per-device window sampling. Keep the same enumeration so the
# sampled windows are identical.
SEED_ORDER_EXTRA = ("WithingsSmartScale",)


# ---------------------------------------------------------------------------
# Exact service features.
# ---------------------------------------------------------------------------

PROTO_RE = re.compile(r"\b(tcp|udp)\b", re.IGNORECASE)
ENDPOINT_RE = re.compile(r"\b(?:dst|src):(\S+)")
PORT_RE = re.compile(r"\b(?:dst-port|src-port):(\d+)")

Feature = tuple[str, str, int]


def normalise_endpoint(value: str) -> str:
    endpoint = value.strip().rstrip(".")
    if endpoint.endswith("/32"):
        endpoint = endpoint[:-3]
    return endpoint.lower()


def _service_features(text: str, proto: str, port: int) -> set[Feature]:
    lowered = text.lower()
    features: set[Feature] = set()
    if "controller:dns" in lowered or (proto in {"tcp", "udp"} and port == 53):
        features.add((proto, "service:dns", 53))
    if "controller:gateway" in lowered:
        features.add((proto, "service:gateway", port))
    if proto == "udp" and port in {67, 68}:
        features.add(("udp", "service:dhcp", 67))
    if "255.255.255.255" in lowered and proto == "udp":
        features.add(("udp", "service:broadcast", port))
    return features


def features_from_ace_text(text: str) -> set[Feature]:
    """Features for one compact reference ACE line."""
    proto_match = PROTO_RE.search(text)
    port_match = PORT_RE.search(text)
    if not proto_match or not port_match:
        return set()
    proto = proto_match.group(1).lower()
    port = int(port_match.group(1))
    features = _service_features(text, proto, port)
    endpoint_match = ENDPOINT_RE.search(text)
    if endpoint_match:
        endpoint = normalise_endpoint(endpoint_match.group(1))
        if endpoint and not endpoint.startswith("controller:"):
            features.add((proto, endpoint, port))
    return features


def features_from_flow(
    text: str,
    ip_protocol: int,
    remote_ip: str,
    remote_port: int,
) -> set[Feature]:
    """Features for one runtime flow row."""
    if ip_protocol == 6:
        proto = "tcp"
    elif ip_protocol == 17:
        proto = "udp"
    else:
        proto = f"proto{int(ip_protocol)}"
    port = int(remote_port)
    features = _service_features(text, proto, port)
    endpoint_match = ENDPOINT_RE.search(text)
    endpoint = (
        normalise_endpoint(endpoint_match.group(1))
        if endpoint_match
        else normalise_endpoint(remote_ip)
    )
    if endpoint and not endpoint.startswith("controller:"):
        features.add((proto, endpoint, port))
    return features


def load_reference_features(compact_dir: Path) -> dict[str, set[Feature]]:
    device_features: dict[str, set[Feature]] = {}
    for path in sorted(compact_dir.glob("*.txt")):
        features: set[Feature] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            features.update(features_from_ace_text(line))
        device_features[path.stem] = features
    if not device_features:
        raise FileNotFoundError(f"No compact profiles in {compact_dir}")
    return device_features


# ---------------------------------------------------------------------------
# Traces.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trace:
    device_name: str
    gt_mud: str
    texts: tuple[str, ...]  # one runtime ACE text per flow, temporal order
    features: tuple[frozenset, ...]  # exact features per flow


def read_traces(runtime_dir: Path) -> list[Trace]:
    traces = []
    for path in sorted(runtime_dir.glob("*_runtime_aces.csv")):
        device_name = path.name.split("_")[0]
        gt_mud = DEVICE_TO_MUD.get(device_name)
        if gt_mud is None:
            raise ValueError(f"No reference profile mapping for {device_name}")
        texts: list[str] = []
        features: list[frozenset] = []
        feature_cache: dict[str, frozenset] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for idx, row in enumerate(csv.DictReader(handle), start=1):
                if int(row["flow_idx"]) != idx:
                    raise ValueError(f"Non-consecutive flow_idx in {path.name}")
                text = row["runtime_ace"]
                texts.append(text)
                cached = feature_cache.get(text)
                if cached is None:
                    cached = frozenset(
                        features_from_flow(
                            text,
                            int(row["ip_protocol"]),
                            row["remote_ip"],
                            int(row["remote_port"]),
                        )
                    )
                    feature_cache[text] = cached
                features.append(cached)
        if not texts:
            raise ValueError(f"Empty trace: {path.name}")
        traces.append(
            Trace(
                device_name=device_name,
                gt_mud=gt_mud,
                texts=tuple(texts),
                features=tuple(features),
            )
        )
    if not traces:
        raise FileNotFoundError(f"No runtime CSV files in {runtime_dir}")
    return traces


def seed_index(traces: list[Trace]) -> dict[str, int]:
    """Per-device index in the original 27-trace sorted enumeration."""
    names = sorted({t.device_name for t in traces} | set(SEED_ORDER_EXTRA))
    return {name: idx for idx, name in enumerate(names)}
