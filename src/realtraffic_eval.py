#!/usr/bin/env python3
"""Reproduce the paper's real traffic evaluation from the shipped CSV files.

The 26 files in ``data/runtime_aces/`` hold real IoT traffic flows, one row
per flow, already converted to compact ACE text (the ``runtime_ace`` column).
This script matches those flows against the 28 reference MUD profiles with
four methods and reproduces the paper's two experiments:

  cumulative   Identification as flows accumulate in temporal order,
               including the rank distribution over the first 10,000 flows.
  windows      Identification from disjoint 50-flow windows (9,023 windows
               across 25 devices), binned by exact-overlap score.

Both experiments need per-flow embeddings. Build them once with:

  embed        Embed every distinct runtime ACE text with BGE-M3, whiten
               with the transform fitted on the raw reference bank, and save
               a local runtime embedding bank.

Matching methods:

- Exact hit count: a flow scores 1 for a device when it shares a service
  feature (protocol, endpoint, port; wildcard DNS/DHCP/gateway/broadcast
  services) with the device's profile. Scores accumulate over repeats.
- Jaccard: feature-set overlap between the unique observed features and each
  profile's features.
- Mean Pool: cosine between the normalized mean of the observed whitened
  ACE embeddings (repeats kept) and each profile's mean-pooled signature.
- MaxSim: each flow scores its best cosine against each profile's ACEs;
  scores accumulate over repeats.

A method only predicts a device when a single candidate holds the top score;
tied leaders count as not identified.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gen_whiten_emb import apply_whitening, fit_whitening

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME_DIR = REPO_ROOT / "data" / "runtime_aces"
DEFAULT_COMPACT_DIR = REPO_ROOT / "data" / "ref_mud" / "compact"
DEFAULT_RAW_NPZ = (
    REPO_ROOT / "data" / "ref_embeddings" / "bge" / "per_ace" / "raw"
    / "reference_per_ace.npz"
)
DEFAULT_RUNTIME_NPZ = (
    REPO_ROOT / "data" / "runtime_embeddings"
    / "runtime_texts_whitened_k256.npz"
)

# Tie thresholds, matching the original evaluation code.
CUMULATIVE_TIE_EPSILON = 1e-6
WINDOW_TIE_EPSILON = 1e-9
RANK_TIE_EPSILON = 1e-9

WHITEN_K = 256

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

# The original evaluation enumerated 27 traces (including WithingsSmartScale,
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


# ---------------------------------------------------------------------------
# Embeddings.
# ---------------------------------------------------------------------------


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return values / norms


def load_reference_bank(raw_npz: Path):
    """Whiten the raw reference bank locally; keep the fitted transform."""
    data = np.load(raw_npz, allow_pickle=True)
    raw = data["embeddings"].astype(np.float32, copy=False)
    mean, components, singular_values = fit_whitening(raw)
    whitened = l2_normalise(
        apply_whitening(raw, mean, components, singular_values, WHITEN_K, raw.shape[0])
    )
    devices = [str(value) for value in data["devices"]]
    ace_texts = [str(value) for value in data["ace_texts"]]
    transform = {
        "mean": mean,
        "components": components,
        "singular_values": singular_values,
        "n_reference": raw.shape[0],
    }
    return whitened, devices, ace_texts, transform


def run_embed(args: argparse.Namespace) -> None:
    from gen_emb import build_model, encode_texts

    traces = read_traces(args.runtime_dir)
    distinct = sorted({text for trace in traces for text in trace.texts})
    whitened_ref, _, ref_texts, transform = load_reference_bank(args.raw_npz)

    vector_by_text: dict[str, np.ndarray] = {}
    for idx, text in enumerate(ref_texts):
        vector_by_text.setdefault(text, whitened_ref[idx])
    reused = [text for text in distinct if text in vector_by_text]
    new_texts = [text for text in distinct if text not in vector_by_text]

    print(f"{len(distinct)} distinct runtime ACE texts from {len(traces)} traces.")
    print(f"{len(reused)} match a reference ACE; embedding {len(new_texts)} new texts...")
    model = build_model(args.model_name, args.device)
    raw = encode_texts(model, new_texts, args.batch_size)
    whitened_new = l2_normalise(
        apply_whitening(
            raw,
            transform["mean"],
            transform["components"],
            transform["singular_values"],
            WHITEN_K,
            transform["n_reference"],
        )
    )
    vector_by_text.update(zip(new_texts, whitened_new))

    embeddings = np.vstack([vector_by_text[text] for text in distinct])
    args.runtime_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.runtime_npz,
        texts=np.asarray(distinct, dtype=object),
        embeddings=embeddings.astype(np.float32),
        model_name=np.asarray(args.model_name),
        whiten_k=np.asarray(WHITEN_K),
    )
    print(f"Saved {args.runtime_npz} ({embeddings.shape[0]}x{embeddings.shape[1]}).")


def load_runtime_bank(runtime_npz: Path) -> dict[str, np.ndarray]:
    if not runtime_npz.exists():
        raise FileNotFoundError(
            f"Runtime embedding bank not found: {runtime_npz}\n"
            "Build it first: python src/realtraffic_eval.py embed"
        )
    data = np.load(runtime_npz, allow_pickle=True)
    texts = [str(value) for value in data["texts"]]
    embeddings = l2_normalise(data["embeddings"])
    return dict(zip(texts, embeddings))


# ---------------------------------------------------------------------------
# Scoring.
# ---------------------------------------------------------------------------


@dataclass
class ScoreTables:
    devices: list[str]  # 28 reference stems, sorted
    ref_features: list[set]  # feature set per device
    ref_signatures: np.ndarray  # (28, k) mean-pooled signatures
    # Per unique text within one trace:
    maxsim: np.ndarray  # (U, 28) best cosine per device
    exact: np.ndarray  # (U, 28) binary feature hit per device
    vectors: np.ndarray  # (U, k) whitened flow embedding
    text_index: dict[str, int]


def build_score_tables(
    trace: Trace,
    devices: list[str],
    ref_features: dict[str, set],
    ref_rows: dict[str, np.ndarray],
    ref_signatures: np.ndarray,
    vector_by_text: dict[str, np.ndarray],
) -> ScoreTables:
    order: dict[str, int] = {}
    first_features: list[frozenset] = []
    for text, feats in zip(trace.texts, trace.features):
        if text not in order:
            order[text] = len(order)
            first_features.append(feats)
    missing = [text for text in order if text not in vector_by_text]
    if missing:
        raise ValueError(
            f"{len(missing)} runtime texts missing from the embedding bank "
            f"(first: {missing[0]!r}). Re-run the embed step."
        )
    vectors = np.vstack([vector_by_text[text] for text in order])
    maxsim = np.empty((len(order), len(devices)), dtype=np.float32)
    for col, device in enumerate(devices):
        maxsim[:, col] = (vectors @ ref_rows[device].T).max(axis=1)
    exact = np.empty((len(order), len(devices)), dtype=np.float32)
    for row, feats in enumerate(first_features):
        for col, device in enumerate(devices):
            exact[row, col] = 1.0 if feats & ref_features[device] else 0.0
    return ScoreTables(
        devices=devices,
        ref_features=[ref_features[d] for d in devices],
        ref_signatures=ref_signatures,
        maxsim=maxsim,
        exact=exact,
        vectors=vectors,
        text_index=order,
    )


def unique_leader_correct(
    scores: np.ndarray, gt_col: int, epsilon: float
) -> np.ndarray:
    """True where the ground-truth device is the single leader."""
    top = scores.max(axis=1)
    n_tied = (scores >= top[:, None] - epsilon).sum(axis=1)
    return (n_tied == 1) & (scores.argmax(axis=1) == gt_col)


def conservative_rank(scores: np.ndarray, gt_col: int, epsilon: float) -> np.ndarray:
    gt = scores[:, gt_col][:, None]
    better = (scores > gt + epsilon).sum(axis=1)
    tied = (np.abs(scores - gt) <= epsilon).sum(axis=1)
    return better + tied


def cumulative_jaccard(trace: Trace, tables: ScoreTables) -> np.ndarray:
    """(n_flows, 28) Jaccard between observed unique features and each profile."""
    n_dev = len(tables.devices)
    sizes = np.asarray([len(f) for f in tables.ref_features], dtype=np.float64)
    inter = np.zeros(n_dev, dtype=np.float64)
    seen: set = set()
    out = np.empty((len(trace.texts), n_dev), dtype=np.float32)
    for row, feats in enumerate(trace.features):
        for feature in feats:
            if feature not in seen:
                seen.add(feature)
                for col, ref in enumerate(tables.ref_features):
                    if feature in ref:
                        inter[col] += 1.0
        union = len(seen) + sizes - inter
        with np.errstate(invalid="ignore"):
            out[row] = np.where(union > 0, inter / union, 0.0)
    return out


def per_flow_matrix(trace: Trace, table: np.ndarray, text_index: dict[str, int]) -> np.ndarray:
    rows = np.asarray([text_index[text] for text in trace.texts], dtype=np.int64)
    return table[rows]


def cumulative_scores(trace: Trace, tables: ScoreTables) -> dict[str, np.ndarray]:
    """Per-flow cumulative score matrices for the four methods."""
    exact_flow = per_flow_matrix(trace, tables.exact, tables.text_index)
    maxsim_flow = per_flow_matrix(trace, tables.maxsim, tables.text_index)
    vec_flow = per_flow_matrix(trace, tables.vectors, tables.text_index)
    pooled = l2_normalise(np.cumsum(vec_flow, axis=0, dtype=np.float64))
    return {
        "exact_hit_count": np.cumsum(exact_flow, axis=0, dtype=np.float64),
        "jaccard": cumulative_jaccard(trace, tables),
        "mean_pool": pooled @ tables.ref_signatures.T,
        "maxsim": np.cumsum(maxsim_flow, axis=0, dtype=np.float64),
    }


METHOD_ORDER = ("jaccard", "exact_hit_count", "mean_pool", "maxsim")
METHOD_LABELS = {
    "jaccard": "Jaccard",
    "exact_hit_count": "Exact hit count",
    "mean_pool": "Mean Pool",
    "maxsim": "MaxSim",
}


def prepare(args: argparse.Namespace):
    traces = read_traces(args.runtime_dir)
    ref_features = load_reference_features(args.compact_dir)
    whitened_ref, ref_devices, ref_texts, _ = load_reference_bank(args.raw_npz)
    devices = sorted(set(ref_devices))
    ref_rows = {
        device: whitened_ref[[i for i, d in enumerate(ref_devices) if d == device]]
        for device in devices
    }
    ref_signatures = l2_normalise(
        np.vstack(
            [ref_rows[device].mean(axis=0, dtype=np.float64) for device in devices]
        )
    )
    vector_by_text = load_runtime_bank(args.runtime_npz)
    for idx, text in enumerate(ref_texts):
        vector_by_text.setdefault(text, whitened_ref[idx])
    total_flows = sum(len(t.texts) for t in traces)
    print(
        f"{len(traces)} traces, {total_flows} flows, "
        f"{len(devices)} candidate profiles."
    )
    return traces, devices, ref_features, ref_rows, ref_signatures, vector_by_text


def run_cumulative(args: argparse.Namespace) -> None:
    traces, devices, ref_features, ref_rows, ref_signatures, vectors = prepare(args)
    checkpoints = [1, 2, 3, 5, 10, 20, 50, 100, 500, 1000, 5000, 10000, 50000]
    correct_at = {m: {} for m in METHOD_ORDER}
    final_correct = {m: 0 for m in METHOD_ORDER}
    active_at = {}
    rank_counts = {m: np.zeros(len(devices) + 1, dtype=np.int64) for m in METHOD_ORDER}
    rank_queries = 0
    convergence_rows = []

    for trace in traces:
        tables = build_score_tables(
            trace, devices, ref_features, ref_rows, ref_signatures, vectors
        )
        gt_col = devices.index(trace.gt_mud)
        cumulative = cumulative_scores(trace, tables)
        n = len(trace.texts)
        cap = min(n, 10000)
        rank_queries += cap
        for method, scores in cumulative.items():
            correct = unique_leader_correct(scores, gt_col, CUMULATIVE_TIE_EPSILON)
            ranks = conservative_rank(scores[:cap], gt_col, RANK_TIE_EPSILON)
            counts = np.bincount(
                np.minimum(ranks, len(devices)), minlength=len(devices) + 1
            )
            rank_counts[method] += counts
            final_correct[method] += int(correct[n - 1])
            for k in checkpoints:
                if k <= n:
                    correct_at[method].setdefault(k, 0)
                    correct_at[method][k] += int(correct[k - 1])
            if args.full_curve_output:
                for i in range(n):
                    convergence_rows.append(
                        (method, trace.device_name, i + 1, int(correct[i]))
                    )
        for k in checkpoints:
            if k <= n:
                active_at[k] = active_at.get(k, 0) + 1

    print("\nTop-1 correct traces as flows accumulate (correct/active):")
    header = "flows".rjust(7) + "".join(m.rjust(18) for m in METHOD_ORDER)
    print(header)
    for k in checkpoints:
        if k not in active_at:
            continue
        row = f"{k:7d}"
        for method in METHOD_ORDER:
            row += f"{correct_at[method][k]:>13d}/{active_at[k]:<4d}"
        print(row)
    row = "  final"
    for method in METHOD_ORDER:
        row += f"{final_correct[method]:>13d}/{len(traces):<4d}"
    print(row + "   (each trace at its last flow)")

    print("\nRank of the correct device over the first 10,000 flows per trace:")
    print("method             top1    top3    top5")
    summary_rank = {}
    for method in METHOD_ORDER:
        counts = rank_counts[method]
        top1 = counts[1] / rank_queries
        top3 = counts[1:4].sum() / rank_queries
        top5 = counts[1:6].sum() / rank_queries
        summary_rank[method] = {"top1": top1, "top3": top3, "top5": top5}
        print(f"{method:<17s}{top1:8.4f}{top3:8.4f}{top5:8.4f}")

    if args.output:
        payload = {
            "checkpoints": {
                str(k): {
                    "active": active_at[k],
                    **{m: correct_at[m][k] for m in METHOD_ORDER},
                }
                for k in checkpoints
                if k in active_at
            },
            "final": {
                "active": len(traces),
                **{m: final_correct[m] for m in METHOD_ORDER},
            },
            "rank_distribution": {
                "queries": rank_queries,
                **{m: summary_rank[m] for m in METHOD_ORDER},
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"Saved {args.output}.")
    if args.full_curve_output:
        args.full_curve_output.parent.mkdir(parents=True, exist_ok=True)
        with args.full_curve_output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["method", "device_name", "flow_idx", "correct"])
            writer.writerows(convergence_rows)
        print(f"Saved {args.full_curve_output}.")


def window_starts(
    n_flows: int, window_size: int, windows_per_device: int, seed: int
) -> np.ndarray:
    """Disjoint window starts, identical to the original sampling."""
    max_windows = n_flows // window_size
    if max_windows <= 0:
        return np.asarray([], dtype=int)
    rng = np.random.default_rng(seed)
    leftover = n_flows - max_windows * window_size
    offset = int(rng.integers(0, leftover + 1)) if leftover else 0
    starts = offset + np.arange(max_windows, dtype=int) * window_size
    if len(starts) > windows_per_device:
        starts = rng.choice(starts, size=windows_per_device, replace=False)
    return np.sort(starts.astype(int))


def run_windows(args: argparse.Namespace) -> None:
    traces, devices, ref_features, ref_rows, ref_signatures, vectors = prepare(args)
    seed_by_device = seed_index(traces)
    rows = []
    for trace in traces:
        tables = build_score_tables(
            trace, devices, ref_features, ref_rows, ref_signatures, vectors
        )
        gt_col = devices.index(trace.gt_mud)
        starts = window_starts(
            len(trace.texts),
            args.window_size,
            args.windows_per_device,
            args.seed + seed_by_device[trace.device_name] * 9973,
        )
        if len(starts) == 0:
            print(
                f"Skipping {trace.device_name}: fewer than "
                f"{args.window_size} flows."
            )
            continue
        exact_flow = per_flow_matrix(trace, tables.exact, tables.text_index)
        maxsim_flow = per_flow_matrix(trace, tables.maxsim, tables.text_index)
        vec_flow = per_flow_matrix(trace, tables.vectors, tables.text_index)
        for start in starts:
            end = int(start + args.window_size)
            window_scores = {
                "exact_hit_count": exact_flow[start:end].sum(axis=0) / args.window_size,
                "maxsim": maxsim_flow[start:end].sum(axis=0) / args.window_size,
                "mean_pool": l2_normalise(
                    vec_flow[start:end].sum(axis=0, dtype=np.float64)[None, :]
                )[0]
                @ ref_signatures.T,
            }
            window_features: set = set()
            for feats in trace.features[start:end]:
                window_features.update(feats)
            window_scores["jaccard"] = np.asarray(
                [
                    len(window_features & ref) / len(window_features | ref)
                    if window_features | ref
                    else 0.0
                    for ref in tables.ref_features
                ]
            )
            record = {"device": trace.device_name}
            for method, scores in window_scores.items():
                matrix = scores[None, :]
                record[f"{method}_correct"] = bool(
                    unique_leader_correct(matrix, gt_col, WINDOW_TIE_EPSILON)[0]
                )
            record["exact_top_score"] = float(window_scores["exact_hit_count"].max())
            record["exact_gt_score"] = float(window_scores["exact_hit_count"][gt_col])
            rows.append(record)

    subsets = [
        ("All windows", lambda r: True),
        ("Exact-overlap < 0.50", lambda r: r["exact_top_score"] < 0.50),
        ("Exact-overlap < 0.10", lambda r: r["exact_gt_score"] < 0.10),
        ("Exact-overlap = 0", lambda r: r["exact_gt_score"] <= WINDOW_TIE_EPSILON),
    ]
    print(f"\n{len(rows)} windows across {len({r['device'] for r in rows})} devices.")
    print(f"{'Window set':<22s}{'Windows':>9s}{'Devices':>9s}"
          f"{'Exact':>9s}{'MeanPool':>10s}{'MaxSim':>9s}")
    table_payload = []
    for label, keep in subsets:
        subset = [r for r in rows if keep(r)]
        n_dev = len({r["device"] for r in subset})
        counts = {
            m: sum(r[f"{m}_correct"] for r in subset)
            for m in ("exact_hit_count", "mean_pool", "maxsim")
        }
        print(
            f"{label:<22s}{len(subset):>9d}{n_dev:>9d}"
            f"{counts['exact_hit_count']:>9d}{counts['mean_pool']:>10d}"
            f"{counts['maxsim']:>9d}"
        )
        table_payload.append(
            {"window_set": label, "windows": len(subset), "devices": n_dev, **counts}
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(table_payload, indent=2))
        print(f"Saved {args.output}.")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--compact-dir", type=Path, default=DEFAULT_COMPACT_DIR)
    parser.add_argument("--raw-npz", type=Path, default=DEFAULT_RAW_NPZ,
                        help="Raw per-ACE reference bank used for whitening.")
    parser.add_argument("--runtime-npz", type=Path, default=DEFAULT_RUNTIME_NPZ,
                        help="Runtime embedding bank (written by 'embed').")
    sub = parser.add_subparsers(dest="command", required=True)

    embed = sub.add_parser("embed", help="Embed the distinct runtime ACE texts.")
    embed.add_argument("--model-name", default="BAAI/bge-m3")
    embed.add_argument("--device", default=None)
    embed.add_argument("--batch-size", type=int, default=32)

    cumulative = sub.add_parser("cumulative", help="Evidence-accumulation evaluation.")
    cumulative.add_argument("--output", type=Path, default=None)
    cumulative.add_argument("--full-curve-output", type=Path, default=None,
                            help="Optional per-flow correctness CSV (large).")

    windows = sub.add_parser("windows", help="Short-window evaluation.")
    windows.add_argument("--window-size", type=int, default=50)
    windows.add_argument("--windows-per-device", type=int, default=500)
    windows.add_argument("--seed", type=int, default=1729)
    windows.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "embed":
        run_embed(args)
    elif args.command == "cumulative":
        run_cumulative(args)
    else:
        run_windows(args)


if __name__ == "__main__":
    main()
