"""Score runtime traces against the reference MUD profiles.

This module holds the embedding banks and the four matching methods used by
the real traffic evaluation:

- Exact hit count: a flow scores 1 for a device when it shares a service
  feature with the device's profile. Scores accumulate over repeats.
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

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gen_whiten_emb import apply_whitening, fit_whitening
from realtraffic_data import Trace, load_reference_features, read_traces

REPO_ROOT = Path(__file__).resolve().parent.parent
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

METHOD_ORDER = ("jaccard", "exact_hit_count", "mean_pool", "maxsim")


# ---------------------------------------------------------------------------
# Embedding banks.
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


def prepare(args):
    """Load traces, reference features, embedding banks, and signatures."""
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


# ---------------------------------------------------------------------------
# Per-trace score tables.
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


def per_flow_matrix(
    trace: Trace, table: np.ndarray, text_index: dict[str, int]
) -> np.ndarray:
    rows = np.asarray([text_index[text] for text in trace.texts], dtype=np.int64)
    return table[rows]


# ---------------------------------------------------------------------------
# Decisions and ranks.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Cumulative scores.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Window scores.
# ---------------------------------------------------------------------------


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


def window_score_vectors(
    trace: Trace,
    tables: ScoreTables,
    exact_flow: np.ndarray,
    maxsim_flow: np.ndarray,
    vec_flow: np.ndarray,
    start: int,
    end: int,
) -> dict[str, np.ndarray]:
    """Per-device scores for one flow window, for the four methods.

    The three per-flow matrices are computed once per trace with
    ``per_flow_matrix`` and shared across that trace's windows.
    """
    window_size = end - start
    scores = {
        "exact_hit_count": exact_flow[start:end].sum(axis=0) / window_size,
        "maxsim": maxsim_flow[start:end].sum(axis=0) / window_size,
        "mean_pool": l2_normalise(
            vec_flow[start:end].sum(axis=0, dtype=np.float64)[None, :]
        )[0]
        @ tables.ref_signatures.T,
    }
    window_features: set = set()
    for feats in trace.features[start:end]:
        window_features.update(feats)
    scores["jaccard"] = np.asarray(
        [
            len(window_features & ref) / len(window_features | ref)
            if window_features | ref
            else 0.0
            for ref in tables.ref_features
        ]
    )
    return scores
