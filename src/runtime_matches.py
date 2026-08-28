"""Compare matching methods on synthetic runtime observations.

The script creates repeatable queries from the shipped per-ACE embedding bank.
It compares exact overlap, mean pooling, and MaxSim.

Modes:

- single-unseen: one query per ACE; that ACE is removed from all references.
- unseen-family: one query per (device, ACE family); the whole family is
  removed from all references. Families are found by clustering ACE embeddings.
- unseen-set: three ACEs drawn from distinct families, removed from all
  references. Ten seeded queries per eligible device.
- strict-unseen: a simpler demo where N random ACEs are removed everywhere.
- partial: some query ACEs remain in the references and the rest are removed.

Scoring follows the paper's abstention rule: a method only makes a prediction
when its best score is positive and a single device holds it. Otherwise the
query counts as a miss. The summary also reports paired bootstrap confidence
intervals for the Top-1 difference between MaxSim and each baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from runtime_score import (
    asymmetric_maxsim,
    clean_device_name,
    exact_hit_count,
    jaccard,
)

METHODS = ["jaccard", "exact_hit_count", "mean_pool", "maxsim"]


@dataclass(frozen=True)
class AceBank:
    path: Path
    embeddings: np.ndarray
    labels: list[str]
    clean_labels: list[str]
    ace_texts: list[str]
    indices_by_device: OrderedDict[str, np.ndarray]


@dataclass(frozen=True)
class Episode:
    episode_id: str
    mode: str
    expected_device: str
    query_indices: tuple[int, ...]
    exact_indices: tuple[int, ...]
    unseen_indices: tuple[int, ...]
    removed_texts: frozenset[str]


def normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


def normalise_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def stable_seed_offset(name: str) -> int:
    """Deterministic per-name seed offset, stable across Python runs."""
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)


def build_ace_bank(
    path: Path,
    embeddings: np.ndarray,
    labels: list[str],
    ace_texts: list[str],
) -> AceBank:
    embeddings = normalise_rows(embeddings.astype(np.float32, copy=False))
    if len(labels) != embeddings.shape[0] or len(ace_texts) != embeddings.shape[0]:
        raise ValueError(f"{path} has mismatched row counts.")

    clean_labels = [clean_device_name(label) for label in labels]
    grouped: dict[str, list[int]] = {}
    for idx, device in enumerate(clean_labels):
        grouped.setdefault(device, []).append(idx)

    return AceBank(
        path=path,
        embeddings=embeddings,
        labels=labels,
        clean_labels=clean_labels,
        ace_texts=ace_texts,
        indices_by_device=OrderedDict(
            (device, np.asarray(indices, dtype=np.int64))
            for device, indices in sorted(grouped.items())
        ),
    )


def load_ace_bank(path: Path) -> AceBank:
    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data or "ace_texts" not in data:
        raise ValueError(f"{path} must contain embeddings and ace_texts arrays.")

    label_key = "devices" if "devices" in data else "names" if "names" in data else None
    if label_key is None:
        raise ValueError(f"{path} must contain either devices or names labels.")

    return build_ace_bank(
        path,
        data["embeddings"],
        [str(value) for value in data[label_key]],
        [str(value) for value in data["ace_texts"]],
    )


def unique_indices_for_device(bank: AceBank, device: str) -> list[int]:
    """Return one row index per unique ACE text for a device."""
    by_text: OrderedDict[str, int] = OrderedDict()
    for idx in bank.indices_by_device[device]:
        by_text.setdefault(bank.ace_texts[int(idx)], int(idx))
    return list(by_text.values())


# ---------------------------------------------------------------------------
# ACE families: reciprocal nearest-neighbour clustering of ACE embeddings.
# ---------------------------------------------------------------------------


def family_space_vectors(raw_npz: Path, unique_texts: list[str]) -> np.ndarray:
    """Whiten the unique raw ACE embeddings to form the family space.

    Each unique ACE text is represented by the mean of its raw embedding rows
    (the same text can appear in several profiles). The means are whitened
    with a transform fitted on the means themselves, keeping up to 256
    dimensions, and L2-normalised so cosine similarity is a dot product.
    """
    data = np.load(raw_npz, allow_pickle=True)
    if "embeddings" not in data or "ace_texts" not in data:
        raise ValueError(f"{raw_npz} must contain embeddings and ace_texts arrays.")
    raw = data["embeddings"].astype(np.float32, copy=False)
    rows_by_text: dict[str, list[int]] = {}
    for idx, text in enumerate(data["ace_texts"]):
        rows_by_text.setdefault(str(text), []).append(idx)
    missing = [text for text in unique_texts if text not in rows_by_text]
    if missing:
        raise ValueError(f"{raw_npz} is missing {len(missing)} ACE texts, e.g. {missing[0]!r}")

    unique = np.vstack(
        [raw[rows_by_text[text]].mean(axis=0) for text in unique_texts]
    ).astype(np.float32)

    centered = unique - unique.mean(axis=0)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    noise = max(centered.shape) * np.finfo(centered.dtype).eps * singular_values.max()
    keep = min(256, int((singular_values > noise).sum()))
    scale = np.sqrt(unique.shape[0] - 1) / singular_values[:keep]
    return normalise_rows((centered @ components[:keep].T) * scale)


def build_families(
    vectors: np.ndarray,
    *,
    top_k: int,
    threshold: float,
) -> list[list[int]]:
    """Group vectors into families via reciprocal top-k neighbours.

    Two ACEs join the same family when each is among the other's top-k cosine
    neighbours and their similarity is at least the threshold. Families are the
    connected components of these mutual links, sorted largest first.
    """
    count = vectors.shape[0]
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    neighbours = np.argsort(-sims, axis=1)[:, :top_k]
    neighbour_sets = [
        {int(j) for j in row if sims[i, j] >= threshold}
        for i, row in enumerate(neighbours)
    ]

    parent = list(range(count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(count):
        for j in neighbour_sets[i]:
            if i in neighbour_sets[j]:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(count):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda members: (-len(members), min(members)))


def family_by_text(bank: AceBank, raw_npz: Path, top_k: int, threshold: float) -> dict[str, str]:
    """Map every unique ACE text to a family id such as fam_0001."""
    unique_texts: list[str] = []
    seen: set[str] = set()
    for text in bank.ace_texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)
    vectors = family_space_vectors(raw_npz, unique_texts)
    families: dict[str, str] = {}
    for number, members in enumerate(build_families(vectors, top_k=top_k, threshold=threshold), 1):
        for member in members:
            families[unique_texts[member]] = f"fam_{number:04d}"
    return families


# ---------------------------------------------------------------------------
# Episode generation.
# ---------------------------------------------------------------------------


def make_unseen_episode(
    bank: AceBank, mode: str, device: str, name: str, query: tuple[int, ...]
) -> Episode:
    return Episode(
        episode_id=f"{device}/{mode}/{name}",
        mode=mode,
        expected_device=device,
        query_indices=query,
        exact_indices=(),
        unseen_indices=query,
        removed_texts=frozenset(bank.ace_texts[idx] for idx in query),
    )


def build_single_unseen_episodes(bank: AceBank) -> list[Episode]:
    """One episode per (device, unique ACE); the ACE is removed everywhere."""
    episodes: list[Episode] = []
    for device in bank.indices_by_device:
        for num, idx in enumerate(unique_indices_for_device(bank, device)):
            episodes.append(
                make_unseen_episode(bank, "single-unseen", device, f"{num:04d}", (idx,))
            )
    return episodes


def build_family_episodes(bank: AceBank, families: dict[str, str]) -> list[Episode]:
    """One episode per (device, family) with at least two members in the device."""
    family_sizes = Counter(families.values())
    episodes: list[Episode] = []
    for device in bank.indices_by_device:
        buckets: OrderedDict[str, list[int]] = OrderedDict()
        for idx in unique_indices_for_device(bank, device):
            buckets.setdefault(families[bank.ace_texts[idx]], []).append(idx)
        for family_id, indices in buckets.items():
            if family_sizes[family_id] < 2 or len(indices) < 2:
                continue
            episodes.append(
                make_unseen_episode(bank, "unseen-family", device, family_id, tuple(indices))
            )
    return episodes


def build_set_episodes(
    bank: AceBank,
    families: dict[str, str],
    *,
    seeds_per_device: int,
    min_profile_size: int,
    query_size: int,
    base_seed: int,
) -> list[Episode]:
    """Seeded queries of ACEs drawn from distinct families where possible."""
    episodes: list[Episode] = []
    for device in bank.indices_by_device:
        unique_indices = unique_indices_for_device(bank, device)
        if len(unique_indices) < min_profile_size:
            continue
        buckets: dict[str, list[int]] = {}
        for idx in unique_indices:
            buckets.setdefault(families[bank.ace_texts[idx]], []).append(idx)
        family_ids = sorted(buckets)
        for offset in range(seeds_per_device):
            rng = random.Random(base_seed + offset * 1009 + stable_seed_offset(device))
            if len(family_ids) >= query_size:
                chosen = sorted(rng.sample(family_ids, query_size))
                query = tuple(sorted(rng.choice(buckets[fid]) for fid in chosen))
            else:
                query = tuple(sorted(rng.sample(unique_indices, query_size)))
            episodes.append(
                make_unseen_episode(bank, "unseen-set", device, f"{offset:03d}", query)
            )
    return episodes


def build_episodes(
    bank: AceBank,
    *,
    mode: str,
    episodes_per_device: int,
    query_size: int,
    exact_count: int,
    unseen_count: int,
    seed: int,
) -> list[Episode]:
    rng = random.Random(seed)
    episodes: list[Episode] = []

    for device in bank.indices_by_device:
        unique_indices = unique_indices_for_device(bank, device)
        if mode == "strict-unseen":
            if len(unique_indices) <= query_size:
                continue
            for episode_num in range(episodes_per_device):
                query = tuple(rng.sample(unique_indices, query_size))
                episodes.append(
                    make_unseen_episode(bank, mode, device, f"{episode_num:03d}", query)
                )
        elif mode == "partial":
            total = exact_count + unseen_count
            if total <= 0:
                raise ValueError("partial mode needs at least one exact or unseen ACE.")
            if len(unique_indices) <= total:
                continue
            for episode_num in range(episodes_per_device):
                selected = rng.sample(unique_indices, total)
                exact = tuple(selected[:exact_count])
                unseen = tuple(selected[exact_count:])
                episodes.append(
                    Episode(
                        episode_id=f"{device}/partial/{episode_num:03d}",
                        mode=mode,
                        expected_device=device,
                        query_indices=tuple(selected),
                        exact_indices=exact,
                        unseen_indices=unseen,
                        removed_texts=frozenset(bank.ace_texts[idx] for idx in unseen),
                    )
                )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    if not episodes:
        raise ValueError("No episodes generated. Try smaller query counts.")
    return episodes


def mean_pool(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros(matrix.shape[1], dtype=np.float32)
    return normalise_vector(matrix.mean(axis=0))


# ---------------------------------------------------------------------------
# Scoring with the abstention rule.
# ---------------------------------------------------------------------------


def rank_scores(scores: dict[str, float], expected_device: str, top_k: int) -> dict[str, object]:
    """Rank devices by score, applying the paper's abstention rule.

    A method abstains when its best score is exactly zero, because it then
    has no evidence for any device; the query counts as a miss. Ties between
    positive scores keep the deterministic device-name order used throughout.
    """
    ranked = sorted(
        ({"device": device, "score": float(score)} for device, score in scores.items()),
        key=lambda row: (-row["score"], row["device"]),
    )
    best = ranked[0]["score"]
    abstained = best == 0.0
    devices = [row["device"] for row in ranked]
    rank = devices.index(expected_device) + 1 if expected_device in devices else None
    top1_correct = not abstained and rank == 1
    return {
        "rank": None if abstained else rank,
        "abstained": abstained,
        "top1_correct": top1_correct,
        "top": [
            {"device": row["device"], "score": round(float(row["score"]), 6)}
            for row in ranked[:top_k]
        ],
    }


def score_query(
    bank: AceBank,
    query_texts: list[str],
    query_vectors: np.ndarray,
    *,
    expected_device: str,
    removed_texts: frozenset[str] = frozenset(),
    removed_scope: str = "all",
    top_k: int = 5,
) -> dict[str, object]:
    """Score one query against every device with all four methods.

    removed_scope controls where removed_texts are held out: "all" removes
    them from every candidate profile, "expected" only from the true device.
    """
    query_text_set = frozenset(query_texts)
    query_mean = mean_pool(query_vectors)

    method_scores: dict[str, dict[str, float]] = {method: {} for method in METHODS}
    for device in bank.indices_by_device:
        if removed_scope == "all" or device == expected_device:
            indices = np.asarray(
                [
                    int(idx)
                    for idx in bank.indices_by_device[device]
                    if bank.ace_texts[int(idx)] not in removed_texts
                ],
                dtype=np.int64,
            )
        else:
            indices = bank.indices_by_device[device]
        ref_texts = frozenset(bank.ace_texts[int(idx)] for idx in indices)
        ref_vectors = bank.embeddings[indices] if len(indices) else bank.embeddings[:0]

        method_scores["jaccard"][device] = jaccard(query_text_set, ref_texts)
        method_scores["exact_hit_count"][device] = exact_hit_count(query_text_set, ref_texts)
        method_scores["mean_pool"][device] = float(query_mean @ mean_pool(ref_vectors))
        method_scores["maxsim"][device] = asymmetric_maxsim(query_vectors, ref_vectors)

    return {
        method: rank_scores(scores, expected_device, top_k)
        for method, scores in method_scores.items()
    }


def score_episode(bank: AceBank, episode: Episode, top_k: int) -> dict[str, object]:
    return score_query(
        bank,
        [bank.ace_texts[idx] for idx in episode.query_indices],
        bank.embeddings[np.asarray(episode.query_indices, dtype=np.int64)],
        expected_device=episode.expected_device,
        removed_texts=episode.removed_texts,
        removed_scope="all",
        top_k=top_k,
    )


def summarise_results(scored: list[dict[str, object]], top_k: int) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for method in METHODS:
        rows = [item["scores"][method] for item in scored]
        n = len(rows)
        ranks = [row["rank"] for row in rows if row["rank"] is not None]
        summary[method] = {
            "top1": sum(1 for row in rows if row["top1_correct"]) / n,
            f"top{top_k}": sum(1 for rank in ranks if rank <= top_k) / n,
            "mrr": sum(1.0 / rank for rank in ranks) / n,
            "abstain_rate": sum(1 for row in rows if row["abstained"]) / n,
        }
    return summary


# ---------------------------------------------------------------------------
# Paired bootstrap confidence intervals.
# ---------------------------------------------------------------------------


def bootstrap_mean_ci(
    values: np.ndarray, *, resamples: int, seed: int, batch: int = 1000
) -> tuple[float, float]:
    """95% percentile bootstrap interval for the mean of values."""
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, batch):
        count = min(batch, resamples - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def cluster_bootstrap_ci(
    diffs: np.ndarray,
    clusters: list[str],
    *,
    resamples: int,
    seed: int,
    batch: int = 1000,
) -> tuple[float, float]:
    """95% cluster bootstrap interval for the pooled mean of diffs.

    Whole clusters are resampled with replacement and the statistic is the
    size-weighted mean over the resampled clusters, so correlated queries
    from the same cluster move together.
    """
    by_cluster: OrderedDict[str, list[float]] = OrderedDict()
    for cluster, diff in zip(clusters, diffs):
        by_cluster.setdefault(cluster, []).append(float(diff))
    sums = np.asarray([sum(values) for values in by_cluster.values()])
    sizes = np.asarray([len(values) for values in by_cluster.values()])

    rng = np.random.default_rng(seed)
    stats = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, batch):
        count = min(batch, resamples - start)
        indices = rng.integers(0, sums.size, size=(count, sums.size))
        stats[start : start + count] = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def paired_top1_bootstrap(
    scored: list[dict[str, object]], *, resamples: int, seed: int
) -> dict[str, object]:
    """Paired Top-1 differences of MaxSim against each baseline.

    The episode interval resamples queries independently. The cluster
    interval resamples whole clusters (the expected device by default), so
    correlated queries from one device cannot narrow the interval.
    """
    clusters = [item.get("cluster_key", item["expected_device"]) for item in scored]
    correct = {
        method: np.asarray(
            [1.0 if item["scores"][method]["top1_correct"] else 0.0 for item in scored]
        )
        for method in METHODS
    }

    comparisons: dict[str, object] = {}
    for baseline in ("jaccard", "exact_hit_count", "mean_pool"):
        diffs = correct["maxsim"] - correct[baseline]
        comparisons[f"maxsim_minus_{baseline}"] = {
            "difference": float(diffs.mean()),
            "episode_ci95": list(bootstrap_mean_ci(diffs, resamples=resamples, seed=seed)),
            "cluster_ci95": list(
                cluster_bootstrap_ci(diffs, clusters, resamples=resamples, seed=seed)
            ),
        }
    return comparisons


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def episode_record(bank: AceBank, episode: Episode, scores: dict[str, object]) -> dict[str, object]:
    return {
        "episode_id": episode.episode_id,
        "mode": episode.mode,
        "expected_device": episode.expected_device,
        "query_aces": [bank.ace_texts[idx] for idx in episode.query_indices],
        "exact_aces": [bank.ace_texts[idx] for idx in episode.exact_indices],
        "unseen_aces": [bank.ace_texts[idx] for idx in episode.unseen_indices],
        "scores": scores,
    }


def run_demo(
    *,
    embedding_npz: Path,
    raw_npz: Path,
    mode: str,
    episodes_per_device: int,
    query_size: int,
    exact_count: int,
    unseen_count: int,
    set_seeds: int,
    min_profile_size: int,
    family_top_k: int,
    family_threshold: float,
    seed: int,
    top_k: int,
    examples: int,
    bootstrap_resamples: int,
) -> dict[str, object]:
    bank = load_ace_bank(embedding_npz)

    if mode == "single-unseen":
        episodes = build_single_unseen_episodes(bank)
    elif mode in ("unseen-family", "unseen-set"):
        families = family_by_text(bank, raw_npz, family_top_k, family_threshold)
        if mode == "unseen-family":
            episodes = build_family_episodes(bank, families)
        else:
            episodes = build_set_episodes(
                bank,
                families,
                seeds_per_device=set_seeds,
                min_profile_size=min_profile_size,
                query_size=query_size,
                base_seed=seed,
            )
    else:
        episodes = build_episodes(
            bank,
            mode=mode,
            episodes_per_device=episodes_per_device,
            query_size=query_size,
            exact_count=exact_count,
            unseen_count=unseen_count,
            seed=seed,
        )
    if not episodes:
        raise ValueError("No episodes generated.")

    scored = [
        {
            "episode": episode,
            "expected_device": episode.expected_device,
            "scores": score_episode(bank, episode, top_k),
        }
        for episode in episodes
    ]

    result: dict[str, object] = {
        "config": {
            "embedding_npz": str(embedding_npz),
            "mode": mode,
            "seed": seed,
            "top_k": top_k,
        },
        "episode_count": len(episodes),
        "device_count": len({episode.expected_device for episode in episodes}),
        "summary": summarise_results(scored, top_k),
        "examples": [
            episode_record(bank, item["episode"], item["scores"])
            for item in scored[:examples]
        ],
    }
    if bootstrap_resamples > 0:
        result["paired_top1_bootstrap"] = paired_top1_bootstrap(
            scored, resamples=bootstrap_resamples, seed=seed
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--embedding-npz",
        type=Path,
        default=Path("data/ref_embeddings/bge/per_ace/whitened_k256/reference_per_ace_whitened_k256.npz"),
        help="Per-ACE embedding bank with embeddings, device labels, and ace_texts.",
    )
    parser.add_argument(
        "--raw-npz",
        type=Path,
        default=Path("data/ref_embeddings/bge/per_ace/raw/reference_per_ace.npz"),
        help="Raw per-ACE bank used to build the ACE family space.",
    )
    parser.add_argument(
        "--mode",
        choices=["single-unseen", "unseen-family", "unseen-set", "strict-unseen", "partial"],
        default="strict-unseen",
        help="Synthetic observation type.",
    )
    parser.add_argument(
        "--episodes-per-device",
        type=int,
        default=5,
        help="Observations per device in strict-unseen and partial modes.",
    )
    parser.add_argument(
        "--query-size",
        type=int,
        default=3,
        help="Number of query ACEs for strict-unseen and unseen-set modes.",
    )
    parser.add_argument(
        "--exact-count",
        type=int,
        default=2,
        help="Number of exact ACEs retained in partial mode.",
    )
    parser.add_argument(
        "--unseen-count",
        type=int,
        default=2,
        help="Number of query ACEs removed from references in partial mode.",
    )
    parser.add_argument(
        "--set-seeds",
        type=int,
        default=10,
        help="Seeded queries per device in unseen-set mode.",
    )
    parser.add_argument(
        "--min-profile-size",
        type=int,
        default=12,
        help="Minimum unique ACEs a device needs for unseen-set mode.",
    )
    parser.add_argument(
        "--family-top-k",
        type=int,
        default=5,
        help="Neighbours considered when clustering ACEs into families.",
    )
    parser.add_argument(
        "--family-threshold",
        type=float,
        default=0.75,
        help="Minimum cosine similarity for a reciprocal family link.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1729,
        help="Random seed for repeatable episodes.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of ranked devices kept for each query.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="Number of example episodes included in the result.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10000,
        help="Bootstrap resamples for the paired Top-1 intervals. 0 disables.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def print_summary(result: dict[str, object]) -> None:
    print(
        f"Generated {result['episode_count']} {result['config']['mode']} episodes "
        f"from {result['device_count']} devices."
    )
    print("method             top1    topK     mrr   abstain")
    for method, row in result["summary"].items():
        topk_key = next(key for key in row if key.startswith("top") and key != "top1")
        print(
            f"{method:<18} "
            f"{row['top1']:.4f}  {row[topk_key]:.4f}  {row['mrr']:.4f}  {row['abstain_rate']:.4f}"
        )
    for name, row in result.get("paired_top1_bootstrap", {}).items():
        low, high = row["episode_ci95"]
        cluster_low, cluster_high = row["cluster_ci95"]
        print(
            f"{name}: {row['difference']:+.4f} "
            f"(episode 95% CI [{low:+.4f}, {high:+.4f}], "
            f"cluster 95% CI [{cluster_low:+.4f}, {cluster_high:+.4f}])"
        )


def main() -> None:
    args = parse_args()
    result = run_demo(
        embedding_npz=args.embedding_npz,
        raw_npz=args.raw_npz,
        mode=args.mode,
        episodes_per_device=args.episodes_per_device,
        query_size=args.query_size,
        exact_count=args.exact_count,
        unseen_count=args.unseen_count,
        set_seeds=args.set_seeds,
        min_profile_size=args.min_profile_size,
        family_top_k=args.family_top_k,
        family_threshold=args.family_threshold,
        seed=args.seed,
        top_k=args.top_k,
        examples=args.examples,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    print_summary(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.output}.")


if __name__ == "__main__":
    main()
