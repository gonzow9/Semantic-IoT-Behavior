"""Small Section 3 demo: semantic matching under evolving runtime behavior.

This module builds lightweight synthetic runtime-observation episodes from the
shipped per-ACE embedding bank. It is intentionally narrower than the original
research code: it demonstrates the Section 3 idea without regenerating the full
conference-paper experiment grid.

Two episode types are supported:

* strict-unseen: sampled query ACEs are removed from every candidate reference
  profile before scoring, so exact overlap has no evidence.
* partial: each query contains some exact ACEs that remain in the reference and
  some unseen ACEs removed from every candidate reference profile.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from semantic_iot_behavior.score import (
    asymmetric_maxsim,
    clean_device_name,
    exact_hit_count,
    jaccard,
)


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


def load_ace_bank(path: Path) -> AceBank:
    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data or "ace_texts" not in data:
        raise ValueError(f"{path} must contain embeddings and ace_texts arrays.")

    label_key = "devices" if "devices" in data else "names" if "names" in data else None
    if label_key is None:
        raise ValueError(f"{path} must contain either devices or names labels.")

    embeddings = normalise_rows(data["embeddings"].astype(np.float32, copy=False))
    labels = [str(value) for value in data[label_key]]
    ace_texts = [str(value) for value in data["ace_texts"]]
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


def unique_indices_for_device(bank: AceBank, device: str) -> list[int]:
    """Return one row index per unique ACE text for a device."""
    by_text: OrderedDict[str, int] = OrderedDict()
    for idx in bank.indices_by_device[device]:
        by_text.setdefault(bank.ace_texts[int(idx)], int(idx))
    return list(by_text.values())


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
                removed = frozenset(bank.ace_texts[idx] for idx in query)
                episodes.append(
                    Episode(
                        episode_id=f"{device}/strict-unseen/{episode_num:03d}",
                        mode=mode,
                        expected_device=device,
                        query_indices=query,
                        exact_indices=(),
                        unseen_indices=query,
                        removed_texts=removed,
                    )
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
                query = tuple(selected)
                removed = frozenset(bank.ace_texts[idx] for idx in unseen)
                episodes.append(
                    Episode(
                        episode_id=f"{device}/partial/{episode_num:03d}",
                        mode=mode,
                        expected_device=device,
                        query_indices=query,
                        exact_indices=exact,
                        unseen_indices=unseen,
                        removed_texts=removed,
                    )
                )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    if not episodes:
        raise ValueError("No episodes generated. Try smaller query counts.")
    return episodes


def reference_indices_for_episode(bank: AceBank, episode: Episode, device: str) -> np.ndarray:
    removed = episode.removed_texts
    return np.asarray(
        [
            int(idx)
            for idx in bank.indices_by_device[device]
            if bank.ace_texts[int(idx)] not in removed
        ],
        dtype=np.int64,
    )


def reference_texts_for_episode(bank: AceBank, episode: Episode, device: str) -> frozenset[str]:
    return frozenset(
        bank.ace_texts[int(idx)] for idx in reference_indices_for_episode(bank, episode, device)
    )


def mean_pool(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros(matrix.shape[1], dtype=np.float32)
    return normalise_vector(matrix.mean(axis=0))


def rank_scores(scores: dict[str, float], expected_device: str, top_k: int) -> dict[str, object]:
    ranked = sorted(
        ({"device": device, "score": float(score)} for device, score in scores.items()),
        key=lambda row: (-row["score"], row["device"]),
    )
    devices = [row["device"] for row in ranked]
    rank = devices.index(expected_device) + 1 if expected_device in devices else None
    return {
        "rank": rank,
        "top": [
            {"device": row["device"], "score": round(float(row["score"]), 6)}
            for row in ranked[:top_k]
        ],
    }


def score_episode(bank: AceBank, episode: Episode, top_k: int) -> dict[str, object]:
    query_texts = frozenset(bank.ace_texts[idx] for idx in episode.query_indices)
    query_vectors = bank.embeddings[np.asarray(episode.query_indices, dtype=np.int64)]
    query_mean = mean_pool(query_vectors)

    jaccard_scores: dict[str, float] = {}
    hit_scores: dict[str, float] = {}
    mean_scores: dict[str, float] = {}
    maxsim_scores: dict[str, float] = {}

    for device in bank.indices_by_device:
        ref_indices = reference_indices_for_episode(bank, episode, device)
        ref_texts = frozenset(bank.ace_texts[int(idx)] for idx in ref_indices)
        ref_vectors = bank.embeddings[ref_indices] if len(ref_indices) else bank.embeddings[:0]

        jaccard_scores[device] = jaccard(query_texts, ref_texts)
        hit_scores[device] = exact_hit_count(query_texts, ref_texts)
        mean_scores[device] = float(query_mean @ mean_pool(ref_vectors))
        maxsim_scores[device] = asymmetric_maxsim(query_vectors, ref_vectors)

    return {
        "jaccard": rank_scores(jaccard_scores, episode.expected_device, top_k),
        "exact_hit_count": rank_scores(hit_scores, episode.expected_device, top_k),
        "mean_pool": rank_scores(mean_scores, episode.expected_device, top_k),
        "maxsim": rank_scores(maxsim_scores, episode.expected_device, top_k),
    }


def summarise_results(scored: list[dict[str, object]], top_k: int) -> dict[str, dict[str, float]]:
    methods = ["jaccard", "exact_hit_count", "mean_pool", "maxsim"]
    summary: dict[str, dict[str, float]] = {}
    for method in methods:
        ranks = [item["scores"][method]["rank"] for item in scored]
        valid = [int(rank) for rank in ranks if rank is not None]
        n = len(ranks)
        summary[method] = {
            "top1": sum(1 for rank in valid if rank == 1) / n,
            f"top{top_k}": sum(1 for rank in valid if rank <= top_k) / n,
            "mrr": float(np.mean([1.0 / rank for rank in valid])) if valid else 0.0,
            "mean_rank": float(np.mean(valid)) if valid else 0.0,
        }
    return summary


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
    mode: str,
    episodes_per_device: int,
    query_size: int,
    exact_count: int,
    unseen_count: int,
    seed: int,
    top_k: int,
    examples: int,
) -> dict[str, object]:
    bank = load_ace_bank(embedding_npz)
    episodes = build_episodes(
        bank,
        mode=mode,
        episodes_per_device=episodes_per_device,
        query_size=query_size,
        exact_count=exact_count,
        unseen_count=unseen_count,
        seed=seed,
    )
    scored = [
        {
            "episode": episode,
            "scores": score_episode(bank, episode, top_k),
        }
        for episode in episodes
    ]

    summary = summarise_results(scored, top_k)
    return {
        "config": {
            "embedding_npz": str(embedding_npz),
            "mode": mode,
            "episodes_per_device": episodes_per_device,
            "query_size": query_size,
            "exact_count": exact_count if mode == "partial" else 0,
            "unseen_count": unseen_count if mode == "partial" else query_size,
            "seed": seed,
            "top_k": top_k,
        },
        "episode_count": len(episodes),
        "device_count": len({episode.expected_device for episode in episodes}),
        "summary": summary,
        "examples": [
            episode_record(bank, item["episode"], item["scores"])
            for item in scored[:examples]
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-npz",
        type=Path,
        default=Path("embeddings/bge_m3/reference_per_ace_whitened_k256.npz"),
        help="Per-ACE embedding bank with embeddings, device labels, and ace_texts.",
    )
    parser.add_argument(
        "--mode",
        choices=["strict-unseen", "partial"],
        default="strict-unseen",
        help="Synthetic runtime-observation episode type.",
    )
    parser.add_argument("--episodes-per-device", type=int, default=5)
    parser.add_argument(
        "--query-size",
        type=int,
        default=3,
        help="Number of query ACEs for strict-unseen mode.",
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
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def print_summary(result: dict[str, object]) -> None:
    print(
        f"Generated {result['episode_count']} {result['config']['mode']} episodes "
        f"from {result['device_count']} devices."
    )
    print("method             top1    topK     mrr   mean_rank")
    for method, row in result["summary"].items():
        topk_key = next(key for key in row if key.startswith("top") and key != "top1")
        print(
            f"{method:<18} "
            f"{row['top1']:.3f}  {row[topk_key]:.3f}  {row['mrr']:.3f}  {row['mean_rank']:.2f}"
        )


def main() -> None:
    args = parse_args()
    result = run_demo(
        embedding_npz=args.embedding_npz,
        mode=args.mode,
        episodes_per_device=args.episodes_per_device,
        query_size=args.query_size,
        exact_count=args.exact_count,
        unseen_count=args.unseen_count,
        seed=args.seed,
        top_k=args.top_k,
        examples=args.examples,
    )
    print_summary(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.output}.")


if __name__ == "__main__":
    main()
