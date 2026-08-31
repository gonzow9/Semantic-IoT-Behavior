"""Score queries against every device with the four matching methods.

Scoring rule: a method only makes a prediction when its best score is positive and a single device holds it.
"""

from __future__ import annotations

import numpy as np

from matching.bank import AceBank, mean_pool
from matching.episodes import Episode
from runtime_score import (
    asymmetric_maxsim,
    exact_hit_count,
    jaccard,
)

METHODS = ["jaccard", "exact_hit_count", "mean_pool", "maxsim"]


def rank_scores(scores: dict[str, float], expected_device: str, top_k: int) -> dict[str, object]:
    """Rank devices by score

    A method abstains when its best score is exactly zero because it then
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
