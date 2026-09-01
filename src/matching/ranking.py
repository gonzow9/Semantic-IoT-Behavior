"""Tie-aware ranking metrics for the candidate device scores."""

from __future__ import annotations

from math import isfinite

TIE_TOLERANCE = 1e-8


def _same_score(left: float, right: float) -> bool:
    return abs(left - right) <= TIE_TOLERANCE


def evaluate_ranking(
    scores: dict[str, float], expected_device: str, top_k: int
) -> dict[str, object]:
    """Evaluate one score vector without using device-name order for ties.

    Credit is averaged uniformly across the ranks taken by a tied score block. A score vector containing only zeros has no identification evidence,
    so the method abstains and receives zero credit.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not scores:
        raise ValueError("At least one candidate score is required.")
    if expected_device not in scores:
        raise ValueError(f"Expected device {expected_device!r} is missing from scores.")
    if any(not isfinite(float(score)) for score in scores.values()):
        raise ValueError("All candidate scores must be finite.")

    rows = [(str(device), float(score)) for device, score in scores.items()]
    ranked = sorted(rows, key=lambda item: (-item[1], item[0]))
    correct_score = float(scores[expected_device])
    best_score = max(score for _, score in rows)
    higher_count = sum(
        score > correct_score + TIE_TOLERANCE for _, score in rows
    )
    correct_tie_size = sum(
        _same_score(score, correct_score) for _, score in rows
    )
    correct_rank_first = higher_count + 1
    correct_rank_last = higher_count + correct_tie_size

    top_tied_devices = sorted(
        device for device, score in rows if _same_score(score, best_score)
    )
    best_tie_size = len(top_tied_devices)
    correct_tied_for_best = _same_score(correct_score, best_score)
    abstained = all(abs(score) <= TIE_TOLERANCE for _, score in rows)

    if abstained:
        top1_credit = 0.0
        topk_credit = 0.0
        reciprocal_rank = 0.0
    else:
        top1_credit = 1.0 / best_tie_size if correct_tied_for_best else 0.0
        places_in_topk = min(max(top_k - higher_count, 0), correct_tie_size)
        topk_credit = places_in_topk / correct_tie_size
        reciprocal_rank = sum(
            1.0 / rank
            for rank in range(correct_rank_first, correct_rank_last + 1)
        ) / correct_tie_size

    return {
        "abstained": abstained,
        "top1_credit": top1_credit,
        "topk_credit": topk_credit,
        "reciprocal_rank": reciprocal_rank,
        "correct_rank_first": None if abstained else correct_rank_first,
        "correct_rank_last": None if abstained else correct_rank_last,
        "correct_score_tie_size": correct_tie_size,
        "best_score_tie_size": best_tie_size,
        "top_tied_devices": top_tied_devices,
        "unique_prediction": (
            top_tied_devices[0] if not abstained and best_tie_size == 1 else None
        ),
        "top": [
            {"device": device, "score": round(score, 6)}
            for device, score in ranked[:top_k]
        ],
    }
