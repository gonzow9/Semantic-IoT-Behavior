"""Paired bootstrap confidence intervals for Top-1 differences."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from matching.scoring import METHODS


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
    size-weighted mean over the resampled clusters so correlated queries
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
