"""Simple exact-overlap and per-ACE MaxSim retrieval scoring."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np

TIE_TOL = 1e-12


def clean_device_name(name: str) -> str:
    clean = name[:-8] if name.endswith("_compact") else name
    for marker in ("_test_", "_loo_", "_family_", "_ipbits"):
        if marker in clean:
            return clean.split(marker, 1)[0]
    return clean


def read_rules(path: Path) -> frozenset[str]:
    return frozenset(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def load_text_profiles(directory: Path, recursive: bool) -> OrderedDict[str, frozenset[str]]:
    pattern = "**/*.txt" if recursive else "*.txt"
    profiles: OrderedDict[str, frozenset[str]] = OrderedDict()
    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue
        name = clean_device_name(path.stem)
        if recursive and len(path.relative_to(directory).parts) > 1:
            name = clean_device_name(path.relative_to(directory).parts[0])
        profiles[str(path.relative_to(directory))] = read_rules(path)
    if not profiles:
        raise ValueError(f"No compact .txt profiles found in {directory}")
    return profiles


def jaccard(query: frozenset[str], reference: frozenset[str]) -> float:
    union = len(query | reference)
    return len(query & reference) / union if union else 0.0


def exact_hit_count(query: frozenset[str], reference: frozenset[str]) -> float:
    return float(len(query & reference))


TEXT_SCORERS: dict[str, Callable[[frozenset[str], frozenset[str]], float]] = {
    "jaccard": jaccard,
    "exact_hit_count": exact_hit_count,
}


def rank_text_profiles(
    reference_dir: Path,
    query_dir: Path,
    *,
    method: str,
    top_k: int,
) -> dict[str, object]:
    references = load_text_profiles(reference_dir, recursive=False)
    reference_by_device = {
        clean_device_name(Path(name).stem): rules for name, rules in references.items()
    }
    queries = load_text_profiles(query_dir, recursive=True)
    scorer = TEXT_SCORERS[method]

    top1_hits = 0
    topk_hits = 0
    reciprocal_ranks: list[float] = []
    results: dict[str, object] = {}
    for query_name, query_rules in queries.items():
        expected = clean_device_name(Path(query_name).parts[0] if "/" in query_name else Path(query_name).stem)
        ranked = sorted(
            (
                {
                    "device": device,
                    "score": float(scorer(query_rules, reference_rules)),
                    "hit_count": len(query_rules & reference_rules),
                }
                for device, reference_rules in reference_by_device.items()
            ),
            key=lambda row: (-row["score"], row["device"]),
        )
        devices = [row["device"] for row in ranked]
        rank = devices.index(expected) + 1 if expected in devices else None
        if rank == 1:
            top1_hits += 1
        if rank is not None and rank <= top_k:
            topk_hits += 1
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
        results[query_name] = {
            "expected": expected,
            "rank": rank,
            "top": ranked[:top_k],
        }

    n = len(queries)
    return {
        "method": method,
        "queries": n,
        "top1_accuracy": top1_hits / n,
        f"top{top_k}_accuracy": topk_hits / n,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "results": results,
    }


def load_per_ace_bank(path: Path) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data:
        raise ValueError(f"{path} is missing an embeddings array")
    label_key = "devices" if "devices" in data else "names"
    embeddings = data["embeddings"].astype(np.float32, copy=False)
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    raw_labels = [str(value) for value in data[label_key]]
    clean_labels = [clean_device_name(value) for value in raw_labels]
    ace_texts = [str(value) for value in data["ace_texts"]] if "ace_texts" in data else []
    return embeddings, raw_labels, clean_labels, ace_texts


def grouped_vectors(embeddings: np.ndarray, labels: list[str]) -> OrderedDict[str, np.ndarray]:
    groups: OrderedDict[str, list[np.ndarray]] = OrderedDict()
    for vector, label in zip(embeddings, labels):
        groups.setdefault(label, []).append(vector)
    return OrderedDict((label, np.vstack(rows)) for label, rows in sorted(groups.items()))


def asymmetric_maxsim(query: np.ndarray, reference: np.ndarray) -> float:
    if query.shape[0] == 0 or reference.shape[0] == 0:
        return 0.0
    return float(np.max(query @ reference.T, axis=1).mean())


def rank_maxsim(reference_npz: Path, query_npz: Path, *, top_k: int) -> dict[str, object]:
    ref_embs, _ref_raw_labels, ref_clean_labels, _ = load_per_ace_bank(reference_npz)
    query_embs, query_raw_labels, _query_clean_labels, _ = load_per_ace_bank(query_npz)
    references = grouped_vectors(ref_embs, ref_clean_labels)
    queries = grouped_vectors(query_embs, query_raw_labels)

    top1_hits = 0
    topk_hits = 0
    reciprocal_ranks: list[float] = []
    results: dict[str, object] = {}
    for query_label, query_vectors in queries.items():
        expected = clean_device_name(query_label)
        ranked = sorted(
            (
                {"device": device, "score": asymmetric_maxsim(query_vectors, reference_vectors)}
                for device, reference_vectors in references.items()
            ),
            key=lambda row: (-row["score"], row["device"]),
        )
        devices = [row["device"] for row in ranked]
        rank = devices.index(expected) + 1 if expected in devices else None
        if rank == 1:
            top1_hits += 1
        if rank is not None and rank <= top_k:
            topk_hits += 1
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
        results[query_label] = {"expected": expected, "rank": rank, "top": ranked[:top_k]}

    n = len(queries)
    return {
        "method": "asymmetric_maxsim",
        "queries": n,
        "top1_accuracy": top1_hits / n,
        f"top{top_k}_accuracy": topk_hits / n,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    exact = subparsers.add_parser("exact")
    exact.add_argument("--reference-dir", type=Path, default=Path("data/mud_compact"))
    exact.add_argument("--query-dir", type=Path, required=True)
    exact.add_argument("--method", choices=sorted(TEXT_SCORERS), default="jaccard")
    exact.add_argument("--top-k", type=int, default=5)
    exact.add_argument("--output", type=Path, required=True)

    maxsim = subparsers.add_parser("maxsim")
    maxsim.add_argument("--reference-npz", type=Path, required=True)
    maxsim.add_argument("--query-npz", type=Path, required=True)
    maxsim.add_argument("--top-k", type=int, default=5)
    maxsim.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "exact":
        result = rank_text_profiles(
            args.reference_dir,
            args.query_dir,
            method=args.method,
            top_k=args.top_k,
        )
    else:
        result = rank_maxsim(args.reference_npz, args.query_npz, top_k=args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}.")


if __name__ == "__main__":
    main()
