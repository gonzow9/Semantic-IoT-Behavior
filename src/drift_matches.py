"""Drifted-endpoint and mixed partial observation experiments.

Two experiment types:

- drift: hostnames in selected ACEs are changed while protocol and port stay
  the same. The query contains only the drifted ACEs. The "full" subset uses
  every device; "high-domain" keeps devices with at least ten domain ACEs.
- mixed: a runtime query mixes exact ACEs, drifted ACEs, and optionally one
  unseen ACE. A grid over the retained fraction, the drift fraction, and the
  unseen count is evaluated, and Top-1 is also grouped by the number of exact
  ACE matches against the correct reference profile.

Drifted ACE texts are new strings, so they must be embedded. The first run
downloads the BGE-M3 model. All embeddings (references and queries) are
whitened with one transform fitted on the raw reference bank.

The hostname mutations live in ``drift/perturb.py`` and the query
construction in ``drift/queries.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drift.queries import (
    Query,
    build_drift_queries,
    build_mixed_queries,
    read_compact_profiles,
)
from gen_emb import build_model, encode_texts
from gen_whiten_emb import apply_whitening, fit_whitening
from matching.bank import AceBank, build_ace_bank
from matching.scoring import METHODS, score_query, summarise_results
from matching.stats import paired_top1_bootstrap


# ---------------------------------------------------------------------------
# Embedding and scoring.
# ---------------------------------------------------------------------------


def load_whitened_reference(raw_npz: Path) -> tuple[AceBank, dict[str, object]]:
    """Whiten the raw reference bank locally and keep the transform."""
    data = np.load(raw_npz, allow_pickle=True)
    raw = data["embeddings"].astype(np.float32, copy=False)
    mean, components, singular_values = fit_whitening(raw)
    whitened = apply_whitening(raw, mean, components, singular_values, 256, raw.shape[0])
    bank = build_ace_bank(
        raw_npz,
        whitened,
        [str(value) for value in data["devices"]],
        [str(value) for value in data["ace_texts"]],
    )
    transform = {
        "mean": mean,
        "components": components,
        "singular_values": singular_values,
        "n_reference": raw.shape[0],
    }
    return bank, transform


def vectors_for_queries(
    bank: AceBank,
    transform: dict[str, object],
    queries: list[Query],
    *,
    model_name: str,
    device: str | None,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Map every query ACE text to a whitened vector.

    Texts already in the reference bank reuse its rows. New (drifted) texts
    are embedded with the sentence-transformer model and whitened with the
    reference transform.
    """
    vector_by_text: dict[str, np.ndarray] = {}
    for idx, text in enumerate(bank.ace_texts):
        vector_by_text.setdefault(text, bank.embeddings[idx])

    new_texts = sorted(
        {text for query in queries for text in query.query_texts if text not in vector_by_text}
    )
    if new_texts:
        print(f"Embedding {len(new_texts)} drifted ACE texts with {model_name}...")
        model = build_model(model_name, device)
        raw = encode_texts(model, new_texts, batch_size)
        whitened = apply_whitening(
            raw,
            transform["mean"],
            transform["components"],
            transform["singular_values"],
            256,
            transform["n_reference"],
        )
        vector_by_text.update(zip(new_texts, whitened))
    return vector_by_text


def score_queries(
    bank: AceBank,
    queries: list[Query],
    vector_by_text: dict[str, np.ndarray],
    *,
    top_k: int,
) -> list[dict[str, object]]:
    scored = []
    for query in queries:
        vectors = np.vstack([vector_by_text[text] for text in query.query_texts])
        scored.append(
            {
                "query": query,
                "expected_device": query.expected_device,
                "cluster_key": query.cluster_key,
                "scores": score_query(
                    bank,
                    list(query.query_texts),
                    vectors,
                    expected_device=query.expected_device,
                    removed_texts=query.removed_texts,
                    removed_scope=query.removed_scope,
                    top_k=top_k,
                ),
            }
        )
    return scored


def exact_hit_bin(hits: int) -> str:
    if hits <= 0:
        return "0"
    if hits <= 2:
        return "1-2"
    if hits <= 5:
        return "3-5"
    return ">5"


def summarise_bins(scored: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Top-1 accuracy per method, grouped by exact hits against the truth."""
    bins: dict[str, dict[str, object]] = {}
    for label in ("0", "1-2", "3-5", ">5"):
        rows = [item for item in scored if exact_hit_bin(item["query"].exact_hits) == label]
        if not rows:
            continue
        bins[label] = {"queries": len(rows)}
        for method in METHODS:
            correct = sum(1 for item in rows if item["scores"][method]["top1_correct"])
            bins[label][method] = correct / len(rows)
    return bins


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--compact-dir",
        type=Path,
        default=Path("data/ref_mud/compact"),
        help="Directory containing the reference compact .txt profiles.",
    )
    parser.add_argument(
        "--raw-npz",
        type=Path,
        default=Path("data/ref_embeddings/bge/per_ace/raw/reference_per_ace.npz"),
        help="Raw per-ACE reference bank used for whitening and scoring.",
    )
    parser.add_argument("--model-name", default="BAAI/bge-m3", help="Embedding model.")
    parser.add_argument("--device", default=None, help="Model device (cpu, cuda, mps).")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size.")
    parser.add_argument("--seed", type=int, default=1729, help="Base random seed.")
    parser.add_argument("--top-k", type=int, default=5, help="Ranked devices kept per query.")
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10000,
        help="Bootstrap resamples for the paired Top-1 intervals. 0 disables.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    drift = subparsers.add_parser(
        "drift",
        help="Queries containing only ACEs with drifted hostnames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    drift.add_argument(
        "--subset",
        choices=["full", "high-domain"],
        default="full",
        help="Device subset: all devices, or devices rich in domain ACEs.",
    )
    drift.add_argument("--variants", type=int, default=10, help="Drift variants per device.")
    drift.add_argument(
        "--fraction",
        type=float,
        default=0.10,
        help="Fraction of eligible ACEs perturbed per variant (capped at 3).",
    )
    drift.add_argument(
        "--high-domain-threshold",
        type=int,
        default=10,
        help="Minimum domain ACEs for the high-domain subset.",
    )

    mixed = subparsers.add_parser(
        "mixed",
        help="Queries mixing exact, drifted, and unseen ACEs over a grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mixed.add_argument(
        "--retained-fractions",
        type=float,
        nargs="+",
        default=[0.10, 0.25, 0.50],
        help="Fractions of the profile observed at runtime.",
    )
    mixed.add_argument(
        "--domain-fractions",
        type=float,
        nargs="+",
        default=[0.00, 0.25, 0.50, 1.00],
        help="Fractions of retained domain ACEs whose hostnames drift.",
    )
    mixed.add_argument(
        "--novel-counts",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Number of query ACEs made unseen in the correct reference.",
    )
    mixed.add_argument(
        "--seeds-per-device",
        type=int,
        default=10,
        help="Seeded queries per device per grid cell.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = read_compact_profiles(args.compact_dir)
    bank, transform = load_whitened_reference(args.raw_npz)

    if args.command == "drift":
        queries = build_drift_queries(
            profiles,
            subset=args.subset,
            variants=args.variants,
            fraction=args.fraction,
            seed=args.seed,
            high_domain_threshold=args.high_domain_threshold,
        )
        config = {
            "command": "drift",
            "subset": args.subset,
            "variants": args.variants,
            "fraction": args.fraction,
        }
    else:
        queries = []
        for retained_fraction in args.retained_fractions:
            for domain_fraction in args.domain_fractions:
                for novel_count in args.novel_counts:
                    queries.extend(
                        build_mixed_queries(
                            profiles,
                            retained_fraction=retained_fraction,
                            domain_fraction=domain_fraction,
                            novel_count=novel_count,
                            seeds_per_device=args.seeds_per_device,
                            seed=args.seed,
                        )
                    )
        config = {
            "command": "mixed",
            "retained_fractions": args.retained_fractions,
            "domain_fractions": args.domain_fractions,
            "novel_counts": args.novel_counts,
            "seeds_per_device": args.seeds_per_device,
        }
    if not queries:
        raise ValueError("No queries generated.")

    vector_by_text = vectors_for_queries(
        bank,
        transform,
        queries,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )
    scored = score_queries(bank, queries, vector_by_text, top_k=args.top_k)

    result: dict[str, object] = {
        "config": {**config, "seed": args.seed, "top_k": args.top_k},
        "query_count": len(queries),
        "device_count": len({query.expected_device for query in queries}),
        "summary": summarise_results(scored, args.top_k),
    }
    if args.command == "mixed":
        result["by_exact_hits"] = summarise_bins(scored)
    if args.bootstrap_resamples > 0:
        result["paired_top1_bootstrap"] = paired_top1_bootstrap(
            scored, resamples=args.bootstrap_resamples, seed=args.seed
        )

    print(
        f"Scored {result['query_count']} {config['command']} queries "
        f"from {result['device_count']} devices."
    )
    print("method             top1    topK     mrr   abstain")
    for method, row in result["summary"].items():
        topk_key = next(key for key in row if key.startswith("top") and key != "top1")
        print(
            f"{method:<18} "
            f"{row['top1']:.4f}  {row[topk_key]:.4f}  {row['mrr']:.4f}  {row['abstain_rate']:.4f}"
        )
    for label, row in result.get("by_exact_hits", {}).items():
        parts = "  ".join(f"{method}={row[method]:.4f}" for method in METHODS)
        print(f"exact hits {label:>3} ({row['queries']:>4} queries): {parts}")
    for name, row in result.get("paired_top1_bootstrap", {}).items():
        low, high = row["episode_ci95"]
        cluster_low, cluster_high = row["cluster_ci95"]
        print(
            f"{name}: {row['difference']:+.4f} "
            f"(episode 95% CI [{low:+.4f}, {high:+.4f}], "
            f"cluster 95% CI [{cluster_low:+.4f}, {cluster_high:+.4f}])"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.output}.")


if __name__ == "__main__":
    main()
