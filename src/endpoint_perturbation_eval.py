"""Run endpoint perturbation and mixed partial observation.

Two controlled conditions are available:

- endpoint-perturbation: domain names in selected ACEs are changed while
  protocol, direction, and port stay the same. The query contains only the
  endpoint-perturbed ACEs. The "full" subset uses every device;
  "high-domain" keeps devices with at least ten perturbable domain-name ACEs.
- mixed-partial-observation: a query combines exact ACEs,
  endpoint-perturbed ACEs, and optionally one unseen ACE. A grid over the
  retained fraction, perturbation fraction, and unseen count is evaluated.
  Top-1 is also grouped by the number of exact ACE matches against the source
  reference profile.

Endpoint-perturbed ACE texts are new strings, so they must be embedded. The first run
downloads the BGE-M3 model. All embeddings (references and queries) are
whitened with one transform fitted on the raw reference bank.

The hostname changes live in ``endpoint_perturbation/perturb.py`` and the query
construction in ``endpoint_perturbation/queries.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from endpoint_perturbation.queries import (
    Query,
    build_endpoint_perturbation_queries,
    build_mixed_partial_observation_queries,
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

    Texts already in the reference bank reuse its rows. New endpoint-perturbed
    texts are embedded with the sentence-transformer model and whitened with
    the reference transform.
    """
    vector_by_text: dict[str, np.ndarray] = {}
    for idx, text in enumerate(bank.ace_texts):
        vector_by_text.setdefault(text, bank.embeddings[idx])

    new_texts = sorted(
        {text for query in queries for text in query.query_texts if text not in vector_by_text}
    )
    if new_texts:
        print(
            f"Embedding {len(new_texts)} endpoint-perturbed ACE texts "
            f"with {model_name}..."
        )
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
            credit = sum(
                float(item["scores"][method]["top1_credit"]) for item in rows
            )
            bins[label][method] = credit / len(rows)
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

    subparsers = parser.add_subparsers(dest="condition", required=True)

    endpoint_perturbation = subparsers.add_parser(
        "endpoint-perturbation",
        help="Queries containing only ACEs with perturbed domain names.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    endpoint_perturbation.add_argument(
        "--subset",
        choices=["full", "high-domain"],
        default="full",
        help="Device subset: all devices, or devices rich in domain ACEs.",
    )
    endpoint_perturbation.add_argument(
        "--variants",
        type=int,
        default=10,
        help="Endpoint-perturbation variants per device.",
    )
    endpoint_perturbation.add_argument(
        "--perturbation-fraction",
        type=float,
        default=0.10,
        help="Fraction of eligible ACEs perturbed per variant (capped at 3).",
    )
    endpoint_perturbation.add_argument(
        "--high-domain-threshold",
        type=int,
        default=10,
        help="Minimum domain ACEs for the high-domain subset.",
    )

    mixed_partial_observation = subparsers.add_parser(
        "mixed-partial-observation",
        help="Queries combining exact, endpoint-perturbed, and unseen ACEs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mixed_partial_observation.add_argument(
        "--retained-fractions",
        type=float,
        nargs="+",
        default=[0.10, 0.25, 0.50],
        help="Fractions of the profile observed at runtime.",
    )
    mixed_partial_observation.add_argument(
        "--perturbation-fractions",
        type=float,
        nargs="+",
        default=[0.00, 0.25, 0.50, 1.00],
        help="Fractions of retained domain-name ACEs to perturb.",
    )
    mixed_partial_observation.add_argument(
        "--unseen-counts",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Number of query ACEs made unseen in the correct reference.",
    )
    mixed_partial_observation.add_argument(
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

    if args.condition == "endpoint-perturbation":
        queries = build_endpoint_perturbation_queries(
            profiles,
            subset=args.subset,
            variants=args.variants,
            perturbation_fraction=args.perturbation_fraction,
            seed=args.seed,
            high_domain_threshold=args.high_domain_threshold,
        )
        config = {
            "condition": "endpoint-perturbation",
            "subset": args.subset,
            "variants": args.variants,
            "perturbation_fraction": args.perturbation_fraction,
        }
    else:
        queries = []
        for retained_fraction in args.retained_fractions:
            for perturbation_fraction in args.perturbation_fractions:
                for unseen_count in args.unseen_counts:
                    queries.extend(
                        build_mixed_partial_observation_queries(
                            profiles,
                            retained_fraction=retained_fraction,
                            perturbation_fraction=perturbation_fraction,
                            unseen_count=unseen_count,
                            seeds_per_device=args.seeds_per_device,
                            seed=args.seed,
                        )
                    )
        config = {
            "condition": "mixed-partial-observation",
            "retained_fractions": args.retained_fractions,
            "perturbation_fractions": args.perturbation_fractions,
            "unseen_counts": args.unseen_counts,
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
    if args.condition == "mixed-partial-observation":
        result["by_exact_hits"] = summarise_bins(scored)
    if args.bootstrap_resamples > 0:
        result["paired_top1_bootstrap"] = paired_top1_bootstrap(
            scored, resamples=args.bootstrap_resamples, seed=args.seed
        )

    print(
        f"Scored {result['query_count']} {config['condition']} queries "
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
        low, high = row["query_ci95"]
        cluster_low, cluster_high = row["cluster_ci95"]
        print(
            f"{name}: {row['difference']:+.4f} "
            f"(query 95% CI [{low:+.4f}, {high:+.4f}], "
            f"cluster 95% CI [{cluster_low:+.4f}, {cluster_high:+.4f}])"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.output}.")


if __name__ == "__main__":
    main()
