"""Compare matching methods using controlled ACE queries.

The script creates repeatable controlled queries from the supplied per-ACE
embedding bank. It compares exact overlap, mean pooling, and MaxSim.

Conditions:

- single-unseen: one query per ACE; that ACE is removed from all references.
- unseen-family: one query per (device, ACE family); the whole family is
  removed from all references. Families are found by clustering ACE embeddings.
- unseen-set: three ACEs drawn from distinct families, removed from all
  references. Ten seeded queries per eligible device.

Positive score ties receive fractional Top-1 credit. Reciprocal rank is
averaged over the ranks occupied by a tie. If all candidate scores are zero,
the method abstains and receives zero credit. The summary also reports paired
bootstrap confidence intervals for the Top-1 difference between MaxSim and
each baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matching.bank import AceBank, load_ace_bank
from matching.episodes import (
    Episode,
    build_family_episodes,
    build_set_episodes,
    build_single_unseen_episodes,
)
from matching.families import family_by_text
from matching.scoring import score_episode, summarise_results
from matching.stats import paired_top1_bootstrap


def query_record(bank: AceBank, episode: Episode, scores: dict[str, object]) -> dict[str, object]:
    return {
        "query_id": episode.episode_id,
        "condition": episode.mode,
        "expected_device": episode.expected_device,
        "query_aces": [bank.ace_texts[idx] for idx in episode.query_indices],
        "exact_aces": [bank.ace_texts[idx] for idx in episode.exact_indices],
        "unseen_aces": [bank.ace_texts[idx] for idx in episode.unseen_indices],
        "scores": scores,
    }


def run_controlled_evaluation(
    *,
    embedding_npz: Path,
    raw_npz: Path,
    condition: str,
    query_size: int,
    queries_per_device: int,
    min_profile_size: int,
    family_top_k: int,
    family_threshold: float,
    seed: int,
    top_k: int,
    examples: int,
    bootstrap_resamples: int,
) -> dict[str, object]:
    bank = load_ace_bank(embedding_npz)

    if condition == "single-unseen":
        episodes = build_single_unseen_episodes(bank)
    else:
        families = family_by_text(bank, raw_npz, family_top_k, family_threshold)
        if condition == "unseen-family":
            episodes = build_family_episodes(bank, families)
        else:
            episodes = build_set_episodes(
                bank,
                families,
                seeds_per_device=queries_per_device,
                min_profile_size=min_profile_size,
                query_size=query_size,
                base_seed=seed,
            )
    if not episodes:
        raise ValueError("No controlled queries generated.")

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
            "condition": condition,
            "seed": seed,
            "top_k": top_k,
        },
        "query_count": len(episodes),
        "device_count": len({episode.expected_device for episode in episodes}),
        "summary": summarise_results(scored, top_k),
        "examples": [
            query_record(bank, item["episode"], item["scores"])
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
        "--condition",
        choices=["single-unseen", "unseen-family", "unseen-set"],
        default="single-unseen",
        help="Controlled query type.",
    )
    parser.add_argument(
        "--query-size",
        type=int,
        default=3,
        help="Number of ACEs in each unseen-set query.",
    )
    parser.add_argument(
        "--queries-per-device",
        type=int,
        default=10,
        help="Seeded queries per eligible device in the unseen-set condition.",
    )
    parser.add_argument(
        "--min-profile-size",
        type=int,
        default=12,
        help="Minimum unique ACEs a device needs for the unseen-set condition.",
    )
    parser.add_argument(
        "--family-top-k",
        type=int,
        default=5,
        help="Neighbors considered when clustering ACEs into families.",
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
        help="Random seed for repeatable query generation.",
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
        help="Number of example queries included in the result.",
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
        f"Scored {result['query_count']} {result['config']['condition']} queries "
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
        low, high = row["query_ci95"]
        cluster_low, cluster_high = row["cluster_ci95"]
        print(
            f"{name}: {row['difference']:+.4f} "
            f"(query 95% CI [{low:+.4f}, {high:+.4f}], "
            f"cluster 95% CI [{cluster_low:+.4f}, {cluster_high:+.4f}])"
        )


def main() -> None:
    args = parse_args()
    result = run_controlled_evaluation(
        embedding_npz=args.embedding_npz,
        raw_npz=args.raw_npz,
        condition=args.condition,
        query_size=args.query_size,
        queries_per_device=args.queries_per_device,
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
