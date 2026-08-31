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

Scoring rule: a method only makes a prediction when its best score is positive and a single device holds it. Otherwise the
query counts as a miss. The summary also reports paired bootstrap confidence
intervals for the Top-1 difference between MaxSim and each baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matching.bank import AceBank, load_ace_bank
from matching.episodes import (
    Episode,
    build_episodes,
    build_family_episodes,
    build_set_episodes,
    build_single_unseen_episodes,
)
from matching.families import family_by_text
from matching.scoring import score_episode, summarise_results
from matching.stats import paired_top1_bootstrap


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
