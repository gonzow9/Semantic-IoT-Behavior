"""Query construction for endpoint perturbation and mixed partial observation.

- Endpoint perturbation queries contain only ACEs with perturbed hostnames.
- Mixed partial observation queries combine exact, endpoint-perturbed, and
  optionally unseen ACEs over a grid of retained fraction, perturbation
  fraction, and unseen count.
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from endpoint_perturbation.perturb import (
    MAX_SELECTED_RULES_PER_PROFILE,
    has_perturbable_domain,
    hash_device,
    perturb_rule_with_retries,
)
from matching.bank import stable_seed_offset
from runtime_score import clean_device_name

@dataclass(frozen=True)
class Query:
    query_id: str
    expected_device: str
    cluster_key: str
    query_texts: tuple[str, ...]
    removed_texts: frozenset[str]
    removed_scope: str
    exact_hits: int


def read_compact_profiles(compact_dir: Path) -> OrderedDict[str, tuple[str, ...]]:
    profiles: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    for path in sorted(compact_dir.glob("*.txt")):
        rules = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        profiles[clean_device_name(path.stem)] = rules
    if not profiles:
        raise ValueError(f"No compact .txt profiles found in {compact_dir}")
    return profiles


def build_endpoint_perturbation_queries(
    profiles: OrderedDict[str, tuple[str, ...]],
    *,
    subset: str,
    variants: int,
    perturbation_fraction: float,
    seed: int,
    high_domain_threshold: int,
) -> list[Query]:
    reference_texts = frozenset(text for rules in profiles.values() for text in rules)
    queries: list[Query] = []
    for device, rules in profiles.items():
        candidate_count = sum(1 for rule in rules if has_perturbable_domain(rule))
        if subset == "high-domain" and candidate_count < high_domain_threshold:
            continue
        if candidate_count == 0:
            continue
        for variant in range(variants):
            rng = random.Random(
                seed + variant * 997 + candidate_count * 131 + hash_device(device)
            )
            eligible = [idx for idx, rule in enumerate(rules) if has_perturbable_domain(rule)]
            target = min(
                MAX_SELECTED_RULES_PER_PROFILE,
                max(1, round(len(eligible) * perturbation_fraction)),
            )
            selected = sorted(rng.sample(eligible, k=min(target, len(eligible))))
            perturbed_rules = [
                perturbed
                for idx in selected
                if (perturbed := perturb_rule_with_retries(rules[idx], rng, reference_texts))
                is not None
            ]
            if not perturbed_rules:
                continue
            queries.append(
                Query(
                    query_id=f"{device}/endpoint-perturbation/v{variant:02d}",
                    expected_device=device,
                    cluster_key=device,
                    query_texts=tuple(perturbed_rules),
                    removed_texts=frozenset(perturbed_rules),
                    removed_scope="all",
                    exact_hits=0,
                )
            )
    return queries


def build_mixed_partial_observation_queries(
    profiles: OrderedDict[str, tuple[str, ...]],
    *,
    retained_fraction: float,
    perturbation_fraction: float,
    unseen_count: int,
    seeds_per_device: int,
    seed: int,
) -> list[Query]:
    reference_texts = frozenset(text for rules in profiles.values() for text in rules)
    queries: list[Query] = []
    for device, rules in profiles.items():
        for offset in range(seeds_per_device):
            episode_seed = seed + offset * 1009 + stable_seed_offset(device)

            retained_rng = random.Random(
                episode_seed + stable_seed_offset(f"retain:{retained_fraction:.6f}")
            )
            count = min(len(rules), max(1, math.floor(len(rules) * retained_fraction + 0.5)))
            retained = sorted(retained_rng.sample(range(len(rules)), k=count))
            if unseen_count > 0 and len(retained) <= unseen_count:
                continue
            query_rules = [rules[idx] for idx in retained]

            perturbation_rng = random.Random(
                episode_seed + stable_seed_offset(f"domain:{perturbation_fraction:.6f}")
            )
            eligible = [
                pos for pos, rule in enumerate(query_rules) if has_perturbable_domain(rule)
            ]
            target = 0
            if perturbation_fraction > 0.0 and eligible:
                target = min(
                    len(eligible),
                    max(1, math.floor(len(eligible) * perturbation_fraction + 0.5)),
                )
            perturbed_count = 0
            if target:
                for pos in sorted(perturbation_rng.sample(eligible, k=target)):
                    perturbed = perturb_rule_with_retries(
                        query_rules[pos], perturbation_rng, reference_texts
                    )
                    if perturbed is not None:
                        query_rules[pos] = perturbed
                        perturbed_count += 1
            if perturbation_fraction > 0.0 and (not eligible or perturbed_count == 0):
                continue

            removed: frozenset[str] = frozenset()
            if unseen_count > 0:
                unseen_rng = random.Random(
                    episode_seed + stable_seed_offset(f"novel:{unseen_count}")
                )
                positions = sorted(unseen_rng.sample(range(len(query_rules)), k=unseen_count))
                removed = frozenset(rules[retained[pos]] for pos in positions)

            expected_reference = frozenset(rules) - removed
            queries.append(
                Query(
                    query_id=(
                        f"{device}/mixed-partial-observation/"
                        f"r{retained_fraction:.2f}_p{perturbation_fraction:.2f}"
                        f"_k{unseen_count}_s{offset:02d}"
                    ),
                    expected_device=device,
                    cluster_key=f"{device}/s{offset:02d}",
                    query_texts=tuple(query_rules),
                    removed_texts=removed,
                    removed_scope="expected",
                    exact_hits=len(frozenset(query_rules) & expected_reference),
                )
            )
    return queries
