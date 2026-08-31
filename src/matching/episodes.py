"""Synthetic episode construction for the controlled evaluation modes."""

from __future__ import annotations

import random
from collections import Counter, OrderedDict
from dataclasses import dataclass

from matching.bank import AceBank, stable_seed_offset, unique_indices_for_device


@dataclass(frozen=True)
class Episode:
    episode_id: str
    mode: str
    expected_device: str
    query_indices: tuple[int, ...]
    exact_indices: tuple[int, ...]
    unseen_indices: tuple[int, ...]
    removed_texts: frozenset[str]


def make_unseen_episode(
    bank: AceBank, mode: str, device: str, name: str, query: tuple[int, ...]
) -> Episode:
    return Episode(
        episode_id=f"{device}/{mode}/{name}",
        mode=mode,
        expected_device=device,
        query_indices=query,
        exact_indices=(),
        unseen_indices=query,
        removed_texts=frozenset(bank.ace_texts[idx] for idx in query),
    )


def build_single_unseen_episodes(bank: AceBank) -> list[Episode]:
    """One episode per (device, unique ACE); the ACE is removed everywhere."""
    episodes: list[Episode] = []
    for device in bank.indices_by_device:
        for num, idx in enumerate(unique_indices_for_device(bank, device)):
            episodes.append(
                make_unseen_episode(bank, "single-unseen", device, f"{num:04d}", (idx,))
            )
    return episodes


def build_family_episodes(bank: AceBank, families: dict[str, str]) -> list[Episode]:
    """One episode per (device, family) with at least two members in the device."""
    family_sizes = Counter(families.values())
    episodes: list[Episode] = []
    for device in bank.indices_by_device:
        buckets: OrderedDict[str, list[int]] = OrderedDict()
        for idx in unique_indices_for_device(bank, device):
            buckets.setdefault(families[bank.ace_texts[idx]], []).append(idx)
        for family_id, indices in buckets.items():
            if family_sizes[family_id] < 2 or len(indices) < 2:
                continue
            episodes.append(
                make_unseen_episode(bank, "unseen-family", device, family_id, tuple(indices))
            )
    return episodes


def build_set_episodes(
    bank: AceBank,
    families: dict[str, str],
    *,
    seeds_per_device: int,
    min_profile_size: int,
    query_size: int,
    base_seed: int,
) -> list[Episode]:
    """Seeded queries of ACEs drawn from distinct families where possible."""
    episodes: list[Episode] = []
    for device in bank.indices_by_device:
        unique_indices = unique_indices_for_device(bank, device)
        if len(unique_indices) < min_profile_size:
            continue
        buckets: dict[str, list[int]] = {}
        for idx in unique_indices:
            buckets.setdefault(families[bank.ace_texts[idx]], []).append(idx)
        family_ids = sorted(buckets)
        for offset in range(seeds_per_device):
            rng = random.Random(base_seed + offset * 1009 + stable_seed_offset(device))
            if len(family_ids) >= query_size:
                chosen = sorted(rng.sample(family_ids, query_size))
                query = tuple(sorted(rng.choice(buckets[fid]) for fid in chosen))
            else:
                query = tuple(sorted(rng.sample(unique_indices, query_size)))
            episodes.append(
                make_unseen_episode(bank, "unseen-set", device, f"{offset:03d}", query)
            )
    return episodes


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
                episodes.append(
                    make_unseen_episode(bank, mode, device, f"{episode_num:03d}", query)
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
                episodes.append(
                    Episode(
                        episode_id=f"{device}/partial/{episode_num:03d}",
                        mode=mode,
                        expected_device=device,
                        query_indices=tuple(selected),
                        exact_indices=exact,
                        unseen_indices=unseen,
                        removed_texts=frozenset(bank.ace_texts[idx] for idx in unseen),
                    )
                )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    if not episodes:
        raise ValueError("No episodes generated. Try smaller query counts.")
    return episodes
