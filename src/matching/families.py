"""ACE families: reciprocal nearest-neighbour clustering of ACE embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from matching.bank import AceBank, normalise_rows


def family_space_vectors(raw_npz: Path, unique_texts: list[str]) -> np.ndarray:
    """Whiten the unique raw ACE embeddings to form the family space.

    Each unique ACE text is represented by the mean of its raw embedding rows
    (the same text can appear in several profiles). The means are whitened
    with a transform fitted on the means themselves, keeping up to 256
    dimensions, and L2-normalised so cosine similarity is a dot product.
    """
    data = np.load(raw_npz, allow_pickle=True)
    if "embeddings" not in data or "ace_texts" not in data:
        raise ValueError(f"{raw_npz} must contain embeddings and ace_texts arrays.")
    raw = data["embeddings"].astype(np.float32, copy=False)
    rows_by_text: dict[str, list[int]] = {}
    for idx, text in enumerate(data["ace_texts"]):
        rows_by_text.setdefault(str(text), []).append(idx)
    missing = [text for text in unique_texts if text not in rows_by_text]
    if missing:
        raise ValueError(f"{raw_npz} is missing {len(missing)} ACE texts, e.g. {missing[0]!r}")

    unique = np.vstack(
        [raw[rows_by_text[text]].mean(axis=0) for text in unique_texts]
    ).astype(np.float32)

    centered = unique - unique.mean(axis=0)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    noise = max(centered.shape) * np.finfo(centered.dtype).eps * singular_values.max()
    keep = min(256, int((singular_values > noise).sum()))
    scale = np.sqrt(unique.shape[0] - 1) / singular_values[:keep]
    return normalise_rows((centered @ components[:keep].T) * scale)


def build_families(
    vectors: np.ndarray,
    *,
    top_k: int,
    threshold: float,
) -> list[list[int]]:
    """Group vectors into families via reciprocal top-k neighbours.

    Two ACEs join the same family when each is among the other's top-k cosine
    neighbours and their similarity is at least the threshold. Families are the
    connected components of these mutual links, sorted largest first.
    """
    count = vectors.shape[0]
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    neighbours = np.argsort(-sims, axis=1)[:, :top_k]
    neighbour_sets = [
        {int(j) for j in row if sims[i, j] >= threshold}
        for i, row in enumerate(neighbours)
    ]

    parent = list(range(count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(count):
        for j in neighbour_sets[i]:
            if i in neighbour_sets[j]:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(count):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda members: (-len(members), min(members)))


def family_by_text(bank: AceBank, raw_npz: Path, top_k: int, threshold: float) -> dict[str, str]:
    """Map every unique ACE text to a family id such as fam_0001."""
    unique_texts: list[str] = []
    seen: set[str] = set()
    for text in bank.ace_texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)
    vectors = family_space_vectors(raw_npz, unique_texts)
    families: dict[str, str] = {}
    for number, members in enumerate(build_families(vectors, top_k=top_k, threshold=threshold), 1):
        for member in members:
            families[unique_texts[member]] = f"fam_{number:04d}"
    return families
