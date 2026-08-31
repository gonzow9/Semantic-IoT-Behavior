"""The per-ACE embedding bank and small vector helpers."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from runtime_score import clean_device_name


@dataclass(frozen=True)
class AceBank:
    path: Path
    embeddings: np.ndarray
    labels: list[str]
    clean_labels: list[str]
    ace_texts: list[str]
    indices_by_device: OrderedDict[str, np.ndarray]


def normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


def normalise_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def stable_seed_offset(name: str) -> int:
    """Deterministic per-name seed offset, stable across Python runs."""
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)


def build_ace_bank(
    path: Path,
    embeddings: np.ndarray,
    labels: list[str],
    ace_texts: list[str],
) -> AceBank:
    embeddings = normalise_rows(embeddings.astype(np.float32, copy=False))
    if len(labels) != embeddings.shape[0] or len(ace_texts) != embeddings.shape[0]:
        raise ValueError(f"{path} has mismatched row counts.")

    clean_labels = [clean_device_name(label) for label in labels]
    grouped: dict[str, list[int]] = {}
    for idx, device in enumerate(clean_labels):
        grouped.setdefault(device, []).append(idx)

    return AceBank(
        path=path,
        embeddings=embeddings,
        labels=labels,
        clean_labels=clean_labels,
        ace_texts=ace_texts,
        indices_by_device=OrderedDict(
            (device, np.asarray(indices, dtype=np.int64))
            for device, indices in sorted(grouped.items())
        ),
    )


def load_ace_bank(path: Path) -> AceBank:
    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data or "ace_texts" not in data:
        raise ValueError(f"{path} must contain embeddings and ace_texts arrays.")

    label_key = "devices" if "devices" in data else "names" if "names" in data else None
    if label_key is None:
        raise ValueError(f"{path} must contain either devices or names labels.")

    return build_ace_bank(
        path,
        data["embeddings"],
        [str(value) for value in data[label_key]],
        [str(value) for value in data["ace_texts"]],
    )


def unique_indices_for_device(bank: AceBank, device: str) -> list[int]:
    """Return one row index per unique ACE text for a device."""
    by_text: OrderedDict[str, int] = OrderedDict()
    for idx in bank.indices_by_device[device]:
        by_text.setdefault(bank.ace_texts[int(idx)], int(idx))
    return list(by_text.values())


def mean_pool(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros(matrix.shape[1], dtype=np.float32)
    return normalise_vector(matrix.mean(axis=0))
