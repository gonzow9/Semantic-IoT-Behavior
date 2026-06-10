"""Reference-only truncated PCA whitening for embedding banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


def fit_whitening(reference_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_embeddings = l2_normalise(reference_embeddings.astype(np.float32, copy=False))
    mean = reference_embeddings.mean(axis=0)
    centered = reference_embeddings - mean
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    threshold = max(centered.shape) * np.finfo(centered.dtype).eps * singular_values.max()
    keep = singular_values > threshold
    return mean, components[keep], singular_values[keep]


def apply_whitening(
    embeddings: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    singular_values: np.ndarray,
    k: int,
    n_reference: int,
) -> np.ndarray:
    embeddings = l2_normalise(embeddings.astype(np.float32, copy=False))
    effective_k = min(k, components.shape[0])
    projected = (embeddings - mean) @ components[:effective_k].T
    scale = np.sqrt(n_reference - 1) / singular_values[:effective_k]
    return l2_normalise(projected * scale)


def load_bank(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    key = "embeddings" if "embeddings" in data else "original_embeddings"
    embeddings = data[key].astype(np.float32, copy=False)
    extras = {name: data[name] for name in data.files if name != key}
    return embeddings, extras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--metadata", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_embeddings, _ = load_bank(args.reference)
    input_embeddings, extras = load_bank(args.input)
    mean, components, singular_values = fit_whitening(reference_embeddings)
    transformed = apply_whitening(
        input_embeddings,
        mean,
        components,
        singular_values,
        args.k,
        reference_embeddings.shape[0],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, embeddings=transformed, **extras)
    metadata = {
        "reference": str(args.reference),
        "input": str(args.input),
        "output": str(args.output),
        "reference_rows": int(reference_embeddings.shape[0]),
        "input_rows": int(input_embeddings.shape[0]),
        "input_dim": int(reference_embeddings.shape[1]),
        "rank": int(components.shape[0]),
        "k": int(min(args.k, components.shape[0])),
    }
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} with shape {transformed.shape}.")


if __name__ == "__main__":
    main()
