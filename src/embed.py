"""Generate whole-profile, mean-pooled, or per-ACE embedding banks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


def compact_files(input_dir: Path) -> list[Path]:
    files = sorted(path for path in input_dir.rglob("*.txt") if path.is_file())
    if not files:
        raise ValueError(f"No compact .txt files found in {input_dir}")
    return files


def read_rules(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_model(model_name: str, device: str | None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Install the embedding dependencies first: python -m pip install -r requirements.txt"
        ) from exc
    return SentenceTransformer(model_name, device=device)


def encode_texts(model, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)


def embed_per_ace(model, files: list[Path], batch_size: int) -> tuple[np.ndarray, list[str], list[str]]:
    devices: list[str] = []
    ace_texts: list[str] = []
    for path in files:
        rules = read_rules(path)
        devices.extend([path.stem] * len(rules))
        ace_texts.extend(rules)
    if not ace_texts:
        raise ValueError("No ACE lines found in compact input files.")
    embeddings = encode_texts(model, ace_texts, batch_size)
    return embeddings, devices, ace_texts


def embed_whole(model, files: list[Path], batch_size: int) -> tuple[np.ndarray, list[str]]:
    texts = [path.read_text(encoding="utf-8") for path in files]
    embeddings = encode_texts(model, texts, batch_size)
    names = [path.stem for path in files]
    return embeddings, names


def embed_mean_pool(model, files: list[Path], batch_size: int) -> tuple[np.ndarray, list[str]]:
    vectors: list[np.ndarray] = []
    names: list[str] = []
    for path in files:
        rules = read_rules(path)
        if not rules:
            continue
        rule_vectors = encode_texts(model, rules, batch_size)
        vectors.append(rule_vectors.mean(axis=0))
        names.append(path.stem)
    if not vectors:
        raise ValueError("No ACE lines found in compact input files.")
    return l2_normalise(np.vstack(vectors).astype(np.float32)), names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/mud/mud_compact"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool", choices=["whole", "mean-ace", "per-ace"], default="per-ace")
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model(args.model_name, args.device)
    files = compact_files(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.pool == "per-ace":
        embeddings, devices, ace_texts = embed_per_ace(model, files, args.batch_size)
        np.savez(
            args.output,
            embeddings=embeddings,
            devices=np.array(devices),
            ace_texts=np.array(ace_texts),
        )
    elif args.pool == "mean-ace":
        embeddings, names = embed_mean_pool(model, files, args.batch_size)
        np.savez(args.output, embeddings=embeddings, names=np.array(names))
    else:
        embeddings, names = embed_whole(model, files, args.batch_size)
        np.savez(args.output, embeddings=embeddings, names=np.array(names))

    print(f"Saved {args.output} with shape {embeddings.shape}.")


if __name__ == "__main__":
    main()
