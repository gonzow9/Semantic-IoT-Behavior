# Semantic IoT Behavior

This repository contains the data artifacts and small reproduction scripts for
semantic identification of IoT devices from MUD behavioral primitives.

The pipeline is:

1. Convert each MUD Access Control Entry (ACE) into one compact behavior line.
2. Embed each behavior line.
3. Optionally whiten the embedding space.
4. Compare runtime behavior with reference devices using exact overlap or
   semantic MaxSim scoring.

## What Is Included

- 28 public MUD profiles in `data/mud/mud_raw/`
- Compact ACE text in `data/mud/mud_compact/`
- BGE-M3 reference embedding banks in `data/embeddings/bge_m3/`
- OpenAI `text-embedding-3-large` reference banks in `data/embeddings/openai/`
- Geometry, controlled-runtime, and real-traffic summaries in `analysis/`
- Small Python scripts in `src/` for the main pipeline.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick Checks

Inspect the shipped BGE-M3 per-ACE embedding bank:

```bash
python - <<'PY'
import numpy as np

bank = np.load("data/embeddings/bge_m3/reference_per_ace.npz", allow_pickle=True)
print(bank["embeddings"].shape)
print(bank.files)
PY
```

Run an exact-overlap self-retrieval sanity check:

```bash
python src/score.py exact \
  --reference-dir data/mud/mud_compact \
  --query-dir data/mud/mud_compact \
  --method jaccard \
  --output analysis/local_exact_self.json
```

Run the small Section 3 evolving-runtime demo:

```bash
python src/runtime_behavior_demo.py \
  --mode strict-unseen \
  --query-size 3 \
  --episodes-per-device 3 \
  --output analysis/local_section3_strict_unseen_demo.json
```

Use `--embedding-npz data/embeddings/openai/reference_per_ace_whitened_k256.npz`
to run the demo with the shipped OpenAI embeddings.

## Reproducing Core Artifacts

Regenerate compact ACE text:

```bash
python src/compact_mud.py \
  --input-dir data/mud/mud_raw \
  --output-dir data/mud/mud_compact
```

Regenerate the BGE-M3 per-ACE reference bank:

```bash
python src/embed.py \
  --input-dir data/mud/mud_compact \
  --pool per-ace \
  --model-name BAAI/bge-m3 \
  --output data/embeddings/bge_m3/reference_per_ace.npz
```

Apply reference-only whitening:

```bash
python src/whiten.py \
  --reference data/embeddings/bge_m3/reference_per_ace.npz \
  --input data/embeddings/bge_m3/reference_per_ace.npz \
  --output data/embeddings/bge_m3/reference_per_ace_whitened_k256.npz \
  --metadata analysis/local_whitening_metadata.json \
  --k 256
```

For query banks, keep the same `--reference` and change only `--input` and
`--output`. This keeps whitening fitted on the canonical reference bank.

The OpenAI banks are shipped artifacts. Regenerating them requires API
credentials, so they are not part of the default local reproduction path.

## Main Artifacts

| Path | Contents |
| --- | --- |
| `data/mud/mud_raw/` | 28 canonical MUD JSON profiles |
| `data/mud/mud_compact/` | Compact ACE text plus `reduction_stats.json` |
| `data/embeddings/bge_m3/reference_per_ace_whitened_k256.npz` | Main whitened BGE-M3 per-ACE bank |
| `data/embeddings/openai/reference_per_ace_whitened_k256.npz` | Whitened OpenAI per-ACE bank |
| `analysis/geometry/` | Embedding geometry diagnostics |
| `analysis/controlled_runtime/` | Controlled runtime evaluation summaries |
| `analysis/real_traffic/` | Real-traffic summary outputs |

The compact canonical data contains 1023 ACE instances and 710 unique compact
ACE lines. See `data/README.md` for data-specific notes.

## Paper Context

This repository supports the paper:

> Semantic Identification of IoT Devices from Behavioral Primitives

The paper evaluates MUD ACE embeddings under three settings:

- embedding geometry on canonical MUD profiles
- controlled runtime variations, including endpoint drift and partial observation
- real IoT traffic converted into ACE-like behavioral primitives

## Citation

```bibtex
@misc{witt2026semanticidentificationiotdevices,
  title={Semantic Identification of IoT Devices from Behavioral Primitives},
  author={Samuel Witt and Hassan Habibi Gharakheili},
  year={2026},
  eprint={2606.12793},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2606.12793},
}
```
