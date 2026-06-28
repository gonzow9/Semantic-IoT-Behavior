# Semantic IoT Behavior

This repository contains the data artifacts along with scripts for
semantic identification of IoT devices from MUD behavioral primitives.

The pipeline is:

1. Convert each MUD Access Control Entry (ACE) into a compact behavior line.
2. Embed each behavior line.
3. Optionally whiten the embedding space.
4. Compare runtime behavior with reference devices using exact overlap or
   semantic MaxSim scoring.

The data was constructed by analyzing a public dataset of MUD files from [UNSW IoT Analytics](https://iotanalytics.unsw.edu.au/mudprofiles.html), collected by researchers at UNSW Sydney.

## What Is Included

- 28 public MUD profiles in `data/ref_mud/raw/`
- Compact ACE text in `data/ref_mud/compact/`
- Real-traffic runtime ACE CSVs in `data/runtime_aces/real_traffic/`
- BGE-M3 reference embeddings in `data/ref_embeddings/bge/`
- OpenAI `text-embedding-3-large` reference embeddings in `data/ref_embeddings/openai/`
- Python scripts in `src/` for the main pipeline.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick Checks

Inspect one shipped BGE-M3 whole-profile embedding:

```bash
python - <<'PY'
import json

path = "data/ref_embeddings/bge/whole/compact/amazonEchoMud_embedding_compact.json"
with open(path, encoding="utf-8") as handle:
    item = json.load(handle)
print(item["device"])
print(item["embedding_dim"])
print(len(item["embedding"]))
PY
```

Inspect the shipped BGE-M3 per-ACE embedding bank:

```bash
python - <<'PY'
import numpy as np

bank = np.load("data/ref_embeddings/bge/per_ace/raw/reference_per_ace.npz", allow_pickle=True)
print(bank["embeddings"].shape)
print(bank.files)
PY
```

Run an exact-overlap self-retrieval sanity check:

```bash
python src/runtime_score.py exact \
  --reference-dir data/ref_mud/compact \
  --query-dir data/ref_mud/compact \
  --method jaccard \
  --output tmp/local_exact_self.json
```

Run the small Section 3 evolving-runtime demo:

```bash
python src/runtime_matches.py \
  --mode strict-unseen \
  --query-size 3 \
  --episodes-per-device 3 \
  --output tmp/local_section3_strict_unseen_demo.json
```

Use `--embedding-npz data/ref_embeddings/openai/per_ace/whitened_k256/reference_per_ace_whitened_k256.npz`
to run the demo with the shipped OpenAI embeddings.

## Reproducing Core Artifacts

Regenerate compact ACE text:

```bash
python src/convert_mud_compact.py \
  --input-dir data/ref_mud/raw \
  --output-dir data/ref_mud/compact
```

Regenerate the BGE-M3 per-ACE reference bank:

```bash
python src/gen_emb.py \
  --input-dir data/ref_mud/compact \
  --pool per-ace \
  --model-name BAAI/bge-m3 \
  --output data/ref_embeddings/bge/per_ace/raw/reference_per_ace.npz
```

Apply reference-only whitening:

```bash
python src/gen_whiten_emb.py \
  --reference data/ref_embeddings/bge/per_ace/raw/reference_per_ace.npz \
  --input data/ref_embeddings/bge/per_ace/raw/reference_per_ace.npz \
  --output data/ref_embeddings/bge/per_ace/whitened_k256/reference_per_ace_whitened_k256.npz \
  --metadata tmp/local_whitening_metadata.json \
  --k 256
```

For query banks, keep the same `--reference` and change only `--input` and
`--output`. This keeps whitening fitted on the canonical reference bank.

The OpenAI banks are shipped artifacts. Regenerating them requires API
credentials, so they are not part of the default local reproduction path.

## Main Artifacts

| Path | Contents |
| --- | --- |
| `data/ref_mud/raw/` | 28 canonical MUD JSON profiles |
| `data/ref_mud/compact/` | Compact ACE text plus `reduction_stats.json` |
| `data/runtime_aces/real_traffic/` | Per-device runtime flow-to-ACE CSVs for the 810,490-flow real-traffic run |
| `data/ref_embeddings/bge/whole/raw/` | Per-device BGE-M3 embeddings for raw MUD JSON |
| `data/ref_embeddings/bge/whole/compact/` | Per-device BGE-M3 embeddings for compact MUD text |
| `data/ref_embeddings/openai/whole/raw/` | Per-device OpenAI embeddings for raw MUD JSON |
| `data/ref_embeddings/openai/whole/compact/` | Per-device OpenAI embeddings for compact MUD text |
| `data/ref_embeddings/bge/per_ace/whitened_k256/reference_per_ace_whitened_k256.npz` | Main whitened BGE-M3 per-ACE bank |
| `data/ref_embeddings/openai/per_ace/whitened_k256/reference_per_ace_whitened_k256.npz` | Whitened OpenAI per-ACE bank |

The compact canonical data contains 1023 ACE instances and 710 unique compact
ACE lines. See `data/README.md` for data-specific notes.

## Cite Our Data and Code

```bibtex
@misc{witt2026semanticidentifyiot,
  title={Semantic Identification of IoT Devices from Behavioral Primitives},
  author={Samuel Witt and Hassan Habibi Gharakheili},
  year={2026},
  eprint={2606.12793},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2606.12793},
}
```
