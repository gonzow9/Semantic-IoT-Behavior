# Semantic IoT Behavior

This repository contains the compact data artifacts and pipeline used
for semantic identification of IoT devices from MUD behavioral primitives.

The core idea is simple: convert each MUD Access Control Entry (ACE) into one
compact behavioral text line, embed each ACE, optionally whiten the embedding
space, and compare runtime behavior against reference devices using exact
overlap or semantic MaxSim scoring.

## What Is Included

- 28 public MUD profiles in `data/mud_raw/`
- Compact ACE text in `data/mud_compact/`
- 1023 ACE instances, 710 unique compact ACE lines
- BGE-M3 reference embedding banks in `embeddings/bge_m3/`
- OpenAI `text-embedding-3-large` 1024-dimensional reference banks in `embeddings/openai/`
- Controlled-runtime and real-traffic summary results in `results/`
- Paper figures in `figures/paper/`
- Small Python modules for the main pipeline

## Repository Layout

```text
data/
  mud_raw/                       original canonical MUD JSON files
  mud_compact/                   one compact ACE text file per device
embeddings/
  bge_m3/                        reference embedding banks
  openai/                        OpenAI reference embedding banks
results/
  geometry/                      embedding geometry diagnostics
  controlled_runtime/            controlled evaluation summaries
  real_traffic/                  real-flow evaluation summaries
figures/
  paper/                         selected paper figures
src/semantic_iot_behavior/       minimal reproduction code
docs/                            artifact and reproduction notes
```

## Quickstart

Create an environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Inspect the shipped BGE-M3 per-ACE embedding bank:

```bash
python - <<'PY'
import numpy as np

data = np.load("embeddings/bge_m3/reference_per_ace.npz", allow_pickle=True)
print(data["embeddings"].shape)
print(data.files)
PY
```

Regenerate compact ACE text from raw MUD JSON:

```bash
python -m semantic_iot_behavior.compact_mud \
  --input-dir data/mud_raw \
  --output-dir data/mud_compact
```

Regenerate the BGE-M3 per-ACE reference bank:

```bash
python -m semantic_iot_behavior.embed \
  --input-dir data/mud_compact \
  --pool per-ace \
  --output embeddings/bge_m3/reference_per_ace.npz
```

The OpenAI embedding banks in `embeddings/openai/` are included as shipped
artifacts. They were generated with `text-embedding-3-large` using 1024 output
dimensions.

Apply reference-only whitening:

```bash
python -m semantic_iot_behavior.whiten \
  --reference embeddings/bge_m3/reference_per_ace.npz \
  --input embeddings/bge_m3/reference_per_ace.npz \
  --output embeddings/bge_m3/reference_per_ace_whitened_k256.npz \
  --k 256
```

Run an exact-overlap self-retrieval smoke check:

```bash
python -m semantic_iot_behavior.score exact \
  --reference-dir data/mud_compact \
  --query-dir data/mud_compact \
  --method jaccard \
  --output results/local_exact_self.json
```

Run a small Section 3-style evolving-runtime demo:

```bash
python -m semantic_iot_behavior.runtime_behavior_demo \
  --mode strict-unseen \
  --query-size 3 \
  --episodes-per-device 3
```

This samples runtime ACEs from each device, removes those exact ACEs from every
candidate reference profile, and compares exact overlap against mean-pooled
semantic scoring and ACE-level MaxSim. A mixed partial-observation version is:

```bash
python -m semantic_iot_behavior.runtime_behavior_demo \
  --mode partial \
  --exact-count 2 \
  --unseen-count 2 \
  --episodes-per-device 3
```

Use `--embedding-npz embeddings/openai/reference_per_ace_whitened_k256.npz` to
run the same demo with the shipped OpenAI embeddings.

## Main Artifacts

The most useful embedding file is:

```text
embeddings/bge_m3/reference_per_ace_whitened_k256.npz
```

It contains:

- `embeddings`: `(1023, 256)` float32 whitened ACE vectors
- `devices`: device label for each ACE row
- `ace_texts`: compact ACE text for each row
- `names`: retained compatibility label array

See `docs/contents.md` for the full file inventory.

## Paper Context

This repository supports the paper:

> Semantic Identification of IoT Devices from Behavioral Primitives

The paper evaluates MUD ACE embeddings under three settings:

- embedding geometry on canonical MUD profiles
- controlled runtime variations, including endpoint drift and partial observation
- real IoT traffic converted into ACE-like behavioral primitives
