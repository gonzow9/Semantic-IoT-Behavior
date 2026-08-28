# Semantic IoT Behavior

Identify IoT devices by comparing their network behavior with known MUD
profiles. The repository includes the data and embeddings needed to run a
small demonstration immediately.

The main pipeline is:

1. Convert MUD Access Control Entries (ACEs) into short behavior lines.
2. Embed each behavior line.
3. Optionally whiten the embeddings.
4. Compare observed behavior with reference devices.

The reference MUD profiles come from the [UNSW IoT Analytics
dataset](https://iotanalytics.unsw.edu.au/mudprofiles.html).

## Quick Start

Clone the repository and enter its directory:

```bash
git clone https://github.com/gonzow9/Semantic-IoT-Behavior.git
cd Semantic-IoT-Behavior
```

Then create the Python environment and run the demo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/runtime_matches.py --examples 0
```

The final command runs a synthetic matching experiment with the shipped
BGE-M3 embeddings. It does not need an API key or a model download.

The first output line should be:

```text
Generated 140 strict-unseen episodes from 28 devices.
```

You should see a table with these four methods:

- `jaccard`: exact set overlap
- `exact_hit_count`: number of identical ACEs
- `mean_pool`: similarity between one average vector per profile
- `maxsim`: semantic matching between individual ACEs

The demo creates synthetic observations from the reference data. It is useful
for checking the method, but it is not a direct evaluation of the real-traffic
CSV files.

## Choose a Task

| Task | Command |
| --- | --- |
| Run the synthetic demo | `python src/runtime_matches.py` |
| Reproduce the unseen-ACE experiments | `python src/runtime_matches.py --mode single-unseen` |
| Reproduce the endpoint-drift experiments | `python src/drift_matches.py drift --help` |
| Reproduce the mixed partial observation grid | `python src/drift_matches.py mixed --help` |
| Compare compact text with exact matching | `python src/runtime_score.py exact --help` |
| Compare per-ACE embeddings with MaxSim | `python src/runtime_score.py maxsim --help` |
| Convert MUD JSON to compact text | `python src/convert_mud_compact.py --help` |
| Create BGE-M3 embeddings | `python src/gen_emb.py --help` |
| Whiten an embedding bank | `python src/gen_whiten_emb.py --help` |

All scripts support `--help`. The help text lists every argument and any
available default value.

## Run the Synthetic Demo

The default `strict-unseen` mode removes every query ACE from all reference
profiles before scoring. Exact matching therefore has no useful evidence.

```bash
python src/runtime_matches.py \
  --mode strict-unseen \
  --query-size 3 \
  --episodes-per-device 3 \
  --output tmp/strict_unseen_demo.json
```

The `partial` mode keeps some exact ACEs and removes the rest:

```bash
python src/runtime_matches.py \
  --mode partial \
  --exact-count 2 \
  --unseen-count 2 \
  --episodes-per-device 3 \
  --output tmp/partial_demo.json
```

Use the shipped OpenAI embeddings instead:

```bash
python src/runtime_matches.py \
  --embedding-npz data/ref_embeddings/openai/per_ace/whitened_k256/reference_per_ace_whitened_k256.npz
```

### Demo Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--embedding-npz` | Shipped whitened BGE-M3 bank | Per-ACE embedding bank to use |
| `--mode` | `strict-unseen` | `strict-unseen` or `partial` |
| `--episodes-per-device` | `5` | Synthetic observations created for each device |
| `--query-size` | `3` | ACEs per query in `strict-unseen` mode |
| `--exact-count` | `2` | ACEs kept in the references in `partial` mode |
| `--unseen-count` | `2` | ACEs removed from the references in `partial` mode |
| `--seed` | `1729` | Random seed for repeatable results |
| `--top-k` | `5` | Number of ranked devices kept per query |
| `--examples` | `3` | Example episodes included in the result |
| `--output` | None | Optional JSON output path |

`--query-size` only affects `strict-unseen`. `--exact-count` and
`--unseen-count` only affect `partial`.

## Reproduce the Controlled Experiments

These commands reproduce the paper's controlled evaluation. Scoring follows
the paper's abstention rule: a method only makes a prediction when its best
score is positive, so exact matching scores zero when the query shares no
ACE with any reference profile. Each run also reports paired bootstrap
confidence intervals (10,000 resamples) for the Top-1 difference between
MaxSim and each baseline.

The three unseen-behavior settings run on the shipped embeddings and need no
model download:

```bash
python src/runtime_matches.py --mode single-unseen   # 1023 episodes
python src/runtime_matches.py --mode unseen-family   # 103 episodes
python src/runtime_matches.py --mode unseen-set      # 240 episodes
```

- `single-unseen`: one query per ACE, removed from every reference profile.
- `unseen-family`: ACEs are clustered into families of related behaviors
  (reciprocal top-5 neighbours with cosine at least 0.75 in a whitened
  space). Each query removes one whole family from a device.
- `unseen-set`: three ACEs drawn from distinct families, removed everywhere.

The two drifted-endpoint settings change hostnames in selected ACEs while
protocol and port stay the same. The drifted texts are new strings, so they
must be embedded; the first run downloads the BGE-M3 model:

```bash
python src/drift_matches.py drift --subset full         # 280 queries
python src/drift_matches.py drift --subset high-domain  # 140 queries
```

The mixed partial observation grid builds queries that combine exact,
drifted, and unseen ACEs. Results are also grouped by the number of exact
ACE matches against the correct reference profile:

```bash
python src/drift_matches.py mixed   # 5772 queries
```

Devices often share ACEs (DNS, NTP, DHCP), so two devices can receive
exactly the same score for a query. The tie order then depends on
floating-point noise in the embeddings. This can move Top-1 by a few
queries per setting, between runs on different machines and against the
numbers reported in the paper.

## Score Prepared Queries

### Exact Text Matching

This command compares compact `.txt` profiles with the 28 reference devices:

```bash
python src/runtime_score.py exact \
  --reference-dir data/ref_mud/compact \
  --query-dir data/ref_mud/compact \
  --method jaccard \
  --output tmp/exact_self_check.json
```

This is a self-check, so the expected top-1 accuracy is `1.0`.

Each non-empty line in a query file must be one compact ACE. For example:

```text
egress ipv4 tcp (direction-initiated:from-device) dst:api.example.com dst-port:443
```

The expected device name is taken from the query filename. For multiple
observations of one device, put them in a directory named after that device:

```text
queries/
└── amazonEchoMud/
    ├── observation_001.txt
    └── observation_002.txt
```

Exact matching arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--reference-dir` | `data/ref_mud/compact` | Reference `.txt` profiles |
| `--query-dir` | Required | Query `.txt` profiles |
| `--method` | `jaccard` | `jaccard` or `exact_hit_count` |
| `--top-k` | `5` | Number of ranked devices kept per query |
| `--output` | Required | JSON output path |

### Semantic MaxSim Matching

MaxSim needs prepared per-ACE `.npz` banks:

```bash
python src/runtime_score.py maxsim \
  --reference-npz data/ref_embeddings/bge/per_ace/whitened_k256/reference_per_ace_whitened_k256.npz \
  --query-npz data/ref_embeddings/bge/per_ace/whitened_k256/reference_per_ace_whitened_k256.npz \
  --output tmp/maxsim_self_check.json
```

This is also a self-check, so the expected top-1 accuracy is `1.0`.

Each bank must contain:

- `embeddings`: one row per ACE
- `devices` or `names`: the device label for each row
- `ace_texts`: recommended, and required by `runtime_matches.py`

MaxSim arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--reference-npz` | Required | Reference per-ACE embedding bank |
| `--query-npz` | Required | Query per-ACE embedding bank |
| `--top-k` | `5` | Number of ranked devices kept per query |
| `--output` | Required | JSON output path |

## Real-Traffic CSV Files

`data/runtime_aces/` contains 26 real-traffic CSV files with 810,490 flow
rows. The `runtime_ace` column contains the compact behavior text for each
flow.

The current command-line scripts do not read these CSV files directly. To use
them with `runtime_score.py`, first prepare either:

- compact `.txt` query profiles for exact matching, or
- a labelled per-ACE `.npz` query bank for MaxSim matching.

Do not pass a CSV file directly to `--query-dir` or `--query-npz`.

## Rebuild the Main Artifacts

You do not need to rebuild anything to run the examples above.

### 1. Convert MUD JSON to Compact Text

```bash
python src/convert_mud_compact.py \
  --input-dir data/ref_mud/raw \
  --output-dir data/ref_mud/compact
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--input-dir` | `data/ref_mud/raw` | Directory containing MUD JSON files |
| `--output-dir` | `data/ref_mud/compact` | Directory for compact `.txt` files and reduction statistics |

### 2. Create BGE-M3 Embeddings

The first run downloads the selected model. A GPU is optional.

```bash
python src/gen_emb.py \
  --input-dir data/ref_mud/compact \
  --pool per-ace \
  --model-name BAAI/bge-m3 \
  --output data/ref_embeddings/bge/per_ace/raw/reference_per_ace.npz
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--input-dir` | `data/ref_mud/compact` | Directory containing compact `.txt` profiles |
| `--output` | Required | Output `.npz` path |
| `--pool` | `per-ace` | `per-ace`, `mean-ace`, or `whole` |
| `--model-name` | `BAAI/bge-m3` | Sentence Transformers model name or path |
| `--device` | Automatic | Model device, such as `cpu`, `cuda`, or `mps` |
| `--batch-size` | `32` | Texts encoded in each batch |

The pooling choices are:

- `per-ace`: one vector per ACE; use this for MaxSim
- `mean-ace`: embed each ACE, then average the vectors for each device
- `whole`: embed the full compact profile as one text

The shipped OpenAI embeddings are data artifacts. This repository does not
include a script for regenerating them.

### 3. Whiten an Embedding Bank

```bash
python src/gen_whiten_emb.py \
  --reference data/ref_embeddings/bge/per_ace/raw/reference_per_ace.npz \
  --input data/ref_embeddings/bge/per_ace/raw/reference_per_ace.npz \
  --output data/ref_embeddings/bge/per_ace/whitened_k256/reference_per_ace_whitened_k256.npz \
  --metadata tmp/whitening_metadata.json \
  --k 256
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--reference` | Required | Bank used to fit whitening |
| `--input` | Required | Bank to transform |
| `--output` | Required | Output `.npz` path |
| `--k` | `256` | Maximum number of output dimensions |
| `--metadata` | None | Optional JSON metadata path |

When whitening query embeddings, keep `--reference` set to the canonical
reference bank. Change only `--input` and `--output`. This prevents query data
from affecting the whitening transform.

## Repository Data

| Path | Contents |
| --- | --- |
| `data/ref_mud/raw/` | 28 canonical MUD JSON profiles |
| `data/ref_mud/compact/` | Compact ACE text and `reduction_stats.json` |
| `data/runtime_aces/` | 26 real-traffic runtime ACE CSV files |
| `data/ref_embeddings/bge/` | Shipped BGE-M3 embeddings |
| `data/ref_embeddings/openai/` | Shipped OpenAI embeddings |

The compact reference data contains 1,023 ACE instances and 710 unique compact
ACE lines. See [`data/README.md`](data/README.md) for more detail.

## Citation

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
