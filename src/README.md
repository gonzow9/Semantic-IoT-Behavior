# Source Code Guide

This directory contains the code for preparing ACE representations, matching
queries against reference MUD profiles, and running the controlled and real
traffic evaluations.

Run the commands in this guide from the repo root.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The supplied reference embeddings are enough to run the unseen ACE conditions.
Endpoint perturbation and real traffic embedding generation use BGE-M3 and
download the model on the first run. A GPU is very helpful but not required.

## Code Map

| Path | Purpose |
| --- | --- |
| `convert_mud_compact.py` | Convert MUD JSON into compact ACE text |
| `gen_emb.py` | Create whole-profile, mean-pooled, or per-ACE embeddings |
| `gen_whiten_emb.py` | Fit reference-only whitening and transform an embedding bank |
| `controlled_eval.py` | Run the unseen ACE controlled conditions |
| `endpoint_perturbation_eval.py` | Run endpoint perturbation and mixed partial observation |
| `realtraffic_eval.py` | Build real traffic embeddings and run both traffic evaluations |
| `runtime_score.py` | Score user-prepared text or embedding queries |
| `matching/` | Shared reference-bank, query, scoring, and statistics code |
| `endpoint_perturbation/` | Endpoint perturbation and mixed-query construction |
| `realtraffic/` | Real traffic loading and scoring code |

## Output Metrics

The controlled evaluation commands print a terminal summary with one row for
each matching method. The columns are:

| Column | Meaning |
| --- | --- |
| `top1` | Average Top-1 credit, split equally among devices tied for the highest score |
| `topK` | Average top-k credit, split when a tie crosses the top-k boundary |
| `mrr` | Mean reciprocal rank, averaged over the positions occupied by a tie |
| `abstain` | Fraction where the method has no positive identification evidence |

The default value of `k` is five. Use `--top-k` to change it.

The controlled evaluation and prepared-query commands use a fixed absolute
tie tolerance of `1e-8`. If several devices share the highest positive score,
Top-1 credit is divided equally among them. If all candidate scores are zero,
the method abstains and receives zero Top-1, top-k, and MRR credit. This occurs
for exact matching when no query ACE has an identical counterpart in any
candidate profile.

By default, controlled evaluations also report paired bootstrap confidence
intervals for the Top-1 difference between MaxSim and each baseline. Set
`--bootstrap-resamples 0` for a faster run without these intervals. Use
`--output PATH` to save the full result as JSON.

## Controlled Evaluation

Controlled queries isolate how identification changes as exact ACE overlap is
reduced. Each query has a known source device and is scored against all 28
candidate MUD profiles.

### Unseen ACEs

An unseen ACE query contains behavior that has no exact counterpart in the
candidate repository. Before scoring, each selected ACE is removed from every
candidate profile in which it occurs.

Run the three conditions:

```bash
python src/controlled_eval.py --condition single-unseen
python src/controlled_eval.py --condition unseen-family
python src/controlled_eval.py --condition unseen-set
```

The conditions differ as follows:

- `single-unseen` creates one query for each of the 1,023 device and ACE pairs.
- `unseen-family` groups related ACEs using reciprocal top-five neighbors with
  cosine similarity of at least 0.75. It creates 103 queries containing two to
  four related ACEs.
- `unseen-set` samples three ACEs from distinct families. It creates ten seeded
  queries for each of 24 eligible devices, giving 240 queries.

All three conditions have zero literal ACE overlap by construction.

### Endpoint Perturbation

Endpoint perturbation changes domain names while keeping the protocol,
direction, and port unchanged. The changes include numeric modification,
regional token insertion, and domain-suffix substitution. Each generated ACE
is checked against all reference ACE texts. A generated ACE is rejected if it
creates an exact match.

Run the condition over all 28 devices:

```bash
python src/endpoint_perturbation_eval.py \
  endpoint-perturbation \
  --subset full
```

This creates ten endpoint-perturbation queries per device, giving 280 queries.

Run the condition over the 14 devices with at least ten perturbable domain-name
ACEs:

```bash
python src/endpoint_perturbation_eval.py \
  endpoint-perturbation \
  --subset high-domain
```

This creates 140 queries. Both endpoint perturbation conditions have zero
literal ACE overlap by construction.

### Mixed Partial Observation

Mixed partial observation creates queries with different combinations of:

- ACEs that remain exact matches.
- ACEs with endpoint perturbation.
- One ACE that may be made unseen in the correct reference profile.

Run the full grid:

```bash
python src/endpoint_perturbation_eval.py mixed-partial-observation
```

The grid varies the retained profile fraction over `0.10`, `0.25`, and `0.50`;
the endpoint-perturbation fraction over `0.00`, `0.25`, `0.50`, and `1.00`;
and the unseen ACE count over `0` and `1`. Ten seeded queries are attempted per
device and grid cell. We exclude 240 cases where making one ACE unseen would
leave no known ACE in the query. We also exclude 708 cases where endpoint
perturbation is requested but the sampled query has no perturbable hostname
ACE. This leaves 5,772 controlled queries.

The terminal output first gives the overall summary. It then groups Top-1
accuracy by the number of exact ACE hits against the source profile: `0`,
`1-2`, `3-5`, and `>5`.

### Repeatability

Some devices share infrastructure ACEs such as DNS, NTP, and DHCP. Two devices
can therefore receive the same score. Tie-aware scoring prevents device-name
order from deciding the result. Query construction is repeatable because all
random choices use fixed seeds.

## Real Traffic Evaluation

The 26 CSV files in `data/runtime_aces/` contain 810,490 real IoT flows. Each
row contains compact ACE text in the `runtime_ace` column. Rows remain in their
original arrival order.

Repeated flow observations are retained for exact ACE-hit count, Mean Pool,
and MaxSim. Jaccard uses the set of unique observed features. The exact-overlap
score is the exact ACE-hit count divided by the number of observed flows. A
trace or window is counted as identified only when the correct device is the
only top-scoring candidate. Tied leaders count as not identified.

The traffic evaluations need an embedding bank for the distinct runtime ACE
texts. Build it once:

```bash
python src/realtraffic_eval.py embed
```

This command embeds each distinct runtime ACE text, applies the whitening
transform fitted on the reference ACE corpus, and saves the result under
`data/runtime_embeddings/`. This generated directory is not tracked by Git.

### Identification as Evidence Accumulates

```bash
python src/realtraffic_eval.py cumulative \
  --output tmp/realtraffic_cumulative.json
```

For each trace, the first `k` flows form the query at observation point `k`.
The output reports Top-1 identification as flows accumulate. It also reports
the distribution of the correct-device rank over the first 10,000 flows of
each trace.

### Identification from Short Traffic Windows

```bash
python src/realtraffic_eval.py windows \
  --output tmp/realtraffic_windows.json
```

This evaluation randomly samples up to 500 non-overlapping 50-flow windows per
device using a fixed seed. It creates 9,023 windows across 25 devices. The
output reports identification over all windows and over groups with
exact-overlap scores below `0.50`, below `0.10`, and equal to zero.

Use `--window-size` or `--windows-per-device` to evaluate another window
configuration.

## Score Prepared Queries

`runtime_score.py` scores queries that you prepare separately. It does not
construct any controlled condition.

### Exact Matching

Each non-empty line in a query `.txt` file must contain one compact ACE. The
expected device label is read from the filename. For several queries from the
same device, place the files in a directory named after that device.

```text
queries/
└── amazonEchoMud/
    ├── query_001.txt
    └── query_002.txt
```

Run exact Jaccard matching:

```bash
python src/runtime_score.py exact \
  --reference-dir data/ref_mud/compact \
  --query-dir queries \
  --method jaccard \
  --output tmp/exact_results.json
```

Use `--method exact_hit_count` for the number of exact ACE matches.

### MaxSim Matching

MaxSim uses labeled per-ACE `.npz` banks:

```bash
python src/runtime_score.py maxsim \
  --reference-npz data/ref_embeddings/bge/per_ace/whitened_k256/reference_per_ace_whitened_k256.npz \
  --query-npz path/to/query_per_ace_whitened_k256.npz \
  --output tmp/maxsim_results.json
```

Each bank must contain:

- `embeddings`: one row per ACE.
- `devices` or `names`: one device label per row.
- `ace_texts`: recommended, and required by `controlled_eval.py`.

## Rebuild the Reference Artifacts

The supplied artifacts are ready to use. The following commands rebuild them
from the reference MUD JSON files.

### Convert MUD JSON to Compact ACE Text

```bash
python src/convert_mud_compact.py \
  --input-dir data/ref_mud/raw \
  --output-dir data/ref_mud/compact
```

### Create Per-ACE BGE-M3 Embeddings

```bash
python src/gen_emb.py \
  --input-dir data/ref_mud/compact \
  --pool per-ace \
  --model-name BAAI/bge-m3 \
  --output data/ref_embeddings/bge/per_ace/raw/reference_per_ace.npz
```

The available pooling choices are:

- `per-ace`: keep one vector for each ACE. Use this layout for MaxSim.
- `mean-ace`: embed each ACE, then average the vectors for each device.
- `whole`: encode the full compact profile as one text.

The supplied OpenAI embeddings are data artifacts. This repository does not
include a script for regenerating them.

### Whiten an Embedding Bank

```bash
python src/gen_whiten_emb.py \
  --reference data/ref_embeddings/bge/per_ace/raw/reference_per_ace.npz \
  --input data/ref_embeddings/bge/per_ace/raw/reference_per_ace.npz \
  --output data/ref_embeddings/bge/per_ace/whitened_k256/reference_per_ace_whitened_k256.npz \
  --metadata tmp/whitening_metadata.json \
  --k 256
```

Whitening parameters are fitted only on the reference ACE corpus. When
transforming another embedding bank, keep `--reference` set to the raw
reference bank and change only `--input` and `--output`. This prevents query
data from affecting embedding calibration.

## Command Help

Every command supports `--help`. Commands with subcommands also provide help
for each subcommand. For example:

```bash
python src/endpoint_perturbation_eval.py --help
python src/endpoint_perturbation_eval.py endpoint-perturbation --help
python src/realtraffic_eval.py windows --help
```
