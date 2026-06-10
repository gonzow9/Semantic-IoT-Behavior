# Contents

This repo is a curated artifact repository, not a full copy of the original
experiment workspace.

## Data

| Path | Contents |
| --- | --- |
| `data/mud_raw/` | 28 canonical MUD JSON profiles. |
| `data/mud_compact/` | 28 compact ACE text files plus `reduction_stats.json`. |

The compact canonical data contains 1023 ACE instances and 710 unique ACE
lines. The compact representation reduces the canonical JSON byte count by
about 88%.

## Embeddings

| File | Arrays | Notes |
| --- | --- | --- |
| `embeddings/bge_m3/reference_per_ace.npz` | `embeddings (1023, 1024)`, `devices (1023,)`, `ace_texts (1023,)` | Main raw BGE-M3 ACE bank. |
| `embeddings/bge_m3/reference_per_ace_whitened_k256.npz` | `embeddings (1023, 256)`, `names`, `devices`, `ace_texts` | Reference-only whitened ACE bank used for semantic MaxSim scoring. |
| `embeddings/bge_m3/reference_whole_profile_json.npz` | `original_embeddings (28, 1024)`, `original_names (28,)` | Whole-profile raw JSON embedding baseline. |
| `embeddings/bge_m3/reference_whole_profile_compact.npz` | `embeddings (28, 1024)`, `names (28,)` | Whole-profile compact text embedding baseline. |
| `embeddings/bge_m3/reference_mean_pool_per_ace.npz` | `embeddings (28, 1024)`, `names (28,)` | Device vectors produced by mean-pooling per-ACE embeddings. |
| `embeddings/openai/reference_per_ace.npz` | `embeddings (1023, 1024)`, `devices (1023,)`, `ace_texts (1023,)` | OpenAI `text-embedding-3-large` per-ACE bank requested at 1024 dimensions. |
| `embeddings/openai/reference_per_ace_whitened_k256.npz` | `embeddings (1023, 256)`, `names`, `devices`, `ace_texts` | Reference-only whitened OpenAI per-ACE bank. |
| `embeddings/openai/reference_whole_profile_compact.npz` | `embeddings (28, 1024)`, `names (28,)` | OpenAI compact whole-profile embedding baseline. |
| `embeddings/openai/reference_mean_pool_per_ace.npz` | `embeddings (28, 1024)`, `names (28,)` | OpenAI mean-pooled per-ACE device vectors. |

## Results

| Path | Contents |
| --- | --- |
| `results/geometry/` | Intrinsic geometry diagnostics for whole-profile, mean-pooled, and per-ACE embeddings. |
| `results/controlled_runtime/` | Summary JSON files for domain perturbation, IP perturbation, and partial-runtime drift evaluations. |
| `results/real_traffic/` | Small summary CSV/JSON outputs for the real-flow evaluation. |
| `figures/paper/` | Selected PDF figures used in the paper. |

## Code

| Path | Contents |
| --- | --- |
| `src/semantic_iot_behavior/compact_mud.py` | Converts MUD JSON profiles into compact ACE text. |
| `src/semantic_iot_behavior/embed.py` | Regenerates local BGE-M3 embedding banks. |
| `src/semantic_iot_behavior/whiten.py` | Applies reference-only truncated PCA whitening. |
| `src/semantic_iot_behavior/score.py` | Provides exact-overlap and MaxSim retrieval helpers. |
| `src/semantic_iot_behavior/runtime_behavior_demo.py` | Small Section 3 evolving-runtime behavior demo. |
