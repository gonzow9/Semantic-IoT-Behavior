# Semantic Matching of Behavioral Primitives for MUD-Based IoT Device Identification

This repository contains code, data, and derived artifacts for identifying IoT
devices from their communication behavior. It compares exact and semantic
matching against reference Manufacturer Usage Description (MUD) profiles.

## Problem and Approach

A network may need to identify whether a connected device is a camera, smart
plug, voice assistant, or another IoT device. This identity can support access
control, policy enforcement, and network monitoring.

A MUD profile describes the communication allowed for one device type. It is a
set of Access Control Entries (ACEs). Each ACE is one behavioral primitive that
contains a protocol, endpoint, direction, and port. This repository converts
each ACE into compact ACE text such as:

```text
egress ipv4 tcp (direction-initiated:from-device) dst:tech.carematix.com dst-port:8777
```

The method keeps ACEs separate and compares observed behavior with the ACEs in
each reference MUD profile:

1. Convert MUD JSON into compact ACE text.
2. Encode each ACE as a BGE-M3 embedding with 1,024 dimensions.
3. Whiten the embeddings using a transform fitted only on the reference ACEs.
   The supplied evaluation artifacts retain 256 principal components.
4. Rank the candidate device profiles with exact or semantic matching.

The matching methods are:

- **Jaccard:** exact set overlap between the query ACEs and a reference profile.
- **Exact ACE-hit count:** the number of query ACEs that appear identically in
  a reference profile. The real traffic evaluation normalizes this count by
  the number of query observations when reporting the exact-overlap score.
- **Mean Pool:** cosine similarity between one average embedding for the query
  and one average embedding for each reference profile.
- **MaxSim:** match each query ACE to its most similar ACE in a reference
  profile, then average those best-match similarities.

Exact matching is strong when observed behavior has literal ACE overlap with a
reference profile. Semantic matching provides a complementary signal when
literal overlap is limited.

## Evaluations

The repository contains two forms of evaluation.

**Controlled evaluation** changes the amount and composition of exact ACE
overlap:

- **Unseen ACEs:** selected query ACEs are removed from the candidate MUD
  profiles before scoring.
- **Endpoint perturbation:** domain names are changed while protocol, direction,
  and port are preserved. Generated ACEs are checked to ensure that they do not
  exactly match any reference ACE.
- **Mixed partial observation:** a query may contain exact ACEs,
  endpoint-perturbed ACEs, and an ACE made unseen in the source profile.

**Real traffic evaluation** converts real IoT flows into ACE-like behavioral
primitives. It measures identification as flows accumulate and within separate
50-flow windows.

## Included Artifacts

- 28 public reference MUD profiles.
- 1,023 ACE instances, including 710 unique compact ACE texts.
- BGE-M3 and OpenAI reference embedding artifacts.
- 26 real IoT traffic traces containing 810,490 flows.
- Python code for representation, matching, controlled evaluation, and real
  traffic evaluation.

The reference MUD profiles come from the public
[UNSW IoT Analytics MUD dataset](https://iotanalytics.unsw.edu.au/mudprofiles.html).
See [data/README.md](data/README.md) for the file layout and data formats.

## Quick Start

Run all commands from the repository root:

```bash
git clone https://github.com/gonzow9/Semantic-IoT-Behavior.git
cd Semantic-IoT-Behavior
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The following command runs the single unseen ACE condition with the supplied
BGE-M3 embeddings. It does not require an API key or model download.

```bash
python src/controlled_eval.py \
  --condition single-unseen \
  --examples 0 \
  --bootstrap-resamples 0
```

The command creates 1,023 controlled queries. Each query contains one ACE, and
that ACE is removed from every candidate profile before scoring. The command
then prints a terminal summary with one row per matching method:

```text
Scored 1023 single-unseen queries from 28 devices.
method             top1    topK     mrr   abstain
jaccard            0.0000  0.0000  0.0000  1.0000
exact_hit_count    0.0000  0.0000  0.0000  1.0000
mean_pool          0.6393  0.7595  0.7040  0.0000
maxsim             0.6551  0.7950  0.7221  0.0000
```

The summary columns mean:

- `top1`: average Top-1 credit. A unique correct leader receives full credit.
  Credit is split equally when several devices tie for the highest score.
- `topK`: average credit for placing the correct device within the retained
  top-k candidates. The default is top five. Credit is split when a tie crosses
  the top-k boundary.
- `mrr`: mean reciprocal rank. Reciprocal rank is averaged over tied positions.
- `abstain`: fraction where every candidate receives a zero score.

An all-zero score vector is treated as an abstention and receives zero credit.
Score ties use a fixed absolute tolerance of `1e-8`.

## Run an Evaluation

| Task | Command |
| --- | --- |
| Single unseen ACE | `python src/controlled_eval.py --condition single-unseen` |
| Unseen ACE family | `python src/controlled_eval.py --condition unseen-family` |
| Unseen ACE set | `python src/controlled_eval.py --condition unseen-set` |
| Endpoint perturbation, all devices | `python src/endpoint_perturbation_eval.py endpoint-perturbation --subset full` |
| Endpoint perturbation, domain-rich devices | `python src/endpoint_perturbation_eval.py endpoint-perturbation --subset high-domain` |
| Mixed partial observation | `python src/endpoint_perturbation_eval.py mixed-partial-observation` |
| Build embeddings for the real traffic flows | `python src/realtraffic_eval.py embed` |
| Evaluate traffic as flows accumulate | `python src/realtraffic_eval.py cumulative` |
| Evaluate separate 50-flow windows | `python src/realtraffic_eval.py windows` |

Endpoint perturbation and real traffic embedding generation download BGE-M3 on
the first run. A GPU is helpful but not required.

See [src/README.md](src/README.md) for complete commands, inputs, outputs, and
artifact rebuilding instructions. Every command also supports `--help`.

## Repository Layout

| Path | Contents |
| --- | --- |
| `src/` | Representation, matching, and evaluation code |
| `data/ref_mud/raw/` | 28 reference MUD JSON profiles |
| `data/ref_mud/compact/` | Compact ACE text for the reference profiles |
| `data/ref_embeddings/` | Supplied reference embedding artifacts |
| `data/runtime_aces/` | Real traffic flows converted to compact ACE text |
| `data/runtime_embeddings/` | Locally generated runtime embeddings, not tracked by Git |

## Citation

```bibtex
@misc{witt2026semanticidentifyiot,
  title={Semantic Matching of Behavioral Primitives for MUD-Based IoT Device Identification},
  author={Samuel Witt and Hassan Habibi Gharakheili},
  year={2026},
  eprint={2606.12793},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2606.12793},
}
```

## License

See [LICENSE.md](LICENSE.md).
