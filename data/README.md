# Data and Artifacts

This directory contains the reference MUD profiles, compact ACE text, supplied
reference embeddings, and real traffic traces used by the code in `src/`.

## Reference MUD Profiles

`data/ref_mud/raw/` contains 28 public MUD profiles used as reference profiles.
The profiles contain 1,023 ACE instances, including 710 unique ACEs.

Each JSON file describes the allowed communication behavior of one IoT device
type. An Access Control Entry (ACE) is one rule containing a protocol, endpoint,
direction, and port.

## Compact ACE Text

`data/ref_mud/compact/` contains one compact text file per reference profile.
Each non-empty line is one ACE. The compact representation removes repeated
JSON structure and retains the fields used for matching:

- Ingress or egress direction.
- Local and controller information.
- IP version.
- Transport protocol.
- Endpoint domain name or network address.
- Source or destination port.

Example:

```text
egress ipv4 tcp (direction-initiated:from-device) dst:api.amazonalexa.com dst-port:443
```

`reduction_stats.json` records the size and token-count reduction produced by
the conversion.

## Reference Embeddings

`data/ref_embeddings/` contains embeddings grouped by encoder and
representation.

- `bge/` contains artifacts generated with
  [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3).
- `openai/` contains artifacts generated with OpenAI
  `text-embedding-3-large`.
- `*/whole/raw/` contains one embedding for each full MUD JSON document.
- `*/whole/compact/` contains one embedding for each compact whole profile.
- `*/per_ace/raw/` contains one raw embedding for each ACE instance.
- `*/per_ace/whitened_k256/` contains per-ACE embeddings after reference-only
  whitening with 256 retained principal components.

The per-ACE `.npz` banks include an embedding matrix, device labels, and ACE
text. The accompanying metadata CSV files map matrix rows to source devices and
ACEs.

## Real Traffic Traces

`data/runtime_aces/` contains 26 CSV files with 810,490 real IoT flows. There is
one file per device trace. Rows remain in their original arrival order.

The `runtime_ace` column contains one ACE-like behavioral primitive for each
flow. These primitives use the same protocol, endpoint, direction, and port
fields as the compact reference ACEs.

The trace preparation retains IP flows with transport protocol and port
information. It excludes non-IP traffic and flows without transport-layer
ports. Each flow is oriented as ingress or egress relative to the device.
Remote IP addresses are replaced with hostnames when the trace contains a DNS
mapping. Otherwise, the remote address is kept as an IPv4 `/32` or IPv6 `/128`
endpoint.

`src/realtraffic_eval.py` reads these CSV files directly. Build the local
runtime embedding bank with:

```bash
python src/realtraffic_eval.py embed
```

Then run identification as flows accumulate or within separate traffic
windows:

```bash
python src/realtraffic_eval.py cumulative
python src/realtraffic_eval.py windows
```

See [`src/README.md`](../src/README.md) for the evaluation procedure and output
definitions.

## Generated Runtime Embeddings

`data/runtime_embeddings/` is created locally by
`python src/realtraffic_eval.py embed`. It stores the BGE-M3 embeddings for the
distinct runtime ACE texts after applying the whitening transform fitted on the
reference ACE corpus.

This directory is not tracked by Git because it can be rebuilt from the CSV
files.
