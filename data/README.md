# Data Notes

## Canonical MUD Profiles

The canonical dataset contains 28 public MUD profiles used as reference device
profiles. The files are stored in `data/ref_mud/raw/`.

## Compact ACE Text

The compact representation strips JSON structure and keeps the behavioral
fields used for semantic comparison:

- ingress or egress direction
- local/controller hints
- IP version
- protocol
- endpoint domain or network
- source or destination port

Example:

```text
egress ipv4 tcp (direction-initiated:from-device) dst:api.amazonalexa.com dst-port:443
```

The compact canonical files are stored in `data/ref_mud/compact/`.

## Reference Embeddings

Reference embeddings are stored in `data/ref_embeddings/`. They are grouped by
encoder and then by representation.

- BGE-M3 artifacts use the [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
  model.
- `data/ref_embeddings/bge/whole/raw/`: one JSON file per device for full raw MUD JSON
  embeddings.
- `data/ref_embeddings/bge/whole/compact/`: one JSON file per device for whole compact
  text embeddings.
- `data/ref_embeddings/openai/whole/raw/`: one JSON file per device for OpenAI
  full raw MUD JSON embeddings.
- `data/ref_embeddings/openai/whole/compact/`: one JSON file per device for OpenAI
  whole compact text embeddings.
- `data/ref_embeddings/*/per_ace/raw/`: one `.npz` matrix per encoder plus a CSV row
  map.
- `data/ref_embeddings/*/per_ace/whitened_k256/`: whitened per-ACE `.npz` matrices plus
  CSV row maps.

## Runtime Traffic Traces

`data/runtime_aces/` contains 26 real IoT traffic traces. There is one CSV file
per device type and 810,490 rows in total.

The `runtime_ace` column contains one compact behavior line for each flow. The
current command-line scripts do not read the CSV files directly. Prepare the
runtime ACEs as compact `.txt` query profiles for exact matching or as a
labelled per-ACE `.npz` bank for MaxSim matching. See the main README for the
required formats and commands.
