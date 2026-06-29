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

Reference embeddings are stored in `data/ref_embeddings/`, split first by encoder and
then by representation.

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

This folder provides 26 real loT traffic traces in the form of one CSV file per device type, each containing runtime ACE rows (a total of 810490 rows).

One can use our code stored in sro to convert individual ACE rows to embeddings and preform matching against reference profiles.
