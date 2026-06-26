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

## Real Traffic Evaluation

The paper's real-traffic evaluation used external IoT traffic traces and
converted observed flows into ACE-like behavioral primitives.

The converted runtime ACE rows used by the final full-trace experiment are
stored in `data/runtime_aces/real_traffic/`, with one CSV per device trace. The export contains 26 traces and 810490 retained flow rows.