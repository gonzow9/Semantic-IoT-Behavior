# Data Notes

## Canonical MUD Profiles

The canonical dataset contains 28 public MUD profiles used as reference device
profiles. The files are stored in `data/mud/mud_raw/`.

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

The compact canonical files are stored in `data/mud/mud_compact/`.

## Real Traffic Evaluation

The paper's real-traffic evaluation used external IoT traffic traces and
converted observed flows into ACE-like behavioral primitives. The raw traces
and large intermediate flow artifacts are not included in this repository.

The included real-traffic files in `analysis/real_traffic/` are summary
outputs and small CSVs needed for the final behavior.
