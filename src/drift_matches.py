"""Drifted-endpoint and mixed partial observation experiments.

Two experiment types from the paper:

- drift: hostnames in selected ACEs are changed while protocol and port stay
  the same. The query contains only the drifted ACEs. The "full" subset uses
  every device; "high-domain" keeps devices with at least ten domain ACEs.
- mixed: a runtime query mixes exact ACEs, drifted ACEs, and optionally one
  unseen ACE. A grid over the retained fraction, the drift fraction, and the
  unseen count is evaluated, and Top-1 is also grouped by the number of exact
  ACE matches against the correct reference profile.

Drifted ACE texts are new strings, so they must be embedded. The first run
downloads the BGE-M3 model. All embeddings (references and queries) are
whitened with one transform fitted on the raw reference bank.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gen_emb import build_model, encode_texts
from gen_whiten_emb import apply_whitening, fit_whitening
from runtime_matches import (
    METHODS,
    AceBank,
    build_ace_bank,
    paired_top1_bootstrap,
    score_query,
    stable_seed_offset,
    summarise_results,
)
from runtime_score import clean_device_name

DOMAIN_TOKEN_RE = re.compile(
    r"\b(?P<field>src|dst):(?P<host>[a-z0-9.][a-z0-9.\-]*[a-z0-9])(?=\s|$)",
    re.IGNORECASE,
)
COMPOUND_SUFFIXES = ("com.au", "co.uk")
RESERVED_EXACT = {"localhost", "example.com", "example.org", "example.net"}
RESERVED_SUFFIXES = (".invalid", ".test")
REGION_SWAPS = {
    "us": "eu",
    "eu": "us",
    "au": "us",
    "uk": "eu",
    "ca": "us",
    "jp": "sg",
    "sg": "jp",
}
INJECTED_REGION_SWAPS = {"ap": "eu", "na": "eu", "emea": "us"}
REGION_TOKENS = ("us", "eu", "ap", "na", "emea")
NUMERIC_INSERT_VALUES = ("2", "3", "4")
TLD_SWAPS = {
    "com": ("com.au", "co.uk", "de", "eu", "io", "net"),
    "net": ("com", "org", "io"),
    "org": ("com", "net"),
    "com.au": ("com", "co.uk"),
}
MAX_MUTATION_ATTEMPTS = 8
MAX_SELECTED_RULES_PER_PROFILE = 3


# ---------------------------------------------------------------------------
# Hostname eligibility and mutations.
# ---------------------------------------------------------------------------


def _is_ipv4_literal(value: str) -> bool:
    try:
        ipaddress.IPv4Address(value)
        return True
    except ValueError:
        return False


def is_perturbable_domain(host: str) -> bool:
    value = host.lower().rstrip(".")
    if "." not in value or _is_ipv4_literal(value):
        return False
    labels = value.split(".")
    if any(not label for label in labels):
        return False
    tld = labels[-1]
    if len(tld) < 2 or not tld.isalpha():
        return False
    if value in RESERVED_EXACT:
        return False
    if any(value.endswith(suffix) for suffix in RESERVED_SUFFIXES):
        return False
    if any(value.endswith("." + reserved) for reserved in RESERVED_EXACT):
        return False
    return True


def has_perturbable_domain(rule: str) -> bool:
    lowered = rule.lower()
    if "controller:" in lowered or "eth:" in lowered or "mac:" in lowered:
        return False
    if lowered.startswith("egress local ipv4") or lowered.startswith("ingress local ipv4"):
        return False
    return any(
        is_perturbable_domain(match.group("host"))
        for match in DOMAIN_TOKEN_RE.finditer(rule)
    )


def _registered_suffix_len(host: str) -> int:
    labels = host.split(".")
    for suffix in COMPOUND_SUFFIXES:
        suffix_labels = suffix.split(".")
        if host.endswith("." + suffix) and len(labels) > len(suffix_labels):
            return len(suffix_labels) + 1
    return 2


def _numeric_increment(label: str) -> str | None:
    match = re.search(r"(\d+)(?!.*\d)", label)
    if match is None:
        return None
    digits = match.group(1)
    incremented = str(int(digits) + 1).zfill(len(digits))
    return f"{label[:match.start(1)]}{incremented}{label[match.end(1):]}"


def _numeric_insert(label: str, rng: random.Random) -> str | None:
    if re.search(r"\d", label) or _region_token_present(label):
        return None
    inserted = f"{label}{rng.choice(NUMERIC_INSERT_VALUES)}"
    return inserted if len(inserted) <= 63 else None


def _region_suffix_swap(label: str) -> str | None:
    match = re.match(r"^(?P<base>.+)-(?P<region>us|eu|au|uk|ca|jp|sg|ap|na|emea)$", label)
    if match is None:
        return None
    region = match.group("region")
    swapped = REGION_SWAPS.get(region, INJECTED_REGION_SWAPS.get(region))
    return f"{match.group('base')}-{swapped}"


def _region_token_present(label: str) -> bool:
    if label in REGION_TOKENS:
        return True
    return any(
        label.startswith(f"{token}-")
        or label.endswith(f"-{token}")
        or f"-{token}-" in label
        for token in REGION_TOKENS
    )


def _region_token_inject(
    labels: list[str], *, position: int, rng: random.Random
) -> list[str] | None:
    region = rng.choice(REGION_TOKENS)
    updated = list(labels)
    if rng.choice((True, False)):
        candidate = f"{updated[position]}-{region}"
        if len(candidate) > 63:
            return None
        updated[position] = candidate
    else:
        updated.insert(position, region)
    return updated


def _label_rename(label: str) -> str | None:
    if label.endswith("-alt"):
        return None
    renamed = f"{label}-alt"
    if len(renamed) > 63:
        renamed = f"{label[:59]}-alt"
    return renamed


def _effective_tld(labels: list[str]) -> tuple[str, int]:
    host = ".".join(labels)
    for suffix in COMPOUND_SUFFIXES:
        suffix_labels = suffix.split(".")
        if host.endswith("." + suffix) and len(labels) > len(suffix_labels):
            return suffix, len(suffix_labels)
    return labels[-1], 1


def _tld_swap(labels: list[str], rng: random.Random) -> list[str] | None:
    current, suffix_label_count = _effective_tld(labels)
    options = TLD_SWAPS.get(current)
    if not options:
        return None
    updated = labels[:-suffix_label_count] + rng.choice(options).split(".")
    return updated if updated != labels else None


def perturb_host(host: str, rng: random.Random) -> str:
    """Rewrite one hostname with a small, realistic mutation."""
    if not is_perturbable_domain(host):
        return host

    labels = host.lower().rstrip(".").split(".")
    suffix_len = _registered_suffix_len(".".join(labels))
    mutable = list(range(max(0, len(labels) - suffix_len)))
    probe = random.Random(0)

    applicable: list[str] = []
    if any(_numeric_increment(labels[pos]) is not None for pos in mutable):
        applicable.append("numeric_increment")
    if any(_numeric_insert(labels[pos], probe) is not None for pos in mutable):
        applicable.append("numeric_insert")
    if any(_region_suffix_swap(labels[pos]) is not None for pos in mutable):
        applicable.append("region_suffix_swap")
    if mutable and not any(_region_token_present(labels[pos]) for pos in mutable):
        applicable.append("region_token_inject")
    if _tld_swap(labels, random.Random(0)) is not None:
        applicable.append("tld_swap")

    if applicable:
        mutation = rng.choice(applicable)
        if mutation == "numeric_increment":
            positions = [pos for pos in mutable if _numeric_increment(labels[pos]) is not None]
            position = rng.choice(positions)
            labels[position] = _numeric_increment(labels[position])
            return ".".join(labels)
        if mutation == "numeric_insert":
            positions = [pos for pos in mutable if _numeric_insert(labels[pos], probe) is not None]
            position = rng.choice(positions)
            labels[position] = _numeric_insert(labels[position], rng)
            return ".".join(labels)
        if mutation == "region_suffix_swap":
            positions = [pos for pos in mutable if _region_suffix_swap(labels[pos]) is not None]
            position = rng.choice(positions)
            labels[position] = _region_suffix_swap(labels[position])
            return ".".join(labels)
        if mutation == "region_token_inject":
            position = rng.choice(mutable)
            updated = _region_token_inject(labels, position=position, rng=rng)
            if updated is not None:
                return ".".join(updated)
        if mutation == "tld_swap":
            updated = _tld_swap(labels, rng)
            if updated is not None:
                return ".".join(updated)

    fallback = [
        (position, renamed)
        for position in mutable
        if (renamed := _label_rename(labels[position])) is not None
    ]
    if not fallback:
        return host
    position, renamed = rng.choice(fallback)
    labels[position] = renamed
    return ".".join(labels)


def perturb_ace(rule: str, rng: random.Random) -> str:
    """Rewrite every eligible hostname in one compact ACE line."""

    def replace(match: re.Match[str]) -> str:
        host = match.group("host")
        perturbed = perturb_host(host, rng)
        if perturbed == host.lower().rstrip("."):
            return match.group(0)
        return f"{match.group('field')}:{perturbed}"

    return DOMAIN_TOKEN_RE.sub(replace, rule)


def perturb_rule_with_retries(
    rule: str, rng: random.Random, reference_texts: frozenset[str]
) -> str | None:
    """Perturb one ACE, rejecting results that collide with any reference ACE."""
    for _attempt in range(MAX_MUTATION_ATTEMPTS):
        candidate = perturb_ace(rule, rng)
        if candidate == rule:
            continue
        if candidate in reference_texts:
            continue
        return candidate
    return None


def hash_device(device: str) -> int:
    """Deterministic compact hash, safe across Python processes."""
    value = 0
    for char in device:
        value = ((value * 131) + ord(char)) % 1_000_000_007
    return value


# ---------------------------------------------------------------------------
# Query generation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Query:
    query_id: str
    expected_device: str
    cluster_key: str
    query_texts: tuple[str, ...]
    removed_texts: frozenset[str]
    removed_scope: str
    exact_hits: int


def read_compact_profiles(compact_dir: Path) -> OrderedDict[str, tuple[str, ...]]:
    profiles: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    for path in sorted(compact_dir.glob("*.txt")):
        rules = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        profiles[clean_device_name(path.stem)] = rules
    if not profiles:
        raise ValueError(f"No compact .txt profiles found in {compact_dir}")
    return profiles


def build_drift_queries(
    profiles: OrderedDict[str, tuple[str, ...]],
    *,
    subset: str,
    variants: int,
    fraction: float,
    seed: int,
    high_domain_threshold: int,
) -> list[Query]:
    reference_texts = frozenset(text for rules in profiles.values() for text in rules)
    queries: list[Query] = []
    for device, rules in profiles.items():
        candidate_count = sum(1 for rule in rules if has_perturbable_domain(rule))
        if subset == "high-domain" and candidate_count < high_domain_threshold:
            continue
        if candidate_count == 0:
            continue
        for variant in range(variants):
            rng = random.Random(
                seed + variant * 997 + candidate_count * 131 + hash_device(device)
            )
            eligible = [idx for idx, rule in enumerate(rules) if has_perturbable_domain(rule)]
            target = min(
                MAX_SELECTED_RULES_PER_PROFILE, max(1, round(len(eligible) * fraction))
            )
            selected = sorted(rng.sample(eligible, k=min(target, len(eligible))))
            drifted = [
                perturbed
                for idx in selected
                if (perturbed := perturb_rule_with_retries(rules[idx], rng, reference_texts))
                is not None
            ]
            if not drifted:
                continue
            queries.append(
                Query(
                    query_id=f"{device}/drift/v{variant:02d}",
                    expected_device=device,
                    cluster_key=device,
                    query_texts=tuple(drifted),
                    removed_texts=frozenset(drifted),
                    removed_scope="all",
                    exact_hits=0,
                )
            )
    return queries


def build_mixed_queries(
    profiles: OrderedDict[str, tuple[str, ...]],
    *,
    retained_fraction: float,
    domain_fraction: float,
    novel_count: int,
    seeds_per_device: int,
    seed: int,
) -> list[Query]:
    reference_texts = frozenset(text for rules in profiles.values() for text in rules)
    queries: list[Query] = []
    for device, rules in profiles.items():
        for offset in range(seeds_per_device):
            episode_seed = seed + offset * 1009 + stable_seed_offset(device)

            retained_rng = random.Random(
                episode_seed + stable_seed_offset(f"retain:{retained_fraction:.6f}")
            )
            count = min(len(rules), max(1, math.floor(len(rules) * retained_fraction + 0.5)))
            retained = sorted(retained_rng.sample(range(len(rules)), k=count))
            if novel_count > 0 and len(retained) <= novel_count:
                continue
            query_rules = [rules[idx] for idx in retained]

            domain_rng = random.Random(
                episode_seed + stable_seed_offset(f"domain:{domain_fraction:.6f}")
            )
            eligible = [
                pos for pos, rule in enumerate(query_rules) if has_perturbable_domain(rule)
            ]
            target = 0
            if domain_fraction > 0.0 and eligible:
                target = min(
                    len(eligible), max(1, math.floor(len(eligible) * domain_fraction + 0.5))
                )
            changed = 0
            if target:
                for pos in sorted(domain_rng.sample(eligible, k=target)):
                    perturbed = perturb_rule_with_retries(
                        query_rules[pos], domain_rng, reference_texts
                    )
                    if perturbed is not None:
                        query_rules[pos] = perturbed
                        changed += 1
            if domain_fraction > 0.0 and (not eligible or changed == 0):
                continue

            removed: frozenset[str] = frozenset()
            if novel_count > 0:
                novel_rng = random.Random(
                    episode_seed + stable_seed_offset(f"novel:{novel_count}")
                )
                positions = sorted(novel_rng.sample(range(len(query_rules)), k=novel_count))
                removed = frozenset(rules[retained[pos]] for pos in positions)

            expected_reference = frozenset(rules) - removed
            queries.append(
                Query(
                    query_id=(
                        f"{device}/mixed/r{retained_fraction:.2f}_d{domain_fraction:.2f}"
                        f"_k{novel_count}_s{offset:02d}"
                    ),
                    expected_device=device,
                    cluster_key=f"{device}/s{offset:02d}",
                    query_texts=tuple(query_rules),
                    removed_texts=removed,
                    removed_scope="expected",
                    exact_hits=len(frozenset(query_rules) & expected_reference),
                )
            )
    return queries


# ---------------------------------------------------------------------------
# Embedding and scoring.
# ---------------------------------------------------------------------------


def load_whitened_reference(raw_npz: Path) -> tuple[AceBank, dict[str, object]]:
    """Whiten the raw reference bank locally and keep the transform."""
    data = np.load(raw_npz, allow_pickle=True)
    raw = data["embeddings"].astype(np.float32, copy=False)
    mean, components, singular_values = fit_whitening(raw)
    whitened = apply_whitening(raw, mean, components, singular_values, 256, raw.shape[0])
    bank = build_ace_bank(
        raw_npz,
        whitened,
        [str(value) for value in data["devices"]],
        [str(value) for value in data["ace_texts"]],
    )
    transform = {
        "mean": mean,
        "components": components,
        "singular_values": singular_values,
        "n_reference": raw.shape[0],
    }
    return bank, transform


def vectors_for_queries(
    bank: AceBank,
    transform: dict[str, object],
    queries: list[Query],
    *,
    model_name: str,
    device: str | None,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Map every query ACE text to a whitened vector.

    Texts already in the reference bank reuse its rows. New (drifted) texts
    are embedded with the sentence-transformer model and whitened with the
    reference transform.
    """
    vector_by_text: dict[str, np.ndarray] = {}
    for idx, text in enumerate(bank.ace_texts):
        vector_by_text.setdefault(text, bank.embeddings[idx])

    new_texts = sorted(
        {text for query in queries for text in query.query_texts if text not in vector_by_text}
    )
    if new_texts:
        print(f"Embedding {len(new_texts)} drifted ACE texts with {model_name}...")
        model = build_model(model_name, device)
        raw = encode_texts(model, new_texts, batch_size)
        whitened = apply_whitening(
            raw,
            transform["mean"],
            transform["components"],
            transform["singular_values"],
            256,
            transform["n_reference"],
        )
        vector_by_text.update(zip(new_texts, whitened))
    return vector_by_text


def score_queries(
    bank: AceBank,
    queries: list[Query],
    vector_by_text: dict[str, np.ndarray],
    *,
    top_k: int,
) -> list[dict[str, object]]:
    scored = []
    for query in queries:
        vectors = np.vstack([vector_by_text[text] for text in query.query_texts])
        scored.append(
            {
                "query": query,
                "expected_device": query.expected_device,
                "cluster_key": query.cluster_key,
                "scores": score_query(
                    bank,
                    list(query.query_texts),
                    vectors,
                    expected_device=query.expected_device,
                    removed_texts=query.removed_texts,
                    removed_scope=query.removed_scope,
                    top_k=top_k,
                ),
            }
        )
    return scored


def exact_hit_bin(hits: int) -> str:
    if hits <= 0:
        return "0"
    if hits <= 2:
        return "1-2"
    if hits <= 5:
        return "3-5"
    return ">5"


def summarise_bins(scored: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Top-1 accuracy per method, grouped by exact hits against the truth."""
    bins: dict[str, dict[str, object]] = {}
    for label in ("0", "1-2", "3-5", ">5"):
        rows = [item for item in scored if exact_hit_bin(item["query"].exact_hits) == label]
        if not rows:
            continue
        bins[label] = {"queries": len(rows)}
        for method in METHODS:
            correct = sum(1 for item in rows if item["scores"][method]["top1_correct"])
            bins[label][method] = correct / len(rows)
    return bins


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--compact-dir",
        type=Path,
        default=Path("data/ref_mud/compact"),
        help="Directory containing the reference compact .txt profiles.",
    )
    parser.add_argument(
        "--raw-npz",
        type=Path,
        default=Path("data/ref_embeddings/bge/per_ace/raw/reference_per_ace.npz"),
        help="Raw per-ACE reference bank used for whitening and scoring.",
    )
    parser.add_argument("--model-name", default="BAAI/bge-m3", help="Embedding model.")
    parser.add_argument("--device", default=None, help="Model device (cpu, cuda, mps).")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size.")
    parser.add_argument("--seed", type=int, default=1729, help="Base random seed.")
    parser.add_argument("--top-k", type=int, default=5, help="Ranked devices kept per query.")
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10000,
        help="Bootstrap resamples for the paired Top-1 intervals. 0 disables.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    drift = subparsers.add_parser(
        "drift",
        help="Queries containing only ACEs with drifted hostnames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    drift.add_argument(
        "--subset",
        choices=["full", "high-domain"],
        default="full",
        help="Device subset: all devices, or devices rich in domain ACEs.",
    )
    drift.add_argument("--variants", type=int, default=10, help="Drift variants per device.")
    drift.add_argument(
        "--fraction",
        type=float,
        default=0.10,
        help="Fraction of eligible ACEs perturbed per variant (capped at 3).",
    )
    drift.add_argument(
        "--high-domain-threshold",
        type=int,
        default=10,
        help="Minimum domain ACEs for the high-domain subset.",
    )

    mixed = subparsers.add_parser(
        "mixed",
        help="Queries mixing exact, drifted, and unseen ACEs over a grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mixed.add_argument(
        "--retained-fractions",
        type=float,
        nargs="+",
        default=[0.10, 0.25, 0.50],
        help="Fractions of the profile observed at runtime.",
    )
    mixed.add_argument(
        "--domain-fractions",
        type=float,
        nargs="+",
        default=[0.00, 0.25, 0.50, 1.00],
        help="Fractions of retained domain ACEs whose hostnames drift.",
    )
    mixed.add_argument(
        "--novel-counts",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Number of query ACEs made unseen in the correct reference.",
    )
    mixed.add_argument(
        "--seeds-per-device",
        type=int,
        default=10,
        help="Seeded queries per device per grid cell.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = read_compact_profiles(args.compact_dir)
    bank, transform = load_whitened_reference(args.raw_npz)

    if args.command == "drift":
        queries = build_drift_queries(
            profiles,
            subset=args.subset,
            variants=args.variants,
            fraction=args.fraction,
            seed=args.seed,
            high_domain_threshold=args.high_domain_threshold,
        )
        config = {
            "command": "drift",
            "subset": args.subset,
            "variants": args.variants,
            "fraction": args.fraction,
        }
    else:
        queries = []
        for retained_fraction in args.retained_fractions:
            for domain_fraction in args.domain_fractions:
                for novel_count in args.novel_counts:
                    queries.extend(
                        build_mixed_queries(
                            profiles,
                            retained_fraction=retained_fraction,
                            domain_fraction=domain_fraction,
                            novel_count=novel_count,
                            seeds_per_device=args.seeds_per_device,
                            seed=args.seed,
                        )
                    )
        config = {
            "command": "mixed",
            "retained_fractions": args.retained_fractions,
            "domain_fractions": args.domain_fractions,
            "novel_counts": args.novel_counts,
            "seeds_per_device": args.seeds_per_device,
        }
    if not queries:
        raise ValueError("No queries generated.")

    vector_by_text = vectors_for_queries(
        bank,
        transform,
        queries,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )
    scored = score_queries(bank, queries, vector_by_text, top_k=args.top_k)

    result: dict[str, object] = {
        "config": {**config, "seed": args.seed, "top_k": args.top_k},
        "query_count": len(queries),
        "device_count": len({query.expected_device for query in queries}),
        "summary": summarise_results(scored, args.top_k),
    }
    if args.command == "mixed":
        result["by_exact_hits"] = summarise_bins(scored)
    if args.bootstrap_resamples > 0:
        result["paired_top1_bootstrap"] = paired_top1_bootstrap(
            scored, resamples=args.bootstrap_resamples, seed=args.seed
        )

    print(
        f"Scored {result['query_count']} {config['command']} queries "
        f"from {result['device_count']} devices."
    )
    print("method             top1    topK     mrr   abstain")
    for method, row in result["summary"].items():
        topk_key = next(key for key in row if key.startswith("top") and key != "top1")
        print(
            f"{method:<18} "
            f"{row['top1']:.4f}  {row[topk_key]:.4f}  {row['mrr']:.4f}  {row['abstain_rate']:.4f}"
        )
    for label, row in result.get("by_exact_hits", {}).items():
        parts = "  ".join(f"{method}={row[method]:.4f}" for method in METHODS)
        print(f"exact hits {label:>3} ({row['queries']:>4} queries): {parts}")
    for name, row in result.get("paired_top1_bootstrap", {}).items():
        low, high = row["episode_ci95"]
        cluster_low, cluster_high = row["cluster_ci95"]
        print(
            f"{name}: {row['difference']:+.4f} "
            f"(episode 95% CI [{low:+.4f}, {high:+.4f}], "
            f"cluster 95% CI [{cluster_low:+.4f}, {cluster_high:+.4f}])"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.output}.")


if __name__ == "__main__":
    main()
