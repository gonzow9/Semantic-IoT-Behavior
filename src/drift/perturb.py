"""Hostname perturbation for the drifted-endpoint experiments.

Rewrites hostnames in compact ACE lines with small, realistic mutations
(numeric changes, region tokens, domain-suffix swaps) while protocol and
port stay the same. Perturbed texts that collide with any reference ACE
are rejected and retried.
"""

from __future__ import annotations

import ipaddress
import random
import re

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

