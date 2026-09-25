"""Deterministic conservative normalization for names and addresses."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Sequence, Set

import pandas as pd


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w]+", flags=re.UNICODE)
_NUMBER_RE = re.compile(r"(?<!\w)\d+[a-z]?(?!\w)", flags=re.IGNORECASE)
_POSTCODE_RE = re.compile(r"(?<!\d)(?:\d{5}(?:-\d{4})?|\d{6})(?!\d)")

# Kept deliberately short and suffix-oriented. The full normalized name is
# always retained, so stripping cannot erase the original signal.
LEGAL_FORMS: Set[str] = {
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "llc",
    "llp",
    "ltd",
    "limited",
    "private",
    "pvt",
    "plc",
    "sa",
    "sas",
    "sarl",
    "eurl",
    "snc",
    "cie",
}


def _strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_text(value: object, *, normalize_connectors: bool = False) -> str:
    """Normalize Unicode, case, punctuation, and whitespace.

    The function does not transliterate non-Latin scripts. Diacritics are folded
    to make accented/unaccented Latin variants comparable.
    """

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = _strip_accents(text)
    if normalize_connectors:
        text = re.sub(r"(?<!\w)[&+](?!\w)", " and ", text)
    text = _PUNCT_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_country(value: object) -> str:
    """Normalize any country label without a closed vocabulary."""

    if value is None:
        return ""
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", str(value)).strip()).casefold()


def detect_legal_forms(normalized_name: str) -> List[str]:
    tokens = normalized_name.split()
    detected: List[str] = []
    while tokens and tokens[-1] in LEGAL_FORMS:
        detected.append(tokens.pop())
    detected.reverse()
    return detected


def strip_legal_forms(normalized_name: str) -> str:
    tokens = normalized_name.split()
    while tokens and tokens[-1] in LEGAL_FORMS:
        tokens.pop()
    return " ".join(tokens) if tokens else normalized_name


def sorted_token_view(normalized_name: str) -> str:
    return " ".join(sorted(normalized_name.split()))


def simple_acronym(core_name: str) -> str:
    tokens = core_name.split()
    if len(tokens) < 2 or len(tokens) > 10:
        return ""
    return "".join(token[0] for token in tokens if token)


def generate_name_views(value: object) -> Dict[str, object]:
    raw = "" if value is None else str(value)
    full = normalize_text(raw, normalize_connectors=True)
    core = strip_legal_forms(full)
    return {
        "name_raw": raw,
        "name_full": full,
        "name_core": core,
        "name_sorted": sorted_token_view(core),
        "name_tokens": tuple(full.split()),
        "legal_form": " ".join(detect_legal_forms(full)),
        "name_acronym": simple_acronym(core),
    }


def extract_numeric_tokens(normalized_address: str) -> List[str]:
    return _NUMBER_RE.findall(normalized_address)


def extract_postcode(normalized_address: str) -> str:
    matches = _POSTCODE_RE.findall(normalized_address)
    return matches[-1].replace("-", "") if matches else ""


def generate_address_views(value: object) -> Dict[str, object]:
    raw = "" if value is None else str(value)
    full = normalize_text(raw)
    # Tier 0 keeps the core view light; future country-aware dictionaries can
    # extend this without changing the feature schema.
    core = full
    numbers = tuple(extract_numeric_tokens(core))
    return {
        "address_raw": raw,
        "address_full": full,
        "address_core": core,
        "address_tokens": tuple(core.split()),
        "address_numeric_tokens": numbers,
        "postcode": extract_postcode(core),
    }


def normalize_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy of a source frame suitable for Parquet caching."""

    required = {"entity_id", "business_name", "business_address", "country"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cannot normalize records; missing columns: {sorted(missing)}")

    result = frame.copy()
    names = result["business_name"].map(generate_name_views)
    addresses = result["business_address"].map(generate_address_views)
    for column in (
        "name_raw",
        "name_full",
        "name_core",
        "name_sorted",
        "legal_form",
        "name_acronym",
    ):
        result[column] = names.map(lambda views, key=column: views[key])
    result["name_tokens"] = names.map(lambda views: " ".join(views["name_tokens"]))

    for column in ("address_raw", "address_full", "address_core", "postcode"):
        result[column] = addresses.map(lambda views, key=column: views[key])
    result["address_tokens"] = addresses.map(
        lambda views: " ".join(views["address_tokens"])
    )
    result["address_numeric_tokens"] = addresses.map(
        lambda views: " ".join(views["address_numeric_tokens"])
    )
    result["country_norm"] = result["country"].map(normalize_country)
    return result


def informative_tokens(
    token_string: str,
    *,
    minimum_length: int = 3,
) -> Set[str]:
    return {
        token
        for token in token_string.split()
        if len(token) >= minimum_length and not token.isdigit()
    }

