"""Deterministic Source-1-level validation splits."""

from __future__ import annotations

import hashlib
from typing import Iterable, Set, Tuple

import pandas as pd


def stable_fraction(entity_id: str, seed: int = 42) -> float:
    payload = f"{seed}:{entity_id}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return value / float(2**64)


def validation_mask(
    entity_ids: Iterable[str],
    *,
    validation_fraction: float,
    seed: int,
) -> pd.Series:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    values = list(entity_ids)
    return pd.Series(
        [stable_fraction(value, seed) < validation_fraction for value in values],
        index=getattr(entity_ids, "index", None),
        dtype=bool,
    )


def split_source1_ids(
    source1: pd.DataFrame,
    *,
    validation_fraction: float = 0.10,
    seed: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """Split whole S1 entities; associated pairs are never split independently."""

    if "entity_id" not in source1:
        raise ValueError("source1 frame requires entity_id")
    mask = validation_mask(
        source1["entity_id"], validation_fraction=validation_fraction, seed=seed
    )
    validation = set(source1.loc[mask.to_numpy(), "entity_id"])
    training = set(source1.loc[~mask.to_numpy(), "entity_id"])
    if validation & training:
        raise AssertionError("Source-1 split leaked entity IDs between partitions")
    return training, validation


def select_validation_entities(
    source1: pd.DataFrame,
    *,
    validation_fraction: float,
    seed: int,
    limit: int,
) -> pd.DataFrame:
    """Select a bounded deterministic validation subset while preserving S1 rows."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    mask = validation_mask(
        source1["entity_id"], validation_fraction=validation_fraction, seed=seed
    )
    return source1.loc[mask.to_numpy()].head(limit).reset_index(drop=True)

