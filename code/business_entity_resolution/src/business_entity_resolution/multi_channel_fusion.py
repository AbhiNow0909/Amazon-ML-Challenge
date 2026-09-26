"""Deterministic fusion of legacy, name, and address retrieval channels."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .candidate_fusion import prepare_fusion_universe


def prepare_multi_channel_universe(
    legacy_candidates: pd.DataFrame,
    name_candidates: pd.DataFrame,
    address_candidates: pd.DataFrame,
    *,
    name_top_k: int,
    address_top_k: int,
) -> pd.DataFrame:
    """Outer-join all channels while retaining scores, ranks, and provenance."""

    if address_top_k <= 0:
        raise ValueError("address_top_k must be positive")
    universe = prepare_fusion_universe(
        legacy_candidates, name_candidates, tfidf_top_k=name_top_k
    )
    address_provenance = [
        column
        for column in (
            "address_route_postcode",
            "address_route_postcode_prefix",
            "address_route_numeric",
            "address_route_rare_token",
            "address_route_country_fallback",
            "address_route_count",
            "address_preblock_pool_size",
        )
        if column in address_candidates.columns
    ]
    address = address_candidates.loc[
        address_candidates["address_tfidf_rank"] <= address_top_k,
        [
            "source1_entity_id",
            "candidate_entity_id",
            "address_tfidf_similarity",
            "address_tfidf_rank",
        ]
        + address_provenance,
    ].copy()
    address["has_address_tfidf"] = np.uint8(1)
    universe = universe.merge(
        address,
        on=["source1_entity_id", "candidate_entity_id"],
        how="outer",
        validate="one_to_one",
    )
    universe["has_legacy"] = universe["has_legacy"].fillna(0).astype("uint8")
    universe["has_tfidf"] = universe["has_tfidf"].fillna(0).astype("uint8")
    universe["has_address_tfidf"] = (
        universe["has_address_tfidf"].fillna(0).astype("uint8")
    )
    universe["legacy_rank"] = universe["legacy_rank"].fillna(0).astype("uint16")
    universe["name_tfidf_rank"] = (
        universe["name_tfidf_rank"].fillna(0).astype("uint16")
    )
    universe["address_tfidf_rank"] = (
        universe["address_tfidf_rank"].fillna(0).astype("uint16")
    )
    universe["name_tfidf_similarity"] = (
        universe["name_tfidf_similarity"].fillna(0.0).astype("float32")
    )
    universe["address_tfidf_similarity"] = (
        universe["address_tfidf_similarity"].fillna(0.0).astype("float32")
    )
    for column in address_provenance:
        universe[column] = pd.to_numeric(
            universe[column].fillna(0), downcast="unsigned"
        )
    universe["block_address_tfidf"] = universe["has_address_tfidf"]
    universe["is_address_tfidf_only"] = (
        (universe["has_address_tfidf"] == 1)
        & (universe["has_legacy"] == 0)
        & (universe["has_tfidf"] == 0)
    ).astype("uint8")
    universe["retrieval_channel_count"] = universe[
        ["has_legacy", "has_tfidf", "has_address_tfidf"]
    ].sum(axis=1).astype("uint8")
    universe["candidate_source"] = universe["candidate_source"].fillna(
        universe["candidate_entity_id"].str[:2]
    )
    universe["same_country_block"] = universe["same_country_block"].fillna(
        universe["has_address_tfidf"]
    ).astype("uint8")
    if universe.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise AssertionError("Multi-channel universe contains duplicate pairs")
    return universe.reset_index(drop=True)


def _score_rrf(
    universe: pd.DataFrame,
    *,
    legacy_weight: float,
    name_weight: float,
    address_weight: float,
    rrf_constant: float,
) -> pd.DataFrame:
    if min(legacy_weight, name_weight, address_weight, rrf_constant) <= 0:
        raise ValueError("RRF weights and constant must be positive")
    result = universe.copy()
    result["fusion_score"] = (
        np.where(
            result["legacy_rank"] > 0,
            legacy_weight / (rrf_constant + result["legacy_rank"]),
            0.0,
        )
        + np.where(
            result["name_tfidf_rank"] > 0,
            name_weight / (rrf_constant + result["name_tfidf_rank"]),
            0.0,
        )
        + np.where(
            result["address_tfidf_rank"] > 0,
            address_weight / (rrf_constant + result["address_tfidf_rank"]),
            0.0,
        )
    ).astype("float32")
    return result


def _sort_and_cap(frame: pd.DataFrame, *, final_k: int, strategy: str) -> pd.DataFrame:
    if final_k <= 0:
        raise ValueError("final_k must be positive")
    large = np.iinfo(np.uint16).max
    result = frame.copy()
    result["_legacy_sort"] = result["legacy_rank"].replace(0, large)
    result["_name_sort"] = result["name_tfidf_rank"].replace(0, large)
    result["_address_sort"] = result["address_tfidf_rank"].replace(0, large)
    result = result.sort_values(
        [
            "source1_entity_id",
            "fusion_score",
            "retrieval_channel_count",
            "_legacy_sort",
            "_name_sort",
            "_address_sort",
            "candidate_entity_id",
        ],
        ascending=[True, False, False, True, True, True, True],
        kind="mergesort",
    )
    result["fusion_rank"] = (
        result.groupby("source1_entity_id", sort=False).cumcount() + 1
    ).astype("uint16")
    result = result.loc[result["fusion_rank"] <= final_k].copy()
    counts = result.groupby("source1_entity_id").size().astype("uint16")
    result["candidate_count_for_s1"] = result["source1_entity_id"].map(counts)
    result["cheap_rank"] = result["fusion_rank"]
    result["fusion_strategy"] = strategy
    result.drop(columns=["_legacy_sort", "_name_sort", "_address_sort"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    if result.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise AssertionError("Multi-channel fusion emitted duplicate pairs")
    if result.groupby("source1_entity_id").size().max() > final_k:
        raise AssertionError("Multi-channel fusion exceeded final K")
    return result


def multi_channel_rrf(
    universe: pd.DataFrame,
    *,
    final_k: int,
    legacy_weight: float,
    name_weight: float,
    address_weight: float,
    rrf_constant: float,
) -> pd.DataFrame:
    scored = _score_rrf(
        universe,
        legacy_weight=legacy_weight,
        name_weight=name_weight,
        address_weight=address_weight,
        rrf_constant=rrf_constant,
    )
    return _sort_and_cap(
        scored,
        final_k=final_k,
        strategy=(
            f"rrf_lw{legacy_weight:g}_nw{name_weight:g}_"
            f"aw{address_weight:g}_c{rrf_constant:g}"
        ),
    )


def protected_legacy_multi_channel_rrf(
    universe: pd.DataFrame,
    *,
    final_k: int,
    minimum_legacy: int,
    legacy_weight: float,
    name_weight: float,
    address_weight: float,
    rrf_constant: float,
) -> pd.DataFrame:
    """Guarantee legacy slots, then fill remaining slots by three-way RRF."""

    if minimum_legacy < 0 or minimum_legacy > final_k:
        raise ValueError("minimum_legacy must be between zero and final_k")
    scored = _score_rrf(
        universe,
        legacy_weight=legacy_weight,
        name_weight=name_weight,
        address_weight=address_weight,
        rrf_constant=rrf_constant,
    )
    pieces: List[pd.DataFrame] = []
    for _, group in scored.groupby("source1_entity_id", sort=False):
        protected = group.loc[group["has_legacy"] == 1].sort_values(
            ["legacy_rank", "candidate_entity_id"], kind="mergesort"
        ).head(minimum_legacy)
        remaining = group.loc[~group.index.isin(protected.index)].sort_values(
            ["fusion_score", "retrieval_channel_count", "candidate_entity_id"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        pieces.append(
            pd.concat(
                [protected, remaining.head(final_k - len(protected))],
                ignore_index=False,
            )
        )
    selected = pd.concat(pieces, ignore_index=True) if pieces else scored.head(0)
    return _sort_and_cap(
        selected,
        final_k=final_k,
        strategy=(
            f"protected{minimum_legacy}_lw{legacy_weight:g}_nw{name_weight:g}_"
            f"aw{address_weight:g}_c{rrf_constant:g}"
        ),
    )
