"""Interpretable fusion of legacy and name TF-IDF candidate rankings."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .blocking import PROVENANCE_COLUMNS


LEGACY_PROVENANCE_COLUMNS = [
    column for column in PROVENANCE_COLUMNS if column != "block_name_tfidf"
]


def prepare_fusion_universe(
    legacy_candidates: pd.DataFrame,
    tfidf_candidates: pd.DataFrame,
    *,
    tfidf_top_k: int,
) -> pd.DataFrame:
    """Create one deduplicated pair table with both retrieval systems' evidence."""

    if tfidf_top_k <= 0:
        raise ValueError("tfidf_top_k must be positive")
    legacy = legacy_candidates.copy()
    tfidf = tfidf_candidates.loc[
        tfidf_candidates["name_tfidf_rank"] <= tfidf_top_k
    ].copy()
    legacy["has_legacy"] = 1
    tfidf["has_tfidf"] = 1
    keep_legacy = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "same_country_block",
        "cheap_score",
        "cheap_rank",
        "cheap_score_margin_to_next",
        "rare_name_token_hits",
        "rare_address_token_hits",
        *LEGACY_PROVENANCE_COLUMNS,
        "has_legacy",
    ]
    missing = set(keep_legacy).difference(legacy.columns)
    if missing:
        raise ValueError(f"Legacy candidates lack fusion fields: {sorted(missing)}")
    keep_tfidf = [
        "source1_entity_id",
        "candidate_entity_id",
        "name_tfidf_similarity",
        "name_tfidf_rank",
        "has_tfidf",
    ]
    missing = set(keep_tfidf).difference(tfidf.columns)
    if missing:
        raise ValueError(f"TF-IDF candidates lack fusion fields: {sorted(missing)}")

    union = legacy[keep_legacy].merge(
        tfidf[keep_tfidf],
        on=["source1_entity_id", "candidate_entity_id"],
        how="outer",
        validate="one_to_one",
    )
    union["has_legacy"] = union["has_legacy"].fillna(0).astype("uint8")
    union["has_tfidf"] = union["has_tfidf"].fillna(0).astype("uint8")
    union["legacy_rank"] = union["cheap_rank"].fillna(0).astype("uint16")
    union["legacy_cheap_score"] = union["cheap_score"].fillna(0.0).astype("float32")
    union["name_tfidf_rank"] = union["name_tfidf_rank"].fillna(0).astype("uint16")
    union["name_tfidf_similarity"] = union["name_tfidf_similarity"].fillna(
        0.0
    ).astype("float32")
    union["block_name_tfidf"] = union["has_tfidf"].astype("uint8")
    union["is_tfidf_only"] = (
        (union["has_tfidf"] == 1) & (union["has_legacy"] == 0)
    ).astype("uint8")
    union["is_legacy_only"] = (
        (union["has_legacy"] == 1) & (union["has_tfidf"] == 0)
    ).astype("uint8")
    union["supported_by_both"] = (
        (union["has_legacy"] == 1) & (union["has_tfidf"] == 1)
    ).astype("uint8")

    for column in LEGACY_PROVENANCE_COLUMNS:
        union[column] = union[column].fillna(0).astype("uint8")
    union["legacy_blocking_channel_count"] = union[
        LEGACY_PROVENANCE_COLUMNS
    ].sum(axis=1).astype("uint8")
    union["candidate_source"] = union["candidate_source"].fillna(
        union["candidate_entity_id"].str[:2]
    )
    # TF-IDF is country-sharded, so a TF-IDF-only pair has same-country support.
    union["same_country_block"] = union["same_country_block"].fillna(
        union["has_tfidf"]
    ).astype("uint8")
    for column in (
        "rare_name_token_hits",
        "rare_address_token_hits",
    ):
        union[column] = union[column].fillna(0).astype("uint8")
    union["cheap_score"] = union["legacy_cheap_score"].astype("float32")
    union["cheap_score_margin_to_next"] = union[
        "cheap_score_margin_to_next"
    ].fillna(0.0).astype("float32")
    union["candidate_count_raw"] = union.groupby("source1_entity_id")[
        "candidate_entity_id"
    ].transform("size").astype("uint16")
    if union.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise AssertionError("Fusion universe contains duplicate candidate pairs")
    return union.reset_index(drop=True)


def _finalize(
    selected: pd.DataFrame,
    *,
    final_k: int,
    strategy: str,
) -> pd.DataFrame:
    if final_k <= 0:
        raise ValueError("final_k must be positive")
    result = selected.sort_values(
        [
            "source1_entity_id",
            "fusion_score",
            "supported_by_both",
            "legacy_rank_sort",
            "tfidf_rank_sort",
            "candidate_entity_id",
        ],
        ascending=[True, False, False, True, True, True],
        kind="mergesort",
    ).copy()
    result["fusion_rank"] = (
        result.groupby("source1_entity_id", sort=False).cumcount() + 1
    ).astype("uint16")
    result = result.loc[result["fusion_rank"] <= final_k].copy()
    counts = result.groupby("source1_entity_id").size().astype("uint16")
    result["candidate_count_for_s1"] = result["source1_entity_id"].map(counts).astype(
        "uint16"
    )
    # Existing downstream feature/submission code uses cheap_rank as the final
    # candidate order. Legacy rank remains separately available.
    result["cheap_rank"] = result["fusion_rank"].astype("uint16")
    result["fusion_strategy"] = strategy
    result.drop(columns=["legacy_rank_sort", "tfidf_rank_sort"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    if result.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise AssertionError("Fusion emitted duplicate pairs")
    if result.groupby("source1_entity_id").size().max() > final_k:
        raise AssertionError("Fusion exceeded final K")
    return result


def _rank_helpers(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    large = np.iinfo(np.uint16).max
    result["legacy_rank_sort"] = result["legacy_rank"].replace(0, large)
    result["tfidf_rank_sort"] = result["name_tfidf_rank"].replace(0, large)
    return result


def quota_fusion(
    universe: pd.DataFrame,
    *,
    final_k: int,
    legacy_quota: int,
    tfidf_addition_quota: int,
) -> pd.DataFrame:
    """Reserve slots for legacy pairs and genuinely TF-IDF-only additions."""

    if legacy_quota < 0 or tfidf_addition_quota < 0:
        raise ValueError("Fusion quotas must be non-negative")
    if legacy_quota + tfidf_addition_quota > final_k:
        raise ValueError("Reserved quotas exceed final K")
    pieces: List[pd.DataFrame] = []
    for _, group in universe.groupby("source1_entity_id", sort=False):
        group = _rank_helpers(group)
        legacy = group.loc[group["has_legacy"] == 1].sort_values(
            ["legacy_rank_sort", "candidate_entity_id"], kind="mergesort"
        )
        chosen = legacy.head(legacy_quota)
        chosen_ids = set(chosen["candidate_entity_id"])
        additions = group.loc[
            (group["is_tfidf_only"] == 1)
            & ~group["candidate_entity_id"].isin(chosen_ids)
        ].sort_values(["tfidf_rank_sort", "candidate_entity_id"], kind="mergesort")
        chosen = pd.concat(
            [chosen, additions.head(tfidf_addition_quota)], ignore_index=False
        )
        chosen_ids = set(chosen["candidate_entity_id"])
        remaining_slots = final_k - len(chosen)
        if remaining_slots > 0:
            remaining = group.loc[~group["candidate_entity_id"].isin(chosen_ids)].copy()
            # Unused quota first returns to legacy rank, then to TF-IDF rank.
            remaining["fill_legacy_priority"] = 1 - remaining["has_legacy"]
            remaining = remaining.sort_values(
                [
                    "fill_legacy_priority",
                    "legacy_rank_sort",
                    "tfidf_rank_sort",
                    "candidate_entity_id",
                ],
                kind="mergesort",
            ).drop(columns="fill_legacy_priority")
            chosen = pd.concat([chosen, remaining.head(remaining_slots)], ignore_index=False)
        chosen = chosen.copy()
        chosen["fusion_score"] = np.arange(len(chosen), 0, -1, dtype="float32")
        pieces.append(chosen)
    selected = pd.concat(pieces, ignore_index=True) if pieces else universe.head(0).copy()
    return _finalize(
        selected,
        final_k=final_k,
        strategy=f"quota_legacy{legacy_quota}_tfidf{tfidf_addition_quota}",
    )


def reciprocal_rank_fusion(
    universe: pd.DataFrame,
    *,
    final_k: int,
    legacy_weight: float,
    tfidf_weight: float,
    rrf_constant: float,
) -> pd.DataFrame:
    if min(legacy_weight, tfidf_weight, rrf_constant) <= 0:
        raise ValueError("RRF weights and constant must be positive")
    result = _rank_helpers(universe)
    legacy_part = np.where(
        result["legacy_rank"] > 0,
        legacy_weight / (rrf_constant + result["legacy_rank"]),
        0.0,
    )
    tfidf_part = np.where(
        result["name_tfidf_rank"] > 0,
        tfidf_weight / (rrf_constant + result["name_tfidf_rank"]),
        0.0,
    )
    result["fusion_score"] = (legacy_part + tfidf_part).astype("float32")
    return _finalize(
        result,
        final_k=final_k,
        strategy=(
            f"rrf_lw{legacy_weight:g}_tw{tfidf_weight:g}_c{rrf_constant:g}"
        ),
    )


def naive_score_fusion(
    universe: pd.DataFrame,
    *,
    final_k: int,
    tfidf_weight: float,
    same_country_weight: float,
) -> pd.DataFrame:
    """Reproduce E1's raw cheap-score union for displacement analysis."""

    result = _rank_helpers(universe)
    result["fusion_score"] = (
        result["legacy_cheap_score"]
        + tfidf_weight * result["name_tfidf_similarity"]
        + same_country_weight * result["is_tfidf_only"]
    ).astype("float32")
    return _finalize(result, final_k=final_k, strategy="naive_score_union")


def _group_minmax(values: pd.Series, present: pd.Series) -> pd.Series:
    result = pd.Series(0.0, index=values.index, dtype="float32")
    active = values.loc[present]
    if active.empty:
        return result
    minimum = float(active.min())
    maximum = float(active.max())
    if maximum == minimum:
        result.loc[active.index] = 1.0
    else:
        result.loc[active.index] = ((active - minimum) / (maximum - minimum)).astype(
            "float32"
        )
    return result


def normalized_score_fusion(
    universe: pd.DataFrame,
    *,
    final_k: int,
    legacy_alpha: float,
    tfidf_beta: float,
    provenance_bonus: float,
) -> pd.DataFrame:
    if legacy_alpha < 0 or tfidf_beta < 0 or provenance_bonus < 0:
        raise ValueError("Normalized-score weights must be non-negative")
    result = _rank_helpers(universe)
    legacy_norm_parts = []
    tfidf_norm_parts = []
    for _, group in result.groupby("source1_entity_id", sort=False):
        legacy_norm_parts.append(
            _group_minmax(group["legacy_cheap_score"], group["has_legacy"] == 1)
        )
        tfidf_norm_parts.append(
            _group_minmax(
                group["name_tfidf_similarity"], group["has_tfidf"] == 1
            )
        )
    legacy_norm = pd.concat(legacy_norm_parts).sort_index()
    tfidf_norm = pd.concat(tfidf_norm_parts).sort_index()
    result["normalized_legacy_score"] = legacy_norm.astype("float32")
    result["normalized_tfidf_score"] = tfidf_norm.astype("float32")
    result["fusion_score"] = (
        legacy_alpha * result["normalized_legacy_score"]
        + tfidf_beta * result["normalized_tfidf_score"]
        + provenance_bonus * result["supported_by_both"]
    ).astype("float32")
    return _finalize(
        result,
        final_k=final_k,
        strategy=(
            f"normalized_a{legacy_alpha:g}_b{tfidf_beta:g}_p{provenance_bonus:g}"
        ),
    )


def hybrid_quota_rank_fusion(
    universe: pd.DataFrame,
    *,
    final_k: int,
    minimum_legacy: int,
    legacy_weight: float,
    tfidf_weight: float,
    rrf_constant: float,
) -> pd.DataFrame:
    """Guarantee legacy slots, then fill remaining slots by RRF."""

    if minimum_legacy < 0 or minimum_legacy > final_k:
        raise ValueError("minimum_legacy must be between zero and final_k")
    ranked = reciprocal_rank_fusion(
        universe,
        final_k=max(final_k, int(universe.groupby("source1_entity_id").size().max())),
        legacy_weight=legacy_weight,
        tfidf_weight=tfidf_weight,
        rrf_constant=rrf_constant,
    )
    pieces: List[pd.DataFrame] = []
    for _, group in ranked.groupby("source1_entity_id", sort=False):
        group = _rank_helpers(group)
        protected = group.loc[group["has_legacy"] == 1].sort_values(
            ["legacy_rank_sort", "candidate_entity_id"], kind="mergesort"
        ).head(minimum_legacy)
        protected_ids = set(protected["candidate_entity_id"])
        fill = group.loc[~group["candidate_entity_id"].isin(protected_ids)].sort_values(
            ["fusion_rank", "candidate_entity_id"], kind="mergesort"
        )
        pieces.append(
            pd.concat(
                [protected, fill.head(final_k - len(protected))], ignore_index=False
            )
        )
    selected = pd.concat(pieces, ignore_index=True) if pieces else universe.head(0).copy()
    return _finalize(
        selected,
        final_k=final_k,
        strategy=(
            f"hybrid_legacy{minimum_legacy}_lw{legacy_weight:g}_"
            f"tw{tfidf_weight:g}_c{rrf_constant:g}"
        ),
    )
