"""Compact, LightGBM-ready numeric features for blocked candidate pairs."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, Iterator, List, Set

import numpy as np
import pandas as pd

from .blocking import PROVENANCE_COLUMNS


IDENTIFIER_COLUMNS = ["source1_entity_id", "candidate_entity_id"]

FEATURE_COLUMNS = [
    "name_full_exact",
    "name_core_exact",
    "name_sorted_exact",
    "name_token_jaccard",
    "name_shared_token_count",
    "name_token_count_difference",
    "name_length_ratio",
    "name_character_similarity",
    "name_prefix_ratio",
    "name_suffix_ratio",
    "address_exact",
    "address_token_jaccard",
    "address_shared_token_count",
    "numeric_token_equal",
    "numeric_token_overlap",
    "numeric_token_conflict",
    "address_length_ratio",
    "address_missing_left",
    "address_missing_right",
    "same_country",
    "country_conflict",
    "country_missing_left",
    "country_missing_right",
    "both_country_present",
    "source_is_s2",
    "source_is_s3",
    *PROVENANCE_COLUMNS,
    "blocking_channel_count",
    "rare_name_token_hits",
    "rare_address_token_hits",
    "name_tfidf_similarity",
    "name_tfidf_rank",
    "legacy_blocking_channel_count",
    "legacy_rank",
    "legacy_cheap_score",
    "is_tfidf_only",
    "is_legacy_only",
    "supported_by_both",
    "fusion_score",
    "fusion_rank",
    "candidate_count_for_s1",
    "cheap_score",
    "cheap_rank",
    "cheap_score_margin_to_next",
]

MODEL_RELATIVE_FEATURE_COLUMNS = [
    "relative_name_similarity_rank",
    "relative_address_similarity_rank",
    "relative_tfidf_similarity_rank",
    "relative_fusion_score_rank",
    "top_name_similarity_margin",
    "top_address_similarity_margin",
    "top_tfidf_similarity_margin",
    "fusion_score_gap_to_best",
    "close_name_competitor_count",
]


def _tokens(value: object) -> Set[str]:
    return set(str(value).split()) if value else set()


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _length_ratio(left: str, right: str) -> float:
    longest = max(len(left), len(right))
    return min(len(left), len(right)) / longest if longest else 1.0


def _edge_ratio(left: str, right: str, *, reverse: bool = False) -> float:
    if reverse:
        left, right = left[::-1], right[::-1]
    limit = min(len(left), len(right))
    shared = 0
    for i in range(limit):
        if left[i] != right[i]:
            break
        shared += 1
    longest = max(len(left), len(right))
    return shared / longest if longest else 1.0


def _pair_features(left: pd.Series, right: pd.Series, candidate: pd.Series) -> Dict[str, object]:
    left_name = str(left["name_full"])
    right_name = str(right["name_full"])
    left_name_tokens = _tokens(left["name_tokens"])
    right_name_tokens = _tokens(right["name_tokens"])
    left_addr = str(left["address_full"])
    right_addr = str(right["address_full"])
    left_addr_tokens = _tokens(left["address_tokens"])
    right_addr_tokens = _tokens(right["address_tokens"])
    left_numbers = _tokens(left["address_numeric_tokens"])
    right_numbers = _tokens(right["address_numeric_tokens"])
    left_country = str(left["country_norm"])
    right_country = str(right["country_norm"])

    both_numbers = bool(left_numbers and right_numbers)
    features: Dict[str, object] = {
        "name_full_exact": bool(left_name and left_name == right_name),
        "name_core_exact": bool(
            left["name_core"] and left["name_core"] == right["name_core"]
        ),
        "name_sorted_exact": bool(
            left["name_sorted"] and left["name_sorted"] == right["name_sorted"]
        ),
        "name_token_jaccard": _jaccard(left_name_tokens, right_name_tokens),
        "name_shared_token_count": len(left_name_tokens & right_name_tokens),
        "name_token_count_difference": abs(
            len(left_name_tokens) - len(right_name_tokens)
        ),
        "name_length_ratio": _length_ratio(left_name, right_name),
        "name_character_similarity": SequenceMatcher(
            None, left_name, right_name, autojunk=False
        ).ratio(),
        "name_prefix_ratio": _edge_ratio(left_name, right_name),
        "name_suffix_ratio": _edge_ratio(left_name, right_name, reverse=True),
        "address_exact": bool(left_addr and left_addr == right_addr),
        "address_token_jaccard": _jaccard(left_addr_tokens, right_addr_tokens),
        "address_shared_token_count": len(left_addr_tokens & right_addr_tokens),
        "numeric_token_equal": bool(both_numbers and left_numbers == right_numbers),
        "numeric_token_overlap": _jaccard(left_numbers, right_numbers)
        if both_numbers
        else 0.0,
        "numeric_token_conflict": bool(both_numbers and not (left_numbers & right_numbers)),
        "address_length_ratio": _length_ratio(left_addr, right_addr),
        "address_missing_left": not bool(left_addr),
        "address_missing_right": not bool(right_addr),
        "same_country": bool(left_country and left_country == right_country),
        "country_conflict": bool(
            left_country and right_country and left_country != right_country
        ),
        "country_missing_left": not bool(left_country),
        "country_missing_right": not bool(right_country),
        "both_country_present": bool(left_country and right_country),
        "source_is_s2": str(candidate["candidate_entity_id"]).startswith("S2-"),
        "source_is_s3": str(candidate["candidate_entity_id"]).startswith("S3-"),
    }
    channel_count = 0
    for column in PROVENANCE_COLUMNS:
        value = bool(candidate[column])
        features[column] = value
        channel_count += int(value)
    features["blocking_channel_count"] = channel_count
    for column in (
        "rare_name_token_hits",
        "rare_address_token_hits",
        "name_tfidf_similarity",
        "name_tfidf_rank",
        "candidate_count_for_s1",
        "cheap_score",
        "cheap_rank",
        "cheap_score_margin_to_next",
    ):
        features[column] = candidate[column]
    # E1.1 fusion metadata is optional so the unchanged Tier 0 candidate table
    # remains a valid input to this shared feature builder.
    features["legacy_blocking_channel_count"] = candidate.get(
        "legacy_blocking_channel_count", channel_count
    )
    features["legacy_rank"] = candidate.get("legacy_rank", candidate["cheap_rank"])
    features["legacy_cheap_score"] = candidate.get(
        "legacy_cheap_score", candidate["cheap_score"]
    )
    features["is_tfidf_only"] = candidate.get("is_tfidf_only", 0)
    features["is_legacy_only"] = candidate.get(
        "is_legacy_only", int(not bool(candidate.get("block_name_tfidf", 0)))
    )
    features["supported_by_both"] = candidate.get("supported_by_both", 0)
    features["fusion_score"] = candidate.get("fusion_score", candidate["cheap_score"])
    features["fusion_rank"] = candidate.get("fusion_rank", candidate["cheap_rank"])
    return features


def iter_feature_chunks(
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    feed: pd.DataFrame,
    *,
    chunk_size: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Yield feature chunks so full-scale scoring need not materialize all rows."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    left = source1.set_index("entity_id", verify_integrity=True)
    right = feed.set_index("entity_id", verify_integrity=True)
    missing_left = set(candidates["source1_entity_id"]) - set(left.index)
    missing_right = set(candidates["candidate_entity_id"]) - set(right.index)
    if missing_left or missing_right:
        raise ValueError(
            f"Candidate references unknown records; S1={list(missing_left)[:5]}, "
            f"feed={list(missing_right)[:5]}"
        )

    for start in range(0, len(candidates), chunk_size):
        block = candidates.iloc[start : start + chunk_size]
        rows: List[Dict[str, object]] = []
        for candidate in block.to_dict(orient="records"):
            s1_id = candidate["source1_entity_id"]
            target_id = candidate["candidate_entity_id"]
            values = {
                "source1_entity_id": s1_id,
                "candidate_entity_id": target_id,
                **_pair_features(
                    left.loc[s1_id], right.loc[target_id], pd.Series(candidate)
                ),
            }
            rows.append(values)
        result = pd.DataFrame(rows, columns=[*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS])
        yield compact_feature_dtypes(result)


def compact_feature_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    boolean_columns = [
        column
        for column in FEATURE_COLUMNS
        if column.endswith("_exact")
        or column.startswith("block_")
        or column
        in {
            "numeric_token_equal",
            "numeric_token_conflict",
            "address_missing_left",
            "address_missing_right",
            "same_country",
            "country_conflict",
            "country_missing_left",
            "country_missing_right",
            "both_country_present",
            "source_is_s2",
            "source_is_s3",
            "is_tfidf_only",
            "is_legacy_only",
            "supported_by_both",
        }
    ]
    for column in boolean_columns:
        result[column] = result[column].astype("uint8")
    count_columns = [
        "name_shared_token_count",
        "name_token_count_difference",
        "address_shared_token_count",
        "blocking_channel_count",
        "rare_name_token_hits",
        "rare_address_token_hits",
        "name_tfidf_rank",
        "legacy_blocking_channel_count",
        "legacy_rank",
        "fusion_rank",
        "candidate_count_for_s1",
        "cheap_rank",
    ]
    for column in count_columns:
        result[column] = pd.to_numeric(result[column], downcast="unsigned")
    float_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in boolean_columns and column not in count_columns
    ]
    for column in float_columns:
        result[column] = result[column].astype("float32")
    return result


def build_feature_table(
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    feed: pd.DataFrame,
    *,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    chunks = list(
        iter_feature_chunks(
            candidates, source1, feed, chunk_size=chunk_size
        )
    )
    if not chunks:
        return pd.DataFrame(columns=[*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS])
    return pd.concat(chunks, ignore_index=True)


def _deterministic_descending_rank(
    frame: pd.DataFrame, score_column: str
) -> pd.Series:
    ordered = frame.sort_values(
        ["source1_entity_id", score_column, "candidate_entity_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    ranks = (
        ordered.groupby("source1_entity_id", sort=False).cumcount() + 1
    ).astype("uint16")
    return ranks.reindex(frame.index)


def _top_margin(frame: pd.DataFrame, score_column: str) -> pd.Series:
    ordered = frame.sort_values(
        ["source1_entity_id", score_column, "candidate_entity_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    positions = ordered.groupby("source1_entity_id", sort=False).cumcount()
    top = ordered.loc[positions == 0].set_index("source1_entity_id")[score_column]
    second = (
        ordered.loc[positions == 1]
        .set_index("source1_entity_id")[score_column]
        .reindex(top.index)
        .fillna(top)
    )
    margins = (top - second).astype("float32")
    return frame["source1_entity_id"].map(margins).astype("float32")


def add_candidate_relative_features(
    features: pd.DataFrame,
    *,
    close_competitor_tolerance: float = 0.05,
) -> pd.DataFrame:
    """Add label-free features calculated within each S1 candidate set."""

    if close_competitor_tolerance < 0:
        raise ValueError("close_competitor_tolerance must be non-negative")
    required = {
        "source1_entity_id",
        "candidate_entity_id",
        "name_character_similarity",
        "address_token_jaccard",
        "name_tfidf_similarity",
        "fusion_score",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Cannot build relative features; missing {sorted(missing)}")
    result = features.copy()
    rank_specs = {
        "relative_name_similarity_rank": "name_character_similarity",
        "relative_address_similarity_rank": "address_token_jaccard",
        "relative_tfidf_similarity_rank": "name_tfidf_similarity",
        "relative_fusion_score_rank": "fusion_score",
    }
    for output, score in rank_specs.items():
        result[output] = _deterministic_descending_rank(result, score)
    result["top_name_similarity_margin"] = _top_margin(
        result, "name_character_similarity"
    )
    result["top_address_similarity_margin"] = _top_margin(
        result, "address_token_jaccard"
    )
    result["top_tfidf_similarity_margin"] = _top_margin(
        result, "name_tfidf_similarity"
    )
    best_fusion = result.groupby("source1_entity_id")["fusion_score"].transform(
        "max"
    )
    result["fusion_score_gap_to_best"] = (
        best_fusion - result["fusion_score"]
    ).astype("float32")
    best_name = result.groupby("source1_entity_id")[
        "name_character_similarity"
    ].transform("max")
    close = result["name_character_similarity"] >= (
        best_name - close_competitor_tolerance
    )
    result["close_name_competitor_count"] = close.groupby(
        result["source1_entity_id"]
    ).transform("sum").astype("uint16")
    return result
