"""Bounded, provenance-aware Tier 0 candidate generation."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import DefaultDict, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .normalize import informative_tokens
from .tfidf_retrieval import retrieve_name_tfidf


PROVENANCE_COLUMNS = [
    "block_exact_full_name",
    "block_exact_core_name",
    "block_exact_sorted_name",
    "block_rare_name_token",
    "block_exact_address",
    "block_rare_address_token",
    "block_cross_country_exact_name",
    "block_name_tfidf",
]


def _validate_inputs(source1: pd.DataFrame, feed: pd.DataFrame) -> None:
    normalized = {
        "entity_id",
        "name_full",
        "name_core",
        "name_sorted",
        "name_tokens",
        "address_full",
        "address_tokens",
        "country_norm",
    }
    for label, frame in (("source1", source1), ("feed", feed)):
        missing = normalized.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing normalized columns: {sorted(missing)}")
    invalid = ~feed["entity_id"].str.startswith(("S2-", "S3-"))
    if invalid.any():
        raise ValueError(
            f"Feed contains non-S2/S3 IDs: {feed.loc[invalid, 'entity_id'].head(5).tolist()}"
        )


def _build_indexes(
    feed: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[
    Dict[object, List[int]],
    Dict[object, List[int]],
    Dict[object, List[int]],
    Dict[object, List[int]],
    Dict[object, List[int]],
    Dict[Tuple[str, str], List[int]],
    Dict[Tuple[str, str], List[int]],
    Counter,
    Counter,
]:
    """Build every Tier 0 index in two feed passes.

    Keeping the passes consolidated is essential at full scale: separate
    ``iterrows`` passes for every channel dominated the initial validation run.
    """

    exact_full_count: Counter = Counter()
    exact_core_count: Counter = Counter()
    exact_sorted_count: Counter = Counter()
    exact_address_count: Counter = Counter()
    global_full_count: Counter = Counter()
    name_df: Counter = Counter()
    address_df: Counter = Counter()

    columns = [
        "name_full",
        "name_core",
        "name_sorted",
        "name_tokens",
        "address_full",
        "address_tokens",
        "country_norm",
    ]
    for row in feed[columns].itertuples(index=False, name=None):
        name_full, name_core, name_sorted, name_tokens, address_full, address_tokens, country = row
        if name_full:
            exact_full_count[(country, name_full)] += 1
            global_full_count[name_full] += 1
        if name_core:
            exact_core_count[(country, name_core)] += 1
        if name_sorted:
            exact_sorted_count[(country, name_sorted)] += 1
        if address_full:
            exact_address_count[(country, address_full)] += 1
        for token in informative_tokens(
            name_tokens, minimum_length=config.minimum_token_length
        ):
            name_df[(country, token)] += 1
        for token in informative_tokens(
            address_tokens, minimum_length=config.minimum_token_length
        ):
            address_df[(country, token)] += 1

    exact_full: DefaultDict[object, List[int]] = defaultdict(list)
    exact_core: DefaultDict[object, List[int]] = defaultdict(list)
    exact_sorted: DefaultDict[object, List[int]] = defaultdict(list)
    exact_address: DefaultDict[object, List[int]] = defaultdict(list)
    global_exact_full: DefaultDict[object, List[int]] = defaultdict(list)
    rare_name: DefaultDict[Tuple[str, str], List[int]] = defaultdict(list)
    rare_address: DefaultDict[Tuple[str, str], List[int]] = defaultdict(list)
    name_limit = min(config.rare_name_token_max_df, config.posting_list_cap)
    address_limit = min(config.rare_address_token_max_df, config.posting_list_cap)

    for idx, row in enumerate(feed[columns].itertuples(index=False, name=None)):
        name_full, name_core, name_sorted, name_tokens, address_full, address_tokens, country = row
        if name_full:
            country_key = (country, name_full)
            if exact_full_count[country_key] <= config.posting_list_cap:
                exact_full[country_key].append(idx)
            if global_full_count[name_full] <= config.posting_list_cap:
                global_exact_full[name_full].append(idx)
        if name_core:
            key = (country, name_core)
            if exact_core_count[key] <= config.posting_list_cap:
                exact_core[key].append(idx)
        if name_sorted:
            key = (country, name_sorted)
            if exact_sorted_count[key] <= config.posting_list_cap:
                exact_sorted[key].append(idx)
        if address_full:
            key = (country, address_full)
            if exact_address_count[key] <= config.posting_list_cap:
                exact_address[key].append(idx)
        for token in informative_tokens(
            name_tokens, minimum_length=config.minimum_token_length
        ):
            key = (country, token)
            if name_df[key] <= name_limit:
                rare_name[key].append(idx)
        for token in informative_tokens(
            address_tokens, minimum_length=config.minimum_token_length
        ):
            key = (country, token)
            if address_df[key] <= address_limit:
                rare_address[key].append(idx)

    return (
        dict(exact_full),
        dict(exact_core),
        dict(exact_sorted),
        dict(exact_address),
        dict(global_exact_full),
        dict(rare_name),
        dict(rare_address),
        name_df,
        address_df,
    )


def _empty_support() -> Dict[str, object]:
    support: Dict[str, object] = {column: False for column in PROVENANCE_COLUMNS}
    support["rare_name_token_hits"] = 0
    support["rare_address_token_hits"] = 0
    support["name_tfidf_similarity"] = np.float32(0.0)
    support["name_tfidf_rank"] = 0
    return support


def generate_candidates(
    source1: pd.DataFrame,
    feed: pd.DataFrame,
    config: PipelineConfig,
    *,
    max_k: Optional[int] = None,
    name_tfidf_candidates: Optional[pd.DataFrame] = None,
    name_tfidf_profile: Optional[Mapping[str, object]] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Generate a bounded candidate union without forming a Cartesian product."""

    started = time.perf_counter()
    _validate_inputs(source1, feed)
    if max_k is None:
        max_k = max(config.evaluation_ks)
    if max_k <= 0:
        raise ValueError("max_k must be positive")

    tfidf_merge_seconds = 0.0
    tfidf_by_s1: DefaultDict[str, List[Tuple[str, float, int]]] = defaultdict(list)
    effective_tfidf_profile: Mapping[str, object] = name_tfidf_profile or {}
    if config.use_name_tfidf:
        if name_tfidf_candidates is None:
            name_tfidf_candidates, generated_profile = retrieve_name_tfidf(
                source1, feed, config
            )
            effective_tfidf_profile = generated_profile
        merge_started = time.perf_counter()
        for row in name_tfidf_candidates.itertuples(index=False):
            tfidf_by_s1[row.source1_entity_id].append(
                (
                    row.candidate_entity_id,
                    float(row.name_tfidf_similarity),
                    int(row.name_tfidf_rank),
                )
            )
        tfidf_merge_seconds += time.perf_counter() - merge_started

    # Stable positional index makes posting lists compact and merge-free.
    feed = feed.reset_index(drop=True)
    (
        exact_full,
        exact_core,
        exact_sorted,
        exact_address,
        global_exact_full,
        rare_name,
        rare_address,
        name_df,
        address_df,
    ) = _build_indexes(feed, config)
    feed_position = (
        {entity_id: idx for idx, entity_id in enumerate(feed["entity_id"].tolist())}
        if config.use_name_tfidf
        else {}
    )

    rows: List[Dict[str, object]] = []
    raw_counts: Dict[str, int] = {}

    for s1 in source1.itertuples(index=False):
        country = s1.country_norm
        candidate_support: Dict[int, Dict[str, object]] = {}

        def add(
            indices: Iterable[int], flag: str, hit_column: Optional[str] = None
        ) -> None:
            for idx in indices:
                support = candidate_support.setdefault(idx, _empty_support())
                support[flag] = True
                if hit_column:
                    support[hit_column] = int(support[hit_column]) + 1

        add(exact_full.get((country, s1.name_full), ()), "block_exact_full_name")
        add(exact_core.get((country, s1.name_core), ()), "block_exact_core_name")
        add(
            exact_sorted.get((country, s1.name_sorted), ()),
            "block_exact_sorted_name",
        )
        add(
            exact_address.get((country, s1.address_full), ()) if s1.address_full else (),
            "block_exact_address",
        )

        for token in informative_tokens(
            s1.name_tokens, minimum_length=config.minimum_token_length
        ):
            add(
                rare_name.get((country, token), ()),
                "block_rare_name_token",
                "rare_name_token_hits",
            )
        for token in informative_tokens(
            s1.address_tokens, minimum_length=config.minimum_token_length
        ):
            add(
                rare_address.get((country, token), ()),
                "block_rare_address_token",
                "rare_address_token_hits",
            )

        # Open-set-safe escape hatch: exact normalized names are allowed across
        # countries, but only cross-country records receive this provenance flag.
        if s1.name_full:
            for idx in global_exact_full.get(s1.name_full, ()):
                if feed.at[idx, "country_norm"] != country:
                    add((idx,), "block_cross_country_exact_name")

        if config.use_name_tfidf:
            merge_started = time.perf_counter()
            for target_id, similarity, rank in tfidf_by_s1.get(s1.entity_id, ()):
                idx = feed_position.get(target_id)
                if idx is None:
                    continue
                support = candidate_support.setdefault(idx, _empty_support())
                support["block_name_tfidf"] = True
                support["name_tfidf_similarity"] = max(
                    float(support["name_tfidf_similarity"]), similarity
                )
                current_rank = int(support["name_tfidf_rank"])
                support["name_tfidf_rank"] = rank if current_rank == 0 else min(current_rank, rank)
            tfidf_merge_seconds += time.perf_counter() - merge_started

        raw_counts[s1.entity_id] = len(candidate_support)
        for idx, support in candidate_support.items():
            candidate = feed.iloc[idx]
            score = (
                config.weight_exact_full_name * int(support["block_exact_full_name"])
                + config.weight_exact_core_name * int(support["block_exact_core_name"])
                + config.weight_exact_sorted_name
                * int(support["block_exact_sorted_name"])
                + config.weight_rare_name_token
                * min(int(support["rare_name_token_hits"]), 3)
                + config.weight_exact_address * int(support["block_exact_address"])
                + config.weight_rare_address_token
                * min(int(support["rare_address_token_hits"]), 3)
                + config.weight_same_country
                * int(candidate["country_norm"] == country)
                + config.weight_name_tfidf
                * float(support["name_tfidf_similarity"])
            )
            row = {
                "source1_entity_id": s1.entity_id,
                "candidate_entity_id": candidate["entity_id"],
                "candidate_source": candidate["entity_id"][:2],
                "same_country_block": candidate["country_norm"] == country,
                "cheap_score": np.float32(score),
                **support,
            }
            rows.append(row)

    if not rows:
        columns = [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "same_country_block",
            "cheap_score",
            *PROVENANCE_COLUMNS,
            "rare_name_token_hits",
            "rare_address_token_hits",
            "name_tfidf_similarity",
            "name_tfidf_rank",
            "candidate_count_raw",
            "candidate_count_for_s1",
            "cheap_rank",
            "cheap_score_margin_to_next",
        ]
        return pd.DataFrame(columns=columns), {
            "runtime_seconds": time.perf_counter() - started,
            "pairs_before_cap": 0,
            "pairs_after_cap": 0,
            "max_k": max_k,
        }

    candidates = pd.DataFrame(rows)
    if candidates.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise AssertionError("Candidate union failed to deduplicate a pair")
    candidates["candidate_count_raw"] = candidates["source1_entity_id"].map(raw_counts)
    candidates = candidates.sort_values(
        ["source1_entity_id", "cheap_score", "candidate_entity_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    candidates["cheap_rank"] = (
        candidates.groupby("source1_entity_id", sort=False).cumcount() + 1
    ).astype("uint16")
    before_cap = len(candidates)
    candidates = candidates.loc[candidates["cheap_rank"] <= max_k].copy()
    final_counts = candidates.groupby("source1_entity_id").size().astype("uint16")
    candidates["candidate_count_for_s1"] = (
        candidates["source1_entity_id"].map(final_counts).astype("uint16")
    )
    next_score = candidates.groupby("source1_entity_id", sort=False)["cheap_score"].shift(-1)
    candidates["cheap_score_margin_to_next"] = (
        candidates["cheap_score"] - next_score.fillna(0.0)
    ).astype("float32")

    bool_columns = [*PROVENANCE_COLUMNS, "same_country_block"]
    candidates[bool_columns] = candidates[bool_columns].astype("uint8")
    candidates["rare_name_token_hits"] = candidates["rare_name_token_hits"].astype(
        "uint8"
    )
    candidates["rare_address_token_hits"] = candidates[
        "rare_address_token_hits"
    ].astype("uint8")
    candidates["name_tfidf_similarity"] = candidates[
        "name_tfidf_similarity"
    ].astype("float32")
    candidates["name_tfidf_rank"] = candidates["name_tfidf_rank"].astype("uint16")
    candidates.reset_index(drop=True, inplace=True)

    metadata = {
        "runtime_seconds": time.perf_counter() - started,
        "pairs_before_cap": before_cap,
        "pairs_after_cap": len(candidates),
        "max_k": max_k,
        "source1_entities": len(source1),
        "feed_entities": len(feed),
        "name_token_vocabulary": len(name_df),
        "address_token_vocabulary": len(address_df),
        "maximum_raw_candidates": max(raw_counts.values(), default=0),
        "tfidf_candidate_merge_seconds": tfidf_merge_seconds,
        "tfidf_profile": dict(effective_tfidf_profile),
    }
    return candidates, metadata


def candidates_at_k(candidates: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 0:
        raise ValueError("k must be positive")
    result = candidates.loc[candidates["cheap_rank"] <= k].copy()
    counts = result.groupby("source1_entity_id").size().astype("uint16")
    result["candidate_count_for_s1"] = (
        result["source1_entity_id"].map(counts).astype("uint16")
    )
    return result.reset_index(drop=True)
