from dataclasses import replace

import pandas as pd

from business_entity_resolution.address_preblocked_retrieval import (
    AddressPreblockStrategy,
    prepare_address_tfidf_index,
    retrieve_from_prepared_address_index,
)
from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.multi_channel_fusion import (
    prepare_multi_channel_universe,
    protected_legacy_multi_channel_rrf,
)
from business_entity_resolution.normalize import normalize_records


def _frame(rows):
    return normalize_records(
        pd.DataFrame(
            rows,
            columns=["entity_id", "business_name", "business_address", "country"],
        )
    )


def _prepared(feed):
    config = replace(
        PipelineConfig(),
        tfidf_address_max_features=None,
        tfidf_address_top_k=3,
    )
    return prepare_address_tfidf_index(
        feed,
        config,
        maximum_numeric_df=20,
        maximum_rare_token_df=20,
        maximum_postcode_prefix_df=20,
    )


def test_postcode_numeric_and_rare_token_scopes_recover_candidates():
    source1 = _frame(
        [
            ["S1-1", "A", "12 West Pine Street 02139", "US"],
            ["S1-2", "B", "77 Market Lane", "US"],
            ["S1-3", "C", "Cedar Arcade", "US"],
        ]
    )
    feed = _frame(
        [
            ["S2-1", "X", "14 West Pine St 02139", "US"],
            ["S2-2", "Y", "77 Market Ln", "US"],
            ["S3-3", "Z", "Cedar Arcad", "US"],
            ["S3-4", "Q", "900 Completely Different Boulevard", "US"],
        ]
    )
    strategy = AddressPreblockStrategy(
        name="routes", numeric_max_df=20, rare_token_max_df=20, pool_cap=20
    )
    result, profile = retrieve_from_prepared_address_index(
        source1, _prepared(feed), strategy, top_k=3
    )
    indexed = result.set_index(["source1_entity_id", "candidate_entity_id"])
    assert indexed.loc[("S1-1", "S2-1"), "address_route_postcode"] == 1
    assert indexed.loc[("S1-2", "S2-2"), "address_route_numeric"] == 1
    assert indexed.loc[("S1-3", "S3-3"), "address_route_rare_token"] == 1
    assert profile["dense_similarity_constructed"] is False
    assert profile["candidate_comparisons"] < len(source1) * len(feed)


def test_country_fallback_is_explicit_when_no_structural_key_qualifies():
    source1 = _frame([["S1-1", "A", "Main Road", "US"]])
    feed = _frame(
        [
            ["S2-1", "X", "Main Road East", "US"],
            ["S2-2", "Y", "Main Road West", "US"],
        ]
    )
    strategy = AddressPreblockStrategy(
        name="fallback",
        numeric_max_df=1,
        rare_token_max_df=1,
        rare_tokens_per_query=2,
        pool_cap=10,
    )
    result, profile = retrieve_from_prepared_address_index(
        source1, _prepared(feed), strategy, top_k=1
    )
    assert len(result) == 1
    assert result.iloc[0].address_route_country_fallback == 1
    assert profile["fallback_queries"] == 1
    assert profile["candidate_comparisons"] == len(feed)


def test_retrieval_is_deterministic_bounded_and_deduplicated():
    source1 = _frame([["S1-1", "A", "42 Oak Street 10001", "US"]])
    feed = _frame(
        [[f"S2-{i}", "X", f"42 Oak Street Unit {i} 10001", "US"] for i in range(8)]
    )
    prepared = _prepared(feed)
    strategy = AddressPreblockStrategy(
        name="bounded", numeric_max_df=20, rare_token_max_df=20, pool_cap=5
    )
    first, _ = retrieve_from_prepared_address_index(
        source1, prepared, strategy, top_k=2
    )
    second, _ = retrieve_from_prepared_address_index(
        source1, prepared, strategy, top_k=2
    )
    pd.testing.assert_frame_equal(first, second)
    assert len(first) <= 2
    assert not first.duplicated(["source1_entity_id", "candidate_entity_id"]).any()
    assert first.address_preblock_pool_size.min() >= 5


def test_multi_channel_union_preserves_route_provenance_and_final_k():
    legacy = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-1"],
            "candidate_source": ["S2"],
            "cheap_score": [5.0],
            "cheap_rank": [1],
            "same_country_block": [1],
            "cheap_score_margin_to_next": [5.0],
            "rare_name_token_hits": [0],
            "rare_address_token_hits": [0],
            "block_exact_full_name": [1],
            "block_exact_core_name": [1],
            "block_exact_sorted_name": [1],
            "block_rare_name_token": [0],
            "block_exact_address": [0],
            "block_rare_address_token": [0],
            "block_cross_country_exact_name": [0],
        }
    )
    # prepare_fusion_universe accepts missing blocker flags and fills them.
    name = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-2"],
            "name_tfidf_similarity": [0.8],
            "name_tfidf_rank": [1],
        }
    )
    address = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-1"],
            "candidate_entity_id": ["S2-1", "S3-3"],
            "address_tfidf_similarity": [0.9, 0.7],
            "address_tfidf_rank": [1, 2],
            "address_route_postcode": [1, 0],
            "address_route_numeric": [1, 0],
            "address_route_rare_token": [0, 1],
            "address_route_country_fallback": [0, 0],
            "address_route_count": [2, 1],
            "address_preblock_pool_size": [3, 3],
        }
    )
    universe = prepare_multi_channel_universe(
        legacy, name, address, name_top_k=3, address_top_k=3
    )
    result = protected_legacy_multi_channel_rrf(
        universe,
        final_k=2,
        minimum_legacy=1,
        legacy_weight=2.0,
        name_weight=1.0,
        address_weight=1.0,
        rrf_constant=20.0,
    )
    assert not universe.duplicated(["source1_entity_id", "candidate_entity_id"]).any()
    assert len(result) == 2
    assert "address_route_postcode" in result
    assert universe.set_index("candidate_entity_id").loc[
        "S2-1", "address_route_postcode"
    ] == 1
