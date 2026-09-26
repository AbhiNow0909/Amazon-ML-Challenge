from dataclasses import replace

import pandas as pd

from business_entity_resolution.blocking import generate_candidates
from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.normalize import normalize_records
from business_entity_resolution.tfidf_retrieval import (
    retrieve_address_tfidf,
    retrieve_name_tfidf,
)


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=["entity_id", "business_name", "business_address", "country"],
    )


def _config(top_k=3):
    return replace(
        PipelineConfig(),
        use_name_tfidf=True,
        tfidf_name_top_k=top_k,
        tfidf_query_chunk_size=1,
        tfidf_max_features=None,
        tfidf_min_df=1,
        posting_list_cap=20,
    )


def test_sparse_tfidf_recovers_typo_and_abbreviation_overlap():
    source1 = normalize_records(
        _frame(
            [
                ["S1-1", "Acme Pharmaceuticals", "1 Main St", "US"],
                ["S1-2", "International Business Machines", "2 Main St", "US"],
            ]
        )
    )
    feed = normalize_records(
        _frame(
            [
                ["S2-1", "Acme Farmaceuticals", "1 Main St", "US"],
                ["S2-2", "Intl Business Machines", "2 Main St", "US"],
                ["S2-3", "Completely Different", "3 Main St", "US"],
            ]
        )
    )
    result, profile = retrieve_name_tfidf(source1, feed, _config(top_k=2))
    pairs = set(zip(result.source1_entity_id, result.candidate_entity_id))
    assert ("S1-1", "S2-1") in pairs
    assert ("S1-2", "S2-2") in pairs
    assert profile["dense_similarity_constructed"] is False
    assert profile["total_index_nnz"] > 0
    assert profile["maximum_similarity_chunk_nnz"] <= len(feed)


def test_tfidf_is_bounded_country_sharded_and_deterministic():
    source1 = normalize_records(
        _frame(
            [
                ["S1-1", "Maison Bleue", "", "France"],
                ["S1-2", "Alpha Services", "", "US"],
                ["S1-3", "Sharma Trading", "", "India"],
            ]
        )
    )
    feed_rows = [
        [f"S2-{i}", f"Alpha Service Number {i}", "", "US"] for i in range(10)
    ]
    feed_rows += [
        ["S3-20", "Maison-Bleue!", "", "France"],
        ["S3-21", "Sharma Traders", "", "India"],
        ["S3-22", "Maison Bleue", "", "Canada"],
    ]
    feed = normalize_records(_frame(feed_rows))
    first, _ = retrieve_name_tfidf(source1, feed, _config(top_k=2))
    second, _ = retrieve_name_tfidf(source1, feed, _config(top_k=2))
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    assert len(first) <= len(source1) * 2
    assert len(first) < len(source1) * len(feed)
    # TF-IDF is country-sharded; cross-country exact fallback belongs to the
    # existing exact blocker rather than this channel.
    assert not (
        (first.source1_entity_id == "S1-1")
        & (first.candidate_entity_id == "S3-22")
    ).any()


def test_blocking_integration_deduplicates_and_records_tfidf_provenance():
    source1 = normalize_records(
        _frame([["S1-1", "Acme, Incorporated", "12 Main Road", "US"]])
    )
    feed = normalize_records(
        _frame(
            [
                ["S2-1", "Acme Incorporated", "12 Main Road", "US"],
                ["S2-2", "Acne Incorporatd", "14 Main Road", "US"],
            ]
        )
    )
    config = _config(top_k=2)
    tfidf, profile = retrieve_name_tfidf(source1, feed, config)
    candidates, _ = generate_candidates(
        source1,
        feed,
        config,
        max_k=5,
        name_tfidf_candidates=tfidf,
        name_tfidf_profile=profile,
    )
    assert candidates.duplicated(["source1_entity_id", "candidate_entity_id"]).sum() == 0
    exact = candidates.set_index("candidate_entity_id").loc["S2-1"]
    assert exact["block_exact_full_name"] == 1
    assert exact["block_name_tfidf"] == 1
    assert exact["name_tfidf_similarity"] > 0
    assert exact["name_tfidf_rank"] >= 1


def test_address_tfidf_is_sparse_bounded_and_recovers_format_noise():
    source1 = normalize_records(
        _frame([["S1-1", "Unrelated Name", "13818 184th Street, Arlington, WA", "US"]])
    )
    feed = normalize_records(
        _frame(
            [
                ["S2-1", "Trade Name", "13816, 184st St Arington Washington", "US"],
                ["S2-2", "Other", "99 South Road, Miami, FL", "US"],
                ["S2-3", "Other Two", "88 North Road, Boston, MA", "US"],
            ]
        )
    )
    config = replace(
        _config(top_k=2),
        tfidf_address_top_k=1,
        tfidf_address_max_features=None,
    )
    result, profile = retrieve_address_tfidf(source1, feed, config)
    assert result.candidate_entity_id.tolist() == ["S2-1"]
    assert result.address_tfidf_rank.max() == 1
    assert len(result) <= len(source1)
    assert profile["dense_similarity_constructed"] is False
    assert profile["total_index_nnz"] > 0
