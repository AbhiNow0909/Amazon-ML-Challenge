import pandas as pd

from business_entity_resolution.blocking import (
    generate_candidates,
    token_df_eligible,
)
from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.features import build_feature_table
from business_entity_resolution.normalize import normalize_records


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=["entity_id", "business_name", "business_address", "country"],
    )


def test_exact_normalized_name_recovers_pair_and_deduplicates_blockers():
    source1 = normalize_records(
        _frame([["S1-1", "Acme Private Limited", "12 Main Road", "India"]])
    )
    feed = normalize_records(
        _frame(
            [
                ["S2-1", "ACME Pvt Ltd", "12 Main Rd", "India"],
                ["S3-1", "Acme Private Limited", "12 Main Road", "India"],
                ["S2-2", "Other Business", "99 Elsewhere", "India"],
            ]
        )
    )
    config = PipelineConfig(
        rare_name_token_max_df=10,
        rare_address_token_max_df=10,
        posting_list_cap=10,
    )
    candidates, _ = generate_candidates(source1, feed, config, max_k=10)
    assert "S3-1" in set(candidates["candidate_entity_id"])
    assert candidates.duplicated(["source1_entity_id", "candidate_entity_id"]).sum() == 0
    recovered = candidates.set_index("candidate_entity_id").loc["S3-1"]
    assert recovered["block_exact_full_name"] == 1
    assert recovered["block_exact_address"] == 1


def test_candidates_are_bounded_and_never_include_s1():
    source1 = normalize_records(_frame([["S1-1", "Common Shop", "1 Road", "US"]]))
    feed = normalize_records(
        _frame(
            [
                [f"S2-{i}", f"Common Shop {i}", f"{i} Road", "US"]
                for i in range(20)
            ]
        )
    )
    config = PipelineConfig(
        rare_name_token_max_df=100,
        rare_address_token_max_df=100,
        posting_list_cap=100,
    )
    candidates, _ = generate_candidates(source1, feed, config, max_k=5)
    assert len(candidates) <= 5
    assert not candidates["candidate_entity_id"].str.startswith("S1-").any()
    assert len(candidates) <= len(source1) * 5


def test_cross_country_exact_name_fallback_is_open_set_safe():
    source1 = normalize_records(_frame([["S1-1", "Maison Bleue", "1 Rue A", "France"]]))
    feed = normalize_records(_frame([["S2-1", "Maison Bleue", "1 Road A", "Canada"]]))
    candidates, _ = generate_candidates(source1, feed, PipelineConfig(), max_k=10)
    assert list(candidates["candidate_entity_id"]) == ["S2-1"]
    assert candidates.loc[0, "block_cross_country_exact_name"] == 1


def test_features_derive_source_flags_from_candidate_ids():
    source1 = normalize_records(_frame([["S1-1", "Acme", "1 Road", "US"]]))
    feed = normalize_records(_frame([["S2-1", "Acme", "1 Road", "US"]]))
    candidates, _ = generate_candidates(source1, feed, PipelineConfig(), max_k=10)
    features = build_feature_table(candidates, source1, feed)
    assert features.loc[0, "source_is_s2"] == 1
    assert features.loc[0, "source_is_s3"] == 0


def test_density_aware_token_eligibility_modes():
    assert token_df_eligible(
        80, 100_000, absolute_max_df=100, relative_max_df=0.0005, mode="absolute"
    )
    assert not token_df_eligible(
        120, 100_000, absolute_max_df=100, relative_max_df=0.0005, mode="relative"
    )
    assert token_df_eligible(
        120, 1_000_000, absolute_max_df=100, relative_max_df=0.0005, mode="relative"
    )
    assert token_df_eligible(
        80, 100_000, absolute_max_df=100, relative_max_df=0.0005, mode="hybrid"
    )


def test_density_scaled_posting_cap_is_opt_in_and_bounded():
    source1 = normalize_records(_frame([["S1-1", "Common Alpha", "", "US"]]))
    feed = normalize_records(
        _frame([[f"S2-{i}", f"Common Item {i}", "", "US"] for i in range(4)])
    )
    baseline = PipelineConfig(
        rare_name_token_max_df=10,
        posting_list_cap=2,
        rare_posting_cap_max=2,
    )
    scaled = PipelineConfig(
        rare_name_token_max_df=10,
        posting_list_cap=2,
        rare_posting_cap_fraction=1.0,
        rare_posting_cap_max=4,
    )
    baseline_candidates, _ = generate_candidates(source1, feed, baseline, max_k=10)
    scaled_candidates, _ = generate_candidates(source1, feed, scaled, max_k=10)
    assert baseline_candidates.empty
    assert len(scaled_candidates) == 4
    assert scaled_candidates["block_rare_name_token"].eq(1).all()
