import pandas as pd

from business_entity_resolution.blocking import PROVENANCE_COLUMNS
from business_entity_resolution.candidate_fusion import (
    hybrid_quota_rank_fusion,
    normalized_score_fusion,
    prepare_fusion_universe,
    quota_fusion,
    reciprocal_rank_fusion,
)


def _legacy_rows(count=5):
    rows = []
    for rank in range(1, count + 1):
        row = {
            "source1_entity_id": "S1-1",
            "candidate_entity_id": f"S2-L{rank}",
            "candidate_source": "S2",
            "same_country_block": 1,
            "cheap_score": float(20 - rank),
            "cheap_rank": rank,
            "cheap_score_margin_to_next": 1.0,
            "rare_name_token_hits": 1,
            "rare_address_token_hits": 0,
        }
        row.update({column: 0 for column in PROVENANCE_COLUMNS})
        row["block_rare_name_token"] = 1
        rows.append(row)
    return pd.DataFrame(rows)


def _tfidf_rows(ids):
    return pd.DataFrame(
        {
            "source1_entity_id": ["S1-1"] * len(ids),
            "candidate_entity_id": ids,
            "name_tfidf_similarity": [1.0 - i * 0.05 for i in range(len(ids))],
            "name_tfidf_rank": list(range(1, len(ids) + 1)),
        }
    )


def test_prepare_universe_deduplicates_and_preserves_provenance():
    universe = prepare_fusion_universe(
        _legacy_rows(3), _tfidf_rows(["S2-L1", "S3-T1", "S3-T2"]), tfidf_top_k=3
    )

    assert len(universe) == 5
    assert not universe.duplicated(["source1_entity_id", "candidate_entity_id"]).any()
    shared = universe.set_index("candidate_entity_id").loc["S2-L1"]
    assert shared["supported_by_both"] == 1
    assert shared["legacy_rank"] == 1
    assert shared["name_tfidf_rank"] == 1
    assert universe.set_index("candidate_entity_id").loc["S3-T1", "is_tfidf_only"] == 1


def test_quota_preserves_legacy_and_uses_only_new_tfidf_additions():
    universe = prepare_fusion_universe(
        _legacy_rows(5),
        _tfidf_rows(["S2-L1", "S3-T1", "S3-T2", "S3-T3"]),
        tfidf_top_k=4,
    )
    fused = quota_fusion(
        universe, final_k=5, legacy_quota=3, tfidf_addition_quota=2
    )

    assert set(fused.candidate_entity_id) == {
        "S2-L1",
        "S2-L2",
        "S2-L3",
        "S3-T1",
        "S3-T2",
    }
    assert len(fused) == 5


def test_quota_returns_unused_slots_to_legacy_candidates():
    universe = prepare_fusion_universe(
        _legacy_rows(5), _tfidf_rows(["S2-L1"]), tfidf_top_k=1
    )
    fused = quota_fusion(
        universe, final_k=5, legacy_quota=2, tfidf_addition_quota=3
    )

    assert set(fused.candidate_entity_id) == set(_legacy_rows(5).candidate_entity_id)
    assert fused.candidate_count_for_s1.max() == 5


def test_rrf_is_deterministic_bounded_and_rewards_shared_support():
    universe = prepare_fusion_universe(
        _legacy_rows(4), _tfidf_rows(["S3-T1", "S2-L3", "S3-T2"]), tfidf_top_k=3
    )
    first = reciprocal_rank_fusion(
        universe, final_k=3, legacy_weight=2.0, tfidf_weight=1.0, rrf_constant=20
    )
    second = reciprocal_rank_fusion(
        universe.sample(frac=1.0, random_state=7),
        final_k=3,
        legacy_weight=2.0,
        tfidf_weight=1.0,
        rrf_constant=20,
    )

    assert first.candidate_entity_id.tolist() == second.candidate_entity_id.tolist()
    assert "S2-L3" in set(first.candidate_entity_id)
    assert len(first) == 3
    assert first.fusion_rank.max() == 3


def test_normalized_scores_are_group_local_and_bounded():
    universe = prepare_fusion_universe(
        _legacy_rows(3), _tfidf_rows(["S3-T1", "S2-L2", "S3-T2"]), tfidf_top_k=3
    )
    fused = normalized_score_fusion(
        universe,
        final_k=4,
        legacy_alpha=0.7,
        tfidf_beta=0.3,
        provenance_bonus=0.1,
    )

    assert fused["normalized_legacy_score"].between(0, 1).all()
    assert fused["normalized_tfidf_score"].between(0, 1).all()
    assert len(fused) == 4


def test_hybrid_protects_true_legacy_candidate_from_tfidf_displacement():
    legacy = _legacy_rows(5)
    # Give TF-IDF-only rows perfect scores. The minimum legacy quota must still
    # retain the first four legacy candidates, including the synthetic truth.
    tfidf = _tfidf_rows([f"S3-T{i}" for i in range(1, 11)])
    universe = prepare_fusion_universe(legacy, tfidf, tfidf_top_k=10)
    fused = hybrid_quota_rank_fusion(
        universe,
        final_k=5,
        minimum_legacy=4,
        legacy_weight=1.0,
        tfidf_weight=10.0,
        rrf_constant=20,
    )

    assert "S2-L4" in set(fused.candidate_entity_id)
    assert int(fused.has_legacy.sum()) >= 4
    assert len(fused) == 5
    assert {
        "legacy_rank",
        "legacy_cheap_score",
        "name_tfidf_rank",
        "name_tfidf_similarity",
        "is_tfidf_only",
        "is_legacy_only",
        "supported_by_both",
    }.issubset(fused.columns)
