import pandas as pd

from business_entity_resolution.candidate_fusion import LEGACY_PROVENANCE_COLUMNS
from business_entity_resolution.multi_channel_fusion import (
    multi_channel_rrf,
    prepare_multi_channel_universe,
    protected_legacy_multi_channel_rrf,
)


def _legacy(count=5):
    rows = []
    for rank in range(1, count + 1):
        row = {
            "source1_entity_id": "S1-1",
            "candidate_entity_id": f"S2-L{rank}",
            "candidate_source": "S2",
            "same_country_block": 1,
            "cheap_score": float(10 - rank),
            "cheap_rank": rank,
            "cheap_score_margin_to_next": 1.0,
            "rare_name_token_hits": 1,
            "rare_address_token_hits": 0,
        }
        row.update({column: 0 for column in LEGACY_PROVENANCE_COLUMNS})
        row["block_rare_name_token"] = 1
        rows.append(row)
    return pd.DataFrame(rows)


def _name():
    return pd.DataFrame(
        {
            "source1_entity_id": ["S1-1"] * 3,
            "candidate_entity_id": ["S2-L1", "S3-N1", "S3-N2"],
            "name_tfidf_similarity": [1.0, 0.9, 0.8],
            "name_tfidf_rank": [1, 2, 3],
        }
    )


def _address():
    return pd.DataFrame(
        {
            "source1_entity_id": ["S1-1"] * 3,
            "candidate_entity_id": ["S2-L1", "S3-N1", "S3-A1"],
            "address_tfidf_similarity": [1.0, 0.95, 0.9],
            "address_tfidf_rank": [1, 2, 3],
        }
    )


def test_three_channel_dedup_provenance_determinism_and_final_bound():
    universe = prepare_multi_channel_universe(
        _legacy(), _name(), _address(), name_top_k=3, address_top_k=3
    )
    assert not universe.duplicated(["source1_entity_id", "candidate_entity_id"]).any()
    shared = universe.set_index("candidate_entity_id").loc["S2-L1"]
    assert shared.retrieval_channel_count == 3
    assert shared.has_address_tfidf == 1
    assert universe.set_index("candidate_entity_id").loc[
        "S3-A1", "is_address_tfidf_only"
    ] == 1
    first = multi_channel_rrf(
        universe,
        final_k=4,
        legacy_weight=2,
        name_weight=1,
        address_weight=0.5,
        rrf_constant=20,
    )
    second = multi_channel_rrf(
        universe.sample(frac=1, random_state=3),
        final_k=4,
        legacy_weight=2,
        name_weight=1,
        address_weight=0.5,
        rrf_constant=20,
    )
    assert first.candidate_entity_id.tolist() == second.candidate_entity_id.tolist()
    assert len(first) == 4
    assert first.fusion_rank.max() == 4


def test_protected_fusion_preserves_legacy_quota():
    universe = prepare_multi_channel_universe(
        _legacy(), _name(), _address(), name_top_k=3, address_top_k=3
    )
    fused = protected_legacy_multi_channel_rrf(
        universe,
        final_k=5,
        minimum_legacy=4,
        legacy_weight=2,
        name_weight=1,
        address_weight=1,
        rrf_constant=20,
    )
    assert int(fused.has_legacy.sum()) >= 4
    assert len(fused) == 5


def test_three_channel_fusion_enforces_final_k_50():
    universe = prepare_multi_channel_universe(
        _legacy(60), _name(), _address(), name_top_k=3, address_top_k=3
    )
    fused = multi_channel_rrf(
        universe,
        final_k=50,
        legacy_weight=2,
        name_weight=1,
        address_weight=1,
        rrf_constant=20,
    )
    assert len(fused) == 50
    assert fused.fusion_rank.max() == 50
