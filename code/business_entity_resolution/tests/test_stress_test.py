import pandas as pd
import pytest

from business_entity_resolution.stress_test import (
    FROZEN_CANDIDATE_CONFIG,
    _deterministic_hash,
    feed_at_density,
)


def test_deterministic_hash_is_repeatable_and_seeded():
    values = pd.Series(["S2-1", "S2-2", "S2-3"])
    first = _deterministic_hash(values, 17, "source2")
    second = _deterministic_hash(values, 17, "source2")
    different = _deterministic_hash(values, 18, "source2")
    assert first.equals(second)
    assert not first.equals(different)


def test_density_subsets_are_nested_and_keep_all_truth_targets():
    rows = []
    for source in ("source2", "source3"):
        rows.append(
            {
                "entity_id": f"{source}-truth",
                "_feed_source": source,
                "_is_distractor": 0,
                "_distractor_rank": 0,
            }
        )
        for rank in range(1, 6):
            rows.append(
                {
                    "entity_id": f"{source}-{rank}",
                    "_feed_source": source,
                    "_is_distractor": 1,
                    "_distractor_rank": rank,
                }
            )
    feed = pd.DataFrame(rows)
    small = feed_at_density(feed, 4)
    large = feed_at_density(feed, 8)
    assert set(small.entity_id).issubset(set(large.entity_id))
    assert int(small._is_distractor.sum()) == 4
    assert int(large._is_distractor.sum()) == 8
    assert {"source2-truth", "source3-truth"}.issubset(set(small.entity_id))


def test_density_rejects_unbalanced_total():
    feed = pd.DataFrame(
        {
            "entity_id": ["S2-1"],
            "_is_distractor": [0],
            "_distractor_rank": [0],
        }
    )
    with pytest.raises(ValueError, match="positive even"):
        feed_at_density(feed, 3)


def test_e21_frozen_candidate_configuration():
    assert FROZEN_CANDIDATE_CONFIG == {
        "fusion": "rrf",
        "legacy_weight": 2.0,
        "tfidf_weight": 1.0,
        "rrf_constant": 20.0,
        "tfidf_top_k": 30,
        "final_k": 50,
    }
