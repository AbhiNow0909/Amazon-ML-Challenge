import numpy as np
import pandas as pd
import pytest

from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.features import add_candidate_relative_features
from business_entity_resolution.train_lgbm import (
    decode_probabilities,
    fit_lgbm_classifier,
    predict_probabilities,
    split_pair_features,
    validate_model_feature_columns,
)


def test_entity_level_pair_split_has_no_overlap():
    frame = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-2"],
            "candidate_entity_id": ["S2-1", "S3-1", "S2-2", "S3-2"],
        }
    )
    training, validation = split_pair_features(frame, {"S1-1"}, {"S1-2"})
    assert set(training.source1_entity_id) == {"S1-1"}
    assert set(validation.source1_entity_id) == {"S1-2"}
    assert set(training.source1_entity_id).isdisjoint(validation.source1_entity_id)


def test_model_feature_validation_rejects_leakage_and_schema_mismatch():
    frame = pd.DataFrame({"safe": [0.1], "label": [1]})
    assert validate_model_feature_columns(frame, ["safe"]) == ["safe"]
    with pytest.raises(ValueError, match="Forbidden"):
        validate_model_feature_columns(frame, ["safe", "label"])
    with pytest.raises(ValueError, match="lacks"):
        validate_model_feature_columns(frame, ["missing"])


def test_relative_features_are_deterministic_and_candidate_local():
    frame = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-1", "S1-1"],
            "candidate_entity_id": ["S2-B", "S2-A", "S3-C"],
            "name_character_similarity": [0.9, 0.9, 0.4],
            "address_token_jaccard": [0.1, 0.8, 0.2],
            "name_tfidf_similarity": [0.7, 0.5, 0.1],
            "fusion_score": [0.2, 0.4, 0.1],
        }
    )
    first = add_candidate_relative_features(frame)
    second = add_candidate_relative_features(frame.sample(frac=1, random_state=4)).set_index(
        "candidate_entity_id"
    )
    first = first.set_index("candidate_entity_id")
    for candidate in first.index:
        assert first.loc[candidate, "relative_name_similarity_rank"] == second.loc[
            candidate, "relative_name_similarity_rank"
        ]
    assert first.loc["S2-A", "relative_name_similarity_rank"] == 1
    assert first.loc["S2-A", "relative_address_similarity_rank"] == 1
    assert first.loc["S2-A", "fusion_score_gap_to_best"] == pytest.approx(0.0)
    assert first.loc["S2-A", "close_name_competitor_count"] == 2


def test_threshold_decoding_supports_zero_and_multiple_matches():
    frame = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S3-1", "S2-2"],
        }
    )
    predictions = decode_probabilities(frame, [0.9, 0.8, 0.1], 0.5)
    assert predictions["S1-1"] == {"S2-1", "S3-1"}
    assert "S1-2" not in predictions


def _tiny_training_frames():
    rng = np.random.RandomState(7)
    rows = 120
    x1 = rng.normal(size=rows).astype("float32")
    x2 = rng.normal(size=rows).astype("float32")
    label = (x1 + 0.3 * x2 > 0).astype("uint8")
    frame = pd.DataFrame(
        {
            "source1_entity_id": [f"S1-{i // 4}" for i in range(rows)],
            "candidate_entity_id": [f"S2-{i}" for i in range(rows)],
            "x1": x1,
            "x2": x2,
            "label": label,
        }
    )
    return frame.iloc[:80].copy(), frame.iloc[80:].copy()


def test_probability_shape_range_feature_order_and_determinism():
    training, validation = _tiny_training_frames()
    config = PipelineConfig(
        lgbm_num_leaves=7,
        lgbm_min_data_in_leaf=2,
        lgbm_max_rounds=40,
        lgbm_early_stopping_rounds=5,
    )
    first, _ = fit_lgbm_classifier(
        training, validation, feature_columns=["x1", "x2"], config=config
    )
    second, _ = fit_lgbm_classifier(
        training, validation, feature_columns=["x1", "x2"], config=config
    )
    first_probabilities = predict_probabilities(first, validation, ["x1", "x2"])
    second_probabilities = predict_probabilities(second, validation, ["x1", "x2"])
    assert first_probabilities.shape == (len(validation),)
    assert np.all((first_probabilities >= 0) & (first_probabilities <= 1))
    assert np.allclose(first_probabilities, second_probabilities)
    with pytest.raises(ValueError, match="lacks"):
        predict_probabilities(first, validation.drop(columns="x2"), ["x1", "x2"])
