import pytest

from business_entity_resolution.evaluation import entity_metrics, evaluate_predictions


def test_empty_truth_empty_prediction_scores_one():
    assert entity_metrics(set(), set()) == {
        "precision": 1.0,
        "recall": 1.0,
        "f0_5": 1.0,
    }


def test_empty_truth_nonempty_prediction_scores_zero():
    assert entity_metrics(set(), {"S2-1"})["f0_5"] == 0.0


def test_perfect_match_scores_one():
    assert entity_metrics({"S2-1", "S3-1"}, {"S2-1", "S3-1"})["f0_5"] == 1.0


def test_partial_multi_match_uses_f05_formula():
    metrics = entity_metrics({"S2-1", "S3-1"}, {"S2-1"})
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f0_5"] == pytest.approx(5 / 6)


def test_macro_metric_preserves_entity_boundaries():
    result = evaluate_predictions(
        ["S1-1", "S1-2"],
        {"S1-1": {"S2-1"}, "S1-2": set()},
        {"S1-1": {"S2-1"}, "S1-2": set()},
    )
    assert result["macro_f0_5"] == 1.0
    assert result["pair_micro_precision"] == 1.0
    assert result["pair_micro_recall"] == 1.0

