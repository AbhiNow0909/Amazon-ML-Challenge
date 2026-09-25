"""Small deterministic E0 matcher operating on the ML-ready feature table."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .evaluation import evaluate_predictions, pairs_to_mapping


def score_rule_baseline(features: pd.DataFrame) -> pd.DataFrame:
    """Attach a precision-oriented rule score without changing feature semantics."""

    result = features.copy()
    score = (
        0.50 * result["name_character_similarity"].to_numpy(dtype="float32")
        + 0.25 * result["name_token_jaccard"].to_numpy(dtype="float32")
        + 0.20 * result["address_token_jaccard"].to_numpy(dtype="float32")
        + 0.05 * result["name_length_ratio"].to_numpy(dtype="float32")
    )

    exact_name = (
        result["name_full_exact"].astype(bool)
        | result["name_core_exact"].astype(bool)
        | result["name_sorted_exact"].astype(bool)
    ).to_numpy()
    same_country = result["same_country"].astype(bool).to_numpy()
    address_support = (
        (result["address_token_jaccard"].to_numpy() >= 0.25)
        | (result["numeric_token_overlap"].to_numpy() > 0)
        | result["address_exact"].astype(bool).to_numpy()
    )
    strong_address = (
        (result["address_token_jaccard"].to_numpy() >= 0.60)
        | result["address_exact"].astype(bool).to_numpy()
    )
    no_numeric_conflict = ~result["numeric_token_conflict"].astype(bool).to_numpy()

    score = np.maximum(
        score,
        np.where(result["name_sorted_exact"].astype(bool), 0.85, 0.0),
    )
    score = np.maximum(
        score,
        np.where(result["name_core_exact"].astype(bool), 0.88, 0.0),
    )
    score = np.maximum(
        score,
        np.where(result["name_full_exact"].astype(bool), 0.90, 0.0),
    )
    score = np.maximum(score, np.where(exact_name & same_country, 0.93, 0.0))
    score = np.maximum(
        score, np.where(exact_name & same_country & address_support, 0.97, 0.0)
    )
    score = np.maximum(
        score,
        np.where(
            (result["name_character_similarity"].to_numpy() >= 0.92)
            & strong_address
            & same_country
            & no_numeric_conflict,
            0.95,
            0.0,
        ),
    )
    score = np.maximum(
        score,
        np.where(
            result["address_exact"].astype(bool).to_numpy()
            & (result["name_token_jaccard"].to_numpy() >= 0.50)
            & same_country,
            0.94,
            0.0,
        ),
    )

    score -= np.where(result["country_conflict"].astype(bool), 0.35, 0.0)
    score -= np.where(result["numeric_token_conflict"].astype(bool), 0.12, 0.0)
    result["rule_score"] = np.clip(score, 0.0, 1.0).astype("float32")
    return result


def predict_from_scores(
    scored: pd.DataFrame,
    threshold: float,
) -> Dict[str, Set[str]]:
    selected = scored.loc[scored["rule_score"] >= threshold]
    return pairs_to_mapping(selected)


def tune_rule_threshold(
    scored: pd.DataFrame,
    source1_ids: Sequence[str],
    truth: Mapping[str, Set[str]],
    thresholds: Iterable[float],
) -> Tuple[float, Dict[str, object], pd.DataFrame]:
    """Tune only against the supplied validation truth; ties prefer precision."""

    rows = []
    best_threshold = None
    best_metrics = None
    for threshold in thresholds:
        predictions = predict_from_scores(scored, float(threshold))
        metrics = evaluate_predictions(source1_ids, truth, predictions)
        rows.append({"threshold": float(threshold), **metrics})
        key = (
            metrics["macro_f0_5"],
            metrics["pair_micro_precision"],
            float(threshold),
        )
        if best_metrics is None or key > (
            best_metrics["macro_f0_5"],
            best_metrics["pair_micro_precision"],
            float(best_threshold),
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_threshold is None or best_metrics is None:
        raise ValueError("At least one rule threshold is required")
    return best_threshold, best_metrics, pd.DataFrame(rows)

