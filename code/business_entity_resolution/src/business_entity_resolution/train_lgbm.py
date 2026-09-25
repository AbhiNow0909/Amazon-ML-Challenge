"""E2 LightGBM pair classifier on the existing E1.1 candidate features."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from .candidate_fusion import prepare_fusion_universe, reciprocal_rank_fusion
from .config import DEFAULT_CONFIG, PipelineConfig
from .evaluation import (
    candidate_metrics,
    evaluate_breakdowns,
    evaluate_predictions,
    pairs_to_mapping,
)
from .features import (
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_RELATIVE_FEATURE_COLUMNS,
    add_candidate_relative_features,
    build_feature_table,
)
from .io_utils import (
    ground_truth_to_mapping,
    load_ground_truth,
    validate_ground_truth_frame,
    write_parquet_cache,
)
from .labels import build_pair_labels
from .rules import predict_from_scores, score_rule_baseline
from .split import split_source1_ids


MODEL_FEATURE_COLUMNS = [*FEATURE_COLUMNS, *MODEL_RELATIVE_FEATURE_COLUMNS]
FORBIDDEN_MODEL_COLUMNS = {
    "source1_entity_id",
    "candidate_entity_id",
    "label",
    "rule_score",
    "ground_truth_match_count",
}


def _peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def validate_model_feature_columns(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
) -> List[str]:
    """Return a stable numeric feature list after explicit leakage checks."""

    columns = list(feature_columns)
    if len(columns) != len(set(columns)):
        raise ValueError("Model feature list contains duplicate columns")
    leaked = set(columns) & FORBIDDEN_MODEL_COLUMNS
    if leaked:
        raise ValueError(f"Forbidden model features: {sorted(leaked)}")
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Feature table lacks model columns: {sorted(missing)}")
    nonnumeric = [
        column for column in columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if nonnumeric:
        raise ValueError(f"Model features must be numeric: {nonnumeric}")
    if frame[columns].isna().any().any():
        bad = frame[columns].columns[frame[columns].isna().any()].tolist()
        raise ValueError(f"Model features contain missing values: {bad}")
    return columns


def split_pair_features(
    frame: pd.DataFrame,
    training_ids: Set[str],
    validation_ids: Set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if training_ids & validation_ids:
        raise ValueError("Training and validation S1 sets overlap")
    known = training_ids | validation_ids
    unknown = set(frame["source1_entity_id"]) - known
    if unknown:
        raise ValueError(f"Pairs have unassigned S1 entities: {sorted(unknown)[:5]}")
    training = frame.loc[frame["source1_entity_id"].isin(training_ids)].copy()
    validation = frame.loc[frame["source1_entity_id"].isin(validation_ids)].copy()
    observed_overlap = set(training["source1_entity_id"]) & set(
        validation["source1_entity_id"]
    )
    if observed_overlap:
        raise AssertionError("Pair split leaked Source-1 entities")
    return training.reset_index(drop=True), validation.reset_index(drop=True)


def sample_hard_negatives(
    training: pd.DataFrame,
    *,
    negatives_per_positive: int,
) -> pd.DataFrame:
    """Keep all positives and bounded, evidence-rich negatives per S1."""

    if negatives_per_positive <= 0:
        raise ValueError("negatives_per_positive must be positive")
    pieces: List[pd.DataFrame] = []
    for _, group in training.groupby("source1_entity_id", sort=False):
        positives = group.loc[group["label"] == 1]
        negatives = group.loc[group["label"] == 0].copy()
        negative_limit = max(negatives_per_positive, len(positives) * negatives_per_positive)
        if not negatives.empty:
            negatives["_hardness"] = (
                2.0 / (1.0 + negatives["relative_fusion_score_rank"])
                + 1.0 / (1.0 + negatives["relative_name_similarity_rank"])
                + 0.75 / (1.0 + negatives["relative_tfidf_similarity_rank"])
                + 0.50 / (1.0 + negatives["relative_address_similarity_rank"])
                + 0.20 * negatives["name_character_similarity"]
                + 0.15 * negatives["address_token_jaccard"]
                + 0.10 * negatives["name_tfidf_similarity"]
                + 0.02 * negatives["legacy_cheap_score"]
            )
            negatives = negatives.sort_values(
                ["_hardness", "candidate_entity_id"],
                ascending=[False, True],
                kind="mergesort",
            ).head(negative_limit)
            negatives = negatives.drop(columns="_hardness")
        pieces.extend([positives, negatives])
    sampled = pd.concat(pieces, ignore_index=True)
    return sampled.sort_values(
        ["source1_entity_id", "candidate_entity_id"], kind="mergesort"
    ).reset_index(drop=True)


def fit_lgbm_classifier(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    config: PipelineConfig,
) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
    columns = validate_model_feature_columns(training, feature_columns)
    validate_model_feature_columns(validation, columns)
    started = time.perf_counter()
    cpu_started = time.process_time()
    model = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=config.lgbm_learning_rate,
        num_leaves=config.lgbm_num_leaves,
        feature_fraction=config.lgbm_feature_fraction,
        bagging_fraction=config.lgbm_bagging_fraction,
        bagging_freq=config.lgbm_bagging_freq,
        min_data_in_leaf=config.lgbm_min_data_in_leaf,
        n_estimators=config.lgbm_max_rounds,
        random_state=config.seed,
        bagging_seed=config.seed,
        feature_fraction_seed=config.seed,
        data_random_seed=config.seed,
        deterministic=True,
        force_col_wise=True,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        training[columns],
        training["label"],
        eval_set=[(validation[columns], validation["label"])],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(
                config.lgbm_early_stopping_rounds,
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )
    wall = time.perf_counter() - started
    cpu = time.process_time() - cpu_started
    return model, {
        "training_seconds": wall,
        "training_cpu_seconds": cpu,
        "average_cpu_utilization_percent": 100.0 * cpu / wall if wall else 0.0,
    }


def predict_probabilities(
    model: lgb.LGBMClassifier,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    columns = validate_model_feature_columns(frame, feature_columns)
    probabilities = np.asarray(
        model.predict_proba(
            frame[columns], num_iteration=model.best_iteration_
        )[:, 1],
        dtype="float64",
    )
    if probabilities.shape != (len(frame),):
        raise AssertionError("LightGBM probability shape mismatch")
    if not np.isfinite(probabilities).all() or not (
        (probabilities >= 0.0) & (probabilities <= 1.0)
    ).all():
        raise AssertionError("LightGBM probabilities must be finite in [0, 1]")
    return probabilities


def decode_probabilities(
    frame: pd.DataFrame,
    probabilities: Sequence[float],
    threshold: float,
) -> Dict[str, Set[str]]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    values = np.asarray(probabilities)
    if values.shape != (len(frame),):
        raise ValueError("Probability vector length does not match candidate rows")
    selected = frame.loc[values >= threshold, IDENTIFIER_COLUMNS]
    return pairs_to_mapping(selected)


def _threshold_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    source1_ids: Sequence[str],
    truth: Mapping[str, Set[str]],
) -> Dict[str, object]:
    predictions = decode_probabilities(frame, probabilities, threshold)
    return {
        "threshold": float(threshold),
        **evaluate_predictions(source1_ids, truth, predictions),
    }


def tune_probability_threshold(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    source1_ids: Sequence[str],
    truth: Mapping[str, Set[str]],
) -> Tuple[float, Dict[str, object], pd.DataFrame, float]:
    started = time.perf_counter()
    broad = np.unique(
        np.concatenate(
            [
                np.array([0.001, 0.002, 0.005]),
                np.arange(0.01, 1.00, 0.01),
                np.array([0.995, 0.998, 0.999]),
            ]
        )
    )
    broad_rows = [
        _threshold_metrics(frame, probabilities, value, source1_ids, truth)
        for value in broad
    ]
    broad_best = max(
        broad_rows,
        key=lambda row: (
            row["macro_f0_5"],
            row["pair_micro_precision"],
            row["threshold"],
        ),
    )
    center = float(broad_best["threshold"])
    local = np.arange(max(0.0001, center - 0.02), min(0.9999, center + 0.02) + 0.0005, 0.0005)
    evaluated = {round(float(row["threshold"]), 7): row for row in broad_rows}
    for value in local:
        key = round(float(value), 7)
        if key not in evaluated:
            evaluated[key] = _threshold_metrics(
                frame, probabilities, float(value), source1_ids, truth
            )
    rows = list(evaluated.values())
    best = max(
        rows,
        key=lambda row: (
            row["macro_f0_5"],
            row["pair_micro_precision"],
            row["threshold"],
        ),
    )
    table = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    return (
        float(best["threshold"]),
        best,
        table,
        time.perf_counter() - started,
    )


def _probability_diagnostics(labels: pd.Series, probabilities: np.ndarray) -> Dict[str, object]:
    result: Dict[str, object] = {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }
    for label, name in ((0, "negative"), (1, "positive")):
        values = probabilities[labels.to_numpy() == label]
        result[f"{name}_probability_distribution"] = {
            "count": int(len(values)),
            "minimum": float(values.min()),
            "p05": float(np.percentile(values, 5)),
            "p25": float(np.percentile(values, 25)),
            "median": float(np.median(values)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
            "maximum": float(values.max()),
        }
    return result


def _latest_cache(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No cache matches {directory / pattern}")
    return matches[-1]


def resolve_e2_inputs(config: PipelineConfig) -> Dict[str, Path]:
    normalized_dir = config.cache_dir / "normalized"
    candidate_dir = config.cache_dir / "candidates"
    source1 = _latest_cache(
        normalized_dir, "tier0_validation_s1_*_source1.parquet"
    )
    suffix = "_source1.parquet"
    tag = source1.name[: -len(suffix)]
    match = re.search(r"tier0_validation_s1_(\d+)_", tag)
    if match is None:
        raise ValueError(f"Cannot derive validation size from cache name: {source1}")
    s1_count = int(match.group(1))
    feed = normalized_dir / f"{tag}_feed.parquet"
    legacy = candidate_dir / f"{tag}_candidates.parquet"
    for path in (feed, legacy):
        if not path.exists():
            raise FileNotFoundError(f"Companion E2 cache is missing: {path}")
    tfidf = _latest_cache(
        candidate_dir, f"e1_name_tfidf_s1_{s1_count}_*_tfidf_only.parquet"
    )
    return {
        "source1": source1,
        "feed": feed,
        "legacy_candidates": legacy,
        "tfidf_candidates": tfidf,
    }


def _load_truth_subset(
    config: PipelineConfig, source1_ids: Set[str]
) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    path = config.train_dir / "train_ground_truth.tsv"
    selected: List[pd.DataFrame] = []
    for chunk in load_ground_truth(path, chunksize=config.chunk_size, validate=False):
        validate_ground_truth_frame(chunk, check_duplicates=False, path=path)
        part = chunk.loc[chunk["source1_entity_id"].isin(source1_ids)]
        if not part.empty:
            selected.append(part)
    frame = pd.concat(selected, ignore_index=True)
    validate_ground_truth_frame(frame, path=path)
    if set(frame["source1_entity_id"]) != source1_ids:
        raise RuntimeError("Ground truth does not cover every E2 Source-1 entity")
    return frame, ground_truth_to_mapping(frame)


def _pair_summary(frame: pd.DataFrame) -> Dict[str, object]:
    positives = int(frame["label"].sum())
    negatives = len(frame) - positives
    return {
        "entities": int(frame["source1_entity_id"].nunique()),
        "pairs": len(frame),
        "positive_pairs": positives,
        "negative_pairs": negatives,
        "positive_negative_ratio": positives / negatives if negatives else None,
        "average_pairs_per_s1": float(
            frame.groupby("source1_entity_id").size().mean()
        ),
        "pairs_by_source": {
            "S2": int(frame["source_is_s2"].sum()),
            "S3": int(frame["source_is_s3"].sum()),
        },
        "positives_by_source": {
            "S2": int(frame.loc[frame["label"] == 1, "source_is_s2"].sum()),
            "S3": int(frame.loc[frame["label"] == 1, "source_is_s3"].sum()),
        },
    }


def _false_positive_category(row: pd.Series) -> str:
    exact_name = bool(
        row["name_full_exact"] or row["name_core_exact"] or row["name_sorted_exact"]
    )
    if row["numeric_token_conflict"]:
        return "numeric-address conflict"
    if exact_name and row["address_token_jaccard"] < 0.20:
        return "same name / different address"
    if row["address_exact"] and row["name_character_similarity"] < 0.65:
        return "same address / different business"
    if row["name_core_exact"] and not row["name_full_exact"]:
        return "legal suffix only"
    if exact_name and row["address_token_jaccard"] < 0.50:
        return "location ambiguity"
    if row["name_character_similarity"] < 0.55:
        return "strong lexical noise"
    if row["name_shared_token_count"] <= 1:
        return "generic/common business name"
    return "other high similarity collision"


def _example_payload(
    row: pd.Series,
    source1: pd.DataFrame,
    feed: pd.DataFrame,
) -> Dict[str, object]:
    left = source1.loc[row["source1_entity_id"]]
    right = feed.loc[row["candidate_entity_id"]]
    return {
        "source1_entity_id": row["source1_entity_id"],
        "candidate_entity_id": row["candidate_entity_id"],
        "source1_name": left["business_name"],
        "candidate_name": right["business_name"],
        "source1_address": left["business_address"],
        "candidate_address": right["business_address"],
        "source1_country": left["country"],
        "candidate_country": right["country"],
        "name_character_similarity": float(row["name_character_similarity"]),
        "name_token_jaccard": float(row["name_token_jaccard"]),
        "address_token_jaccard": float(row["address_token_jaccard"]),
        "numeric_token_conflict": int(row["numeric_token_conflict"]),
        "name_tfidf_similarity": float(row["name_tfidf_similarity"]),
        "name_tfidf_rank": int(row["name_tfidf_rank"]),
        "fusion_rank": int(row["fusion_rank"]),
        "probability": float(row["lgbm_probability"]),
        "truth_label": int(row["label"]),
    }


def _error_analysis(
    validation: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    validation_ids: Sequence[str],
    truth: Mapping[str, Set[str]],
    source1_records: pd.DataFrame,
    feed_records: pd.DataFrame,
) -> Dict[str, object]:
    scored = validation.copy()
    scored["lgbm_probability"] = probabilities.astype("float32")
    candidate_pairs = set(zip(scored.source1_entity_id, scored.candidate_entity_id))
    truth_pairs = {
        (s1, target) for s1 in validation_ids for target in truth.get(s1, set())
    }
    predictions = decode_probabilities(scored, probabilities, threshold)
    predicted_pairs = {
        (s1, target) for s1 in validation_ids for target in predictions.get(s1, set())
    }
    retrieval_failures = truth_pairs - candidate_pairs
    classifier_failures = (truth_pairs & candidate_pairs) - predicted_pairs
    false_positives = predicted_pairs - truth_pairs
    indexed = scored.set_index(
        ["source1_entity_id", "candidate_entity_id"],
        drop=False,
        verify_integrity=True,
    )
    source1 = source1_records.set_index("entity_id", verify_integrity=True)
    feed = feed_records.set_index("entity_id", verify_integrity=True)

    fp_rows = [indexed.loc[pair] for pair in false_positives]
    fp_rows.sort(key=lambda row: float(row["lgbm_probability"]), reverse=True)
    fp_categories = Counter(_false_positive_category(row) for row in fp_rows)
    fp_examples = []
    for row in fp_rows[:15]:
        payload = _example_payload(row, source1, feed)
        payload["category"] = _false_positive_category(row)
        fp_examples.append(payload)

    fn_rows = [indexed.loc[pair] for pair in classifier_failures]
    fn_rows.sort(key=lambda row: float(row["lgbm_probability"]))
    fn_examples = []
    for row in fn_rows[:15]:
        payload = _example_payload(row, source1, feed)
        payload["strongest_features"] = {
            "name_core_exact": int(row["name_core_exact"]),
            "address_exact": int(row["address_exact"]),
            "same_country": int(row["same_country"]),
            "numeric_token_overlap": float(row["numeric_token_overlap"]),
            "legacy_cheap_score": float(row["legacy_cheap_score"]),
            "supported_by_both": int(row["supported_by_both"]),
        }
        fn_examples.append(payload)

    retrieval_examples = []
    for s1_id, target_id in sorted(retrieval_failures)[:15]:
        left = source1.loc[s1_id]
        right = feed.loc[target_id]
        retrieval_examples.append(
            {
                "source1_entity_id": s1_id,
                "candidate_entity_id": target_id,
                "source1_name": left["business_name"],
                "candidate_name": right["business_name"],
                "source1_address": left["business_address"],
                "candidate_address": right["business_address"],
                "source1_country": left["country"],
                "candidate_country": right["country"],
            }
        )
    return {
        "retrieval_false_negatives": len(retrieval_failures),
        "retrieval_false_negative_examples": retrieval_examples,
        "classifier_false_negatives": len(classifier_failures),
        "classifier_false_negative_examples": fn_examples,
        "false_positives": len(false_positives),
        "false_positive_categories": dict(fp_categories),
        "high_confidence_false_positive_examples": fp_examples,
    }


def _feature_importance(
    model: lgb.LGBMClassifier, feature_columns: Sequence[str]
) -> Dict[str, object]:
    booster = model.booster_
    gains = booster.feature_importance(importance_type="gain")
    splits = booster.feature_importance(importance_type="split")
    rows = [
        {"feature": feature, "gain": float(gain), "split": int(split)}
        for feature, gain, split in zip(feature_columns, gains, splits)
    ]
    top_gain = sorted(rows, key=lambda row: row["gain"], reverse=True)[:20]
    top_split = sorted(rows, key=lambda row: row["split"], reverse=True)[:20]
    total_gain = float(np.sum(gains))
    groups = {
        "country": ["same_country", "country_conflict", "country_missing_left", "country_missing_right"],
        "exact_name": ["name_full_exact", "name_core_exact", "name_sorted_exact"],
        "tfidf": ["name_tfidf_similarity", "name_tfidf_rank", "relative_tfidf_similarity_rank"],
        "address_numbers": ["numeric_token_equal", "numeric_token_overlap", "numeric_token_conflict"],
    }
    gain_map = {row["feature"]: row["gain"] for row in rows}
    group_shares = {
        name: sum(gain_map.get(feature, 0.0) for feature in group) / total_gain
        if total_gain
        else 0.0
        for name, group in groups.items()
    }
    return {
        "top_20_gain": top_gain,
        "top_20_split": top_split,
        "selected_signal_group_gain_share": group_shares,
        "all_features": rows,
    }


def _artifact_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def run_e2_experiment(
    config: PipelineConfig = DEFAULT_CONFIG,
    *,
    persist: bool = True,
) -> Dict[str, object]:
    total_started = time.perf_counter()
    inputs = resolve_e2_inputs(config)
    source1 = pd.read_parquet(inputs["source1"])
    feed = pd.read_parquet(inputs["feed"])
    legacy_candidates = pd.read_parquet(inputs["legacy_candidates"])
    tfidf_candidates = pd.read_parquet(inputs["tfidf_candidates"])
    if source1.empty:
        raise ValueError("E2 Source-1 cache is empty")

    feature_started = time.perf_counter()
    universe = prepare_fusion_universe(
        legacy_candidates, tfidf_candidates, tfidf_top_k=30
    )
    candidates = reciprocal_rank_fusion(
        universe,
        final_k=50,
        legacy_weight=2.0,
        tfidf_weight=1.0,
        rrf_constant=20.0,
    )
    _, truth = _load_truth_subset(config, set(source1["entity_id"]))
    base_features = build_feature_table(
        candidates, source1, feed, chunk_size=config.chunk_size
    )
    features, label_metrics = build_pair_labels(base_features, truth)
    features = add_candidate_relative_features(
        features,
        close_competitor_tolerance=config.e2_close_competitor_tolerance,
    )
    feature_columns = validate_model_feature_columns(features)
    feature_preparation_seconds = time.perf_counter() - feature_started

    training_ids, validation_ids = split_source1_ids(
        source1,
        validation_fraction=config.e2_validation_fraction,
        seed=config.e2_split_seed,
    )
    training, validation = split_pair_features(features, training_ids, validation_ids)
    validation_id_list = sorted(validation_ids)
    validation_truth = {s1: truth.get(s1, set()) for s1 in validation_ids}
    validation_candidates = candidates.loc[
        candidates["source1_entity_id"].isin(validation_ids)
    ]
    candidate_report = candidate_metrics(
        validation_id_list, validation_truth, validation_candidates
    )

    hard_training = sample_hard_negatives(
        training,
        negatives_per_positive=config.e2_hard_negatives_per_positive,
    )
    variants = {
        "all_candidate_negatives": training,
        "hard_negative_5_to_1": hard_training,
    }
    variant_reports: Dict[str, object] = {}
    trained_models: Dict[str, lgb.LGBMClassifier] = {}
    validation_probabilities: Dict[str, np.ndarray] = {}
    for name, train_frame in variants.items():
        model, timing = fit_lgbm_classifier(
            train_frame,
            validation,
            feature_columns=feature_columns,
            config=config,
        )
        inference_started = time.perf_counter()
        probabilities = predict_probabilities(model, validation, feature_columns)
        inference_seconds = time.perf_counter() - inference_started
        threshold, metrics, threshold_table, threshold_seconds = tune_probability_threshold(
            validation,
            probabilities,
            validation_id_list,
            validation_truth,
        )
        variant_reports[name] = {
            "training_pairs": _pair_summary(train_frame),
            "best_iteration": int(model.best_iteration_),
            "best_probability_threshold": threshold,
            "entity_metrics": metrics,
            "pair_classification": _probability_diagnostics(
                validation["label"], probabilities
            ),
            "training_performance": timing,
            "validation_inference_seconds": inference_seconds,
            "threshold_sweep_seconds": threshold_seconds,
            "threshold_grid": threshold_table["threshold"].tolist(),
            "threshold_local_results": threshold_table.loc[
                (threshold_table["threshold"] >= threshold - 0.005)
                & (threshold_table["threshold"] <= threshold + 0.005)
            ].to_dict(orient="records"),
        }
        trained_models[name] = model
        validation_probabilities[name] = probabilities

    selected_variant = max(
        variant_reports,
        key=lambda name: (
            variant_reports[name]["entity_metrics"]["macro_f0_5"],
            variant_reports[name]["entity_metrics"]["pair_micro_precision"],
        ),
    )
    selected_model = trained_models[selected_variant]
    selected_probabilities = validation_probabilities[selected_variant]
    selected_report = variant_reports[selected_variant]
    selected_threshold = float(selected_report["best_probability_threshold"])
    selected_predictions = decode_probabilities(
        validation, selected_probabilities, selected_threshold
    )

    e0_scored = score_rule_baseline(validation)
    e0_predictions = predict_from_scores(e0_scored, 0.70)
    e0_metrics = evaluate_predictions(
        validation_id_list, validation_truth, e0_predictions
    )
    validation_source1 = source1.loc[source1["entity_id"].isin(validation_ids)]
    breakdowns = evaluate_breakdowns(
        validation_source1, validation_truth, selected_predictions
    )
    e0_breakdowns = evaluate_breakdowns(
        validation_source1, validation_truth, e0_predictions
    )
    error_analysis = _error_analysis(
        validation,
        selected_probabilities,
        selected_threshold,
        validation_id_list,
        validation_truth,
        source1,
        feed,
    )
    importance = _feature_importance(selected_model, feature_columns)
    model_metrics = selected_report["entity_metrics"]

    artifact_context = {
        "experiment": "E2_lightgbm_pair_classifier",
        "lightgbm_version": lgb.__version__,
        "config": config.as_dict(),
        "inputs": {name: str(path) for name, path in inputs.items()},
        "candidate_configuration": {
            "fusion": "rrf",
            "legacy_weight": 2.0,
            "tfidf_weight": 1.0,
            "rrf_constant": 20.0,
            "tfidf_top_k": 30,
            "final_k": 50,
        },
        "feature_columns": feature_columns,
        "selected_variant": selected_variant,
        "train_entities": len(training_ids),
        "validation_entities": len(validation_ids),
    }
    artifact_id = _artifact_hash(artifact_context)
    report: Dict[str, object] = {
        **artifact_context,
        "artifact_id": artifact_id,
        "training_split_seed": config.e2_split_seed,
        "model_seed": config.seed,
        "training_pairs_all": _pair_summary(training),
        "validation_pairs": _pair_summary(validation),
        "label_metrics_full_universe": label_metrics,
        "candidate_metrics_validation": candidate_report,
        "model_variants": variant_reports,
        "selected_threshold": selected_threshold,
        "selected_model_metrics": model_metrics,
        "e0_threshold": 0.70,
        "e0_metrics_same_validation": e0_metrics,
        "lightgbm_breakdowns": breakdowns,
        "e0_breakdowns": e0_breakdowns,
        "oracle_gap": float(candidate_report["oracle_macro_f0_5"])
        - float(model_metrics["macro_f0_5"]),
        "error_analysis": error_analysis,
        "feature_importance": importance,
        "performance": {
            "feature_preparation_seconds": feature_preparation_seconds,
            "total_experiment_seconds": time.perf_counter() - total_started,
            "peak_rss_mib": _peak_rss_mib(),
        },
    }

    if persist:
        model_dir = config.models_dir
        prediction_dir = config.cache_dir / "predictions"
        experiment_dir = config.experiments_dir
        for directory in (model_dir, prediction_dir, experiment_dir):
            directory.mkdir(parents=True, exist_ok=True)
        prefix = f"e2_lgbm_{artifact_id}"
        paths = {
            "model": model_dir / f"{prefix}_model.txt",
            "features": model_dir / f"{prefix}_features.json",
            "config": model_dir / f"{prefix}_config.json",
            "threshold": model_dir / f"{prefix}_threshold.json",
            "metadata": model_dir / f"{prefix}_metadata.json",
            "predictions": prediction_dir / f"{prefix}_validation.parquet",
            "report": experiment_dir / f"{prefix}_metrics.json",
        }
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite existing E2 artifacts: {existing}"
            )
        selected_model.booster_.save_model(
            str(paths["model"]), num_iteration=selected_model.best_iteration_
        )
        paths["features"].write_text(
            json.dumps(feature_columns, indent=2) + "\n", encoding="utf-8"
        )
        paths["config"].write_text(
            json.dumps(artifact_context, indent=2, sort_keys=True, default=_json_default)
            + "\n",
            encoding="utf-8",
        )
        paths["threshold"].write_text(
            json.dumps(
                {"selected_threshold": selected_threshold}, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        probability_frame = validation[IDENTIFIER_COLUMNS + ["label"]].copy()
        probability_frame["lgbm_probability"] = selected_probabilities.astype(
            "float32"
        )
        write_parquet_cache(
            probability_frame,
            paths["predictions"],
            {
                "artifact_id": artifact_id,
                "selected_variant": selected_variant,
                "selected_threshold": selected_threshold,
            },
        )
        report["artifacts"] = {name: str(path) for name, path in paths.items()}
        paths["metadata"].write_text(
            json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        paths["report"].write_text(
            json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-persist", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_e2_experiment(DEFAULT_CONFIG, persist=not args.no_persist)
    print(json.dumps(report, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
