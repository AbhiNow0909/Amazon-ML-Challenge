"""E2.1 larger-universe distractor-density stress test."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import time
from collections import Counter
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from .blocking import generate_candidates
from .candidate_fusion import prepare_fusion_universe, reciprocal_rank_fusion
from .config import DEFAULT_CONFIG, PipelineConfig
from .evaluation import candidate_metrics, evaluate_breakdowns, evaluate_predictions
from .features import add_candidate_relative_features, build_feature_table
from .features import IDENTIFIER_COLUMNS
from .io_utils import (
    ground_truth_to_mapping,
    load_ground_truth,
    load_source,
    validate_ground_truth_frame,
    validate_source_frame,
    write_parquet_cache,
)
from .labels import build_pair_labels
from .normalize import normalize_records
from .split import select_validation_entities, split_source1_ids
from .tfidf_retrieval import retrieve_name_tfidf
from .train_lgbm import (
    MODEL_FEATURE_COLUMNS,
    _error_analysis,
    _feature_importance,
    _json_default,
    _pair_summary,
    _probability_diagnostics,
    decode_probabilities,
    fit_lgbm_classifier,
    predict_probabilities,
    split_pair_features,
    tune_probability_threshold,
    validate_model_feature_columns,
)


FROZEN_CANDIDATE_CONFIG = {
    "fusion": "rrf",
    "legacy_weight": 2.0,
    "tfidf_weight": 1.0,
    "rrf_constant": 20.0,
    "tfidf_top_k": 30,
    "final_k": 50,
}
OLD_E2_THRESHOLD = 0.759
IMPORTANCE_FEATURES = [
    "address_token_jaccard",
    "numeric_token_overlap",
    "fusion_score",
    "relative_address_similarity_rank",
    "name_character_similarity",
    "name_token_jaccard",
    "name_tfidf_similarity",
    "relative_tfidf_similarity_rank",
    "numeric_token_conflict",
]


def _event(name: str, **values: object) -> None:
    print(json.dumps({"event": name, **values}, default=_json_default), flush=True)


def _peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _select_source1(config: PipelineConfig, limit: int) -> pd.DataFrame:
    selected: List[pd.DataFrame] = []
    count = 0
    path = config.train_dir / "train_source1.tsv"
    for chunk in load_source(path, "source1", chunksize=config.chunk_size, validate=False):
        validate_source_frame(chunk, "source1", check_duplicates=False, path=path)
        part = select_validation_entities(
            chunk,
            validation_fraction=config.validation_fraction,
            seed=config.seed,
            limit=limit - count,
        )
        if not part.empty:
            selected.append(part)
            count += len(part)
        if count >= limit:
            break
    result = pd.concat(selected, ignore_index=True).head(limit)
    validate_source_frame(result, "source1", path=path)
    if len(result) != limit:
        raise RuntimeError(f"Requested {limit} deterministic S1 rows, found {len(result)}")
    return result


def _load_truth(
    config: PipelineConfig, source1_ids: Set[str]
) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    selected: List[pd.DataFrame] = []
    path = config.train_dir / "train_ground_truth.tsv"
    for chunk in load_ground_truth(path, chunksize=config.chunk_size, validate=False):
        validate_ground_truth_frame(chunk, check_duplicates=False, path=path)
        part = chunk.loc[chunk["source1_entity_id"].isin(source1_ids)]
        if not part.empty:
            selected.append(part)
    frame = pd.concat(selected, ignore_index=True)
    validate_ground_truth_frame(frame, path=path)
    if set(frame["source1_entity_id"]) != source1_ids:
        raise RuntimeError("Stress-test truth does not cover every selected S1")
    return frame, ground_truth_to_mapping(frame)


def _deterministic_hash(values: pd.Series, seed: int, salt: str) -> pd.Series:
    payload = values.astype(str) + f"|{seed}|{salt}"
    return pd.util.hash_pandas_object(payload, index=False).astype("uint64")


def load_deterministic_feed_sample(
    config: PipelineConfig,
    *,
    source: str,
    required_ids: Set[str],
    distractor_limit: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Retain all truth targets and the lowest deterministic hashes."""

    if distractor_limit <= 0:
        raise ValueError("distractor_limit must be positive")
    started = time.perf_counter()
    number = source[-1]
    path = config.train_dir / f"train_source{number}.tsv"
    required_parts: List[pd.DataFrame] = []
    pool: Optional[pd.DataFrame] = None
    scanned = 0
    for chunk in load_source(path, source, chunksize=config.chunk_size, validate=False):
        validate_source_frame(chunk, source, check_duplicates=False, path=path)
        scanned += len(chunk)
        required_mask = chunk["entity_id"].isin(required_ids)
        required = chunk.loc[required_mask]
        if not required.empty:
            required_parts.append(required)
        eligible = chunk.loc[~required_mask].copy()
        eligible["_sample_hash"] = _deterministic_hash(
            eligible["entity_id"], seed, source
        ).to_numpy()
        eligible["_sample_hash2"] = _deterministic_hash(
            eligible["entity_id"], seed + 1, source
        ).to_numpy()
        pool = (
            eligible.reset_index(drop=True)
            if pool is None
            else pd.concat([pool, eligible], ignore_index=True)
        )
        if len(pool) > 2 * distractor_limit:
            pool = pool.nsmallest(
                distractor_limit, ["_sample_hash", "_sample_hash2"]
            ).reset_index(drop=True)
    if pool is None:
        raise RuntimeError(f"No eligible distractors found in {source}")
    distractors = pool.nsmallest(
        distractor_limit, ["_sample_hash", "_sample_hash2"]
    ).sort_values(
        ["_sample_hash", "_sample_hash2", "entity_id"], kind="mergesort"
    )
    if len(distractors) != distractor_limit:
        raise RuntimeError(
            f"{source} contains only {len(distractors)} eligible distractors"
        )
    distractors = distractors.copy()
    distractors["_is_distractor"] = np.uint8(1)
    distractors["_distractor_rank"] = np.arange(
        1, len(distractors) + 1, dtype="uint32"
    )
    required = pd.concat(required_parts, ignore_index=True).drop_duplicates("entity_id")
    found = set(required["entity_id"])
    if found != required_ids:
        raise RuntimeError(
            f"Missing required {source} targets: {sorted(required_ids - found)[:5]}"
        )
    required = required.copy()
    required["_sample_hash"] = np.uint64(0)
    required["_sample_hash2"] = np.uint64(0)
    required["_is_distractor"] = np.uint8(0)
    required["_distractor_rank"] = np.uint32(0)
    result = pd.concat([required, distractors], ignore_index=True)
    result["_feed_source"] = source
    return result, {
        "source": source,
        "rows_scanned": scanned,
        "required_rows": len(required),
        "distractor_rows": len(distractors),
        "runtime_seconds": time.perf_counter() - started,
    }


def feed_at_density(feed: pd.DataFrame, total_distractors: int) -> pd.DataFrame:
    """Return a nested, source-balanced density subset."""

    if total_distractors <= 0 or total_distractors % 2:
        raise ValueError("total_distractors must be a positive even number")
    per_source = total_distractors // 2
    mask = (feed["_is_distractor"] == 0) | (
        (feed["_is_distractor"] == 1)
        & (feed["_distractor_rank"] <= per_source)
    )
    result = feed.loc[mask].copy()
    actual = int(result["_is_distractor"].sum())
    if actual != total_distractors:
        raise AssertionError(
            f"Density subset has {actual} distractors; expected {total_distractors}"
        )
    return result.reset_index(drop=True)


def _truth_count_distribution(truth: Mapping[str, Set[str]]) -> Dict[str, int]:
    result = Counter()
    for targets in truth.values():
        count = len(targets)
        result[str(count) if count < 3 else "3+"] += 1
    return dict(result)


def run_retrieval_density(
    source1: pd.DataFrame,
    feed: pd.DataFrame,
    truth: Mapping[str, Set[str]],
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    base_config = replace(config, use_name_tfidf=False)
    tfidf_config = replace(config, use_name_tfidf=True, tfidf_name_top_k=30)
    blocking_started = time.perf_counter()
    legacy, blocking_profile = generate_candidates(
        source1, feed, base_config, max_k=50
    )
    blocking_seconds = time.perf_counter() - blocking_started
    tfidf_started = time.perf_counter()
    tfidf, tfidf_profile = retrieve_name_tfidf(source1, feed, tfidf_config)
    tfidf_seconds = time.perf_counter() - tfidf_started
    fusion_started = time.perf_counter()
    universe = prepare_fusion_universe(legacy, tfidf, tfidf_top_k=30)
    candidates = reciprocal_rank_fusion(
        universe,
        final_k=50,
        legacy_weight=2.0,
        tfidf_weight=1.0,
        rrf_constant=20.0,
    )
    fusion_seconds = time.perf_counter() - fusion_started
    metrics = candidate_metrics(source1["entity_id"].tolist(), truth, candidates)
    return candidates, {
        "feed_rows": len(feed),
        "metrics": metrics,
        "blocking_seconds": blocking_seconds,
        "tfidf_retrieval_seconds": tfidf_seconds,
        "fusion_seconds": fusion_seconds,
        "blocking_profile": blocking_profile,
        "tfidf_profile": tfidf_profile,
        "peak_rss_mib": _peak_rss_mib(),
    }


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(str(left).split())
    right_tokens = set(str(right).split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def _script_mismatch(left: str, right: str) -> bool:
    left_non_ascii = any(ord(char) > 127 and char.isalpha() for char in left)
    right_non_ascii = any(ord(char) > 127 and char.isalpha() for char in right)
    return left_non_ascii != right_non_ascii


def _retrieval_failure_category(left: pd.Series, right: pd.Series) -> str:
    left_name = str(left["name_core"])
    right_name = str(right["name_core"])
    raw = f" {left['business_name']} {right['business_name']} ".casefold()
    character_similarity = SequenceMatcher(
        None, left_name, right_name, autojunk=False
    ).ratio()
    address_similarity = _token_jaccard(
        left["address_tokens"], right["address_tokens"]
    )
    if _script_mismatch(left_name, right_name):
        return "transliteration/script mismatch"
    if character_similarity < 0.35 and address_similarity >= 0.35:
        return "address-only evidence"
    if any(marker in raw for marker in (".com", " dba ", "d/b/a", "trading as", " aka ")):
        return "abbreviation/domain/trade-name"
    length_ratio = min(len(left_name), len(right_name)) / max(
        len(left_name), len(right_name), 1
    )
    if length_ratio < 0.60 and character_similarity >= 0.30:
        return "abbreviation/domain/trade-name"
    if min(len(left_name), len(right_name)) <= 5:
        return "generic/short name"
    if character_similarity < 0.55:
        return "severe spelling corruption"
    return "other"


def _classifier_failure_category(
    row: pd.Series, left: pd.Series, right: pd.Series
) -> str:
    if row["numeric_token_conflict"]:
        return "numeric-address conflict"
    if not str(left["address_full"]) or not str(right["address_full"]):
        return "missing address"
    if _script_mismatch(str(left["name_core"]), str(right["name_core"])):
        return "transliteration"
    if (
        row["name_character_similarity"] >= 0.80
        and row["address_token_jaccard"] < 0.25
    ):
        return "strong name / weak address"
    return "other"


def _categorized_errors(
    validation: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    validation_ids: Sequence[str],
    truth: Mapping[str, Set[str]],
    source1: pd.DataFrame,
    feed: pd.DataFrame,
) -> Dict[str, object]:
    scored = validation.copy()
    scored["lgbm_probability"] = probabilities
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
    left = source1.set_index("entity_id", verify_integrity=True)
    right = feed.set_index("entity_id", verify_integrity=True)
    indexed = scored.set_index(
        ["source1_entity_id", "candidate_entity_id"], verify_integrity=True
    )
    retrieval_categories = Counter(
        _retrieval_failure_category(left.loc[s1], right.loc[target])
        for s1, target in retrieval_failures
    )
    classifier_categories = Counter(
        _classifier_failure_category(
            indexed.loc[(s1, target)], left.loc[s1], right.loc[target]
        )
        for s1, target in classifier_failures
    )

    def with_percentages(counter: Counter, total: int) -> Dict[str, object]:
        return {
            key: {
                "count": value,
                "percentage": 100.0 * value / total if total else 0.0,
            }
            for key, value in sorted(counter.items())
        }

    return {
        "retrieval_failures": len(retrieval_failures),
        "retrieval_failure_percentage_of_truth": 100.0
        * len(retrieval_failures)
        / len(truth_pairs),
        "retrieval_failure_categories": with_percentages(
            retrieval_categories, len(retrieval_failures)
        ),
        "classifier_failures": len(classifier_failures),
        "classifier_failure_percentage_of_retrieved_truth": 100.0
        * len(classifier_failures)
        / max(1, len(truth_pairs & candidate_pairs)),
        "classifier_failure_categories": with_percentages(
            classifier_categories, len(classifier_failures)
        ),
    }


def _importance_comparison(
    importance: Mapping[str, object], previous_report_path: Path
) -> Dict[str, object]:
    previous = json.loads(previous_report_path.read_text(encoding="utf-8"))
    current_rows = importance["all_features"]
    previous_rows = previous["feature_importance"]["all_features"]

    def shares(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
        total = sum(float(row["gain"]) for row in rows)
        values = {str(row["feature"]): float(row["gain"]) for row in rows}
        return {
            feature: values.get(feature, 0.0) / total if total else 0.0
            for feature in IMPORTANCE_FEATURES
        }

    current = shares(current_rows)
    previous_values = shares(previous_rows)
    return {
        feature: {
            "e2_gain_share": previous_values[feature],
            "e21_gain_share": current[feature],
            "change": current[feature] - previous_values[feature],
        }
        for feature in IMPORTANCE_FEATURES
    }


def _artifact_hash(payload: Mapping[str, object]) -> str:
    value = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:12]


def run_e21_experiment(
    config: PipelineConfig = DEFAULT_CONFIG,
    *,
    persist: bool = True,
) -> Dict[str, object]:
    total_started = time.perf_counter()
    universe_started = time.perf_counter()
    source1_raw = _select_source1(config, config.e21_s1_limit)
    truth_frame, truth = _load_truth(config, set(source1_raw["entity_id"]))
    required_s2 = {
        target for targets in truth.values() for target in targets if target.startswith("S2-")
    }
    required_s3 = {
        target for targets in truth.values() for target in targets if target.startswith("S3-")
    }
    maximum_total = max(config.e21_distractor_totals)
    per_source_maximum = maximum_total // 2
    source2_raw, source2_sample_profile = load_deterministic_feed_sample(
        config,
        source="source2",
        required_ids=required_s2,
        distractor_limit=per_source_maximum,
        seed=config.e21_distractor_seed,
    )
    source3_raw, source3_sample_profile = load_deterministic_feed_sample(
        config,
        source="source3",
        required_ids=required_s3,
        distractor_limit=per_source_maximum,
        seed=config.e21_distractor_seed,
    )
    feed_raw_max = pd.concat([source2_raw, source3_raw], ignore_index=True)
    universe_seconds = time.perf_counter() - universe_started
    composition = {
        "source1_rows": len(source1_raw),
        "true_links": sum(len(targets) for targets in truth.values()),
        "required_source2_rows": len(required_s2),
        "required_source3_rows": len(required_s3),
        "distractor_rows": int(feed_raw_max["_is_distractor"].sum()),
        "total_feed_rows": len(feed_raw_max),
        "source1_country_distribution": source1_raw["country"].value_counts().to_dict(),
        "truth_count_distribution": _truth_count_distribution(truth),
        "source2_sampling": source2_sample_profile,
        "source3_sampling": source3_sample_profile,
    }
    _event("e2.1-universe", composition=composition, seconds=universe_seconds)

    normalization_started = time.perf_counter()
    source1 = normalize_records(source1_raw)
    feed_max = normalize_records(feed_raw_max)
    normalization_seconds = time.perf_counter() - normalization_started
    _event(
        "e2.1-normalized",
        source1_rows=len(source1),
        feed_rows=len(feed_max),
        seconds=normalization_seconds,
        peak_rss_mib=_peak_rss_mib(),
    )

    density_results: Dict[str, object] = {}
    primary_candidates: Optional[pd.DataFrame] = None
    primary_feed: Optional[pd.DataFrame] = None
    for density in config.e21_distractor_totals:
        density_feed = feed_at_density(feed_max, density)
        density_started = time.perf_counter()
        candidates, profile = run_retrieval_density(
            source1, density_feed, truth, config
        )
        profile["total_runtime_seconds"] = time.perf_counter() - density_started
        profile["distractor_rows"] = density
        density_results[str(density)] = profile
        _event(
            "e2.1-density",
            distractors=density,
            feed_rows=len(density_feed),
            metrics=profile["metrics"],
            seconds=profile["total_runtime_seconds"],
            peak_rss_mib=_peak_rss_mib(),
        )
        if density == maximum_total:
            primary_candidates = candidates
            primary_feed = density_feed
        else:
            del candidates, density_feed
            gc.collect()
    if primary_candidates is None or primary_feed is None:
        raise AssertionError("Primary density artifacts were not retained")

    feature_started = time.perf_counter()
    base_features = build_feature_table(
        primary_candidates, source1, primary_feed, chunk_size=config.chunk_size
    )
    features, label_metrics = build_pair_labels(base_features, truth)
    features = add_candidate_relative_features(
        features,
        close_competitor_tolerance=config.e2_close_competitor_tolerance,
    )
    feature_columns = validate_model_feature_columns(features, MODEL_FEATURE_COLUMNS)
    feature_seconds = time.perf_counter() - feature_started
    _event(
        "e2.1-features",
        pairs=len(features),
        seconds=feature_seconds,
        peak_rss_mib=_peak_rss_mib(),
    )

    training_ids, validation_ids = split_source1_ids(
        source1,
        validation_fraction=config.e2_validation_fraction,
        seed=config.e2_split_seed,
    )
    training, validation = split_pair_features(features, training_ids, validation_ids)
    validation_ids_list = sorted(validation_ids)
    validation_truth = {s1: truth.get(s1, set()) for s1 in validation_ids}
    validation_candidates = primary_candidates.loc[
        primary_candidates["source1_entity_id"].isin(validation_ids)
    ]
    validation_candidate_metrics = candidate_metrics(
        validation_ids_list, validation_truth, validation_candidates
    )

    model, training_performance = fit_lgbm_classifier(
        training,
        validation,
        feature_columns=feature_columns,
        config=config,
    )
    inference_started = time.perf_counter()
    probabilities = predict_probabilities(model, validation, feature_columns)
    inference_seconds = time.perf_counter() - inference_started
    old_predictions = decode_probabilities(validation, probabilities, OLD_E2_THRESHOLD)
    old_threshold_metrics = evaluate_predictions(
        validation_ids_list, validation_truth, old_predictions
    )
    tuned_threshold, tuned_metrics, threshold_table, threshold_seconds = (
        tune_probability_threshold(
            validation, probabilities, validation_ids_list, validation_truth
        )
    )
    tuned_predictions = decode_probabilities(
        validation, probabilities, tuned_threshold
    )
    pair_diagnostics = _probability_diagnostics(validation["label"], probabilities)
    importance = _feature_importance(model, feature_columns)
    previous_report = max(
        config.experiments_dir.glob("e2_lgbm_*_metrics.json"),
        key=lambda path: path.stat().st_mtime,
    )
    importance_comparison = _importance_comparison(importance, previous_report)

    validation_source1 = source1.loc[source1["entity_id"].isin(validation_ids)]
    model_breakdowns = evaluate_breakdowns(
        validation_source1, validation_truth, tuned_predictions
    )
    country_stability: Dict[str, object] = {}
    for country, group in validation_source1.groupby("country", sort=True):
        ids = group["entity_id"].tolist()
        country_candidates = validation_candidates.loc[
            validation_candidates["source1_entity_id"].isin(ids)
        ]
        retrieval = candidate_metrics(ids, validation_truth, country_candidates)
        model_metrics = evaluate_predictions(ids, validation_truth, tuned_predictions)
        country_stability[str(country)] = {
            "candidate_retrieval": retrieval,
            "model": model_metrics,
        }

    base_errors = _error_analysis(
        validation,
        probabilities,
        tuned_threshold,
        validation_ids_list,
        validation_truth,
        source1,
        primary_feed,
    )
    categorized_errors = _categorized_errors(
        validation,
        probabilities,
        tuned_threshold,
        validation_ids_list,
        validation_truth,
        source1,
        primary_feed,
    )

    artifact_context = {
        "experiment": "E2.1_scale_stress_test",
        "lightgbm_version": lgb.__version__,
        "config": config.as_dict(),
        "candidate_configuration": FROZEN_CANDIDATE_CONFIG,
        "feature_columns": feature_columns,
        "primary_distractor_density": maximum_total,
        "source1_rows": len(source1),
        "previous_e2_report": str(previous_report),
    }
    artifact_id = _artifact_hash(artifact_context)
    report: Dict[str, object] = {
        **artifact_context,
        "artifact_id": artifact_id,
        "composition": composition,
        "train_entities": len(training_ids),
        "validation_entities": len(validation_ids),
        "training_pairs": _pair_summary(training),
        "validation_pairs": _pair_summary(validation),
        "label_metrics_full_universe": label_metrics,
        "density_sensitivity": density_results,
        "validation_candidate_metrics": validation_candidate_metrics,
        "old_threshold": OLD_E2_THRESHOLD,
        "old_threshold_metrics": old_threshold_metrics,
        "retuned_threshold": tuned_threshold,
        "retuned_threshold_metrics": tuned_metrics,
        "threshold_grid": threshold_table["threshold"].tolist(),
        "pair_classification": pair_diagnostics,
        "best_iteration": int(model.best_iteration_),
        "model_breakdowns": model_breakdowns,
        "country_stability": country_stability,
        "feature_importance": importance,
        "feature_importance_comparison": importance_comparison,
        "error_analysis": {**base_errors, **categorized_errors},
        "oracle_gap": float(validation_candidate_metrics["oracle_macro_f0_5"])
        - float(tuned_metrics["macro_f0_5"]),
        "performance": {
            "universe_construction_seconds": universe_seconds,
            "normalization_seconds": normalization_seconds,
            "feature_generation_seconds": feature_seconds,
            "model_training": training_performance,
            "validation_inference_seconds": inference_seconds,
            "threshold_sweep_seconds": threshold_seconds,
            "total_wall_seconds_before_persistence": time.perf_counter() - total_started,
            "peak_rss_mib": _peak_rss_mib(),
        },
    }

    if persist:
        persistence_started = time.perf_counter()
        prefix = f"e21_lgbm_{artifact_id}"
        paths = {
            "model": config.models_dir / f"{prefix}_model.txt",
            "features": config.models_dir / f"{prefix}_features.json",
            "config": config.models_dir / f"{prefix}_config.json",
            "threshold": config.models_dir / f"{prefix}_threshold.json",
            "metadata": config.models_dir / f"{prefix}_metadata.json",
            "predictions": config.cache_dir / "predictions" / f"{prefix}_validation.parquet",
            "normalized_source1": config.cache_dir / "normalized" / f"{prefix}_source1.parquet",
            "normalized_feed": config.cache_dir / "normalized" / f"{prefix}_feed.parquet",
            "candidates": config.cache_dir / "candidates" / f"{prefix}_candidates.parquet",
            "feature_cache": config.cache_dir / "features" / f"{prefix}_features.parquet",
            "report": config.experiments_dir / f"{prefix}_metrics.json",
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite E2.1 artifacts: {existing}")
        model.booster_.save_model(str(paths["model"]), num_iteration=model.best_iteration_)
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
                {"old_threshold": OLD_E2_THRESHOLD, "retuned_threshold": tuned_threshold},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        probabilities_frame = validation[IDENTIFIER_COLUMNS + ["label"]].copy()
        probabilities_frame["lgbm_probability"] = probabilities.astype("float32")
        common_metadata = {"artifact_id": artifact_id, "distractors": maximum_total}
        write_parquet_cache(probabilities_frame, paths["predictions"], common_metadata)
        write_parquet_cache(source1, paths["normalized_source1"], common_metadata)
        write_parquet_cache(primary_feed, paths["normalized_feed"], common_metadata)
        write_parquet_cache(primary_candidates, paths["candidates"], common_metadata)
        write_parquet_cache(features, paths["feature_cache"], common_metadata)
        report["artifacts"] = {name: str(path) for name, path in paths.items()}
        report["performance"]["artifact_persistence_seconds"] = (
            time.perf_counter() - persistence_started
        )
        report["performance"]["total_wall_seconds"] = time.perf_counter() - total_started
        paths["metadata"].write_text(
            json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        paths["report"].write_text(
            json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
    _event(
        "e2.1-complete",
        artifact_id=artifact_id,
        retuned_threshold=tuned_threshold,
        metrics=tuned_metrics,
        total_seconds=report["performance"].get(
            "total_wall_seconds", report["performance"]["total_wall_seconds_before_persistence"]
        ),
        peak_rss_mib=_peak_rss_mib(),
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-persist", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_e21_experiment(DEFAULT_CONFIG, persist=not args.no_persist)
    print(json.dumps(report, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
