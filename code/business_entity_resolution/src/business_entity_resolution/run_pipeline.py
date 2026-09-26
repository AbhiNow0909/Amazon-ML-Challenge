"""Command-line orchestrator for bounded Tier 0 experiments."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .blocking import candidates_at_k, generate_candidates
from .candidate_fusion import (
    hybrid_quota_rank_fusion,
    naive_score_fusion,
    normalized_score_fusion,
    prepare_fusion_universe,
    quota_fusion,
    reciprocal_rank_fusion,
)
from .config import DEFAULT_CONFIG, PipelineConfig
from .evaluation import (
    blocker_recovery,
    candidate_metrics,
    error_diagnostics,
    evaluate_breakdowns,
    evaluate_predictions,
)
from .features import build_feature_table
from .io_utils import (
    GROUND_TRUTH_COLUMNS,
    SOURCE_COLUMNS,
    ground_truth_to_mapping,
    inspect_tsv_schema,
    load_ground_truth,
    load_source,
    validate_ground_truth_frame,
    validate_source_frame,
    write_parquet_cache,
)
from .labels import build_pair_labels
from .normalize import normalize_records
from .rules import predict_from_scores, score_rule_baseline, tune_rule_threshold
from .split import select_validation_entities
from .submission import (
    build_submission_frames,
    validate_submission_frames,
    write_submission,
)
from .tfidf_retrieval import retrieve_name_tfidf


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _print_event(event: str, **payload: object) -> None:
    print(json.dumps({"event": event, **payload}, default=_json_default), flush=True)


def _peak_rss_mib() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_audit_lite(config: PipelineConfig) -> Dict[str, object]:
    files = [
        config.train_dir / "train_source1.tsv",
        config.train_dir / "train_source2.tsv",
        config.train_dir / "train_source3.tsv",
        config.train_dir / "train_ground_truth.tsv",
        config.test_dir / "test_source1.tsv",
        config.test_dir / "test_source2.tsv",
        config.test_dir / "test_source3.tsv",
    ]
    report = {"config": config.as_dict(), "files": [inspect_tsv_schema(path) for path in files]}
    _print_event("audit-lite", report=report)
    return report


def _select_validation_source1(config: PipelineConfig, limit: int) -> pd.DataFrame:
    selected: List[pd.DataFrame] = []
    count = 0
    path = config.train_dir / "train_source1.tsv"
    chunks = load_source(
        path, "source1", chunksize=config.chunk_size, validate=False
    )
    for chunk in chunks:
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
    if not selected:
        raise RuntimeError("Deterministic validation split selected no Source-1 entities")
    return pd.concat(selected, ignore_index=True).head(limit)


def _load_selected_truth(config: PipelineConfig, s1_ids: Set[str]) -> pd.DataFrame:
    selected: List[pd.DataFrame] = []
    path = config.train_dir / "train_ground_truth.tsv"
    chunks = load_ground_truth(path, chunksize=config.chunk_size, validate=False)
    for chunk in chunks:
        validate_ground_truth_frame(chunk, check_duplicates=False, path=path)
        part = chunk.loc[chunk["source1_entity_id"].isin(s1_ids)]
        if not part.empty:
            selected.append(part)
    result = (
        pd.concat(selected, ignore_index=True)
        if selected
        else pd.DataFrame(columns=GROUND_TRUTH_COLUMNS)
    )
    validate_ground_truth_frame(result, path=path)
    actual = set(result["source1_entity_id"])
    if actual != s1_ids:
        raise RuntimeError(
            f"Selected ground truth coverage mismatch: missing={len(s1_ids-actual)}, "
            f"extra={len(actual-s1_ids)}"
        )
    return result


def _load_feed_subset(
    config: PipelineConfig,
    source: str,
    required_ids: Set[str],
    distractor_limit: int,
) -> pd.DataFrame:
    number = source[-1]
    path = config.train_dir / f"train_source{number}.tsv"
    selected: List[pd.DataFrame] = []
    distractors = 0
    chunks = load_source(path, source, chunksize=config.chunk_size, validate=False)
    for chunk in chunks:
        validate_source_frame(chunk, source, check_duplicates=False, path=path)
        required = chunk.loc[chunk["entity_id"].isin(required_ids)]
        if not required.empty:
            selected.append(required)
        remaining = distractor_limit - distractors
        if remaining > 0:
            distractor = chunk.loc[~chunk["entity_id"].isin(required_ids)].head(remaining)
            if not distractor.empty:
                selected.append(distractor)
                distractors += len(distractor)
    result = pd.concat(selected, ignore_index=True).drop_duplicates("entity_id")
    found = set(result["entity_id"]) & required_ids
    if found != required_ids:
        raise RuntimeError(
            f"Could not load all selected truth targets from {source}: "
            f"missing={sorted(required_ids-found)[:5]}"
        )
    validate_source_frame(result, source, path=path)
    return result


def load_validation_universe(
    config: PipelineConfig,
    *,
    s1_limit: int,
    distractors_per_source: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
    """Build a bounded, self-contained S1 validation universe from training data."""

    started = time.perf_counter()
    source1 = _select_validation_source1(config, s1_limit)
    s1_ids = set(source1["entity_id"])
    truth_frame = _load_selected_truth(config, s1_ids)
    truth = ground_truth_to_mapping(truth_frame)
    required_s2 = {
        target for targets in truth.values() for target in targets if target.startswith("S2-")
    }
    required_s3 = {
        target for targets in truth.values() for target in targets if target.startswith("S3-")
    }
    source2 = _load_feed_subset(
        config, "source2", required_s2, distractors_per_source
    )
    source3 = _load_feed_subset(
        config, "source3", required_s3, distractors_per_source
    )
    _print_event(
        "validation-universe",
        source1_rows=len(source1),
        source2_rows=len(source2),
        source3_rows=len(source3),
        truth_pairs=sum(len(values) for values in truth.values()),
        countries=source1["country"].value_counts().to_dict(),
        runtime_seconds=time.perf_counter() - started,
    )
    return source1, source2, source3, truth_frame, truth


def _cache_artifacts(
    config: PipelineConfig,
    tag: str,
    source1: pd.DataFrame,
    feed: pd.DataFrame,
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    config.ensure_generated_dirs()
    common = {
        "config_fingerprint": config.fingerprint(),
        "tag": tag,
        **dict(metadata),
    }
    write_parquet_cache(
        source1, config.cache_dir / "normalized" / f"{tag}_source1.parquet", common
    )
    write_parquet_cache(
        feed, config.cache_dir / "normalized" / f"{tag}_feed.parquet", common
    )
    write_parquet_cache(
        candidates,
        config.cache_dir / "candidates" / f"{tag}_candidates.parquet",
        common,
    )
    write_parquet_cache(
        features, config.cache_dir / "features" / f"{tag}_features.parquet", common
    )


def run_validation_experiment(
    config: PipelineConfig,
    *,
    s1_limit: int,
    distractors_per_source: int,
    persist: bool,
    stop_after: str = "evaluate",
) -> Tuple[Dict[str, object], Dict[str, object]]:
    total_started = time.perf_counter()
    _print_event("effective-config", config=config.as_dict(), fingerprint=config.fingerprint())
    source1_raw, source2_raw, source3_raw, truth_frame, truth = load_validation_universe(
        config,
        s1_limit=s1_limit,
        distractors_per_source=distractors_per_source,
    )
    if stop_after == "split":
        report = {
            "stage": "split",
            "source1_rows": len(source1_raw),
            "truth_pairs": sum(map(len, truth.values())),
            "runtime_seconds": time.perf_counter() - total_started,
            "peak_rss_mib": _peak_rss_mib(),
        }
        return report, {"source1_raw": source1_raw, "truth": truth}

    normalize_started = time.perf_counter()
    source1 = normalize_records(source1_raw)
    source2 = normalize_records(source2_raw)
    source3 = normalize_records(source3_raw)
    feed = pd.concat([source2, source3], ignore_index=True)
    normalize_seconds = time.perf_counter() - normalize_started
    if stop_after == "normalize":
        report = {
            "stage": "normalize",
            "source1_rows": len(source1),
            "feed_rows": len(feed),
            "normalization_runtime_seconds": normalize_seconds,
            "runtime_seconds": time.perf_counter() - total_started,
            "peak_rss_mib": _peak_rss_mib(),
        }
        return report, {"source1": source1, "feed": feed, "truth": truth}

    candidates_max, blocking_metadata = generate_candidates(
        source1, feed, config, max_k=max(config.evaluation_ks)
    )
    s1_ids = source1["entity_id"].tolist()
    k_metrics: Dict[str, object] = {}
    for k in config.evaluation_ks:
        at_k = candidates_at_k(candidates_max, k)
        k_metrics[str(k)] = candidate_metrics(
            s1_ids,
            truth,
            at_k,
            runtime_seconds=blocking_metadata["runtime_seconds"],
        )
    max_recall = max(
        float(values["pair_candidate_recall"]) for values in k_metrics.values()
    )
    practical_ks = [
        k
        for k in config.evaluation_ks
        if float(k_metrics[str(k)]["pair_candidate_recall"]) >= max_recall - 0.0005
    ]
    selected_k = min(practical_ks)
    selected_candidates = candidates_at_k(candidates_max, selected_k)
    if stop_after == "block":
        report = {
            "stage": "block",
            "candidate_metrics": k_metrics,
            "selected_k": selected_k,
            "blocking": blocking_metadata,
            "runtime_seconds": time.perf_counter() - total_started,
            "peak_rss_mib": _peak_rss_mib(),
        }
        return report, {
            "source1": source1,
            "feed": feed,
            "truth": truth,
            "candidates": selected_candidates,
        }

    feature_started = time.perf_counter()
    features = build_feature_table(
        selected_candidates,
        source1,
        feed,
        chunk_size=config.chunk_size,
    )
    feature_seconds = time.perf_counter() - feature_started
    country_map = source1.set_index("entity_id")["country"].to_dict()
    labeled, label_metrics = build_pair_labels(
        features, truth, source1_country=country_map
    )
    if stop_after == "features":
        report = {
            "stage": "features",
            "candidate_metrics": k_metrics,
            "selected_k": selected_k,
            "labels": label_metrics,
            "feature_runtime_seconds": feature_seconds,
            "runtime_seconds": time.perf_counter() - total_started,
            "peak_rss_mib": _peak_rss_mib(),
        }
        return report, {
            "source1": source1,
            "feed": feed,
            "truth": truth,
            "candidates": selected_candidates,
            "features": labeled,
        }

    scored = score_rule_baseline(labeled)
    threshold, baseline_metrics, threshold_table = tune_rule_threshold(
        scored, s1_ids, truth, config.rule_thresholds
    )
    predictions = predict_from_scores(scored, threshold)
    breakdowns = evaluate_breakdowns(source1_raw, truth, predictions)
    diagnostics = error_diagnostics(
        s1_ids, truth, selected_candidates, scored, predictions
    )
    report: Dict[str, object] = {
        "stage": stop_after,
        "config": config.as_dict(),
        "config_fingerprint": config.fingerprint(),
        "subset": {
            "source1_rows": len(source1),
            "source2_rows": len(source2),
            "source3_rows": len(source3),
            "distractors_per_source_requested": distractors_per_source,
            "truth_pairs": sum(len(values) for values in truth.values()),
            "countries": source1_raw["country"].value_counts().to_dict(),
        },
        "candidate_metrics": k_metrics,
        "selected_k": selected_k,
        "blocking": blocking_metadata,
        "blocking_channel_recovery": blocker_recovery(candidates_max, truth),
        "labels": label_metrics,
        "selected_threshold": threshold,
        "threshold_results": threshold_table.to_dict(orient="records"),
        "baseline": baseline_metrics,
        "breakdowns": breakdowns,
        "diagnostics": diagnostics,
        "normalization_runtime_seconds": normalize_seconds,
        "feature_runtime_seconds": feature_seconds,
        "runtime_seconds": time.perf_counter() - total_started,
        "peak_rss_mib": _peak_rss_mib(),
    }
    tag = f"tier0_validation_s1_{len(source1)}_{config.fingerprint()}"
    if persist:
        _cache_artifacts(
            config,
            tag,
            source1,
            feed,
            candidates_max,
            scored,
            {"selected_k": selected_k, "selected_threshold": threshold},
        )
        config.experiments_dir.mkdir(parents=True, exist_ok=True)
        report_path = config.experiments_dir / f"{tag}_metrics.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        report["report_path"] = str(report_path)

    artifacts: Dict[str, object] = {
        "source1_raw": source1_raw,
        "source2_raw": source2_raw,
        "source3_raw": source3_raw,
        "source1": source1,
        "feed": feed,
        "truth": truth,
        "candidates": selected_candidates,
        "features": scored,
        "predictions": predictions,
        "selected_threshold": threshold,
    }
    return report, artifacts


def _pair_similarity_summary(left: pd.Series, right: pd.Series) -> Dict[str, float]:
    left_name = str(left["name_core"])
    right_name = str(right["name_core"])
    left_tokens = set(left_name.split())
    right_tokens = set(right_name.split())
    left_address = set(str(left["address_tokens"]).split())
    right_address = set(str(right["address_tokens"]).split())

    def jaccard(a: Set[str], b: Set[str]) -> float:
        return len(a & b) / len(a | b) if a or b else 1.0

    return {
        "name_character_similarity": SequenceMatcher(
            None, left_name, right_name, autojunk=False
        ).ratio(),
        "name_token_jaccard": jaccard(left_tokens, right_tokens),
        "address_token_jaccard": jaccard(left_address, right_address),
        "name_length_ratio": min(len(left_name), len(right_name))
        / max(len(left_name), len(right_name))
        if left_name or right_name
        else 1.0,
    }


def _failure_category(left: pd.Series, right: pd.Series) -> str:
    summary = _pair_similarity_summary(left, right)
    left_name = str(left["name_core"])
    right_name = str(right["name_core"])
    left_tokens = left_name.split()
    right_tokens = right_name.split()
    raw = f"{left['business_name']} {right['business_name']}".casefold()
    if any(marker in raw for marker in (" dba ", "d/b/a", "doing business as", " t/a ", " aka ", "trading as")):
        return "DBA/trade-name mismatch"
    if (
        summary["name_character_similarity"] < 0.35
        and summary["address_token_jaccard"] >= 0.40
    ):
        return "address-only evidence"
    left_non_ascii = any(ord(char) > 127 and char.isalpha() for char in left_name)
    right_non_ascii = any(ord(char) > 127 and char.isalpha() for char in right_name)
    if left_non_ascii != right_non_ascii:
        return "transliteration/script mismatch"
    if (
        summary["name_token_jaccard"] >= 0.80
        and left_tokens != right_tokens
        and sorted(left_tokens) == sorted(right_tokens)
    ):
        return "token reordering"
    left_acronym = "".join(token[0] for token in left_tokens if token)
    right_acronym = "".join(token[0] for token in right_tokens if token)
    if (
        left_name.replace(" ", "") == right_acronym
        or right_name.replace(" ", "") == left_acronym
        or (
            summary["name_length_ratio"] < 0.65
            and summary["name_character_similarity"] >= 0.35
        )
    ):
        return "abbreviation"
    if min(len(left_name), len(right_name)) <= 5 or (
        len(left_tokens) <= 1 and len(right_tokens) <= 1
    ):
        return "very short/generic name"
    if (
        summary["name_character_similarity"] < 0.65
        and summary["name_token_jaccard"] < 0.30
    ):
        return "severe spelling corruption"
    return "other"


def _blocked_out_analysis(
    source1: pd.DataFrame,
    feed: pd.DataFrame,
    truth: Mapping[str, Set[str]],
    base_k50: pd.DataFrame,
    tfidf_candidates: pd.DataFrame,
) -> Dict[str, object]:
    base_pairs = set(zip(base_k50.source1_entity_id, base_k50.candidate_entity_id))
    truth_pairs = {(s1, target) for s1, values in truth.items() for target in values}
    previously_blocked = truth_pairs - base_pairs
    tfidf_lookup = {
        (row.source1_entity_id, row.candidate_entity_id): float(row.name_tfidf_similarity)
        for row in tfidf_candidates.itertuples(index=False)
    }
    recovered = {pair for pair in previously_blocked if pair in tfidf_lookup}
    remaining = previously_blocked - recovered
    scores = np.array([tfidf_lookup[pair] for pair in recovered], dtype="float32")
    score_distribution = {
        "minimum": float(scores.min()),
        "p25": float(np.percentile(scores, 25)),
        "median": float(np.median(scores)),
        "p75": float(np.percentile(scores, 75)),
        "p95": float(np.percentile(scores, 95)),
        "maximum": float(scores.max()),
    } if len(scores) else {}

    left = source1.set_index("entity_id", verify_integrity=True)
    right = feed.set_index("entity_id", verify_integrity=True)
    recovered_examples = []
    for s1, target in sorted(recovered, key=lambda pair: tfidf_lookup[pair])[:10]:
        recovered_examples.append(
            {
                "source1_entity_id": s1,
                "candidate_entity_id": target,
                "source1_name": left.loc[s1, "business_name"],
                "candidate_name": right.loc[target, "business_name"],
                "tfidf_similarity": tfidf_lookup[(s1, target)],
            }
        )

    categories: Counter = Counter()
    remaining_examples = []
    for s1, target in sorted(remaining):
        category = _failure_category(left.loc[s1], right.loc[target])
        categories[category] += 1
        if len(remaining_examples) < 12:
            remaining_examples.append(
                {
                    "source1_entity_id": s1,
                    "candidate_entity_id": target,
                    "source1_name": left.loc[s1, "business_name"],
                    "candidate_name": right.loc[target, "business_name"],
                    "category": category,
                }
            )
    return {
        "previously_blocked_true_links": len(previously_blocked),
        "recovered_by_tfidf_top50": len(recovered),
        "recovered_percentage": 100.0 * len(recovered) / len(previously_blocked)
        if previously_blocked
        else 0.0,
        "recovered_similarity_distribution": score_distribution,
        "recovered_examples": recovered_examples,
        "remaining_unrecovered": len(remaining),
        "remaining_failure_categories": dict(categories),
        "remaining_examples": remaining_examples,
    }


def run_e1_experiment(
    config: PipelineConfig,
    *,
    s1_limit: int,
    distractors_per_source: int,
    persist: bool,
) -> Dict[str, object]:
    """Compare E0 blockers with E0 plus sparse name character TF-IDF."""

    total_started = time.perf_counter()
    evaluation_ks = (20, 30, 50, 75, 100)
    base_config = replace(
        config, use_name_tfidf=False, evaluation_ks=evaluation_ks
    )
    e1_config = replace(
        config,
        use_name_tfidf=True,
        evaluation_ks=evaluation_ks,
        candidate_k=50,
    )
    _print_event("effective-config-e1", config=e1_config.as_dict(), fingerprint=e1_config.fingerprint())
    source1_raw, source2_raw, source3_raw, _, truth = load_validation_universe(
        e1_config,
        s1_limit=s1_limit,
        distractors_per_source=distractors_per_source,
    )
    normalize_started = time.perf_counter()
    source1 = normalize_records(source1_raw)
    source2 = normalize_records(source2_raw)
    source3 = normalize_records(source3_raw)
    feed = pd.concat([source2, source3], ignore_index=True)
    normalization_seconds = time.perf_counter() - normalize_started
    s1_ids = source1["entity_id"].tolist()

    base_candidates, base_profile = generate_candidates(
        source1, feed, base_config, max_k=max(evaluation_ks)
    )
    tfidf_candidates, tfidf_profile = retrieve_name_tfidf(
        source1, feed, e1_config
    )
    improved_candidates, improved_profile = generate_candidates(
        source1,
        feed,
        e1_config,
        max_k=max(evaluation_ks),
        name_tfidf_candidates=tfidf_candidates,
        name_tfidf_profile=tfidf_profile,
    )

    existing_metrics: Dict[str, object] = {}
    improved_metrics: Dict[str, object] = {}
    for k in evaluation_ks:
        existing_metrics[str(k)] = candidate_metrics(
            s1_ids, truth, candidates_at_k(base_candidates, k),
            runtime_seconds=base_profile["runtime_seconds"],
        )
        improved_metrics[str(k)] = candidate_metrics(
            s1_ids, truth, candidates_at_k(improved_candidates, k),
            runtime_seconds=(
                float(tfidf_profile["total_runtime_seconds"])
                + float(improved_profile["runtime_seconds"])
            ),
        )

    tfidf_only_metrics: Dict[str, object] = {}
    for k in (10, 20, 30, 50):
        at_k = tfidf_candidates.loc[tfidf_candidates["name_tfidf_rank"] <= k]
        tfidf_only_metrics[str(k)] = candidate_metrics(s1_ids, truth, at_k)

    best_recall = max(
        float(values["pair_candidate_recall"])
        for values in improved_metrics.values()
    )
    practical = [
        k
        for k in evaluation_ks
        if float(improved_metrics[str(k)]["pair_candidate_recall"])
        >= best_recall - 0.002
    ]
    selected_k = min(practical)
    base_k50 = candidates_at_k(base_candidates, 50)
    improved_selected = candidates_at_k(improved_candidates, selected_k)

    feature_started = time.perf_counter()
    base_features = build_feature_table(
        base_k50, source1, feed, chunk_size=e1_config.chunk_size
    )
    improved_features = build_feature_table(
        improved_selected, source1, feed, chunk_size=e1_config.chunk_size
    )
    feature_seconds = time.perf_counter() - feature_started
    base_labeled, _ = build_pair_labels(base_features, truth)
    improved_labeled, improved_label_metrics = build_pair_labels(
        improved_features, truth
    )
    base_scored = score_rule_baseline(base_labeled)
    improved_scored = score_rule_baseline(improved_labeled)
    # Fix the E0 threshold at its previously selected value to isolate retrieval.
    e0_threshold = 0.70
    base_predictions = predict_from_scores(base_scored, e0_threshold)
    improved_predictions = predict_from_scores(improved_scored, e0_threshold)
    base_baseline = evaluate_predictions(s1_ids, truth, base_predictions)
    improved_baseline = evaluate_predictions(s1_ids, truth, improved_predictions)

    blocked_analysis = _blocked_out_analysis(
        source1, feed, truth, base_k50, tfidf_candidates
    )
    report: Dict[str, object] = {
        "experiment": "E1_name_character_tfidf",
        "config": e1_config.as_dict(),
        "config_fingerprint": e1_config.fingerprint(),
        "subset": {
            "source1_rows": len(source1),
            "source2_rows": len(source2),
            "source3_rows": len(source3),
            "truth_pairs": sum(map(len, truth.values())),
            "countries": source1_raw["country"].value_counts().to_dict(),
        },
        "existing_blocker_metrics": existing_metrics,
        "improved_blocker_metrics": improved_metrics,
        "tfidf_only_metrics": tfidf_only_metrics,
        "selected_k": selected_k,
        "selection_rule": "smallest K within 0.002 pair recall of best E1 K",
        "tfidf_profile": tfidf_profile,
        "existing_blocking_profile": base_profile,
        "improved_blocking_profile": improved_profile,
        "blocked_out_analysis": blocked_analysis,
        "e0_threshold_fixed": e0_threshold,
        "e0_before": base_baseline,
        "e0_after": improved_baseline,
        "e0_macro_f0_5_delta": float(improved_baseline["macro_f0_5"])
        - float(base_baseline["macro_f0_5"]),
        "improved_labels": improved_label_metrics,
        "normalization_runtime_seconds": normalization_seconds,
        "feature_runtime_seconds": feature_seconds,
        "runtime_seconds": time.perf_counter() - total_started,
        "peak_rss_mib": _peak_rss_mib(),
    }
    if persist:
        e1_config.ensure_generated_dirs()
        tag = f"e1_name_tfidf_s1_{len(source1)}_{e1_config.fingerprint()}"
        metadata = {
            "config_fingerprint": e1_config.fingerprint(),
            "selected_k": selected_k,
            "e0_threshold": e0_threshold,
        }
        write_parquet_cache(
            tfidf_candidates,
            e1_config.cache_dir / "candidates" / f"{tag}_tfidf_only.parquet",
            metadata,
        )
        write_parquet_cache(
            improved_candidates,
            e1_config.cache_dir / "candidates" / f"{tag}_union.parquet",
            metadata,
        )
        write_parquet_cache(
            improved_scored,
            e1_config.cache_dir / "features" / f"{tag}_features.parquet",
            metadata,
        )
        report_path = e1_config.experiments_dir / f"{tag}_metrics.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        report["report_path"] = str(report_path)
    return report


def _candidate_pair_set(frame: pd.DataFrame) -> Set[Tuple[str, str]]:
    return set(zip(frame["source1_entity_id"], frame["candidate_entity_id"]))


def _numeric_distribution(values: Sequence[float]) -> Dict[str, object]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype="float64")
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "minimum": float(finite.min()),
        "p25": float(np.percentile(finite, 25)),
        "median": float(np.median(finite)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
        "maximum": float(finite.max()),
    }


def _fusion_record(
    *,
    strategy: str,
    parameters: Mapping[str, object],
    tfidf_top_k: int,
    final_k: int,
    fused: pd.DataFrame,
    source1_ids: Sequence[str],
    truth: Mapping[str, Set[str]],
    legacy_true_pairs: Set[Tuple[str, str]],
    legacy_missed_pairs: Set[Tuple[str, str]],
    tfidf_pair_set: Set[Tuple[str, str]],
) -> Dict[str, object]:
    final_pairs = _candidate_pair_set(fused)
    retained_legacy = legacy_true_pairs & final_pairs
    displaced_legacy = legacy_true_pairs - final_pairs
    retrieved_failures = legacy_missed_pairs & tfidf_pair_set
    retained_failures = legacy_missed_pairs & final_pairs
    metrics = candidate_metrics(source1_ids, truth, fused)
    return {
        "strategy": strategy,
        "parameters": dict(parameters),
        "tfidf_top_k": tfidf_top_k,
        "final_k": final_k,
        "metrics": metrics,
        "legacy_true_candidates_at_k50": len(legacy_true_pairs),
        "legacy_true_candidates_retained": len(retained_legacy),
        "legacy_true_candidates_displaced": len(displaced_legacy),
        "legacy_true_candidate_retention_rate": (
            len(retained_legacy) / len(legacy_true_pairs) if legacy_true_pairs else 1.0
        ),
        "legacy_true_candidate_displacement_rate": (
            len(displaced_legacy) / len(legacy_true_pairs) if legacy_true_pairs else 0.0
        ),
        "original_legacy_failures": len(legacy_missed_pairs),
        "original_failures_retrieved_by_tfidf": len(retrieved_failures),
        "original_failures_retained_after_fusion": len(retained_failures),
        "tfidf_retrieved_failures_lost_during_fusion": len(
            retrieved_failures - final_pairs
        ),
    }


def _fusion_sort_key(record: Mapping[str, object]) -> Tuple[float, ...]:
    metrics = record["metrics"]
    return (
        float(metrics["pair_candidate_recall"]),
        float(metrics["oracle_macro_f0_5"]),
        float(record["legacy_true_candidate_retention_rate"]),
        -float(record["final_k"]),
        -float(record["tfidf_top_k"]),
    )


def _apply_fusion_record(
    universes: Mapping[int, pd.DataFrame], record: Mapping[str, object]
) -> pd.DataFrame:
    universe = universes[int(record["tfidf_top_k"])]
    parameters = dict(record["parameters"])
    final_k = int(record["final_k"])
    strategy = str(record["strategy"])
    if strategy == "quota":
        return quota_fusion(universe, final_k=final_k, **parameters)
    if strategy == "rrf":
        return reciprocal_rank_fusion(universe, final_k=final_k, **parameters)
    if strategy == "normalized_score":
        return normalized_score_fusion(universe, final_k=final_k, **parameters)
    if strategy == "hybrid":
        return hybrid_quota_rank_fusion(universe, final_k=final_k, **parameters)
    raise ValueError(f"Unknown fusion strategy: {strategy}")


def _naive_displacement_analysis(
    *,
    source1: pd.DataFrame,
    feed: pd.DataFrame,
    universe: pd.DataFrame,
    naive_k50: pd.DataFrame,
    naive_k100: pd.DataFrame,
    legacy_true_pairs: Set[Tuple[str, str]],
    selected_fused: pd.DataFrame,
) -> Dict[str, object]:
    naive_pairs = _candidate_pair_set(naive_k50)
    displaced = legacy_true_pairs - naive_pairs
    selected_pairs = _candidate_pair_set(selected_fused)
    lookup = universe.set_index(
        ["source1_entity_id", "candidate_entity_id"], verify_integrity=True
    )
    rank100 = naive_k100.set_index(
        ["source1_entity_id", "candidate_entity_id"], verify_integrity=True
    )["fusion_rank"]
    legacy_ranks: List[float] = []
    legacy_scores: List[float] = []
    tfidf_scores: List[float] = []
    blocker_counts: List[float] = []
    final_ranks: List[float] = []
    left = source1.set_index("entity_id", verify_integrity=True)
    right = feed.set_index("entity_id", verify_integrity=True)
    examples: List[Dict[str, object]] = []
    for pair in sorted(displaced):
        row = lookup.loc[pair]
        legacy_ranks.append(float(row["legacy_rank"]))
        legacy_scores.append(float(row["legacy_cheap_score"]))
        tfidf_scores.append(float(row["name_tfidf_similarity"]))
        blocker_counts.append(float(row["legacy_blocking_channel_count"]))
        final_rank = float(rank100.get(pair, np.nan))
        final_ranks.append(final_rank)
        if len(examples) < 12:
            examples.append(
                {
                    "source1_entity_id": pair[0],
                    "candidate_entity_id": pair[1],
                    "source1_name": left.loc[pair[0], "business_name"],
                    "candidate_name": right.loc[pair[1], "business_name"],
                    "legacy_rank": int(row["legacy_rank"]),
                    "legacy_score": float(row["legacy_cheap_score"]),
                    "tfidf_rank": int(row["name_tfidf_rank"]),
                    "tfidf_similarity": float(row["name_tfidf_similarity"]),
                    "legacy_blocking_channels": int(
                        row["legacy_blocking_channel_count"]
                    ),
                    "naive_final_rank": int(final_rank)
                    if np.isfinite(final_rank)
                    else None,
                    "retained_by_selected_fusion": pair in selected_pairs,
                }
            )
    return {
        "naive_k50_legacy_true_candidates_displaced": len(displaced),
        "recovered_by_selected_fusion": len(displaced & selected_pairs),
        "still_displaced_by_selected_fusion": len(displaced - selected_pairs),
        "legacy_rank_distribution": _numeric_distribution(legacy_ranks),
        "legacy_score_distribution": _numeric_distribution(legacy_scores),
        "tfidf_score_distribution": _numeric_distribution(tfidf_scores),
        "legacy_blocker_count_distribution": _numeric_distribution(blocker_counts),
        "naive_final_rank_distribution": _numeric_distribution(final_ranks),
        "examples": examples,
    }


def _count_tsv_data_rows(path: Path) -> int:
    lines = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            lines += block.count(b"\n")
    return max(0, lines - 1)


def run_e11_experiment(
    config: PipelineConfig,
    *,
    s1_limit: int,
    distractors_per_source: int,
    persist: bool,
) -> Dict[str, object]:
    """Optimize legacy/TF-IDF fusion on the fixed bounded E1 universe."""

    total_started = time.perf_counter()
    base_config = replace(config, use_name_tfidf=False, evaluation_ks=(30, 40, 50))
    e11_config = replace(config, use_name_tfidf=True, tfidf_name_top_k=50)
    _print_event(
        "effective-config-e1.1",
        config=e11_config.as_dict(),
        fingerprint=e11_config.fingerprint(),
    )
    source1_raw, source2_raw, source3_raw, _, truth = load_validation_universe(
        e11_config,
        s1_limit=s1_limit,
        distractors_per_source=distractors_per_source,
    )
    normalization_started = time.perf_counter()
    source1 = normalize_records(source1_raw)
    source2 = normalize_records(source2_raw)
    source3 = normalize_records(source3_raw)
    feed = pd.concat([source2, source3], ignore_index=True)
    normalization_seconds = time.perf_counter() - normalization_started
    source1_ids = source1["entity_id"].tolist()
    truth_pairs = {
        (source1_id, target)
        for source1_id, targets in truth.items()
        for target in targets
    }

    blocking_started = time.perf_counter()
    legacy_max, blocking_profile = generate_candidates(
        source1, feed, base_config, max_k=100
    )
    blocking_seconds = time.perf_counter() - blocking_started
    legacy_k50 = candidates_at_k(legacy_max, 50)
    tfidf_started = time.perf_counter()
    tfidf_candidates, tfidf_profile = retrieve_name_tfidf(
        source1, feed, e11_config
    )
    tfidf_seconds = time.perf_counter() - tfidf_started

    legacy_pairs = _candidate_pair_set(legacy_k50)
    legacy_true_pairs = truth_pairs & legacy_pairs
    legacy_missed_pairs = truth_pairs - legacy_pairs
    legacy_metrics = {
        str(k): candidate_metrics(
            source1_ids, truth, candidates_at_k(legacy_k50, k)
        )
        for k in (30, 40, 50)
    }

    universes: Dict[int, pd.DataFrame] = {}
    tfidf_sets: Dict[int, Set[Tuple[str, str]]] = {}
    for top_k in e11_config.fusion_tfidf_top_ks:
        universes[top_k] = prepare_fusion_universe(
            legacy_k50, tfidf_candidates, tfidf_top_k=top_k
        )
        tfidf_sets[top_k] = _candidate_pair_set(
            tfidf_candidates.loc[tfidf_candidates["name_tfidf_rank"] <= top_k]
        )
    # The original E1 global score ranked the complete legacy union before its
    # final cap. Keep a wider legacy view solely for an exact reference and
    # displacement analysis; optimized fusion still protects Tier 0 K=50.
    naive_universe = prepare_fusion_universe(
        legacy_max, tfidf_candidates, tfidf_top_k=50
    )

    fusion_started = time.perf_counter()
    records: List[Dict[str, object]] = []

    def evaluate(
        strategy: str,
        parameters: Mapping[str, object],
        top_k: int,
        final_k: int,
        fused: pd.DataFrame,
    ) -> None:
        records.append(
            _fusion_record(
                strategy=strategy,
                parameters=parameters,
                tfidf_top_k=top_k,
                final_k=final_k,
                fused=fused,
                source1_ids=source1_ids,
                truth=truth,
                legacy_true_pairs=legacy_true_pairs,
                legacy_missed_pairs=legacy_missed_pairs,
                tfidf_pair_set=tfidf_sets[top_k],
            )
        )

    for top_k, universe in universes.items():
        for final_k, legacy_quota, tfidf_quota in e11_config.fusion_quota_configs:
            parameters = {
                "legacy_quota": legacy_quota,
                "tfidf_addition_quota": tfidf_quota,
            }
            evaluate(
                "quota",
                parameters,
                top_k,
                final_k,
                quota_fusion(universe, final_k=final_k, **parameters),
            )
        for legacy_weight, tfidf_weight, constant in e11_config.fusion_rrf_configs:
            parameters = {
                "legacy_weight": legacy_weight,
                "tfidf_weight": tfidf_weight,
                "rrf_constant": constant,
            }
            ranked = reciprocal_rank_fusion(
                universe, final_k=max(e11_config.fusion_final_ks), **parameters
            )
            for final_k in e11_config.fusion_final_ks:
                evaluate(
                    "rrf",
                    parameters,
                    top_k,
                    final_k,
                    candidates_at_k(ranked, final_k),
                )
        for alpha, beta, bonus in e11_config.fusion_normalized_configs:
            parameters = {
                "legacy_alpha": alpha,
                "tfidf_beta": beta,
                "provenance_bonus": bonus,
            }
            ranked = normalized_score_fusion(
                universe, final_k=max(e11_config.fusion_final_ks), **parameters
            )
            for final_k in e11_config.fusion_final_ks:
                evaluate(
                    "normalized_score",
                    parameters,
                    top_k,
                    final_k,
                    candidates_at_k(ranked, final_k),
                )
        for final_k, minimum_legacy in e11_config.fusion_hybrid_minimum_legacy:
            parameters = {
                "minimum_legacy": minimum_legacy,
                "legacy_weight": 3.0,
                "tfidf_weight": 1.0,
                "rrf_constant": 60.0,
            }
            evaluate(
                "hybrid",
                parameters,
                top_k,
                final_k,
                hybrid_quota_rank_fusion(
                    universe, final_k=final_k, **parameters
                ),
            )
    fusion_sweep_seconds = time.perf_counter() - fusion_started

    best_by_strategy = {
        strategy: max(
            (
                record
                for record in records
                if record["strategy"] == strategy and int(record["final_k"]) <= 50
            ),
            key=_fusion_sort_key,
        )
        for strategy in ("quota", "rrf", "normalized_score", "hybrid")
    }
    best_by_final_k = {
        str(final_k): max(
            (record for record in records if int(record["final_k"]) == final_k),
            key=_fusion_sort_key,
        )
        for final_k in e11_config.fusion_final_ks
    }
    selected_record = max(
        (record for record in records if int(record["final_k"]) <= 50),
        key=_fusion_sort_key,
    )
    selected_fusion_started = time.perf_counter()
    selected_fused = _apply_fusion_record(universes, selected_record)
    selected_fusion_seconds = time.perf_counter() - selected_fusion_started

    naive_k50 = naive_score_fusion(
        naive_universe,
        final_k=50,
        tfidf_weight=e11_config.weight_name_tfidf,
        same_country_weight=e11_config.weight_same_country,
    )
    naive_k100 = naive_score_fusion(
        naive_universe,
        final_k=100,
        tfidf_weight=e11_config.weight_name_tfidf,
        same_country_weight=e11_config.weight_same_country,
    )
    naive_metrics = {
        "50": candidate_metrics(source1_ids, truth, naive_k50),
        "100": candidate_metrics(source1_ids, truth, naive_k100),
    }
    displacement_analysis = _naive_displacement_analysis(
        source1=source1,
        feed=feed,
        universe=naive_universe,
        naive_k50=naive_k50,
        naive_k100=naive_k100,
        legacy_true_pairs=legacy_true_pairs,
        selected_fused=selected_fused,
    )

    feature_started = time.perf_counter()
    legacy_features = build_feature_table(
        legacy_k50, source1, feed, chunk_size=e11_config.chunk_size
    )
    fused_features = build_feature_table(
        selected_fused, source1, feed, chunk_size=e11_config.chunk_size
    )
    feature_seconds = time.perf_counter() - feature_started
    legacy_labeled, _ = build_pair_labels(legacy_features, truth)
    fused_labeled, fused_label_metrics = build_pair_labels(fused_features, truth)
    legacy_scored = score_rule_baseline(legacy_labeled)
    fused_scored = score_rule_baseline(fused_labeled)
    e0_threshold = 0.70
    e0_before_predictions = predict_from_scores(legacy_scored, e0_threshold)
    e0_after_predictions = predict_from_scores(fused_scored, e0_threshold)
    e0_before = evaluate_predictions(source1_ids, truth, e0_before_predictions)
    e0_after = evaluate_predictions(source1_ids, truth, e0_after_predictions)

    test_s1_rows = _count_tsv_data_rows(e11_config.test_dir / "test_source1.tsv")
    selected_metrics = selected_record["metrics"]
    projected_test_pairs = {
        "test_source1_rows": test_s1_rows,
        "at_observed_average_candidate_count": int(
            round(test_s1_rows * float(selected_metrics["average_candidate_count"]))
        ),
        "hard_cap_upper_bound": test_s1_rows * int(selected_record["final_k"]),
    }
    report: Dict[str, object] = {
        "experiment": "E1.1_candidate_fusion",
        "config": e11_config.as_dict(),
        "config_fingerprint": e11_config.fingerprint(),
        "subset": {
            "source1_rows": len(source1),
            "source2_rows": len(source2),
            "source3_rows": len(source3),
            "truth_pairs": len(truth_pairs),
            "countries": source1_raw["country"].value_counts().to_dict(),
        },
        "legacy_metrics": legacy_metrics,
        "naive_e1_reference_metrics": naive_metrics,
        "evaluated_configurations": records,
        "best_by_strategy_at_k_lte_50": best_by_strategy,
        "best_by_final_k": best_by_final_k,
        "selected_configuration": selected_record,
        "selection_rule": (
            "maximum pair recall among final K <= 50; ties use oracle F0.5, "
            "legacy retention, then smaller final and TF-IDF K"
        ),
        "naive_k50_displacement_analysis": displacement_analysis,
        "e0_threshold_fixed": e0_threshold,
        "e0_before": e0_before,
        "e0_after": e0_after,
        "e0_macro_f0_5_delta": float(e0_after["macro_f0_5"])
        - float(e0_before["macro_f0_5"]),
        "fused_labels": fused_label_metrics,
        "projected_test_candidate_pairs": projected_test_pairs,
        "performance": {
            "normalization_seconds": normalization_seconds,
            "existing_blocking_seconds": blocking_seconds,
            "existing_blocking_profile": blocking_profile,
            "tfidf_retrieval_seconds": tfidf_seconds,
            "tfidf_profile": tfidf_profile,
            "fusion_sweep_seconds": fusion_sweep_seconds,
            "selected_fusion_seconds": selected_fusion_seconds,
            "feature_generation_seconds": feature_seconds,
            "total_experiment_seconds": time.perf_counter() - total_started,
            "peak_rss_mib": _peak_rss_mib(),
        },
    }
    if persist:
        e11_config.ensure_generated_dirs()
        tag = f"e11_candidate_fusion_s1_{len(source1)}_{e11_config.fingerprint()}"
        metadata = {
            "config_fingerprint": e11_config.fingerprint(),
            "selected_configuration": selected_record,
            "e0_threshold": e0_threshold,
        }
        write_parquet_cache(
            selected_fused,
            e11_config.cache_dir / "candidates" / f"{tag}_candidates.parquet",
            metadata,
        )
        write_parquet_cache(
            fused_scored,
            e11_config.cache_dir / "features" / f"{tag}_features.parquet",
            metadata,
        )
        report_path = e11_config.experiments_dir / f"{tag}_metrics.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        report["report_path"] = str(report_path)
    return report


def run_smoke(config: PipelineConfig) -> Dict[str, object]:
    report, artifacts = run_validation_experiment(
        config,
        s1_limit=50,
        distractors_per_source=500,
        persist=False,
        stop_after="evaluate",
    )
    source1 = artifacts["source1_raw"]
    source2 = artifacts["source2_raw"]
    source3 = artifacts["source3_raw"]
    candidates = artifacts["candidates"]
    predictions = artifacts["predictions"]
    matching, candidate = build_submission_frames(
        source1["entity_id"], candidates, predictions
    )
    valid_targets = set(source2["entity_id"]) | set(source3["entity_id"])
    validate_submission_frames(
        matching,
        candidate,
        source1["entity_id"],
        valid_target_ids=valid_targets,
    )
    with tempfile.TemporaryDirectory(prefix="ber_smoke_") as temp:
        temp_path = Path(temp)
        test_dir = temp_path / "dataset" / "test"
        output_dir = temp_path / "output"
        test_dir.mkdir(parents=True)
        source1[SOURCE_COLUMNS].to_csv(
            test_dir / "test_source1.tsv", sep="\t", index=False
        )
        source2[SOURCE_COLUMNS].to_csv(
            test_dir / "test_source2.tsv", sep="\t", index=False
        )
        source3[SOURCE_COLUMNS].to_csv(
            test_dir / "test_source3.tsv", sep="\t", index=False
        )
        matching_path, candidate_path = write_submission(matching, candidate, output_dir)
        completed = subprocess.run(
            [
                sys.executable,
                str(config.validator_path),
                "--matching",
                str(matching_path),
                "--candidate",
                str(candidate_path),
                "--test-dir",
                str(test_dir),
                "--check-ids",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Official validator failed smoke output:\n{completed.stdout}\n{completed.stderr}"
            )
        report["official_validator"] = completed.stdout.strip()
    report["smoke_passed"] = True
    return report


def validate_existing_output(config: PipelineConfig, check_ids: bool) -> int:
    command = [
        sys.executable,
        str(config.validator_path),
        "--matching",
        str(config.output_dir / "matching_results.tsv"),
        "--candidate",
        str(config.output_dir / "candidate_pairs.tsv"),
        "--test-dir",
        str(config.test_dir),
    ]
    if check_ids:
        command.append("--check-ids")
    return subprocess.run(command, check=False).returncode


def submit_from_caches(config: PipelineConfig, confirmed: bool) -> None:
    if not confirmed:
        raise RuntimeError(
            "Full test output is guarded. Re-run with --confirm-full-test only after "
            "validation is accepted and test caches have been generated."
        )
    raise RuntimeError(
        "Tier 0 validation is complete before full test cache generation by design. "
        "Generate reviewed test candidate/feature caches in the next authorized run."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "audit-lite",
            "split",
            "normalize",
            "block",
            "features",
            "baseline",
            "evaluate",
            "e1",
            "e1.1",
            "e2",
            "e2.1",
            "e2.2",
            "e2.3",
            "submit",
            "validate",
            "smoke",
        ],
    )
    parser.add_argument("--validation-s1-limit", type=int, default=None)
    parser.add_argument("--distractors-per-source", type=int, default=None)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--use-name-tfidf", action="store_true")
    parser.add_argument("--tfidf-name-top-k", type=int, default=None)
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--check-ids", action="store_true")
    parser.add_argument("--confirm-full-test", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = DEFAULT_CONFIG
    if args.candidate_k is not None:
        config = replace(config, candidate_k=args.candidate_k)
    if args.use_name_tfidf:
        config = replace(config, use_name_tfidf=True)
    if args.tfidf_name_top_k is not None:
        config = replace(config, tfidf_name_top_k=args.tfidf_name_top_k)
    s1_limit = args.validation_s1_limit or config.validation_s1_limit
    distractors = (
        args.distractors_per_source or config.validation_distractors_per_source
    )
    if args.stage == "audit-lite":
        run_audit_lite(config)
        return 0
    if args.stage == "smoke":
        report = run_smoke(config)
        _print_event("smoke-complete", report=report)
        return 0
    if args.stage == "validate":
        return validate_existing_output(config, args.check_ids)
    if args.stage == "submit":
        submit_from_caches(config, args.confirm_full_test)
        return 0
    if args.stage == "e1":
        report = run_e1_experiment(
            config,
            s1_limit=s1_limit,
            distractors_per_source=distractors,
            persist=args.persist,
        )
        _print_event("e1-complete", report=report)
        return 0
    if args.stage == "e1.1":
        report = run_e11_experiment(
            config,
            s1_limit=s1_limit,
            distractors_per_source=distractors,
            persist=args.persist,
        )
        _print_event("e1.1-complete", report=report)
        return 0
    if args.stage == "e2":
        # Lazy import keeps Tier 0/E1 usable in environments where the optional
        # E2 dependency has not been installed.
        from .train_lgbm import run_e2_experiment

        report = run_e2_experiment(config, persist=args.persist)
        _print_event("e2-complete", report=report)
        return 0
    if args.stage == "e2.1":
        from .stress_test import run_e21_experiment

        report = run_e21_experiment(config, persist=args.persist)
        _print_event(
            "e2.1-report",
            artifact_id=report["artifact_id"],
            report_path=report.get("artifacts", {}).get("report"),
        )
        return 0
    if args.stage == "e2.2":
        from .retrieval_repair import run_e22_experiment

        report = run_e22_experiment(config, persist=args.persist)
        _print_event(
            "e2.2-report",
            artifact_id=report["artifact_id"],
            report_path=report.get("artifacts", {}).get("report"),
        )
        return 0
    if args.stage == "e2.3":
        from .retrieval_optimization import run_e23_experiment

        report = run_e23_experiment(config, persist=args.persist)
        _print_event(
            "e2.3-report",
            artifact_id=report["artifact_id"],
            report_path=report.get("artifacts", {}).get("report"),
        )
        return 0

    stop_after = "evaluate" if args.stage in {"baseline", "evaluate"} else args.stage
    report, _ = run_validation_experiment(
        config,
        s1_limit=s1_limit,
        distractors_per_source=distractors,
        persist=args.persist,
        stop_after=stop_after,
    )
    _print_event("stage-complete", report=report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
