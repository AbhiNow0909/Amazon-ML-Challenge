"""Exact entity-level F0.5 and candidate diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .blocking import PROVENANCE_COLUMNS


def entity_metrics(truth: Set[str], prediction: Set[str]) -> Dict[str, float]:
    """Calculate one entity's metrics using the challenge singleton convention."""

    if not truth and not prediction:
        return {"precision": 1.0, "recall": 1.0, "f0_5": 1.0}
    if not truth or not prediction:
        return {"precision": 0.0, "recall": 0.0, "f0_5": 0.0}
    true_positive = len(truth & prediction)
    precision = true_positive / len(prediction)
    recall = true_positive / len(truth)
    denominator = 0.25 * precision + recall
    f0_5 = 1.25 * precision * recall / denominator if denominator else 0.0
    return {"precision": precision, "recall": recall, "f0_5": f0_5}


def pairs_to_mapping(
    pairs: pd.DataFrame,
    *,
    selected_column: Optional[str] = None,
) -> Dict[str, Set[str]]:
    if selected_column is not None:
        pairs = pairs.loc[pairs[selected_column].astype(bool)]
    mapping: Dict[str, Set[str]] = defaultdict(set)
    for s1, target in zip(pairs["source1_entity_id"], pairs["candidate_entity_id"]):
        mapping[s1].add(target)
    return dict(mapping)


def evaluate_predictions(
    source1_ids: Iterable[str],
    truth: Mapping[str, Set[str]],
    predictions: Mapping[str, Set[str]],
) -> Dict[str, object]:
    ids = list(source1_ids)
    per_entity = [
        entity_metrics(set(truth.get(s1, set())), set(predictions.get(s1, set())))
        for s1 in ids
    ]
    truth_pairs = {(s1, target) for s1 in ids for target in truth.get(s1, set())}
    predicted_pairs = {
        (s1, target) for s1 in ids for target in predictions.get(s1, set())
    }
    true_positive = len(truth_pairs & predicted_pairs)
    micro_precision = true_positive / len(predicted_pairs) if predicted_pairs else 0.0
    micro_recall = true_positive / len(truth_pairs) if truth_pairs else 1.0
    predicted_counts = [len(predictions.get(s1, set())) for s1 in ids]
    return {
        "entities": len(ids),
        "macro_f0_5": float(np.mean([row["f0_5"] for row in per_entity]))
        if ids
        else 0.0,
        "entity_macro_precision": float(
            np.mean([row["precision"] for row in per_entity])
        )
        if ids
        else 0.0,
        "entity_macro_recall": float(np.mean([row["recall"] for row in per_entity]))
        if ids
        else 0.0,
        "pair_micro_precision": micro_precision,
        "pair_micro_recall": micro_recall,
        "true_pairs": len(truth_pairs),
        "predicted_pairs": len(predicted_pairs),
        "true_positive_pairs": true_positive,
        "empty_predictions": sum(count == 0 for count in predicted_counts),
        "nonempty_predictions": sum(count > 0 for count in predicted_counts),
        "average_predicted_matches": float(np.mean(predicted_counts)) if ids else 0.0,
    }


def evaluate_breakdowns(
    source1: pd.DataFrame,
    truth: Mapping[str, Set[str]],
    predictions: Mapping[str, Set[str]],
) -> Dict[str, object]:
    by_country: Dict[str, object] = {}
    for country, group in source1.groupby("country", dropna=False, sort=True):
        by_country[str(country)] = evaluate_predictions(
            group["entity_id"], truth, predictions
        )

    buckets: Dict[str, List[str]] = {"0": [], "1": [], "2": [], "3+": []}
    for s1 in source1["entity_id"]:
        count = len(truth.get(s1, set()))
        bucket = str(count) if count < 3 else "3+"
        buckets[bucket].append(s1)
    by_truth_count = {
        bucket: evaluate_predictions(ids, truth, predictions)
        for bucket, ids in buckets.items()
        if ids
    }
    return {"by_country": by_country, "by_true_match_count": by_truth_count}


def candidate_metrics(
    source1_ids: Sequence[str],
    truth: Mapping[str, Set[str]],
    candidates: pd.DataFrame,
    *,
    runtime_seconds: Optional[float] = None,
) -> Dict[str, object]:
    candidate_map = pairs_to_mapping(candidates)
    total_true_pairs = sum(len(truth.get(s1, set())) for s1 in source1_ids)
    recovered_pairs = sum(
        len(set(truth.get(s1, set())) & candidate_map.get(s1, set()))
        for s1 in source1_ids
    )
    entity_recalls: List[float] = []
    oracle: Dict[str, Set[str]] = {}
    for s1 in source1_ids:
        targets = set(truth.get(s1, set()))
        recovered = targets & candidate_map.get(s1, set())
        oracle[s1] = recovered
        entity_recalls.append(len(recovered) / len(targets) if targets else 1.0)

    counts = np.array([len(candidate_map.get(s1, set())) for s1 in source1_ids])
    oracle_metrics = evaluate_predictions(source1_ids, truth, oracle)
    result: Dict[str, object] = {
        "pair_candidate_recall": recovered_pairs / total_true_pairs
        if total_true_pairs
        else 1.0,
        "entity_candidate_recall": float(np.mean(entity_recalls))
        if entity_recalls
        else 1.0,
        "oracle_macro_f0_5": oracle_metrics["macro_f0_5"],
        "true_pairs": total_true_pairs,
        "recovered_true_pairs": recovered_pairs,
        "missed_true_pairs": total_true_pairs - recovered_pairs,
        "total_candidate_pairs": int(counts.sum()),
        "average_candidate_count": float(counts.mean()) if len(counts) else 0.0,
        "median_candidate_count": float(np.median(counts)) if len(counts) else 0.0,
        "p95_candidate_count": float(np.percentile(counts, 95)) if len(counts) else 0.0,
        "maximum_candidate_count": int(counts.max()) if len(counts) else 0,
    }
    if runtime_seconds is not None:
        result["runtime_seconds"] = runtime_seconds
    return result


def blocker_recovery(
    candidates: pd.DataFrame,
    truth: Mapping[str, Set[str]],
) -> Dict[str, object]:
    total = sum(len(values) for values in truth.values())
    result: Dict[str, object] = {}
    for column in PROVENANCE_COLUMNS:
        pairs = candidates.loc[candidates[column].astype(bool)]
        recovered = sum(
            target in truth.get(s1, set())
            for s1, target in zip(
                pairs["source1_entity_id"], pairs["candidate_entity_id"]
            )
        )
        result[column] = {
            "recovered_true_pairs": int(recovered),
            "recall": recovered / total if total else 1.0,
        }
    return result


def error_diagnostics(
    source1_ids: Sequence[str],
    truth: Mapping[str, Set[str]],
    candidates: pd.DataFrame,
    scored_features: pd.DataFrame,
    predictions: Mapping[str, Set[str]],
) -> Dict[str, object]:
    candidate_pairs = set(
        zip(candidates["source1_entity_id"], candidates["candidate_entity_id"])
    )
    blocked_out = []
    rejected = []
    for s1 in source1_ids:
        for target in truth.get(s1, set()):
            pair = (s1, target)
            if pair not in candidate_pairs:
                blocked_out.append(pair)
            elif target not in predictions.get(s1, set()):
                rejected.append(pair)

    predicted_pairs = {
        (s1, target) for s1 in source1_ids for target in predictions.get(s1, set())
    }
    truth_pairs = {(s1, target) for s1 in source1_ids for target in truth.get(s1, set())}
    false_pairs = predicted_pairs - truth_pairs
    lookup = scored_features.set_index(
        ["source1_entity_id", "candidate_entity_id"], verify_integrity=True
    )
    patterns = Counter()
    for pair in false_pairs:
        row = lookup.loc[pair]
        if row["name_full_exact"] and not row["address_exact"]:
            patterns["exact name, different address"] += 1
        elif row["numeric_token_conflict"]:
            patterns["address-number conflict"] += 1
        elif row["address_exact"] and not row["name_core_exact"]:
            patterns["exact address, different name"] += 1
        else:
            patterns["other similarity collision"] += 1
    return {
        "blocked_out_true_pairs": len(blocked_out),
        "blocked_out_examples": [list(pair) for pair in blocked_out[:10]],
        "candidate_rejected_true_pairs": len(rejected),
        "candidate_rejected_examples": [list(pair) for pair in rejected[:10]],
        "false_positive_patterns": dict(patterns),
    }

