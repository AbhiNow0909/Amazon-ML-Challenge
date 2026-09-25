"""Construct realistic pair labels from blocked candidates."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Mapping, Set, Tuple

import pandas as pd


def build_pair_labels(
    features: pd.DataFrame,
    truth: Mapping[str, Set[str]],
    *,
    source1_country: Mapping[str, str] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Label blocked pairs; blocking-generated nonmatches are the negatives."""

    required = {"source1_entity_id", "candidate_entity_id"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Feature table is missing identifiers: {sorted(missing)}")
    result = features.copy()
    result["label"] = [
        int(target_id in truth.get(s1_id, set()))
        for s1_id, target_id in zip(
            result["source1_entity_id"], result["candidate_entity_id"]
        )
    ]
    result["label"] = result["label"].astype("uint8")

    all_truth = {(s1, target) for s1, targets in truth.items() for target in targets}
    recovered = set(
        zip(
            result.loc[result["label"] == 1, "source1_entity_id"],
            result.loc[result["label"] == 1, "candidate_entity_id"],
        )
    )
    positives_by_source = Counter(target[:2] for _, target in recovered)
    positives_by_country = Counter()
    if source1_country is not None:
        positives_by_country.update(source1_country.get(s1, "") for s1, _ in recovered)

    positives = int(result["label"].sum())
    negatives = len(result) - positives
    metrics: Dict[str, object] = {
        "total_candidate_pairs": len(result),
        "recovered_positives": positives,
        "missed_positives": len(all_truth - recovered),
        "negatives": negatives,
        "positive_negative_ratio": positives / negatives if negatives else None,
        "positives_by_source": dict(positives_by_source),
        "positives_by_country": dict(positives_by_country),
    }
    return result, metrics

