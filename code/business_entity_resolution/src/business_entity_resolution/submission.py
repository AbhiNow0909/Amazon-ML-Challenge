"""Build and strictly validate the two challenge submission TSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

from .io_utils import DataValidationError


MATCHING_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]


def _candidate_lists(candidates: pd.DataFrame) -> Dict[str, Sequence[str]]:
    if candidates.empty:
        return {}
    ordered = candidates.sort_values(
        ["source1_entity_id", "cheap_rank", "candidate_entity_id"], kind="mergesort"
    )
    return (
        ordered.groupby("source1_entity_id", sort=False)["candidate_entity_id"]
        .apply(list)
        .to_dict()
    )


def build_submission_frames(
    source1_ids: Iterable[str],
    candidates: pd.DataFrame,
    predictions: Mapping[str, Set[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ids = list(source1_ids)
    if len(ids) != len(set(ids)):
        raise DataValidationError("Duplicate Source-1 IDs supplied to submission builder")
    candidate_lists = _candidate_lists(candidates)
    matching_rows = []
    candidate_rows = []
    for s1 in ids:
        candidate_values = list(candidate_lists.get(s1, []))
        candidate_set = set(candidate_values)
        matched_set = set(predictions.get(s1, set()))
        missing = matched_set - candidate_set
        if missing:
            raise DataValidationError(
                f"Predictions for {s1} are absent from its candidate list: {sorted(missing)[:5]}"
            )
        matched_values = [value for value in candidate_values if value in matched_set]
        matching_rows.append(
            {
                "source1_entity_id": s1,
                "matched_entity_ids": ",".join(matched_values),
            }
        )
        candidate_rows.append(
            {
                "source1_entity_id": s1,
                "candidate_entity_ids": ",".join(candidate_values),
            }
        )
    return (
        pd.DataFrame(matching_rows, columns=MATCHING_COLUMNS),
        pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS),
    )


def _parse_id_list(raw: str) -> Sequence[str]:
    return str(raw).split(",") if str(raw) else []


def validate_submission_frames(
    matching: pd.DataFrame,
    candidate: pd.DataFrame,
    required_s1_ids: Iterable[str],
    *,
    valid_target_ids: Optional[Set[str]] = None,
) -> None:
    required = set(required_s1_ids)
    if list(matching.columns) != MATCHING_COLUMNS:
        raise DataValidationError(f"Invalid matching columns: {list(matching.columns)}")
    if list(candidate.columns) != CANDIDATE_COLUMNS:
        raise DataValidationError(f"Invalid candidate columns: {list(candidate.columns)}")

    for label, frame, value_column in (
        ("matching", matching, "matched_entity_ids"),
        ("candidate", candidate, "candidate_entity_ids"),
    ):
        if frame["source1_entity_id"].duplicated().any():
            raise DataValidationError(f"Duplicate S1 rows in {label} output")
        actual = set(frame["source1_entity_id"])
        if actual != required:
            raise DataValidationError(
                f"{label} S1 coverage mismatch: missing={len(required-actual)}, "
                f"extra={len(actual-required)}"
            )
        for s1, raw in zip(frame["source1_entity_id"], frame[value_column]):
            values = _parse_id_list(raw)
            if len(values) != len(set(values)):
                raise DataValidationError(f"Duplicate IDs within {label} row {s1}")
            invalid_prefix = [
                value for value in values if not value.startswith(("S2-", "S3-"))
            ]
            if invalid_prefix:
                raise DataValidationError(
                    f"Invalid target prefixes in {label} row {s1}: {invalid_prefix[:5]}"
                )
            if valid_target_ids is not None:
                unknown = set(values) - valid_target_ids
                if unknown:
                    raise DataValidationError(
                        f"Unknown target IDs in {label} row {s1}: {sorted(unknown)[:5]}"
                    )

    candidate_map = {
        s1: set(_parse_id_list(raw))
        for s1, raw in zip(
            candidate["source1_entity_id"], candidate["candidate_entity_ids"]
        )
    }
    for s1, raw in zip(matching["source1_entity_id"], matching["matched_entity_ids"]):
        missing = set(_parse_id_list(raw)) - candidate_map[s1]
        if missing:
            raise DataValidationError(
                f"Matched IDs absent from candidates for {s1}: {sorted(missing)[:5]}"
            )


def write_submission(
    matching: pd.DataFrame,
    candidate: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    matching.to_csv(matching_path, sep="\t", index=False, encoding="utf-8")
    candidate.to_csv(candidate_path, sep="\t", index=False, encoding="utf-8")
    return matching_path, candidate_path

