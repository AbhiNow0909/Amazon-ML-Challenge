import pandas as pd
import pytest

from business_entity_resolution.io_utils import DataValidationError
from business_entity_resolution.submission import (
    build_submission_frames,
    validate_submission_frames,
)


def _candidates():
    return pd.DataFrame(
        [
            ["S1-1", "S2-1", 1],
            ["S1-1", "S3-1", 2],
        ],
        columns=["source1_entity_id", "candidate_entity_id", "cheap_rank"],
    )


def test_submission_preserves_empty_rows_and_subset_constraint():
    matching, candidate = build_submission_frames(
        ["S1-1", "S1-2"], _candidates(), {"S1-1": {"S2-1"}}
    )
    assert len(matching) == 2
    assert matching.loc[matching["source1_entity_id"] == "S1-2", "matched_entity_ids"].item() == ""
    validate_submission_frames(
        matching,
        candidate,
        ["S1-1", "S1-2"],
        valid_target_ids={"S2-1", "S3-1"},
    )


def test_match_outside_candidates_is_rejected():
    with pytest.raises(DataValidationError, match="absent"):
        build_submission_frames(["S1-1"], _candidates(), {"S1-1": {"S2-999"}})


def test_duplicate_and_invalid_ids_are_rejected():
    matching = pd.DataFrame(
        [["S1-1", "S2-1,S2-1"]],
        columns=["source1_entity_id", "matched_entity_ids"],
    )
    candidate = pd.DataFrame(
        [["S1-1", "S2-1"]],
        columns=["source1_entity_id", "candidate_entity_ids"],
    )
    with pytest.raises(DataValidationError, match="Duplicate"):
        validate_submission_frames(matching, candidate, ["S1-1"])

    matching.loc[0, "matched_entity_ids"] = "S4-1"
    candidate.loc[0, "candidate_entity_ids"] = "S4-1"
    with pytest.raises(DataValidationError, match="prefix"):
        validate_submission_frames(matching, candidate, ["S1-1"])


def test_duplicate_s1_rows_are_rejected():
    matching = pd.DataFrame(
        [["S1-1", ""], ["S1-1", ""]],
        columns=["source1_entity_id", "matched_entity_ids"],
    )
    candidate = pd.DataFrame(
        [["S1-1", ""]],
        columns=["source1_entity_id", "candidate_entity_ids"],
    )
    with pytest.raises(DataValidationError, match="Duplicate S1"):
        validate_submission_frames(matching, candidate, ["S1-1"])

