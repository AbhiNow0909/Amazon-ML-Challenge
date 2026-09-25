import pandas as pd
import pytest

from business_entity_resolution.io_utils import (
    DataValidationError,
    load_source,
    validate_source_frame,
)


def test_tsv_parsing_preserves_strings_and_empty_fields(tmp_path):
    path = tmp_path / "source.tsv"
    path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-001\tAcme\t\tUS\n",
        encoding="utf-8",
    )
    frame = load_source(path, "source2")
    assert frame.loc[0, "entity_id"] == "S2-001"
    assert frame.loc[0, "business_address"] == ""
    assert all(dtype == object for dtype in frame.dtypes)


def test_invalid_source_prefix_is_rejected():
    frame = pd.DataFrame(
        [["S1-1", "Acme", "1 Main St", "US"]],
        columns=["entity_id", "business_name", "business_address", "country"],
    )
    with pytest.raises(DataValidationError, match="prefix"):
        validate_source_frame(frame, "source2")


def test_duplicate_ids_are_rejected():
    frame = pd.DataFrame(
        [
            ["S3-1", "A", "", "France"],
            ["S3-1", "B", "", "France"],
        ],
        columns=["entity_id", "business_name", "business_address", "country"],
    )
    with pytest.raises(DataValidationError, match="Duplicate"):
        validate_source_frame(frame, "source3")

