from dataclasses import replace

import pandas as pd

from business_entity_resolution.config import DEFAULT_CONFIG
from business_entity_resolution.production_submission import (
    COUNTRY_BLEND_CONFIG,
    OPEN_SET_BLEND_CONFIG,
    FrozenEnsemble,
    _partition_normalized_test,
)


def test_country_blend_and_open_set_fallback_are_exact():
    ensemble = FrozenEnsemble.__new__(FrozenEnsemble)
    assert ensemble.decision_config("india") == COUNTRY_BLEND_CONFIG["india"]
    assert ensemble.decision_config("us") == COUNTRY_BLEND_CONFIG["us"]
    assert ensemble.decision_config("france") == OPEN_SET_BLEND_CONFIG
    assert ensemble.decision_config("future-country") == OPEN_SET_BLEND_CONFIG
    assert OPEN_SET_BLEND_CONFIG == {
        "lgb_weight": 0.50,
        "xgb_weight": 0.50,
        "threshold": 0.736413,
    }


def test_normalized_country_partition_is_complete_and_reusable(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    rows = {
        "source1": [
            ["S1-1", "Cafe Bleu", "10 Rue A", "France"],
            ["S1-2", "Alpha", "20 Main St", "US"],
            ["S1-3", "Beta", "3 MG Road", "India"],
        ],
        "source2": [
            ["S2-1", "Cafe Bleu", "10 Rue A", "France"],
            ["S2-2", "Alpha", "20 Main St", "US"],
            ["S2-3", "Beta", "3 MG Road", "India"],
        ],
        "source3": [
            ["S3-1", "Cafe Bleu", "10 Rue A", "France"],
            ["S3-2", "Alpha", "20 Main St", "US"],
            ["S3-3", "Beta", "3 MG Road", "India"],
        ],
    }
    columns = ["entity_id", "business_name", "business_address", "country"]
    for source, values in rows.items():
        pd.DataFrame(values, columns=columns).to_csv(
            test_dir / f"test_{source}.tsv", sep="\t", index=False
        )

    config = replace(DEFAULT_CONFIG, test_dir=test_dir, chunk_size=2)
    work_dir = tmp_path / "production"
    first = _partition_normalized_test(config, work_dir)
    second = _partition_normalized_test(config, work_dir)

    assert first == second
    assert first["sources"]["source1"]["rows"] == 3
    assert first["sources"]["source1"]["country_rows"] == {
        "france": 1,
        "india": 1,
        "us": 1,
    }
    for source in rows:
        for path in first["sources"][source]["paths"].values():
            normalized = pd.read_parquet(path)
            assert len(normalized) == 1
            assert "name_core" in normalized.columns
    india_s1 = pd.read_parquet(first["sources"]["source1"]["paths"]["india"])
    assert india_s1.loc[0, "_row_order"] == 2
