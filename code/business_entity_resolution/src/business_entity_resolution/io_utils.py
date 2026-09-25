"""Safe TSV loading, validation, and cache helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence, Set, Union

import pandas as pd


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
SOURCE_PREFIXES = {"source1": "S1-", "source2": "S2-", "source3": "S3-"}


class DataValidationError(ValueError):
    """Raised when a challenge input or generated artifact violates its schema."""


def read_tsv(
    path: Union[str, Path],
    *,
    chunksize: Optional[int] = None,
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """Read a challenge TSV without interpreting empty strings as missing values."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        chunksize=chunksize,
        encoding="utf-8",
    )


def validate_columns(
    frame: pd.DataFrame,
    expected: Sequence[str],
    *,
    path: Optional[Union[str, Path]] = None,
) -> None:
    actual = list(frame.columns)
    if actual != list(expected):
        location = f" in {path}" if path else ""
        raise DataValidationError(
            f"Unexpected columns{location}: {actual}; expected exactly {list(expected)}"
        )


def validate_source_frame(
    frame: pd.DataFrame,
    source: str,
    *,
    check_duplicates: bool = True,
    path: Optional[Union[str, Path]] = None,
) -> None:
    """Validate one source frame without changing it."""

    if source not in SOURCE_PREFIXES:
        raise ValueError(f"Unknown source {source!r}; expected one of {sorted(SOURCE_PREFIXES)}")
    validate_columns(frame, SOURCE_COLUMNS, path=path)
    ids = frame["entity_id"]
    if ids.eq("").any():
        raise DataValidationError(f"Empty entity_id found in {path or source}")
    prefix = SOURCE_PREFIXES[source]
    invalid = ~ids.str.startswith(prefix)
    if invalid.any():
        examples = ids[invalid].head(5).tolist()
        raise DataValidationError(
            f"Invalid {source} ID prefix in {path or source}: {examples}; expected {prefix}"
        )
    if check_duplicates and ids.duplicated().any():
        examples = ids[ids.duplicated(keep=False)].drop_duplicates().head(5).tolist()
        raise DataValidationError(f"Duplicate entity IDs in {path or source}: {examples}")


def validate_ground_truth_frame(
    frame: pd.DataFrame,
    *,
    check_duplicates: bool = True,
    path: Optional[Union[str, Path]] = None,
) -> None:
    validate_columns(frame, GROUND_TRUTH_COLUMNS, path=path)
    ids = frame["source1_entity_id"]
    invalid = ~ids.str.startswith("S1-")
    if invalid.any():
        raise DataValidationError(
            f"Ground truth contains invalid S1 IDs: {ids[invalid].head(5).tolist()}"
        )
    if check_duplicates and ids.duplicated().any():
        examples = ids[ids.duplicated(keep=False)].drop_duplicates().head(5).tolist()
        raise DataValidationError(f"Ground truth contains duplicate S1 rows: {examples}")

    bad_targets = []
    for raw in frame["matched_entity_ids"]:
        if not raw:
            continue
        values = raw.split(",")
        if len(values) != len(set(values)):
            raise DataValidationError("Ground truth contains duplicate IDs within a match list")
        bad_targets.extend(
            value for value in values if not value.startswith(("S2-", "S3-"))
        )
        if len(bad_targets) >= 5:
            break
    if bad_targets:
        raise DataValidationError(f"Ground truth contains invalid target IDs: {bad_targets[:5]}")


def load_source(
    path: Union[str, Path],
    source: str,
    *,
    chunksize: Optional[int] = None,
    validate: bool = True,
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """Load a source file, optionally as validated chunks.

    Duplicate checking across chunks is performed by ``iter_source_chunks``. A
    single-frame load checks duplicates directly.
    """

    loaded = read_tsv(path, chunksize=chunksize)
    if chunksize is not None:
        return iter_source_chunks(loaded, source, path=path, validate=validate)
    assert isinstance(loaded, pd.DataFrame)
    if validate:
        validate_source_frame(loaded, source, path=path)
    return loaded


def iter_source_chunks(
    chunks: Iterable[pd.DataFrame],
    source: str,
    *,
    path: Optional[Union[str, Path]] = None,
    validate: bool = True,
) -> Iterator[pd.DataFrame]:
    """Yield chunks while detecting duplicates both within and across chunks."""

    seen: Set[str] = set()
    for chunk in chunks:
        if validate:
            validate_source_frame(chunk, source, path=path)
            overlap = seen.intersection(chunk["entity_id"])
            if overlap:
                raise DataValidationError(
                    f"Duplicate entity IDs across chunks in {path or source}: "
                    f"{sorted(overlap)[:5]}"
                )
            seen.update(chunk["entity_id"])
        yield chunk


def load_ground_truth(
    path: Union[str, Path],
    *,
    chunksize: Optional[int] = None,
    validate: bool = True,
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    loaded = read_tsv(path, chunksize=chunksize)
    if chunksize is not None:
        return iter_ground_truth_chunks(loaded, path=path, validate=validate)
    assert isinstance(loaded, pd.DataFrame)
    if validate:
        validate_ground_truth_frame(loaded, path=path)
    return loaded


def iter_ground_truth_chunks(
    chunks: Iterable[pd.DataFrame],
    *,
    path: Optional[Union[str, Path]] = None,
    validate: bool = True,
) -> Iterator[pd.DataFrame]:
    seen: Set[str] = set()
    for chunk in chunks:
        if validate:
            validate_ground_truth_frame(chunk, path=path)
            overlap = seen.intersection(chunk["source1_entity_id"])
            if overlap:
                raise DataValidationError(
                    f"Duplicate ground-truth S1 IDs across chunks: {sorted(overlap)[:5]}"
                )
            seen.update(chunk["source1_entity_id"])
        yield chunk


def inspect_tsv_schema(path: Union[str, Path]) -> Dict[str, object]:
    """Return a lightweight header/sample/size inspection without loading the file."""

    path = Path(path)
    sample = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        nrows=5,
        encoding="utf-8",
    )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "columns": list(sample.columns),
        "sample_rows": sample.to_dict(orient="records"),
    }


def ground_truth_to_mapping(frame: pd.DataFrame) -> Dict[str, Set[str]]:
    validate_ground_truth_frame(frame)
    return {
        row.source1_entity_id: set(row.matched_entity_ids.split(","))
        if row.matched_entity_ids
        else set()
        for row in frame.itertuples(index=False)
    }


def write_parquet_cache(
    frame: pd.DataFrame,
    path: Union[str, Path],
    metadata: Mapping[str, object],
) -> None:
    """Write a Parquet artifact and a JSON sidecar describing its configuration."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def read_parquet_cache(
    path: Union[str, Path],
    *,
    expected_fingerprint: Optional[str] = None,
) -> pd.DataFrame:
    path = Path(path)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Incomplete cache artifact: {path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if expected_fingerprint and metadata.get("config_fingerprint") != expected_fingerprint:
        raise DataValidationError(
            f"Cache {path} was built with configuration "
            f"{metadata.get('config_fingerprint')}, expected {expected_fingerprint}"
        )
    return pd.read_parquet(path)

