"""Checkpointed full-test E2.3 retrieval and LGBM/XGBoost submission runner.

The module deliberately has no training path.  It loads the frozen E2.3 model
artifacts, processes one country at a time, scores bounded Source-1 batches, and
assembles the two official TSV files only after every checkpoint is complete.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import resource
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from tqdm.auto import tqdm

from .address_preblocked_retrieval import (
    prepare_address_tfidf_index,
    retrieve_from_prepared_address_index,
    strategy_from_config,
)
from .blocking import (
    generate_legacy_candidates_from_prepared,
    prepare_legacy_index,
)
from .config import DEFAULT_CONFIG, PipelineConfig
from .features import (
    _deterministic_descending_rank,
    add_candidate_relative_features,
    build_feature_table,
    build_feature_table_fast_exact,
)
from .io_utils import (
    SOURCE_COLUMNS,
    DataValidationError,
    load_source,
    validate_source_frame,
)
from .multi_channel_fusion import (
    prepare_multi_channel_universe,
    protected_legacy_multi_channel_rrf_fast_exact,
)
from .normalize import normalize_country, normalize_records
from .submission import (
    build_submission_frames,
    validate_submission_frames,
    write_submission,
)
from .tfidf_retrieval import (
    prepare_name_tfidf_index,
    retrieve_name_tfidf_from_prepared,
)


LOGGER = logging.getLogger(__name__)
PRODUCTION_VERSION = "e23_ensemble_production_v1"

DEFAULT_LGB_MODEL = "cache/models/e23_lgbm_38a0b640f78b.txt"
DEFAULT_XGB_MODEL = "cache/models/e24_xgb_38a0b640f78b.json"
DEFAULT_FEATURE_METADATA = (
    "cache/model_features/e23_retrieval_38a0b640f78b_training_metadata.json"
)
DEFAULT_LGB_METADATA = "cache/models/e23_lgbm_38a0b640f78b_metadata.json"
DEFAULT_XGB_METADATA = "cache/models/e24_xgb_38a0b640f78b_metadata.json"

COUNTRY_BLEND_CONFIG: Mapping[str, Mapping[str, float]] = {
    "india": {
        "lgb_weight": 0.35,
        "xgb_weight": 0.65,
        "threshold": 0.7739178356713426,
    },
    "us": {
        "lgb_weight": 0.90,
        "xgb_weight": 0.10,
        "threshold": 0.7403607214428858,
    },
}
OPEN_SET_BLEND_CONFIG: Mapping[str, float] = {
    "lgb_weight": 0.50,
    "xgb_weight": 0.50,
    "threshold": 0.736413,
}

ZERO_FILL_CANDIDATE_COLUMNS = [
    "cheap_score",
    "cheap_score_margin_to_next",
    "rare_name_token_hits",
    "rare_address_token_hits",
    "block_exact_full_name",
    "block_exact_core_name",
    "block_exact_sorted_name",
    "block_rare_name_token",
    "block_exact_address",
    "block_rare_address_token",
    "block_cross_country_exact_name",
    "legacy_cheap_score",
    "block_name_tfidf",
    "is_tfidf_only",
    "is_legacy_only",
    "supported_by_both",
    "legacy_blocking_channel_count",
    "candidate_count_raw",
]

E23_EXTRA_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "address_tfidf_similarity",
    "address_tfidf_rank",
    "address_route_postcode",
    "address_route_postcode_prefix",
    "address_route_numeric",
    "address_route_rare_token",
    "address_route_country_fallback",
    "address_route_count",
    "address_preblock_pool_size",
    "has_address_tfidf",
    "block_address_tfidf",
    "is_address_tfidf_only",
    "retrieval_channel_count",
]


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _event(name: str, **payload: object) -> None:
    print(json.dumps({"event": name, **payload}, default=_json_default), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="snappy")
    temporary.replace(path)


def _file_signature(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _runtime_hardware(requested_xgb_device: str) -> Dict[str, object]:
    """Detect available hardware without changing any installed components."""

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    except Exception as error:  # pragma: no cover - environment-specific
        cuda_available = False
        gpu_name = None
        LOGGER.warning("CUDA detection failed: %s", error)
    if requested_xgb_device == "cuda" and not cuda_available:
        raise RuntimeError("--xgb-device cuda requested but CUDA is unavailable")
    xgb_device = (
        "cuda"
        if requested_xgb_device == "cuda"
        or (requested_xgb_device == "auto" and cuda_available)
        else "cpu"
    )
    return {
        "cpu_count": os.cpu_count(),
        "cuda": cuda_available,
        "gpu": gpu_name,
        "xgb_device": xgb_device,
        "gpu_stages": ["xgboost_inference"] if xgb_device == "cuda" else [],
        "cpu_stages": [
            "legacy_blocking",
            "name_tfidf_sparse_retrieval",
            "address_tfidf_sparse_retrieval",
            "fusion",
            "feature_generation",
            "lightgbm_inference",
        ],
    }


def _gpu_snapshot() -> Mapping[str, object]:
    try:
        completed = subprocess.run(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        utilization, used, total = [
            int(value.strip()) for value in completed.stdout.splitlines()[0].split(",")
        ]
        return {
            "utilization_percent": utilization,
            "memory_used_mib": used,
            "memory_total_mib": total,
        }
    except Exception:
        return {"available": False}


def _normalized_checkpoints_valid(config: PipelineConfig, work_dir: Path) -> bool:
    manifest_path = work_dir / "normalized" / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_inputs = {
            source: _file_signature(config.test_dir / f"test_source{source[-1]}.tsv")
            for source in ("source1", "source2", "source3")
        }
        return (
            manifest.get("version") == PRODUCTION_VERSION
            and manifest.get("inputs") == expected_inputs
            and all(
                Path(path).is_file()
                for entry in manifest.get("sources", {}).values()
                for path in entry.get("paths", {}).values()
            )
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _country_slug(country: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in country)
    return safe.strip("_") or "missing"


def _production_config(config: PipelineConfig) -> PipelineConfig:
    """Return the frozen E2.3 retrieval configuration."""

    return replace(
        config,
        use_name_tfidf=True,
        tfidf_name_top_k=30,
        use_address_tfidf=True,
        use_preblocked_address_tfidf=True,
        tfidf_address_top_k=20,
        address_preblock_use_postcode=True,
        address_preblock_use_postcode_prefix=False,
        address_preblock_numeric_max_df=5_000,
        address_preblock_rare_token_max_df=500,
        address_preblock_rare_tokens_per_query=3,
        address_preblock_pool_cap=15_000,
        address_preblock_fallback_if_empty=True,
        address_preblock_fallback_char_features=3,
        address_preblock_fallback_char_max_df=5_000,
    )


class FrozenEnsemble:
    """Validated models, feature order, and country decision rules."""

    def __init__(
        self,
        repo_root: Path,
        *,
        lgb_model: str = DEFAULT_LGB_MODEL,
        xgb_model: str = DEFAULT_XGB_MODEL,
        feature_metadata: str = DEFAULT_FEATURE_METADATA,
        lgb_metadata: str = DEFAULT_LGB_METADATA,
        xgb_metadata: str = DEFAULT_XGB_METADATA,
        xgb_device: str = "cpu",
    ) -> None:
        self.paths = {
            "lgb_model": repo_root / lgb_model,
            "xgb_model": repo_root / xgb_model,
            "feature_metadata": repo_root / feature_metadata,
            "lgb_metadata": repo_root / lgb_metadata,
            "xgb_metadata": repo_root / xgb_metadata,
        }
        missing = [str(path) for path in self.paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing frozen production artifacts: {missing}")
        feature_meta = json.loads(
            self.paths["feature_metadata"].read_text(encoding="utf-8")
        )
        lgb_meta = json.loads(self.paths["lgb_metadata"].read_text(encoding="utf-8"))
        xgb_meta = json.loads(self.paths["xgb_metadata"].read_text(encoding="utf-8"))
        feature_lists = [
            feature_meta["model_features"],
            lgb_meta["model_features"],
            xgb_meta["model_features"],
        ]
        if not all(values == feature_lists[0] for values in feature_lists[1:]):
            raise DataValidationError("Frozen model feature metadata does not agree")
        self.feature_columns = list(feature_lists[0])
        if len(self.feature_columns) != 75 or len(set(self.feature_columns)) != 75:
            raise DataValidationError("Expected exactly 75 unique model features")
        self.lgb_best_iteration = int(lgb_meta["best_iteration"])
        self.xgb_best_iteration = int(xgb_meta["best_iteration"])
        self.lgb = lgb.Booster(model_file=str(self.paths["lgb_model"]))
        self.xgb = xgb.Booster()
        self.xgb.load_model(self.paths["xgb_model"])
        self.xgb_device = xgb_device
        self.xgb.set_param({"device": xgb_device})
        self.xgb_exact_cpu: Optional[xgb.Booster] = None
        if xgb_device == "cuda":
            self.xgb_exact_cpu = xgb.Booster()
            self.xgb_exact_cpu.load_model(self.paths["xgb_model"])
            self.xgb_exact_cpu.set_param({"device": "cpu"})
        self.last_prediction_profile: Dict[str, float] = {}
        if self.lgb.num_feature() != 75:
            raise DataValidationError("LightGBM artifact does not expose 75 features")
        if self.xgb.num_features() != 75:
            raise DataValidationError("XGBoost artifact does not expose 75 features")
        if self.xgb.feature_names is not None and list(self.xgb.feature_names) != self.feature_columns:
            raise DataValidationError("XGBoost model feature order differs from metadata")
        self.artifact_hashes = {
            name: _sha256(path) for name, path in self.paths.items()
        }

    def decision_config(self, country_norm: str) -> Mapping[str, float]:
        return COUNTRY_BLEND_CONFIG.get(country_norm, OPEN_SET_BLEND_CONFIG)

    def predict(self, features: pd.DataFrame, country_norm: str) -> pd.DataFrame:
        missing = [column for column in self.feature_columns if column not in features]
        if missing:
            raise DataValidationError(f"Production features are missing: {missing}")
        matrix = features[self.feature_columns]
        if list(matrix.columns) != self.feature_columns:
            raise AssertionError("Model feature order changed")
        values = matrix.to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(values).all():
            raise DataValidationError("Model feature matrix contains NaN or infinity")
        xgb_matrix_started = time.perf_counter()
        xgb_matrix = xgb.DMatrix(matrix, feature_names=self.feature_columns)
        xgb_matrix_seconds = time.perf_counter() - xgb_matrix_started

        def predict_lgb():
            stage = time.perf_counter()
            scores = self.lgb.predict(
                matrix, num_iteration=self.lgb_best_iteration
            ).astype("float32")
            return scores, time.perf_counter() - stage

        def predict_xgb(booster: xgb.Booster):
            stage = time.perf_counter()
            scores = booster.predict(
                xgb_matrix,
                iteration_range=(0, self.xgb_best_iteration + 1),
            ).astype("float32")
            return scores, time.perf_counter() - stage

        gpu_difference = 0.0
        gpu_seconds = 0.0
        if self.xgb_exact_cpu is not None:
            # Run CPU-exact and GPU traversal concurrently.  The CPU scores are
            # retained to reproduce validation bit-for-bit; CUDA traversal is
            # checked continuously and overlaps the LightGBM/CPU work.
            with ThreadPoolExecutor(max_workers=3) as executor:
                lgb_future = executor.submit(predict_lgb)
                cpu_future = executor.submit(predict_xgb, self.xgb_exact_cpu)
                gpu_future = executor.submit(predict_xgb, self.xgb)
                lgb_scores, lgb_seconds = lgb_future.result()
                xgb_scores, xgb_seconds = cpu_future.result()
                gpu_scores, gpu_seconds = gpu_future.result()
            gpu_difference = float(np.max(np.abs(xgb_scores - gpu_scores)))
        else:
            lgb_scores, lgb_seconds = predict_lgb()
            xgb_scores, xgb_seconds = predict_xgb(self.xgb)
        self.last_prediction_profile = {
            "lgb_inference_seconds": lgb_seconds,
            "xgb_matrix_seconds": xgb_matrix_seconds,
            "xgb_inference_seconds": xgb_seconds,
            "xgb_gpu_shadow_seconds": gpu_seconds,
            "xgb_gpu_cpu_max_abs_difference": gpu_difference,
            "xgb_scores_retained_from": "cpu-exact",
        }
        decision = self.decision_config(country_norm)
        blended = (
            float(decision["lgb_weight"]) * lgb_scores
            + float(decision["xgb_weight"]) * xgb_scores
        ).astype("float32")
        result = features[["source1_entity_id", "candidate_entity_id"]].copy()
        result["lgb_probability"] = lgb_scores
        result["xgb_probability"] = xgb_scores
        result["blend_probability"] = blended
        result["is_match"] = blended >= float(decision["threshold"])
        return result


def build_e23_model_features(
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    feed: pd.DataFrame,
    model_features: Sequence[str],
    *,
    chunk_size: int,
    close_competitor_tolerance: float,
    fast_exact: bool = False,
    prepared_feed_index: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Reproduce the exact validated notebook feature construction."""

    cleaned = candidates.copy()
    missing_zero_columns = [
        column for column in ZERO_FILL_CANDIDATE_COLUMNS if column not in cleaned
    ]
    if missing_zero_columns:
        raise DataValidationError(
            f"Candidate table lacks validated neutral-fill fields: {missing_zero_columns}"
        )
    cleaned[ZERO_FILL_CANDIDATE_COLUMNS] = cleaned[
        ZERO_FILL_CANDIDATE_COLUMNS
    ].fillna(0)
    builder = build_feature_table_fast_exact if fast_exact else build_feature_table
    if fast_exact:
        base = builder(
            cleaned,
            source1,
            feed,
            chunk_size=chunk_size,
            prepared_feed_index=prepared_feed_index,
        )
    else:
        base = builder(cleaned, source1, feed, chunk_size=chunk_size)
    base = add_candidate_relative_features(
        base, close_competitor_tolerance=close_competitor_tolerance
    )
    missing_extra = [column for column in E23_EXTRA_COLUMNS if column not in cleaned]
    if missing_extra:
        raise DataValidationError(
            f"Candidate table lacks E2.3 retrieval fields: {missing_extra}"
        )
    features = base.merge(
        cleaned[E23_EXTRA_COLUMNS],
        on=["source1_entity_id", "candidate_entity_id"],
        how="left",
        validate="one_to_one",
    )
    binary = [
        "address_route_postcode",
        "address_route_postcode_prefix",
        "address_route_numeric",
        "address_route_rare_token",
        "address_route_country_fallback",
        "has_address_tfidf",
        "block_address_tfidf",
        "is_address_tfidf_only",
    ]
    counts = [
        "address_tfidf_rank",
        "address_route_count",
        "address_preblock_pool_size",
        "retrieval_channel_count",
    ]
    features[binary] = features[binary].fillna(0).astype("uint8")
    features["address_tfidf_similarity"] = features[
        "address_tfidf_similarity"
    ].fillna(0).astype("float32")
    features[counts] = features[counts].fillna(0).astype("uint32")
    features["relative_address_tfidf_rank"] = _deterministic_descending_rank(
        features, "address_tfidf_similarity"
    ).astype("uint32")
    best = features.groupby("source1_entity_id")[
        "address_tfidf_similarity"
    ].transform("max")
    features["address_tfidf_gap_to_best"] = (
        best - features["address_tfidf_similarity"]
    ).astype("float32")
    missing_model = [column for column in model_features if column not in features]
    if missing_model:
        raise DataValidationError(f"Missing exact model features: {missing_model}")
    if list(features[list(model_features)].columns) != list(model_features):
        raise AssertionError("Exact 75-feature ordering was not preserved")
    return features


def _prediction_mapping(scored: pd.DataFrame) -> Dict[str, Set[str]]:
    positive = scored.loc[scored["is_match"]]
    return (
        positive.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(set)
        .to_dict()
        if not positive.empty
        else {}
    )


def _retrieve_score_batch(
    source1_batch: pd.DataFrame,
    feature_feed: pd.DataFrame,
    legacy_index: object,
    name_index: object,
    address_index: object,
    ensemble: FrozenEnsemble,
    config: PipelineConfig,
    prepared_feature_feed_index: Optional[pd.DataFrame] = None,
    parallel_retrieval: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    total_started = time.perf_counter()
    country_norm = str(source1_batch["country_norm"].iloc[0])

    def legacy_task():
        stage = time.perf_counter()
        values = generate_legacy_candidates_from_prepared(
            source1_batch, legacy_index, config, max_k=50
        )
        return values, time.perf_counter() - stage

    def name_task():
        stage = time.perf_counter()
        values = retrieve_name_tfidf_from_prepared(
            source1_batch, name_index, config, top_k=30
        )
        return values, time.perf_counter() - stage

    def address_task():
        stage = time.perf_counter()
        values = retrieve_from_prepared_address_index(
            source1_batch,
            address_index,
            strategy_from_config(config, name="broad_union"),
            top_k=20,
        )
        return values, time.perf_counter() - stage

    retrieval_started = time.perf_counter()
    if parallel_retrieval:
        with ThreadPoolExecutor(max_workers=3) as executor:
            legacy_future = executor.submit(legacy_task)
            name_future = executor.submit(name_task)
            address_future = executor.submit(address_task)
            (legacy, legacy_profile), legacy_seconds = legacy_future.result()
            (name, name_profile), name_seconds = name_future.result()
            (address, address_profile), address_seconds = address_future.result()
    else:
        (legacy, legacy_profile), legacy_seconds = legacy_task()
        (name, name_profile), name_seconds = name_task()
        (address, address_profile), address_seconds = address_task()
    retrieval_wall_seconds = time.perf_counter() - retrieval_started
    fusion_started = time.perf_counter()
    universe = prepare_multi_channel_universe(
        legacy, name, address, name_top_k=30, address_top_k=20
    )
    candidates = protected_legacy_multi_channel_rrf_fast_exact(
        universe,
        final_k=50,
        minimum_legacy=35,
        legacy_weight=2.0,
        name_weight=1.0,
        address_weight=1.0,
        rrf_constant=20.0,
    )
    fusion_seconds = time.perf_counter() - fusion_started
    feature_started = time.perf_counter()
    features = build_e23_model_features(
        candidates,
        source1_batch,
        feature_feed,
        ensemble.feature_columns,
        chunk_size=config.chunk_size,
        close_competitor_tolerance=config.e2_close_competitor_tolerance,
        fast_exact=True,
        prepared_feed_index=prepared_feature_feed_index,
    )
    feature_seconds = time.perf_counter() - feature_started
    model_started = time.perf_counter()
    scored = ensemble.predict(features, country_norm)
    model_seconds = time.perf_counter() - model_started
    submission_started = time.perf_counter()
    predictions = _prediction_mapping(scored)
    matching, candidate = build_submission_frames(
        source1_batch["entity_id"], candidates, predictions
    )
    validate_submission_frames(
        matching,
        candidate,
        source1_batch["entity_id"],
        valid_target_ids=set(candidates["candidate_entity_id"]),
    )
    submission_seconds = time.perf_counter() - submission_started
    profile = {
        "source1_rows": len(source1_batch),
        "candidate_pairs": len(candidates),
        "predicted_pairs": int(scored["is_match"].sum()),
        "empty_predictions": int(matching["matched_entity_ids"].eq("").sum()),
        "legacy": legacy_profile,
        "name": name_profile,
        "address": address_profile,
        "timing": {
            "legacy_seconds": legacy_seconds,
            "name_tfidf_seconds": name_seconds,
            "address_tfidf_seconds": address_seconds,
            "retrieval_wall_seconds": retrieval_wall_seconds,
            "retrieval_parallel": parallel_retrieval,
            "fusion_seconds": fusion_seconds,
            "feature_seconds": feature_seconds,
            "model_seconds": model_seconds,
            **ensemble.last_prediction_profile,
            "submission_seconds": submission_seconds,
            "total_seconds": time.perf_counter() - total_started,
        },
    }
    return matching, candidate, profile


def _bounded_source_sample(
    path: Path, source: str, per_country: int
) -> pd.DataFrame:
    pieces: List[pd.DataFrame] = []
    counts: Dict[str, int] = {}
    offset = 0
    for chunk in load_source(path, source, chunksize=50_000, validate=False):
        validate_source_frame(chunk, source, check_duplicates=True, path=path)
        if source == "source1":
            chunk = chunk.copy()
            chunk["_row_order"] = np.arange(offset, offset + len(chunk), dtype=np.int64)
        offset += len(chunk)
        for country, group in chunk.groupby("country", sort=False):
            remaining = per_country - counts.get(country, 0)
            if remaining > 0:
                selected = group.head(remaining)
                pieces.append(selected)
                counts[country] = counts.get(country, 0) + len(selected)
        if counts and all(value >= per_country for value in counts.values()) and len(counts) >= 3:
            break
    if not pieces:
        raise RuntimeError(f"No smoke rows loaded from {path}")
    return pd.concat(pieces, ignore_index=True)


def _cross_country_rows_in_memory(
    source1: pd.DataFrame, feed: pd.DataFrame, posting_cap: int
) -> pd.DataFrame:
    query_names = set(source1["name_full"]) - {""}
    relevant = feed.loc[feed["name_full"].isin(query_names)].copy()
    counts = relevant["name_full"].value_counts()
    relevant = relevant.loc[relevant["name_full"].map(counts).le(posting_cap)]
    country = str(source1["country_norm"].iloc[0])
    return relevant.loc[relevant["country_norm"] != country]


def run_smoke(
    config: PipelineConfig,
    ensemble: FrozenEnsemble,
    *,
    s1_per_country: int,
    feed_per_source_country: int,
    output_dir: Path,
) -> Dict[str, object]:
    """Run all production stages on a tiny real test subset."""

    started = time.perf_counter()
    raw_s1 = _bounded_source_sample(
        config.test_dir / "test_source1.tsv", "source1", s1_per_country
    )
    raw_s2 = _bounded_source_sample(
        config.test_dir / "test_source2.tsv", "source2", feed_per_source_country
    )
    raw_s3 = _bounded_source_sample(
        config.test_dir / "test_source3.tsv", "source3", feed_per_source_country
    )
    source1 = normalize_records(raw_s1)
    source2 = normalize_records(raw_s2)
    source3 = normalize_records(raw_s3)
    all_feed = pd.concat([source2, source3], ignore_index=True)
    matching_parts: List[pd.DataFrame] = []
    candidate_parts: List[pd.DataFrame] = []
    profiles: Dict[str, object] = {}
    for country_norm, country_s1 in source1.groupby("country_norm", sort=True):
        main_feed = all_feed.loc[all_feed["country_norm"] == country_norm].copy()
        cross = _cross_country_rows_in_memory(country_s1, all_feed, config.posting_list_cap)
        feature_feed = pd.concat([main_feed, cross], ignore_index=True).drop_duplicates(
            "entity_id"
        )
        feature_feed_index = feature_feed.set_index("entity_id", verify_integrity=True)
        legacy_index = prepare_legacy_index(feature_feed, config)
        name_index = prepare_name_tfidf_index(main_feed, config)
        address_index = prepare_address_tfidf_index(
            main_feed,
            config,
            maximum_numeric_df=config.address_preblock_numeric_max_df,
            maximum_rare_token_df=config.address_preblock_rare_token_max_df,
            maximum_postcode_prefix_df=config.address_preblock_postcode_prefix_max_df,
            postcode_prefix_length=config.address_preblock_postcode_prefix_length,
        )
        matching, candidate, profile = _retrieve_score_batch(
            country_s1,
            feature_feed,
            legacy_index,
            name_index,
            address_index,
            ensemble,
            config,
            feature_feed_index,
        )
        order = country_s1.set_index("entity_id")["_row_order"]
        matching["_row_order"] = matching["source1_entity_id"].map(order)
        candidate["_row_order"] = candidate["source1_entity_id"].map(order)
        matching_parts.append(matching)
        candidate_parts.append(candidate)
        profiles[country_norm] = profile
        del legacy_index, name_index, address_index
        del feature_feed_index
        gc.collect()
    matching = pd.concat(matching_parts, ignore_index=True).sort_values("_row_order")
    candidate = pd.concat(candidate_parts, ignore_index=True).sort_values("_row_order")
    matching.drop(columns="_row_order", inplace=True)
    candidate.drop(columns="_row_order", inplace=True)
    validate_submission_frames(
        matching,
        candidate,
        raw_s1["entity_id"],
        valid_target_ids=set(all_feed["entity_id"]),
    )
    matching_path, candidate_path = write_submission(matching, candidate, output_dir)
    with tempfile.TemporaryDirectory(prefix="ber-smoke-validator-") as temporary:
        test_dir = Path(temporary) / "test"
        test_dir.mkdir(parents=True)
        raw_s1[SOURCE_COLUMNS].to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)
        raw_s2[SOURCE_COLUMNS].to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)
        raw_s3[SOURCE_COLUMNS].to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)
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
            f"Official smoke validator failed:\n{completed.stdout}\n{completed.stderr}"
        )
    report = {
        "source1_rows": len(raw_s1),
        "source2_rows": len(raw_s2),
        "source3_rows": len(raw_s3),
        "countries": source1["country_norm"].value_counts().to_dict(),
        "candidate_rows": int(
            candidate["candidate_entity_ids"].map(
                lambda value: len(value.split(",")) if value else 0
            ).sum()
        ),
        "matched_rows": int(
            matching["matched_entity_ids"].map(
                lambda value: len(value.split(",")) if value else 0
            ).sum()
        ),
        "empty_predictions": int(matching["matched_entity_ids"].eq("").sum()),
        "profiles": profiles,
        "matching_path": str(matching_path),
        "candidate_path": str(candidate_path),
        "official_validator": completed.stdout.strip(),
        "runtime_seconds": time.perf_counter() - started,
    }
    _event("production-smoke-complete", report=report)
    return report


def _partition_normalized_test(
    config: PipelineConfig, work_dir: Path
) -> Mapping[str, object]:
    """Normalize raw test sources once into resumable country Parquet shards."""

    normalized_dir = work_dir / "normalized"
    manifest_path = normalized_dir / "manifest.json"
    expected_inputs = {
        source: _file_signature(config.test_dir / f"test_source{source[-1]}.tsv")
        for source in ("source1", "source2", "source3")
    }
    manifest: Dict[str, object] = {
        "version": PRODUCTION_VERSION,
        "inputs": expected_inputs,
        "sources": {},
    }
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if loaded.get("version") == PRODUCTION_VERSION and loaded.get("inputs") == expected_inputs:
            manifest = loaded

    for source in ("source1", "source2", "source3"):
        existing = manifest.get("sources", {}).get(source)
        if existing and all(Path(path).is_file() for path in existing["paths"].values()):
            _event("normalized-source-reused", source=source, rows=existing["rows"])
            continue
        source_path = config.test_dir / f"test_source{source[-1]}.tsv"
        writers: Dict[str, pq.ParquetWriter] = {}
        temporary_paths: Dict[str, Path] = {}
        final_paths: Dict[str, Path] = {}
        country_rows: Dict[str, int] = {}
        total_rows = 0
        try:
            chunks = load_source(
                source_path, source, chunksize=config.chunk_size, validate=False
            )
            for chunk in tqdm(chunks, desc=f"Normalize {source}", unit="chunk"):
                validate_source_frame(
                    chunk, source, check_duplicates=True, path=source_path
                )
                if source == "source1":
                    chunk = chunk.copy()
                    chunk["_row_order"] = np.arange(
                        total_rows, total_rows + len(chunk), dtype=np.int64
                    )
                total_rows += len(chunk)
                normalized = normalize_records(chunk)
                for country, group in normalized.groupby("country_norm", sort=True):
                    slug = _country_slug(country)
                    final = normalized_dir / source / f"{slug}.parquet"
                    temporary = final.with_suffix(".parquet.tmp")
                    final.parent.mkdir(parents=True, exist_ok=True)
                    if country not in writers:
                        if temporary.exists():
                            temporary.unlink()
                        table = pa.Table.from_pandas(group, preserve_index=False)
                        writers[country] = pq.ParquetWriter(
                            temporary, table.schema, compression="snappy"
                        )
                        temporary_paths[country] = temporary
                        final_paths[country] = final
                    else:
                        table = pa.Table.from_pandas(
                            group, schema=writers[country].schema, preserve_index=False
                        )
                    writers[country].write_table(table)
                    country_rows[country] = country_rows.get(country, 0) + len(group)
        finally:
            for writer in writers.values():
                writer.close()
        for country, temporary in temporary_paths.items():
            temporary.replace(final_paths[country])
        source_entry = {
            "rows": total_rows,
            "country_rows": country_rows,
            "paths": {country: str(path) for country, path in final_paths.items()},
        }
        manifest.setdefault("sources", {})[source] = source_entry
        _atomic_json(manifest_path, manifest)
        _event("normalized-source-complete", source=source, **source_entry)
    return manifest


def _load_country_source(
    manifest: Mapping[str, object], source: str, country: str
) -> pd.DataFrame:
    path = Path(manifest["sources"][source]["paths"][country])
    return pd.read_parquet(path)


def _load_cross_country_rows(
    manifest: Mapping[str, object],
    source1: pd.DataFrame,
    main_feed: pd.DataFrame,
    country: str,
    posting_cap: int,
) -> pd.DataFrame:
    """Load only globally exact-name rows needed by the validated fallback."""

    names = set(source1["name_full"]) - {""}
    relevant_parts = [main_feed.loc[main_feed["name_full"].isin(names)]]
    cross_parts: List[pd.DataFrame] = []
    for source in ("source2", "source3"):
        for other_country, raw_path in manifest["sources"][source]["paths"].items():
            if other_country == country:
                continue
            parquet = pq.ParquetFile(raw_path)
            for batch in parquet.iter_batches(batch_size=100_000):
                frame = batch.to_pandas()
                selected = frame.loc[frame["name_full"].isin(names)]
                if not selected.empty:
                    cross_parts.append(selected)
                    relevant_parts.append(selected)
    relevant = pd.concat(relevant_parts, ignore_index=True)
    counts = relevant["name_full"].value_counts()
    if not cross_parts:
        return main_feed.head(0).copy()
    cross = pd.concat(cross_parts, ignore_index=True)
    return cross.loc[cross["name_full"].map(counts).le(posting_cap)].copy()


def _checkpoint_paths(country_dir: Path, batch_number: int) -> Dict[str, Path]:
    stem = f"batch-{batch_number:06d}"
    return {
        "matching": country_dir / f"{stem}-matching.parquet",
        "candidate": country_dir / f"{stem}-candidate.parquet",
        "metadata": country_dir / f"{stem}.json",
    }


def _valid_batch_checkpoint(
    paths: Mapping[str, Path], run_hash: str, start: int, end: int
) -> bool:
    if not all(path.is_file() for path in paths.values()):
        return False
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    return (
        metadata.get("run_hash") == run_hash
        and metadata.get("start") == start
        and metadata.get("end") == end
    )


def run_full_production(
    config: PipelineConfig,
    ensemble: FrozenEnsemble,
    *,
    work_dir: Path,
    output_dir: Path,
    s1_batch_size: int,
) -> Dict[str, object]:
    """Run resumable full inference and invoke the official validator."""

    if s1_batch_size <= 0:
        raise ValueError("s1_batch_size must be positive")
    started = time.perf_counter()
    manifest = _partition_normalized_test(config, work_dir)
    countries = sorted(manifest["sources"]["source1"]["paths"])
    retrieval_payload = {
        "version": PRODUCTION_VERSION,
        "model_hashes": ensemble.artifact_hashes,
        "features": ensemble.feature_columns,
        "blend": {**COUNTRY_BLEND_CONFIG, "open_set": OPEN_SET_BLEND_CONFIG},
        "retrieval": {
            "name_top_k": 30,
            "address_top_k": 20,
            "legacy_quota": 35,
            "final_k": 50,
            "rrf": [2.0, 1.0, 1.0, 20.0],
            "address_strategy": strategy_from_config(config).__dict__,
        },
        "input_signatures": manifest["inputs"],
        "s1_batch_size": s1_batch_size,
    }
    run_hash = hashlib.sha256(
        json.dumps(retrieval_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    run_dir = work_dir / "runs" / run_hash
    _atomic_json(run_dir / "configuration.json", retrieval_payload)
    country_reports: Dict[str, object] = {}

    for country in countries:
        country_started = time.perf_counter()
        source1 = _load_country_source(manifest, "source1", country)
        feed = pd.concat(
            [
                _load_country_source(manifest, "source2", country),
                _load_country_source(manifest, "source3", country),
            ],
            ignore_index=True,
        )
        if source1.empty or feed.empty:
            raise RuntimeError(f"Country shard {country!r} is unexpectedly empty")
        cross = _load_cross_country_rows(
            manifest, source1, feed, country, config.posting_list_cap
        )
        feature_feed = pd.concat([feed, cross], ignore_index=True).drop_duplicates(
            "entity_id"
        )
        feature_feed_index = feature_feed.set_index("entity_id", verify_integrity=True)
        _event(
            "production-country-index-start",
            country=country,
            source1_rows=len(source1),
            feed_rows=len(feed),
            cross_country_exact_rows=len(cross),
        )
        legacy_index = prepare_legacy_index(feature_feed, config)
        name_index = prepare_name_tfidf_index(feed, config)
        address_index = prepare_address_tfidf_index(
            feed,
            config,
            maximum_numeric_df=config.address_preblock_numeric_max_df,
            maximum_rare_token_df=config.address_preblock_rare_token_max_df,
            maximum_postcode_prefix_df=config.address_preblock_postcode_prefix_max_df,
            postcode_prefix_length=config.address_preblock_postcode_prefix_length,
        )
        country_dir = run_dir / "shards" / _country_slug(country)
        batch_reports = []
        ranges = list(range(0, len(source1), s1_batch_size))
        for batch_number, start in enumerate(
            tqdm(ranges, desc=f"Score {country}", unit="batch")
        ):
            end = min(start + s1_batch_size, len(source1))
            paths = _checkpoint_paths(country_dir, batch_number)
            if _valid_batch_checkpoint(paths, run_hash, start, end):
                batch_reports.append({"start": start, "end": end, "reused": True})
                _event(
                    "production-batch-reused",
                    country=country,
                    batch_number=batch_number,
                    start=start,
                    end=end,
                )
                continue
            batch_started = time.perf_counter()
            batch = source1.iloc[start:end].copy()
            matching, candidate, profile = _retrieve_score_batch(
                batch,
                feature_feed,
                legacy_index,
                name_index,
                address_index,
                ensemble,
                config,
                feature_feed_index,
            )
            order = batch.set_index("entity_id")["_row_order"]
            matching["_row_order"] = matching["source1_entity_id"].map(order)
            candidate["_row_order"] = candidate["source1_entity_id"].map(order)
            _atomic_parquet(matching, paths["matching"])
            _atomic_parquet(candidate, paths["candidate"])
            metadata = {
                "run_hash": run_hash,
                "country": country,
                "batch_number": batch_number,
                "start": start,
                "end": end,
                "first_s1": str(batch["entity_id"].iloc[0]),
                "last_s1": str(batch["entity_id"].iloc[-1]),
                "profile": profile,
                "runtime_seconds": time.perf_counter() - batch_started,
            }
            _atomic_json(paths["metadata"], metadata)
            batch_reports.append(metadata)
            _event(
                "production-batch-complete",
                country=country,
                batch_number=batch_number,
                start=start,
                end=end,
                rows=len(batch),
                candidate_pairs=profile["candidate_pairs"],
                rows_per_second=len(batch) / metadata["runtime_seconds"],
                peak_rss_mib=_peak_rss_mib(),
                timing=profile["timing"],
            )
        generated_reports = [
            report for report in batch_reports if not report.get("reused", False)
        ]
        country_reports[country] = {
            "source1_rows": len(source1),
            "feed_rows": len(feed),
            "cross_country_exact_rows": len(cross),
            "batches": len(ranges),
            "runtime_seconds": time.perf_counter() - country_started,
            "decision": ensemble.decision_config(country),
            "batch_reports": batch_reports,
            "candidate_pairs_generated": sum(
                int(report["profile"]["candidate_pairs"])
                for report in generated_reports
            ),
            "peak_rss_mib": _peak_rss_mib(),
            "gpu": _gpu_snapshot(),
        }
        _event("production-country-complete", country=country, **country_reports[country])
        del source1, feed, cross, feature_feed
        del feature_feed_index
        del legacy_index, name_index, address_index
        gc.collect()

    matching_parts: List[pd.DataFrame] = []
    candidate_parts: List[pd.DataFrame] = []
    for country in countries:
        country_dir = run_dir / "shards" / _country_slug(country)
        source1_rows = int(manifest["sources"]["source1"]["country_rows"][country])
        for batch_number, start in enumerate(range(0, source1_rows, s1_batch_size)):
            paths = _checkpoint_paths(country_dir, batch_number)
            end = min(start + s1_batch_size, source1_rows)
            if not _valid_batch_checkpoint(paths, run_hash, start, end):
                raise RuntimeError(f"Missing production checkpoint: {paths}")
            matching_parts.append(pd.read_parquet(paths["matching"]))
            candidate_parts.append(pd.read_parquet(paths["candidate"]))
    matching = pd.concat(matching_parts, ignore_index=True).sort_values(
        "_row_order", kind="mergesort"
    )
    candidate = pd.concat(candidate_parts, ignore_index=True).sort_values(
        "_row_order", kind="mergesort"
    )
    matching.drop(columns="_row_order", inplace=True)
    candidate.drop(columns="_row_order", inplace=True)
    expected_s1 = sum(
        int(value)
        for value in manifest["sources"]["source1"]["country_rows"].values()
    )
    if len(matching) != expected_s1 or len(candidate) != expected_s1:
        raise DataValidationError("Assembled output row count does not match test Source-1")
    required_s1_ids: List[str] = []
    for country in countries:
        source1_path = Path(manifest["sources"]["source1"]["paths"][country])
        required_s1_ids.extend(
            pd.read_parquet(source1_path, columns=["entity_id"])["entity_id"].tolist()
        )
    validate_submission_frames(matching, candidate, required_s1_ids)
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
            str(config.test_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Official full validator failed:\n{completed.stdout}\n{completed.stderr}"
        )
    report = {
        "run_hash": run_hash,
        "countries": country_reports,
        "matching_path": str(matching_path),
        "candidate_path": str(candidate_path),
        "source1_rows": expected_s1,
        "candidate_pairs": int(
            candidate["candidate_entity_ids"].map(
                lambda value: len(value.split(",")) if value else 0
            ).sum()
        ),
        "empty_predictions": int(matching["matched_entity_ids"].eq("").sum()),
        "peak_rss_mib": _peak_rss_mib(),
        "official_validator": completed.stdout.strip(),
        "runtime_seconds": time.perf_counter() - started,
    }
    _atomic_json(run_dir / "production_report.json", report)
    _event("production-submission-complete", report=report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Run tiny real-data smoke test")
    mode.add_argument("--full", action="store_true", help="Run full checkpointed inference")
    parser.add_argument("--s1-batch-size", type=int, default=2_000)
    parser.add_argument(
        "--name-workers",
        type=int,
        default=4,
        help="Parallel exact sparse name-TFIDF query chunks",
    )
    parser.add_argument(
        "--xgb-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use CUDA for frozen XGBoost inference when available",
    )
    parser.add_argument("--smoke-s1-per-country", type=int, default=8)
    parser.add_argument("--smoke-feed-per-source-country", type=int, default=500)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_CONFIG.cache_dir / "production" / PRODUCTION_VERSION,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    if args.name_workers <= 0:
        raise ValueError("--name-workers must be positive")
    config = replace(
        _production_config(DEFAULT_CONFIG),
        tfidf_name_query_workers=args.name_workers,
    )
    hardware = _runtime_hardware(args.xgb_device)
    work_dir = args.work_dir
    normalized_reused = _normalized_checkpoints_valid(config, work_dir)
    ensemble = FrozenEnsemble(
        config.train_dir.parents[2], xgb_device=str(hardware["xgb_device"])
    )
    _event(
        "production-runtime-config",
        effective_s1_batch_size=args.s1_batch_size,
        name_tfidf_workers=args.name_workers,
        normalized_checkpoints_reused=normalized_reused,
        **hardware,
    )
    _event(
        "production-artifacts-verified",
        feature_count=len(ensemble.feature_columns),
        model_hashes=ensemble.artifact_hashes,
        blend={**COUNTRY_BLEND_CONFIG, "open_set": OPEN_SET_BLEND_CONFIG},
    )
    if args.smoke:
        output_dir = args.output_dir or config.output_dir / "smoke"
        run_smoke(
            config,
            ensemble,
            s1_per_country=args.smoke_s1_per_country,
            feed_per_source_country=args.smoke_feed_per_source_country,
            output_dir=output_dir,
        )
        return 0
    output_dir = args.output_dir or config.output_dir
    run_full_production(
        config,
        ensemble,
        work_dir=work_dir,
        output_dir=output_dir,
        s1_batch_size=args.s1_batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
