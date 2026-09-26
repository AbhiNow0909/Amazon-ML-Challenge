"""Sparse, country-sharded character TF-IDF retrieval."""

from __future__ import annotations

import resource
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import PipelineConfig


RESULT_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "name_tfidf_similarity",
    "name_tfidf_rank",
]
ADDRESS_RESULT_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "address_tfidf_similarity",
    "address_tfidf_rank",
]


def _peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _sparse_bytes(matrix: sparse.spmatrix) -> int:
    csr = matrix.tocsr(copy=False)
    return int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes)


def _row_top_k(
    similarities: sparse.csr_matrix,
    query_ids: np.ndarray,
    candidate_ids: np.ndarray,
    top_k: int,
    *,
    similarity_column: str,
    rank_column: str,
) -> List[Dict[str, object]]:
    """Extract row-wise top K directly from CSR buffers without densifying."""

    rows: List[Dict[str, object]] = []
    for row_index, query_id in enumerate(query_ids):
        start = similarities.indptr[row_index]
        end = similarities.indptr[row_index + 1]
        indices = similarities.indices[start:end]
        scores = similarities.data[start:end]
        if not len(scores):
            continue
        positive = scores > 0
        indices = indices[positive]
        scores = scores[positive]
        if len(scores) > top_k:
            selected = np.argpartition(scores, -top_k)[-top_k:]
            indices = indices[selected]
            scores = scores[selected]
        ordered = sorted(
            zip(indices.tolist(), scores.tolist()),
            key=lambda pair: (-pair[1], str(candidate_ids[pair[0]])),
        )
        for rank, (candidate_position, score) in enumerate(ordered, 1):
            rows.append(
                {
                    "source1_entity_id": str(query_id),
                    "candidate_entity_id": str(candidate_ids[candidate_position]),
                    similarity_column: np.float32(score),
                    rank_column: rank,
                }
            )
    return rows


def _retrieve_character_tfidf(
    source1: pd.DataFrame,
    feed: pd.DataFrame,
    *,
    field: str,
    channel: str,
    top_k: int,
    ngram_range: Tuple[int, int],
    query_chunk_size: int,
    max_features: Optional[int],
    min_df: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Run one bounded sparse retrieval channel over dynamic country shards."""

    required = {"entity_id", field, "country_norm"}
    for label, frame in (("source1", source1), ("feed", feed)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} lacks TF-IDF columns: {sorted(missing)}")
    if top_k <= 0:
        raise ValueError("TF-IDF top_k must be positive")
    if query_chunk_size <= 0:
        raise ValueError("tfidf_query_chunk_size must be positive")

    similarity_column = f"{channel}_tfidf_similarity"
    rank_column = f"{channel}_tfidf_rank"
    result_columns = [
        "source1_entity_id",
        "candidate_entity_id",
        similarity_column,
        rank_column,
    ]
    total_started = time.perf_counter()
    fit_seconds = 0.0
    index_seconds = 0.0
    query_transform_seconds = 0.0
    similarity_seconds = 0.0
    extraction_seconds = 0.0
    result_rows: List[Dict[str, object]] = []
    shard_profiles: Dict[str, object] = {}
    maximum_similarity_chunk_nnz = 0
    total_index_nnz = 0
    total_index_sparse_bytes = 0
    total_vocabulary = 0

    countries = sorted(set(source1["country_norm"]))
    for country in countries:
        query_shard = source1.loc[source1["country_norm"] == country]
        feed_shard = feed.loc[feed["country_norm"] == country]
        if query_shard.empty or feed_shard.empty:
            shard_profiles[country] = {
                "query_rows": len(query_shard),
                "index_rows": len(feed_shard),
                "skipped": True,
            }
            continue

        feed_values = feed_shard[field].astype(str).tolist()
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            sublinear_tf=True,
            dtype=np.float32,
            min_df=min_df,
            max_features=max_features,
            lowercase=False,
            norm="l2",
        )
        fit_started = time.perf_counter()
        try:
            vectorizer.fit(feed_values)
        except ValueError as exc:
            if "empty vocabulary" not in str(exc).lower():
                raise
            shard_profiles[country] = {
                "query_rows": len(query_shard),
                "index_rows": len(feed_shard),
                "skipped": True,
                "reason": "empty vocabulary",
            }
            continue
        shard_fit_seconds = time.perf_counter() - fit_started
        fit_seconds += shard_fit_seconds

        index_started = time.perf_counter()
        index_matrix = vectorizer.transform(feed_values).tocsr()
        shard_index_seconds = time.perf_counter() - index_started
        index_seconds += shard_index_seconds
        if not sparse.isspmatrix_csr(index_matrix):
            raise AssertionError("TF-IDF index unexpectedly became dense")

        candidate_ids = feed_shard["entity_id"].to_numpy(dtype=object)
        query_ids = query_shard["entity_id"].to_numpy(dtype=object)
        query_values = query_shard[field].astype(str).to_numpy(dtype=object)
        shard_query_nnz = 0
        shard_similarity_nnz = 0
        shard_max_chunk_nnz = 0

        for start in range(0, len(query_shard), query_chunk_size):
            end = min(start + query_chunk_size, len(query_shard))
            transform_started = time.perf_counter()
            query_matrix = vectorizer.transform(query_values[start:end]).tocsr()
            query_transform_seconds += time.perf_counter() - transform_started
            shard_query_nnz += int(query_matrix.nnz)

            similarity_started = time.perf_counter()
            similarities = (query_matrix @ index_matrix.T).tocsr()
            similarity_seconds += time.perf_counter() - similarity_started
            if not sparse.isspmatrix_csr(similarities):
                raise AssertionError("Similarity multiplication unexpectedly became dense")
            chunk_nnz = int(similarities.nnz)
            shard_similarity_nnz += chunk_nnz
            shard_max_chunk_nnz = max(shard_max_chunk_nnz, chunk_nnz)
            maximum_similarity_chunk_nnz = max(maximum_similarity_chunk_nnz, chunk_nnz)

            extraction_started = time.perf_counter()
            result_rows.extend(
                _row_top_k(
                    similarities,
                    query_ids[start:end],
                    candidate_ids,
                    top_k,
                    similarity_column=similarity_column,
                    rank_column=rank_column,
                )
            )
            extraction_seconds += time.perf_counter() - extraction_started

        vocabulary_size = len(vectorizer.vocabulary_)
        index_bytes = _sparse_bytes(index_matrix)
        total_vocabulary += vocabulary_size
        total_index_nnz += int(index_matrix.nnz)
        total_index_sparse_bytes += index_bytes
        shard_profiles[country] = {
            "query_rows": len(query_shard),
            "index_rows": len(feed_shard),
            "matrix_shape": [int(index_matrix.shape[0]), int(index_matrix.shape[1])],
            "vocabulary_size": vocabulary_size,
            "index_nnz": int(index_matrix.nnz),
            "index_sparse_bytes": index_bytes,
            "query_nnz": shard_query_nnz,
            "similarity_nnz": shard_similarity_nnz,
            "maximum_similarity_chunk_nnz": shard_max_chunk_nnz,
            "vectorizer_fit_seconds": shard_fit_seconds,
            "index_transform_seconds": shard_index_seconds,
            "skipped": False,
        }

    result = pd.DataFrame(result_rows, columns=result_columns)
    if not result.empty:
        result[similarity_column] = result[similarity_column].astype("float32")
        result[rank_column] = pd.to_numeric(result[rank_column], downcast="unsigned")
        if result.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
            raise AssertionError("TF-IDF retrieval emitted a duplicate pair")
        if len(result) > len(source1) * top_k:
            raise AssertionError("TF-IDF retrieval exceeded its top-K bound")

    profile: Dict[str, object] = {
        "channel": channel,
        "field": field,
        "analyzer": "char_wb",
        "ngram_range": list(ngram_range),
        "dtype": "float32",
        "top_k": top_k,
        "query_chunk_size": query_chunk_size,
        "max_features": max_features,
        "min_df": min_df,
        "candidate_pairs": len(result),
        "vectorizer_fit_seconds": fit_seconds,
        "index_transform_seconds": index_seconds,
        "query_transform_seconds": query_transform_seconds,
        "similarity_query_seconds": similarity_seconds,
        "top_k_extraction_seconds": extraction_seconds,
        "total_runtime_seconds": time.perf_counter() - total_started,
        "total_vocabulary_across_shards": total_vocabulary,
        "total_index_nnz": total_index_nnz,
        "total_index_sparse_bytes": total_index_sparse_bytes,
        "maximum_similarity_chunk_nnz": maximum_similarity_chunk_nnz,
        "peak_rss_mib": _peak_rss_mib(),
        "dense_similarity_constructed": False,
        "shards": shard_profiles,
    }
    return result, profile


def retrieve_name_tfidf(
    source1: pd.DataFrame,
    feed: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Retrieve bounded name candidates from sparse country-specific indexes."""

    return _retrieve_character_tfidf(
        source1,
        feed,
        field="name_core",
        channel="name",
        top_k=config.tfidf_name_top_k,
        ngram_range=(config.tfidf_ngram_min, config.tfidf_ngram_max),
        query_chunk_size=config.tfidf_query_chunk_size,
        max_features=config.tfidf_max_features,
        min_df=config.tfidf_min_df,
    )


def retrieve_address_tfidf(
    source1: pd.DataFrame,
    feed: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Retrieve bounded normalized-address candidates without densifying."""

    return _retrieve_character_tfidf(
        source1,
        feed,
        field="address_core",
        channel="address",
        top_k=config.tfidf_address_top_k,
        ngram_range=(
            config.tfidf_address_ngram_min,
            config.tfidf_address_ngram_max,
        ),
        query_chunk_size=config.tfidf_address_query_chunk_size,
        max_features=config.tfidf_address_max_features,
        min_df=config.tfidf_address_min_df,
    )
