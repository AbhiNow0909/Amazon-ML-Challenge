"""Sparse character TF-IDF address retrieval over deterministic pre-blocks.

The E2.2 reference implementation compares every S1 address with every feed
address in its country.  This module preserves the fitted country-level TF-IDF
representation but restricts each sparse multiplication to a recall-oriented
union of postcode, numeric-token, and rare lexical-token postings.  Queries
without a usable pre-block take an explicit country-level fallback route.
"""

from __future__ import annotations

import resource
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import PipelineConfig


ROUTE_POSTCODE = 1
ROUTE_POSTCODE_PREFIX = 2
ROUTE_NUMERIC = 4
ROUTE_RARE_TOKEN = 8
ROUTE_FALLBACK = 16

ADDRESS_PREBLOCK_RESULT_COLUMNS = [
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
]


def _peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _sparse_bytes(matrix: sparse.spmatrix) -> int:
    compressed = matrix if sparse.isspmatrix_csr(matrix) or sparse.isspmatrix_csc(matrix) else matrix.tocsr()
    return int(
        compressed.data.nbytes
        + compressed.indices.nbytes
        + compressed.indptr.nbytes
    )


def _tokens(value: object) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(str(value or "").split()))


def _lexical_tokens(value: object, minimum_length: int) -> Tuple[str, ...]:
    return tuple(
        token
        for token in _tokens(value)
        if len(token) >= minimum_length and not token.isdigit()
    )


@dataclass(frozen=True)
class AddressPreblockStrategy:
    """One interpretable pre-block configuration."""

    name: str
    use_postcode: bool = True
    use_postcode_prefix: bool = False
    postcode_prefix_length: int = 3
    postcode_prefix_max_df: int = 2_000
    numeric_max_df: int = 2_000
    rare_token_max_df: int = 250
    rare_tokens_per_query: int = 2
    pool_cap: int = 10_000
    fallback_if_empty: bool = True
    fallback_char_features: int = 3
    fallback_char_max_df: int = 5_000

    def validate(self) -> None:
        if self.postcode_prefix_length <= 0:
            raise ValueError("postcode_prefix_length must be positive")
        for label, value in (
            ("postcode_prefix_max_df", self.postcode_prefix_max_df),
            ("numeric_max_df", self.numeric_max_df),
            ("rare_token_max_df", self.rare_token_max_df),
            ("pool_cap", self.pool_cap),
        ):
            if value <= 0:
                raise ValueError(f"{label} must be positive")
        if self.rare_tokens_per_query < 0:
            raise ValueError("rare_tokens_per_query cannot be negative")
        if self.fallback_char_features <= 0 or self.fallback_char_max_df <= 0:
            raise ValueError("fallback character limits must be positive")


@dataclass
class _CountryAddressIndex:
    country: str
    vectorizer: TfidfVectorizer
    matrix: sparse.csr_matrix
    fallback_matrix: sparse.csc_matrix
    candidate_ids: np.ndarray
    numeric_df: Counter
    rare_df: Counter
    postcode_df: Counter
    postcode_prefix_df: Counter
    numeric_postings: Mapping[str, np.ndarray]
    rare_postings: Mapping[str, np.ndarray]
    postcode_postings: Mapping[str, np.ndarray]
    postcode_prefix_postings: Mapping[str, np.ndarray]


@dataclass
class PreparedAddressIndex:
    """Reusable country matrices and pre-block postings for several variants."""

    shards: Mapping[str, _CountryAddressIndex]
    profile: Dict[str, object]
    minimum_token_length: int


def strategy_from_config(config: PipelineConfig, *, name: str = "configured") -> AddressPreblockStrategy:
    return AddressPreblockStrategy(
        name=name,
        use_postcode=config.address_preblock_use_postcode,
        use_postcode_prefix=config.address_preblock_use_postcode_prefix,
        postcode_prefix_length=config.address_preblock_postcode_prefix_length,
        postcode_prefix_max_df=config.address_preblock_postcode_prefix_max_df,
        numeric_max_df=config.address_preblock_numeric_max_df,
        rare_token_max_df=config.address_preblock_rare_token_max_df,
        rare_tokens_per_query=config.address_preblock_rare_tokens_per_query,
        pool_cap=config.address_preblock_pool_cap,
        fallback_if_empty=config.address_preblock_fallback_if_empty,
        fallback_char_features=config.address_preblock_fallback_char_features,
        fallback_char_max_df=config.address_preblock_fallback_char_max_df,
    )


def _count_tokens(values: Iterable[object], *, lexical: bool, minimum_length: int) -> Counter:
    counter: Counter = Counter()
    for value in values:
        tokens = _lexical_tokens(value, minimum_length) if lexical else _tokens(value)
        counter.update(tokens)
    return counter


def _postings(
    values: Sequence[object],
    df: Counter,
    *,
    maximum_df: int,
    lexical: bool,
    minimum_length: int,
) -> Dict[str, np.ndarray]:
    lists: DefaultDict[str, List[int]] = defaultdict(list)
    for position, value in enumerate(values):
        tokens = _lexical_tokens(value, minimum_length) if lexical else _tokens(value)
        for token in tokens:
            if df[token] <= maximum_df:
                lists[token].append(position)
    return {token: np.asarray(positions, dtype=np.int32) for token, positions in lists.items()}


def _exact_postings(values: Sequence[object]) -> Tuple[Counter, Dict[str, np.ndarray]]:
    lists: DefaultDict[str, List[int]] = defaultdict(list)
    for position, raw in enumerate(values):
        value = str(raw or "")
        if value:
            lists[value].append(position)
    df = Counter({value: len(positions) for value, positions in lists.items()})
    return df, {
        value: np.asarray(positions, dtype=np.int32)
        for value, positions in lists.items()
    }


def prepare_address_tfidf_index(
    feed: pd.DataFrame,
    config: PipelineConfig,
    *,
    maximum_numeric_df: int,
    maximum_rare_token_df: int,
    maximum_postcode_prefix_df: int,
    postcode_prefix_length: int = 3,
) -> PreparedAddressIndex:
    """Fit one sparse TF-IDF index per country and reusable posting lists."""

    required = {
        "entity_id",
        "address_core",
        "address_tokens",
        "address_numeric_tokens",
        "postcode",
        "country_norm",
    }
    missing = required.difference(feed.columns)
    if missing:
        raise ValueError(f"feed lacks address retrieval columns: {sorted(missing)}")
    if min(maximum_numeric_df, maximum_rare_token_df, maximum_postcode_prefix_df) <= 0:
        raise ValueError("maximum posting DFs must be positive")

    started = time.perf_counter()
    fit_seconds = 0.0
    matrix_seconds = 0.0
    posting_seconds = 0.0
    shards: Dict[str, _CountryAddressIndex] = {}
    shard_profiles: Dict[str, object] = {}
    total_nnz = 0
    total_sparse_bytes = 0
    total_fallback_sparse_bytes = 0
    total_vocabulary = 0
    total_posting_entries = 0

    for country in sorted(set(feed["country_norm"])):
        country_started = time.perf_counter()
        shard = feed.loc[feed["country_norm"] == country].reset_index(drop=True)
        if shard.empty:
            continue
        addresses = shard["address_core"].astype(str).tolist()
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(config.tfidf_address_ngram_min, config.tfidf_address_ngram_max),
            sublinear_tf=True,
            dtype=np.float32,
            min_df=config.tfidf_address_min_df,
            max_features=config.tfidf_address_max_features,
            lowercase=False,
            norm="l2",
        )
        fit_started = time.perf_counter()
        vectorizer.fit(addresses)
        shard_fit = time.perf_counter() - fit_started
        fit_seconds += shard_fit

        matrix_started = time.perf_counter()
        matrix = vectorizer.transform(addresses).tocsr()
        fallback_matrix = matrix.tocsc()
        shard_matrix = time.perf_counter() - matrix_started
        matrix_seconds += shard_matrix
        if not sparse.isspmatrix_csr(matrix):
            raise AssertionError("Address TF-IDF index unexpectedly became dense")

        posting_started = time.perf_counter()
        numeric_values = shard["address_numeric_tokens"].astype(str).tolist()
        lexical_values = shard["address_tokens"].astype(str).tolist()
        postcode_values = shard["postcode"].astype(str).tolist()
        prefix_values = [
            value[:postcode_prefix_length] if len(value) >= postcode_prefix_length else ""
            for value in postcode_values
        ]
        numeric_df = _count_tokens(
            numeric_values, lexical=False, minimum_length=config.minimum_token_length
        )
        rare_df = _count_tokens(
            lexical_values, lexical=True, minimum_length=config.minimum_token_length
        )
        postcode_df, postcode_postings = _exact_postings(postcode_values)
        postcode_prefix_df, all_prefix_postings = _exact_postings(prefix_values)
        numeric_postings = _postings(
            numeric_values,
            numeric_df,
            maximum_df=maximum_numeric_df,
            lexical=False,
            minimum_length=config.minimum_token_length,
        )
        rare_postings = _postings(
            lexical_values,
            rare_df,
            maximum_df=maximum_rare_token_df,
            lexical=True,
            minimum_length=config.minimum_token_length,
        )
        postcode_prefix_postings = {
            token: positions
            for token, positions in all_prefix_postings.items()
            if postcode_prefix_df[token] <= maximum_postcode_prefix_df
        }
        shard_posting = time.perf_counter() - posting_started
        posting_seconds += shard_posting
        posting_entries = sum(
            len(positions)
            for mapping in (
                numeric_postings,
                rare_postings,
                postcode_postings,
                postcode_prefix_postings,
            )
            for positions in mapping.values()
        )
        total_posting_entries += posting_entries

        index = _CountryAddressIndex(
            country=country,
            vectorizer=vectorizer,
            matrix=matrix,
            fallback_matrix=fallback_matrix,
            candidate_ids=shard["entity_id"].to_numpy(dtype=object),
            numeric_df=numeric_df,
            rare_df=rare_df,
            postcode_df=postcode_df,
            postcode_prefix_df=postcode_prefix_df,
            numeric_postings=numeric_postings,
            rare_postings=rare_postings,
            postcode_postings=postcode_postings,
            postcode_prefix_postings=postcode_prefix_postings,
        )
        shards[country] = index
        index_bytes = _sparse_bytes(matrix)
        total_nnz += int(matrix.nnz)
        total_sparse_bytes += index_bytes
        fallback_bytes = _sparse_bytes(fallback_matrix)
        total_fallback_sparse_bytes += fallback_bytes
        total_vocabulary += len(vectorizer.vocabulary_)
        shard_profiles[country] = {
            "index_rows": len(shard),
            "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "vocabulary_size": len(vectorizer.vocabulary_),
            "index_nnz": int(matrix.nnz),
            "index_sparse_bytes": index_bytes,
            "fallback_csc_sparse_bytes": fallback_bytes,
            "numeric_token_count": len(numeric_df),
            "rare_token_count": len(rare_df),
            "postcode_count": len(postcode_df),
            "postcode_prefix_count": len(postcode_prefix_df),
            "posting_entries": posting_entries,
            "fit_seconds": shard_fit,
            "matrix_seconds": shard_matrix,
            "posting_seconds": shard_posting,
            "total_seconds": time.perf_counter() - country_started,
        }

    profile: Dict[str, object] = {
        "vectorizer_fit_seconds": fit_seconds,
        "sparse_matrix_construction_seconds": matrix_seconds,
        "posting_index_seconds": posting_seconds,
        "total_build_seconds": time.perf_counter() - started,
        "total_vocabulary_across_shards": total_vocabulary,
        "total_index_nnz": total_nnz,
        "total_index_sparse_bytes": total_sparse_bytes,
        "total_fallback_csc_sparse_bytes": total_fallback_sparse_bytes,
        "total_sparse_resident_bytes": total_sparse_bytes
        + total_fallback_sparse_bytes,
        "total_posting_entries": total_posting_entries,
        "posting_payload_bytes_lower_bound": total_posting_entries * 4,
        "dense_similarity_constructed": False,
        "peak_rss_mib": _peak_rss_mib(),
        "shards": shard_profiles,
    }
    return PreparedAddressIndex(
        shards=shards,
        profile=profile,
        minimum_token_length=config.minimum_token_length,
    )


def _add_posting(
    support: Dict[int, List[int]],
    positions: Optional[np.ndarray],
    route: int,
    df: int,
) -> None:
    if positions is None:
        return
    for raw_position in positions:
        position = int(raw_position)
        existing = support.get(position)
        if existing is None:
            support[position] = [route, df]
        else:
            existing[0] |= route
            existing[1] = min(existing[1], df)


def _candidate_pool(
    row: object,
    shard: _CountryAddressIndex,
    strategy: AddressPreblockStrategy,
    minimum_token_length: int,
) -> Tuple[np.ndarray, Dict[int, int], bool, int]:
    support: Dict[int, List[int]] = {}
    postcode = str(getattr(row, "postcode") or "")
    if strategy.use_postcode and postcode:
        _add_posting(
            support,
            shard.postcode_postings.get(postcode),
            ROUTE_POSTCODE,
            int(shard.postcode_df.get(postcode, 0)),
        )
    if strategy.use_postcode_prefix and len(postcode) >= strategy.postcode_prefix_length:
        prefix = postcode[: strategy.postcode_prefix_length]
        prefix_df = int(shard.postcode_prefix_df.get(prefix, 0))
        if 0 < prefix_df <= strategy.postcode_prefix_max_df:
            _add_posting(
                support,
                shard.postcode_prefix_postings.get(prefix),
                ROUTE_POSTCODE_PREFIX,
                prefix_df,
            )

    for token in _tokens(getattr(row, "address_numeric_tokens")):
        df = int(shard.numeric_df.get(token, 0))
        if 0 < df <= strategy.numeric_max_df:
            _add_posting(
                support, shard.numeric_postings.get(token), ROUTE_NUMERIC, df
            )

    eligible_rare = []
    for token in _lexical_tokens(
        getattr(row, "address_tokens"), minimum_token_length
    ):
        df = int(shard.rare_df.get(token, 0))
        if 0 < df <= strategy.rare_token_max_df:
            eligible_rare.append((df, token))
    eligible_rare.sort(key=lambda item: (item[0], item[1]))
    for df, token in eligible_rare[: strategy.rare_tokens_per_query]:
        _add_posting(
            support, shard.rare_postings.get(token), ROUTE_RARE_TOKEN, df
        )

    fallback = not support and strategy.fallback_if_empty and bool(
        str(getattr(row, "address_core") or "")
    )
    raw_pool_size = len(support)
    if fallback:
        positions = np.asarray([], dtype=np.int32)
        flags = {}
    else:
        if len(support) > strategy.pool_cap:
            selected = sorted(
                support,
                key=lambda position: (
                    -bin(int(support[position][0])).count("1"),
                    support[position][1],
                    position,
                ),
            )[: strategy.pool_cap]
        else:
            selected = sorted(support)
        positions = np.asarray(selected, dtype=np.int32)
        flags = {position: support[position][0] for position in selected}
    return positions, flags, fallback, raw_pool_size


def _character_fallback_pool(
    query_row: sparse.csr_matrix,
    shard: _CountryAddressIndex,
    strategy: AddressPreblockStrategy,
) -> Tuple[np.ndarray, int]:
    """Bound an otherwise broad fallback with rare active TF-IDF features."""

    query = query_row.tocsr(copy=False)
    features: List[Tuple[int, float, int]] = []
    csc = shard.fallback_matrix
    for feature, weight in zip(query.indices.tolist(), query.data.tolist()):
        df = int(csc.indptr[feature + 1] - csc.indptr[feature])
        if df:
            features.append((df, -float(weight), int(feature)))
    if not features:
        return np.asarray([], dtype=np.int32), 0
    features.sort()
    eligible = [item for item in features if item[0] <= strategy.fallback_char_max_df]
    selected_features = (eligible or features)[: strategy.fallback_char_features]
    support: Dict[int, List[int]] = {}
    for df, _, feature in selected_features:
        start = csc.indptr[feature]
        end = csc.indptr[feature + 1]
        _add_posting(
            support,
            csc.indices[start:end],
            ROUTE_FALLBACK,
            df,
        )
    raw_size = len(support)
    selected = sorted(
        support,
        key=lambda position: (support[position][1], position),
    )[: strategy.pool_cap]
    return np.asarray(selected, dtype=np.int32), raw_size


def _rank_sparse_scores(
    similarities: sparse.csr_matrix,
    pool_positions: np.ndarray,
    candidate_ids: np.ndarray,
    top_k: int,
) -> List[Tuple[int, float]]:
    if not sparse.issparse(similarities):
        raise AssertionError("Address pre-block similarities became dense")
    row = similarities.tocsr(copy=False)
    local_indices = row.indices
    scores = row.data
    positive = scores > 0
    local_indices = local_indices[positive]
    scores = scores[positive]
    if len(scores) > top_k:
        keep = np.argpartition(scores, -top_k)[-top_k:]
        local_indices = local_indices[keep]
        scores = scores[keep]
    ranked = [
        (int(pool_positions[int(local)]), float(score))
        for local, score in zip(local_indices.tolist(), scores.tolist())
    ]
    ranked.sort(key=lambda item: (-item[1], str(candidate_ids[item[0]])))
    return ranked


def retrieve_from_prepared_address_index(
    source1: pd.DataFrame,
    prepared: PreparedAddressIndex,
    strategy: AddressPreblockStrategy,
    *,
    top_k: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Retrieve address neighbors using only sparse, per-query pre-block pools."""

    strategy.validate()
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    required = {
        "entity_id",
        "address_core",
        "address_tokens",
        "address_numeric_tokens",
        "postcode",
        "country_norm",
    }
    missing = required.difference(source1.columns)
    if missing:
        raise ValueError(f"source1 lacks address retrieval columns: {sorted(missing)}")

    started = time.perf_counter()
    query_transform_seconds = 0.0
    pool_seconds = 0.0
    multiplication_seconds = 0.0
    extraction_seconds = 0.0
    rows: List[Dict[str, object]] = []
    shard_profiles: Dict[str, object] = {}
    total_comparisons = 0
    total_similarity_nnz = 0
    total_fallback_queries = 0
    pool_sizes: List[int] = []

    for country in sorted(set(source1["country_norm"])):
        query_shard = source1.loc[source1["country_norm"] == country]
        index = prepared.shards.get(country)
        if index is None or query_shard.empty:
            shard_profiles[country] = {
                "query_rows": len(query_shard),
                "skipped": True,
            }
            continue
        transform_started = time.perf_counter()
        query_matrix = index.vectorizer.transform(
            query_shard["address_core"].astype(str).tolist()
        ).tocsr()
        shard_transform = time.perf_counter() - transform_started
        query_transform_seconds += shard_transform
        shard_comparisons = 0
        shard_nnz = 0
        shard_fallback = 0
        shard_pool_sizes: List[int] = []

        for query_position, row in enumerate(query_shard.itertuples(index=False)):
            pool_started = time.perf_counter()
            positions, flags, fallback, raw_pool_size = _candidate_pool(
                row, index, strategy, prepared.minimum_token_length
            )
            pool_seconds += time.perf_counter() - pool_started
            if fallback:
                fallback_started = time.perf_counter()
                positions, raw_pool_size = _character_fallback_pool(
                    query_matrix[query_position], index, strategy
                )
                pool_seconds += time.perf_counter() - fallback_started
            if not len(positions):
                continue
            pool_size = len(positions)
            pool_sizes.append(pool_size)
            shard_pool_sizes.append(pool_size)
            total_comparisons += pool_size
            shard_comparisons += pool_size
            if fallback:
                total_fallback_queries += 1
                shard_fallback += 1

            multiplication_started = time.perf_counter()
            comparison_matrix = index.matrix[positions]
            similarities = (
                query_matrix[query_position] @ comparison_matrix.T
            ).tocsr()
            multiplication_seconds += time.perf_counter() - multiplication_started
            shard_nnz += int(similarities.nnz)
            total_similarity_nnz += int(similarities.nnz)

            extraction_started = time.perf_counter()
            ranked = _rank_sparse_scores(
                similarities, positions, index.candidate_ids, top_k
            )
            extraction_seconds += time.perf_counter() - extraction_started
            source1_id = str(getattr(row, "entity_id"))
            for rank, (candidate_position, score) in enumerate(ranked, 1):
                route = ROUTE_FALLBACK if fallback else flags.get(candidate_position, 0)
                rows.append(
                    {
                        "source1_entity_id": source1_id,
                        "candidate_entity_id": str(index.candidate_ids[candidate_position]),
                        "address_tfidf_similarity": np.float32(score),
                        "address_tfidf_rank": rank,
                        "address_route_postcode": int(bool(route & ROUTE_POSTCODE)),
                        "address_route_postcode_prefix": int(
                            bool(route & ROUTE_POSTCODE_PREFIX)
                        ),
                        "address_route_numeric": int(bool(route & ROUTE_NUMERIC)),
                        "address_route_rare_token": int(
                            bool(route & ROUTE_RARE_TOKEN)
                        ),
                        "address_route_country_fallback": int(
                            bool(route & ROUTE_FALLBACK)
                        ),
                        "address_route_count": bin(int(route & 15)).count("1")
                        if not fallback
                        else 1,
                        "address_preblock_pool_size": raw_pool_size,
                    }
                )

        shard_profiles[country] = {
            "query_rows": len(query_shard),
            "index_rows": len(index.candidate_ids),
            "query_transform_seconds": shard_transform,
            "candidate_comparisons": shard_comparisons,
            "similarity_nnz": shard_nnz,
            "fallback_queries": shard_fallback,
            "average_pool_size": float(np.mean(shard_pool_sizes))
            if shard_pool_sizes
            else 0.0,
            "p95_pool_size": float(np.percentile(shard_pool_sizes, 95))
            if shard_pool_sizes
            else 0.0,
            "maximum_pool_size": max(shard_pool_sizes, default=0),
            "skipped": False,
        }

    result = pd.DataFrame(rows, columns=ADDRESS_PREBLOCK_RESULT_COLUMNS)
    if not result.empty:
        result["address_tfidf_similarity"] = result[
            "address_tfidf_similarity"
        ].astype("float32")
        for column in ADDRESS_PREBLOCK_RESULT_COLUMNS[3:10]:
            result[column] = pd.to_numeric(result[column], downcast="unsigned")
        if result.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
            raise AssertionError("Pre-blocked address retrieval emitted duplicate pairs")
        if result.groupby("source1_entity_id").size().max() > top_k:
            raise AssertionError("Pre-blocked address retrieval exceeded top_k")

    elapsed = time.perf_counter() - started
    profile: Dict[str, object] = {
        "strategy": strategy.__dict__,
        "query_rows": len(source1),
        "result_pairs": len(result),
        "query_transform_seconds": query_transform_seconds,
        "preblock_pool_seconds": pool_seconds,
        "similarity_multiplication_seconds": multiplication_seconds,
        "top_k_extraction_seconds": extraction_seconds,
        "total_query_seconds": elapsed,
        "queries_per_second": len(source1) / elapsed if elapsed else 0.0,
        "candidate_comparisons": total_comparisons,
        "average_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        "median_pool_size": float(np.median(pool_sizes)) if pool_sizes else 0.0,
        "p95_pool_size": float(np.percentile(pool_sizes, 95)) if pool_sizes else 0.0,
        "maximum_pool_size": max(pool_sizes, default=0),
        "fallback_queries": total_fallback_queries,
        "fallback_fraction": total_fallback_queries / len(source1) if len(source1) else 0.0,
        "similarity_nnz": total_similarity_nnz,
        "dense_similarity_constructed": False,
        "peak_rss_mib": _peak_rss_mib(),
        "shards": shard_profiles,
    }
    return result, profile


def diagnose_address_preblock_pools(
    source1: pd.DataFrame,
    prepared: PreparedAddressIndex,
    strategy: AddressPreblockStrategy,
) -> Dict[str, object]:
    """Measure pool sizes and fallback burden without computing similarities."""

    strategy.validate()
    started = time.perf_counter()
    pool_sizes: List[int] = []
    fallback_queries = 0
    queries_with_no_candidates = 0
    for country in sorted(set(source1["country_norm"])):
        index = prepared.shards.get(country)
        if index is None:
            continue
        query_shard = source1.loc[source1["country_norm"] == country]
        query_matrix = index.vectorizer.transform(
            query_shard["address_core"].astype(str).tolist()
        ).tocsr()
        for query_position, row in enumerate(query_shard.itertuples(index=False)):
            positions, _, fallback, _ = _candidate_pool(
                row, index, strategy, prepared.minimum_token_length
            )
            if fallback:
                positions, _ = _character_fallback_pool(
                    query_matrix[query_position], index, strategy
                )
            if not len(positions):
                queries_with_no_candidates += 1
                continue
            pool_sizes.append(len(positions))
            fallback_queries += int(fallback)
    return {
        "strategy": strategy.__dict__,
        "query_rows": len(source1),
        "queries_with_candidates": len(pool_sizes),
        "queries_with_no_candidates": queries_with_no_candidates,
        "fallback_queries": fallback_queries,
        "fallback_fraction": fallback_queries / len(source1) if len(source1) else 0.0,
        "candidate_comparisons": int(sum(pool_sizes)),
        "average_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        "median_pool_size": float(np.median(pool_sizes)) if pool_sizes else 0.0,
        "p95_pool_size": float(np.percentile(pool_sizes, 95)) if pool_sizes else 0.0,
        "maximum_pool_size": max(pool_sizes, default=0),
        "diagnostic_seconds": time.perf_counter() - started,
    }


def retrieve_address_tfidf_preblocked(
    source1: pd.DataFrame,
    feed: pd.DataFrame,
    config: PipelineConfig,
    *,
    strategy: Optional[AddressPreblockStrategy] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Convenience wrapper for one pre-blocked address retrieval run."""

    selected = strategy or strategy_from_config(config)
    prepared = prepare_address_tfidf_index(
        feed,
        config,
        maximum_numeric_df=selected.numeric_max_df,
        maximum_rare_token_df=selected.rare_token_max_df,
        maximum_postcode_prefix_df=selected.postcode_prefix_max_df,
        postcode_prefix_length=selected.postcode_prefix_length,
    )
    candidates, query_profile = retrieve_from_prepared_address_index(
        source1, prepared, selected, top_k=config.tfidf_address_top_k
    )
    return candidates, {"build": prepared.profile, "query": query_profile}
