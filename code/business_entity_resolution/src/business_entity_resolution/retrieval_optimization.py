"""E2.3 address-TFIDF pre-block optimization on the cached E2.2 universe."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import resource
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Set, Tuple

import numpy as np
import pandas as pd

from .address_preblocked_retrieval import (
    AddressPreblockStrategy,
    diagnose_address_preblock_pools,
    prepare_address_tfidf_index,
    retrieve_from_prepared_address_index,
)
from .candidate_fusion import prepare_fusion_universe, reciprocal_rank_fusion
from .config import DEFAULT_CONFIG, PipelineConfig
from .evaluation import candidate_metrics
from .io_utils import write_parquet_cache
from .multi_channel_fusion import (
    prepare_multi_channel_universe,
    protected_legacy_multi_channel_rrf,
)
from .retrieval_repair import (
    _pair_set,
    _remaining_failure_categories,
    _retrieval_bundle,
    _truth_pairs,
)
from .split import split_source1_ids
from .stress_test import _json_default, _load_truth, _retrieval_failure_category


E21_ARTIFACT_ID = "c449c0c6b281"
E22_ARTIFACT_ID = "e7c71a736237"
E22_REPORT_NAME = f"e22_retrieval_{E22_ARTIFACT_ID}_metrics.json"


def _event(name: str, **values: object) -> None:
    print(json.dumps({"event": name, **values}, default=_json_default), flush=True)


def _peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _artifact_hash(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _address_channel_metrics(
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    truth: Mapping[str, Set[str]],
    validation_ids: Set[str],
    base_pairs: Set[Tuple[str, str]],
) -> Dict[str, object]:
    bundle = _retrieval_bundle(candidates, source1, truth, validation_ids)
    all_truth = _truth_pairs(truth)
    pairs = _pair_set(candidates)
    held_truth = {
        pair for pair in all_truth if pair[0] in validation_ids
    }
    return {
        **bundle,
        "true_links_retrieved": len(all_truth & pairs),
        "heldout_true_links_retrieved": len(held_truth & pairs),
        "incremental_true_links_vs_legacy_name": len(
            (all_truth - base_pairs) & pairs
        ),
        "heldout_incremental_true_links_vs_legacy_name": len(
            (held_truth - base_pairs) & pairs
        ),
    }


def _strategy_payload(strategy: AddressPreblockStrategy) -> Dict[str, object]:
    return dict(strategy.__dict__)


def _examples(
    pairs: Set[Tuple[str, str]],
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    feed: pd.DataFrame,
    *,
    limit: int = 12,
) -> List[Dict[str, object]]:
    left = source1.set_index("entity_id", verify_integrity=True)
    right = feed.set_index("entity_id", verify_integrity=True)
    lookup = candidates.set_index(
        ["source1_entity_id", "candidate_entity_id"], verify_integrity=True
    )
    result: List[Dict[str, object]] = []
    for pair in sorted(pairs)[:limit]:
        s1, target = pair
        row = lookup.loc[pair] if pair in lookup.index else None
        result.append(
            {
                "source1_entity_id": s1,
                "candidate_entity_id": target,
                "source1_name": left.at[s1, "business_name"],
                "candidate_name": right.at[target, "business_name"],
                "source1_address": left.at[s1, "business_address"],
                "candidate_address": right.at[target, "business_address"],
                "country": left.at[s1, "country_norm"],
                "failure_category": _retrieval_failure_category(
                    left.loc[s1], right.loc[target]
                ),
                "address_tfidf_similarity": float(row["address_tfidf_similarity"])
                if row is not None
                else None,
                "address_tfidf_rank": int(row["address_tfidf_rank"])
                if row is not None
                else None,
                "routes": {
                    column: int(row[column])
                    for column in (
                        "address_route_postcode",
                        "address_route_postcode_prefix",
                        "address_route_numeric",
                        "address_route_rare_token",
                        "address_route_country_fallback",
                    )
                    if row is not None and column in row.index
                },
            }
        )
    return result


def _category_recovery(
    failures: Set[Tuple[str, str]],
    original_pairs: Set[Tuple[str, str]],
    optimized_pairs: Set[Tuple[str, str]],
    source1: pd.DataFrame,
    feed: pd.DataFrame,
) -> Dict[str, object]:
    left = source1.set_index("entity_id", verify_integrity=True)
    right = feed.set_index("entity_id", verify_integrity=True)
    grouped: Dict[str, Set[Tuple[str, str]]] = {}
    for pair in failures:
        category = _retrieval_failure_category(left.loc[pair[0]], right.loc[pair[1]])
        grouped.setdefault(category, set()).add(pair)
    return {
        category: {
            "previous_failures": len(pairs),
            "original_address_recovered": len(pairs & original_pairs),
            "optimized_address_recovered": len(pairs & optimized_pairs),
            "optimized_final_recovered": None,
        }
        for category, pairs in sorted(grouped.items())
    }


def _runtime_projection(
    build_profile: Mapping[str, object],
    query_profile: Mapping[str, object],
    *,
    validation_feed_rows: int,
    validation_queries: int,
    test_feed_rows: int = 9_969_589,
    test_queries: int = 1_732_544,
) -> Dict[str, object]:
    feed_scale = test_feed_rows / validation_feed_rows
    query_scale = test_queries / validation_queries
    build_seconds = float(build_profile["total_build_seconds"]) * feed_scale
    query_seconds = float(query_profile["total_query_seconds"]) * query_scale
    matrix_bytes = int(build_profile["total_index_sparse_bytes"] * feed_scale)
    posting_lower_bound = int(
        build_profile["posting_payload_bytes_lower_bound"] * feed_scale
    )
    return {
        "test_source1_rows": test_queries,
        "test_feed_rows": test_feed_rows,
        "feed_scale": feed_scale,
        "query_scale": query_scale,
        "projected_build_seconds_linear_observed": build_seconds,
        "projected_query_seconds_observed_throughput": query_seconds,
        "projected_total_seconds": build_seconds + query_seconds,
        "projected_total_hours": (build_seconds + query_seconds) / 3600.0,
        "projected_sparse_matrix_bytes": matrix_bytes,
        "projected_posting_payload_bytes_lower_bound": posting_lower_bound,
        "note": (
            "Projection uses measured build throughput and per-query scoped retrieval. "
            "Production should process country shards sequentially to bound dataframe, "
            "matrix, and posting-list residency."
        ),
    }


def run_e23_experiment(
    config: PipelineConfig = DEFAULT_CONFIG, *, persist: bool = True
) -> Dict[str, object]:
    """Profile E2.2 and compare bounded address pre-block variants."""

    total_started = time.perf_counter()
    report_path = config.experiments_dir / E22_REPORT_NAME
    e22 = json.loads(report_path.read_text(encoding="utf-8"))
    artifacts = e22["artifacts"]
    source1_path = Path(e22["cache_reuse"]["reused_normalized_source1"])
    feed_path = Path(e22["cache_reuse"]["reused_normalized_feed"])

    cache_started = time.perf_counter()
    source1 = pd.read_parquet(source1_path)
    feed = pd.read_parquet(feed_path)
    legacy = pd.read_parquet(artifacts["best_legacy"])
    name_candidates = pd.read_parquet(artifacts["name_candidates"])
    original_address = pd.read_parquet(artifacts["address_candidates"])
    original_selected = pd.read_parquet(artifacts["selected_candidates"])
    e21_report = json.loads(
        (config.experiments_dir / f"e21_lgbm_{E21_ARTIFACT_ID}_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    e21_candidates_path = Path(e21_report["artifacts"]["candidates"])
    e21_candidates = pd.read_parquet(e21_candidates_path)
    cache_seconds = time.perf_counter() - cache_started
    if len(source1) != 10_000 or len(feed) != 1_034_502:
        raise RuntimeError("Cached E2.1/E2.2 universe dimensions changed")

    truth_started = time.perf_counter()
    _, truth = _load_truth(config, set(source1["entity_id"]))
    truth_seconds = time.perf_counter() - truth_started
    training_ids, validation_ids = split_source1_ids(
        source1,
        validation_fraction=config.e2_validation_fraction,
        seed=config.e2_split_seed,
    )
    all_truth_pairs = _truth_pairs(truth)
    held_truth_pairs = {
        pair for pair in all_truth_pairs if pair[0] in validation_ids
    }
    e21_pairs = _pair_set(e21_candidates)
    prior_failures = held_truth_pairs - e21_pairs

    base_universe = prepare_fusion_universe(legacy, name_candidates, tfidf_top_k=30)
    legacy_name = reciprocal_rank_fusion(
        base_universe,
        final_k=50,
        legacy_weight=2.0,
        tfidf_weight=1.0,
        rrf_constant=20.0,
    )
    base_pairs = _pair_set(legacy_name)
    del base_universe
    gc.collect()

    strategies = [
        AddressPreblockStrategy(
            name="postcode_numeric",
            use_postcode=True,
            use_postcode_prefix=False,
            numeric_max_df=2_000,
            rare_token_max_df=250,
            rare_tokens_per_query=0,
            pool_cap=10_000,
        ),
        AddressPreblockStrategy(
            name="balanced_union",
            use_postcode=True,
            use_postcode_prefix=False,
            numeric_max_df=2_000,
            rare_token_max_df=250,
            rare_tokens_per_query=2,
            pool_cap=10_000,
        ),
        AddressPreblockStrategy(
            name="balanced_prefix_union",
            use_postcode=True,
            use_postcode_prefix=True,
            postcode_prefix_max_df=2_000,
            numeric_max_df=2_000,
            rare_token_max_df=250,
            rare_tokens_per_query=2,
            pool_cap=10_000,
        ),
        AddressPreblockStrategy(
            name="broad_union",
            use_postcode=True,
            use_postcode_prefix=False,
            numeric_max_df=5_000,
            rare_token_max_df=500,
            rare_tokens_per_query=3,
            pool_cap=15_000,
        ),
    ]
    _event(
        "e2.3-cache-reused",
        source1_rows=len(source1),
        feed_rows=len(feed),
        validation_entities=len(validation_ids),
        cache_seconds=cache_seconds,
        truth_seconds=truth_seconds,
    )

    build_started = time.perf_counter()
    prepared = prepare_address_tfidf_index(
        feed,
        config,
        maximum_numeric_df=max(s.numeric_max_df for s in strategies),
        maximum_rare_token_df=max(s.rare_token_max_df for s in strategies),
        maximum_postcode_prefix_df=max(s.postcode_prefix_max_df for s in strategies),
        postcode_prefix_length=3,
    )
    build_seconds = time.perf_counter() - build_started
    _event(
        "e2.3-index-built",
        seconds=build_seconds,
        profile=prepared.profile,
    )

    original_address_pairs = _pair_set(original_address)
    original_final_pairs = _pair_set(original_selected)
    original_address_metrics = _address_channel_metrics(
        original_address, source1, truth, validation_ids, base_pairs
    )
    original_final_metrics = _retrieval_bundle(
        original_selected, source1, truth, validation_ids
    )

    results: Dict[str, object] = {}
    frames: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    best_name = ""
    best_key = None
    pool_diagnostics = {
        strategy.name: diagnose_address_preblock_pools(source1, prepared, strategy)
        for strategy in strategies
    }
    _event("e2.3-pool-diagnostics", diagnostics=pool_diagnostics)
    # The postcode/numeric-only control is intentionally diagnosed without a
    # full similarity run: its broad-fallback comparison count makes it an
    # operationally invalid full-scale option. The three recall-oriented union
    # strategies receive the complete retrieval and fusion evaluation.
    for strategy in strategies[1:]:
        address, query_profile = retrieve_from_prepared_address_index(
            source1, prepared, strategy, top_k=20
        )
        address_metrics_20 = _address_channel_metrics(
            address, source1, truth, validation_ids, base_pairs
        )
        per_top_k: Dict[str, object] = {}
        for address_top_k in (10, 20):
            address_subset = address.loc[
                address["address_tfidf_rank"] <= address_top_k
            ].copy()
            universe = prepare_multi_channel_universe(
                legacy,
                name_candidates,
                address_subset,
                name_top_k=30,
                address_top_k=address_top_k,
            )
            fusion_started = time.perf_counter()
            fused = protected_legacy_multi_channel_rrf(
                universe,
                final_k=50,
                minimum_legacy=35,
                legacy_weight=2.0,
                name_weight=1.0,
                address_weight=1.0,
                rrf_constant=20.0,
            )
            fusion_seconds = time.perf_counter() - fusion_started
            bundle = _retrieval_bundle(fused, source1, truth, validation_ids)
            fused_pairs = _pair_set(fused)
            preserved = held_truth_pairs & original_final_pairs & fused_pairs
            old_recovered = prior_failures & fused_pairs
            per_top_k[str(address_top_k)] = {
                "metrics": bundle,
                "fusion_seconds": fusion_seconds,
                "e22_heldout_true_links_preserved": len(preserved),
                "e22_heldout_true_links_lost": len(
                    (held_truth_pairs & original_final_pairs) - fused_pairs
                ),
                "heldout_true_links_recovered_differently": len(
                    (held_truth_pairs - original_final_pairs) & fused_pairs
                ),
                "original_966_e21_failures_recovered": len(old_recovered),
            }
            selection_key = (
                float(bundle["heldout"]["pair_candidate_recall"]),
                float(bundle["heldout"]["oracle_macro_f0_5"]),
                -float(query_profile["total_query_seconds"]),
                -address_top_k,
            )
            label = f"{strategy.name}_top{address_top_k}"
            if best_key is None or selection_key > best_key:
                best_key = selection_key
                best_name = label
            frames[label] = (address_subset, fused)
            del universe
        results[strategy.name] = {
            "strategy": _strategy_payload(strategy),
            "query_profile": query_profile,
            "address_channel_top20": address_metrics_20,
            "final_fusion_by_address_top_k": per_top_k,
        }
        _event(
            "e2.3-strategy",
            strategy=strategy.name,
            query_profile=query_profile,
            top20=per_top_k["20"]["metrics"]["heldout"],
        )
        gc.collect()

    best_address, best_selected = frames[best_name]
    best_strategy_name, best_top_k_text = best_name.rsplit("_top", 1)
    best_top_k = int(best_top_k_text)
    best_result = results[best_strategy_name]
    best_query_profile = best_result["query_profile"]
    best_final_metrics = best_result["final_fusion_by_address_top_k"][
        str(best_top_k)
    ]["metrics"]
    best_address_pairs = _pair_set(best_address)
    best_final_pairs = _pair_set(best_selected)

    original_true_address = all_truth_pairs & original_address_pairs
    optimized_true_address = all_truth_pairs & best_address_pairs
    kept_address_true = original_true_address & optimized_true_address
    lost_address_true = original_true_address - optimized_true_address
    newly_address_true = optimized_true_address - original_true_address
    original_final_true = held_truth_pairs & original_final_pairs
    optimized_final_true = held_truth_pairs & best_final_pairs
    old_failure_recovery = prior_failures & best_final_pairs
    category_recovery = _category_recovery(
        prior_failures,
        original_address_pairs,
        best_address_pairs,
        source1,
        feed,
    )
    left = source1.set_index("entity_id", verify_integrity=True)
    right = feed.set_index("entity_id", verify_integrity=True)
    for category, payload in category_recovery.items():
        category_pairs = {
            pair
            for pair in prior_failures
            if _retrieval_failure_category(left.loc[pair[0]], right.loc[pair[1]])
            == category
        }
        payload["optimized_final_recovered"] = len(category_pairs & best_final_pairs)

    original_profile = e22["address_tfidf"]["profile"]
    optimized_total = float(prepared.profile["total_build_seconds"]) + float(
        best_query_profile["total_query_seconds"]
    )
    original_seconds = float(original_profile["total_runtime_seconds"])
    projection = _runtime_projection(
        prepared.profile,
        best_query_profile,
        validation_feed_rows=len(feed),
        validation_queries=len(source1),
    )
    context = {
        "experiment": "E2.3_address_tfidf_preblock_optimization",
        "source_e22_artifact": E22_ARTIFACT_ID,
        "selected_configuration": best_name,
        "strategies": [_strategy_payload(value) for value in strategies],
        "fusion": {
            "legacy_weight": 2.0,
            "name_weight": 1.0,
            "address_weight": 1.0,
            "rrf_constant": 20.0,
            "protected_legacy_quota": 35,
            "final_k": 50,
        },
    }
    artifact_id = _artifact_hash(context)
    report: Dict[str, object] = {
        **context,
        "artifact_id": artifact_id,
        "cache_reuse": {
            "normalized_source1": str(source1_path),
            "normalized_feed": str(feed_path),
            "legacy_candidates": artifacts["best_legacy"],
            "name_tfidf_candidates": artifacts["name_candidates"],
            "original_address_tfidf_candidates": artifacts["address_candidates"],
            "original_selected_candidates": artifacts["selected_candidates"],
            "e21_candidates": str(e21_candidates_path),
            "source1_rows": len(source1),
            "feed_rows": len(feed),
            "normalization_rerun": False,
            "universe_sampling_rerun": False,
        },
        "split": {
            "seed": config.e2_split_seed,
            "training_entities": len(training_ids),
            "validation_entities": len(validation_ids),
        },
        "cpu": {
            "logical_cores": os.cpu_count(),
            "multiprocessing_used": False,
            "reason": "shared sparse matrices avoid process-level memory duplication",
        },
        "original_e22_profile": original_profile,
        "original_address_metrics": original_address_metrics,
        "original_final_metrics": original_final_metrics,
        "optimized_index_profile": prepared.profile,
        "preblock_pool_diagnostics": pool_diagnostics,
        "strategy_results": results,
        "selected_metrics": best_final_metrics,
        "recall_accounting": {
            "e22_address_true_links": len(original_true_address),
            "e22_address_true_links_preserved": len(kept_address_true),
            "e22_address_true_links_lost": len(lost_address_true),
            "address_true_links_recovered_differently": len(newly_address_true),
            "net_address_true_link_change": len(optimized_true_address)
            - len(original_true_address),
            "e22_final_heldout_true_links": len(original_final_true),
            "e22_final_heldout_true_links_preserved": len(
                original_final_true & optimized_final_true
            ),
            "e22_final_heldout_true_links_lost": len(
                original_final_true - optimized_final_true
            ),
            "new_final_heldout_true_links": len(
                optimized_final_true - original_final_true
            ),
            "net_final_heldout_true_link_change": len(optimized_final_true)
            - len(original_final_true),
            "original_e21_heldout_failures": len(prior_failures),
            "original_address_recovered_from_e21_failures": len(
                prior_failures & original_address_pairs
            ),
            "optimized_address_recovered_from_e21_failures": len(
                prior_failures & best_address_pairs
            ),
            "optimized_final_recovered_from_e21_failures": len(
                old_failure_recovery
            ),
            "remaining_e21_failures": len(prior_failures - best_final_pairs),
            "remaining_failure_categories": _remaining_failure_categories(
                prior_failures - best_final_pairs, source1, feed
            ),
            "recovery_by_original_failure_category": category_recovery,
        },
        "difficult_examples": {
            "original_address_kept": _examples(
                prior_failures & kept_address_true,
                best_address,
                source1,
                feed,
            ),
            "original_address_lost": _examples(
                prior_failures & lost_address_true,
                original_address,
                source1,
                feed,
            ),
        },
        "performance": {
            "cache_load_seconds": cache_seconds,
            "truth_load_seconds": truth_seconds,
            "original_address_retrieval_seconds": original_seconds,
            "optimized_index_build_seconds": prepared.profile[
                "total_build_seconds"
            ],
            "optimized_query_seconds": best_query_profile["total_query_seconds"],
            "optimized_address_total_seconds": optimized_total,
            "speedup_vs_original_including_build": original_seconds
            / optimized_total,
            "speedup_vs_original_query_stage": float(
                original_profile["similarity_query_seconds"]
            )
            / float(best_query_profile["similarity_multiplication_seconds"]),
            "total_wall_seconds_before_persistence": time.perf_counter()
            - total_started,
            "peak_rss_mib": _peak_rss_mib(),
            "full_test_projection": projection,
        },
    }

    if persist:
        persistence_started = time.perf_counter()
        prefix = f"e23_retrieval_{artifact_id}"
        paths = {
            "optimized_address_candidates": config.cache_dir
            / "candidates"
            / f"{prefix}_address_tfidf.parquet",
            "selected_candidates": config.cache_dir
            / "candidates"
            / f"{prefix}_selected.parquet",
            "report": config.experiments_dir / f"{prefix}_metrics.json",
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite E2.3 artifacts: {existing}")
        metadata = {
            "artifact_id": artifact_id,
            "source_e22_artifact": E22_ARTIFACT_ID,
            "configuration": best_name,
        }
        write_parquet_cache(
            best_address, paths["optimized_address_candidates"], metadata
        )
        write_parquet_cache(best_selected, paths["selected_candidates"], metadata)
        report["artifacts"] = {key: str(value) for key, value in paths.items()}
        report["cache_disk_usage_bytes"] = {
            "optimized_address_candidates": paths[
                "optimized_address_candidates"
            ].stat().st_size,
            "selected_candidates": paths["selected_candidates"].stat().st_size,
            "new_candidate_caches_total": paths[
                "optimized_address_candidates"
            ].stat().st_size
            + paths["selected_candidates"].stat().st_size,
            "sparse_index_cached": False,
        }
        report["performance"]["persistence_seconds"] = (
            time.perf_counter() - persistence_started
        )
        report["performance"]["total_wall_seconds"] = (
            time.perf_counter() - total_started
        )
        paths["report"].write_text(
            json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )

    _event(
        "e2.3-complete",
        artifact_id=artifact_id,
        selected_configuration=best_name,
        heldout=best_final_metrics["heldout"],
        speedup=report["performance"]["speedup_vs_original_including_build"],
        peak_rss_mib=report["performance"]["peak_rss_mib"],
    )
    return report
