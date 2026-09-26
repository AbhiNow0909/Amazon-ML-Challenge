"""E2.2 retrieval-density diagnosis and repair on the cached E2.1 universe."""

from __future__ import annotations

import gc
import hashlib
import json
import resource
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .blocking import (
    PROVENANCE_COLUMNS,
    effective_rare_posting_cap,
    generate_candidates,
    token_df_eligible,
)
from .candidate_fusion import prepare_fusion_universe, reciprocal_rank_fusion
from .config import DEFAULT_CONFIG, PipelineConfig
from .evaluation import candidate_metrics
from .io_utils import write_parquet_cache
from .multi_channel_fusion import (
    multi_channel_rrf,
    prepare_multi_channel_universe,
    protected_legacy_multi_channel_rrf,
)
from .normalize import informative_tokens
from .split import split_source1_ids
from .stress_test import _json_default, _load_truth, _retrieval_failure_category, feed_at_density
from .tfidf_retrieval import retrieve_address_tfidf, retrieve_name_tfidf


E21_ARTIFACT_ID = "c449c0c6b281"
E21_REPORT_NAME = f"e21_lgbm_{E21_ARTIFACT_ID}_metrics.json"
CHANNEL_COLUMNS = [
    "block_exact_full_name",
    "block_exact_core_name",
    "block_exact_sorted_name",
    "block_rare_name_token",
    "block_exact_address",
    "block_rare_address_token",
    "block_cross_country_exact_name",
]


def _event(name: str, **values: object) -> None:
    print(json.dumps({"event": name, **values}, default=_json_default), flush=True)


def _peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _pair_set(frame: pd.DataFrame) -> Set[Tuple[str, str]]:
    return set(zip(frame["source1_entity_id"], frame["candidate_entity_id"]))


def _truth_pairs(truth: Mapping[str, Set[str]]) -> Set[Tuple[str, str]]:
    return {(s1, target) for s1, targets in truth.items() for target in targets}


def _truth_link_frame(
    truth: Mapping[str, Set[str]], source1: pd.DataFrame, feed: pd.DataFrame
) -> pd.DataFrame:
    links = pd.DataFrame(
        [(s1, target) for s1, targets in truth.items() for target in targets],
        columns=["source1_entity_id", "candidate_entity_id"],
    )
    left = source1[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country_norm",
            "name_full",
            "name_core",
            "name_sorted",
            "name_tokens",
            "address_full",
            "address_tokens",
        ]
    ].rename(
        columns={
            "entity_id": "source1_entity_id",
            "business_name": "source1_name",
            "business_address": "source1_address",
            "country_norm": "country",
            "name_full": "left_name_full",
            "name_core": "left_name_core",
            "name_sorted": "left_name_sorted",
            "name_tokens": "left_name_tokens",
            "address_full": "left_address_full",
            "address_tokens": "left_address_tokens",
        }
    )
    right = feed[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country_norm",
            "name_full",
            "name_core",
            "name_sorted",
            "name_tokens",
            "address_full",
            "address_tokens",
        ]
    ].rename(
        columns={
            "entity_id": "candidate_entity_id",
            "business_name": "candidate_name",
            "business_address": "candidate_address",
            "country_norm": "candidate_country",
            "name_full": "right_name_full",
            "name_core": "right_name_core",
            "name_sorted": "right_name_sorted",
            "name_tokens": "right_name_tokens",
            "address_full": "right_address_full",
            "address_tokens": "right_address_tokens",
        }
    )
    result = links.merge(left, on="source1_entity_id", validate="many_to_one").merge(
        right, on="candidate_entity_id", validate="many_to_one"
    )
    counts = links.groupby("source1_entity_id").size()
    result["truth_count_bucket"] = result["source1_entity_id"].map(
        lambda value: (
            "0"
            if counts.get(value, 0) == 0
            else str(counts[value])
            if counts[value] < 3
            else "3+"
        )
    )
    return result


def _counter_from_tokens(
    feed: pd.DataFrame, token_column: str, minimum_length: int
) -> Counter:
    values: Counter = Counter()
    for tokens, country in feed[[token_column, "country_norm"]].itertuples(
        index=False, name=None
    ):
        for token in informative_tokens(tokens, minimum_length=minimum_length):
            values[(country, token)] += 1
    return values


def _exact_counter(feed: pd.DataFrame, column: str, *, country: bool = True) -> Counter:
    values: Counter = Counter()
    columns = [column, "country_norm"] if country else [column]
    for row in feed[columns].itertuples(index=False, name=None):
        value = row[0]
        if value:
            values[(row[1], value) if country else value] += 1
    return values


def _channel_summary(
    links: pd.DataFrame, flags: Mapping[str, Sequence[bool]]
) -> Dict[str, object]:
    arrays = {key: np.asarray(value, dtype=bool) for key, value in flags.items()}
    total = len(links)
    result: Dict[str, object] = {}
    for channel, recovered in arrays.items():
        country = {}
        for value, index in links.groupby("country", sort=True).groups.items():
            positions = np.asarray(list(index), dtype=int)
            count = int(recovered[positions].sum())
            country[str(value)] = {
                "true_links": len(positions),
                "recovered_true_links": count,
                "pair_recall": count / len(positions) if len(positions) else 0.0,
            }
        buckets = {}
        for value, index in links.groupby("truth_count_bucket", sort=True).groups.items():
            positions = np.asarray(list(index), dtype=int)
            count = int(recovered[positions].sum())
            buckets[str(value)] = {
                "true_links": len(positions),
                "recovered_true_links": count,
                "pair_recall": count / len(positions) if len(positions) else 0.0,
            }
        others = np.zeros(total, dtype=bool)
        for other_channel, other_recovered in arrays.items():
            if other_channel != channel:
                others |= other_recovered
        count = int(recovered.sum())
        result[channel] = {
            "recovered_true_links": count,
            "pair_recall": count / total if total else 0.0,
            "unique_true_links": int((recovered & ~others).sum()),
            "by_country": country,
            "by_truth_count": buckets,
        }
    overlap = {
        left: {
            right: int((left_values & right_values).sum())
            for right, right_values in arrays.items()
        }
        for left, left_values in arrays.items()
    }
    return {"channels": result, "overlap": overlap}


def diagnose_legacy_channels(
    links: pd.DataFrame, feed: pd.DataFrame, config: PipelineConfig
) -> Tuple[Dict[str, object], Dict[Tuple[str, str], int], Dict[Tuple[str, str], int]]:
    """Diagnose true-pair channel eligibility before candidate ranking."""

    started = time.perf_counter()
    shard_sizes = Counter(feed["country_norm"])
    name_df = _counter_from_tokens(feed, "name_tokens", config.minimum_token_length)
    address_df = _counter_from_tokens(feed, "address_tokens", config.minimum_token_length)
    exact_full = _exact_counter(feed, "name_full")
    exact_core = _exact_counter(feed, "name_core")
    exact_sorted = _exact_counter(feed, "name_sorted")
    exact_address = _exact_counter(feed, "address_full")
    global_full = _exact_counter(feed, "name_full", country=False)

    flags = {column: [] for column in CHANNEL_COLUMNS}
    rare_name_posting_suppressed = 0
    rare_address_posting_suppressed = 0
    exact_posting_suppressed = Counter()
    for row in links.itertuples(index=False):
        same_country = row.country == row.candidate_country
        full_key = (row.country, row.left_name_full)
        core_key = (row.country, row.left_name_core)
        sorted_key = (row.country, row.left_name_sorted)
        address_key = (row.country, row.left_address_full)
        full_same = bool(
            same_country
            and row.left_name_full
            and row.left_name_full == row.right_name_full
        )
        core_same = bool(
            same_country
            and row.left_name_core
            and row.left_name_core == row.right_name_core
        )
        sorted_same = bool(
            same_country
            and row.left_name_sorted
            and row.left_name_sorted == row.right_name_sorted
        )
        address_same = bool(
            same_country
            and row.left_address_full
            and row.left_address_full == row.right_address_full
        )
        flags["block_exact_full_name"].append(
            full_same and exact_full[full_key] <= config.posting_list_cap
        )
        flags["block_exact_core_name"].append(
            core_same and exact_core[core_key] <= config.posting_list_cap
        )
        flags["block_exact_sorted_name"].append(
            sorted_same and exact_sorted[sorted_key] <= config.posting_list_cap
        )
        flags["block_exact_address"].append(
            address_same and exact_address[address_key] <= config.posting_list_cap
        )
        if full_same and exact_full[full_key] > config.posting_list_cap:
            exact_posting_suppressed["exact_full_name"] += 1
        if core_same and exact_core[core_key] > config.posting_list_cap:
            exact_posting_suppressed["exact_core_name"] += 1
        if sorted_same and exact_sorted[sorted_key] > config.posting_list_cap:
            exact_posting_suppressed["exact_sorted_name"] += 1
        if address_same and exact_address[address_key] > config.posting_list_cap:
            exact_posting_suppressed["exact_address"] += 1

        left_name_tokens = set(
            informative_tokens(
                row.left_name_tokens, minimum_length=config.minimum_token_length
            )
        )
        right_name_tokens = set(
            informative_tokens(
                row.right_name_tokens, minimum_length=config.minimum_token_length
            )
        )
        common_name = left_name_tokens & right_name_tokens if same_country else set()
        name_posting_cap = effective_rare_posting_cap(config, shard_sizes[row.country])
        name_eligible = []
        for token in common_name:
            frequency = name_df[(row.country, token)]
            df_ok = token_df_eligible(
                frequency,
                shard_sizes[row.country],
                absolute_max_df=config.rare_name_token_max_df,
                relative_max_df=config.rare_name_token_max_df_fraction,
                mode=config.rare_token_df_mode,
            )
            if df_ok and frequency > name_posting_cap:
                rare_name_posting_suppressed += 1
            name_eligible.append(df_ok and frequency <= name_posting_cap)
        flags["block_rare_name_token"].append(any(name_eligible))

        left_address_tokens = set(
            informative_tokens(
                row.left_address_tokens, minimum_length=config.minimum_token_length
            )
        )
        right_address_tokens = set(
            informative_tokens(
                row.right_address_tokens, minimum_length=config.minimum_token_length
            )
        )
        common_address = (
            left_address_tokens & right_address_tokens if same_country else set()
        )
        address_posting_cap = effective_rare_posting_cap(
            config, shard_sizes[row.country]
        )
        address_eligible = []
        for token in common_address:
            frequency = address_df[(row.country, token)]
            df_ok = token_df_eligible(
                frequency,
                shard_sizes[row.country],
                absolute_max_df=config.rare_address_token_max_df,
                relative_max_df=config.rare_address_token_max_df_fraction,
                mode=config.rare_token_df_mode,
            )
            if df_ok and frequency > address_posting_cap:
                rare_address_posting_suppressed += 1
            address_eligible.append(df_ok and frequency <= address_posting_cap)
        flags["block_rare_address_token"].append(any(address_eligible))
        cross_country = bool(
            not same_country
            and row.left_name_full
            and row.left_name_full == row.right_name_full
            and global_full[row.left_name_full] <= config.posting_list_cap
        )
        flags["block_cross_country_exact_name"].append(cross_country)

    diagnostic = _channel_summary(links, flags)
    diagnostic.update(
        {
            "feed_rows": len(feed),
            "shard_sizes": dict(shard_sizes),
            "name_token_vocabulary": len(name_df),
            "address_token_vocabulary": len(address_df),
            "qualifying_name_tokens": sum(
                token_df_eligible(
                    frequency,
                    shard_sizes[country],
                    absolute_max_df=config.rare_name_token_max_df,
                    relative_max_df=config.rare_name_token_max_df_fraction,
                    mode=config.rare_token_df_mode,
                )
                and frequency <= effective_rare_posting_cap(
                    config, shard_sizes[country]
                )
                for (country, _), frequency in name_df.items()
            ),
            "qualifying_address_tokens": sum(
                token_df_eligible(
                    frequency,
                    shard_sizes[country],
                    absolute_max_df=config.rare_address_token_max_df,
                    relative_max_df=config.rare_address_token_max_df_fraction,
                    mode=config.rare_token_df_mode,
                )
                and frequency <= effective_rare_posting_cap(
                    config, shard_sizes[country]
                )
                for (country, _), frequency in address_df.items()
            ),
            "rare_name_posting_suppressed_truth_token_hits": rare_name_posting_suppressed,
            "rare_address_posting_suppressed_truth_token_hits": rare_address_posting_suppressed,
            "exact_posting_suppressed_true_links": dict(exact_posting_suppressed),
            "runtime_seconds": time.perf_counter() - started,
            "peak_rss_mib": _peak_rss_mib(),
        }
    )
    relevant_name_tokens = {
        (row.country, token)
        for row in links.itertuples(index=False)
        for token in set(
            informative_tokens(
                row.left_name_tokens, minimum_length=config.minimum_token_length
            )
        )
        & set(
            informative_tokens(
                row.right_name_tokens, minimum_length=config.minimum_token_length
            )
        )
    }
    relevant_address_tokens = {
        (row.country, token)
        for row in links.itertuples(index=False)
        for token in set(
            informative_tokens(
                row.left_address_tokens, minimum_length=config.minimum_token_length
            )
        )
        & set(
            informative_tokens(
                row.right_address_tokens, minimum_length=config.minimum_token_length
            )
        )
    }
    compact_name_df = {key: name_df[key] for key in relevant_name_tokens}
    compact_address_df = {key: address_df[key] for key in relevant_address_tokens}
    return diagnostic, compact_name_df, compact_address_df


def _retrieval_bundle(
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    truth: Mapping[str, Set[str]],
    validation_ids: Set[str],
) -> Dict[str, object]:
    all_ids = source1["entity_id"].tolist()
    heldout_ids = sorted(validation_ids)
    heldout_truth = {value: truth.get(value, set()) for value in heldout_ids}
    heldout = candidates.loc[candidates["source1_entity_id"].isin(validation_ids)]
    countries = {}
    for country, group in source1.loc[
        source1["entity_id"].isin(validation_ids)
    ].groupby("country_norm", sort=True):
        ids = group["entity_id"].tolist()
        country_candidates = heldout.loc[heldout["source1_entity_id"].isin(ids)]
        countries[str(country)] = candidate_metrics(ids, heldout_truth, country_candidates)
    return {
        "overall": candidate_metrics(all_ids, truth, candidates),
        "heldout": candidate_metrics(heldout_ids, heldout_truth, heldout),
        "heldout_by_country": countries,
    }


def _address_examples(
    address: pd.DataFrame,
    truth_pairs: Set[Tuple[str, str]],
    previous_failures: Set[Tuple[str, str]],
    source1: pd.DataFrame,
    feed: pd.DataFrame,
) -> Dict[str, object]:
    left = source1.set_index("entity_id", verify_integrity=True)
    right = feed.set_index("entity_id", verify_integrity=True)
    scored = address.sort_values(
        ["address_tfidf_similarity", "source1_entity_id", "candidate_entity_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )

    def row(pair: Tuple[str, str], score: float, rank: int) -> Dict[str, object]:
        s1, target = pair
        return {
            "source1_entity_id": s1,
            "candidate_entity_id": target,
            "source1_name": left.at[s1, "business_name"],
            "candidate_name": right.at[target, "business_name"],
            "source1_address": left.at[s1, "business_address"],
            "candidate_address": right.at[target, "business_address"],
            "country": left.at[s1, "country_norm"],
            "address_tfidf_similarity": score,
            "address_tfidf_rank": rank,
        }

    good = []
    bad = []
    for item in scored.itertuples(index=False):
        pair = (item.source1_entity_id, item.candidate_entity_id)
        payload = row(pair, float(item.address_tfidf_similarity), int(item.address_tfidf_rank))
        if pair in previous_failures and len(good) < 15:
            good.append(payload)
        elif pair not in truth_pairs and len(bad) < 15:
            bad.append(payload)
        if len(good) >= 15 and len(bad) >= 15:
            break
    return {"good_recoveries": good, "high_similarity_bad_neighbors": bad}


def _threshold_crossing_examples(
    links: pd.DataFrame,
    df_100k: Mapping[Tuple[str, str], int],
    df_1m: Mapping[Tuple[str, str], int],
    limit: int,
) -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    for row in links.itertuples(index=False):
        common = set(
            informative_tokens(row.left_name_tokens, minimum_length=3)
        ) & set(
            informative_tokens(row.right_name_tokens, minimum_length=3)
        )
        crossed = sorted(
            (
                token,
                df_100k.get((row.country, token), 0),
                df_1m.get((row.country, token), 0),
            )
            for token in common
            if 0 < df_100k.get((row.country, token), 0) <= limit
            and df_1m.get((row.country, token), 0) > limit
        )
        if not crossed:
            continue
        token, before, after = min(crossed, key=lambda value: value[2])
        examples.append(
            {
                "source1_entity_id": row.source1_entity_id,
                "candidate_entity_id": row.candidate_entity_id,
                "source1_name": row.source1_name,
                "candidate_name": row.candidate_name,
                "country": row.country,
                "token": token,
                "df_100k": before,
                "df_1m": after,
            }
        )
        if len(examples) >= 15:
            break
    return examples


def _remaining_failure_categories(
    failures: Set[Tuple[str, str]], source1: pd.DataFrame, feed: pd.DataFrame
) -> Dict[str, object]:
    left = source1.set_index("entity_id", verify_integrity=True)
    right = feed.set_index("entity_id", verify_integrity=True)
    counts = Counter(
        _retrieval_failure_category(left.loc[s1], right.loc[target])
        for s1, target in failures
    )
    return {
        key: {
            "count": value,
            "percentage": 100.0 * value / len(failures) if failures else 0.0,
        }
        for key, value in sorted(counts.items())
    }


def _artifact_hash(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def run_e22_experiment(
    config: PipelineConfig = DEFAULT_CONFIG, *, persist: bool = True
) -> Dict[str, object]:
    total_started = time.perf_counter()
    e21_report_path = config.experiments_dir / E21_REPORT_NAME
    e21 = json.loads(e21_report_path.read_text(encoding="utf-8"))
    source1_path = Path(e21["artifacts"]["normalized_source1"])
    feed_path = Path(e21["artifacts"]["normalized_feed"])
    baseline_candidates_path = Path(e21["artifacts"]["candidates"])

    cache_started = time.perf_counter()
    source1 = pd.read_parquet(source1_path)
    feed = pd.read_parquet(feed_path)
    baseline_candidates = pd.read_parquet(baseline_candidates_path)
    cache_seconds = time.perf_counter() - cache_started
    if len(source1) != 10_000 or len(feed) != 1_034_502:
        raise RuntimeError("Cached E2.1 universe has unexpected dimensions")
    _event(
        "e2.2-cache-reused",
        source1_rows=len(source1),
        feed_rows=len(feed),
        baseline_candidates=len(baseline_candidates),
        seconds=cache_seconds,
    )

    truth_started = time.perf_counter()
    _, truth = _load_truth(config, set(source1["entity_id"]))
    truth_links = _truth_link_frame(truth, source1, feed)
    all_truth_pairs = _truth_pairs(truth)
    training_ids, validation_ids = split_source1_ids(
        source1,
        validation_fraction=config.e2_validation_fraction,
        seed=config.e2_split_seed,
    )
    heldout_truth_pairs = {
        pair for pair in all_truth_pairs if pair[0] in validation_ids
    }
    baseline_pairs = _pair_set(baseline_candidates)
    previous_failures = heldout_truth_pairs - baseline_pairs
    if len(previous_failures) != 966:
        raise RuntimeError(
            f"Expected 966 cached E2.1 held-out failures, found {len(previous_failures)}"
        )
    truth_seconds = time.perf_counter() - truth_started

    diagnostic_started = time.perf_counter()
    density_diagnostics: Dict[str, object] = {}
    relevant_name_dfs: Dict[int, Dict[Tuple[str, str], int]] = {}
    for density in config.e21_distractor_totals:
        density_feed = feed_at_density(feed, density)
        diagnostic, name_df, _ = diagnose_legacy_channels(
            truth_links, density_feed, config
        )
        density_diagnostics[str(density)] = diagnostic
        relevant_name_dfs[density] = name_df
        _event(
            "e2.2-channel-diagnostic",
            distractors=density,
            qualifying_name_tokens=diagnostic["qualifying_name_tokens"],
            rare_name_true_links=diagnostic["channels"]["block_rare_name_token"][
                "recovered_true_links"
            ],
            seconds=diagnostic["runtime_seconds"],
        )
        del density_feed
        gc.collect()
    diagnostic_seconds = time.perf_counter() - diagnostic_started
    crossing_examples = _threshold_crossing_examples(
        truth_links,
        relevant_name_dfs[100_000],
        relevant_name_dfs[1_000_000],
        config.rare_name_token_max_df,
    )
    del relevant_name_dfs
    gc.collect()

    name_started = time.perf_counter()
    name_config = replace(config, use_name_tfidf=True, tfidf_name_top_k=30)
    name_candidates, name_profile = retrieve_name_tfidf(source1, feed, name_config)
    name_seconds = time.perf_counter() - name_started
    name_pairs = _pair_set(name_candidates)

    address_started = time.perf_counter()
    address_config = replace(
        config, use_address_tfidf=True, tfidf_address_top_k=20
    )
    address_candidates, address_profile = retrieve_address_tfidf(
        source1, feed, address_config
    )
    address_seconds = time.perf_counter() - address_started
    _event(
        "e2.2-sparse-retrieval",
        name_seconds=name_seconds,
        address_seconds=address_seconds,
        name_pairs=len(name_candidates),
        address_pairs=len(address_candidates),
        peak_rss_mib=_peak_rss_mib(),
    )

    address_standalone: Dict[str, object] = {}
    for top_k in (5, 10, 20):
        subset = address_candidates.loc[
            address_candidates["address_tfidf_rank"] <= top_k
        ]
        bundle = _retrieval_bundle(subset, source1, truth, validation_ids)
        pairs = _pair_set(subset)
        recovered = len(all_truth_pairs & pairs)
        incremental = len((all_truth_pairs - baseline_pairs) & pairs)
        bundle.update(
            {
                "top_k": top_k,
                "candidate_pair_precision_proxy": recovered / len(subset),
                "unique_incremental_true_links_vs_e21": incremental,
                "average_additional_candidates_vs_e21": len(pairs - baseline_pairs)
                / len(source1),
            }
        )
        address_standalone[str(top_k)] = bundle

    rare_variant_configs = {
        "absolute_baseline": replace(config, use_name_tfidf=False),
        "relative_df": replace(
            config,
            use_name_tfidf=False,
            rare_token_df_mode="relative",
            rare_name_token_max_df_fraction=0.0005,
            rare_address_token_max_df_fraction=0.00015,
        ),
        "hybrid_df": replace(
            config,
            use_name_tfidf=False,
            rare_token_df_mode="hybrid",
            rare_name_token_max_df_fraction=0.0005,
            rare_address_token_max_df_fraction=0.00015,
        ),
        "hybrid_scaled_postings": replace(
            config,
            use_name_tfidf=False,
            rare_token_df_mode="hybrid",
            rare_name_token_max_df_fraction=0.0005,
            rare_address_token_max_df_fraction=0.00015,
            rare_posting_cap_fraction=0.0006,
            rare_posting_cap_max=350,
        ),
        "hybrid_ranked2_scaled": replace(
            config,
            use_name_tfidf=False,
            rare_token_df_mode="hybrid",
            rare_name_token_max_df_fraction=0.0005,
            rare_address_token_max_df_fraction=0.00015,
            ranked_name_tokens_per_query=2,
            rare_posting_cap_fraction=0.0006,
            rare_posting_cap_max=350,
        ),
    }
    legacy_started = time.perf_counter()
    legacy_frames: Dict[str, pd.DataFrame] = {}
    rare_variant_results: Dict[str, object] = {}
    best_density_name = ""
    best_density_key: Optional[Tuple[float, float]] = None
    best_density_candidates: Optional[pd.DataFrame] = None
    for name, variant_config in rare_variant_configs.items():
        stage_started = time.perf_counter()
        legacy, profile = generate_candidates(
            source1, feed, variant_config, max_k=50
        )
        legacy_frames[name] = legacy
        universe = prepare_fusion_universe(
            legacy, name_candidates, tfidf_top_k=30
        )
        fused = reciprocal_rank_fusion(
            universe,
            final_k=50,
            legacy_weight=2.0,
            tfidf_weight=1.0,
            rrf_constant=20.0,
        )
        bundle = _retrieval_bundle(fused, source1, truth, validation_ids)
        rare_variant_results[name] = {
            "config": {
                "df_mode": variant_config.rare_token_df_mode,
                "name_relative_df": variant_config.rare_name_token_max_df_fraction,
                "address_relative_df": variant_config.rare_address_token_max_df_fraction,
                "ranked_name_tokens_per_query": variant_config.ranked_name_tokens_per_query,
                "posting_cap_fraction": variant_config.rare_posting_cap_fraction,
                "posting_cap_max": variant_config.rare_posting_cap_max,
            },
            "legacy_profile": profile,
            "metrics": bundle,
            "runtime_seconds": time.perf_counter() - stage_started,
        }
        key = (
            float(bundle["heldout"]["pair_candidate_recall"]),
            float(bundle["heldout"]["oracle_macro_f0_5"]),
        )
        if best_density_key is None or key > best_density_key:
            best_density_key = key
            best_density_name = name
            best_density_candidates = fused
        _event(
            "e2.2-rare-variant",
            variant=name,
            heldout=bundle["heldout"],
            seconds=rare_variant_results[name]["runtime_seconds"],
        )
        del universe
        gc.collect()
    legacy_seconds = time.perf_counter() - legacy_started
    if best_density_candidates is None:
        raise AssertionError("No density-aware candidate configuration was selected")

    fusion_started = time.perf_counter()
    fusion_results: Dict[str, object] = {}
    best_fusion_name = ""
    best_fusion_key: Optional[Tuple[float, float, int]] = None
    best_fusion_candidates: Optional[pd.DataFrame] = None
    best_address_top_k = 0
    best_legacy = legacy_frames[best_density_name]
    for address_top_k in (5, 10, 20):
        universe = prepare_multi_channel_universe(
            best_legacy,
            name_candidates,
            address_candidates,
            name_top_k=30,
            address_top_k=address_top_k,
        )
        configurations = [
            ("rrf", 0.5, None),
            ("rrf", 1.0, None),
        ]
        if address_top_k in (10, 20):
            configurations.append(("protected", 1.0, 35))
        for strategy, address_weight, minimum_legacy in configurations:
            if strategy == "rrf":
                fused = multi_channel_rrf(
                    universe,
                    final_k=50,
                    legacy_weight=2.0,
                    name_weight=1.0,
                    address_weight=address_weight,
                    rrf_constant=20.0,
                )
            else:
                fused = protected_legacy_multi_channel_rrf(
                    universe,
                    final_k=50,
                    minimum_legacy=int(minimum_legacy),
                    legacy_weight=2.0,
                    name_weight=1.0,
                    address_weight=address_weight,
                    rrf_constant=20.0,
                )
            label = (
                f"{strategy}_legacy2_name1_address{address_weight:g}_"
                f"top{address_top_k}"
                + (f"_protected{minimum_legacy}" if minimum_legacy else "")
            )
            bundle = _retrieval_bundle(fused, source1, truth, validation_ids)
            fusion_results[label] = {
                "strategy": strategy,
                "legacy_variant": best_density_name,
                "legacy_weight": 2.0,
                "name_weight": 1.0,
                "address_weight": address_weight,
                "address_top_k": address_top_k,
                "minimum_legacy": minimum_legacy,
                "metrics": bundle,
            }
            key = (
                float(bundle["heldout"]["pair_candidate_recall"]),
                float(bundle["heldout"]["oracle_macro_f0_5"]),
                -address_top_k,
            )
            if best_fusion_key is None or key > best_fusion_key:
                best_fusion_key = key
                best_fusion_name = label
                best_fusion_candidates = fused
                best_address_top_k = address_top_k
            _event("e2.2-fusion", configuration=label, heldout=bundle["heldout"])
        del universe
        gc.collect()
    fusion_seconds = time.perf_counter() - fusion_started
    if best_fusion_candidates is None:
        raise AssertionError("No three-channel fusion was selected")

    density_pairs = _pair_set(best_density_candidates)
    best_address = address_candidates.loc[
        address_candidates["address_tfidf_rank"] <= best_address_top_k
    ]
    best_address_pairs = _pair_set(best_address)
    final_pairs = _pair_set(best_fusion_candidates)
    density_recovered = previous_failures & density_pairs
    address_recovered = previous_failures & best_address_pairs
    final_recovered = previous_failures & final_pairs
    remaining = previous_failures - final_pairs
    previously_recovered = heldout_truth_pairs & baseline_pairs
    displaced_previous_true_links = previously_recovered - final_pairs
    failure_recovery = {
        "previous_failures": len(previous_failures),
        "recovered_by_density_aware_fusion": len(density_recovered),
        "recovered_by_address_tfidf_raw": len(address_recovered),
        "density_address_overlap": len(density_recovered & address_recovered),
        "union_retrieved_before_final_fusion": len(
            density_recovered | address_recovered
        ),
        "retained_after_final_fusion": len(final_recovered),
        "previously_recovered_true_links_displaced": len(
            displaced_previous_true_links
        ),
        "retrieved_then_lost_during_final_fusion": len(
            (density_recovered | address_recovered) - final_pairs
        ),
        "remaining_failures": len(remaining),
        "remaining_failure_categories": _remaining_failure_categories(
            remaining, source1, feed
        ),
    }

    best_address_pairs_all = _pair_set(best_address)
    address_counts = Counter(feed["address_full"])
    address_by_id = feed.set_index("entity_id")["address_full"].to_dict()
    generic_false_pairs = 0
    for s1, target in best_address_pairs_all - all_truth_pairs:
        target_address = address_by_id.get(target, "")
        if target_address and address_counts[target_address] > 100:
            generic_false_pairs += 1
    address_safety = {
        "selected_top_k": best_address_top_k,
        "generic_address_false_neighbors_df_gt_100": generic_false_pairs,
        "examples": _address_examples(
            best_address,
            all_truth_pairs,
            previous_failures,
            source1,
            feed,
        ),
    }

    baseline_bundle = _retrieval_bundle(
        baseline_candidates, source1, truth, validation_ids
    )
    artifact_context = {
        "experiment": "E2.2_retrieval_density_repair",
        "source_e21_artifact": E21_ARTIFACT_ID,
        "cache_paths": {
            "source1": str(source1_path),
            "feed": str(feed_path),
            "baseline_candidates": str(baseline_candidates_path),
        },
        "best_density_variant": best_density_name,
        "best_fusion_configuration": best_fusion_name,
        "feature_flags": {
            "address_tfidf": True,
            "density_aware_rare_tokens": True,
        },
    }
    artifact_id = _artifact_hash(artifact_context)
    report: Dict[str, object] = {
        **artifact_context,
        "artifact_id": artifact_id,
        "cache_reuse": {
            "reused_normalized_source1": str(source1_path),
            "reused_normalized_feed": str(feed_path),
            "reused_e21_candidates": str(baseline_candidates_path),
            "source1_rows": len(source1),
            "feed_rows": len(feed),
            "normalization_rerun": False,
            "universe_sampling_rerun": False,
            "truth_scan_required": True,
        },
        "split": {
            "training_entities": len(training_ids),
            "validation_entities": len(validation_ids),
            "seed": config.e2_split_seed,
        },
        "baseline_metrics": baseline_bundle,
        "channel_diagnostics_by_density": density_diagnostics,
        "rare_token_configuration": {
            "absolute_name_max_df": config.rare_name_token_max_df,
            "absolute_address_max_df": config.rare_address_token_max_df,
            "posting_list_cap": config.posting_list_cap,
            "threshold_crossing_examples": crossing_examples,
        },
        "name_tfidf": {
            "profile": name_profile,
            "standalone": _retrieval_bundle(
                name_candidates, source1, truth, validation_ids
            ),
        },
        "address_tfidf": {
            "profile": address_profile,
            "standalone_by_top_k": address_standalone,
            "safety": address_safety,
        },
        "rare_token_variants": rare_variant_results,
        "fusion_results": fusion_results,
        "selected_metrics": fusion_results[best_fusion_name]["metrics"],
        "failure_recovery": failure_recovery,
        "performance": {
            "cache_load_seconds": cache_seconds,
            "truth_load_seconds": truth_seconds,
            "channel_diagnostic_seconds": diagnostic_seconds,
            "name_tfidf_seconds": name_seconds,
            "address_tfidf_seconds": address_seconds,
            "legacy_variant_seconds": legacy_seconds,
            "fusion_seconds": fusion_seconds,
            "total_wall_seconds_before_persistence": time.perf_counter()
            - total_started,
            "peak_rss_mib": _peak_rss_mib(),
        },
    }

    if persist:
        persistence_started = time.perf_counter()
        prefix = f"e22_retrieval_{artifact_id}"
        paths = {
            "name_candidates": config.cache_dir
            / "candidates"
            / f"{prefix}_name_tfidf.parquet",
            "address_candidates": config.cache_dir
            / "candidates"
            / f"{prefix}_address_tfidf.parquet",
            "best_legacy": config.cache_dir
            / "candidates"
            / f"{prefix}_best_legacy.parquet",
            "selected_candidates": config.cache_dir
            / "candidates"
            / f"{prefix}_selected.parquet",
            "report": config.experiments_dir / f"{prefix}_metrics.json",
        }
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite E2.2 artifacts: {existing}")
        metadata = {
            "artifact_id": artifact_id,
            "source_e21_artifact": E21_ARTIFACT_ID,
        }
        write_parquet_cache(name_candidates, paths["name_candidates"], metadata)
        write_parquet_cache(address_candidates, paths["address_candidates"], metadata)
        write_parquet_cache(best_legacy, paths["best_legacy"], metadata)
        write_parquet_cache(
            best_fusion_candidates, paths["selected_candidates"], metadata
        )
        report["artifacts"] = {key: str(value) for key, value in paths.items()}
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
        "e2.2-complete",
        artifact_id=artifact_id,
        best_density_variant=best_density_name,
        best_fusion=best_fusion_name,
        selected_metrics=report["selected_metrics"],
        total_seconds=report["performance"].get(
            "total_wall_seconds",
            report["performance"]["total_wall_seconds_before_persistence"],
        ),
        peak_rss_mib=_peak_rss_mib(),
    )
    return report
