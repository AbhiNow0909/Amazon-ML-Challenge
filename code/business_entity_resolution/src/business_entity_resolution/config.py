"""Central configuration for the Tier 0 pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration shared by all pipeline stages.

    Paths are anchored to the repository rather than the caller's working
    directory, which keeps CLI behavior reproducible.
    """

    seed: int = 42

    train_dir: Path = REPO_ROOT / "student_resource" / "dataset" / "train"
    test_dir: Path = REPO_ROOT / "student_resource" / "dataset" / "test"
    validator_path: Path = (
        REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"
    )
    cache_dir: Path = REPO_ROOT / "cache"
    experiments_dir: Path = REPO_ROOT / "experiments"
    output_dir: Path = REPO_ROOT / "output"
    models_dir: Path = REPO_ROOT / "models" / "lgbm"

    chunk_size: int = 100_000
    validation_fraction: float = 0.10
    validation_s1_limit: int = 2_000
    validation_distractors_per_source: int = 50_000

    candidate_k: int = 30
    evaluation_ks: Tuple[int, ...] = (10, 20, 30, 50)
    rare_name_token_max_df: int = 100
    rare_address_token_max_df: int = 50
    posting_list_cap: int = 250
    minimum_token_length: int = 3

    # Optional E1 name character TF-IDF retrieval. Disabled by default so E0
    # candidate behavior remains unchanged.
    use_name_tfidf: bool = False
    tfidf_name_top_k: int = 50
    tfidf_ngram_min: int = 3
    tfidf_ngram_max: int = 4
    tfidf_query_chunk_size: int = 128
    tfidf_max_features: Optional[int] = 300_000
    tfidf_min_df: int = 1

    # E1.1 validation search space. These settings only affect the explicit
    # E1.1 stage; default Tier 0 and E1 behavior remains unchanged.
    fusion_tfidf_top_ks: Tuple[int, ...] = (20, 30, 50)
    fusion_final_ks: Tuple[int, ...] = (30, 40, 50, 75)
    fusion_quota_configs: Tuple[Tuple[int, int, int], ...] = (
        (30, 20, 10),
        (30, 24, 6),
        (40, 30, 10),
        (40, 32, 8),
        (50, 35, 15),
        (50, 40, 10),
        (50, 45, 5),
        (75, 50, 25),
    )
    fusion_rrf_configs: Tuple[Tuple[float, float, float], ...] = (
        (2.0, 1.0, 20.0),
        (3.0, 1.0, 20.0),
        (3.0, 1.0, 60.0),
    )
    fusion_normalized_configs: Tuple[Tuple[float, float, float], ...] = (
        (0.70, 0.30, 0.10),
        (0.80, 0.20, 0.10),
        (0.65, 0.35, 0.15),
    )
    fusion_hybrid_minimum_legacy: Tuple[Tuple[int, int], ...] = (
        (30, 20),
        (30, 24),
        (40, 30),
        (40, 32),
        (50, 35),
        (50, 40),
        (50, 45),
        (75, 50),
    )

    # E2 pair classifier. The split seed is intentionally distinct from the
    # seed used to select the outer 2,000-S1 development universe.
    e2_split_seed: int = 202603
    e2_validation_fraction: float = 0.25
    e2_hard_negatives_per_positive: int = 5
    e2_close_competitor_tolerance: float = 0.05
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: int = 63
    lgbm_feature_fraction: float = 0.8
    lgbm_bagging_fraction: float = 0.8
    lgbm_bagging_freq: int = 1
    lgbm_min_data_in_leaf: int = 50
    lgbm_max_rounds: int = 2000
    lgbm_early_stopping_rounds: int = 100

    # Cheap blocking score weights.
    weight_exact_full_name: float = 5.0
    weight_exact_core_name: float = 4.5
    weight_exact_sorted_name: float = 4.0
    weight_rare_name_token: float = 1.0
    weight_exact_address: float = 3.0
    weight_rare_address_token: float = 0.5
    weight_same_country: float = 0.5
    weight_name_tfidf: float = 4.0

    # Precision-oriented E0 thresholds are tuned only on validation.
    rule_thresholds: Tuple[float, ...] = field(
        default_factory=lambda: (
            0.70,
            0.75,
            0.80,
            0.84,
            0.87,
            0.90,
            0.92,
            0.94,
            0.96,
            0.98,
        )
    )

    def as_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key, value in tuple(result.items()):
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, tuple):
                result[key] = list(value)
        return result

    def fingerprint(self, length: int = 12) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:length]

    def ensure_generated_dirs(self) -> None:
        """Create generated directories only when a run needs them."""

        for path in (
            self.cache_dir / "normalized",
            self.cache_dir / "candidates",
            self.cache_dir / "features",
            self.cache_dir / "predictions",
            self.experiments_dir,
            self.output_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = PipelineConfig()
