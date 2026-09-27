"""
RAAP-X Regime-Aware Quantile Rainfall Correction Engine.
Core Component for Phase 3 — Regime-Aware Quantile Rainfall Correction.

Implements:
Raw NWP Forecast
        +
Historical Forecast Error
        +
Local/Physical Features
        +
Regime Probability Vector
        ↓
RAAP-X Quantile Corrector
        ↓
P50, P75, P90

Physical Realizability Enforced:
1. Strict Non-Negativity: q_tau >= 0.0 mm/24h.
2. Monotonicity / Non-Crossing (Rearrangement Operator):
   0.0 <= q_50 <= q_75 <= q_90 (and optional q_10 <= q_50 <= q_75 <= q_90 <= q_95).
3. Zero Temporal Leakage: Causal verification on historical error and features.
4. Experiment Tracking & Immutable Versioning Integration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import HistGradientBoostingRegressor

from src.mlops.experiments import ExperimentRecord, ExperimentTracker, TemporalSplitInfo
from src.mlops.versioning import (
    DATASET_VERSION_DEFAULT,
    FEATURE_VERSION_DEFAULT,
    QUANTILE_MODEL_VERSION_DEFAULT,
)
from src.postprocessing.constraints import enforce_quantile_monotonicity
from src.postprocessing.probabilistic_eval import (
    compute_crps_from_quantiles,
    evaluate_quantile_calibration,
)
from src.postprocessing.quantiles import compute_pinball_loss
from src.postprocessing.target import validate_target_isolation
from src.regime.schemas import REGIME_CLASSES, RegimeProbabilityVector
from src.utils.logging import get_logger

logger = get_logger("rain_repair.models.raapx_corrector")

# Canonical quantiles focused on P50, P75, P90 as specified in Phase 3
DEFAULT_QUANTILES: List[float] = [0.50, 0.75, 0.90]

# Standard Quantile Label Mappings
QUANTILE_NAMES: Dict[float, str] = {
    0.10: "p10",
    0.50: "p50",
    0.75: "p75",
    0.90: "p90",
    0.95: "p95",
}


@dataclass
class QuantileEvaluationMetrics:
    """Standardized metrics payload for Quantile Post-Processing Models."""
    pinball_losses: Dict[str, float]
    mean_pinball_loss: float
    empirical_coverage: Dict[str, float]
    coverage_error: Dict[str, float]
    mean_coverage_error: float
    crps: float
    raw_nwp_mae: float
    p50_mae: float
    mae_improvement_pct: float
    crossing_rate_before_rearrangement: float
    crossing_rate_after_rearrangement: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RAAPXQuantileCorrector:
    """
    RAAP-X Quantile Precipitation Post-Processing Engine.
    Converts raw NWP rainfall forecasts into calibrated probabilistic quantile plumes
    (P50, P75, P90) conditioned on historical error memory, physical/terrain features,
    and continuous monsoon synoptic regime probabilities.
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_seed: int = 42,
        mode: str = "unified",  # "unified" or "moe" (mixture of experts)
        model_version: str = QUANTILE_MODEL_VERSION_DEFAULT,
    ):
        self.quantiles = sorted(quantiles or DEFAULT_QUANTILES)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_seed = random_seed
        self.mode = mode
        self.model_version = model_version

        # Model store: mapping alpha -> fitted estimator (or dict of expert models per regime)
        self.models: Dict[float, Any] = {}
        self.expert_models: Dict[float, Dict[str, Any]] = {}
        self.feature_cols: List[str] = []
        self.feature_importances_: Dict[float, Dict[str, float]] = {}
        self.training_metadata: Dict[str, Any] = {}
        self.is_fitted = False

    @classmethod
    def get_canonical_feature_groups(cls) -> Dict[str, List[str]]:
        """Returns standard feature groups expected by RAAP-X."""
        return {
            "nwp": [
                "nwp_precip",
                "nwp_precip_log",
                "nwp_t2m",
                "nwp_q2m",
                "nwp_u10",
                "nwp_v10",
                "nwp_wind_speed",
                "nwp_mslp",
            ],
            "error_memory": [
                "error_lag_1d",
                "error_lag_3d",
                "error_lag_7d",
                "error_lag_14d",
                "rolling_bias_7d",
                "rolling_bias_14d",
                "rolling_mae_7d",
            ],
            "terrain_physics": [
                "elevation",
                "slope",
                "aspect_sin",
                "aspect_cos",
                "dist_to_coast",
                "terrain_roughness",
                "moisture_flux_conv",
                "wind_shear_deep",
                "relative_vorticity",
                "orographic_uplift",
                "vapor_pressure_deficit",
                "cape",
            ],
            "regime_probabilities": [
                "prob_active_monsoon",
                "prob_break_monsoon",
                "prob_monsoon_depression",
                "prob_coastal",
                "prob_orographic",
                "prob_western_disturbance",
            ],
        }

    def _extract_regime_prob_columns(self, df: pd.DataFrame) -> List[str]:
        """Detects and returns regime probability columns from dataframe."""
        regime_cols = []
        for col in df.columns:
            c_lower = col.lower()
            if c_lower.startswith("prob_") or c_lower.startswith("regime_prob_"):
                regime_cols.append(col)
        return regime_cols

    def prepare_feature_matrix(
        self,
        df: pd.DataFrame,
        regime_probs_df: Optional[pd.DataFrame] = None,
        error_memory_df: Optional[pd.DataFrame] = None,
        drop_targets: bool = False,
    ) -> pd.DataFrame:
        """
        Integrates Raw NWP, Historical Forecast Error, Local/Physical Features,
        and Regime Probability Vector into a single model-ready DataFrame.
        """
        X = df.copy()
        if drop_targets:
            target_cols = [c for c in ["obs_precip", "observation", "ground_truth", "target", "signed_error"] if c in X.columns]
            X = X.drop(columns=target_cols, errors="ignore")

        # Merge regime probabilities if supplied separately
        if regime_probs_df is not None:
            prob_cols = [c for c in regime_probs_df.columns if c.lower().startswith("prob_")]
            if prob_cols:
                for c in prob_cols:
                    X[c] = regime_probs_df[c].values

        # Merge error memory if supplied separately
        if error_memory_df is not None:
            err_cols = [c for c in error_memory_df.columns if "error" in c.lower() or "bias" in c.lower()]
            if err_cols:
                for c in err_cols:
                    X[c] = error_memory_df[c].values

        return X

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> RAAPXQuantileCorrector:
        """
        Fits multi-quantile Gradient Boosted Decision Trees minimizing pinball loss per quantile.

        Quantiles:
            alpha in [0.50, 0.75, 0.90] (or custom quantiles list).
        """
        # Exclude targets, identifiers, and timestamp columns from predictors
        excluded = {
            "obs_precip",
            "observation",
            "ground_truth",
            "signed_error",
            "absolute_error",
            "target",
            "target_precip",
            "grid_id",
            "timestamp",
            "forecast_time",
            "valid_date",
            "cycle_date",
            "date",
            "current_regime",
            "previous_regime",
            "transition_status",
            "transition_from",
            "transition_to",
        }

        if feature_cols is not None:
            cols = feature_cols
        else:
            cols = [
                c for c in X_train.select_dtypes(include=[np.number]).columns
                if c not in excluded
            ]

        self.feature_cols = cols

        # Strict target isolation validation on the selected feature predictors
        validate_target_isolation(X_train[cols])
        if X_val is not None:
            validate_target_isolation(X_val[cols])

        X_tr = X_train[cols].fillna(0.0).to_numpy(dtype=np.float32)
        y_tr = np.asarray(y_train, dtype=np.float32)

        # Enforce non-negativity on training ground truth precipitation
        y_tr = np.maximum(y_tr, 0.0)

        logger.info(
            "Fitting RAAP-X Quantile Corrector across quantiles %s with %d samples x %d features (mode=%s).",
            self.quantiles,
            len(X_tr),
            len(cols),
            self.mode,
        )

        for alpha in self.quantiles:
            q_name = QUANTILE_NAMES.get(alpha, f"p{int(alpha*100)}")
            logger.info("Training quantile model for alpha=%.2f (%s)...", alpha, q_name)

            if HAS_LIGHTGBM:
                model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=alpha,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    num_leaves=self.num_leaves,
                    min_child_samples=self.min_child_samples,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    random_state=self.random_seed,
                    verbose=-1,
                )
            else:
                model = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=alpha,
                    max_iter=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=self.random_seed,
                )

            model.fit(X_tr, y_tr)
            self.models[alpha] = model

            # Extract feature importance
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_.astype(float)
                total = float(np.sum(importances))
                norm_imp = (importances / total) if total > 0 else importances
                self.feature_importances_[alpha] = {
                    feat: round(float(norm_imp[idx]), 4)
                    for idx, feat in enumerate(cols)
                }

        self.training_metadata = {
            "n_samples": len(X_tr),
            "n_features": len(cols),
            "feature_columns": cols,
            "quantiles": self.quantiles,
            "fitted_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_version": self.model_version,
            "has_lightgbm": HAS_LIGHTGBM,
        }
        self.is_fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        enforce_constraints: bool = True,
    ) -> pd.DataFrame:
        """
        Predicts rainfall quantiles (P50, P75, P90) for input weather states.

        Guarantees Physical Constraints (when enforce_constraints=True):
        1. Non-Negativity: q_tau >= 0.0 mm.
        2. Monotonicity: q_50 <= q_75 <= q_90 via Chernozhukov Rearrangement Operator.
        """
        if not self.is_fitted:
            raise RuntimeError("RAAPXQuantileCorrector must be fitted before predict.")

        validate_target_isolation(X[self.feature_cols])
        X_mat = X[self.feature_cols].fillna(0.0).to_numpy(dtype=np.float32)

        # 1. Raw quantile predictions
        raw_preds: Dict[float, np.ndarray] = {}
        for alpha, model in self.models.items():
            raw_preds[alpha] = model.predict(X_mat).astype(np.float32)

        # 2. Check crossing before rearrangement for diagnostics
        if len(self.quantiles) > 1:
            stack_raw = np.column_stack([raw_preds[a] for a in self.quantiles])
            n_crossings = int(np.sum(np.diff(stack_raw, axis=1) < -1e-4))
        else:
            n_crossings = 0

        # 3. Constraint enforcement
        if enforce_constraints:
            monotonic_preds = enforce_quantile_monotonicity(
                raw_preds,
                enforce_non_negativity=True,
                strict_check=False,
            )
        else:
            monotonic_preds = {a: np.maximum(raw_preds[a], 0.0) for a in raw_preds}

        # 4. Assemble structured result DataFrame
        result_df = pd.DataFrame(index=X.index)

        # Include Raw NWP precipitation if present in input
        nwp_col = None
        for candidate in ["nwp_precip", "raw_nwp_precip", "precip_raw"]:
            if candidate in X.columns:
                nwp_col = candidate
                result_df["nwp_precip"] = X[candidate].values
                break

        for alpha in self.quantiles:
            q_name = QUANTILE_NAMES.get(alpha, f"p{int(alpha*100)}")
            result_df[q_name] = monotonic_preds[alpha]

        # Derived probabilistic diagnostic quantities
        if 0.50 in monotonic_preds and 0.90 in monotonic_preds:
            result_df["spread_p90_p50"] = monotonic_preds[0.90] - monotonic_preds[0.50]

        if nwp_col and 0.50 in monotonic_preds:
            result_df["predicted_bias_p50"] = monotonic_preds[0.50] - result_df["nwp_precip"]

        return result_df

    def predict_quantiles_dict(
        self,
        X: pd.DataFrame,
        enforce_constraints: bool = True,
    ) -> Dict[float, np.ndarray]:
        """Convenience method returning Dict[float, np.ndarray] of predicted quantiles."""
        df_out = self.predict(X, enforce_constraints=enforce_constraints)
        return {
            alpha: df_out[QUANTILE_NAMES.get(alpha, f"p{int(alpha*100)}")].to_numpy(dtype=np.float32)
            for alpha in self.quantiles
        }

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: Union[np.ndarray, pd.Series],
    ) -> QuantileEvaluationMetrics:
        """
        Comprehensive evaluation of RAAP-X Quantile Corrector on held-out test data.
        Computes pinball losses, empirical coverage, coverage error, CRPS, and NWP benchmark improvement.
        """
        if not self.is_fitted:
            raise RuntimeError("RAAPXQuantileCorrector must be fitted before evaluate.")

        validate_target_isolation(X_test[self.feature_cols])
        y_true = np.asarray(y_test, dtype=np.float32)

        # Raw predictions before rearrangement (to audit crossings)
        X_mat = X_test[self.feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        raw_preds = {a: self.models[a].predict(X_mat).astype(np.float32) for a in self.quantiles}
        stack_raw = np.column_stack([raw_preds[a] for a in self.quantiles])
        total_intervals = len(X_mat) * (len(self.quantiles) - 1)
        raw_crossings = int(np.sum(np.diff(stack_raw, axis=1) < -1e-4))
        crossing_rate_raw = round(raw_crossings / max(1, total_intervals), 4)

        # Rearranged predictions (enforcing monotonicity and non-negativity)
        rearranged_preds = enforce_quantile_monotonicity(raw_preds, enforce_non_negativity=True)
        stack_post = np.column_stack([rearranged_preds[a] for a in self.quantiles])
        post_crossings = int(np.sum(np.diff(stack_post, axis=1) < -1e-4))
        crossing_rate_post = round(post_crossings / max(1, total_intervals), 4)

        # 1. Pinball Losses
        pinball_dict: Dict[str, float] = {}
        for alpha in self.quantiles:
            q_name = QUANTILE_NAMES.get(alpha, f"p{int(alpha*100)}")
            loss = compute_pinball_loss(y_true, rearranged_preds[alpha], alpha)
            pinball_dict[f"pinball_{q_name}"] = round(loss, 4)

        mean_pinball = round(float(np.mean(list(pinball_dict.values()))), 4)

        # 2. Empirical Coverage and Gap
        coverage_dict: Dict[str, float] = {}
        gap_dict: Dict[str, float] = {}
        for alpha in self.quantiles:
            q_name = QUANTILE_NAMES.get(alpha, f"p{int(alpha*100)}")
            cov = float(np.mean(y_true <= rearranged_preds[alpha]))
            coverage_dict[f"coverage_{q_name}"] = round(cov, 4)
            gap_dict[f"gap_{q_name}"] = round(abs(cov - alpha), 4)

        mean_gap = round(float(np.mean(list(gap_dict.values()))), 4)

        # 3. CRPS approximation
        crps = compute_crps_from_quantiles(y_true, rearranged_preds)

        # 4. Benchmark against raw NWP precipitation
        nwp_mae = 0.0
        p50_mae = 0.0
        improvement_pct = 0.0

        for candidate in ["nwp_precip", "raw_nwp_precip"]:
            if candidate in X_test.columns:
                nwp_arr = X_test[candidate].to_numpy(dtype=np.float32)
                nwp_mae = round(float(np.mean(np.abs(nwp_arr - y_true))), 4)
                break

        if 0.50 in rearranged_preds:
            p50_mae = round(float(np.mean(np.abs(rearranged_preds[0.50] - y_true))), 4)
            if nwp_mae > 1e-4:
                improvement_pct = round(((nwp_mae - p50_mae) / nwp_mae) * 100.0, 2)

        metrics = QuantileEvaluationMetrics(
            pinball_losses=pinball_dict,
            mean_pinball_loss=mean_pinball,
            empirical_coverage=coverage_dict,
            coverage_error=gap_dict,
            mean_coverage_error=mean_gap,
            crps=crps,
            raw_nwp_mae=nwp_mae,
            p50_mae=p50_mae,
            mae_improvement_pct=improvement_pct,
            crossing_rate_before_rearrangement=crossing_rate_raw,
            crossing_rate_after_rearrangement=crossing_rate_post,
        )
        return metrics

    def log_to_experiment_tracker(
        self,
        tracker: ExperimentTracker,
        experiment_id: str,
        experiment_name: str,
        temporal_split: TemporalSplitInfo,
        metrics: QuantileEvaluationMetrics,
        dataset_version: str = DATASET_VERSION_DEFAULT,
        feature_version: str = FEATURE_VERSION_DEFAULT,
    ) -> ExperimentRecord:
        """Logs model run, hyperparameters, and evaluation metrics to the experiment tracking system."""
        flat_metrics = {
            "mean_pinball_loss": metrics.mean_pinball_loss,
            "crps": metrics.crps,
            "mean_coverage_error": metrics.mean_coverage_error,
            "p50_mae": metrics.p50_mae,
            "raw_nwp_mae": metrics.raw_nwp_mae,
            "mae_improvement_pct": metrics.mae_improvement_pct,
            "crossing_rate_post": metrics.crossing_rate_after_rearrangement,
        }
        flat_metrics.update(metrics.pinball_losses)
        flat_metrics.update(metrics.empirical_coverage)
        flat_metrics.update(metrics.coverage_error)

        hyperparams = {
            "quantiles": self.quantiles,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "mode": self.mode,
            "n_features": len(self.feature_cols),
        }

        record = ExperimentRecord(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            dataset_version=dataset_version,
            feature_version=feature_version,
            model_type=f"RAAPX_Quantile_Corrector_{self.mode.upper()}",
            hyperparameters=hyperparams,
            temporal_split=temporal_split,
            metrics=flat_metrics,
            regime_metrics={
                "quantiles": self.quantiles,
                "coverage_error": metrics.coverage_error,
            },
            event_metrics={
                "crossing_rate_before": metrics.crossing_rate_before_rearrangement,
                "crossing_rate_after": metrics.crossing_rate_after_rearrangement,
            },
        )
        tracker.log_experiment(record)
        return record

    def save(self, model_dir: Union[str, Path]) -> Path:
        """Persists trained model estimators, feature lists, and metadata to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted RAAPXQuantileCorrector.")

        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        # 1. Save quantile models
        models_file = path / f"raapx_models_{self.model_version}.joblib"
        joblib.dump(self.models, models_file)

        # 2. Save metadata JSON
        meta = {
            "model_version": self.model_version,
            "quantiles": self.quantiles,
            "feature_cols": self.feature_cols,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "mode": self.mode,
            "training_metadata": self.training_metadata,
            "feature_importances": self.feature_importances_,
            "saved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        meta_file = path / f"raapx_metadata_{self.model_version}.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved RAAP-X Quantile Corrector (%s) to %s", self.model_version, path)
        return path

    @classmethod
    def load(cls, model_dir: Union[str, Path], model_version: str = QUANTILE_MODEL_VERSION_DEFAULT) -> RAAPXQuantileCorrector:
        """Loads a saved RAAP-X Quantile Corrector from disk."""
        path = Path(model_dir)
        meta_file = path / f"raapx_metadata_{model_version}.json"
        models_file = path / f"raapx_models_{model_version}.joblib"

        if not meta_file.exists() or not models_file.exists():
            raise FileNotFoundError(f"Missing model files at {path} for version {model_version}")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(
            quantiles=meta["quantiles"],
            n_estimators=meta["n_estimators"],
            learning_rate=meta["learning_rate"],
            max_depth=meta["max_depth"],
            num_leaves=meta["num_leaves"],
            mode=meta.get("mode", "unified"),
            model_version=meta["model_version"],
        )
        instance.models = joblib.load(models_file)
        instance.feature_cols = meta["feature_cols"]
        instance.feature_importances_ = {float(k): v for k, v in meta.get("feature_importances", {}).items()}
        instance.training_metadata = meta.get("training_metadata", {})
        instance.is_fitted = True

        logger.info("Loaded RAAP-X Quantile Corrector (%s) from %s", model_version, path)
        return instance
