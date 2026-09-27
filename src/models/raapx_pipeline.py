"""
RAAP-X Quantile Post-Processing Training & Benchmarking Pipeline (Phase 3).
Coordinates:
1. Feature matrix assembly: Raw NWP + Error Memory + Terrain/Physics + Regime Probabilities.
2. Temporal split: Train (2020-2024), Validation (2025), Test (2026) without data leakage.
3. Automated Ablation Experiments:
   - EXP-03A: Raw NWP baseline (uncalibrated)
   - EXP-03B: Standard Quantile Corrector (without regime probabilities)
   - EXP-03C: RAAP-X Regime-Aware Quantile Corrector (with regime probabilities)
4. Verification & Experiment Logging via ExperimentTracker.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.features.error_memory import ErrorMemoryBuffer
from src.features.pipeline import FeatureEngineeringPipeline
from src.mlops.experiments import ExperimentRecord, ExperimentTracker, TemporalSplitInfo
from src.mlops.versioning import (
    DATASET_VERSION_DEFAULT,
    FEATURE_VERSION_DEFAULT,
    QUANTILE_MODEL_VERSION_DEFAULT,
)
from src.models.raapx_corrector import RAAPXQuantileCorrector, QuantileEvaluationMetrics
from src.postprocessing.constraints import enforce_quantile_monotonicity
from src.postprocessing.probabilistic_eval import compute_crps_from_quantiles
from src.postprocessing.quantiles import compute_pinball_loss
from src.regime.classifier import LightGBMRegimeClassifier
from src.regime.inference import RegimeInferenceEngine
from src.regime.labels import derive_provisional_regime_labels
from src.regime.schemas import REGIME_CLASSES, RegimeProbabilityVector
from src.utils.logging import get_logger

logger = get_logger("rain_repair.models.raapx_pipeline")


def generate_synthetic_multicyle_data(
    n_days: int = 120,
    n_grid_points: int = 15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a realistic multi-season, multi-grid meteorological dataset
    for Phase 3 training and verification covering 2024 (train), 2025 (val), and 2026 (test).
    """
    rng = np.random.RandomState(random_seed)
    records = []

    # Dates spanning 2024 (train), 2025 (val), and 2026 (test)
    years = [2024, 2025, 2026]
    days_per_year = n_days // len(years)

    grid_coords = [
        (18.0 + (i % 4) * 0.25, 73.0 + (i // 4) * 0.25)
        for i in range(n_grid_points)
    ]

    for yr in years:
        for day in range(1, days_per_year + 1):
            date_str = f"{yr}-07-{day:02d}" if day <= 31 else f"{yr}-08-{(day-31):02d}"

            # Meteorological seasonal state
            monsoon_active_pulse = float(np.sin(day / 7.0) > 0.2)
            orographic_boost = rng.uniform(0.0, 15.0)

            for grid_idx, (lat, lon) in enumerate(grid_coords):
                # 1. Physical and Terrain features
                elevation = 50.0 + grid_idx * 60.0
                slope = 0.5 + grid_idx * 0.4
                aspect_sin = float(np.sin(grid_idx))
                aspect_cos = float(np.cos(grid_idx))
                dist_to_coast = max(5.0, 150.0 - grid_idx * 8.0)
                terrain_roughness = 1.0 + grid_idx * 0.2

                moisture_flux_conv = float(rng.uniform(1.0, 10.0) + monsoon_active_pulse * 8.0)
                wind_shear_deep = float(rng.uniform(5.0, 25.0))
                relative_vorticity = float(rng.uniform(-2.0, 6.0) + monsoon_active_pulse * 3.0)
                orographic_uplift = float((elevation / 100.0) * slope * 0.8)
                vapor_pressure_deficit = float(rng.uniform(0.2, 2.5))
                cape = float(rng.uniform(400.0, 2800.0))

                # 2. Ground truth precipitation (physics-guided non-negative)
                base_rain = (
                    moisture_flux_conv * 2.2
                    + orographic_uplift * 1.5
                    + (cape / 400.0) * 1.8
                    + rng.exponential(scale=6.0)
                )
                obs_precip = round(max(0.0, float(base_rain)), 2)

                # 3. NWP forecast with systematic terrain & regime dependent bias
                # e.g., NWP underestimates extreme orographic rain and overestimates drizzle
                nwp_bias = (
                    -0.25 * orographic_uplift * 1.2
                    + (2.0 if obs_precip < 5.0 else -4.5 if obs_precip > 35.0 else 1.0)
                    + rng.normal(0.0, 3.5)
                )
                nwp_precip = round(max(0.0, float(obs_precip + nwp_bias)), 2)

                # 4. Historical Error Memory features (simulated trailing past error)
                error_lag_1d = round(float(nwp_bias + rng.normal(0.0, 1.5)), 2)
                error_lag_3d = round(float(nwp_bias * 0.9 + rng.normal(0.0, 1.0)), 2)
                error_lag_7d = round(float(nwp_bias * 0.85 + rng.normal(0.0, 0.8)), 2)
                error_lag_14d = round(float(nwp_bias * 0.80 + rng.normal(0.0, 0.6)), 2)
                rolling_bias_7d = round(float(nwp_bias * 0.88), 2)
                rolling_bias_14d = round(float(nwp_bias * 0.82), 2)
                rolling_mae_7d = round(float(abs(nwp_bias) + rng.uniform(1.0, 3.0)), 2)

                # 5. Continuous Regime Probability Vector
                # Higher Active Monsoon during pulses; Coastal/Orographic based on terrain
                p_active = max(0.01, 0.45 * monsoon_active_pulse + rng.uniform(0.05, 0.2))
                p_break = max(0.01, 0.45 * (1.0 - monsoon_active_pulse) + rng.uniform(0.02, 0.15))
                p_depression = max(0.01, 0.15 if monsoon_active_pulse and relative_vorticity > 3.0 else 0.03)
                p_coastal = max(0.01, 0.30 if dist_to_coast < 40.0 else 0.05)
                p_orographic = max(0.01, 0.35 if elevation > 300.0 else 0.05)
                p_wd = 0.02

                prob_vec = np.array([p_active, p_break, p_depression, p_coastal, p_orographic, p_wd], dtype=np.float32)
                prob_vec = prob_vec / np.sum(prob_vec)

                records.append({
                    "date": date_str,
                    "grid_id": f"grid_{grid_idx:02d}",
                    "lat": lat,
                    "lon": lon,
                    "nwp_precip": nwp_precip,
                    "obs_precip": obs_precip,
                    # Error memory
                    "error_lag_1d": error_lag_1d,
                    "error_lag_3d": error_lag_3d,
                    "error_lag_7d": error_lag_7d,
                    "error_lag_14d": error_lag_14d,
                    "rolling_bias_7d": rolling_bias_7d,
                    "rolling_bias_14d": rolling_bias_14d,
                    "rolling_mae_7d": rolling_mae_7d,
                    # Terrain and Physics
                    "elevation": elevation,
                    "slope": slope,
                    "aspect_sin": aspect_sin,
                    "aspect_cos": aspect_cos,
                    "dist_to_coast": dist_to_coast,
                    "terrain_roughness": terrain_roughness,
                    "moisture_flux_conv": moisture_flux_conv,
                    "wind_shear_deep": wind_shear_deep,
                    "relative_vorticity": relative_vorticity,
                    "orographic_uplift": orographic_uplift,
                    "vapor_pressure_deficit": vapor_pressure_deficit,
                    "cape": cape,
                    # Soft Regime Probability Vector
                    "prob_active_monsoon": round(float(prob_vec[0]), 4),
                    "prob_break_monsoon": round(float(prob_vec[1]), 4),
                    "prob_monsoon_depression": round(float(prob_vec[2]), 4),
                    "prob_coastal": round(float(prob_vec[3]), 4),
                    "prob_orographic": round(float(prob_vec[4]), 4),
                    "prob_western_disturbance": round(float(prob_vec[5]), 4),
                })

    return pd.DataFrame(records)


class RAAPXTrainingPipeline:
    """
    Automated Training & Evaluation Pipeline for Phase 3:
    Reuses dataset/feature versioning, temporal splits, and experiment tracking.
    """

    def __init__(
        self,
        experiment_tracker: Optional[ExperimentTracker] = None,
        dataset_version: str = DATASET_VERSION_DEFAULT,
        feature_version: str = FEATURE_VERSION_DEFAULT,
    ):
        self.tracker = experiment_tracker or ExperimentTracker()
        self.dataset_version = dataset_version
        self.feature_version = feature_version

    def split_temporal_dataset(
        self,
        df: pd.DataFrame,
        split_info: Optional[TemporalSplitInfo] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset strictly by date to prevent future data leakage.
        Train: <= 2024-12-31
        Val:   2025-01-01 to 2025-12-31
        Test:  >= 2026-01-01
        """
        df_sorted = df.sort_values(by="date").copy()

        train_mask = df_sorted["date"] <= "2024-12-31"
        val_mask = (df_sorted["date"] >= "2025-01-01") & (df_sorted["date"] <= "2025-12-31")
        test_mask = df_sorted["date"] >= "2026-01-01"

        # Fallback if dates are synthetic or within single year
        if not train_mask.any() or not test_mask.any():
            n = len(df_sorted)
            n_train = int(n * 0.6)
            n_val = int(n * 0.2)
            train_df = df_sorted.iloc[:n_train].copy()
            val_df = df_sorted.iloc[n_train : n_train + n_val].copy()
            test_df = df_sorted.iloc[n_train + n_val :].copy()
        else:
            train_df = df_sorted[train_mask].copy()
            val_df = df_sorted[val_mask].copy()
            test_df = df_sorted[test_mask].copy()

        logger.info(
            "Temporal split completed: Train=%d, Val=%d, Test=%d samples.",
            len(train_df),
            len(val_df),
            len(test_df),
        )
        return train_df, val_df, test_df

    def run_benchmark_experiments(
        self,
        df: Optional[pd.DataFrame] = None,
        quantiles: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the three canonical Phase 3 experiments and benchmarks results:
        1. EXP-03A: Raw NWP (Baseline)
        2. EXP-03B: Standard Quantile Model (No Regime Probabilities)
        3. EXP-03C: RAAP-X Quantile Corrector (With Soft Regime Probabilities)
        """
        q_list = quantiles or [0.50, 0.75, 0.90]
        data = df if df is not None else generate_synthetic_multicyle_data()

        split_info = TemporalSplitInfo(
            train_start="2024-06-01",
            train_end="2024-09-30",
            validation_start="2025-06-01",
            validation_end="2025-09-30",
            test_start="2026-06-01",
            test_end="2026-09-30",
        )
        train_df, val_df, test_df = self.split_temporal_dataset(data, split_info)

        y_train = train_df["obs_precip"].values
        y_val = val_df["obs_precip"].values
        y_test = test_df["obs_precip"].values

        # ----------------------------------------------------------------------
        # EXP-03A: Raw NWP Baseline
        # ----------------------------------------------------------------------
        nwp_test = test_df["nwp_precip"].values
        nwp_mae = float(np.mean(np.abs(nwp_test - y_test)))
        raw_pinball = {
            f"pinball_p{int(a*100)}": round(compute_pinball_loss(y_test, nwp_test, a), 4)
            for a in q_list
        }
        mean_raw_pinball = round(float(np.mean(list(raw_pinball.values()))), 4)

        rec_a = ExperimentRecord(
            experiment_id="EXP-03A-RAW-NWP",
            experiment_name="Raw NWP Forecast Baseline",
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            dataset_version=self.dataset_version,
            feature_version=self.feature_version,
            model_type="Raw_NWP_Baseline",
            hyperparameters={"quantiles": q_list},
            temporal_split=split_info,
            metrics={"p50_mae": round(nwp_mae, 4), "mean_pinball_loss": mean_raw_pinball, **raw_pinball},
            regime_metrics={},
            event_metrics={},
        )
        self.tracker.log_experiment(rec_a)

        # ----------------------------------------------------------------------
        # EXP-03B: Standard Quantile Model (Ablation: Without Regime Probabilities)
        # ----------------------------------------------------------------------
        regime_feature_cols = [c for c in train_df.columns if c.startswith("prob_")]
        cols_no_regimes = [
            c for c in train_df.select_dtypes(include=[np.number]).columns
            if c not in ["obs_precip", "lat", "lon"] and c not in regime_feature_cols
        ]

        model_std = RAAPXQuantileCorrector(
            quantiles=q_list,
            n_estimators=100,
            learning_rate=0.05,
            random_seed=42,
        )
        model_std.fit(train_df, y_train, feature_cols=cols_no_regimes)
        metrics_std = model_std.evaluate(test_df, y_test)

        rec_b = model_std.log_to_experiment_tracker(
            tracker=self.tracker,
            experiment_id="EXP-03B-STD-QUANTILE",
            experiment_name="Standard Quantile Regressor (No Regime Intelligence)",
            temporal_split=split_info,
            metrics=metrics_std,
            dataset_version=self.dataset_version,
            feature_version=self.feature_version,
        )

        # ----------------------------------------------------------------------
        # EXP-03C: Full RAAP-X Quantile Corrector (With Regime Probabilities)
        # ----------------------------------------------------------------------
        cols_with_regimes = [
            c for c in train_df.select_dtypes(include=[np.number]).columns
            if c not in ["obs_precip", "lat", "lon"]
        ]

        model_raapx = RAAPXQuantileCorrector(
            quantiles=q_list,
            n_estimators=100,
            learning_rate=0.05,
            random_seed=42,
        )
        model_raapx.fit(train_df, y_train, feature_cols=cols_with_regimes)
        metrics_raapx = model_raapx.evaluate(test_df, y_test)

        rec_c = model_raapx.log_to_experiment_tracker(
            tracker=self.tracker,
            experiment_id="EXP-03C-RAAPX-REGIME",
            experiment_name="RAAP-X Regime-Aware Quantile Corrector",
            temporal_split=split_info,
            metrics=metrics_raapx,
            dataset_version=self.dataset_version,
            feature_version=self.feature_version,
        )

        # Summary comparison table
        summary_rows = [
            {
                "Experiment ID": "EXP-03A-RAW-NWP",
                "Model Description": "Raw NWP Uncorrected Baseline",
                "Mean Pinball Loss": mean_raw_pinball,
                "P50 Pinball": raw_pinball.get("pinball_p50", 0.0),
                "P90 Pinball": raw_pinball.get("pinball_p90", 0.0),
                "P50 MAE (mm)": round(nwp_mae, 4),
                "MAE Improvement (%)": 0.0,
                "Non-Crossing (%)": 100.0,
            },
            {
                "Experiment ID": "EXP-03B-STD-QUANTILE",
                "Model Description": "Standard Quantile Regressor (Ablation)",
                "Mean Pinball Loss": metrics_std.mean_pinball_loss,
                "P50 Pinball": metrics_std.pinball_losses.get("pinball_p50", 0.0),
                "P90 Pinball": metrics_std.pinball_losses.get("pinball_p90", 0.0),
                "P50 MAE (mm)": metrics_std.p50_mae,
                "MAE Improvement (%)": metrics_std.mae_improvement_pct,
                "Non-Crossing (%)": round((1.0 - metrics_std.crossing_rate_after_rearrangement) * 100.0, 1),
            },
            {
                "Experiment ID": "EXP-03C-RAAPX-REGIME",
                "Model Description": "RAAP-X Regime-Aware Quantile Corrector",
                "Mean Pinball Loss": metrics_raapx.mean_pinball_loss,
                "P50 Pinball": metrics_raapx.pinball_losses.get("pinball_p50", 0.0),
                "P90 Pinball": metrics_raapx.pinball_losses.get("pinball_p90", 0.0),
                "P50 MAE (mm)": metrics_raapx.p50_mae,
                "MAE Improvement (%)": metrics_raapx.mae_improvement_pct,
                "Non-Crossing (%)": round((1.0 - metrics_raapx.crossing_rate_after_rearrangement) * 100.0, 1),
            },
        ]
        comparison_df = pd.DataFrame(summary_rows)

        return {
            "comparison_table": comparison_df,
            "raw_nwp_metrics": rec_a.metrics,
            "standard_quantile_metrics": metrics_std.to_dict(),
            "raapx_metrics": metrics_raapx.to_dict(),
            "model_raapx": model_raapx,
            "test_df": test_df,
        }
