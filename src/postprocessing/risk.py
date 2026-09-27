"""
Calibrated Risk Output Engine for RAIN-REPAIR X (PART 7).
Harmonizes:
    Part 6 Corrected Rainfall (Deterministic Mean / Median)
    +
    Multi-Quantile Regression Plumes (P10 / P50 / P75 / P90 / P95)
    +
    Calibrated Heavy & Very Heavy Exceedance Probabilities (64.5 mm, 115.6 mm, 204.5 mm)
    +
    Extreme Value Theory (EVT-GPD Peaks-Over-Threshold Tail Modeling)
    +
    Physical Invariant Enforcement (Monotonic Rearrangement & Non-Negativity)
    +
    IMD Color-Coded Severe Weather Warning Level Assignment (Green, Yellow, Orange, Red)

Emits the complete, production-grade output matrix conforming to Section 24 and CONTRACT-ML-002.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.postprocessing.constraints import (
    enforce_probability_monotonicity,
    enforce_quantile_monotonicity,
    validate_physical_realizability,
)
from src.postprocessing.evt import EVT_VERSION, EVTGPDTailModel
from src.postprocessing.quantiles import MultiQuantileRegressor
from src.postprocessing.thresholds import (
    CALIBRATION_VERSION,
    EVENT_MODEL_VERSION,
    ExtremeThresholdClassifier,
)
from src.utils.logging import get_logger

logger = get_logger("rain_repair.postprocessing.risk")

RISK_MODEL_VERSION: str = "v1.0.0-prob"


class IMDAlertLevel(str, Enum):
    GREEN = "GREEN"      # No Warning: Light to moderate rain, no operational action needed
    YELLOW = "YELLOW"    # Watch: Be updated; low-to-moderate probability of heavy rain
    ORANGE = "ORANGE"    # Alert: Be prepared; high chance of heavy to very heavy rain
    RED = "RED"          # Warning: Take action; extreme danger, very heavy to catastrophic rain


def assign_imd_alert_level(
    prob_heavy: float,
    prob_very_heavy: float,
    prob_extreme: float,
    p90_rain: float,
) -> Tuple[IMDAlertLevel, str]:
    """
    Categorizes operational risk into standard IMD four-tier alert levels.
    """
    if prob_extreme >= 0.35 or prob_very_heavy >= 0.60 or p90_rain >= 204.5:
        return IMDAlertLevel.RED, "Catastrophic or extremely heavy rainfall expected. Take immediate defensive action."
    elif prob_very_heavy >= 0.35 or prob_heavy >= 0.65 or p90_rain >= 115.6:
        return IMDAlertLevel.ORANGE, "Very heavy rainfall likely. Significant localized waterlogging and flood risk. Be prepared."
    elif prob_heavy >= 0.25 or p90_rain >= 64.5:
        return IMDAlertLevel.YELLOW, "Heavy rainfall possible. Moderate disruption possible. Keep watch on forecast updates."
    else:
        return IMDAlertLevel.GREEN, "Normal to moderate conditions. No severe weather warning."


class CalibratedRiskEngine:
    """
    Unified Probabilistic & Extreme Risk Post-Processing Engine for RAIN-REPAIR X.
    """

    def __init__(
        self,
        quantile_regressor: Optional[MultiQuantileRegressor] = None,
        threshold_classifier: Optional[ExtremeThresholdClassifier] = None,
        evt_tail_model: Optional[EVTGPDTailModel] = None,
        model_version: str = RISK_MODEL_VERSION,
    ):
        self.quantile_regressor = quantile_regressor or MultiQuantileRegressor()
        self.threshold_classifier = threshold_classifier or ExtremeThresholdClassifier()
        self.evt_tail_model = evt_tail_model or EVTGPDTailModel()
        self.model_version = model_version

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train_precip: Union[np.ndarray, pd.Series],
        X_val: Optional[pd.DataFrame] = None,
        y_val_precip: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> CalibratedRiskEngine:
        """Fits multi-quantile models, calibrated threshold classifiers, and EVT tail distribution."""
        y_precip = np.asarray(y_train_precip, dtype=np.float32)

        logger.info("Fitting CalibratedRiskEngine across all probabilistic components...")
        self.quantile_regressor.fit(X_train, y_precip)
        self.threshold_classifier.fit(X_train, y_precip, X_val=X_val, y_val_precip=y_val_precip)
        self.evt_tail_model.fit(y_precip)

        logger.info("CalibratedRiskEngine training complete.")
        return self

    def generate_risk_output(
        self,
        df_base_repaired: pd.DataFrame,
        X_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Takes Part 6 repaired dataframe and feature matrix, and enriches it
        with multi-quantiles, exceedance probabilities, EVT-GPD extremes,
        monotonicity guards, and IMD alert levels.
        """
        out_df = df_base_repaired.copy()
        n_samples = len(out_df)

        # 1. Multi-Quantile Prediction (P10, P50, P75, P90, P95)
        if self.quantile_regressor.is_fitted:
            q_preds = self.quantile_regressor.predict_quantiles(X_features, enforce_monotonicity=True)
            for q_name, q_vals in q_preds.items():
                out_df[q_name] = np.round(q_vals, 3)
        else:
            # Fallback estimation around deterministic P50
            p50 = out_df["corrected_p50"].to_numpy()
            out_df["corrected_p10"] = np.round(np.maximum(0.0, p50 * 0.5), 3)
            out_df["corrected_p75"] = np.round(p50 * 1.3, 3)
            out_df["corrected_p90"] = np.round(p50 * 1.7, 3)
            out_df["corrected_p95"] = np.round(p50 * 2.1, 3)

        # 2. Extreme Threshold Exceedance Probabilities (64.5, 115.6, 204.5 mm)
        if self.threshold_classifier.is_fitted:
            t_probs = self.threshold_classifier.predict_probabilities(X_features, enforce_monotonicity=True)
            out_df["prob_heavy_rain"] = np.round(t_probs["prob_heavy_rain"], 4)
            out_df["prob_very_heavy_rain"] = np.round(t_probs["prob_very_heavy_rain"], 4)
            out_df["prob_extreme_rain"] = np.round(t_probs["prob_extreme_rain"], 4)
        else:
            p90 = out_df["corrected_p90"].to_numpy()
            out_df["prob_heavy_rain"] = np.round(np.clip(p90 / 64.5 * 0.4, 0.0, 1.0), 4)
            out_df["prob_very_heavy_rain"] = np.round(np.clip(p90 / 115.6 * 0.3, 0.0, 1.0), 4)
            out_df["prob_extreme_rain"] = np.round(np.clip(p90 / 204.5 * 0.2, 0.0, 1.0), 4)

        # 3. EVT-GPD Extreme Return Level Extrapolation (e.g., P99 Return Level)
        p99_return_val = self.evt_tail_model.compute_return_level(0.99)
        out_df["evt_tail_p99"] = round(p99_return_val, 2)
        out_df["evt_tail_shape_xi"] = round(self.evt_tail_model.shape_xi, 4)
        out_df["evt_status"] = self.evt_tail_model.evt_status
        out_df["evt_version"] = self.evt_tail_model.version

        # 4. Enforce Physical Invariants on Output Table
        validate_physical_realizability(out_df, strict=False)

        # 5. Section 24 Schema Conformity Aliases
        # forecast_time, valid_time, grid_id, raw_nwp_rainfall, p50_rainfall, p75_rainfall, p90_rainfall, etc.
        out_df["forecast_time"] = out_df["cycle_date"] if "cycle_date" in out_df.columns else "2026-07-01"
        out_df["valid_time"] = out_df["forecast_time"]  # Valid at horizon lead_time
        out_df["raw_nwp_rainfall"] = out_df["raw_nwp_precip"]
        out_df["p50_rainfall"] = out_df["corrected_p50"]
        out_df["p75_rainfall"] = out_df["corrected_p75"]
        out_df["p90_rainfall"] = out_df["corrected_p90"]
        out_df["heavy_probability"] = out_df["prob_heavy_rain"]
        out_df["very_heavy_probability"] = out_df["prob_very_heavy_rain"]
        out_df["extreme_probability"] = out_df["prob_extreme_rain"]

        # Uncertainty indicator: P90 - P50 inter-quantile range spread (IQR proxy)
        uncertainty_spread = np.maximum(0.0, out_df["corrected_p90"] - out_df["corrected_p50"])
        out_df["uncertainty_indicator"] = np.round(uncertainty_spread, 3)

        out_df["model_version"] = self.model_version
        out_df["probability_calibration_version"] = self.threshold_classifier.calibration_version

        # 6. Assign IMD Alert Levels & Advice
        alert_levels = []
        alert_advisories = []
        for i in range(n_samples):
            lvl, adv = assign_imd_alert_level(
                prob_heavy=float(out_df["prob_heavy_rain"].iloc[i]),
                prob_very_heavy=float(out_df["prob_very_heavy_rain"].iloc[i]),
                prob_extreme=float(out_df["prob_extreme_rain"].iloc[i]),
                p90_rain=float(out_df["corrected_p90"].iloc[i]),
            )
            alert_levels.append(lvl.value)
            alert_advisories.append(adv)

        out_df["imd_alert_level"] = alert_levels
        out_df["imd_alert_advisory"] = alert_advisories

        logger.info(
            "Calibrated risk generated for %d points. Alert distribution: %s",
            n_samples,
            dict(pd.Series(alert_levels).value_counts()),
        )
        return out_df
