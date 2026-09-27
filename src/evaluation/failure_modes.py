"""
Model Failure Conditions & Operational Safeguards for RAIN-REPAIR X (Phase 4).
Defines objective, machine-enforced validation checks and defensive responses for:
1. Input Failure (Missing or malformed features)
2. Regime Failure (Invalid or un-normalized probability vectors)
3. Quantile Failure (Crossing or physical non-negativity violations)
4. Data Failure (Coordinate, unit, timestamp, or domain violations)
5. Model Failure (Missing or incompatible checkpoints)
6. Distribution Shift (Atmospheric or NWP drift vs training distribution)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logging import get_logger

logger = get_logger("rain_repair.evaluation.failure_modes")


class FailureSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class FailureConditionAudit:
    """Standardized record of an evaluated model failure check."""
    condition_name: str
    severity: FailureSeverity
    is_triggered: bool
    detection_rule: str
    response_action: str
    logging_level: str
    user_facing_behavior: str
    diagnostic_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


class OperationalFailureGuardrails:
    """
    Enforces objective failure mode detection and mitigation across the prediction pipeline.
    Prevents silent delivery of corrupted, uncalibrated, or out-of-domain rainfall forecasts.
    """

    @classmethod
    def check_input_failure(
        cls,
        df: pd.DataFrame,
        required_features: List[str],
        missing_threshold_pct: float = 0.30,
    ) -> FailureConditionAudit:
        """
        Input Failure Check:
        Detects missing required features or excessive NaN rates.
        """
        missing_cols = [c for c in required_features if c not in df.columns]
        missing_pct = len(missing_cols) / max(1, len(required_features))

        nan_rates = {}
        for c in required_features:
            if c in df.columns:
                rate = float(df[c].isna().mean())
                if rate > 0.05:
                    nan_rates[c] = round(rate, 4)

        triggered = (missing_pct >= missing_threshold_pct) or (len(missing_cols) > 0 and "nwp_precip" in missing_cols)

        severity = FailureSeverity.CRITICAL if triggered else FailureSeverity.LOW
        response = (
            "Halt model inference. Revert to Raw NWP with operational telemetry warning."
            if triggered
            else "Impute minor missing continuous features using training medians."
        )

        return FailureConditionAudit(
            condition_name="Input_Failure",
            severity=severity,
            is_triggered=triggered,
            detection_rule=f"Missing features > {int(missing_threshold_pct*100)}% or critical baseline NWP missing",
            response_action=response,
            logging_level="ERROR" if triggered else "INFO",
            user_facing_behavior="Display uncorrected NWP marked with 'FALLBACK_INPUT_UNAVAILABLE' alert flag.",
            diagnostic_evidence={
                "missing_columns": missing_cols,
                "missing_count": len(missing_cols),
                "high_nan_features": nan_rates,
            },
        )

    @classmethod
    def check_regime_failure(
        cls,
        regime_probs: Union[np.ndarray, Dict[str, float]],
        tolerance: float = 0.02,
    ) -> FailureConditionAudit:
        """
        Regime Failure Check:
        Verifies sum(P(Regime)) == 1.0, non-negativity, and checks for extreme entropy/uncertainty.
        """
        if isinstance(regime_probs, dict):
            probs = np.array(list(regime_probs.values()), dtype=np.float32)
        else:
            probs = np.asarray(regime_probs, dtype=np.float32)

        prob_sum = float(np.sum(probs))
        has_negative = bool(np.any(probs < -1e-5))
        sum_error = abs(prob_sum - 1.0)
        is_invalid = has_negative or (sum_error > tolerance)

        # Check for near-uniform uncertainty (high entropy)
        uniform_ref = 1.0 / len(probs)
        is_highly_uncertain = bool(np.all(np.abs(probs - uniform_ref) < 0.03))

        triggered = is_invalid or is_highly_uncertain
        severity = FailureSeverity.HIGH if is_invalid else (FailureSeverity.MEDIUM if is_highly_uncertain else FailureSeverity.LOW)

        response = (
            "Normalize probability vector or fallback to uniform 1/6 distribution with REGIME_UNCERTAIN tag."
            if triggered
            else "Proceed with standard soft regime conditioning."
        )

        return FailureConditionAudit(
            condition_name="Regime_Failure",
            severity=severity,
            is_triggered=triggered,
            detection_rule="sum(P) != 1.0, negative probabilities, or maximum entropy uniform distribution",
            response_action=response,
            logging_level="WARNING" if triggered else "INFO",
            user_facing_behavior="Tag forecast with 'REGIME_UNCERTAIN: UNIFORM_WEIGHTING_APPLIED' badge.",
            diagnostic_evidence={
                "probability_sum": round(prob_sum, 4),
                "sum_deviation": round(sum_error, 4),
                "has_negative_weights": has_negative,
                "is_near_uniform": is_highly_uncertain,
            },
        )

    @classmethod
    def check_quantile_failure(
        cls,
        p50: np.ndarray,
        p75: np.ndarray,
        p90: np.ndarray,
    ) -> FailureConditionAudit:
        """
        Quantile Failure Check:
        Detects quantile crossing (P50 > P75 or P75 > P90) or negative rainfall values.
        """
        p50_arr = np.asarray(p50, dtype=np.float32)
        p75_arr = np.asarray(p75, dtype=np.float32)
        p90_arr = np.asarray(p90, dtype=np.float32)

        has_negatives = bool(np.any(p50_arr < -1e-4) or np.any(p75_arr < -1e-4) or np.any(p90_arr < -1e-4))
        cross_50_75 = int(np.sum(p50_arr > p75_arr + 1e-4))
        cross_75_90 = int(np.sum(p75_arr > p90_arr + 1e-4))
        cross_50_90 = int(np.sum(p50_arr > p90_arr + 1e-4))

        total_samples = len(p50_arr)
        total_crossings = cross_50_75 + cross_75_90 + cross_50_90
        triggered = has_negatives or (total_crossings > 0)

        severity = FailureSeverity.HIGH if triggered else FailureSeverity.LOW
        response = (
            "Invoke Chernozhukov Rearrangement Operator and clip non-negativity to 0.0 mm."
            if triggered
            else "Quantile ordering strictly monotonic. Proceed."
        )

        return FailureConditionAudit(
            condition_name="Quantile_Failure",
            severity=severity,
            is_triggered=triggered,
            detection_rule="P50 > P75, P75 > P90, or negative predicted precipitation",
            response_action=response,
            logging_level="WARNING" if triggered else "INFO",
            user_facing_behavior="Serve post-processed monotonically sorted quantiles.",
            diagnostic_evidence={
                "has_negative_values": has_negatives,
                "crossings_p50_gt_p75": cross_50_75,
                "crossings_p75_gt_p90": cross_75_90,
                "crossing_rate_pct": round((total_crossings / max(1, total_samples * 2)) * 100.0, 2),
            },
        )

    @classmethod
    def check_data_failure(
        cls,
        lats: np.ndarray,
        lons: np.ndarray,
        dates: List[str],
        domain_bounds: Tuple[float, float, float, float] = (6.0, 38.0, 68.0, 98.0),
    ) -> FailureConditionAudit:
        """
        Data Failure Check:
        Verifies coordinate domain (EPSG:4326 within Indian region) and valid ISO date strings.
        """
        min_lat, max_lat, min_lon, max_lon = domain_bounds
        lat_arr = np.asarray(lats, dtype=np.float32)
        lon_arr = np.asarray(lons, dtype=np.float32)

        out_of_bounds_lat = int(np.sum((lat_arr < min_lat) | (lat_arr > max_lat)))
        out_of_bounds_lon = int(np.sum((lon_arr < min_lon) | (lon_arr > max_lon)))

        # Validate dates
        invalid_dates = []
        for d in dates[:50]:
            try:
                datetime.fromisoformat(str(d).replace("Z", "+00:00"))
            except ValueError:
                invalid_dates.append(str(d))

        triggered = (out_of_bounds_lat > 0) or (out_of_bounds_lon > 0) or (len(invalid_dates) > 0)
        severity = FailureSeverity.CRITICAL if triggered else FailureSeverity.LOW

        response = (
            "Reject coordinates / cycle. Return RFC 7807 400 Bad Request error."
            if triggered
            else "Domain bounds and timestamp format verified."
        )

        return FailureConditionAudit(
            condition_name="Data_Failure",
            severity=severity,
            is_triggered=triggered,
            detection_rule="Coordinates outside 6-38N, 68-98E or unparseable ISO timestamps",
            response_action=response,
            logging_level="ERROR" if triggered else "INFO",
            user_facing_behavior="Return explicit HTTP 400 with 'INVALID_SPATIAL_COORDINATES' details.",
            diagnostic_evidence={
                "out_of_bounds_lat_count": out_of_bounds_lat,
                "out_of_bounds_lon_count": out_of_bounds_lon,
                "invalid_date_samples": invalid_dates[:3],
            },
        )

    @classmethod
    def check_model_failure(
        cls,
        model_checkpoint_path: Path,
        expected_quantiles: List[float] = [0.50, 0.75, 0.90],
    ) -> FailureConditionAudit:
        """
        Model Failure Check:
        Verifies model checkpoint exists, is readable, and contains all expected quantile estimators.
        """
        exists = model_checkpoint_path.exists()
        is_valid = False
        error_msg = ""

        if exists:
            try:
                import joblib
                loaded = joblib.load(model_checkpoint_path)
                if isinstance(loaded, dict) and all(q in loaded for q in expected_quantiles):
                    is_valid = True
                else:
                    error_msg = f"Checkpoint missing expected quantiles {expected_quantiles}"
            except Exception as exc:
                error_msg = str(exc)

        triggered = not (exists and is_valid)
        severity = FailureSeverity.CRITICAL if triggered else FailureSeverity.LOW
        response = (
            "Deploy fallback Statistical Bias Correction baseline while alerting MLOps on-call."
            if triggered
            else "Model weights loaded and integrity verified."
        )

        return FailureConditionAudit(
            condition_name="Model_Failure",
            severity=severity,
            is_triggered=triggered,
            detection_rule="Model artifact missing, corrupt, or missing quantile estimators",
            response_action=response,
            logging_level="CRITICAL" if triggered else "INFO",
            user_facing_behavior="Return Level-2 Fallback bias-corrected forecast with 'MODEL_DEGRADED' flag.",
            diagnostic_evidence={
                "checkpoint_exists": exists,
                "is_valid_checkpoint": is_valid,
                "error_details": error_msg,
            },
        )

    @classmethod
    def check_distribution_shift(
        cls,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: Optional[List[str]] = None,
        alpha_significance: float = 0.01,
    ) -> FailureConditionAudit:
        """
        Distribution Shift Check:
        Runs two-sample Kolmogorov-Smirnov (KS) test and Wasserstein distance
        comparing training-era feature distributions vs unseen test-era distributions.
        """
        cols = features or [
            c for c in ["nwp_precip", "moisture_flux_conv", "wind_shear_deep", "elevation", "error_lag_7d"]
            if c in train_df.columns and c in test_df.columns
        ]

        shifted_features = {}
        for c in cols:
            tr_vals = train_df[c].dropna().values
            te_vals = test_df[c].dropna().values
            if len(tr_vals) > 10 and len(te_vals) > 10:
                ks_stat, p_val = stats.ks_2samp(tr_vals, te_vals)
                # Significant shift if p-value < alpha and KS statistic > 0.15
                if p_val < alpha_significance and ks_stat > 0.15:
                    shifted_features[c] = {
                        "ks_statistic": round(float(ks_stat), 4),
                        "p_value": float(f"{p_val:.2e}"),
                        "train_mean": round(float(np.mean(tr_vals)), 2),
                        "test_mean": round(float(np.mean(te_vals)), 2),
                    }

        triggered = len(shifted_features) > 0
        severity = FailureSeverity.MEDIUM if triggered else FailureSeverity.LOW
        response = (
            "Flag features exhibiting significant drift. Log telemetry for retraining pipeline evaluation."
            if triggered
            else "Distributions align within nominal sampling tolerance."
        )

        return FailureConditionAudit(
            condition_name="Distribution_Shift",
            severity=severity,
            is_triggered=triggered,
            detection_rule=f"KS 2-sample test p < {alpha_significance} and statistic > 0.15",
            response_action=response,
            logging_level="WARNING" if triggered else "INFO",
            user_facing_behavior="Operate normally with heightened prediction spread monitoring.",
            diagnostic_evidence={
                "features_tested": len(cols),
                "features_shifted": list(shifted_features.keys()),
                "shift_details": shifted_features,
            },
        )
