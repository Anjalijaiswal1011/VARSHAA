"""
Hierarchical Defensive Fallback Strategy Engine for RAIN-REPAIR X.
Implements the 5-tier graceful degradation hierarchy:
    Tier 0 (FULL):      Full RAIN-REPAIR X (Regimes + Transitions + Memory + Analog)
    Tier 1 (NO_ANALOG): Regime-Aware Model without Analog Memory
    Tier 2 (GENERIC_ML): Generic ML Error Correction (No regime features)
    Tier 3 (STAT_BIAS):  Statistical Lead-Time Bias Correction
    Tier 4 (RAW_NWP):    Uncorrected Raw NWP (Fail-safe pass-through)

Guarantees high operational availability while transparently logging and tagging fallback invocations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.postprocessing.baselines import (
    GenericMLErrorCorrectionBaseline,
    RawNWPBaseline,
    StatisticalBiasCorrectionBaseline,
)
from src.postprocessing.repair_model import (
    SharedRegimeLightGBMRepair,
    SoftMixtureOfExpertsRepair,
)
from src.postprocessing.target import apply_error_correction
from src.utils.logging import get_logger

logger = get_logger("rain_repair.postprocessing.fallback")


class FallbackLevel(str, Enum):
    FULL_RAIN_REPAIR_X = "FULL_RAIN_REPAIR_X"
    REGIME_AWARE_NO_ANALOG = "REGIME_AWARE_NO_ANALOG"
    GENERIC_ML_CORRECTION = "GENERIC_ML_CORRECTION"
    STATISTICAL_BIAS = "STATISTICAL_BIAS"
    RAW_NWP_PASSTHROUGH = "RAW_NWP_PASSTHROUGH"


@dataclass
class FallbackExecutionStatus:
    active_tier: FallbackLevel
    tier_number: int  # 0 to 4
    reason: str
    degraded: bool
    details: Dict[str, Any]


class FallbackRepairController:
    """
    Orchestrates post-processing execution with robust, transparent fallback degradation.
    """

    def __init__(
        self,
        primary_model: Optional[Union[SharedRegimeLightGBMRepair, SoftMixtureOfExpertsRepair]] = None,
        regime_no_analog_model: Optional[SharedRegimeLightGBMRepair] = None,
        generic_ml_model: Optional[GenericMLErrorCorrectionBaseline] = None,
        statistical_bias_model: Optional[StatisticalBiasCorrectionBaseline] = None,
        raw_nwp_baseline: Optional[RawNWPBaseline] = None,
    ):
        self.primary_model = primary_model
        self.regime_no_analog_model = regime_no_analog_model or primary_model
        self.generic_ml_model = generic_ml_model
        self.statistical_bias_model = statistical_bias_model
        self.raw_nwp_baseline = raw_nwp_baseline or RawNWPBaseline()

    def repair_forecast(
        self,
        X: pd.DataFrame,
        nwp_col: str = "nwp_precip",
        has_analog: bool = True,
        has_regime: bool = True,
        has_recent_memory: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, FallbackExecutionStatus]:
        """
        Execute forecast error repair with deterministic fallback routing.

        Returns:
            Tuple of:
            - predicted_error: np.ndarray (mm/24h)
            - corrected_rainfall: np.ndarray (mm/24h, non-negative)
            - status: FallbackExecutionStatus describing tier executed
        """
        raw_nwp = X[nwp_col].to_numpy(dtype=np.float32)

        # -------------------------------------------------------------
        # TIER 0: Full RAIN-REPAIR X
        # -------------------------------------------------------------
        if (
            self.primary_model is not None
            and self.primary_model.is_fitted
            and has_analog
            and has_regime
            and has_recent_memory
        ):
            try:
                pred_err = self.primary_model.predict_error(X)
                corrected = apply_error_correction(raw_nwp, pred_err, enforce_non_negativity=True)
                status = FallbackExecutionStatus(
                    active_tier=FallbackLevel.FULL_RAIN_REPAIR_X,
                    tier_number=0,
                    reason="Optimal execution: all regime, memory, and analog inputs verified.",
                    degraded=False,
                    details={"model_type": type(self.primary_model).__name__},
                )
                return pred_err, corrected, status
            except Exception as exc:
                logger.warning("Tier 0 failed during inference (%s). Falling back to Tier 1.", exc)

        # -------------------------------------------------------------
        # TIER 1: Regime-Aware Model without Analog Memory
        # -------------------------------------------------------------
        active_t1 = self.regime_no_analog_model or self.primary_model
        if active_t1 is not None and active_t1.is_fitted and has_regime:
            try:
                pred_err = active_t1.predict_error(X)
                corrected = apply_error_correction(raw_nwp, pred_err, enforce_non_negativity=True)
                status = FallbackExecutionStatus(
                    active_tier=FallbackLevel.REGIME_AWARE_NO_ANALOG,
                    tier_number=1,
                    reason="Analog memory unavailable or degraded; running regime-aware correction without analog.",
                    degraded=True,
                    details={"model_type": type(active_t1).__name__},
                )
                logger.info("Executing Fallback Tier 1: REGIME_AWARE_NO_ANALOG.")
                return pred_err, corrected, status
            except Exception as exc:
                logger.warning("Tier 1 failed during inference (%s). Falling back to Tier 2.", exc)

        # -------------------------------------------------------------
        # TIER 2: Generic ML Correction (No Regime Info)
        # -------------------------------------------------------------
        if self.generic_ml_model is not None and self.generic_ml_model.is_fitted:
            try:
                pred_err = self.generic_ml_model.predict_error(X)
                corrected = apply_error_correction(raw_nwp, pred_err, enforce_non_negativity=True)
                status = FallbackExecutionStatus(
                    active_tier=FallbackLevel.GENERIC_ML_CORRECTION,
                    tier_number=2,
                    reason="Regime probabilities unavailable or corrupted; running generic tabular ML correction.",
                    degraded=True,
                    details={"model_type": type(self.generic_ml_model).__name__},
                )
                logger.warning("Executing Fallback Tier 2: GENERIC_ML_CORRECTION.")
                return pred_err, corrected, status
            except Exception as exc:
                logger.warning("Tier 2 failed during inference (%s). Falling back to Tier 3.", exc)

        # -------------------------------------------------------------
        # TIER 3: Statistical Bias Correction
        # -------------------------------------------------------------
        if self.statistical_bias_model is not None and self.statistical_bias_model.is_fitted:
            try:
                pred_err = self.statistical_bias_model.predict_error(X)
                corrected = apply_error_correction(raw_nwp, pred_err, enforce_non_negativity=True)
                status = FallbackExecutionStatus(
                    active_tier=FallbackLevel.STATISTICAL_BIAS,
                    tier_number=3,
                    reason="ML models unavailable; executing statistical lead-time bias correction.",
                    degraded=True,
                    details={"model_type": type(self.statistical_bias_model).__name__},
                )
                logger.warning("Executing Fallback Tier 3: STATISTICAL_BIAS.")
                return pred_err, corrected, status
            except Exception as exc:
                logger.warning("Tier 3 failed during inference (%s). Falling back to Tier 4.", exc)

        # -------------------------------------------------------------
        # TIER 4: Raw NWP (Fail-Safe Pass-Through)
        # -------------------------------------------------------------
        pred_err = self.raw_nwp_baseline.predict_error(X)
        corrected = self.raw_nwp_baseline.predict(X, nwp_col=nwp_col)
        status = FallbackExecutionStatus(
            active_tier=FallbackLevel.RAW_NWP_PASSTHROUGH,
            tier_number=4,
            reason="All correction models unavailable or input severely corrupted; emitting uncorrected raw NWP.",
            degraded=True,
            details={"model_type": "RawNWPBaseline"},
        )
        logger.error("CRITICAL: Executing Fallback Tier 4: RAW_NWP_PASSTHROUGH.")
        return pred_err, corrected, status
