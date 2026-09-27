"""
Unified High-Level Post-Processing Engine for RAIN-REPAIR X.
Coordinates:
    Base NWP & Physics Features -> Regime Intelligence & Transitions ->
    3/7/14-Day Trailing Error Memory -> Historical Analog Retrieval ->
    NWP Error DNA Assembly -> Soft Regime Repair Model ->
    Physical Constraint Enforcement (Non-Negativity) -> Fallback Telemetry.

Emits structured post-processing records compliant with CONTRACT-ML-002.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.postprocessing.analog import AnalogMemory, KNNAnalogMemory
from src.postprocessing.baselines import (
    GenericMLErrorCorrectionBaseline,
    RawNWPBaseline,
    StatisticalBiasCorrectionBaseline,
)
from src.postprocessing.error_dna import NWPErrorDNAExtractor
from src.postprocessing.fallback import (
    FallbackExecutionStatus,
    FallbackLevel,
    FallbackRepairController,
)
from src.postprocessing.memory import LeakageSafeErrorMemoryStore
from src.postprocessing.repair_model import (
    SharedRegimeLightGBMRepair,
    SoftMixtureOfExpertsRepair,
)
from src.postprocessing.target import apply_error_correction, validate_target_isolation
from src.regime.inference import RegimeInferenceEngine
from src.regime.schemas import REGIME_CLASSES
from src.utils.logging import get_logger

logger = get_logger("rain_repair.postprocessing.engine")


class RainRepairEngine:
    """
    End-to-End Operational NWP Error Repair Engine.
    Executes context-aware, regime-guided post-processing with zero temporal leakage.
    """

    def __init__(
        self,
        repair_model: Optional[Union[SharedRegimeLightGBMRepair, SoftMixtureOfExpertsRepair]] = None,
        memory_store: Optional[LeakageSafeErrorMemoryStore] = None,
        analog_memory: Optional[AnalogMemory] = None,
        regime_engine: Optional[RegimeInferenceEngine] = None,
        fallback_controller: Optional[FallbackRepairController] = None,
    ):
        self.repair_model = repair_model
        self.memory_store = memory_store or LeakageSafeErrorMemoryStore()
        self.analog_memory = analog_memory
        self.regime_engine = regime_engine
        self.fallback_controller = fallback_controller or FallbackRepairController(
            primary_model=repair_model,
        )

        self.error_dna_extractor = NWPErrorDNAExtractor(
            memory_store=self.memory_store,
            analog_memory=self.analog_memory,
            include_regimes=True,
            include_transition=True,
            include_recent_memory=True,
            include_analog=self.analog_memory is not None,
        )

    def process_forecast_cycle(
        self,
        df_features: pd.DataFrame,
        cycle_date: str,
        regime_df: Optional[pd.DataFrame] = None,
        nwp_col: str = "nwp_precip",
    ) -> pd.DataFrame:
        """
        Process a forecast cycle and emit calibrated, corrected rainfall forecasts.

        Steps:
        1. Target isolation validation.
        2. Derive regime probabilities & transition signals if not pre-computed.
        3. Extract 3/7/14-day memory and analog memory signals.
        4. Assemble structured NWP Error DNA.
        5. Execute repair model with defensive fallback routing.
        6. Enforce non-negativity constraint.
        7. Format and return CONTRACT-ML-002 compatible schema.
        """
        validate_target_isolation(df_features)
        df_work = df_features.copy()

        # Step 2: Regime probabilities
        has_regime = False
        if regime_df is not None:
            active_regime_df = regime_df
            has_regime = True
        elif self.regime_engine is not None:
            try:
                active_regime_df = self.regime_engine.predict_to_dataframe(df_work, update_history=True)
                has_regime = True
            except Exception as exc:
                logger.warning("Regime inference engine failed: %s. Proceeding with fallback.", exc)
                active_regime_df = None
                has_regime = False
        else:
            active_regime_df = None
            has_regime = all(f"prob_{r.lower()}" in df_work.columns for r in REGIME_CLASSES)

        # Step 3 & 4: Extract NWP Error DNA
        has_analog = self.analog_memory is not None
        has_recent_memory = self.memory_store is not None
        error_dna = self.error_dna_extractor.extract_error_dna(
            features_df=df_work,
            regime_df=active_regime_df,
            query_date=cycle_date,
        )

        # Step 5 & 6: Execute repair through fallback controller
        predicted_error, corrected_rainfall, fallback_status = self.fallback_controller.repair_forecast(
            X=error_dna,
            nwp_col=nwp_col,
            has_analog=has_analog,
            has_regime=has_regime,
            has_recent_memory=has_recent_memory,
        )

        raw_nwp = df_work[nwp_col].to_numpy(dtype=np.float32)
        delta_correction = corrected_rainfall - raw_nwp

        # Step 7: Assemble output table conforming to CONTRACT-ML-002
        out_df = pd.DataFrame(index=df_work.index)
        out_df["cycle_date"] = cycle_date
        out_df["lead_time"] = df_work["lead_time"] if "lead_time" in df_work.columns else 24
        out_df["lat"] = df_work["lat"] if "lat" in df_work.columns else 0.0
        out_df["lon"] = df_work["lon"] if "lon" in df_work.columns else 0.0

        out_df["raw_nwp_precip"] = np.round(raw_nwp, 3)
        out_df["predicted_error"] = np.round(predicted_error, 3)
        out_df["delta_correction"] = np.round(delta_correction, 3)
        out_df["corrected_rainfall"] = np.round(corrected_rainfall, 3)
        # Aliased to corrected_p50 for CONTRACT-ML-002 compatibility
        out_df["corrected_p50"] = out_df["corrected_rainfall"]

        # Regime tags
        if active_regime_df is not None and "current_regime" in active_regime_df.columns:
            out_df["active_regime"] = active_regime_df["current_regime"].values
            out_df["transition_status"] = active_regime_df["transition_status"].values
        elif "current_regime" in df_work.columns:
            out_df["active_regime"] = df_work["current_regime"].values
            out_df["transition_status"] = df_work.get("transition_status", "STABLE")
        else:
            out_df["active_regime"] = "NORMAL_TRANSITIONAL"
            out_df["transition_status"] = "STABLE"

        # Attach soft probabilities
        for r_name in REGIME_CLASSES:
            prob_col = f"prob_{r_name.lower()}"
            if active_regime_df is not None and prob_col in active_regime_df.columns:
                out_df[prob_col] = active_regime_df[prob_col].values
            elif prob_col in error_dna.columns:
                out_df[prob_col] = error_dna[prob_col].values
            else:
                out_df[prob_col] = 1.0 / len(REGIME_CLASSES)

        # Telemetry & uncertainty indicators for Part 7 foundation
        out_df["fallback_tier"] = fallback_status.active_tier.value
        out_df["fallback_degraded"] = fallback_status.degraded
        out_df["recent_error_std_7d"] = error_dna.get("recent_error_std_7d", 0.0)
        out_df["analog_std_error"] = error_dna.get("analog_std_error", 0.0)

        logger.info(
            "Cycle %s repaired: %d points. Mean raw: %.2f mm, Mean corrected: %.2f mm. Fallback Tier: %s",
            cycle_date,
            len(out_df),
            float(np.mean(raw_nwp)),
            float(np.mean(corrected_rainfall)),
            fallback_status.active_tier.value,
        )
        return out_df

    def record_ground_truth(
        self,
        verification_date: str,
        verified_df: pd.DataFrame,
        obs_col: str = "obs_precip",
        nwp_col: str = "nwp_precip",
    ) -> None:
        """
        Feed ground truth observations into historical stores once verified.
        Ensures trailing error memory and analog archives stay up to date.
        """
        if self.memory_store is not None:
            self.memory_store.record_verified_cycle(
                verification_date=verification_date,
                df_cycle=verified_df,
                obs_col=obs_col,
                nwp_col=nwp_col,
            )
            logger.info("Recorded verified cycle %s into ErrorMemoryStore.", verification_date)
