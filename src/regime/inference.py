"""
Unified High-Level Inference Engine for Weather Regime Intelligence.
Connects the Supervised Regime Classifier, Probability Calibrator,
and Regime Transition Engine with defensive fallback mechanisms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.regime.calibration import RegimeCalibrator
from src.regime.classifier import LightGBMRegimeClassifier
from src.regime.schemas import (
    REGIME_CLASSES,
    RegimeProbabilityVector,
    RegimeTransitionOutput,
    TransitionStatus,
    WeatherRegime,
)
from src.regime.transition import RegimeHistoryBuffer, RegimeTransitionEngine
from src.utils.logging import get_logger

logger = get_logger("rain_repair.regime.inference")


class RegimeInferenceEngine:
    """
    End-to-End Regime Intelligence Engine for real-time and batch forecast cycles.
    Produces soft probability vectors and transition dynamics with zero temporal leakage.
    """

    def __init__(
        self,
        classifier: Optional[LightGBMRegimeClassifier] = None,
        calibrator: Optional[RegimeCalibrator] = None,
        transition_engine: Optional[RegimeTransitionEngine] = None,
    ):
        self.classifier = classifier
        self.calibrator = calibrator
        self.transition_engine = transition_engine or RegimeTransitionEngine()

    @classmethod
    def from_saved_model(
        cls,
        model_dir: Path,
        model_version: str = "v1.0.0",
        calibrator: Optional[RegimeCalibrator] = None,
    ) -> RegimeInferenceEngine:
        """Load inference engine from saved model artifacts."""
        classifier = LightGBMRegimeClassifier.load_artifact(model_dir, model_version)
        return cls(classifier=classifier, calibrator=calibrator)

    def _fallback_prediction(
        self,
        row: pd.Series,
        reason: str,
    ) -> RegimeTransitionOutput:
        """
        Safe fallback when model fails or features are missing.
        Assigns REGIME_UNCERTAIN and a uniform 1/6 probability distribution.
        Never silently assigns an arbitrary class like ACTIVE_MONSOON.
        """
        uniform_val = round(1.0 / len(REGIME_CLASSES), 4)
        uniform_probs = RegimeProbabilityVector(
            active_monsoon=uniform_val,
            break_monsoon=uniform_val,
            monsoon_depression=uniform_val,
            coastal=uniform_val,
            orographic=uniform_val,
            western_disturbance=uniform_val,
        )

        grid_id = str(row.get("grid_id", "grid_unknown"))
        timestamp = str(row.get("forecast_time", row.get("timestamp", "unknown_time")))

        logger.warning(
            "Fallback invoked for grid %s at %s. Reason: %s",
            grid_id,
            timestamp,
            reason,
        )

        return RegimeTransitionOutput(
            timestamp=timestamp,
            grid_id=grid_id,
            current_regime=WeatherRegime.REGIME_UNCERTAIN.value,
            current_regime_probabilities=uniform_probs.to_dict(),
            previous_regime=None,
            transition_status=TransitionStatus.UNCERTAIN,
            transition_from=None,
            transition_to=None,
            transition_strength=0.0,
            transition_confidence=uniform_val,
        )

    def predict(
        self,
        df: pd.DataFrame,
        update_history: bool = True,
    ) -> List[RegimeTransitionOutput]:
        """
        Execute full regime inference pipeline over a DataFrame of weather contexts.
        Returns typed transition outputs for each grid point.
        """
        if df.empty:
            return []

        # Check model availability
        if self.classifier is None or not self.classifier.is_fitted:
            logger.error("Classifier is unavailable or un-fitted. Triggering safe fallback.")
            return [self._fallback_prediction(row, "Model uninitialized or not fitted") for _, row in df.iterrows()]

        # Validate required feature columns
        required_cols = self.classifier.feature_cols
        if required_cols:
            missing_cols = [c for c in required_cols if c not in df.columns]
            if len(missing_cols) > len(required_cols) * 0.4:
                # If more than 40% of features are completely missing, reject and fallback
                logger.error("Critical features missing (%d/%d): %s. Fallback.", len(missing_cols), len(required_cols), missing_cols[:5])
                return [self._fallback_prediction(row, f"Missing critical features: {missing_cols[:3]}") for _, row in df.iterrows()]

        try:
            # Step 1: Raw probability prediction
            raw_probs = self.classifier.predict_proba(df)

            # Step 2: Calibration (if configured)
            if self.calibrator is not None and self.calibrator.is_calibrated:
                probs = self.calibrator.predict_proba(df, self.classifier.feature_cols)
            else:
                probs = raw_probs

            # Step 3: Typed probability vectors
            prob_vectors = [RegimeProbabilityVector.from_array(probs[i]) for i in range(len(df))]

            # Step 4: Transition engine evaluation
            outputs = self.transition_engine.evaluate_batch(
                df=df,
                prob_vectors=prob_vectors,
            )
            return outputs

        except Exception as exc:
            logger.error("Unexpected error during regime inference: %s. Reverting to safe fallback.", exc)
            return [self._fallback_prediction(row, f"Inference exception: {exc}") for _, row in df.iterrows()]

    def predict_to_dataframe(
        self,
        df: pd.DataFrame,
        update_history: bool = True,
    ) -> pd.DataFrame:
        """
        Inference output formatted as a DataFrame ready for direct ingestion
        by downstream modules (NWP Error DNA and Soft Regime-Aware Repair).
        """
        outputs = self.predict(df, update_history=update_history)
        records = []
        for out in outputs:
            row_dict = {
                "timestamp": out.timestamp,
                "grid_id": out.grid_id,
                "current_regime": out.current_regime,
                "previous_regime": out.previous_regime,
                "transition_status": out.transition_status.value,
                "transition_from": out.transition_from,
                "transition_to": out.transition_to,
                "transition_strength": out.transition_strength,
                "transition_confidence": out.transition_confidence,
            }
            # Unpack the 6 regime probabilities
            for reg, p in out.current_regime_probabilities.items():
                row_dict[f"prob_{reg.lower()}"] = p

            records.append(row_dict)

        return pd.DataFrame(records)
