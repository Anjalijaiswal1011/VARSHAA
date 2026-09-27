"""
Model Selection & Ablation Study Engine for RAIN-REPAIR X Post-Processing.
Executes systematic comparative ablation across 5 candidate architectures:
    Model A: Generic ML Correction (No regimes, no memory, no analog)
    Model B: Regime Features + ML (Option A with P(Regime) vectors)
    Model C: Soft Regime-Aware Experts (Option B Mixture of Experts)
    Model D: Soft Regime + Transition + Recent Memory (3/7/14d)
    Model E: Full RAIN-REPAIR X (Regimes + Transitions + Memory + Analog Retrieval)

Guarantees fair, uncorrupted evaluation on an identical chronologically held-out test split.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.postprocessing.analog import KNNAnalogMemory
from src.postprocessing.baselines import GenericMLErrorCorrectionBaseline, RawNWPBaseline
from src.postprocessing.error_dna import NWPErrorDNAExtractor
from src.postprocessing.evaluation import compute_deterministic_metrics
from src.postprocessing.memory import LeakageSafeErrorMemoryStore
from src.postprocessing.repair_model import (
    SharedRegimeLightGBMRepair,
    SoftMixtureOfExpertsRepair,
)
from src.postprocessing.target import compute_nwp_error_target, validate_target_isolation
from src.utils.logging import get_logger

logger = get_logger("rain_repair.postprocessing.ablation")


class ModelSelectionExperimentRunner:
    """
    Orchestrates the 5-model ablation benchmark on standardized train/val/test splits.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.raw_baseline = RawNWPBaseline()

    def run_ablation(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        memory_store: Optional[LeakageSafeErrorMemoryStore] = None,
        analog_memory: Optional[KNNAnalogMemory] = None,
        obs_col: str = "obs_precip",
        nwp_col: str = "nwp_precip",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run the 5-model ablation experiment.

        Returns:
            - comparison_df: pd.DataFrame summarizing MAE, RMSE, and improvements
            - models_dict: Dict of fitted model instances
        """
        validate_target_isolation(train_df.drop(columns=[obs_col], errors="ignore"))
        validate_target_isolation(test_df.drop(columns=[obs_col], errors="ignore"))

        y_train_target = (train_df[obs_col] - train_df[nwp_col]).to_numpy(dtype=np.float32)
        y_test_obs = test_df[obs_col].to_numpy(dtype=np.float32)
        y_test_raw_nwp = test_df[nwp_col].to_numpy(dtype=np.float32)

        # Baseline: Raw NWP
        raw_metrics = compute_deterministic_metrics(y_test_obs, y_test_raw_nwp)
        logger.info("Raw NWP Test Baseline: MAE=%.4f mm, RMSE=%.4f mm", raw_metrics["mae"], raw_metrics["rmse"])

        fitted_models: Dict[str, Any] = {}
        results: List[Dict[str, Any]] = [
            {
                "model_id": "Baseline_Raw_NWP",
                "model_name": "Raw NWP Forecast",
                "feature_group": "None (Identity)",
                "mae": raw_metrics["mae"],
                "rmse": raw_metrics["rmse"],
                "mae_reduction": 0.0,
                "rmse_reduction": 0.0,
            }
        ]

        # -------------------------------------------------------------
        # MODEL A: Generic ML Correction
        # -------------------------------------------------------------
        logger.info("Fitting Model A: Generic ML Correction...")
        extractor_a = NWPErrorDNAExtractor(
            include_regimes=False,
            include_transition=False,
            include_recent_memory=False,
            include_analog=False,
        )
        X_tr_a = extractor_a.extract_error_dna(train_df)
        X_te_a = extractor_a.extract_error_dna(test_df)

        model_a = GenericMLErrorCorrectionBaseline(random_seed=self.random_seed)
        model_a.fit(X_tr_a, y_train_target)
        pred_a = model_a.predict(X_te_a, nwp_col=nwp_col)
        m_a = compute_deterministic_metrics(y_test_obs, pred_a)
        fitted_models["Model_A"] = model_a
        results.append(
            {
                "model_id": "Model_A",
                "model_name": "Generic ML Correction",
                "feature_group": "NWP + Atmospheric + Terrain + Lead Time",
                "mae": m_a["mae"],
                "rmse": m_a["rmse"],
                "mae_reduction": round(raw_metrics["mae"] - m_a["mae"], 4),
                "rmse_reduction": round(raw_metrics["rmse"] - m_a["rmse"], 4),
            }
        )

        # -------------------------------------------------------------
        # MODEL B: Regime Features + ML (Option A Shared Model)
        # -------------------------------------------------------------
        logger.info("Fitting Model B: Regime Features + ML...")
        extractor_b = NWPErrorDNAExtractor(
            include_regimes=True,
            include_transition=False,
            include_recent_memory=False,
            include_analog=False,
        )
        X_tr_b = extractor_b.extract_error_dna(train_df, regime_df=train_df)
        X_te_b = extractor_b.extract_error_dna(test_df, regime_df=test_df)

        model_b = SharedRegimeLightGBMRepair(random_seed=self.random_seed)
        model_b.fit(X_tr_b, y_train_target)
        pred_b = model_b.predict_corrected_rainfall(X_te_b, nwp_col=nwp_col)
        m_b = compute_deterministic_metrics(y_test_obs, pred_b)
        fitted_models["Model_B"] = model_b
        results.append(
            {
                "model_id": "Model_B",
                "model_name": "Regime Features + ML",
                "feature_group": "Model A + Soft Regime Probabilities P(Regime)",
                "mae": m_b["mae"],
                "rmse": m_b["rmse"],
                "mae_reduction": round(raw_metrics["mae"] - m_b["mae"], 4),
                "rmse_reduction": round(raw_metrics["rmse"] - m_b["rmse"], 4),
            }
        )

        # -------------------------------------------------------------
        # MODEL C: Soft Regime-Aware Experts (Option B MoE)
        # -------------------------------------------------------------
        logger.info("Fitting Model C: Soft Regime-Aware Experts (MoE)...")
        model_c = SoftMixtureOfExpertsRepair(
            min_samples_per_regime=15,
            random_seed=self.random_seed,
        )
        model_c.fit(X_tr_b, y_train_target)
        pred_c = model_c.predict_corrected_rainfall(X_te_b, nwp_col=nwp_col)
        m_c = compute_deterministic_metrics(y_test_obs, pred_c)
        fitted_models["Model_C"] = model_c
        results.append(
            {
                "model_id": "Model_C",
                "model_name": "Soft Regime Experts (MoE)",
                "feature_group": "Model B + 6 Regime Dedicated Expert Regressors",
                "mae": m_c["mae"],
                "rmse": m_c["rmse"],
                "mae_reduction": round(raw_metrics["mae"] - m_c["mae"], 4),
                "rmse_reduction": round(raw_metrics["rmse"] - m_c["rmse"], 4),
            }
        )

        # -------------------------------------------------------------
        # MODEL D: Soft Regime + Transition + Recent Memory
        # -------------------------------------------------------------
        logger.info("Fitting Model D: Soft Regime + Transition + Recent Memory...")
        extractor_d = NWPErrorDNAExtractor(
            memory_store=memory_store,
            include_regimes=True,
            include_transition=True,
            include_recent_memory=True,
            include_analog=False,
        )
        X_tr_d = extractor_d.extract_error_dna(train_df, regime_df=train_df)
        X_te_d = extractor_d.extract_error_dna(test_df, regime_df=test_df)

        model_d = SharedRegimeLightGBMRepair(random_seed=self.random_seed)
        model_d.fit(X_tr_d, y_train_target)
        pred_d = model_d.predict_corrected_rainfall(X_te_d, nwp_col=nwp_col)
        m_d = compute_deterministic_metrics(y_test_obs, pred_d)
        fitted_models["Model_D"] = model_d
        results.append(
            {
                "model_id": "Model_D",
                "model_name": "Regime + Transition + Memory",
                "feature_group": "Model B + Transitions + 3/7/14d Trailing Error",
                "mae": m_d["mae"],
                "rmse": m_d["rmse"],
                "mae_reduction": round(raw_metrics["mae"] - m_d["mae"], 4),
                "rmse_reduction": round(raw_metrics["rmse"] - m_d["rmse"], 4),
            }
        )

        # -------------------------------------------------------------
        # MODEL E: Full RAIN-REPAIR X (Full Error DNA with Analog)
        # -------------------------------------------------------------
        logger.info("Fitting Model E: Full RAIN-REPAIR X with Analog Memory...")
        extractor_e = NWPErrorDNAExtractor(
            memory_store=memory_store,
            analog_memory=analog_memory,
            include_regimes=True,
            include_transition=True,
            include_recent_memory=True,
            include_analog=True,
        )
        X_tr_e = extractor_e.extract_error_dna(train_df, regime_df=train_df)
        X_te_e = extractor_e.extract_error_dna(test_df, regime_df=test_df)

        model_e = SharedRegimeLightGBMRepair(random_seed=self.random_seed)
        model_e.fit(X_tr_e, y_train_target)
        pred_e = model_e.predict_corrected_rainfall(X_te_e, nwp_col=nwp_col)
        m_e = compute_deterministic_metrics(y_test_obs, pred_e)
        fitted_models["Model_E"] = model_e
        results.append(
            {
                "model_id": "Model_E",
                "model_name": "Full RAIN-REPAIR X",
                "feature_group": "Model D + Historical Analog Memory Signals",
                "mae": m_e["mae"],
                "rmse": m_e["rmse"],
                "mae_reduction": round(raw_metrics["mae"] - m_e["mae"], 4),
                "rmse_reduction": round(raw_metrics["rmse"] - m_e["rmse"], 4),
            }
        )

        comparison_df = pd.DataFrame(results)
        return comparison_df, fitted_models
