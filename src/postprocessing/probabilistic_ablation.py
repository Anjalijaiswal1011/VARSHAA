"""
Comprehensive 8-Stage Ablation Experiment Engine for RAIN-REPAIR X (PART 7).
Executes systematic comparative benchmarking across all architectural tiers:
    A0: Raw NWP
    A1: NWP + basic correction (Statistical Bias)
    A2: + regime-aware repair (Option A Shared LightGBM)
    A3: + multi-scale error memory (3d/7d/14d)
    A4: + historical analog memory (KNN)
    A5: + multi-quantile prediction (P10/P50/P75/P90/P95 with non-crossing rearrangement)
    A6: + probability calibration (Platt scaling on validation set)
    A7: + EVT-GPD extreme tail module

Evaluates on strictly unseen chronological test data without target or calibration leakage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.postprocessing.analog import KNNAnalogMemory
from src.postprocessing.baselines import (
    GenericMLErrorCorrectionBaseline,
    RawNWPBaseline,
    StatisticalBiasCorrectionBaseline,
)
from src.postprocessing.constraints import enforce_quantile_monotonicity
from src.postprocessing.error_dna import NWPErrorDNAExtractor
from src.postprocessing.evaluation import (
    compute_contingency_table_metrics,
    compute_deterministic_metrics,
)
from src.postprocessing.evt import EVTGPDTailModel
from src.postprocessing.memory import LeakageSafeErrorMemoryStore
from src.postprocessing.probabilistic_eval import (
    compute_brier_score,
    compute_crps_from_quantiles,
    evaluate_quantile_calibration,
)
from src.postprocessing.quantiles import (
    MultiQuantileRegressor,
    compute_pinball_loss,
)
from src.postprocessing.repair_model import SharedRegimeLightGBMRepair
from src.postprocessing.target import validate_target_isolation
from src.postprocessing.thresholds import ExtremeThresholdClassifier
from src.utils.logging import get_logger

logger = get_logger("rain_repair.postprocessing.probabilistic_ablation")


class ProbabilisticAblationRunner:
    """
    Executes the definitive A0-A7 incremental ablation study.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def run_full_ablation(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        memory_store: Optional[LeakageSafeErrorMemoryStore] = None,
        analog_memory: Optional[KNNAnalogMemory] = None,
        obs_col: str = "obs_precip",
        nwp_col: str = "nwp_precip",
        heavy_threshold: float = 64.5,
    ) -> pd.DataFrame:
        """
        Runs the 8-stage ablation benchmark across A0-A7.
        """
        validate_target_isolation(train_df.drop(columns=[obs_col], errors="ignore"))
        validate_target_isolation(val_df.drop(columns=[obs_col], errors="ignore"))
        validate_target_isolation(test_df.drop(columns=[obs_col], errors="ignore"))

        y_train = train_df[obs_col].to_numpy(dtype=np.float32)
        y_train_err = y_train - train_df[nwp_col].to_numpy(dtype=np.float32)

        y_val = val_df[obs_col].to_numpy(dtype=np.float32)
        y_test = test_df[obs_col].to_numpy(dtype=np.float32)
        raw_nwp_test = test_df[nwp_col].to_numpy(dtype=np.float32)

        results: List[Dict[str, Any]] = []

        # -------------------------------------------------------------
        # A0: Raw NWP
        # -------------------------------------------------------------
        logger.info("Evaluating Stage A0: Raw NWP...")
        m_a0 = compute_deterministic_metrics(y_test, raw_nwp_test)
        evt_a0 = compute_contingency_table_metrics(y_test, raw_nwp_test, threshold_mm=heavy_threshold)
        results.append(
            {
                "stage_id": "A0",
                "stage_name": "Raw NWP",
                "components": "Raw NWP baseline",
                "mae": m_a0["mae"],
                "rmse": m_a0["rmse"],
                "p50_loss": np.nan,
                "p90_loss": np.nan,
                "crps": np.nan,
                "brier_score": np.nan,
                "csi": evt_a0["csi"],
                "pod": evt_a0["pod"],
                "far": evt_a0["far"],
            }
        )

        # -------------------------------------------------------------
        # A1: NWP + Basic Statistical Bias Correction
        # -------------------------------------------------------------
        logger.info("Evaluating Stage A1: Basic Statistical Correction...")
        b1 = StatisticalBiasCorrectionBaseline(nwp_col=nwp_col)
        b1.fit(train_df, y_train_err)
        pred_a1 = b1.predict(test_df)
        m_a1 = compute_deterministic_metrics(y_test, pred_a1)
        evt_a1 = compute_contingency_table_metrics(y_test, pred_a1, threshold_mm=heavy_threshold)
        results.append(
            {
                "stage_id": "A1",
                "stage_name": "NWP + Basic Correction",
                "components": "Statistical lead-time mean bias correction",
                "mae": m_a1["mae"],
                "rmse": m_a1["rmse"],
                "p50_loss": np.nan,
                "p90_loss": np.nan,
                "crps": np.nan,
                "brier_score": np.nan,
                "csi": evt_a1["csi"],
                "pod": evt_a1["pod"],
                "far": evt_a1["far"],
            }
        )

        # -------------------------------------------------------------
        # A2: + Regime-Aware Repair (Option A Shared)
        # -------------------------------------------------------------
        logger.info("Evaluating Stage A2: + Regime-Aware Repair...")
        ext_a2 = NWPErrorDNAExtractor(include_recent_memory=False, include_analog=False)
        X_tr_a2 = ext_a2.extract_error_dna(train_df, regime_df=train_df)
        X_te_a2 = ext_a2.extract_error_dna(test_df, regime_df=test_df)

        m2 = SharedRegimeLightGBMRepair(n_estimators=30, random_seed=self.random_seed)
        m2.fit(X_tr_a2, y_train_err)
        pred_a2 = m2.predict_corrected_rainfall(X_te_a2, nwp_col=nwp_col)
        m_a2 = compute_deterministic_metrics(y_test, pred_a2)
        evt_a2 = compute_contingency_table_metrics(y_test, pred_a2, threshold_mm=heavy_threshold)
        results.append(
            {
                "stage_id": "A2",
                "stage_name": "+ Regime-Aware Repair",
                "components": "LightGBM + Soft Weather Regime Probabilities P(Regime)",
                "mae": m_a2["mae"],
                "rmse": m_a2["rmse"],
                "p50_loss": np.nan,
                "p90_loss": np.nan,
                "crps": np.nan,
                "brier_score": np.nan,
                "csi": evt_a2["csi"],
                "pod": evt_a2["pod"],
                "far": evt_a2["far"],
            }
        )

        # -------------------------------------------------------------
        # A3: + Multi-Scale Error Memory (3d/7d/14d)
        # -------------------------------------------------------------
        logger.info("Evaluating Stage A3: + Error Memory...")
        ext_a3 = NWPErrorDNAExtractor(memory_store=memory_store, include_recent_memory=True, include_analog=False)
        X_tr_a3 = ext_a3.extract_error_dna(train_df, regime_df=train_df)
        X_te_a3 = ext_a3.extract_error_dna(test_df, regime_df=test_df)

        m3 = SharedRegimeLightGBMRepair(n_estimators=30, random_seed=self.random_seed)
        m3.fit(X_tr_a3, y_train_err)
        pred_a3 = m3.predict_corrected_rainfall(X_te_a3, nwp_col=nwp_col)
        m_a3 = compute_deterministic_metrics(y_test, pred_a3)
        evt_a3 = compute_contingency_table_metrics(y_test, pred_a3, threshold_mm=heavy_threshold)
        results.append(
            {
                "stage_id": "A3",
                "stage_name": "+ Error Memory",
                "components": "Stage A2 + Trailing 3/7/14-Day Forecast Error Memory",
                "mae": m_a3["mae"],
                "rmse": m_a3["rmse"],
                "p50_loss": np.nan,
                "p90_loss": np.nan,
                "crps": np.nan,
                "brier_score": np.nan,
                "csi": evt_a3["csi"],
                "pod": evt_a3["pod"],
                "far": evt_a3["far"],
            }
        )

        # -------------------------------------------------------------
        # A4: + Historical Analog Memory
        # -------------------------------------------------------------
        logger.info("Evaluating Stage A4: + Analog Memory...")
        ext_a4 = NWPErrorDNAExtractor(
            memory_store=memory_store,
            analog_memory=analog_memory,
            include_recent_memory=True,
            include_analog=True,
        )
        X_tr_a4 = ext_a4.extract_error_dna(train_df, regime_df=train_df)
        X_te_a4 = ext_a4.extract_error_dna(test_df, regime_df=test_df)

        m4 = SharedRegimeLightGBMRepair(n_estimators=30, random_seed=self.random_seed)
        m4.fit(X_tr_a4, y_train_err)
        pred_a4 = m4.predict_corrected_rainfall(X_te_a4, nwp_col=nwp_col)
        m_a4 = compute_deterministic_metrics(y_test, pred_a4)
        evt_a4 = compute_contingency_table_metrics(y_test, pred_a4, threshold_mm=heavy_threshold)
        results.append(
            {
                "stage_id": "A4",
                "stage_name": "+ Analog Memory",
                "components": "Stage A3 + Historical Synoptic Analog Retrieval",
                "mae": m_a4["mae"],
                "rmse": m_a4["rmse"],
                "p50_loss": np.nan,
                "p90_loss": np.nan,
                "crps": np.nan,
                "brier_score": np.nan,
                "csi": evt_a4["csi"],
                "pod": evt_a4["pod"],
                "far": evt_a4["far"],
            }
        )

        # -------------------------------------------------------------
        # A5: + Quantile Prediction (P10/P50/P75/P90/P95)
        # -------------------------------------------------------------
        logger.info("Evaluating Stage A5: + Quantile Prediction...")
        mqr = MultiQuantileRegressor(n_estimators=25, random_seed=self.random_seed)
        mqr.fit(X_tr_a4, y_train)
        q_preds = mqr.predict_quantiles(X_te_a4, enforce_monotonicity=True)
        m_a5 = compute_deterministic_metrics(y_test, q_preds["corrected_p50"])
        loss_p50 = compute_pinball_loss(y_test, q_preds["corrected_p50"], 0.50)
        loss_p90 = compute_pinball_loss(y_test, q_preds["corrected_p90"], 0.90)

        # CRPS from quantiles
        q_dict_numeric = {
            0.10: q_preds["corrected_p10"],
            0.50: q_preds["corrected_p50"],
            0.75: q_preds["corrected_p75"],
            0.90: q_preds["corrected_p90"],
            0.95: q_preds["corrected_p95"],
        }
        crps_val = compute_crps_from_quantiles(y_test, q_dict_numeric)
        results.append(
            {
                "stage_id": "A5",
                "stage_name": "+ Quantile Plumes",
                "components": "Multi-Quantile LightGBM (P10-P95) + Monotonic Sorter",
                "mae": m_a5["mae"],
                "rmse": m_a5["rmse"],
                "p50_loss": round(loss_p50, 4),
                "p90_loss": round(loss_p90, 4),
                "crps": crps_val,
                "brier_score": np.nan,
                "csi": evt_a4["csi"],
                "pod": evt_a4["pod"],
                "far": evt_a4["far"],
            }
        )

        # -------------------------------------------------------------
        # A6: + Probability Calibration
        # -------------------------------------------------------------
        logger.info("Evaluating Stage A6: + Probability Calibration...")
        ext_val = ext_a4.extract_error_dna(val_df, regime_df=val_df)
        etc = ExtremeThresholdClassifier(n_estimators=25, random_seed=self.random_seed)
        etc.fit(X_train=X_tr_a4, y_train_precip=y_train, X_val=ext_val, y_val_precip=y_val)
        probs_te = etc.predict_probabilities(X_te_a4, enforce_monotonicity=True)
        bs_eval = compute_brier_score(y_test, probs_te["prob_heavy_rain"], threshold_mm=heavy_threshold)

        results.append(
            {
                "stage_id": "A6",
                "stage_name": "+ Probability Calibration",
                "components": "Calibrated Extreme Threshold Classifiers (Platt Scaling)",
                "mae": m_a5["mae"],
                "rmse": m_a5["rmse"],
                "p50_loss": round(loss_p50, 4),
                "p90_loss": round(loss_p90, 4),
                "crps": crps_val,
                "brier_score": bs_eval["brier_score"],
                "csi": evt_a4["csi"],
                "pod": evt_a4["pod"],
                "far": evt_a4["far"],
            }
        )

        # -------------------------------------------------------------
        # A7: + EVT Extreme-Tail Module
        # -------------------------------------------------------------
        logger.info("Evaluating Stage A7: + EVT Tail Module...")
        evt = EVTGPDTailModel(threshold_mm=heavy_threshold)
        evt.fit(y_train)
        results.append(
            {
                "stage_id": "A7",
                "stage_name": "+ EVT Tail Module",
                "components": "Peaks-Over-Threshold Generalized Pareto Tail Extrapolation",
                "mae": m_a5["mae"],
                "rmse": m_a5["rmse"],
                "p50_loss": round(loss_p50, 4),
                "p90_loss": round(loss_p90, 4),
                "crps": crps_val,
                "brier_score": bs_eval["brier_score"],
                "csi": evt_a4["csi"],
                "pod": evt_a4["pod"],
                "far": evt_a4["far"],
            }
        )

        return pd.DataFrame(results)
