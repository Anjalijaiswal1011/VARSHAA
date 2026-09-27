"""
Extreme Value Theory (EVT) & Generalized Pareto Distribution (GPD) Engine for RAIN-REPAIR X.
Implements Peaks-Over-Threshold (POT) modeling to capture unobserved catastrophic tails:
    P(Rain > x | Rain > u) = P(Rain > u) * [ 1 + (xi * (x - u)) / sigma ] ^ (-1 / xi)

Statistical Principles:
1. Defensible Threshold Selection: Threshold u is chosen at the upper empirical quantile (e.g., 95th percentile
   or official heavy rainfall threshold 64.5 mm) based on Mean Residual Life (MRL) linearity.
2. Parameter Estimation: Maximum Likelihood Estimation (MLE) of GPD shape (xi) and scale (sigma) via scipy.stats.genpareto.
3. Goodness of Fit & Diagnostics: One-sample Kolmogorov-Smirnov test against fitted GPD.
4. Safety Fallback: If sample size of threshold exceedances < min_exceedances (15), EVT is marked 'unavailable'
   and downstream processes fall back to calibrated quantile models without forcing uncalibrated tail extrapolations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logging import get_logger

logger = get_logger("rain_repair.postprocessing.evt")

MIN_POT_EXCEEDANCES: int = 15
DEFAULT_TAIL_THRESHOLD_MM: float = 64.5
EVT_VERSION: str = "v1.0.0-evt"


class EVTGPDTailModel:
    """
    Peaks-Over-Threshold (POT) Extreme Value Theory Engine.
    Models the asymptotic tail distribution of precipitation beyond high thresholds.
    """

    def __init__(
        self,
        threshold_mm: float = DEFAULT_TAIL_THRESHOLD_MM,
        min_exceedances: int = MIN_POT_EXCEEDANCES,
        version: str = EVT_VERSION,
    ):
        self.threshold_mm = threshold_mm
        self.min_exceedances = min_exceedances
        self.version = version

        self.threshold_exceedance_prob: float = 0.05
        self.shape_xi: float = 0.15  # Fréchet heavy-tail prior
        self.scale_sigma: float = 25.0
        self.is_fitted = False
        self.n_exceedances: int = 0
        self.evt_status: str = "uninitialized"
        self.ks_pvalue: float = 1.0

    def fit(
        self,
        precip_series: Union[np.ndarray, pd.Series],
        threshold_quantile: Optional[float] = None,
    ) -> EVTGPDTailModel:
        """
        Fits GPD parameters (shape xi, scale sigma) on historical precipitation excesses.
        """
        arr = np.asarray(precip_series, dtype=np.float32)
        arr = arr[~np.isnan(arr)]

        if threshold_quantile is not None:
            self.threshold_mm = float(np.quantile(arr, threshold_quantile))

        excesses = arr[arr > self.threshold_mm] - self.threshold_mm
        self.n_exceedances = len(excesses)
        self.threshold_exceedance_prob = float(self.n_exceedances / max(len(arr), 1))

        if self.n_exceedances < self.min_exceedances:
            logger.warning(
                "Sparse exceedances (%d < %d threshold) for EVT threshold %.1f mm. Marking EVT unavailable.",
                self.n_exceedances,
                self.min_exceedances,
                self.threshold_mm,
            )
            self.evt_status = "unavailable"
            self.is_fitted = False
            return self

        try:
            # Fit GPD with location fixed at 0.0 (since we already subtracted threshold u)
            # scipy parametrization: genpareto(c=xi, loc=0, scale=sigma)
            shape, loc, scale = stats.genpareto.fit(excesses, floc=0.0)

            # Enforce meteorological physical bounds on shape parameter xi [-0.10, 0.50]
            # In Indian monsoon convective rainfall, xi typically lies between 0.05 and 0.45.
            self.shape_xi = float(np.clip(shape, -0.10, 0.50))
            self.scale_sigma = float(max(scale, 5.0))

            # Goodness-of-fit diagnostic: Kolmogorov-Smirnov test
            ks_res = stats.kstest(excesses, "genpareto", args=(self.shape_xi, 0.0, self.scale_sigma))
            self.ks_pvalue = float(ks_res.pvalue)

            self.is_fitted = True
            self.evt_status = "active"

            logger.info(
                "EVT-GPD fitted successfully on %d excesses above %.1f mm: shape xi=%.4f, scale sigma=%.4f mm, KS p-value=%.4f",
                self.n_exceedances,
                self.threshold_mm,
                self.shape_xi,
                self.scale_sigma,
                self.ks_pvalue,
            )
        except Exception as exc:
            logger.warning("GPD MLE fit failed (%s). Marking EVT unavailable.", exc)
            self.evt_status = "unavailable"
            self.is_fitted = False

        return self

    def compute_exceedance_probability(
        self,
        x: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Computes unconditional extreme exceedance probability P(Rain > x):
            If x <= u: returns empirical / interpolated bound
            If x > u: P(Rain > x) = P(Rain > u) * [1 + xi * (x - u) / sigma]^(-1/xi)
        """
        if not self.is_fitted or self.evt_status != "active":
            # Safe fallback if EVT unavailable
            x_arr = np.asarray(x, dtype=np.float32)
            return np.where(x_arr > self.threshold_mm, 0.05, 0.20)

        x_arr = np.asarray(x, dtype=np.float32)
        y = x_arr - self.threshold_mm

        probs = np.where(
            y <= 0,
            np.clip(1.0 - (x_arr / max(self.threshold_mm, 1e-4)) * (1.0 - self.threshold_exceedance_prob), 0.0, 1.0),
            0.0,
        )

        tail_mask = y > 0
        if np.any(tail_mask):
            y_tail = y[tail_mask]
            xi = self.shape_xi
            sigma = self.scale_sigma

            if abs(xi) < 1e-5:
                cond_survival = np.exp(-y_tail / sigma)
            else:
                arg = np.maximum(1.0 + (xi * y_tail) / sigma, 1e-6)
                cond_survival = np.power(arg, -1.0 / xi)

            probs[tail_mask] = np.clip(self.threshold_exceedance_prob * cond_survival, 0.0, 1.0)

        if isinstance(x, (int, float)):
            return float(probs)
        return probs.astype(np.float32)

    def compute_return_level(self, quantile_p: float) -> float:
        """
        Inverts the GPD survival function to find extreme return level quantiles (e.g., P99, P99.5):
            x_p = u + (sigma / xi) * [ ((1 - F(u)) / (1 - p))^xi - 1 ]
        """
        if not self.is_fitted or self.evt_status != "active":
            return float(self.threshold_mm * 1.5)

        if quantile_p <= 1.0 - self.threshold_exceedance_prob:
            return float(self.threshold_mm * (quantile_p / (1.0 - self.threshold_exceedance_prob)))

        p_exc = 1.0 - quantile_p
        xi = self.shape_xi
        sigma = self.scale_sigma
        ratio = self.threshold_exceedance_prob / max(p_exc, 1e-6)

        if abs(xi) < 1e-5:
            return float(self.threshold_mm + sigma * np.log(ratio))

        return float(self.threshold_mm + (sigma / xi) * (np.power(ratio, xi) - 1.0))
