"""
Historical Analog Memory Engine for RAIN-REPAIR X.
Retrieves historically similar synoptic weather/forecast states from a verified archive
and extracts the analog NWP error signal to guide the repair model.

Architectural Abstraction:
    AnalogMemory (ABC) defines the interface, allowing seamless switching
    between exact KD-Tree / BallTree / KNN and approximate FAISS vector indexes.

Strict Anti-Leakage Invariants:
1. Similarity vectors consist strictly of pre-forecast predictors:
   atmospheric dynamics, raw NWP rainfall, lead time, terrain, and soft regime probabilities.
2. Target observations (observed rain / verified error) are stored ONLY in the outcome
   payload and never influence similarity search.
3. Temporal causality: For any query at forecast cycle T, the historical candidate pool
   is strictly constrained to dates verified prior to T (t_hist < T).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.utils.exceptions import TemporalLeakageError
from src.utils.logging import get_logger

logger = get_logger("rain_repair.postprocessing.analog")


class AnalogMemory(ABC):
    """
    Abstract Base Class for Historical Weather Analog Retrieval.
    Enables pluggable retrieval backends (KNN, BallTree, KDTree, FAISS).
    """

    @abstractmethod
    def fit(
        self,
        features: pd.DataFrame,
        errors: Union[np.ndarray, pd.Series],
        dates: Union[List[str], np.ndarray, pd.Series],
        feature_cols: Optional[List[str]] = None,
    ) -> AnalogMemory:
        """Index verified historical forecast situations with their observed NWP errors."""
        pass

    @abstractmethod
    def query(
        self,
        query_features: pd.DataFrame,
        query_dates: Optional[Union[List[str], np.ndarray, pd.Series]] = None,
        k: int = 5,
    ) -> pd.DataFrame:
        """
        Query top-k most similar historical situations and return analog error signals:
            analog_mean_error, analog_std_error, analog_min_distance
        """
        pass


class KNNAnalogMemory(AnalogMemory):
    """
    Scalable Nearest-Neighbors Analog Memory for Weather Forecast Error Retrieval.
    Uses standard feature normalization and Euclidean/Cosine space matching.

    Dataset Scale Rationale:
    For regional monsoon forecasts across sub-regions or cluster centroids (< 100,000 cases),
    BallTree / KDTree in Scikit-Learn executes in < 2ms without heavy external C++ binaries,
    making it ideal for robust, cross-platform deployment while preserving zero leakage.
    """

    DEFAULT_ANALOG_COLS: List[str] = [
        "nwp_precip",
        "u850",
        "v850",
        "rh850",
        "mslp",
        "t2m",
        "dem_slope",
        "elevation",
        "orographic_uplift",
        "vorticity_850",
        "lead_time",
        "prob_active_monsoon",
        "prob_break_monsoon",
        "prob_monsoon_depression",
        "prob_coastal",
        "prob_orographic",
        "prob_western_disturbance",
    ]

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        metric: str = "euclidean",
        algorithm: str = "auto",
        k: int = 5,
    ):
        self.feature_cols = feature_cols or self.DEFAULT_ANALOG_COLS
        self.metric = metric
        self.algorithm = algorithm
        self.k = k

        self.scaler = StandardScaler()
        self.nn_model: Optional[NearestNeighbors] = None
        self._stored_errors: np.ndarray = np.array([])
        self._stored_dates: np.ndarray = np.array([])
        self._is_indexed = False

    def fit(
        self,
        features: pd.DataFrame,
        errors: Union[np.ndarray, pd.Series],
        dates: Union[List[str], np.ndarray, pd.Series],
        feature_cols: Optional[List[str]] = None,
    ) -> KNNAnalogMemory:
        """
        Index historical weather situations.
        All inputs must represent verified historical cycles.
        """
        if feature_cols is not None:
            self.feature_cols = feature_cols

        available_cols = [c for c in self.feature_cols if c in features.columns]
        if not available_cols:
            raise ValueError(f"None of the required analog features {self.feature_cols} found in dataframe.")
        self.feature_cols = available_cols

        X_raw = features[self.feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        self.scaler.fit(X_raw)
        X_scaled = self.scaler.transform(X_raw)

        self._stored_errors = np.asarray(errors, dtype=np.float32).ravel()
        self._stored_dates = np.asarray([str(d) for d in dates])

        if len(X_scaled) != len(self._stored_errors):
            raise ValueError(f"Features length ({len(X_scaled)}) must match errors length ({len(self._stored_errors)})")

        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.k, len(X_scaled)),
            metric=self.metric,
            algorithm=self.algorithm,
        )
        self.nn_model.fit(X_scaled)
        self._is_indexed = True

        logger.info(
            "KNNAnalogMemory indexed %d historical cases across %d contextual features.",
            len(X_scaled),
            len(self.feature_cols),
        )
        return self

    def query(
        self,
        query_features: pd.DataFrame,
        query_dates: Optional[Union[List[str], np.ndarray, pd.Series]] = None,
        k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Query analog historical cases for incoming weather contexts.
        Enforces causal anti-leakage: if query_dates are provided,
        verified candidates occurring at or after the query cycle are masked out.
        """
        if not self._is_indexed or self.nn_model is None:
            raise RuntimeError("KNNAnalogMemory must be fitted before querying.")

        k_neighbors = k or self.k
        k_neighbors = min(k_neighbors, len(self._stored_errors))

        X_query_raw = query_features[self.feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        X_query_scaled = self.scaler.transform(X_query_raw)

        # Retrieve a wider window of neighbors to allow post-filtering for temporal causality
        search_k = min(len(self._stored_errors), max(k_neighbors * 3, 15))
        distances, indices = self.nn_model.kneighbors(X_query_scaled, n_neighbors=search_k)

        n_queries = len(query_features)
        mean_errors = np.zeros(n_queries, dtype=np.float32)
        std_errors = np.zeros(n_queries, dtype=np.float32)
        min_dists = np.zeros(n_queries, dtype=np.float32)

        q_dates_list = (
            [str(d) for d in query_dates]
            if query_dates is not None
            else ["2099-12-31"] * n_queries
        )

        for i in range(n_queries):
            q_date_dt = pd.to_datetime(q_dates_list[i])
            row_indices = indices[i]
            row_dists = distances[i]

            # Filter candidates: strictly candidate_date < query_date
            valid_mask = [
                pd.to_datetime(self._stored_dates[cand_idx]) < q_date_dt
                for cand_idx in row_indices
            ]

            valid_cand_indices = row_indices[valid_mask]
            valid_cand_dists = row_dists[valid_mask]

            if len(valid_cand_indices) == 0:
                # Cold start: neutral signal
                mean_errors[i] = 0.0
                std_errors[i] = 0.0
                min_dists[i] = 999.0
                continue

            # Take top-k causal neighbors
            top_k_indices = valid_cand_indices[:k_neighbors]
            top_k_dists = valid_cand_dists[:k_neighbors]
            cand_errors = self._stored_errors[top_k_indices]

            # Inverse-distance weighting (plus epsilon to avoid division by zero)
            weights = 1.0 / (top_k_dists + 1e-4)
            norm_weights = weights / np.sum(weights)

            mean_errors[i] = float(np.sum(cand_errors * norm_weights))
            std_errors[i] = float(np.std(cand_errors)) if len(cand_errors) > 1 else 0.0
            min_dists[i] = float(top_k_dists[0])

        return pd.DataFrame(
            {
                "analog_mean_error": mean_errors,
                "analog_std_error": std_errors,
                "analog_min_distance": min_dists,
            },
            index=query_features.index,
        )
