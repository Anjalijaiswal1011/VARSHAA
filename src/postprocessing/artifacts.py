"""
Model Artifact Serialization & Versioning Engine for RAIN-REPAIR X Post-Processing.
Persists and loads trained repair models, feature column specifications, target definitions,
data split date ranges, hyperparameters, and experiment manifests without silent overwrites.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib

from src.postprocessing.repair_model import (
    SharedRegimeLightGBMRepair,
    SoftMixtureOfExpertsRepair,
)
from src.regime.schemas import REGIME_CLASSES
from src.utils.config import config
from src.utils.logging import get_logger

logger = get_logger("rain_repair.postprocessing.artifacts")


class ModelArtifactManager:
    """
    Manages versioned persistence and retrieval of post-processing repair models.
    """

    def __init__(self, base_checkpoint_dir: Optional[Path] = None):
        self.base_dir = base_checkpoint_dir or config.paths.models_checkpoint_dir / "nwp_repair"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Union[SharedRegimeLightGBMRepair, SoftMixtureOfExpertsRepair],
        model_version: str = "v1.0.0",
        feature_columns: Optional[List[str]] = None,
        train_period: Optional[Tuple[str, str]] = None,
        val_period: Optional[Tuple[str, str]] = None,
        test_period: Optional[Tuple[str, str]] = None,
        experiment_metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Save serialized model and rich JSON metadata manifest.
        Refuses silent overwrite unless overwrite=True.
        """
        version_dir = self.base_dir / model_version
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / f"repair_model_{model_version}.joblib"
        meta_path = version_dir / f"metadata_{model_version}.json"

        if model_path.exists() and not overwrite:
            raise FileExistsError(
                f"Model artifact {model_path} already exists. Specify a new version or set overwrite=True."
            )

        joblib.dump(model, model_path)

        metadata: Dict[str, Any] = {
            "model_version": model_version,
            "architecture": type(model).__name__,
            "target_definition": "NWP Error = Observed Rainfall - Raw NWP Rainfall",
            "physical_constraint": "Corrected Rainfall = max(0.0, Raw NWP + Predicted Error)",
            "regime_classes": REGIME_CLASSES,
            "feature_columns": feature_columns or getattr(model, "feature_cols", []),
            "hyperparameters": {
                "n_estimators": getattr(model, "n_estimators", None),
                "learning_rate": getattr(model, "learning_rate", None),
                "max_depth": getattr(model, "max_depth", None),
                "random_seed": getattr(model, "random_seed", None),
            },
            "training_period": train_period,
            "validation_period": val_period,
            "test_period": test_period,
            "experiment_metadata": experiment_metadata or {},
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Successfully saved versioned model artifact to %s", model_path)
        return model_path

    def load_model(
        self,
        model_version: str = "v1.0.0",
    ) -> Tuple[Union[SharedRegimeLightGBMRepair, SoftMixtureOfExpertsRepair], Dict[str, Any]]:
        """
        Load model artifact and associated metadata manifest.
        """
        version_dir = self.base_dir / model_version
        model_path = version_dir / f"repair_model_{model_version}.joblib"
        meta_path = version_dir / f"metadata_{model_version}.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")

        model = joblib.load(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        logger.info("Loaded model %s from %s", metadata.get("architecture"), model_path)
        return model, metadata
