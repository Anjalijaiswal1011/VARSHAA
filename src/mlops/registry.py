"""
Model Registry & Safe Rollback Management Engine for RAIN-REPAIR X (PART 10).
Maintains versioned model metadata, lifecycle states, promotion rules, and instant rollback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.mlops.quality_gates import ModelQualityGate, QualityGateResult
from src.mlops.versioning import (
    CALIBRATION_VERSION_DEFAULT,
    DATASET_VERSION_DEFAULT,
    EVT_VERSION_DEFAULT,
    FEATURE_VERSION_DEFAULT,
)
from src.utils.logging import get_logger

logger = get_logger("rain_repair.mlops.registry")

ModelStatus = Literal[
    "trained",
    "validated",
    "candidate",
    "staging",
    "production",
    "retired",
    "rejected",
    "failed",
    "archived",
]

STATUS_TRANSITIONS: Dict[str, List[str]] = {
    "trained": ["validated", "candidate", "failed"],
    "candidate": ["validated", "staging", "production", "rejected", "failed"],
    "validated": ["staging", "production", "retired", "failed", "rejected"],
    "staging": ["production", "archived", "retired", "rejected", "failed"],
    "production": ["archived", "retired", "staging", "failed"],
    "retired": ["staging", "production"],
    "rejected": ["candidate", "trained"],
    "failed": ["trained", "candidate"],
    "archived": ["staging", "production", "retired"],
}



@dataclass
class RegisteredModel:
    """Standardized Model Registry Package conforming to PART 10 Section 10 & Phase 5."""
    model_id: str
    model_name: str
    version: str
    model_type: str
    dataset_version: str
    feature_version: str
    calibration_version: str
    evt_version: str
    training_period: Dict[str, str]
    validation_period: Dict[str, str]
    test_period: Dict[str, str]
    metrics: Dict[str, float]
    status: ModelStatus = "candidate"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    promoted_at: Optional[str] = None
    artifact_path: Optional[str] = None
    quality_gate_result: Optional[Dict[str, Any]] = None
    checksum: Optional[str] = None
    regime_model_version: Optional[str] = "v1.0.0"
    quantiles: Optional[List[float]] = field(default_factory=lambda: [0.50, 0.75, 0.90])
    components: Optional[Dict[str, Dict[str, Any]]] = None
    feature_schema: Optional[List[str]] = None

    @staticmethod
    def compute_checksum(filepath: Union[str, Path]) -> str:
        """Computes SHA-256 hexadecimal digest for a given artifact file."""
        p = Path(filepath)
        if not p.is_file():
            raise FileNotFoundError(f"Artifact file not found: {p}")
        sha256 = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def verify_checksum(self, base_dir: Optional[Path] = None) -> bool:
        """Verifies if the artifact exists and matches registered checksum."""
        if not self.checksum or not self.artifact_path:
            return True
        path = Path(self.artifact_path)
        if not path.is_absolute() and base_dir:
            path = base_dir / path
        if not path.exists():
            return False
        return self.compute_checksum(path) == self.checksum

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """
    Central operational model repository managing candidate evaluation,
    gated promotion, and safe rollback.
    """

    _default_instance: Optional[ModelRegistry] = None

    def __init__(
        self,
        registry_dir: Optional[Path] = None,
        quality_gate: Optional[ModelQualityGate] = None,
    ) -> None:
        self.registry_dir = registry_dir or Path("models/registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.quality_gate = quality_gate or ModelQualityGate()
        self.models: Dict[str, RegisteredModel] = {}
        self.production_history: List[str] = []
        self._load_or_initialize_registry()

    @classmethod
    def get_default_registry(cls) -> ModelRegistry:
        if cls._default_instance is None:
            cls._default_instance = ModelRegistry()
        return cls._default_instance

    def _load_or_initialize_registry(self) -> None:
        """Loads registered models from disk or initializes the initial production baseline."""
        json_files = list(self.registry_dir.glob("*.json"))
        if json_files:
            for f in json_files:
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                        rec = RegisteredModel(**data)
                        self.models[rec.model_id] = rec
                        if rec.status == "production":
                            self.production_history.append(rec.model_id)
                except Exception as e:
                    logger.warning("Could not load registry file %s: %s", f, e)
            logger.info("Loaded %d models from registry directory.", len(self.models))
        else:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            prod_model = RegisteredModel(
                model_id="rrx_v1_0_0_prod",
                model_name="RAIN-REPAIR-X-Operational",
                version="1.0.0",
                model_type="SoftMixtureOfExperts_Quantile_EVT",
                dataset_version=DATASET_VERSION_DEFAULT,
                feature_version=FEATURE_VERSION_DEFAULT,
                calibration_version=CALIBRATION_VERSION_DEFAULT,
                evt_version=EVT_VERSION_DEFAULT,
                training_period={"start": "2020-06-01", "end": "2024-09-30"},
                validation_period={"start": "2025-06-01", "end": "2025-09-30"},
                test_period={"start": "2026-06-01", "end": "2026-09-30"},
                metrics={"rmse": 16.20, "mae": 9.70, "csi_heavy": 0.44, "crps": 7.40, "brier_score": 0.114},
                status="production",
                created_at=now,
                promoted_at=now,
                artifact_path="models/checkpoints/rrx_v1_0_0.joblib",
            )
            cand_model = RegisteredModel(
                model_id="MOD_REPAIR_CANDIDATE_V2.4.1",
                model_name="RAIN-REPAIR-X-Candidate",
                version="2.4.1",
                model_type="SoftMixtureOfExperts_Quantile_EVT",
                dataset_version=DATASET_VERSION_DEFAULT,
                feature_version=FEATURE_VERSION_DEFAULT,
                calibration_version=CALIBRATION_VERSION_DEFAULT,
                evt_version=EVT_VERSION_DEFAULT,
                training_period={"start": "2020-06-01", "end": "2024-09-30"},
                validation_period={"start": "2025-06-01", "end": "2025-09-30"},
                test_period={"start": "2026-06-01", "end": "2026-09-30"},
                metrics={"rmse": 15.90, "mae": 9.50, "csi_heavy": 0.46, "crps": 7.20, "brier_score": 0.108},
                status="candidate",
                created_at=now,
                artifact_path="models/checkpoints/rrx_v2_4_1_cand.joblib",
            )
            self.models[prod_model.model_id] = prod_model
            self.models[cand_model.model_id] = cand_model
            self.production_history.append(prod_model.model_id)
            self._save_model_record(prod_model)
            self._save_model_record(cand_model)

    def _save_model_record(self, model: RegisteredModel) -> Path:
        file_path = self.registry_dir / f"{model.model_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(model.to_dict(), f, indent=2)
        return file_path

    def register_model(
        self,
        model_name: str,
        version: str,
        model_type: str,
        metrics: Dict[str, float],
        training_period: Optional[Dict[str, str]] = None,
        validation_period: Optional[Dict[str, str]] = None,
        test_period: Optional[Dict[str, str]] = None,
        artifact_path: Optional[str] = None,
        initial_status: ModelStatus = "candidate",
        manifest: Optional[Any] = None,
        temporal_split: Optional[Any] = None,
        status: Optional[str] = None,
        checksum: Optional[str] = None,
        regime_model_version: Optional[str] = "v1.0.0",
        quantiles: Optional[List[float]] = None,
        components: Optional[Dict[str, Dict[str, Any]]] = None,
        feature_schema: Optional[List[str]] = None,
    ) -> RegisteredModel:
        """Registers a new model package in the registry."""
        model_id = f"{model_name.lower().replace('-', '_')}_{version.replace('.', '_')}"
        d_ver = manifest.dataset_version if manifest else DATASET_VERSION_DEFAULT
        f_ver = manifest.feature_version if manifest else FEATURE_VERSION_DEFAULT
        c_ver = manifest.calibration_version if manifest else CALIBRATION_VERSION_DEFAULT
        e_ver = manifest.evt_version if manifest else EVT_VERSION_DEFAULT

        t_period = training_period or (
            {"start": temporal_split.train_start, "end": temporal_split.train_end}
            if temporal_split else {"start": "2020-06-01", "end": "2024-09-30"}
        )
        v_period = validation_period or (
            {"start": temporal_split.validation_start, "end": temporal_split.validation_end}
            if temporal_split else {"start": "2025-06-01", "end": "2025-09-30"}
        )
        s_period = test_period or (
            {"start": temporal_split.test_start, "end": temporal_split.test_end}
            if temporal_split else {"start": "2026-06-01", "end": "2026-09-30"}
        )

        final_status = status or initial_status
        rec = RegisteredModel(
            model_id=model_id,
            model_name=model_name,
            version=version,
            model_type=model_type,
            dataset_version=d_ver,
            feature_version=f_ver,
            calibration_version=c_ver,
            evt_version=e_ver,
            training_period=t_period,
            validation_period=v_period,
            test_period=s_period,
            metrics=metrics,
            status=final_status,
            artifact_path=artifact_path,
            checksum=checksum,
            regime_model_version=regime_model_version,
            quantiles=quantiles or [0.50, 0.75, 0.90],
            components=components,
            feature_schema=feature_schema,
        )
        self.models[model_id] = rec
        self._save_model_record(rec)
        logger.info("Registered model %s (Status: %s).", model_id, final_status)
        return rec

    def get_model(self, model_id: str) -> Optional[RegisteredModel]:
        return self.models.get(model_id)

    def get_production_model(self) -> Optional[RegisteredModel]:
        """Returns the currently active production model, falling back safely to the primary baseline."""
        for m in self.models.values():
            if m.status == "production":
                return m
        # Operational fallback: ensure the registry is never left without an active production model
        for mid in ["rrx_v1_0_0_prod", "MOD_REPAIR_CANDIDATE_V2.4.1"]:
            if mid in self.models:
                m = self.models[mid]
                m.status = "production"
                self._save_model_record(m)
                return m
        if self.models:
            first_m = next(iter(self.models.values()))
            first_m.status = "production"
            self._save_model_record(first_m)
            return first_m
        return None


    def list_models(self, status_filter: Optional[str] = None) -> List[RegisteredModel]:
        """Lists all registered models, optionally filtered by status."""
        results = []
        for m in self.models.values():
            if status_filter is None or m.status == status_filter:
                results.append(m)
        return results

    def promote_model(
        self,
        model_id: str,
        target_status: ModelStatus,
        val_predictions: Optional[pd.DataFrame] = None,
        promoted_by: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[QualityGateResult]]:
        """
        Promotes a model through lifecycle stages.
        Executes strict quality gate before staging or production promotion.
        """
        model = self.models.get(model_id)
        if not model:
            return False, f"Model '{model_id}' not found in registry.", None

        norm_target = str(target_status).lower()
        current_status = model.status.lower()
        allowed_targets = STATUS_TRANSITIONS.get(current_status, [])
        if norm_target not in allowed_targets:
            return False, f"Invalid state transition: '{current_status}' -> '{norm_target}'. Allowed: {allowed_targets}", None

        gate_res: Optional[QualityGateResult] = None
        if norm_target in ["validated", "staging", "production"]:
            prod_model = self.get_production_model()
            prod_metrics = prod_model.metrics if prod_model else None

            gate_res = self.quality_gate.evaluate_candidate(
                candidate_metrics=model.metrics,
                production_metrics=prod_metrics,
                val_predictions=val_predictions,
                target_status=target_status,
            )

            if not gate_res.gate_passed and norm_target in ["staging", "production"]:
                model.status = "rejected"
                model.quality_gate_result = gate_res.to_dict()
                self._save_model_record(model)
                logger.warning("Promotion rejected for %s: %s", model_id, gate_res.failure_reasons)
                return False, f"Quality gate failed: {gate_res.failure_reasons}", gate_res

        if norm_target == "production":
            current_prod = self.get_production_model()
            if current_prod and current_prod.model_id != model_id:
                current_prod.status = "retired"
                self._save_model_record(current_prod)
                logger.info("Retired previous production model: %s", current_prod.model_id)
            self.production_history.append(model_id)

        model.status = norm_target
        model.promoted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if gate_res:
            model.quality_gate_result = gate_res.to_dict()
        self._save_model_record(model)
        logger.info("Successfully promoted model %s to %s by %s.", model_id, target_status, promoted_by or "system")
        return True, f"Successfully promoted to {target_status}.", gate_res

    def rollback_to_previous_production(
        self,
        reason: Optional[str] = None,
        triggered_by: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Safely rolls back from current production model to the immediately preceding
        validated production model without retraining.
        """
        current_prod = self.get_production_model()
        if not current_prod:
            return False, "No active production model found."

        predecessor_id: Optional[str] = None
        for mid in reversed(self.production_history):
            if mid != current_prod.model_id and mid in self.models:
                predecessor_id = mid
                break

        if not predecessor_id:
            for m in self.models.values():
                if m.model_id != current_prod.model_id and m.status in ["archived", "retired", "staging", "validated"]:
                    predecessor_id = m.model_id
                    break

        if not predecessor_id:
            return False, "No prior valid model available in registry to rollback to."

        current_prod.status = "archived"
        self._save_model_record(current_prod)

        prev_model = self.models[predecessor_id]
        prev_model.status = "production"
        prev_model.promoted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._save_model_record(prev_model)

        logger.info("Safe rollback executed: %s -> %s (Reason: %s, By: %s)", current_prod.model_id, prev_model.model_id, reason, triggered_by)
        return True, f"Successfully rolled back from '{current_prod.model_id}' to '{prev_model.model_id}'."

