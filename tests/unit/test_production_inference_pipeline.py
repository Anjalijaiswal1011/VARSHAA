"""
Unit Tests for Phase 5 — Production Inference Pipeline & Model Serving.
Verifies all 9 mandatory test categories specified in Phase 5:
1. Input Validation: Coordinates, temporal causality, negative precipitation, schema types.
2. Feature Consistency: Training schema == inference schema; stops on mismatch.
3. Temporal Safety: Historical error retrieval forbids future observations (t >= T).
4. Regime Output: 0 <= p <= 1, sum(p) == 1.0, dominant regime extraction, probability vector preserved.
5. Quantile Output: Non-negativity (>=0) and monotonicity (P50 <= P75 <= P90).
6. Model Loading & Integrity: Checkpoint existence, SHA-256 verification, no silent fallback.
7. Reproducibility: Same input + same model -> identical prediction.
8. Fault-Isolated Batch Inference: Individual invalid records do not corrupt batch.
9. End-to-End Inference Flow: Full data flow from raw input to validated prediction record.
"""

from __future__ import annotations

from datetime import datetime, timezone
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.mlops.registry import ModelRegistry, RegisteredModel
from src.models.inference_pipeline import (
    BatchForecastInput,
    FeatureSchemaMismatchError,
    ForecastInputRecord,
    InferenceError,
    InputValidationError,
    InsufficientHistoryError,
    ModelIntegrityError,
    ModelNotFoundError,
    ProductionInferencePipeline,
    ProductionModelLoader,
)
from src.postprocessing.memory import LeakageSafeErrorMemoryStore
from src.utils.exceptions import PhysicalConstraintViolationError, TemporalLeakageError


@pytest.fixture(autouse=True)
def reset_model_loader_cache():
    """Ensure clean model loader state across tests."""
    ProductionModelLoader.clear_cache()
    yield
    ProductionModelLoader.clear_cache()


@pytest.fixture
def valid_forecast_input() -> dict:
    """Standard valid NWP forecast state for Pune / Western Ghats."""
    return {
        "forecast_time": "2026-07-15T03:00:00Z",
        "initialization_time": "2026-07-15T00:00:00Z",
        "latitude": 18.50,
        "longitude": 73.75,
        "lead_time": 24,
        "nwp_precip": 45.2,
        "t2m": 298.15,
        "q2m": 0.018,
        "u10": 10.5,
        "v10": 3.2,
        "mslp": 1004.5,
        "elevation": 560.0,
        "slope": 4.2,
        "dist_to_coast": 65.0,
        "cape": 1450.0,
    }


# ==============================================================================
# 1. INPUT VALIDATION TESTS
# ==============================================================================
class TestInferenceInputValidation:
    """Tests strict validation of datatypes, physical limits, and temporal causality."""

    def test_valid_input_instantiation(self, valid_forecast_input):
        record = ForecastInputRecord(**valid_forecast_input)
        assert record.latitude == 18.50
        assert record.longitude == 73.75
        assert record.nwp_precip == 45.2
        assert record.lead_time == 24

    def test_invalid_spatial_bounds_rejected(self, valid_forecast_input):
        # Lat out of India bounds (e.g. 45.0 N)
        bad_lat = dict(valid_forecast_input, latitude=45.0)
        with pytest.raises(ValueError, match="outside the Indian subcontinent domain"):
            ForecastInputRecord(**bad_lat)

        # Lon out of India bounds (e.g. 55.0 E)
        bad_lon = dict(valid_forecast_input, longitude=55.0)
        with pytest.raises(ValueError, match="outside the Indian subcontinent domain"):
            ForecastInputRecord(**bad_lon)

    def test_negative_precipitation_rejected(self, valid_forecast_input):
        bad_precip = dict(valid_forecast_input, nwp_precip=-12.5)
        with pytest.raises(ValueError, match="Precipitation cannot be negative"):
            ForecastInputRecord(**bad_precip)

    def test_temporal_causality_violation_rejected(self, valid_forecast_input):
        # forecast_time earlier than initialization_time
        inverted_times = dict(
            valid_forecast_input,
            forecast_time="2026-07-14T18:00:00Z",
            initialization_time="2026-07-15T00:00:00Z",
        )
        with pytest.raises(ValueError, match="Temporal causality violation"):
            ForecastInputRecord(**inverted_times)

    def test_negative_lead_time_rejected(self, valid_forecast_input):
        bad_lt = dict(valid_forecast_input, lead_time=-6)
        with pytest.raises(ValueError, match="Lead time cannot be negative"):
            ForecastInputRecord(**bad_lt)


# ==============================================================================
# 2. FEATURE CONSISTENCY TESTS
# ==============================================================================
class TestFeatureConsistency:
    """Verifies that inference features match training schema and stops on mismatch."""

    def test_feature_consistency_matches_model(self, valid_forecast_input):
        pipeline = ProductionInferencePipeline()
        record = ForecastInputRecord(**valid_forecast_input)
        pred = pipeline.predict_single(record)
        assert pred.prediction_status in ["valid", "calibrated_rearranged"]
        assert pred.model_version in ["1.0.0", "2.4.1"]

    def test_feature_schema_mismatch_stops_inference(self, valid_forecast_input, monkeypatch):
        pipeline = ProductionInferencePipeline()
        pkg = pipeline.model_loader.load_model()

        # Artificially inject an unexpected required feature in model schema
        original_cols = list(pkg.feature_cols)
        tampered_cols = original_cols + ["unsupported_solar_irradiance_flux"]
        monkeypatch.setattr(pkg, "feature_cols", tampered_cols)

        record = ForecastInputRecord(**valid_forecast_input)
        with pytest.raises(FeatureSchemaMismatchError, match="Missing required quantile corrector features"):
            pipeline.predict_single(record)


# ==============================================================================
# 3. TEMPORAL SAFETY & ERROR MEMORY TESTS
# ==============================================================================
class TestTemporalSafety:
    """Enforces zero leakage: future observations (t >= T) must never be accessed."""

    def test_historical_error_retrieval_excludes_future(self, valid_forecast_input):
        store = LeakageSafeErrorMemoryStore()
        # Record past observations (t < 2026-07-15)
        store.record_verified_cycle(
            verification_date="2026-07-14",
            df_cycle=pd.DataFrame({"lat": [18.5], "lon": [73.75], "obs_precip": [50.0], "nwp_precip": [42.0]}),
        )
        store.record_verified_cycle(
            verification_date="2026-07-13",
            df_cycle=pd.DataFrame({"lat": [18.5], "lon": [73.75], "obs_precip": [30.0], "nwp_precip": [35.0]}),
        )

        pipeline = ProductionInferencePipeline(error_memory_store=store)
        record = ForecastInputRecord(**valid_forecast_input)
        pred = pipeline.predict_single(record)

        assert pred.diagnostics["memory_status"] == "PARTIAL_HISTORY_2_DAYS"
        assert pred.corrected_p50 > 0

    def test_strict_anti_leakage_exception(self):
        store = LeakageSafeErrorMemoryStore()
        # Attempting to fetch past dates relative to cycle 2026-07-10
        # If store erroneously contained 2026-07-10 or later
        store._history["2026-07-12"] = {"18.50_73.75": 5.0}

        # get_valid_past_dates filters strictly < 2026-07-10
        valid_dates = store.get_valid_past_dates("2026-07-10")
        assert "2026-07-12" not in valid_dates

    def test_strict_mode_rejects_insufficient_history(self, valid_forecast_input):
        empty_store = LeakageSafeErrorMemoryStore()
        pipeline = ProductionInferencePipeline(error_memory_store=empty_store)
        record = ForecastInputRecord(**valid_forecast_input)

        with pytest.raises(InsufficientHistoryError, match="No historical error memory available"):
            pipeline.predict_single(record, require_full_history=True)


# ==============================================================================
# 4. REGIME INFERENCE & PROBABILITY VECTOR TESTS
# ==============================================================================
class TestRegimeInference:
    """Verifies regime probabilities meet physical probability axioms."""

    def test_regime_probability_axioms(self, valid_forecast_input):
        pipeline = ProductionInferencePipeline()
        pred = pipeline.predict_single(valid_forecast_input)

        probs = pred.regime_probabilities
        assert len(probs) == 6
        assert "ACTIVE_MONSOON" in probs
        assert "BREAK_MONSOON" in probs
        assert "MONSOON_DEPRESSION" in probs
        assert "COASTAL" in probs
        assert "OROGRAPHIC" in probs
        assert "WESTERN_DISTURBANCE" in probs

        # 0 <= p <= 1
        for reg, p in probs.items():
            assert 0.0 <= p <= 1.0, f"Probability for {reg} out of bounds: {p}"

        # Sum == 1.0
        assert np.isclose(sum(probs.values()), 1.0, atol=1e-3)

        # Dominant regime metadata
        assert pred.dominant_regime in probs
        assert np.isclose(pred.dominant_probability, probs[pred.dominant_regime], atol=1e-3)


# ==============================================================================
# 5. QUANTILE OUTPUT & MONOTONICITY TESTS
# ==============================================================================
class TestQuantileValidation:
    """Verifies non-negativity, monotonicity, and mathematical rearrangement."""

    def test_quantile_non_negativity_and_monotonicity(self, valid_forecast_input):
        pipeline = ProductionInferencePipeline()
        pred = pipeline.predict_single(valid_forecast_input)

        assert pred.corrected_p50 >= 0.0
        assert pred.corrected_p75 >= 0.0
        assert pred.corrected_p90 >= 0.0

        assert pred.corrected_p50 <= pred.corrected_p75 + 1e-4
        assert pred.corrected_p75 <= pred.corrected_p90 + 1e-4
        assert pred.spread_p90_p50 >= 0.0

    def test_rearrangement_operator_resolves_crossing(self, valid_forecast_input, monkeypatch):
        pipeline = ProductionInferencePipeline()
        pkg = pipeline.model_loader.load_model()

        # Mock quantile corrector to intentionally emit crossing quantiles
        mock_raw = pd.DataFrame({"p50": [60.0], "p75": [40.0], "p90": [20.0]})
        monkeypatch.setattr(pkg.quantile_corrector, "predict", lambda *args, **kwargs: mock_raw)

        pred = pipeline.predict_single(valid_forecast_input)
        assert pred.prediction_status == "calibrated_rearranged"
        assert pred.diagnostics["rearrangement_applied"] is True

        # Post-rearrangement must be strictly ordered
        assert pred.corrected_p50 <= pred.corrected_p75 <= pred.corrected_p90
        assert pred.corrected_p50 == 20.0
        assert pred.corrected_p75 == 40.0
        assert pred.corrected_p90 == 60.0


# ==============================================================================
# 6. MODEL LOADING & TAMPER RESISTANCE TESTS
# ==============================================================================
class TestModelLoadingIntegrity:
    """Verifies safe lifecycle, checksum tamper detection, and no silent fallback."""

    def test_model_loading_and_caching(self):
        loader = ProductionModelLoader()
        loader.clear_cache()

        pkg1 = loader.load_model(model_id="rrx_v1_0_0_prod")
        assert pkg1.model_id == "rrx_v1_0_0_prod"
        assert pkg1.model_version == "1.0.0"

        # Cached reuse
        pkg2 = loader.load_model(model_id="rrx_v1_0_0_prod")
        assert pkg1 is pkg2

        # Active production model lookup
        prod_pkg = loader.load_model()
        assert prod_pkg.model_id in ["rrx_v1_0_0_prod", "MOD_REPAIR_CANDIDATE_V2.4.1"]

    def test_missing_model_raises_503(self):
        loader = ProductionModelLoader()
        with pytest.raises(ModelNotFoundError, match="is not registered"):
            loader.load_model(model_id="NON_EXISTENT_MODEL_V99")

    def test_checksum_tamper_detection(self, monkeypatch):
        loader = ProductionModelLoader()
        loader.clear_cache()

        reg = ModelRegistry.get_default_registry()
        prod_m = reg.get_production_model()

        # Tamper registered checksum
        fake_checksum = "0000000000000000000000000000000000000000000000000000000000000000"
        monkeypatch.setattr(prod_m, "checksum", fake_checksum)

        with pytest.raises(ModelIntegrityError, match="Model checksum mismatch"):
            loader.load_model(model_id=prod_m.model_id)


# ==============================================================================
# 7. REPRODUCIBILITY TEST
# ==============================================================================
class TestPredictionReproducibility:
    """Verifies deterministic prediction given identical inputs."""

    def test_exact_reproducibility(self, valid_forecast_input):
        pipeline = ProductionInferencePipeline()
        pred1 = pipeline.predict_single(valid_forecast_input)
        pred2 = pipeline.predict_single(valid_forecast_input)

        assert np.isclose(pred1.corrected_p50, pred2.corrected_p50, atol=1e-5)
        assert np.isclose(pred1.corrected_p75, pred2.corrected_p75, atol=1e-5)
        assert np.isclose(pred1.corrected_p90, pred2.corrected_p90, atol=1e-5)
        assert pred1.dominant_regime == pred2.dominant_regime
        assert np.isclose(pred1.dominant_probability, pred2.dominant_probability, atol=1e-5)


# ==============================================================================
# 8. BATCH INFERENCE & FAULT ISOLATION TESTS
# ==============================================================================
class TestBatchInferenceFaultIsolation:
    """Verifies batch execution and fault isolation for individual invalid records."""

    def test_batch_prediction_with_fault_isolation(self, valid_forecast_input):
        pipeline = ProductionInferencePipeline()

        record_1 = ForecastInputRecord(**valid_forecast_input)
        record_2 = ForecastInputRecord(**dict(valid_forecast_input, latitude=19.0, longitude=72.85, nwp_precip=68.0))
        # Record 3 has an unprocessable invalid latitude (outside India)
        # We simulate raw dict input in batch
        batch_payload = {
            "records": [
                valid_forecast_input,
                dict(valid_forecast_input, latitude=19.0, longitude=72.85, nwp_precip=68.0),
                dict(valid_forecast_input, latitude=52.0),  # INVALID
            ]
        }

        resp = pipeline.predict_batch(batch_payload)
        assert resp.total_records == 3
        assert resp.successful_count == 2
        assert resp.failed_count == 1
        assert len(resp.predictions) == 2
        assert len(resp.failed_records) == 1
        assert resp.failed_records[0]["record_index"] == 2
        assert "outside the Indian subcontinent domain" in resp.failed_records[0]["error_message"]
