"""
Operational Performance & Latency Benchmark Script for RAIN-REPAIR X (Phase 5).
Measures real physical timings for:
- Model checkpoint load & integrity check
- Single prediction (N=1)
- Small batch (N=10)
- Larger batch (N=50)
Captures exact End-to-End Prediction Record without fabrication.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

from src.models.inference_pipeline import (
    ForecastInputRecord,
    ProductionInferencePipeline,
    ProductionModelLoader,
)
from src.postprocessing.memory import LeakageSafeErrorMemoryStore


def create_sample_input(lat: float, lon: float, nwp_precip: float, lead_time: int = 24) -> dict:
    return {
        "forecast_time": "2026-07-15T03:00:00Z",
        "initialization_time": "2026-07-15T00:00:00Z",
        "latitude": lat,
        "longitude": lon,
        "lead_time": lead_time,
        "nwp_precip": nwp_precip,
        "t2m": 298.15,
        "q2m": 0.018,
        "u10": 9.2,
        "v10": 3.5,
        "mslp": 1004.2,
        "elevation": 560.0,
        "slope": 4.5,
        "dist_to_coast": 65.0,
        "cape": 1500.0,
    }


def run_benchmark():
    print("=" * 70)
    print("RAIN-REPAIR X — PHASE 5 PRODUCTION INFERENCE BENCHMARK")
    print("=" * 70)

    # 1. Measure Cold Start Model Loading
    ProductionModelLoader.clear_cache()
    t0 = time.perf_counter()
    loader = ProductionModelLoader()
    pkg = loader.load_model()
    t_cold_load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[1] Cold Start Model Load Time: {t_cold_load_ms:.2f} ms")

    # Warm load
    t0 = time.perf_counter()
    pkg_warm = loader.load_model()
    t_warm_load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"    Warm Cached Model Load Time: {t_warm_load_ms:.4f} ms")
    print(f"    Loaded Model ID: {pkg.model_id}, Version: {pkg.model_version}")

    # Seed Error Memory Store with past verification cycles
    store = LeakageSafeErrorMemoryStore()
    for day in range(1, 15):
        date_str = f"2026-07-{day:02d}"
        store.record_verified_cycle(
            verification_date=date_str,
            df_cycle=pd.DataFrame({
                "lat": [18.50, 19.00, 11.70, 30.30],
                "lon": [73.75, 72.85, 76.10, 78.05],
                "obs_precip": [48.0, 72.0, 90.0, 50.0],
                "nwp_precip": [45.0, 68.0, 82.0, 55.0],
            }),
        )

    pipeline = ProductionInferencePipeline(model_loader=loader, error_memory_store=store)

    # 2. End-to-End Single Prediction (N=1)
    single_input = create_sample_input(lat=18.50, lon=73.75, nwp_precip=45.2, lead_time=24)
    print("\n[2] End-to-End Single Prediction (N=1)")
    
    t0 = time.perf_counter()
    pred_single = pipeline.predict_single(single_input)
    t_single_total_ms = (time.perf_counter() - t0) * 1000.0

    timings = pred_single.diagnostics["timings_ms"]
    print(f"    Total Latency:            {t_single_total_ms:.2f} ms")
    print(f"    - Model Loading:          {timings['model_load']:.2f} ms")
    print(f"    - Feature Gen & Memory:   {timings['feature_generation']:.2f} ms")
    print(f"    - Regime Inference:       {timings['regime_inference']:.2f} ms")
    print(f"    - Quantile Inference:     {timings['quantile_inference']:.2f} ms")
    print(f"    P50: {pred_single.corrected_p50:.2f} mm | P75: {pred_single.corrected_p75:.2f} mm | P90: {pred_single.corrected_p90:.2f} mm")
    print(f"    Dominant Regime: {pred_single.dominant_regime} ({pred_single.dominant_probability*100:.1f}%)")
    print(f"    Prediction Status: {pred_single.prediction_status}")

    # 3. Small Batch Prediction (N=10)
    print("\n[3] Small Batch Prediction (N=10)")
    batch_10_inputs = [
        create_sample_input(
            lat=18.0 + (i % 3) * 0.5,
            lon=73.0 + (i // 3) * 0.5,
            nwp_precip=20.0 + i * 5.0,
            lead_time=24 * ((i % 5) + 1),
        )
        for i in range(10)
    ]
    t0 = time.perf_counter()
    resp_10 = pipeline.predict_batch({"records": batch_10_inputs})
    t_batch_10_ms = (time.perf_counter() - t0) * 1000.0
    avg_per_rec_10 = t_batch_10_ms / 10.0
    throughput_10 = 10.0 / (t_batch_10_ms / 1000.0)

    print(f"    Total Batch Latency (N=10): {t_batch_10_ms:.2f} ms")
    print(f"    Average Latency / Record:   {avg_per_rec_10:.2f} ms/record")
    print(f"    Throughput:                 {throughput_10:.1f} records/sec")
    print(f"    Successful: {resp_10.successful_count}/{resp_10.total_records}")

    # 4. Larger Batch Prediction (N=50)
    print("\n[4] Larger Batch Prediction (N=50)")
    batch_50_inputs = [
        create_sample_input(
            lat=12.0 + (i % 10) * 1.5,
            lon=72.0 + (i // 10) * 2.0,
            nwp_precip=10.0 + (i % 15) * 4.0,
            lead_time=24 * ((i % 5) + 1),
        )
        for i in range(50)
    ]
    t0 = time.perf_counter()
    resp_50 = pipeline.predict_batch({"records": batch_50_inputs})
    t_batch_50_ms = (time.perf_counter() - t0) * 1000.0
    avg_per_rec_50 = t_batch_50_ms / 50.0
    throughput_50 = 50.0 / (t_batch_50_ms / 1000.0)

    print(f"    Total Batch Latency (N=50): {t_batch_50_ms:.2f} ms")
    print(f"    Average Latency / Record:   {avg_per_rec_50:.2f} ms/record")
    print(f"    Throughput:                 {throughput_50:.1f} records/sec")
    print(f"    Successful: {resp_50.successful_count}/{resp_50.total_records}")

    # 5. Output End-to-End JSON Artifact
    output_record_json = pred_single.model_dump_json(indent=2)
    print("\n[5] Actual Captured Prediction Record:")
    print(output_record_json)

    benchmark_summary = {
        "benchmark_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cold_start_model_load_ms": round(t_cold_load_ms, 2),
        "warm_cached_model_load_ms": round(t_warm_load_ms, 4),
        "single_prediction": {
            "request_size": 1,
            "total_latency_ms": round(t_single_total_ms, 2),
            "feature_generation_ms": timings["feature_generation"],
            "regime_inference_ms": timings["regime_inference"],
            "quantile_inference_ms": timings["quantile_inference"],
            "captured_prediction": pred_single.model_dump(),
        },
        "small_batch": {
            "request_size": 10,
            "total_latency_ms": round(t_batch_10_ms, 2),
            "latency_per_record_ms": round(avg_per_rec_10, 2),
            "throughput_records_sec": round(throughput_10, 1),
        },
        "larger_batch": {
            "request_size": 50,
            "total_latency_ms": round(t_batch_50_ms, 2),
            "latency_per_record_ms": round(avg_per_rec_50, 2),
            "throughput_records_sec": round(throughput_50, 1),
        },
    }

    with open("models/experiments/phase5_performance_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)
    print("\nSaved benchmark results to models/experiments/phase5_performance_benchmark.json")


if __name__ == "__main__":
    run_benchmark()
