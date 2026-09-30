"""
Performance Measurement Benchmark Script for Phase 8 Forecast Product APIs.
Measures:
1. Single District Request (GET /api/v1/forecasts/districts/MH_PUNE)
2. Multiple District Request (GET /api/v1/forecasts/districts?limit=50)
3. Grid Forecast Request (GET /api/v1/forecasts/grid?limit=50)
4. Large Geographic Request (GET /api/v1/forecasts/grid with Bounding Box)

Captures:
- End-to-end response latency (ms)
- Service query & aggregation time (ms)
- Pydantic serialization time (ms)
- Peak memory allocation (KB)
"""

from __future__ import annotations

import json
import time
import tracemalloc
from typing import Any, Dict, List
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import ForecastService

client = TestClient(app)
service = ForecastService()


def benchmark_endpoint(name: str, url: str, num_runs: int = 5) -> Dict[str, Any]:
    """Runs a URL benchmark across num_runs and averages metrics."""
    latencies: List[float] = []

    # Warmup
    _ = client.get(url)

    tracemalloc.start()
    for _ in range(num_runs):
        t0 = time.perf_counter()
        resp = client.get(url)
        t1 = time.perf_counter()
        assert resp.status_code == 200, f"Benchmark failed for {url}: {resp.status_code}"
        latencies.append((t1 - t0) * 1000.0)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    return {
        "benchmark_name": name,
        "url": url,
        "runs": num_runs,
        "avg_latency_ms": round(avg_latency, 2),
        "min_latency_ms": round(min_latency, 2),
        "max_latency_ms": round(max_latency, 2),
        "peak_memory_kb": round(peak_mem / 1024.0, 2),
    }


def benchmark_internal_breakdown() -> Dict[str, Any]:
    """Measures query time vs serialization time for single district & grid."""
    # 1. Single district breakdown
    t0 = time.perf_counter()
    ok, code, single_dict = service.get_product_single_district_forecast("MH_PUNE", lead_time=24)
    t1 = time.perf_counter()
    query_time_single = (t1 - t0) * 1000.0

    from backend.schemas import CanonicalDistrictForecastRecord
    t2 = time.perf_counter()
    _ = CanonicalDistrictForecastRecord(**single_dict).model_dump()
    t3 = time.perf_counter()
    serialization_time_single = (t3 - t2) * 1000.0

    # 2. Multi-district breakdown
    t4 = time.perf_counter()
    ok, multi_dict = service.get_product_district_forecasts(lead_time=24, limit=50)
    t5 = time.perf_counter()
    query_time_multi = (t5 - t4) * 1000.0

    from backend.schemas import DistrictForecastProductResponse
    t6 = time.perf_counter()
    _ = DistrictForecastProductResponse(**multi_dict).model_dump()
    t7 = time.perf_counter()
    serialization_time_multi = (t7 - t6) * 1000.0

    # 3. Grid breakdown
    t8 = time.perf_counter()
    ok, grid_dict = service.get_product_grid_forecasts(lead_time=24, limit=50)
    t9 = time.perf_counter()
    query_time_grid = (t9 - t8) * 1000.0

    from backend.schemas import GridForecastProductResponse
    t10 = time.perf_counter()
    _ = GridForecastProductResponse(**grid_dict).model_dump()
    t11 = time.perf_counter()
    serialization_time_grid = (t11 - t10) * 1000.0

    return {
        "single_district": {
            "query_time_ms": round(query_time_single, 2),
            "serialization_time_ms": round(serialization_time_single, 2),
            "total_ms": round(query_time_single + serialization_time_single, 2),
        },
        "multi_district": {
            "query_time_ms": round(query_time_multi, 2),
            "serialization_time_ms": round(serialization_time_multi, 2),
            "total_ms": round(query_time_multi + serialization_time_multi, 2),
        },
        "grid": {
            "query_time_ms": round(query_time_grid, 2),
            "serialization_time_ms": round(serialization_time_grid, 2),
            "total_ms": round(query_time_grid + serialization_time_grid, 2),
        },
    }


def main():
    print("=" * 70)
    print("PHASE 8 FORECAST PRODUCT API PERFORMANCE BENCHMARKS")
    print("=" * 70)

    endpoints = [
        ("Single District Request", "/api/v1/forecasts/districts/MH_PUNE?lead_time=24"),
        ("Multiple District Request", "/api/v1/forecasts/districts?lead_time=24&limit=50"),
        ("Grid Forecast Request", "/api/v1/forecasts/grid?lead_time=24&limit=50"),
        ("Large Geographic Request (BBox)", "/api/v1/forecasts/grid?lead_time=24&min_lat=10.0&max_lat=30.0&min_lon=70.0&max_lon=90.0&limit=100"),
    ]

    results = []
    for name, url in endpoints:
        res = benchmark_endpoint(name, url, num_runs=10)
        results.append(res)
        print(f"[*] {name:32s} | Avg: {res['avg_latency_ms']:6.2f} ms | Min: {res['min_latency_ms']:6.2f} ms | Max: {res['max_latency_ms']:6.2f} ms | Peak Mem: {res['peak_memory_kb']:7.2f} KB")

    print("\n" + "-" * 70)
    print("INTERNAL QUERY & SERIALIZATION LATENCY BREAKDOWN")
    print("-" * 70)
    breakdown = benchmark_internal_breakdown()
    for cat, vals in breakdown.items():
        print(f"[*] {cat:20s} | Query: {vals['query_time_ms']:6.2f} ms | Serialization: {vals['serialization_time_ms']:6.2f} ms | Sum: {vals['total_ms']:6.2f} ms")

    output_data = {
        "endpoint_benchmarks": results,
        "internal_breakdown": breakdown,
    }

    with open("logs/phase8_performance_metrics.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("\n[+] Benchmark metrics saved to logs/phase8_performance_metrics.json")


if __name__ == "__main__":
    main()
