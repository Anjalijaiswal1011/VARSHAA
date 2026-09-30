# RAAP-X Phase 8 — District & Grid Forecast Product API Reference

## 1. Overview
The RAAP-X District/Grid Forecast Product API exposes post-processed, regime-aware rainfall predictions with preserved uncertainty ($P_{50}, P_{75}, P_{90}$), calibrated extreme risk probabilities, and strict administrative boundary lineage.

The product layer serves pre-computed and verified forecasts produced by the operational pipeline. It does **not** recompute machine learning models within GET requests.

Base URL: `/api/v1`

---

## 2. Endpoints

### 2.1 Grid Forecast Product API
* **Method & Path**: `GET /api/v1/forecasts/grid`
* **Summary**: Query verified grid-level predictions with uncertainty quantiles, synoptic regime distributions, and risk exceedances.
* **Query Parameters**:
  | Parameter | Type | Required | Default | Description |
  | :--- | :--- | :--- | :--- | :--- |
  | `lead_time` | `int` | No | `24` | Forecast lead time in hours (`24, 48, 72, 96, 120`). |
  | `forecast_time` | `string` | No | `None` | Initialization or cycle date filter (`YYYY-MM-DD`). |
  | `min_lat` | `float` | No | `None` | South boundary latitude in WGS84 (`6.0` to `38.5`). |
  | `max_lat` | `float` | No | `None` | North boundary latitude in WGS84 (`6.0` to `38.5`). |
  | `min_lon` | `float` | No | `None` | West boundary longitude in WGS84 (`68.0` to `98.0`). |
  | `max_lon` | `float` | No | `None` | East boundary longitude in WGS84 (`68.0` to `98.0`). |
  | `latitude` | `float` | No | `None` | Proximity query point latitude. |
  | `longitude` | `float` | No | `None` | Proximity query point longitude. |
  | `radius_km` | `float` | No | `None` | Spatial radius around proximity point in km. |
  | `grid_id` | `string` | No | `None` | Exact grid point identifier (e.g. `G_18.50_73.75`). |
  | `limit` | `int` | No | `50` | Page size limit (`1` to `1000`). |
  | `offset` | `int` | No | `0` | Page offset. |

* **Status Codes**:
  * `200 OK`: Validated grid records returned.
  * `400 Bad Request`: Invalid coordinates, illegal bounding box, or unsupported lead time (RFC 7807).

* **Example Response**:
```json
{
  "pagination": {
    "total_count": 10,
    "limit": 1,
    "offset": 0,
    "has_more": true
  },
  "forecast_time": "2026-07-16T03:00:00Z",
  "lead_time": 24,
  "boundary_version": "IMD-LGD-2026.1",
  "model_version": "v1.0.0-prob",
  "records": [
    {
      "prediction_id": "pred_G_18.50_73.75_24_2026-07-15",
      "forecast_time": "2026-07-16T03:00:00Z",
      "initialization_time": "2026-07-15T00:00:00Z",
      "lead_time": 24,
      "latitude": 18.5,
      "longitude": 73.75,
      "grid_id": "G_18.50_73.75",
      "raw_nwp_rainfall": 49.0,
      "corrected_p50": 56.2,
      "corrected_p75": 71.94,
      "corrected_p90": 88.8,
      "spread_p90_p50": 32.6,
      "regime_probabilities": {
        "ACTIVE_MONSOON": 0.75,
        "BREAK_MONSOON": 0.05,
        "MONSOON_DEPRESSION": 0.05,
        "WESTERN_DISTURBANCE": 0.05,
        "OFFSHORE_TROUGH": 0.05,
        "NORMAL_TRANSITIONAL": 0.05
      },
      "dominant_regime": "ACTIVE_MONSOON",
      "dominant_probability": 0.75,
      "heavy_rainfall_probability": 0.5786,
      "extreme_rainfall_probability": 0.0955,
      "model_version": "v1.0.0-prob",
      "regime_model_version": "v1.0.0-regime-lgbm",
      "feature_version": "v1.0.0-phys",
      "dataset_version": "IMD-ERA5-v2.1",
      "boundary_version": "IMD-LGD-2026.1",
      "prediction_status": "VALID",
      "created_at": "2026-09-28T22:27:18Z",
      "rainfall": {
        "raw_nwp": 49.0,
        "p50": 56.2,
        "p75": 71.94,
        "p90": 88.8,
        "spread": 32.6,
        "difference": 7.2
      },
      "regime": {
        "dominant": "ACTIVE_MONSOON",
        "dominant_probability": 0.75,
        "probabilities": { ... }
      },
      "risk": {
        "heavy_rainfall_probability": 0.5786,
        "extreme_rainfall_probability": 0.0955,
        "warning_level": "ORANGE"
      },
      "metadata": {
        "model_version": "v1.0.0-prob",
        "regime_model_version": "v1.0.0-regime-lgbm",
        "feature_version": "v1.0.0-phys",
        "dataset_version": "IMD-ERA5-v2.1",
        "boundary_version": "IMD-LGD-2026.1",
        "pipeline_run_id": null
      }
    }
  ]
}
```

---

### 2.2 District Forecast Product API
* **Method & Path**: `GET /api/v1/forecasts/districts`
* **Summary**: Returns area-weighted district-level forecasts aggregated via Ray-Casting against `IMD-LGD-2026.1` boundaries.
* **Query Parameters**:
  | Parameter | Type | Required | Default | Description |
  | :--- | :--- | :--- | :--- | :--- |
  | `lead_time` | `int` | No | `24` | Forecast lead time in hours (`24, 48, 72, 96, 120`). |
  | `forecast_time` | `string` | No | `None` | Forecast cycle date filter (`YYYY-MM-DD`). |
  | `district_id` | `string` | No | `None` | District administrative code (e.g. `MH_PUNE`). |
  | `state` | `string` | No | `None` | State name filter (e.g. `Maharashtra`). |
  | `limit` | `int` | No | `50` | Pagination page limit. |
  | `offset` | `int` | No | `0` | Pagination offset. |

* **Status Codes**:
  * `200 OK`: District forecasts returned successfully.
  * `400 Bad Request`: Unsupported lead time or invalid parameters (RFC 7807).

---

### 2.3 Single District Forecast Product API
* **Method & Path**: `GET /api/v1/forecasts/districts/{district_id}`
* **Summary**: Returns the latest valid forecast for a single district, including model explanation metadata, uncertainty quantiles, and version lineage.
* **Path Parameters**:
  | Parameter | Type | Required | Description |
  | :--- | :--- | :--- | :--- |
  | `district_id` | `string` | Yes | Official district code (e.g. `MH_PUNE`, `KL_WAYANAD`). |
* **Query Parameters**:
  | Parameter | Type | Required | Default | Description |
  | :--- | :--- | :--- | :--- | :--- |
  | `lead_time` | `int` | No | `24` | Lead time in hours (`24, 48, 72, 96, 120`). |
  | `forecast_time` | `string` | No | `None` | Optional cycle date (`YYYY-MM-DD`). |

* **Status Codes**:
  * `200 OK`: Single district forecast returned.
  * `404 Not Found`: District ID not in catalog or forecast unavailable (RFC 7807).
  * `400 Bad Request`: Invalid lead time (RFC 7807).

* **Example Response**:
```json
{
  "district_id": "MH_PUNE",
  "district_name": "Pune",
  "state": "Maharashtra",
  "forecast_time": "2026-07-16T03:00:00Z",
  "initialization_time": "2026-07-15T00:00:00Z",
  "lead_time": 24,
  "raw_nwp_rainfall": 49.0,
  "corrected_p50": 56.2,
  "corrected_p75": 71.94,
  "corrected_p90": 88.8,
  "spread_p90_p50": 32.6,
  "heavy_rainfall_probability": 0.5786,
  "extreme_rainfall_probability": 0.0955,
  "dominant_regime": "ACTIVE_MONSOON",
  "regime_probabilities": {
    "ACTIVE_MONSOON": 0.75,
    "BREAK_MONSOON": 0.05,
    "MONSOON_DEPRESSION": 0.05,
    "WESTERN_DISTURBANCE": 0.05,
    "OFFSHORE_TROUGH": 0.05,
    "NORMAL_TRANSITIONAL": 0.05
  },
  "model_version": "v1.0.0-prob",
  "regime_model_version": "v1.0.0-regime-lgbm",
  "feature_version": "v1.0.0-phys",
  "dataset_version": "IMD-ERA5-v2.1",
  "boundary_version": "IMD-LGD-2026.1",
  "pipeline_run_id": null,
  "prediction_status": "VALID",
  "created_at": "2026-09-28T22:27:18Z",
  "status": "VALID",
  "district": {
    "id": "MH_PUNE",
    "name": "Pune",
    "state": "Maharashtra"
  },
  "rainfall": {
    "raw_nwp": 49.0,
    "p50": 56.2,
    "p75": 71.94,
    "p90": 88.8,
    "spread": 32.6,
    "difference": 7.2
  },
  "regime": {
    "dominant": "ACTIVE_MONSOON",
    "dominant_probability": 0.75,
    "probabilities": { ... }
  },
  "risk": {
    "heavy_rainfall_probability": 0.5786,
    "extreme_rainfall_probability": 0.0955,
    "warning_level": "ORANGE"
  },
  "explanation": {
    "dominant_regime": "ACTIVE_MONSOON",
    "regime_confidence": 0.75,
    "forecast_spread": 32.6,
    "top_features": [
      {
        "feature": "raw_nwp_precip",
        "contribution_mm": 49.0,
        "category": "nwp_rainfall",
        "description": "Baseline NWP forecast predicts 49.0 mm accumulated precipitation."
      },
      {
        "feature": "error_memory_lag3",
        "contribution_mm": 7.2,
        "category": "recent_error",
        "description": "Multi-scale error memory applied an adjustment of +7.2 mm."
      },
      {
        "feature": "regime_prob_active_monsoon",
        "contribution_mm": 6.32,
        "category": "regime_probability",
        "description": "Synoptic regime classified as ACTIVE_MONSOON with 75.0% confidence."
      }
    ],
    "feature_contributions": {
      "raw_nwp_precip": 49.0,
      "error_memory_lag3": 7.2,
      "regime_prob_active_monsoon": 6.32
    },
    "user_friendly_summary": "Pune forecast: 56.2 mm expected under ACTIVE_MONSOON regime. Adjusted upward from 49.0 mm NWP baseline driven by persistent recent underprediction offset.",
    "model_version": "v1.0.0-prob",
    "explanation_version": "v1.0.0-shap-tree"
  },
  "metadata": {
    "model_version": "v1.0.0-prob",
    "regime_model_version": "v1.0.0-regime-lgbm",
    "feature_version": "v1.0.0-phys",
    "dataset_version": "IMD-ERA5-v2.1",
    "boundary_version": "IMD-LGD-2026.1",
    "pipeline_run_id": null
  }
}
```

---

## 3. RFC 7807 Error Contract
All API errors return a standard Problem Details JSON response:
```json
{
  "status": 404,
  "error_code": "DISTRICT_NOT_FOUND",
  "message": "District with ID 'MH_UNKNOWN' not found in official boundary dataset (IMD-LGD-2026.1).",
  "timestamp": "2026-09-28T22:27:18Z"
}
```

Standard Error Codes:
- `400 INVALID_FORECAST_PARAMETERS`: Illegal coordinates, invalid bounding box range, or unsupported lead time.
- `404 DISTRICT_NOT_FOUND`: District code does not exist in the administrative boundary dataset.
- `404 FORECAST_UNAVAILABLE`: District exists, but no verified forecast cycle is currently available.
- `503 FORECAST_SERVICE_UNAVAILABLE`: Internal storage or upstream model service unavailable.

---

## 4. Status Life Cycle
* `VALID`: All physical and probabilistic checks passed (monotonic $P_{50} \le P_{75} \le P_{90}$, non-negative rainfall, normalized regime probability vector $\sum P \approx 1.0$).
* `PARTIAL`: Forecast published with missing spatial/temporal coverage explicitly flagged.
* `STALE`: Forecast cycle age exceeds configured freshness threshold (>48h).
* `INVALID`: Verification checks failed (e.g. quantile crossing or out-of-bounds coordinates). **Never exposed as normal successful responses.**
* `UNAVAILABLE`: No records exist matching query criteria.
