# Interface Contract: Backend API Service (BACKEND -> FRONTEND)

**Contract ID:** CONTRACT-API-003  
**Version:** 1.0.0  
**Owner:** Team C (Backend Engineering)  
**Consumer:** Team D (Frontend / UI)  
**Status:** Frozen for Foundation Phase  
**Base URL:** `/api/v1`  

---

## 1. Overview & General Standards

- **Protocol:** HTTP/1.1 or HTTP/2 over HTTPS (HTTP for local development).
- **Format:** Strict JSON (`application/json`) or GeoJSON (`application/geo+json`).
- **Date/Time:** ISO 8601 strings in UTC (`YYYY-MM-DDTHH:MM:SSZ`).
- **Error Standard:** RFC 7807 Problem Details compliant JSON with fields: `status`, `error_code`, `message`, `timestamp`.

---

## 2. API Endpoints

### 2.1 System Health & Metadata
- **`GET /api/v1/health`**
  - **Purpose:** Liveness and readiness probe.
  - **Response 200 OK:**
    ```json
    {
      "status": "healthy",
      "version": "1.0.0",
      "environment": "development",
      "latest_available_cycle": "2026-07-15"
    }
    ```

---

### 2.2 Latest Cycle Forecast Summary
- **`GET /api/v1/forecast/latest`**
  - **Query Parameters:**
    - `lead_time` (optional, integer, default: `24`): `24 | 48 | 72 | 96 | 120`
  - **Response 200 OK:**
    ```json
    {
      "cycle_date": "2026-07-15",
      "lead_time_hours": 24,
      "valid_time_utc": "2026-07-16T03:00:00Z",
      "active_synoptic_regime": "ACTIVE_MONSOON",
      "regime_confidence": 0.88,
      "grid_summary": {
        "min_corrected_mm": 0.0,
        "max_corrected_mm": 182.4,
        "mean_corrected_mm": 18.6,
        "heavy_rain_points_count": 42
      }
    }
    ```

---

### 2.3 Grid Point Forecast (Point Query)
- **`GET /api/v1/forecast/grid`**
  - **Query Parameters:**
    - `lat` (required, float, e.g. `18.5204`): Latitude
    - `lon` (required, float, e.g. `73.8567`): Longitude
    - `cycle_date` (optional, string, e.g. `2026-07-15`)
  - **Response 200 OK:**
    ```json
    {
      "query_coords": { "lat": 18.52, "lon": 73.86 },
      "nearest_grid_coords": { "lat": 18.50, "lon": 73.75 },
      "cycle_date": "2026-07-15",
      "time_series": [
        {
          "lead_time": 24,
          "valid_date": "2026-07-16",
          "raw_nwp_mm": 45.2,
          "corrected_p10_mm": 38.0,
          "corrected_p50_mm": 52.4,
          "corrected_p90_mm": 68.1,
          "delta_mm": 7.2,
          "prob_heavy_rain": 0.35,
          "prob_very_heavy_rain": 0.08,
          "active_regime": "ACTIVE_MONSOON"
        }
      ]
    }
    ```
  - **Response 400 Bad Request:** Coordinates out of India domain.

---

### 2.4 District Administrative Forecast (GeoJSON Layer)
- **`GET /api/v1/forecast/districts`**
  - **Query Parameters:**
    - `lead_time` (optional, integer, default: `24`)
    - `state` (optional, string, e.g. `Maharashtra`)
  - **Response 200 OK:** `FeatureCollection` (GeoJSON)
    ```json
    {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "id": "MH_PUNE",
          "properties": {
            "district_id": "MH_PUNE",
            "district_name": "Pune",
            "state_name": "Maharashtra",
            "lead_time": 24,
            "mean_rainfall_mm": 38.4,
            "max_rainfall_mm": 112.5,
            "prob_heavy_rain": 0.65,
            "prob_very_heavy_rain": 0.22,
            "warning_level": "ORANGE",
            "active_regime": "ACTIVE_MONSOON"
          },
          "geometry": {
            "type": "Polygon",
            "coordinates": [...]
          }
        }
      ]
    }
    ```

---

### 2.5 Verification & Accuracy Metrics
- **`GET /api/v1/verification/summary`**
  - **Query Parameters:**
    - `season` (optional, string, default: `monsoon_2026`)
  - **Response 200 OK:**
    ```json
    {
      "evaluation_period": "2026-06-01 to 2026-09-30",
      "metrics": {
        "raw_nwp_rmse": 24.8,
        "corrected_rmse": 16.2,
        "rmse_improvement_pct": 34.68,
        "raw_nwp_mae": 14.3,
        "corrected_mae": 9.7,
        "heavy_rain_csi_raw": 0.28,
        "heavy_rain_csi_corrected": 0.44,
        "crps_raw": 11.2,
        "crps_corrected": 7.4
      }
    }
    ```

---

### 2.6 Model Transparency & Explainability
- **`GET /api/v1/explainability/summary`**
  - **Response 200 OK:**
    ```json
    {
      "cycle_date": "2026-07-15",
      "top_contributing_features": [
        { "feature": "error_memory_lag1", "mean_abs_shap": 4.82, "description": "Recent 24h NWP bias memory" },
        { "feature": "dem_slope", "mean_abs_shap": 3.41, "description": "Topographic orographic uplift" },
        { "feature": "rh850", "mean_abs_shap": 2.95, "description": "Low-level moisture availability" },
        { "feature": "raw_nwp_precip", "mean_abs_shap": 2.10, "description": "Raw NWP baseline prediction" }
      ]
    }
    ```

---

## 3. Warning Level Categorical Mapping (IMD Standard)

| Alert Level | Condition Criteria | Color Code | Action Required |
| :--- | :--- | :--- | :--- |
| **GREEN** | $P(\text{Heavy Rain}) < 0.25$ and Max $< 64.5\text{ mm}$ | `#22c55e` | No advisory / Normal monitoring |
| **YELLOW** | $0.25 \le P(\text{Heavy Rain}) < 0.50$ OR $64.5 \le \text{Max} < 115.6\text{ mm}$ | `#eab308` | Watch & stay updated |
| **ORANGE** | $0.50 \le P(\text{Heavy Rain}) < 0.75$ OR $115.6 \le \text{Max} < 204.5\text{ mm}$ | `#f97316` | Be prepared / Action alerts |
| **RED** | $P(\text{Heavy Rain}) \ge 0.75$ OR $\text{Max} \ge 204.5\text{ mm}$ | `#ef4444` | Take action / Severe warning |
