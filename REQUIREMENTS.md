# System Requirements Specification — RAIN-REPAIR X (VARSHAA)

**Project:** Regime-Aware AI Post-Processing of Indian Monsoon Rainfall Forecasts  
**Version:** 1.0.0-draft  
**Status:** Baseline Approved  
**Author:** Tech Lead & Solutions Architect  

---

## 1. Introduction & Mission Statement

Numerical Weather Prediction (NWP) models (e.g., IMD GFS, NCMRWF NCUM, ECMWF) exhibit systematic biases, spatial displacement errors, and regime-dependent over/under-forecasting across the Indian subcontinent during the southwest and northeast monsoon seasons. 

**RAIN-REPAIR X (VARSHAA)** is designed as an operational, regime-aware AI/ML post-processing pipeline that:
1. Ingests raw NWP forecasts and meteorological observations.
2. Identifies active synoptic and mesoscale weather regimes (e.g., Active/Break monsoon phases, Monsoon Depressions, Western Disturbances, Offshore Troughs).
3. Applies machine learning corrections conditioned on regime context, spatial topography, climatology, and recent forecast error memory.
4. Generates both deterministic and calibrated probabilistic rainfall forecasts (quantiles, heavy/extreme rainfall thresholds) at high-resolution grid and district administrative levels.
5. Exposes explainability metrics and verification analytics via an open API and interactive dashboard.

---

## 2. Functional Requirements (FR)

### FR-01: NWP Forecast Ingestion
- **FR-01.1:** The system shall ingest multi-lead-time Numerical Weather Prediction forecasts (e.g., Day 1 to Day 5 / 24h to 120h lead times) in standard meteorological formats (NetCDF4, GRIB2, or converted tabular/GeoTIFF).
- **FR-01.2:** Ingested variables shall include at minimum total precipitation, surface temperature, humidity, mean sea-level pressure, and zonal/meridional wind vectors at standard pressure levels (850 hPa, 500 hPa, 200 hPa).
- **FR-01.3:** The system must validate spatial grids and temporal alignment upon ingestion, rejecting corrupted or incomplete forecast cycles.

### FR-02: Observational & Reanalysis Data Ingestion
- **FR-02.1:** The system shall ingest ground-truth observational datasets (e.g., IMD gridded rainfall 0.25°/0.1°, AWS/ARG station observations, or GPM IMERG satellite estimates) for model training, recent error memory, and verification.
- **FR-02.2:** Observation time stamps must align precisely with NWP accumulation windows (e.g., 03:00 UTC to 03:00 UTC 24-hour accumulations).

### FR-03: Static & Climatological Feature Ingestion
- **FR-03.1:** The system shall ingest high-resolution digital elevation model (DEM) data (or derived slope/aspect/topographic position index) to account for orographic enhancement (e.g., Western Ghats, Northeast Himalayas).
- **FR-03.2:** The system shall incorporate historical rainfall climatology (e.g., day-of-year mean, 90th percentile historical baseline) to ground predictions in historical distribution limits.

### FR-04: Recent Forecast-Error Memory
- **FR-04.1:** The system shall compute and maintain a rolling forecast error memory (e.g., NWP minus Observation over past 1 to 7 days) to capture persistent model drift, bias trends, and regional over-estimation.

### FR-05: Weather Regime Identification
- **FR-05.1:** The system shall classify large-scale and regional meteorological states into distinct monsoon regimes (e.g., Active Monsoon, Break Monsoon, Low Pressure Area/Depression, Normal/Transitional) using objective atmospheric indices (monsoon trough position, low-level jet strength, wind shear).
- **FR-05.2:** Regime classification may be discrete (categorical state) or probabilistic (soft assignment weights across regimes).

### FR-06: Post-Processing & Rainfall Correction
- **FR-06.1:** The system shall apply AI/ML correction algorithms conditioned on the identified regime, recent error memory, and topographic features.
- **FR-06.2:** The model must mitigate common NWP pathologies (e.g., "drizzle bias" / over-forecasting light rain, and under-forecasting extreme convective bursts).

### FR-07: Probabilistic & Extreme Rainfall Forecasting
- **FR-07.1:** The system shall generate probabilistic rainfall distributions (e.g., calibrated quantiles: P10, P50, P75, P90, P95).
- **FR-07.2:** The system shall compute exceedance probabilities for standard IMD operational thresholds:
  - Heavy Rainfall: $\ge 64.5\text{ mm/day}$
  - Very Heavy Rainfall: $\ge 115.6\text{ mm/day}$
  - Extremely Heavy Rainfall: $\ge 204.5\text{ mm/day}$

### FR-08: Spatial Aggregation & District-Level Mapping
- **FR-08.1:** The system shall project corrected grid forecasts onto official Indian district and sub-division boundary polygons (GeoJSON/Shapefile).
- **FR-08.2:** The system shall compute district-level summary metrics: spatial mean, maximum, area-weighted precipitation, and categorical IMD warning color code (Green, Yellow, Orange, Red).

### FR-09: Forecast Verification & Operational Metrics
- **FR-09.1:** The system shall continuously compute deterministic verification metrics against ground truth: Root Mean Square Error (RMSE), Mean Absolute Error (MAE), Mean Bias Error (MBE), Critical Success Index (CSI), Threat Score (TS), and False Alarm Ratio (FAR).
- **FR-09.2:** The system shall compute probabilistic verification metrics: Continuous Ranked Probability Score (CRPS), Brier Score (BS), and Reliability Diagrams.

### FR-10: Explainability & Diagnostic Attribution
- **FR-10.1:** The system shall generate feature attribution metrics (e.g., SHAP values, regime routing weights) explaining why a specific raw NWP forecast was corrected up or down.

### FR-11: API & Data Dissemination
- **FR-11.1:** The system shall expose a structured REST API (FastAPI) serving point forecasts, district summaries, raster/grid layers, and historical verification statistics in JSON/GeoJSON.
- **FR-11.2:** API endpoints shall be fully documented using OpenAPI / Swagger specifications.

### FR-12: Operational Web Dashboard
- **FR-12.1:** The web platform shall provide an interactive map viewer displaying raw NWP vs. AI-corrected rainfall layers, district risk alerts, time-series meteograms, and verification cards.

---

## 3. Non-Functional Requirements (NFR)

### NFR-01: Reliability & Fault Tolerance
- Ingestion failures or missing non-critical features (e.g., missing day-3 satellite product) must trigger graceful fallback (e.g., fall back to climatology + raw NWP without crashing the pipeline).
- Pipeline runs must be idempotent; rerunning an identical cycle must yield identical results without duplicate record creation.

### NFR-02: Performance & Latency
- Daily operational inference run for the entire pan-India domain ($\sim 0.25^\circ$ or defined sub-region) must complete within $\le 5\text{ minutes}$ on standard GPU/CPU compute nodes.
- API response times for district forecast queries must be $\le 200\text{ ms}$ for 95th percentile requests ($P95$).

### NFR-03: Scalability
- The architecture must support expansion from regional pilot domains (e.g., Maharashtra or Western Ghats) to pan-India grid scale without architectural refactoring.
- Storage must cleanly partition raw, interim, and processed artifacts by forecast cycle date (`YYYYMMDD`).

### NFR-04: Maintainability & Modularity
- Pipeline stages (Ingestion $\rightarrow$ Feature Engineering $\rightarrow$ Regime ID $\rightarrow$ ML Model $\rightarrow$ District Aggregation) must be decoupled via typed interface contracts.
- Codebase must maintain $\ge 80\%$ test coverage on core utility and processing modules.

### NFR-05: Reproducibility
- All model training runs must log exact Git commit hash, dataset version tag, configuration YAML hash, random seed, and serialized model artifact.
- Pipeline execution must produce reproducible results given the same seed and input data.

### NFR-06: Security & Secrets Management
- Zero hardcoded credentials; all secrets (database URLs, API tokens) must be injected strictly via environment variables (`.env`).
- API endpoints must enforce input validation, rate limiting, and CORS restrictions.

### NFR-07: Explainability & Transparency
- Post-processed outputs must never be an uninterpretable "black box". Operational forecasters must have access to regime classification rationale and error correction deltas ($\Delta \text{Rainfall} = \text{Corrected} - \text{Raw}$).

### NFR-08: Observability & Logging
- Structured JSON logging must be implemented across all ingestion, feature transformation, inference, and API requests.
- System metrics (execution duration, memory consumption, missing value percentages) must be logged per cycle.

---

## 4. Traceability Matrix

| Requirement | Pipeline Stage | Responsible Module | Verification Strategy |
| :--- | :--- | :--- | :--- |
| FR-01, FR-02, FR-03 | Data Ingestion | `src/data/` | Unit tests on netCDF/CSV parsing, spatial bounds validation |
| FR-04, FR-05 | Feature Engineering | `src/features/`, `src/regime/` | Schema validation, statistical distribution tests |
| FR-06, FR-07 | ML Post-Processing | `src/models/`, `src/postprocessing/` | Inference integration tests, contract compliance checks |
| FR-08 | Spatial Aggregation | `src/postprocessing/district.py` | Polygon intersection tests, boundary integrity checks |
| FR-09, FR-10 | Validation & XAI | `src/validation/`, `src/models/explain.py` | Metric equation tests, SHAP summary validation |
| FR-11 | API Dissemination | `backend/` | Automated OpenAPI schema tests, HTTP route integration tests |
| FR-12 | Web Platform | `frontend/` | UI component tests, API response parsing tests |
