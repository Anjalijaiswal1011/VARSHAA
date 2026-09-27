# Master System Architecture — RAIN-REPAIR X (VARSHAA)

**Status:** Approved  
**Version:** 1.0.0  
**Domain:** Meteorological AI Post-Processing & Operational Decision Support  

---

## 1. Architectural Philosophy

The RAIN-REPAIR X system follows a **Modular Monorepo / Layered Pipeline Architecture**. 
Rather than fragmenting the project into complex distributed microservices prematurely, we decouple components through **strict typed interfaces, data contracts, and filesystem/API boundaries**. This allows independent development by specialized sub-teams (Data, ML, Backend, Frontend, GIS) while keeping local testing, execution, and deployment streamlined.

---

## 2. End-to-End Master Dataflow Pipeline

```
+-------------------------------------------------------------------------------------------------------------------------+
| [1. DATA SOURCES]                                                                                                       |
|  - IMD Gridded Rainfall (0.25°/0.1°)    - Global/Regional NWP (GFS / NCUM / ECMWF)                                      |
|  - SRTM / Copernicus Topography (DEM)   - Historical Climatology & Reanalysis (ERA5/IMD)                                |
+-------------------------------------------------------------------------------------------------------------------------+
                                                                |
                                                                v
+-------------------------------------------------------------------------------------------------------------------------+
| [2. DATA INGESTION] (`src/data/ingest.py`)                                                                              |
|  - Pull / Load raw forecast & observation files (NetCDF4, GRIB2, CSV, GeoTIFF)                                          |
|  - Stored immutably in `data/raw/{source}/{cycle_date}/`                                                                |
+-------------------------------------------------------------------------------------------------------------------------+
                                                                |
                                                                v
+-------------------------------------------------------------------------------------------------------------------------+
| [3. DATA VALIDATION & STANDARDIZATION] (`src/data/validate.py`, `src/data/standardize.py`)                              |
|  - Coordinate integrity verification (EPSG:4326, Lat: 6°-38°N, Lon: 68°-98°E)                                           |
|  - Temporal accumulation alignment (03:00 to 03:00 UTC 24h totals)                                                      |
|  - Missing coordinate/variable check; NaN bounds thresholding                                                           |
|  - Validated grids stored in `data/interim/`                                                                             |
+-------------------------------------------------------------------------------------------------------------------------+
                                                                |
                                                                v
+-------------------------------------------------------------------------------------------------------------------------+
| [4. FEATURE PIPELINE] (`src/features/`)                                                                                 |
|  - Atmospheric Features: Relative Humidity, Wind Shear, Vorticity, MSLP Gradients                                       |
|  - Topographic Features: Elevation, Slope, Aspect, Distance to Coast, Ridge Barriers                                   |
|  - Climatological Baseline: Day-of-year mean, 90th percentile baseline                                                   |
|  - Recent Error Memory: Lag 1-to-7 day NWP minus Observation bias buffer                                                |
|  - Outputs written to `data/processed/{cycle_date}/features.parquet` (or `.nc`)                                         |
+-------------------------------------------------------------------------------------------------------------------------+
                                                                |
                                                                v
+-------------------------------------------------------------------------------------------------------------------------+
| [5. ML & REGIME PIPELINE] (`src/regime/`, `src/models/`)                                                                |
|  - 5A: Regime Classifier (`src/regime/classifier.py`): Synoptic classification (Active, Break, Low Pressure, Normal)   |
|  - 5B: Regime-Conditioned ML Core (`src/models/train.py`, `src/models/predict.py`):                                      |
|       * Gradient Boosted Decision Trees / Quantile Regressors (LightGBM/XGBoost)                                        |
|       * Quantile outputs: P10, P50 (median), P75, P90, P95                                                              |
|       * Extreme Rainfall Threshold Probabilities: P(Rain >= 64.5mm), P(Rain >= 115.6mm), P(Rain >= 204.5mm)            |
|  - 5C: Explainability (`src/models/explain.py`): SHAP attributions, Regime weight contributions                         |
+-------------------------------------------------------------------------------------------------------------------------+
                                                                |
                                                                v
+-------------------------------------------------------------------------------------------------------------------------+
| [6. POST-PROCESSING & PHYSICAL CONSTRAINTS] (`src/postprocessing/constraints.py`)                                      |
|  - Physical realizability verification (Rainfall >= 0.0 mm; realistic physical upper bound filters)                     |
|  - Quantile crossing prevention (P10 <= P50 <= P75 <= P90 <= P95)                                                        |
+-------------------------------------------------------------------------------------------------------------------------+
                                                                |
                                                                v
+-------------------------------------------------------------------------------------------------------------------------+
| [7. VALIDATION ENGINE] (`src/validation/metrics.py`)                                                                    |
|  - Deterministic metrics: RMSE, MAE, Mean Bias, Threat Score (TS), Critical Success Index (CSI), False Alarm Ratio (FAR)|
|  - Probabilistic metrics: Continuous Ranked Probability Score (CRPS), Brier Score (BS), Reliability Diagrams             |
|  - Verification logs saved to `data/processed/{cycle_date}/verification_summary.json`                                   |
+-------------------------------------------------------------------------------------------------------------------------+
                                                                |
                                                                v
+-------------------------------------------------------------------------------------------------------------------------+
| [8. OUTPUT GENERATION & GIS AGGREGATION] (`src/postprocessing/district.py`)                                             |
|  - Spatial vector intersection: Regridded predictions x District Boundary GeoJSON                                       |
|  - District metrics: Area-weighted Mean, District Max, Probability of Heavy Rain (>64.5mm)                             |
|  - IMD color code alert assigned (Green, Yellow, Orange, Red)                                                           |
|  - Final output JSON/GeoJSON written to serving cache / DB                                                              |
+-------------------------------------------------------------------------------------------------------------------------+
                                                                |
                                                                v
+-------------------------------------------------------------------------------------------------------------------------+
| [9. BACKEND API SERVICE] (`backend/`)                                                                                   |
|  - High-performance FastAPI server                                                                                      |
|  - Endpoints:                                                                                                           |
|      /api/v1/forecast/latest                                                                                            |
|      /api/v1/forecast/grid?lat=...&lon=...                                                                              |
|      /api/v1/forecast/district/{district_id}                                                                            |
|      /api/v1/verification/summary                                                                                       |
|      /api/v1/explainability/feature-importance                                                                          |
+-------------------------------------------------------------------------------------------------------------------------+
                                                                |
                                                                v
+-------------------------------------------------------------------------------------------------------------------------+
| [10. WEB APPLICATION] (`frontend/`)                                                                                     |
|  - Interactive Dashboard (Vite + React / Modern Vanilla Web Components)                                                |
|  - Leaflet / MapLibre GIS choropleth & raster overlays                                                                  |
|  - Raw NWP vs. Corrected AI Comparison slider                                                                           |
|  - District alert status board & Meteogram time-series visualization                                                     |
+-------------------------------------------------------------------------------------------------------------------------+
                                                                |
                                                                v
+-------------------------------------------------------------------------------------------------------------------------+
| [11. OBSERVABILITY & MONITORING] (`src/utils/logging.py`, `src/utils/health.py`)                                        |
|  - Pipeline runtime, stage latency, data drift, and null rate logging                                                   |
|  - Standardized structured logs & health check endpoints                                                                 |
+-------------------------------------------------------------------------------------------------------------------------+
```

---

## 3. Subsystem Breakdown

### 3.1 Data Layer (`src/data/`)
- Pure I/O and spatial-temporal transformation functions.
- Independent of machine learning algorithms.
- Validates data against `data-contract.md` before passing to downstreams.

### 3.2 Feature & Regime Layer (`src/features/`, `src/regime/`)
- Computes meteorological indices, synoptic classifications, and temporal lag errors.
- Ensures all transformations use strictly past/current data (zero future data leakage).

### 3.3 ML Modeling Layer (`src/models/`)
- Encapsulates model training, hyperparameter configuration, model persistence, and batch inference.
- Emits standardized outputs compliant with `ml-contract.md`.

### 3.4 Spatial & Post-Processing Layer (`src/postprocessing/`)
- Translates numerical tensors into actionable administrative risk products.
- Applies spatial zonal operations against Indian district boundary shapefiles.

### 3.5 Dissemination Layer (`backend/` & `frontend/`)
- Decoupled client-server interaction via HTTP REST APIs.
- The web app never accesses disk datasets directly, only consuming clean JSON/GeoJSON.
