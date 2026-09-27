# System Boundary Definition — RAIN-REPAIR X (VARSHAA)

**Status:** Approved  
**Version:** 1.0.0  
**Scope:** SIH 2026 Core Engineering Boundary  

---

## 1. System Scope Overview

The boundary definition prevents scope creep, delineates clear engineering responsibilities, and clarifies the separation between external dependencies, internal core pipelines, and downstream consumer applications.

```
+----------------------------------------------------------------------------------------------------+
|                                         EXTERNAL SYSTEMS                                           |
|  - IMD Gridded Rainfall Archive   - GFS/NCUM Forecast Portals   - SRTM/Copernicus DEM              |
|  - Real-time Telemetry / AWS      - Admin Boundary Repositories - Static Climatological Atlases    |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v Ingestion Gateway (ETL)
+----------------------------------------------------------------------------------------------------+
|                                           SYSTEM BOUNDARY                                          |
|                                                                                                    |
|  [1. INPUTS]                                                                                       |
|     * Raw NWP Forecasts (NetCDF4 / GRIB2 / CSV)                                                    |
|     * Ground Observations (Gridded / Station CSV)                                                  |
|     * Elevation Topography & Terrain Grids (GeoTIFF)                                               |
|     * District Boundary GeoJSON / Polygons (EPSG:4326)                                             |
|                                                                                                    |
|  [2. DATA & FEATURE PROCESSING]                                                                    |
|     * Coordinate standardizer & Spatial Regridding (Target: 0.25° standard lat/lon)                |
|     * Temporal Accumulation Aggregation (03:00 to 03:00 UTC 24h window)                            |
|     * Rolling Forecast Error Memory (Lag 1 to 7 Days Bias Buffer)                                  |
|     * Topographic & Climatological Index Extraction                                                |
|                                                                                                    |
|  [3. ML & REGIME PIPELINE]                                                                         |
|     * Synoptic & Mesoscale Weather Regime Classifier (Rule/Cluster/Supervised)                     |
|     * Regime-Conditioned ML Correction Engine (Quantile / GBDT / Probabilistic Regressor)          |
|     * Uncertainty & Extreme Rainfall Probability Estimator (64.5mm, 115.6mm, 204.5mm)              |
|     * Feature Attribution & Model Explainability Module (SHAP / Feature Impact)                    |
|                                                                                                    |
|  [4. POST-PROCESSING & VALIDATION]                                                                 |
|     * Physical Realizability Filter (Non-negativity, Climatological caps)                          |
|     * Spatial Vector Aggregation (Zonal Statistics -> District Mean/Max/Quantiles)                 |
|     * Verification Engine (RMSE, MAE, MBE, CSI, FAR, CRPS, Brier Score)                           |
|                                                                                                    |
|  [5. BACKEND API SERVICE]                                                                          |
|     * REST API Server (FastAPI)                                                                    |
|     * Forecast Query Service (Grid point, District, Time-series)                                   |
|     * GeoJSON District Layer & Alert Level Generator                                               |
|     * Verification Analytics & Metrics Endpoints                                                   |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v Dissemination
+----------------------------------------------------------------------------------------------------+
|                                     USER APPLICATIONS (CLIENTS)                                    |
|  - Operational Forecaster Dashboard (React/Vite Web App)                                           |
|  - District Disaster Management Exporters (CSV/JSON/GeoJSON Downloads)                             |
|  - Automated Alert & Notification Consumers                                                        |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. In-Scope vs. Out-of-Scope Matrix

| Domain | In-Scope (Strictly Included) | Out-of-Scope (Strictly Excluded) | Rationale |
| :--- | :--- | :--- | :--- |
| **Numerical Weather Prediction** | Ingestion of raw model outputs (GFS/NCUM/ECMWF). | Running or solving primitive hydrostatic atmospheric equations; compiling WRF or running NWP physics simulations locally. | NWP computation requires supercomputing clusters (PARAM/Pratyush). Our system post-processes existing NWP forecasts. |
| **Data Ingestion** | Reading NetCDF/GRIB/CSV, spatial regridding, temporal alignment. | Scraping proprietary websites without permission; managing physical weather stations. | System operates on open IMD/NCMRWF/GPM research data and public distribution channels. |
| **ML Post-Processing** | Tabular/spatial AI correction, regime routing, quantile estimation, uncertainty calibration. | 100-billion parameter generic Foundation LLMs; training climate models from scratch. | Targeted, domain-informed ML models (LightGBM/XGBoost/Quantile Forests/UNet) are computationally efficient, verifiable, and explainable. |
| **Spatial Scale** | Pan-India domain (Latitude: 6.0°N to 38.0°N, Longitude: 68.0°E to 98.0°E) and selected high-impact regional test beds. | Global weather forecasting; ultra-localized microclimate street-canyon CFD modeling. | Focus is aligned with Indian Monsoon dynamics and administrative disaster boundaries. |
| **Application Layer** | REST API providing JSON/GeoJSON forecasts; interactive forecaster dashboard with GIS layers. | Full ERP software; mobile apps with SMS gateways; direct siren actuation systems. | SIH demonstration demands an intuitive operational dashboard and clean API integration. |
| **Operational Infrastructure** | Docker-compatible backend and frontend architectures; reproducible local environment setup. | Multi-cloud multi-region Kubernetes clusters; multi-million dollar streaming message brokers. | Keeps architecture lean, testable, and deployable on standard workstation/cloud instances. |

---

## 3. Explicit Boundaries Between Subsystems

1. **Boundary A: Ingestion $\leftrightarrow$ Processing**
   - Ingestion downloads raw files into `data/raw/` preserving raw fidelity.
   - Processing reads `data/raw/`, validates coordinates and timestamps, and writes validated, regridded data into `data/interim/`.

2. **Boundary B: Processing $\leftrightarrow$ ML Pipeline**
   - Feature engineering generates normalized tabular/multidimensional tensors in `data/processed/`.
   - ML components consume strictly defined feature contracts and never directly parse raw GRIB2/NetCDF files.

3. **Boundary C: ML Pipeline $\leftrightarrow$ Post-Processing & GIS**
   - ML model outputs gridded predictions and probability quantiles.
   - Spatial aggregation joins grid cells with district polygon masks to produce administrative summaries.

4. **Boundary D: Post-Processing $\leftrightarrow$ Backend API**
   - Post-processed forecasts are indexed and exposed via structured service layers.
   - Backend acts as an API gateway; it does not perform raw model training or heavy data transformation on request.

5. **Boundary E: Backend API $\leftrightarrow$ Frontend**
   - Frontend is decoupled; it consumes JSON/GeoJSON over HTTP REST and renders interactive maps and charts. Frontend contains no ML logic.
