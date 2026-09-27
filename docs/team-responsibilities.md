# Team Responsibilities & Work Boundaries — RAIN-REPAIR X (VARSHAA)

**Status:** Approved  
**Version:** 1.0.0  
**Purpose:** Establish clear ownership, prevent merge conflicts, enforce decoupled workflows, and define formal handoff criteria across sub-teams.

---

## Team Matrix Summary

```
                      +---------------------------------------+
                      |       SENIOR DEV / TECH LEAD          |
                      | (Architecture, Review, Governance)    |
                      +---------------------------------------+
                                          |
        +------------------+--------------+---------------+------------------+
        |                  |                              |                  |
+---------------+  +---------------+              +---------------+  +---------------+
|    TEAM A     |  |    TEAM B     |              |    TEAM C     |  |    TEAM D     |
|   Data Eng    |  |    ML / AI    |              |    Backend    |  |  Frontend/UI  |
+---------------+  +---------------+              +---------------+  +---------------+
        |                  |                              |                  |
        +------------------+--------------+---------------+------------------+
                                          |
                      +-------------------+-------------------+
                      |                   |                   |
               +---------------+   +---------------+   +---------------+
               |    TEAM E     |   |    TEAM F     |   |    TEAM G     |
               |     GIS       |   | DevOps/MLOps  |   |  QA / Testing |
               +---------------+   +---------------+   +---------------+
```

---

## 1. TEAM A — Data Engineering

- **Primary Responsibility:** Ingestion of raw NWP forecasts (NetCDF/GRIB) and ground observations (IMD/GPM); spatial regridding to standard 0.25° grid; temporal alignment to 24h accumulation cycles (03:00 UTC); missing value handling; creation of intermediate datasets.
- **Inputs:** Raw data sources (`data/raw/`), data download scripts, coordinate specifications.
- **Outputs:** Validated and regridded spatial grids in `data/interim/`, ingestion metadata logs.
- **Files Owned (Exclusive):**
  - `src/data/*`
  - `scripts/download_sample_data.py`
  - `tests/data_tests/*`
- **Files They May Modify (With Review):**
  - `configs/base_config.yaml` (data paths section only)
  - `requirements/data.txt`
- **Dependencies:** None (first in pipeline).
- **Handoff Conditions to Team B (ML):**
  - Data conforms 100% to `docs/contracts/data-contract.md`.
  - All test cases in `tests/data_tests/` pass with zero coordinate or timestamp misalignments.

---

## 2. TEAM B — Machine Learning & AI

- **Primary Responsibility:** Feature extraction (vorticity, moisture convergence, rolling forecast error memory, DEM slope/elevation); monsoon regime classification; training quantile regression models (LightGBM/XGBoost); uncertainty estimation; extreme rainfall threshold probabilities; SHAP explainability.
- **Inputs:** Validated interim datasets from `data/interim/`, feature specs, target labels.
- **Outputs:** Serialized model artifacts in `models/checkpoints/`, inference predictions table/tensor conforming to ML contract, verification metrics.
- **Files Owned (Exclusive):**
  - `src/features/*`
  - `src/regime/*`
  - `src/models/*`
  - `src/validation/*`
  - `notebooks/*`
- **Files They May Modify (With Review):**
  - `requirements/ml.txt`
  - `configs/base_config.yaml` (model parameters section)
- **Dependencies:** Team A (data contract).
- **Handoff Conditions to Team C (Backend) & Team E (GIS):**
  - Inference output conforms strictly to `docs/contracts/ml-contract.md`.
  - Serialized model runs standalone inference without crashing.
  - Verification baseline scores documented in validation summaries.

---

## 3. TEAM C — Backend Engineering

- **Primary Responsibility:** FastAPI service layer implementation; REST API endpoints for forecast retrieval (point grid, district summary, time-series meteogram, verification); caching; query parameter validation; integration with post-processed inference stores.
- **Inputs:** Post-processed forecast datasets (`data/processed/` or database) conforming to ML and Post-Processing contracts.
- **Outputs:** OpenAPI / Swagger documented REST API serving JSON/GeoJSON.
- **Files Owned (Exclusive):**
  - `backend/*`
- **Files They May Modify (With Review):**
  - `requirements/api.txt`
  - `tests/integration/test_api*.py`
- **Dependencies:** Team B & Team E (output formats and district schemas).
- **Handoff Conditions to Team D (Frontend):**
  - API endpoints conform 100% to `docs/contracts/api-contract.md`.
  - Automated tests pass; Swagger docs accessible at `/docs`.

---

## 4. TEAM D — Frontend / UI Engineering

- **Primary Responsibility:** Interactive forecaster dashboard; dual-map comparison viewer (Raw NWP vs. RAIN-REPAIR X); district risk alert matrix (Green/Yellow/Orange/Red); meteogram chart visualizations; model transparency & attribution modal.
- **Inputs:** REST API endpoints from Backend (`/api/v1/*`).
- **Outputs:** Production web application build in `frontend/dist/`.
- **Files Owned (Exclusive):**
  - `frontend/*`
- **Files They May Modify (With Review):**
  - None outside `frontend/`.
- **Dependencies:** Team C (backend API endpoints).
- **Handoff Conditions to Tech Lead:**
  - Responsive, error-free UI handling API loading, empty states, and errors gracefully.
  - Zero hardcoded mock data in production builds.

---

## 5. TEAM E — GIS & Spatial Visualization

- **Primary Responsibility:** District administrative boundaries management (GeoJSON/Shapefiles); spatial intersection / zonal statistics (grid-to-district aggregation); color code alert classification according to IMD criteria; spatial projection standardization (EPSG:4326).
- **Inputs:** Grid forecasts from Team B, District shapefiles from `data/external/`.
- **Outputs:** District-aggregated forecast files (`district_forecasts.json`/`geojson`), zonal masks.
- **Files Owned (Exclusive):**
  - `src/postprocessing/district.py`
  - `src/postprocessing/constraints.py`
  - `data/external/boundaries/*`
- **Files They May Modify (With Review):**
  - `src/postprocessing/*`
- **Dependencies:** Team B (grid predictions), Team A (coordinate standard).
- **Handoff Conditions to Team C (Backend):**
  - GeoJSON attributes contain valid district IDs, names, mean/max rainfall, alert levels.

---

## 6. TEAM F — DevOps & MLOps

- **Primary Responsibility:** Containerization (Dockerfiles, docker-compose); CI/CD workflows (GitHub Actions for linting, type-checking, and tests); environment reproducible setups; artifact versioning policies.
- **Inputs:** Dockerfiles, dependency lists, CI scripts.
- **Outputs:** Passing CI builds, container images, deployment guides.
- **Files Owned (Exclusive):**
  - `.github/workflows/*`
  - `deployment/*`
- **Files They May Modify (With Review):**
  - `.gitignore`, `.env.example`
- **Dependencies:** All teams (for build scripts).
- **Handoff Conditions:**
  - One-click build via `docker-compose up` runs without errors.

---

## 7. TEAM G — QA & Testing

- **Primary Responsibility:** Automated test suite management; regression testing; boundary condition tests (zero rain, extreme 500mm rain, missing dates); test data fixtures; API schema validation.
- **Inputs:** Code modules from all teams.
- **Outputs:** Test suites in `tests/`, test execution reports, bug issues.
- **Files Owned (Exclusive):**
  - `tests/unit/*`
  - `tests/integration/*`
  - `tests/conftest.py`
- **Files They May Modify (With Review):**
  - None outside `tests/`.
- **Dependencies:** All teams.
- **Handoff Conditions:**
  - Full test suite passes under `pytest` with coverage report.

---

## 8. SENIOR DEVELOPER / TECH LEAD

- **Role & Governance:**
  - Architecture governance and contract change approvals.
  - Final reviewer for all Pull Requests into `development` and `main`.
  - Cross-team conflict resolution and dependency unblocking.
  - Verification of non-functional requirements (security, performance, reproducibility).
