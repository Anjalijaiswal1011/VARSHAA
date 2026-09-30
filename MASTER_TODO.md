# Master Engineering Task Board — RAIN-REPAIR X (VARSHAA)

**Status:** Active  
**Governance:** Tech Lead / Senior Architect  
**Legend:**  
- `[x]` Completed & Verified  
- `[ ]` Pending Execution / Backlog  

---

## P0 — Engineering Foundation (PART 0)

| Task ID | Task Description | Owner | Dependency | Status | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **P0-01** | Repository Audit & Classification | Tech Lead | None | `[x]` | `PROJECT_AUDIT.md` completed without deleting existing work |
| **P0-02** | Requirement Baseline Specification | Tech Lead | P0-01 | `[x]` | `REQUIREMENTS.md` with Functional & Non-Functional requirements |
| **P0-03** | System Boundary & Scope Definition | Tech Lead | P0-02 | `[x]` | `SYSTEM_BOUNDARY.md` defining in-scope vs. out-of-scope |
| **P0-04** | Master System Architecture Design | Tech Lead | P0-03 | `[x]` | `docs/architecture/system-architecture.md` dataflow diagram |
| **P0-05** | Repository Directory Structure & Scaffolding | Tech Lead | P0-04 | `[x]` | Full folder hierarchy, `.gitignore`, `.env.example`, `LICENSE`, requirements |
| **P0-06** | Team Work Separation & Ownership Matrix | Tech Lead | P0-05 | `[x]` | `docs/team-responsibilities.md` for 7 sub-teams |
| **P0-07** | Interface Contracts Definition | Tech Lead | P0-06 | `[x]` | `docs/contracts/` (Data, ML, and API contracts frozen) |
| **P0-08** | Data & Model Versioning Strategy | Tech Lead | P0-07 | `[x]` | `docs/versioning-strategy.md` defining traceability equation |
| **P0-09** | Configuration Management Engine | Tech Lead | P0-05 | `[x]` | `configs/base_config.yaml` & `src/utils/config.py` loaded |
| **P0-10** | Coding Standards & Contributing Guide | Tech Lead | P0-05 | `[x]` | `CONTRIBUTING.md` with Python & JS standards |
| **P0-11** | Git Collaboration & Branching Workflow | Tech Lead | P0-05 | `[x]` | `docs/git-workflow.md` documenting branch lifecycle |
| **P0-12** | Foundation Testing Suite & Fixtures | Team G | P0-09 | `[x]` | Pytest passing unit, integration, and contract tests |
| **P0-13** | Structured Logging & Domain Exceptions | Tech Lead | P0-09 | `[x]` | `src/utils/logging.py` & `src/utils/exceptions.py` implemented |
| **P0-14** | Security Baseline & Secret Protection | Tech Lead | P0-09 | `[x]` | `docs/security-baseline.md` documented |
| **P0-15** | Senior Developer Architecture Review | Tech Lead | P0-01..14 | `[x]` | `docs/senior-developer-review.md` audit completed |
| **P0-16** | Foundation Quality Gate Audit | Tech Lead | P0-15 | `[x]` | `docs/architecture-quality-gate.md` scored $\ge 9.0/10$ |
| **P0-17** | Developer Onboarding Guide | Tech Lead | P0-05 | `[x]` | `DEVELOPER_SETUP.md` with 10 clear onboarding sections |
| **P0-18** | Master Roadmap & Task Board Setup | Tech Lead | All P0 | `[x]` | `MASTER_TODO.md` tracking all stages P0 to P9 |

---

## P1 — Data Engineering & Pipelines

| Task ID | Task Description | Owner | Dependency | Status | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **P1-01** | Raw NWP Ingestion Parser (NetCDF4/GRIB2) | Team A | P0-07 | `[x]` | Ingests multi-lead GFS/NCUM forecasts into memory |
| **P1-02** | Observation Ingestion Parser (IMD/GPM) | Team A | P0-07 | `[x]` | Ingests ground-truth observations on 0.25° grid |
| **P1-03** | Spatial Regridding Engine (0.25° EPSG:4326) | Team A | P1-01, P1-02 | `[x]` | Interpolates inconsistent grids to standard 0.25° grid |
| **P1-04** | Temporal Alignment & Accumulation Aggregator | Team A | P1-03 | `[x]` | Aggregates 24h totals from 03:00 to 03:00 UTC |
| **P1-05** | Missing Data & Outlier Quality Control | Team A | P1-04 | `[x]` | Replaces invalid sentinels with NaN; validates thresholds |
| **P1-06** | Data Contract Automated Validator | Team A, G | P1-05 | `[x]` | Automated validation of `data/interim/` against CONTRACT-DATA-001 |

---

## P2 — ML & Feature Engineering

| Task ID | Task Description | Owner | Dependency | Status | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **P2-01** | DEM Topography & Slope Feature Extraction | Team B | P1-06 | `[x]` | Computes elevation, slope, and aspect grids |
| **P2-02** | Rolling Forecast-Error Memory Buffer | Team B | P1-06 | `[x]` | Computes 1-to-14 day lag error matrices ($NWP - Obs$) with causal guardrails |
| **P2-03** | Atmospheric Dynamics Features (Shear, Vorticity) | Team B | P1-06 | `[x]` | Extracts moisture flux convergence, U/V wind shear, and vorticity |
| **P2-04** | Monsoon Synoptic Regime Classifier | Team B | P2-03 | `[x]` | Classifies Active, Break, Depression, and Normal states |
| **P2-05** | Baseline Deterministic Post-Processing Regressor | Team B | P2-04 | `[x]` | Trained LightGBM/XGBoost baseline model |
| **P2-06** | Quantile Regression Engine (P10, P50, P75, P90, P95) | Team B | P2-05 | `[x]` | Multi-quantile models with pinball loss optimization |
| **P2-07** | Extreme Rainfall Threshold Classifier | Team B | P2-05 | `[x]` | Predicts probabilities for $\ge 64.5$ mm, $\ge 115.6$ mm, $\ge 204.5$ mm |
| **P2-08** | Physical Constraints & Monotonicity Sorter | Team B | P2-06 | `[x]` | Enforces non-negativity and prevents quantile crossing |
| **P2-09** | Feature Attribution & Explainability (SHAP) | Team B | P2-06 | `[x]` | Emits local and global feature importance attributions |

---

## P3 — Backend API Engineering

| Task ID | Task Description | Owner | Dependency | Status | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **P3-01** | FastAPI Core Server & Routing Skeleton | Team C | P0-07 | `[x]` | Server boots with `/health` and OpenAPI docs |
| **P3-02** | Latest Forecast Retrieval Endpoint | Team C | P3-01, P2-08 | `[x]` | `/api/v1/forecast/latest` serving validated cycle summary |
| **P3-03** | Point-Grid Timeseries Endpoint | Team C | P3-01, P2-08 | `[x]` | `/api/v1/forecast/grid` returning lead-time meteogram data |
| **P3-04** | District Administrative Forecast Endpoint | Team C | P3-01, P5-02 | `[x]` | `/api/v1/forecast/districts` returning GeoJSON alerts |
| **P3-05** | Verification Metrics Endpoint | Team C | P3-01, P7-03 | `[x]` | `/api/v1/verification/summary` returning RMSE/CSI/CRPS |
| **P3-06** | Explainability & Attribution Endpoint | Team C | P3-01, P2-09 | `[x]` | `/api/v1/explainability/summary` serving SHAP values |

---

## P4 — Frontend Operational Dashboard

| Task ID | Task Description | Owner | Dependency | Status | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **P4-01** | Frontend Scaffolding (Vite + React / UI Shell) | Team D | P0-05 | `[x]` | Responsive dashboard shell with dark/light themes (`frontend/src/App.tsx`) |
| **P4-02** | GIS Map Canvas (Leaflet/MapLibre Integration) | Team D | P4-01 | `[x]` | Interactive India map with pan/zoom and layer toggling (`IndiaForecastMap.tsx`) |
| **P4-03** | Raw vs. AI-Corrected Dual View / Slider | Team D | P4-02, P3-02 | `[x]` | Visual split-screen comparison of NWP vs. Corrected rain (`NwpVsRaapxChart.tsx`) |
| **P4-04** | District Warning Level Choropleth Overlay | Team D | P4-02, P3-04 | `[x]` | Color-coded alert overlay (Green/Yellow/Orange/Red) (`DistrictDetailPanel.tsx`) |
| **P4-05** | Station/Grid Point Meteogram Chart | Team D | P4-01, P3-03 | `[x]` | Quantile plume chart (P10-P50-P90) over 5 forecast days (`LeadTimeBarChart.tsx`) |
| **P4-06** | Model Explainability & Regime Attribution Modal | Team D | P4-01, P3-06 | `[x]` | Visual feature contribution bar charts and active regime tag (`ExplainabilityPanel.tsx`, `WhyRaapxStoryFlow.tsx`) |

---

## P5 — GIS & Spatial Aggregation

| Task ID | Task Description | Owner | Dependency | Status | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **P5-01** | District Shapefile & GeoJSON Normalization | Team E | P0-05 | `[x]` | Validated India district polygons in EPSG:4326 |
| **P5-02** | Spatial Zonal Statistics Engine | Team E | P5-01, P2-08 | `[x]` | Computes district mean, max, and exceedance probability |
| **P5-03** | IMD Alert Level Assignment Engine | Team E | P5-02 | `[x]` | Assigns Green, Yellow, Orange, Red warning codes |

---

## P6 — MLOps & CI/CD

| Task ID | Task Description | Owner | Dependency | Status | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **P6-01** | GitHub Actions Automated Lint & Test Workflow | Team F | P0-12 | `[x]` | Pull requests automatically execute pytest & ruff (`.github/workflows/ci.yml`) |
| **P6-02** | Multi-Stage Backend Dockerfile | Team F | P3-01 | `[x]` | Containerized FastAPI backend $< 500\text{ MB}$ (`deployment/Dockerfile.backend`) |
| **P6-03** | Docker Compose Full Monorepo Orchestration | Team F | P6-02, P4-01 | `[x]` | Single command `docker-compose up` launches entire stack (`docker-compose.yml`) |

---

## P7 — Quality Assurance & Verification

| Task ID | Task Description | Owner | Dependency | Status | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **P7-01** | Deterministic Verification Engine (RMSE, MAE, CSI) | Team G | P1-06, P2-08 | `[x]` | Computes standard meteorological verification metrics |
| **P7-02** | Probabilistic Verification Engine (CRPS, Brier) | Team G | P2-08 | `[x]` | Evaluates reliability diagrams and quantile calibration |
| **P7-03** | End-to-End Pipeline Regression Test | Team G | P1..P5 | `[x]` | Simulated cycle execution tests from data to API |

---

## P8 — Production Deployment

| Task ID | Task Description | Owner | Dependency | Status | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **P8-01** | Cloud VM / Server Provisioning & Storage Mount | Team F | P6-03 | `[x]` | Host environment configured with persistent storage (`docs/deployment/cloud-provisioning.md`) |
| **P8-02** | Reverse Proxy & SSL Setup (Nginx / Caddy) | Team F | P8-01 | `[x]` | Secure HTTPS endpoints with domain name routing (`deployment/nginx-ssl.conf`, `docs/deployment/ssl-reverse-proxy.md`) |

---

## P9 — SIH Demonstration & Pitch

| Task ID | Task Description | Owner | Dependency | Status | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **P9-01** | High-Impact Demonstration Case Studies (e.g. Floods) | All Teams | P1..P7 | `[x]` | Documented case studies showing AI improvement (`docs/case_studies/floods_and_extreme_weather.md`) |
| **P9-02** | Pitch Presentation Deck & Live Demo Script | Tech Lead | All Teams | `[x]` | Ready for judges with live operational walkthrough (`docs/presentation/SIH_FINAL_PITCH_DECK.md`, `LIVE_DEMO_SCRIPT.md`, `docs/slides/sih_presentation_deck.html`) |
