# VARSHAA — RAIN-REPAIR X

> **Regime-Aware AI Post-Processing of Indian Monsoon Rainfall Forecasts**

RAAP-X is a regime-aware AI post-processing system for improving Indian monsoon rainfall forecasts using NWP data, atmospheric features, terrain, recent forecast-error memory, probabilistic regime routing, quantile prediction, uncertainty estimation, and GIS-based district-level forecasting.

---

## 1. Project Overview & Architecture

Numerical Weather Prediction (NWP) forecasts (IMD GFS, NCMRWF NCUM, ECMWF) often suffer from structural biases, light-rain drizzle bias, orographic displacement, and under-prediction of convective extreme events during the Indian Monsoon. **RAIN-REPAIR X** constructs a domain-informed AI post-processing pipeline that:

1. **Ingests & Regrids** multi-lead NWP forecasts and observation fields to a standard 0.25° coordinate grid.
2. **Classifies Synoptic Regimes** (Active/Break monsoon, low-pressure depressions, offshore troughs) to dynamically condition model behavior.
3. **Engineers Meteorological & Topographic Features** incorporating DEM orography, moisture flux convergence, and rolling forecast-error memory.
4. **Predicts Quantiles & Calibrated Probabilities** ($P_{10}, P_{50}, P_{75}, P_{90}, P_{95}$) and threshold exceedance probabilities for IMD Heavy ($\ge 64.5$ mm), Very Heavy ($\ge 115.6$ mm), and Extremely Heavy ($\ge 204.5$ mm) rainfall.
5. **Aggregates to Administrative Districts** using zonal geometry to emit actionable early warnings (Green / Yellow / Orange / Red).
6. **Serves Predictions & Transparency Metrics** via a FastAPI backend and interactive web GIS dashboard.

---

## 2. Repository Layout

```text
├── README.md                     # Project overview and quickstart
├── LICENSE                       # MIT License
├── .gitignore                    # Comprehensive meteorological & environment ignores
├── .env.example                  # Environment configuration template
├── requirements.txt              # Top-level pip installer pointer
├── requirements/                 # Layered dependency specifications (base, data, ml, api, dev)
├── PROJECT_AUDIT.md              # Initial repository audit report
├── REQUIREMENTS.md               # Functional & Non-Functional requirement baseline
├── SYSTEM_BOUNDARY.md            # System scope and external boundaries
├── DEVELOPER_SETUP.md            # New developer onboarding and run guide
├── MASTER_TODO.md                # Comprehensive cross-team milestone task board
├── CONTRIBUTING.md               # Coding standards and contribution guidelines
│
├── docs/                         # Architecture, contracts, git workflow, and guides
│   ├── architecture/             # Master system and dataflow documentation
│   ├── contracts/                # Typed data, ML, and API interface contracts
│   ├── git-workflow.md           # Branching and review conventions
│   └── security-baseline.md      # Security and secrets management policies
│
├── data/                         # Data storage (git-ignored, partitioned)
│   ├── raw/                      # Immutable raw NetCDF/GRIB/CSV files
│   ├── interim/                  # Validated, regridded intermediate grids
│   ├── processed/                # Normalized feature matrices and tabular inputs
│   └── external/                 # Static DEM topography and district GeoJSON
│
├── src/                          # Core Python application package
│   ├── data/                     # Ingestion, validation, and regridding
│   ├── features/                 # Physical indices, error memory, and terrain features
│   ├── regime/                   # Synoptic monsoon regime classifier
│   ├── models/                   # Quantile models, training, inference, and SHAP explainability
│   ├── postprocessing/           # Physical bounds enforcement and district GIS aggregation
│   ├── validation/               # Deterministic and probabilistic verification engine
│   └── utils/                    # Config management, structured logging, custom exceptions
│
├── backend/                      # FastAPI REST service and route handlers
├── frontend/                     # Interactive React/Vite GIS forecaster dashboard
├── notebooks/                    # Exploratory prototypes and EDA notebooks
├── configs/                      # Base system and logging YAML configurations
├── scripts/                      # Operational pipeline scripts and orchestration
├── tests/                        # Comprehensive test suite (unit, integration, data_tests)
└── deployment/                   # Dockerfiles and deployment manifests
```

---

## 3. Quick Links

- [System Requirements Specification](REQUIREMENTS.md)
- [System Boundary Definition](SYSTEM_BOUNDARY.md)
- [Master System Architecture](docs/architecture/system-architecture.md)
- [Team Work Responsibilities](docs/team-responsibilities.md)
- [Interface Contracts](docs/contracts/)
  - [Data Contract](docs/contracts/data-contract.md)
  - [ML Contract](docs/contracts/ml-contract.md)
  - [API Contract](docs/contracts/api-contract.md)
- [Developer Setup & Onboarding Guide](DEVELOPER_SETUP.md)
- [Master Roadmap & Task Board](MASTER_TODO.md)
- [Contributing Guidelines & Standards](CONTRIBUTING.md)
