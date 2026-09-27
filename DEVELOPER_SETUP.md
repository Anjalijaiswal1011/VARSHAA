# Developer Setup & Onboarding Guide — RAIN-REPAIR X (VARSHAA)

Welcome to the **RAIN-REPAIR X (VARSHAA)** development team for SIH 2026. This guide contains everything you need to set up your local development environment and start contributing effectively.

---

## 1. Prerequisites

Ensure you have the following installed on your operating system (Windows, Linux, or macOS):
- **Python:** Version `3.10`, `3.11`, `3.12`, or `3.13`
- **Git:** Version `2.30+`
- **Node.js & npm:** Version `18.0+` (Required only for Team D Frontend)
- **Virtual Environment Tool:** `venv` (bundled with Python) or `conda`

---

## 2. Repository Setup

Clone the repository and enter the project root directory:

```bash
git clone https://github.com/Anjalijaiswal1011/VARSHAA.git
cd VARSHAA
```

---

## 3. Environment Setup

Create and activate an isolated Python virtual environment:

### Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Linux / macOS (Bash):
```bash
python3 -m venv venv
source venv/bin/activate
```

Next, initialize your local environment configuration:
```bash
# Copy template to .env
cp .env.example .env
```
*(On Windows PowerShell: `Copy-Item .env.example .env`)*

---

## 4. Dependency Installation

The repository provides layered dependency requirements tailored to different sub-teams:

```bash
# Option A: Full Developer Setup (recommended for all devs)
pip install -r requirements/dev.txt

# Option B: Data Engineering only
pip install -r requirements/data.txt

# Option C: Machine Learning only
pip install -r requirements/ml.txt

# Option D: Backend API only
pip install -r requirements/api.txt
```

Verify that the local package can be imported:
```bash
python -c "import src; print('VARSHAA package initialized successfully, version:', src.__version__)"
```

---

## 5. Project Structure Overview

```text
VARSHAA/
├── configs/              # System and logging configuration files
├── data/                 # Raw, interim, processed, and external datasets (git-ignored)
├── docs/                 # Architecture, contracts, and security specifications
├── models/checkpoints/   # Serialized ML model weights and metadata
├── notebooks/            # Exploratory research and analysis notebooks
├── requirements/         # Modular Python dependency specifications
├── scripts/              # CLI execution and orchestration scripts
├── src/                  # Core Python modules
│   ├── data/             # Ingestion, validation, regridding
│   ├── features/         # Physical indices, error memory, topography
│   ├── regime/           # Synoptic monsoon regime classifier
│   ├── models/           # Quantile models, training, inference, explainability
│   ├── postprocessing/   # Physical constraints, district GIS zonal aggregation
│   ├── validation/       # Deterministic & probabilistic verification engine
│   └── utils/            # Config parser, structured logging, custom exceptions
├── backend/              # FastAPI REST API service
├── frontend/             # React/Vite operational dashboard
├── tests/                # Unit, integration, and data contract tests
└── deployment/           # Dockerfiles and deployment manifests
```

---

## 6. How to Run Existing Components

### Test Configuration Loading:
```bash
python -c "from src.utils.config import config; print('Loaded Project:', config.project_name); print('Spatial CRS:', config.spatial.crs)"
```

### Test Structured Logging:
```bash
python -c "from src.utils.logging import setup_logger; log = setup_logger(); log.info('Logging system verified!')"
```

---

## 7. How to Run Tests

Execute the full automated test suite:
```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with test coverage report
pytest --cov=src tests/

# Run only data contract schema tests
pytest tests/data_tests/ -v
```

---

## 8. Where Each Team Works

Refer to `docs/team-responsibilities.md` for explicit boundaries:
- **Team A (Data Engineering):** `src/data/`, `scripts/download_sample_data.py`, `tests/data_tests/`
- **Team B (ML / AI):** `src/features/`, `src/regime/`, `src/models/`, `src/validation/`, `notebooks/`
- **Team C (Backend):** `backend/`, `tests/integration/test_api*.py`
- **Team D (Frontend / UI):** `frontend/`
- **Team E (GIS / Visualization):** `src/postprocessing/district.py`, `data/external/`
- **Team F (DevOps / MLOps):** `deployment/`, `.github/workflows/`
- **Team G (QA / Testing):** `tests/unit/`, `tests/integration/`

---

## 9. Git & GitHub Workflow

We follow standard branch isolation. Never push directly to `main` or `development`.
1. Pull latest `development`: `git checkout development && git pull`
2. Create feature branch: `git checkout -b feature/<your-feature-name>`
3. Run tests locally: `pytest tests/`
4. Commit using conventional format: `feat(scope): concise description`
5. Push and open a Pull Request into `development`.

See `docs/git-workflow.md` for details.

---

## 10. Where Documentation Lives

All system documentation is stored inside the repository:
- **System Requirements:** [REQUIREMENTS.md](REQUIREMENTS.md)
- **System Boundaries:** [SYSTEM_BOUNDARY.md](SYSTEM_BOUNDARY.md)
- **Master Architecture:** [docs/architecture/system-architecture.md](docs/architecture/system-architecture.md)
- **Team Responsibilities:** [docs/team-responsibilities.md](docs/team-responsibilities.md)
- **Interface Contracts:**
  - Data Contract: [docs/contracts/data-contract.md](docs/contracts/data-contract.md)
  - ML Contract: [docs/contracts/ml-contract.md](docs/contracts/ml-contract.md)
  - API Contract: [docs/contracts/api-contract.md](docs/contracts/api-contract.md)
- **Coding Standards:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Master Roadmap & Task Board:** [MASTER_TODO.md](MASTER_TODO.md)
