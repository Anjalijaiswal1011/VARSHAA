# PART 10 — MLOps, Model Lifecycle & Continuous Reliability Foundation

## 1. Executive Summary & Purpose
RAIN-REPAIR X (**VARSHAA**) is a production-grade operational system designed for Regime-Aware NWP Error Repair and Extreme Rainfall Intelligence across the Indian subcontinent. Following the deployment of the production backend (Part 9), Part 10 implements the end-to-end MLOps lifecycle, experiment tracking, model registry governance, time-aware evaluation, drift detection, and controlled retraining pipelines.

This architecture ensures that model predictions remain accurate, calibrated, and reliable post-deployment without introducing temporal leakage, unvalidated automatic deployments, or silent model degradation.

---

## 2. Model Lifecycle State Machine
```text
           DATA INGESTION (IMD / NCMRWF / ERA5)
                           ↓
                 DATASET VERSIONING
                           ↓
                 FEATURE VERSIONING
                           ↓
               TIME-AWARE DATA SPLIT
          [Train: 2020-2024 | Val: 2025 | Test: 2026]
                           ↓
                   MODEL TRAINING
                           ↓
               EXPERIMENT TRACKING
            (Baselines A-F & Ablations)
                           ↓
           PHYSICAL INVARIANT AUDIT
        (Non-negativity, Quantile Monotonicity)
                           ↓
               MODEL REGISTRATION
                  (Candidate)
                           ↓
              RIGOROUS QUALITY GATE
        (RMSE <= 5% degradation, Brier <= 0.15,
      Heavy CSI >= 0.35, Regime N >= 25 check)
             ┌─────────────┴─────────────┐
             ↓                           ↓
      [PASSED: Staging]          [FAILED: Rejected]
             ↓
        SMOKE TEST
             ↓
        PRODUCTION
             │
     ┌───────┴────────────────────────┐
     ↓                                ↓
 DATA DRIFT (PSI, KS)       CONCEPT DRIFT (Error, Brier)
     ↓                                ↓
 3/7/14-Day Memory Health    Analog Retrieval Health
     └───────┬────────────────────────┘
             ↓
  RETRAINING DECISION ENGINE
  (Scheduled / Drift / Performance Degradation)
             ↓
  SAFE ROLLBACK (Predecessor Instant Restoration)
```

---

## 3. Subsystem Architecture

### 3.1 Immutable System Versioning (`src/mlops/versioning.py`)
Tracks 10 distinct, immutable system versions with a deterministic SHA-256 manifest hash:
- `dataset_version`: e.g., `IMD-ERA5-v2.1`
- `feature_version`: e.g., `v1.0.0-phys`
- `regime_model_version`: e.g., `v1.0.0-regime-lgb`
- `repair_model_version`: e.g., `v1.0.0-repair-smoe`
- `quantile_model_version`: e.g., `v1.0.0-quantile-pinball`
- `calibration_version`: e.g., `v1.0.0-platt`
- `evt_version`: e.g., `v1.0.0-evt-gpd`
- `boundary_version`: `IMD-LGD-2026.1`
- `api_version`: `v1.0.0`
- `deployment_version`: `1.0.0-prod`

### 3.2 Time-Aware Experiment Tracking (`src/mlops/experiments.py`)
Guarantees temporal separation with strict chronological partitioning:
- **Train Period**: `2020-06-01` to `2024-09-30`
- **Validation Period**: `2025-06-01` to `2025-09-30`
- **Test Period**: `2026-06-01` to `2026-09-30`

Tracks canonical reference Baselines A through F:
1. **Experiment A (Raw NWP)**: Global deterministic benchmark.
2. **Experiment B (Traditional Bias)**: Linear/quantile mapping.
3. **Experiment C (Generic ML)**: Monolithic LightGBM.
4. **Experiment D (Regime-Aware)**: Soft Mixture of Experts.
5. **Experiment E (RAIN-REPAIR X)**: Regime + Error Memory + Multi-Quantile.
6. **Experiment F (RAIN-REPAIR X + EVT)**: Full system with Extreme Value Theory.

### 3.3 Quality Gates & Invariants (`src/mlops/quality_gates.py`)
Enforces five mandatory promotion gates:
1. **RMSE Regression Limit**: Candidate RMSE cannot degrade by > 5% over active production.
2. **Probability Calibration**: Brier Score for heavy rainfall must remain $\le 0.15$.
3. **Extreme Event Skill**: Critical Success Index (CSI) for heavy rainfall ($\ge 64.5\text{ mm}$) must be $\ge 0.35$.
4. **Regime-Wise Sufficiency**: Evaluates 6 synoptic regimes. Regimes with sample size $N < 25$ are audited and explicitly tagged `INSUFFICIENT_SAMPLE` rather than reporting fabricated conclusions.
5. **Physical Invariants**:
   - Non-negative precipitation: $\forall i, \hat{y}_i \ge 0$.
   - Monotonic quantile order: $\hat{y}_{P10} \le \hat{y}_{P50} \le \hat{y}_{P75} \le \hat{y}_{P90} \le \hat{y}_{P95}$.

### 3.4 Operational Model Registry (`src/mlops/registry.py`)
Manages lifecycle stages: `candidate` $\to$ `validated` $\to$ `staging` $\to$ `production` $\to$ `archived` / `rejected`.
- **RBAC Protected**: Model promotions and rollbacks require `admin` authorization.
- **Safe Rollback**: Reverts active production immediately to the predecessor model without retraining.

### 3.5 Statistical Drift & Memory Health Monitor (`src/mlops/drift.py`)
- **Data Drift**: Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov (KS) tests across atmospheric features.
  - $\text{PSI} < 0.10$: Stable (`HEALTHY`)
  - $0.10 \le \text{PSI} < 0.25$: Moderate shift (`WARNING`)
  - $\text{PSI} \ge 0.25$: Severe shift (`DRIFT_DETECTED`)
- **Performance Drift**: Tracks recent verified RMSE, MAE, and Brier degradation vs production baseline.
- **Error Memory Health**: Audits 3-day, 7-day, and 14-day error rolling buffers ensuring availability $\ge 90\%$ and temporal leakage safeguards.
- **Analog Memory Health**: Tracks valid candidate counts ($N \ge 3$) and cosine similarity ($\ge 0.50$). If similarity is insufficient, analog explanations are safely withheld (`analog_available = false`).

### 3.6 Controlled Retraining Pipeline (`src/mlops/retraining.py`)
Controlled, non-automatic retraining triggered only when:
- Forecast performance degrades beyond tolerance, OR
- Statistically significant data drift occurs across multiple key atmospheric predictors, OR
- A scheduled review occurs with verified observations.

---

## 4. REST API Endpoint Specifications (`/api/v1/mlops/*`)

| Method | Endpoint | Authorization | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/mlops/registry` | Public / Bearer | Catalog of all registered models and active production model |
| `GET` | `/api/v1/mlops/experiments` | Public / Bearer | Baselines A–F comparison, temporal split info, ablation gains |
| `GET` | `/api/v1/mlops/drift/status` | Public / Bearer | Feature PSI/KS, performance drift, error/analog memory health |
| `POST` | `/api/v1/mlops/promote` | `admin` | Promotes model through quality gates |
| `POST` | `/api/v1/mlops/rollback` | `admin` | Instant safe rollback to prior production version |
| `POST` | `/api/v1/mlops/retrain/evaluate` | `forecaster` / `admin` | Controlled retraining evaluation and staging promotion |

---

## 5. Verification & Test Suite
All 118 unit and integration tests passing:
- `tests/unit/test_mlops_lifecycle.py`: 14 comprehensive unit tests verifying immutability, quality gates, drift statistics, memory health, and registry transitions.
- `tests/integration/test_mlops_api.py`: 6 integration tests verifying REST endpoints, RBAC permissions, and HTTP status codes.
- Pre-existing unit and integration test suites: 98 tests verifying Parts 1 through 9.
