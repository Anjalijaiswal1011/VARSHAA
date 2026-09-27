# Data & Model Versioning Strategy — RAIN-REPAIR X (VARSHAA)

**Status:** Approved  
**Version:** 1.0.0  
**Goal:** Guarantee full scientific and engineering reproducibility across weather forecast post-processing runs.

---

## 1. The Reproducibility Equation

Every trained model and inference forecast in RAIN-REPAIR X must be strictly traceable to this exact lineage tuple:

$$\text{Forecast Model} = f(\text{Code Version}, \text{Config Version}, \text{Dataset Version}, \text{Feature Version}, \text{Random Seed})$$

If any component changes, a new version artifact is generated.

---

## 2. Versioning Taxonomy

```
+--------------------------------------------------------------------------------------------------+
| CODE VERSION       -> Git Commit Hash (e.g., git-7a47451) + Git Tag (v1.0.0-phase1)              |
| CONFIG VERSION     -> SHA-256 Hash of YAML Config File (e.g., cfg-8f12a3d)                       |
| DATASET VERSION    -> Date-range + Source identifier (e.g., dset-imd_gfs-2020_2024-v1)           |
| FEATURE VERSION    -> Feature Definition Hash (e.g., feat-orog_lag7_regime-v1.2)                 |
| MODEL VERSION      -> Semantic Version + Model Type (e.g., model-lgbm_quantile-v1.0.0)           |
| EXPERIMENT ID      -> ISO Date + Run UUID (e.g., exp-20260927-01-lgb_active)                    |
+--------------------------------------------------------------------------------------------------+
```

---

## 3. Storage & Metadata Conventions

### 3.1 Model Checkpoint Metadata (`model_metadata.json`)
Every saved model in `models/checkpoints/{model_name}/` must include an adjacent `metadata.json` containing:

```json
{
  "model_id": "model-lgbm_quantile_p50-v1.0.0",
  "model_type": "LightGBMRegressor",
  "created_at": "2026-09-27T12:00:00Z",
  "code_git_commit": "7a47451b6e4d",
  "config_hash": "a1b2c3d4e5f6",
  "dataset_version": "dset-imd_gfs-2020_2024-v1",
  "feature_columns": [
    "raw_nwp_precip",
    "elevation",
    "dem_slope",
    "rh850",
    "u850",
    "v850",
    "error_memory_lag1",
    "regime_encoded"
  ],
  "target_variable": "obs_precip",
  "random_seed": 42,
  "hyperparameters": {
    "objective": "quantile",
    "alpha": 0.50,
    "n_estimators": 300,
    "learning_rate": 0.05
  },
  "metrics": {
    "val_rmse": 16.24,
    "val_mae": 9.71
  }
}
```

### 3.2 Dataset Version Partitioning
Data storage follows deterministic folder partitioning:
```text
data/
├── raw/
│   ├── gfs/YYYY/MM/DD/gfs_YYYYMMDD_t00z.nc
│   └── imd_obs/YYYY/imd_precip_YYYY.nc
├── interim/
│   └── regridded_025/YYYY/regridded_YYYYMMDD.nc
└── processed/
    └── feature_store_v1/YYYY/features_YYYYMMDD.parquet
```

---

## 4. Operational Ingestion Run Versioning
When the daily operational post-processing pipeline executes, it generates a run manifest stored at `data/processed/{cycle_date}/manifest.json` recording:
- Input raw NWP file path and SHA-256 checksum.
- Active model version loaded.
- Number of valid grid cells processed.
- Execution start time, completion time, and runtime duration.
- Logging trace ID.
