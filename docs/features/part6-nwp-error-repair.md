# RAIN-REPAIR X: PART 6 — NWP Error DNA, Multi-Scale Error Memory & Soft Regime-Aware Repair

**Status:** IMPLEMENTED  
**Layer:** Core Post-Processing Engine (`src/postprocessing/`)  
**Interface Contract:** CONTRACT-ML-002  

---

## 1. NWP Error Definition & Mathematical Formulation

Systematic and context-dependent forecast error in RAIN-REPAIR X is strictly defined as:

$$\text{NWP Error} = \text{Observed Rainfall} - \text{Raw NWP Rainfall}$$

- **Signed Error ($e = y_{\text{obs}} - y_{\text{nwp}}$):**
  - $e > 0$: NWP under-forecast (dry bias / missed rain).
  - $e < 0$: NWP over-forecast (wet bias / false alarm).
- **Corrected Rainfall Re-construction:**
  $$\hat{y}_{\text{corrected}} = \max\left(0.0, \; y_{\text{nwp}} + \hat{e}\right)$$
- **Physical Invariant:** All post-processed precipitation is constrained to $\ge 0.0\text{ mm/24h}$. Negative precipitation is physically non-realizable.
- **Target Isolation:** Ground-truth observation ($y_{\text{obs}}$) and target error ($e$) are strictly isolated from the predictor matrix $X$ at inference time.

---

## 2. NWP Error DNA Representation

The **Error DNA** represents the multidimensional contextual fingerprint of numerical weather prediction error:

$$\text{Error DNA} = f(\text{Context}_{\text{Atmo}}, \text{Terrain}, \text{LeadTime}, P(\text{Regime}), \text{Transition}, \text{Memory}_{\text{Recent}}, \text{Analog}_{\text{Hist}})$$

Implemented in `src/postprocessing/error_dna.py` via `NWPErrorDNAExtractor`:

1. **Base NWP & Lead Time:** `nwp_precip`, `nwp_precip_log`, `lead_time`, `lead_time_days`.
2. **Location & Terrain:** `lat`, `lon`, `elevation`, `dem_slope`, `dem_aspect`, `dist_to_coast`.
3. **Atmospheric Dynamics & Physics:** `t2m`, `rh850`, `mslp`, `u850`, `v850`, `wind_speed_850`, `wind_direction_850`, `vapor_pressure_proxy`, `orographic_uplift`, `moisture_flux_conv`, `vorticity_850`, `bulk_wind_shear`, `windward_leeward_index`.
4. **Spatial Neighborhood & Climatology:** `nwp_precip_spatial_mean_3x3`, `nwp_precip_spatial_max_3x3`, `nwp_precip_spatial_std_3x3`, `clim_mean_doy`, `nwp_clim_anomaly`, `doy_sin`, `doy_cos`.
5. **Soft Weather Regime Probabilities:** `prob_active_monsoon`, `prob_break_monsoon`, `prob_monsoon_depression`, `prob_coastal`, `prob_orographic`, `prob_western_disturbance`.
6. **Regime Transition State:** `transition_strength`, `transition_confidence`, `is_transition_state`.
7. **Multi-Scale Trailing Error Memory:** `recent_bias_{3,7,14}d`, `recent_mae_{3,7,14}d`, `recent_rmse_{3,7,14}d`, `recent_error_std_{3,7,14}d`.
8. **Historical Analog Signals:** `analog_mean_error`, `analog_std_error`, `analog_min_distance`.

---

## 3. Multi-Scale Recent Error Memory (3d, 7d, 14d)

Implemented in `src/postprocessing/memory.py` via `LeakageSafeErrorMemoryStore`:
- Tracks verified forecast errors across grid points and verification cycles.
- For a forecast issued for cycle date $T$:
  - $T-1, T-2, T-3$: 3-day trailing window (`recent_bias_3d`, `recent_mae_3d`, `recent_rmse_3d`, `recent_error_std_3d`).
  - $T-1, \dots, T-7$: 7-day trailing window (`recent_bias_7d`, `recent_mae_7d`, `recent_rmse_7d`, `recent_error_std_7d`).
  - $T-1, \dots, T-14$: 14-day trailing window (`recent_bias_14d`, `recent_mae_14d`, `recent_rmse_14d`, `recent_error_std_14d`).

---

## 4. Leakage Prevention Architecture

Zero temporal leakage is enforced by three automated architectural guardrails:
1. **Target Isolation Audit:** `validate_target_isolation` scans predictor matrices and raises `TemporalLeakageError` if `obs_precip`, `signed_error`, or target aliases are present.
2. **Memory Temporal Barrier:** `LeakageSafeErrorMemoryStore.get_valid_past_dates` enforces $t_{\text{verified}} < T_{\text{forecast}}$. Accessing verification data at or beyond $T$ immediately triggers a fatal `TemporalLeakageError`.
3. **Analog Query Masking:** `KNNAnalogMemory.query` filters the candidate index dynamically, ensuring only cases verified before query cycle $T$ are eligible for nearest-neighbor aggregation.

---

## 5. Historical Analog Memory & Vector Search

Implemented in `src/postprocessing/analog.py`:
- **Abstraction:** `AnalogMemory(ABC)` enables pluggable search backends.
- **Implementation:** `KNNAnalogMemory` uses normalized feature representations of pre-forecast atmospheric dynamics, terrain, and regime probabilities.
- **Why KNN / KDTree over FAISS for Current Scale:**
  - For operational regional forecast matrices ($< 100,000$ points), Scikit-Learn `NearestNeighbors` executes in $< 2\text{ ms}$ with zero external C++ binary dependency.
  - Pluggable `AnalogMemory` interface allows seamless substitution with GPU-accelerated FAISS when scaling to all-India 10-year reanalysis datasets without rewriting the repair model.
- **Extracted Signals:**
  - `analog_mean_error`: Distance-weighted average verified error among top-$k$ nearest synoptic analogs.
  - `analog_std_error`: Dispersion / volatility among analogs (uncertainty proxy for Part 7).
  - `analog_min_distance`: Distance to closest analog (novelty / out-of-distribution detector).

---

## 6. Soft Regime-Aware Repair Architectures

Implemented in `src/postprocessing/repair_model.py`:

### Option A: Shared LightGBM Regressor
- Single unified gradient-boosted tree model trained on full Error DNA including soft probability vector:
  $$\hat{e} = f_{\text{LightGBM}}(X_{\text{ErrorDNA}}, P(\text{Active}), \dots, P(\text{WD}))$$

### Option B: Soft Mixture of Experts (MoE)
- Up to 6 dedicated regime-specific expert regressors.
- Output blended using continuous soft regime probabilities:
  $$\hat{e} = \sum_{k=1}^6 P(\text{Regime}_k) \times \hat{e}_k(X)$$
- **Sample Sufficiency Gate:** If any regime has $< 25$ training samples, Option B automatically falls back to Option A (the shared model) for that regime, preventing overfitting on rare regimes.

---

## 7. Baseline Correction Models

Implemented in `src/postprocessing/baselines.py`:
- **BASELINE 1: Raw NWP:** Predicts zero error; $\hat{y} = y_{\text{nwp}}$.
- **BASELINE 2: Statistical Bias Correction:** Computes empirical mean error by lead time:
  $$\hat{y} = \max\left(0, \; y_{\text{nwp}} + \overline{e}(h)\right)$$
- **BASELINE 3: Generic ML Error Correction:** LightGBM regressor on NWP, atmospheric, and terrain features **strictly omitting** regime probabilities, transition states, and analog memory.

---

## 8. Model Selection & Ablation Study

Implemented in `src/postprocessing/ablation.py` via `ModelSelectionExperimentRunner`:
- Compares:
  - **Baseline:** Raw NWP
  - **Model A:** Generic ML Correction
  - **Model B:** Regime Features + ML (Option A)
  - **Model C:** Soft Regime Experts (Option B MoE)
  - **Model D:** Soft Regime + Transition + Recent Memory (3/7/14d)
  - **Model E:** Full RAIN-REPAIR X (Model D + Analog Retrieval)
- Evaluated on chronologically held-out test data.

---

## 9. Fallback Strategy Hierarchy

Implemented in `src/postprocessing/fallback.py` via `FallbackRepairController`:

```
   Tier 0: Full RAIN-REPAIR X (Regimes + Transitions + Memory + Analog)
                           ↓ (Analog service drops)
   Tier 1: Regime-Aware Model without Analog Memory
                           ↓ (Regime classifier fails / features missing)
   Tier 2: Generic ML Correction (NWP + Atmospheric + Terrain)
                           ↓ (ML model unavailable)
   Tier 3: Statistical Lead-Time Bias Correction
                           ↓ (All correction services fail)
   Tier 4: Raw NWP Pass-Through (Fail-Safe)
```

Every forecast record tags `fallback_tier` and `fallback_degraded` in telemetry. Fallbacks are never hidden.

---

## 10. Implementation Roadmap Matrix

| Component | Status | Location |
| :--- | :---: | :--- |
| NWP Error Target ($Obs - NWP$) | **IMPLEMENTED** | `src/postprocessing/target.py` |
| Non-Negativity Handling | **IMPLEMENTED** | `src/postprocessing/target.py` |
| Target Isolation Verification | **IMPLEMENTED** | `src/postprocessing/target.py` |
| 3/7/14-Day Trailing Error Memory | **IMPLEMENTED** | `src/postprocessing/memory.py` |
| Leakage-Safe Memory Store | **IMPLEMENTED** | `src/postprocessing/memory.py` |
| Historical Analog Memory (KNN) | **IMPLEMENTED** | `src/postprocessing/analog.py` |
| Pluggable `AnalogMemory` Interface | **IMPLEMENTED** | `src/postprocessing/analog.py` |
| Structured Error DNA Extractor | **IMPLEMENTED** | `src/postprocessing/error_dna.py` |
| Baseline 1 (Raw NWP) | **IMPLEMENTED** | `src/postprocessing/baselines.py` |
| Baseline 2 (Statistical Bias) | **IMPLEMENTED** | `src/postprocessing/baselines.py` |
| Baseline 3 (Generic ML) | **IMPLEMENTED** | `src/postprocessing/baselines.py` |
| Option A (Shared LightGBM) | **IMPLEMENTED** | `src/postprocessing/repair_model.py` |
| Option B (Soft Mixture of Experts) | **IMPLEMENTED** | `src/postprocessing/repair_model.py` |
| 5-Tier Fallback Controller | **IMPLEMENTED** | `src/postprocessing/fallback.py` |
| 5-Model Ablation Runner | **IMPLEMENTED** | `src/postprocessing/ablation.py` |
| Continuous & Event Verification | **IMPLEMENTED** | `src/postprocessing/evaluation.py` |
| Regime-Wise Performance Breakdown | **IMPLEMENTED** | `src/postprocessing/evaluation.py` |
| Lead-Time Breakdown | **IMPLEMENTED** | `src/postprocessing/evaluation.py` |
| Spatial Grid Error Summary | **IMPLEMENTED** | `src/postprocessing/evaluation.py` |
| Versioned Model Artifact Manager | **IMPLEMENTED** | `src/postprocessing/artifacts.py` |
| End-to-End `RainRepairEngine` | **IMPLEMENTED** | `src/postprocessing/engine.py` |
| Multi-Quantile P10/P50/P90 Regression | **PLANNED (Part 7)** | `src/postprocessing/quantiles.py` |
| EVT / Generalized Pareto Tail Fitting | **PLANNED (Part 7)** | `src/postprocessing/evt.py` |
| GIS Dashboard UI Integration | **PLANNED (Part 8/9)** | `frontend/` |
