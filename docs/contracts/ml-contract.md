# Interface Contract: Machine Learning Layer (ML -> DOWNSTREAM)

**Contract ID:** CONTRACT-ML-002  
**Version:** 1.0.0  
**Owner:** Team B (Machine Learning & AI)  
**Consumers:** Team C (Backend), Team E (GIS), Team G (Validation)  
**Status:** Frozen for Foundation Phase  

---

## 1. Overview & Purpose
This contract establishes the output structure of the ML post-processing engine. Downstream services (Backend, GIS aggregation, Verification) rely strictly on these schemas.

---

## 2. Model Inference Output Schema

The ML prediction engine emits a structured dictionary, Parquet table, or xarray Dataset with the following fields per grid point `(lat, lon, lead_time)`:

### 2.1 Grid-Level Prediction Fields

| Field Name | Type | Required? | Units | Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cycle_date` | `string` | **Required** | ISO 8601 | `YYYY-MM-DD` | Forecast cycle reference date |
| `lead_time` | `int32` | **Required** | hours | `[24, 48, 72, 96, 120]` | Forecast lead time |
| `lat` | `float32` | **Required** | degrees | `[6.00, 38.00]` | Latitude (0.25° grid) |
| `lon` | `float32` | **Required** | degrees | `[68.00, 98.00]` | Longitude (0.25° grid) |
| `raw_nwp_precip` | `float32` | **Required** | $\text{mm/24h}$ | $[0.0, 1500.0]$ | Baseline raw NWP forecast |
| `corrected_p50` | `float32` | **Required** | $\text{mm/24h}$ | $[0.0, 1500.0]$ | Calibrated deterministic/median estimate |
| `corrected_p10` | `float32` | **Required** | $\text{mm/24h}$ | $[0.0, 1500.0]$ | 10th percentile lower bound |
| `corrected_p75` | `float32` | Optional | $\text{mm/24h}$ | $[0.0, 1500.0]$ | 75th percentile bound |
| `corrected_p90` | `float32` | **Required** | $\text{mm/24h}$ | $[0.0, 1500.0]$ | 90th percentile upper bound |
| `corrected_p95` | `float32` | Optional | $\text{mm/24h}$ | $[0.0, 1500.0]$ | 95th percentile extreme bound |
| `prob_heavy_rain` | `float32` | **Required** | probability | $[0.00, 1.00]$ | $P(\text{Rain} \ge 64.5\text{ mm})$ |
| `prob_very_heavy_rain`| `float32` | **Required** | probability | $[0.00, 1.00]$ | $P(\text{Rain} \ge 115.6\text{ mm})$ |
| `prob_extreme_rain` | `float32` | Optional | probability | $[0.00, 1.00]$ | $P(\text{Rain} \ge 204.5\text{ mm})$ |
| `active_regime` | `string` | **Required** | category | See Regimes | Classified monsoon regime |
| `regime_confidence` | `float32` | Optional | probability | $[0.00, 1.00]$ | Confidence / soft-weight of regime assignment |
| `delta_correction` | `float32` | **Required** | $\text{mm/24h}$ | $[-500.0, 500.0]$ | `corrected_p50 - raw_nwp_precip` |

---

## 3. Monsoon Regime Categorical Taxonomy

The `active_regime` field must emit one of the following standardized strings:
- `"ACTIVE_MONSOON"`: Strong low-level westerly jet, trough south of normal position, widespread rainfall.
- `"BREAK_MONSOON"`: Trough shifted to Himalayan foothills, suppressed peninsula rainfall.
- `"MONSOON_DEPRESSION"`: Low pressure area / depression over Bay of Bengal or central India.
- `"WESTERN_DISTURBANCE"`: Mid-latitude synoptic system impacting northwest/northern India.
- `"OFFSHORE_TROUGH"`: Trough along Arabian Sea coast (Western Ghats convection).
- `"NORMAL_TRANSITIONAL"`: Baseline climatological monsoon state.

---

## 4. Physical Realizability Invariants (Mandatory)

The ML output MUST satisfy the following physical constraints before release:
1. **Non-Negativity:** All precipitation values must be $\ge 0.0\text{ mm}$. Negative values are physical impossibilities.
2. **Quantile Monotonicity:** Quantiles must never cross:
   $$\text{corrected\_p10} \le \text{corrected\_p50} \le \text{corrected\_p75} \le \text{corrected\_p90} \le \text{corrected\_p95}$$
3. **Probability Bounds:** All probability fields must fall strictly within $[0.000, 1.000]$.

Violation of any invariant will trigger `PhysicalConstraintViolationError` in `src/postprocessing/constraints.py`.

---

## 5. Model Explainability Output (Attribution)

For transparent decision-making, the model must export a feature attribution summary:
- **Format:** JSON dictionary per cycle.
- **Fields:**
  - `top_features`: List of strings ranked by mean absolute SHAP value (e.g., `["error_memory_lag1", "dem_slope", "rh850", "raw_nwp_precip"]`).
  - `regime_impact`: Factor multiplier or delta associated with current regime.
