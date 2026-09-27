# Feature Catalogue & Dictionary — RAIN-REPAIR X (VARSHAA)

**Module:** `src/features`  
**Schema Registry:** `src.features.schema.FEATURE_REGISTRY`  
**Status:** Approved & Frozen for Downstream Model Development  
**Target Domain:** Indian Monsoon Hydrometeorology & Orography  

---

## 1. Feature Architecture Overview

The RAIN-REPAIR X feature engineering layer translates aligned Numerical Weather Prediction (NWP) rasters, Digital Elevation Models (DEM), and historical error stores into 32 scientifically grounded, physically interpretable predictors.

```
ALIGNED DATA TENSORS (NWP, Obs, DEM, Climatology)
                      │
                      ▼
+─────────────────────────────────────────────────────────────+
|               FEATURE ENGINEERING PIPELINE                  |
|                                                             |
|  [Group A: NWP Baseline]        - Precip, Log-precip, Leads |
|  [Group B: Atmospheric State]   - T2m, RH, MSLP, U, V, Speed|
|  [Group C: Physics-Guided]      - Orographic Uplift, MFC,   |
|                                   Vorticity, Shear, Aspect  |
|  [Group D: Terrestrial Topo]    - Elevation, Slope, Aspect, |
|                                   Coastal Dist, Exposure    |
|  [Group E: Spatial Context]     - 3x3 Mean, Max, Std, Grad  |
|  [Group F: Climatology Anomaly] - DOY Mean, NWP Anomaly     |
|  [Group G: Temporal Cycles]     - DOY Sin/Cos, Month Sin/Cos|
|  [Group H: Multi-Scale Memory]  - Lag 1, Lag 3, 7, 14 Errors|
|  [Group I: Regime Support]      - WD Shear, Trough Grad,    |
|                                   Offshore Coastal Shear    |
+─────────────────────────────────────────────────────────────+
                      │
                      ▼
MODEL-READY FEATURE TABLE (15,609 grid cells x 32 features)
```

---

## 2. Comprehensive Feature Dictionary

| Feature Name | Group | Source | Unit | Formula / Formulation | Physical Meaning | Consuming Models | Leakage Risk |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| `nwp_precip` | NWP | GFS / NCUM | mm/24h | Raw accumulated precipitation | Baseline uncorrected forecast | Quantile, Prob, Error, Regime | SAFE |
| `nwp_precip_log` | NWP | Derived | log(mm+1) | $\ln(1 + \text{nwp\_precip})$ | Compresses heavy right-hand skewness | Quantile, Prob | SAFE |
| `lead_time` | NWP | Metadata | hours | $24, 48, 72, 96, 120$ | Forecast lead time horizon | Quantile, Prob, Regime | SAFE |
| `lead_time_days` | NWP | Derived | days | $\text{lead\_time} / 24.0$ | Continuous lead time factor | Quantile | SAFE |
| `t2m` | ATMOSPHERIC | GFS | K | 2m Air Temperature | Surface thermal buoyancy & energy | Regime, Quantile | SAFE |
| `rh850` | ATMOSPHERIC | GFS | % | Relative humidity at 850 hPa | Lower-tropospheric moisture saturation | Regime, Quantile, Prob | SAFE |
| `mslp` | ATMOSPHERIC | GFS | hPa | Mean Sea-Level Pressure | Synoptic troughs, depressions, highs | Regime, Transition | SAFE |
| `u850` | ATMOSPHERIC | GFS | m/s | Zonal wind at 850 hPa | Low-level westerly monsoon jet strength | Regime, Quantile | SAFE |
| `v850` | ATMOSPHERIC | GFS | m/s | Meridional wind at 850 hPa | Cross-equatorial flow & vorticity flux | Regime, Quantile | SAFE |
| `wind_speed_850` | ATMOSPHERIC | Derived | m/s | $\sqrt{u_{850}^2 + v_{850}^2}$ | Kinetic intensity of low-level monsoon jet | Regime, Quantile | SAFE |
| `wind_direction_850`| ATMOSPHERIC | Derived | degrees | $\text{atan2}(-u, -v) \pmod{360}$ | Direction wind blows from (0-360°) | Regime | SAFE |
| `vapor_pressure_proxy`| ATMOSPHERIC | Derived | hPa | $\text{RH} \cdot e_{sat}(T_{2m})$ (Tetens) | Absolute water vapor content in column | Quantile, Prob | SAFE |
| `orographic_uplift` | PHYSICS | Derived | m/s | $u \frac{\partial h}{\partial x} + v \frac{\partial h}{\partial y}$ | Mechanical lift on Western Ghats / Himalayas | Quantile, Prob, Regime | SAFE |
| `moisture_flux_conv`| PHYSICS | Derived | 1/s | $-\nabla \cdot (\text{RH}_{850} \vec{V}_{850})$ | Dynamic moisture convergence fueling rain | Quantile, Prob | SAFE |
| `vorticity_850` | PHYSICS | Derived | 1/s | $\frac{\partial v_{850}}{\partial x} - \frac{\partial u_{850}}{\partial y}$ | Cyclonic vortex rotation in depressions | Regime, Transition | SAFE |
| `bulk_wind_shear` | PHYSICS | Derived | m/s | $\|\vec{V}_{200} - \vec{V}_{850}\|$ | Deep convective storm organization potential | Prob, Regime | SAFE |
| `windward_leeward_index`| PHYSICS | Derived | [-1, 1] | $\cos(\theta_{wind} - (\theta_{aspect} + 180^\circ))$ | Windward upslope (+1) vs leeward shadow (-1) | Quantile | SAFE |
| `elevation` | TERRAIN | SRTM DEM | meters | Surface elevation above sea level | Altitudinal condensation level | Quantile, Regime | SAFE |
| `dem_slope` | TERRAIN | Derived | degrees | Local terrain gradient slope $\alpha$ | Steepness of orographic ramp | Quantile | SAFE |
| `dem_aspect` | TERRAIN | Derived | degrees | Azimuth orientation of mountain slope | Direction barrier faces relative to wind | Quantile | SAFE |
| `dist_to_coast` | TERRAIN | Vector GIS | km | Minimum Euclidean distance to sea | Maritime boundary layer moderation | Regime, Quantile | SAFE |
| `relative_elevation_exposure`| TERRAIN | Derived | meters | $h - \text{mean}_{3\times3}(h)$ | Ridge peaks ($>0$) vs valley basins ($<0$) | Quantile | SAFE |
| `nwp_precip_spatial_mean_3x3`| SPATIAL | Derived | mm/24h | Uniform filter 3x3 mean | Mesoscale surrounding precipitation buffer | Quantile, Prob | SAFE |
| `nwp_precip_spatial_max_3x3` | SPATIAL | Derived | mm/24h | Maximum filter 3x3 peak | Adjacent intense convective cell detection | Prob | SAFE |
| `nwp_precip_spatial_std_3x3` | SPATIAL | Derived | mm/24h | Local 3x3 standard deviation | Convective patchiness & spatial variability | Prob | SAFE |
| `nwp_precip_spatial_gradient`| SPATIAL | Derived | mm/km | Magnitude of 2D spatial gradient | Frontal rainband edges & storm borders | Quantile | SAFE |
| `clim_mean_doy` | CLIMATOLOGY | IMD Archive | mm/24h | 30-Year day-of-year historical mean | Climatological baseline expectation | Quantile, Regime | SAFE |
| `nwp_clim_anomaly`| CLIMATOLOGY | Derived | mm/24h | $\text{NWP} - \text{clim\_mean\_doy}$ | Abnormal wet/dry forecast departures | Quantile, Prob | SAFE |
| `doy_sin`, `doy_cos` | TEMPORAL | Calendar | [-1, 1] | $\sin, \cos(2\pi \cdot \text{DOY} / 365.25)$ | Smooth annual solar progression | Regime, Quantile | SAFE |
| `month_sin`, `month_cos`| TEMPORAL | Calendar | [-1, 1] | $\sin, \cos(2\pi \cdot M / 12)$ | Monsoon seasonal phase representation | Regime | SAFE |
| `error_memory_lag1` | ERROR_MEMORY| Error Store | mm/24h | $\text{NWP}_{t-1} - \text{Obs}_{t-1}$ | Immediate 24h persistent model bias | Error, Quantile | SAFE |
| `error_memory_lag3_mean`| ERROR_MEMORY| Error Store | mm/24h | Trailing 3-day average bias | Short-term synoptic error trend | Error, Quantile | SAFE |
| `error_memory_lag7_mean`| ERROR_MEMORY| Error Store | mm/24h | Trailing 7-day average bias | Weekly synoptic model drift | Error, Quantile | SAFE |
| `error_memory_lag14_mean`| ERROR_MEMORY| Error Store | mm/24h | Trailing 14-day average bias | Intra-seasonal error memory | Error, Quantile | SAFE |
| `wd_shear_proxy` | REGIME_SUPPORT | Derived | m/s | $\max(u_{200}, 0) \cdot \mathbb{I}(\text{lat} \ge 24^\circ)$ | Western Disturbance upper jet indicator | Regime | SAFE |
| `monsoon_trough_mslp_gradient`| REGIME_SUPPORT | Derived | hPa/deg | $\partial(\text{MSLP}) / \partial(\text{Lat})$ | Monsoon Trough axis position (Active/Break) | Regime, Transition | SAFE |
| `offshore_trough_coastal_shear`| REGIME_SUPPORT | Derived | m/s | $v_{850} \cdot \mathbb{I}(\text{lat} \in [10, 18], d_{coast} < 80)$ | Arabian Sea offshore trough indicator | Regime | SAFE |

---

## 3. Downstream Model Consumption Mapping

1. **Weather Regime Classifier (`src/regime/`):** Consumes $U_{850}, V_{850}, \text{WindSpeed}, \text{WindDir}, \text{MSLP}, \text{RH}_{850}, \zeta_{850}, \text{Shear}, \text{Trough Gradient}, \text{WD Proxy}, \text{Offshore Shear}, \text{DOY Harmonics}$.
2. **Regime Transition Engine (`src/regime/`):** Tracks temporal derivatives of regime probabilities and synoptic pressure/vorticity gradients.
3. **NWP Error DNA & Memory Models (`src/models/`):** Consumes multi-scale error memory lags ($1, 3, 7, 14\text{ days}$) and spatial error persistence.
4. **Quantile Models ($P_{10}, P_{50}, P_{75}, P_{90}, P_{95}$):** Consumes the complete 32-feature predictor matrix $X$.
5. **Heavy / Very Heavy Rainfall Classifiers ($\ge 64.5\text{ mm}, \ge 115.6\text{ mm}$):** Focuses heavily on moisture convergence, orographic uplift, vapor pressure, and spatial maximum $3\times3$.
