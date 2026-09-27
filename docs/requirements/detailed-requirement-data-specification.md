# RAIN-REPAIR X — Detailed Requirement & Data Specification

**Document Version:** 2.0.0  
**Phase:** PART 1 — Requirement & Data Specification  
**Status:** Approved for Implementation Planning  
**Lead Roles:** Senior Software Architect + Senior AI/ML Engineer + Senior Data Engineer  

---

## 1. Requirement Summary

**RAIN-REPAIR X** is a domain-informed, regime-aware meteorological artificial intelligence post-processing system engineered specifically for the complex dynamics of the Indian Monsoon and associated weather systems.

Numerical Weather Prediction (NWP) models (e.g., IMD GFS, NCMRWF NCUM, ECMWF IFS) exhibit persistent, systematic errors across India:
- **Drizzle bias / Spatial over-prediction:** Predicting light precipitation ($0.1 - 5\text{ mm/day}$) across vast areas that remain dry.
- **Orographic displacement:** Misplacing extreme crest-line precipitation along the Western Ghats and Himalayan foothills.
- **Convective under-forecasting:** Underestimating peak rainfall rates during active monsoon depressions and tropical cloud bursts.
- **Regime dependency:** Model biases change radically depending on the active synoptic state (e.g., an active trough versus a break period versus a Western Disturbance).

RAIN-REPAIR X establishes a multi-stage correction architecture:
1. **Physics-guided Feature Extraction:** Ingestion of thermodynamics, moisture dynamics, and topographic barriers.
2. **Soft Synoptic Regime Classification & Transition Detection:** Probabilistic routing into Active, Break, Low/Depression, Western Disturbance, Offshore Trough, and Normal states.
3. **NWP Error DNA & Multi-Scale Memory:** Dynamic retrieval of rolling 3/7/14-day persistent errors and historical synoptic analogs.
4. **Calibrated Quantile & Extreme Value Modeling:** Deterministic $P_{50}$, spread ($P_{10}, P_{75}, P_{90}$), categorical IMD exceedance probabilities ($\ge 64.5\text{ mm}, \ge 115.6\text{ mm}$), with Extreme Value Theory (EVT/GPD) for high-impact distribution tails.
5. **Operational Administrative Dissemination:** Seamless aggregation from $0.25^\circ$ grid to district warning polygons with full SHAP transparency and historical replay capabilities.

---

## 2. SIH Requirement Matrix

| Requirement | Why it exists | How RAIN-REPAIR X addresses it | Required Input | Expected Output | Validation Metric | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **Weather Regime Identification** | NWP systematic bias shifts fundamentally between synoptic states. | Multi-class soft regime estimator conditioned on low-level winds, shear, and MSLP. | $U_{850}, V_{850}, \text{MSLP}, \text{RH}_{850}$, shear indices | Soft regime probability vector $\vec{\pi} \in [0, 1]^K$ | Macro F1-score, Precision/Recall, Brier Score | **P0 (Must)** |
| **Active Monsoon State** | Deep trough south of normal position; heavy widespread rainfall. | Amplifies moisture-convergence weighting and adjusts orographic uplift. | $U_{850}$ jet core speed, Monsoon Trough axis lat | Active regime weight $P(\text{Active})$ | Detection POD, Regime F1 | **P0 (Must)** |
| **Break Monsoon State** | Trough shifts to Himalayan foothills; peninsular dry spells. | Dampens peninsular convective forecasts; reduces false alarm drizzle. | Rainfall latitudinal profile, MSLP anomaly | Break regime weight $P(\text{Break})$ | FAR reduction, Critical Success Index (CSI) | **P0 (Must)** |
| **Low / Monsoon Depression** | Intense cyclonic vortices causing severe flooding across Central India. | Tracks depression center; models asymmetric rain shields via vortex features. | Vorticity at 850 hPa, MSLP minima, wind divergence | Depression probability, localized intensity boost | Equitable Threat Score (ETS), CSI $\ge 64.5\text{ mm}$ | **P0 (Must)** |
| **Coastal Rainfall Dynamics** | Land-sea thermal contrasts trigger rapid coastal squall lines. | Incorporates distance to coast, sea-surface temperature, and onshore moisture flux. | Distance to coast, shoreline vector, low-level wind | Coastal zone bias correction factor | Coastal station RMSE, MAE | **P1 (High)** |
| **Orographic Precipitation** | Western Ghats & Himalayas force mechanical air lifting. | Ingests DEM elevation, slope, aspect, and wind-slope dot product ($\vec{V} \cdot \nabla h$). | DEM elevation, slope, aspect, low-level wind vector | Orographic condensation rate feature, localized correction | Spatial RMSE along Western Ghats transects | **P0 (Must)** |
| **Western Disturbance (WD)** | Winter/pre-monsoon extra-tropical storms impacting North/NW India. | Ingests upper-tropospheric jet stream dynamics (200 hPa winds, geopotential height). | $U_{200}, V_{200}, Z_{500}$, NW India humidity | WD regime probability and NW India precipitation adjustment | Northern India winter/spring CSI, MAE | **P1 (High)** |
| **NWP Rainfall Correction** | Raw NWP has spatial drift and systematic scale bias. | Regime-conditioned gradient boosted quantile regression. | Raw NWP precip, regime weights, physics features | Corrected $P_{50}$ deterministic rainfall (mm) | RMSE, MAE, Mean Bias Error (MBE) | **P0 (Must)** |
| **Heavy Rain Probability ($\ge 64.5\text{ mm}$)** | Operational disaster management threshold (IMD Yellow/Orange alert). | Calibrated binary classifier / CDF threshold evaluation. | Quantile predictions, atmospheric moisture, error memory | $P(\text{Rain} \ge 64.5\text{ mm}) \in [0.0, 1.0]$ | Brier Score, Reliability Diagram, ROC-AUC | **P0 (Must)** |
| **Very Heavy Rain ($\ge 115.6\text{ mm}$)** | Severe urban and riverine flood trigger (IMD Orange/Red alert). | Extreme quantile estimation ($P_{90}, P_{95}$) with probability calibration. | Atmospheric instability, precipitable water, error DNA | $P(\text{Rain} \ge 115.6\text{ mm}) \in [0.0, 1.0]$ | CSI $\ge 115.6$, Precision-Recall AUC | **P0 (Must)** |
| **Extreme Rainfall Handling ($\ge 204.5\text{ mm}$)** | Flash floods, landslides; empirical distributions fail in the fat tail. | Extreme Value Theory (EVT) / Generalized Pareto Distribution (GPD) modeling. | Tail residuals exceeding 95th percentile | Calibrated tail exceedance probability & return levels | Tail Quantile Loss, Extreme CSI | **P2 (Advanced)** |
| **Grid-Level Forecast** | Localized hydrological and precision agricultural decision-making. | Regular $0.25^\circ \times 0.25^\circ$ gridded prediction fields. | Ingested features on $0.25^\circ$ grid | 2D gridded NetCDF/GeoTIFF raster array | Fractions Skill Score (FSS) | **P0 (Must)** |
| **District-Level Forecast** | State and District Disaster Management Authority (DDMA) protocols. | Spatial polygon zonal statistics (area-weighted mean, max, alert levels). | Grid predictions, official District GeoJSON boundaries | District JSON/GeoJSON with IMD alert color code | District-level MAE, Alert Accuracy | **P0 (Must)** |
| **Verification & Replay** | Operational credibility requires continuous auditable benchmarking. | Daily automated verification engine and historical date range replay runner. | Historical forecasts, IMD ground-truth observations | Automated verification reports (RMSE, CSI, CRPS) | Metric verification pipeline pass rate | **P0 (Must)** |
| **Explainability** | Forecasters reject black-box AI without physical reasoning. | Tree SHAP feature attributions and regime influence contributions. | Trained model, prediction feature vector | Top-5 driving features with directional impact | Forecaster interpretability audit | **P1 (High)** |
| **Operational Usability** | Emergency response requires sub-second queries and interactive visual maps. | FastAPI REST endpoints + Leaflet/MapLibre web GIS dashboard. | Post-processed outputs stored in indexed cache/DB | Interactive dual-map view, meteograms, alerts | API response time $< 200\text{ ms}$, UI responsiveness | **P0 (Must)** |

---

## 3. Functional Requirements (FR)

- **FR-01 (Accept NWP Forecast Data):** The system shall ingest multi-lead-time (24h, 48h, 72h, 96h, 120h) Numerical Weather Prediction precipitation and surface fields in NetCDF4, GRIB2, or pre-extracted tabular formats.
- **FR-02 (Accept Multi-Level Atmospheric Variables):** The system shall ingest multi-level isobaric fields (850 hPa, 700 hPa, 500 hPa, 200 hPa) including horizontal wind vectors ($U, V$), relative humidity ($RH$), temperature ($T$), geopotential height ($Z$), and mean sea-level pressure ($MSLP$).
- **FR-03 (Process Terrain & Geographic Data):** The system shall ingest and preprocess high-resolution Digital Elevation Model (DEM) data to compute static geographic rasters: elevation ($h$), terrain slope ($\alpha$), terrain aspect ($\theta$), distance to coastline ($d_{coast}$), and land-sea mask.
- **FR-04 (Generate Physics-Guided Features):** The system shall compute domain-specific physical diagnostic quantities:
  - Moisture Flux Divergence (MFD): $\nabla \cdot (q \vec{V})$
  - Orographic Uplift Index: $w_{orog} = \vec{V}_{850} \cdot \nabla h$
  - Low-Level Bulk Wind Shear: $|\vec{V}_{850} - \vec{V}_{surface}|$
  - Deep-Layer Wind Shear: $|\vec{V}_{200} - \vec{V}_{850}|$
  - Saturated Vapor Pressure & Dewpoint Depression.
- **FR-05 (Estimate Soft Weather-Regime Probabilities):** The system shall evaluate active synoptic atmospheric states and output a continuous probability vector $\vec{\pi} = [\pi_{active}, \pi_{break}, \pi_{depression}, \pi_{wd}, \pi_{offshore}, \pi_{normal}]$ such that $\sum \pi_k = 1.0$.
- **FR-06 (Detect Regime Transitions):** The system shall compute regime transition gradients:
  $$\Delta \vec{\pi}_t = \vec{\pi}_t - \vec{\pi}_{t-24h}$$
  and flag impending synoptic phase shifts (e.g., Active-to-Break or incoming Western Disturbance).
- **FR-07 (Generate NWP Error DNA):** The system shall compute spatial error profiles classifying the historical error mode of the NWP model at each location:
  $$\text{Error DNA} = [\text{Mean Bias}, \text{False Alarm Tendency}, \text{Peak Displacement Vector}, \text{Spread-Error Ratio}]$$
- **FR-08 (Retrieve Recent Forecast-Error Memory):** The system shall maintain and query rolling spatial error buffers over 3-day, 7-day, and 14-day trailing lookback windows:
  $$\text{Error}_t(\tau) = \text{NWP}_{t-\tau} - \text{Observation}_{t-\tau} \quad \text{for } \tau \in \{1, 2, 3, 7, 14\}$$
- **FR-09 (Retrieve Similar Historical Forecast Cases):** The system shall index historical forecast feature vectors and retrieve top-$K$ historical atmospheric analogs based on cosine/Euclidean distance of synoptic regime, moisture profile, and seasonal day-of-year.
- **FR-10 (Generate Regime-Aware Rainfall Correction):** The system shall compute deterministic corrected precipitation forecasts ($P_{50}$) dynamically weighted by active regime probabilities.
- **FR-11 (Generate Quantile Forecasts P50/P75/P90):** The system shall produce calibrated quantile estimates ($P_{10}, P_{50}, P_{75}, P_{90}, P_{95}$) satisfying physical non-crossing monotonicity.
- **FR-12 (Generate Calibrated Heavy/Very-Heavy Rainfall Probabilities):** The system shall evaluate calibrated probabilities for operational IMD thresholds:
  $$P(\text{Rain} \ge 64.5\text{ mm}), \quad P(\text{Rain} \ge 115.6\text{ mm})$$
  using Platt scaling or isotonic regression to ensure probability reliability.
- **FR-13 (Extreme Tail Modeling via EVT/GPD — Advanced Module):** Where enabled, the system shall fit a Generalized Pareto Distribution (GPD) to residual exceedances over a high threshold $u$ (95th percentile) to estimate extreme return levels ($\ge 204.5\text{ mm}$).
- **FR-14 (Generate Grid-Level Forecast):** The system shall export corrected forecasts and probabilistic fields onto a regular $0.25^\circ \times 0.25^\circ$ spatial grid across the Indian domain ($6^\circ-38^\circ\text{N}$, $68^\circ-98^\circ\text{E}$).
- **FR-15 (Aggregate Grid Predictions to District Level):** The system shall intersect grid forecasts with official Survey of India district boundaries using zonal operations to compute:
  - District Area-Weighted Mean Rainfall
  - District Maximum Peak Rainfall
  - District Exceedance Probability for Heavy / Very Heavy Rain
  - IMD Four-Color Warning Alert Level (Green, Yellow, Orange, Red).
- **FR-16 (Generate Uncertainty & Explanation Metadata):** The system shall calculate forecast spread ($P_{90} - P_{10}$) as an uncertainty proxy and generate local SHAP attributions identifying the top factors driving the correction.
- **FR-17 (Support Historical Replay):** The system shall allow execution over historical date intervals to simulate real-time operations and generate retrospective forecasts.
- **FR-18 (Calculate Operational Verification Metrics):** The system shall evaluate forecasts against ground-truth observations, computing RMSE, MAE, CSI, FAR, ETS, Brier Score, and Fractions Skill Score (FSS).
- **FR-19 (Store Eligible Errors for Scheduled Updates):** The system shall archive daily paired forecast-observation instances into an offline append-only retraining store for scheduled periodic model refreshes.

---

## 4. Non-Functional Requirements (NFR)

- **NFR-01 (Accuracy & Bias Correction):**
  - The system must achieve a reduction of $\ge 20\%$ in Root Mean Square Error (RMSE) compared to raw NWP over the monsoon test period.
  - The Critical Success Index (CSI) for Heavy Rainfall ($\ge 64.5\text{ mm}$) must show positive skill improvement over raw NWP.
- **NFR-02 (Reliability & Robustness):**
  - If optional atmospheric fields or recent error buffers are unavailable, the pipeline must fall back gracefully to baseline NWP + climatology without terminating.
  - Missing data sentinels (e.g., `-999.0`, `9999`) must never propagate into feature matrices.
- **NFR-03 (Interpretability & Transparency):**
  - Every forecast cycle must provide top feature attributions and active regime probability distributions.
  - No "black-box-only" predictions; forecasters must see $\Delta = \text{Corrected} - \text{Raw NWP}$.
- **NFR-04 (Scalability & Throughput):**
  - The inference pipeline must process the full pan-India grid ($\approx 15,600$ grid points $\times 5$ lead times) in under $180\text{ seconds}$ on standard multi-core compute.
- **NFR-05 (Maintainability & Modularity):**
  - Feature extraction, regime routing, quantile regressors, and GIS aggregation must reside in independent modules with zero cyclic dependencies.
  - Test coverage on data processing and transformation functions must exceed $80\%$.
- **NFR-06 (API Performance):**
  - Point-grid and district query endpoints must respond with $P_{95} \le 200\text{ ms}$ under concurrent load of 50 requests/sec.
- **NFR-07 (Data Validation & Contract Enforceability):**
  - All inter-module exchanges must be strictly validated against Pydantic schemas and interface contracts (`CONTRACT-DATA-001`, `CONTRACT-ML-002`, `CONTRACT-API-003`).
- **NFR-08 (Security & Isolation):**
  - Zero hardcoded credentials in source code.
  - Public endpoints must sanitize user inputs and restrict CORS access.
- **NFR-09 (Reproducibility & Lineage):**
  - Every trained model must log its exact lineage: Git commit SHA, config YAML hash, dataset time range, feature schema hash, and random seed.
- **NFR-10 (Deployment & Containerization):**
  - The application backend must run as a stateless containerized service (Docker) deployable on cloud or on-premise infrastructure.

---

## 5. Data Requirement Analysis

### Data Inventory Table

| Dataset | Variables | Spatial Resolution | Temporal Resolution | Source | Role | Required / Optional | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **NWP Precipitation (GFS/NCUM)** | Total 24h accumulated precipitation (`tp`) | $0.25^\circ \times 0.25^\circ$ (or $0.12^\circ$ regridded) | Daily (00 UTC run, Day 1 to Day 5 leads) | IMD / NCMRWF / NOAA AWS | Primary forecast baseline & predictor | **Required** | Access latency; format shifts between GRIB2 and NetCDF4 |
| **NWP Dynamic Wind Fields** | Zonal ($U$) and Meridional ($V$) wind at 850 hPa, 500 hPa, 200 hPa | $0.25^\circ \times 0.25^\circ$ | Daily / 6-hourly | IMD / NCMRWF / NOAA GFS | Low-level jet tracking, shear, WD detection | **Required** | Variable naming divergence; vertical level availability |
| **NWP Moisture & Pressure** | Relative Humidity ($RH_{850}$), Specific Humidity ($q$), MSLP, Geopotential ($Z_{500}$) | $0.25^\circ \times 0.25^\circ$ | Daily / 6-hourly | IMD / NCMRWF / NOAA GFS | Moisture flux convergence, regime classification | **Required** | Incomplete pressure levels in open portals (**TO BE VERIFIED**) |
| **Ground Truth Observed Rainfall** | 24h accumulated rainfall (03 UTC to 03 UTC) | $0.25^\circ \times 0.25^\circ$ | Daily | IMD Gridded Rainfall Archive (1901-present) | Ground truth target label, error memory computation, verification | **Required** | Latency of recent observation release; missing gauge interpolation artifacts |
| **High-Resolution Satellite Rainfall** | Precipitation rate / calibrated accumulation | $0.1^\circ \times 0.1^\circ$ | Daily / 3-hourly | NASA GPM IMERG / INSAT-3D | Near-real-time proxy observations when IMD gridded has release lag | **Optional** | Satellite infrared/radar bias over complex topography |
| **Digital Elevation Model (DEM)** | Surface elevation ($h$) | $\approx 90\text{m}$ (SRTM) aggregated to $0.25^\circ$ | Static | NASA SRTM / Copernicus DEM 30m | Orographic uplift, slope, aspect calculation | **Required** | Low risk; publicly available and static |
| **District Boundary Polygons** | Administrative district geometries, state names, district IDs | Vector polygon | Static (Survey of India 2024 update) | Survey of India / BharatMaps / Datameet | Spatial intersection, zonal statistics, alert mapping | **Required** | District name spelling mismatches; disputed or bifurcated boundaries |
| **Historical Climatology** | Day-of-year mean rainfall, 90th percentile baseline | $0.25^\circ \times 0.25^\circ$ | Daily climatology (30-year baseline 1991-2020) | Derived from IMD gridded archive | Climatological reference feature | **Required** | Requires pre-computation across historical archive |
| **Historical Synoptic Regimes Catalogue** | Classified historical monsoon dates (Active, Break, Low) | Pan-India / Regional | Daily | IMD Monsoon Reports / Objective wind-pressure index | Training target for regime classifier | **Required (Derived)** | Subjective manual catalog vs. objective automated index |
| **Trailing Forecast-Error Store** | $NWP - Obs$ error grids for lag 1 to 14 days | $0.25^\circ \times 0.25^\circ$ | Daily | RAIN-REPAIR X internal pipeline store | Short-term error memory and Error DNA | **Required** | Cold-start at beginning of season or missing observation days |

---

## 6. Data Availability & Feasibility

### Availability Classification

| Dataset Component | Availability Status | Feasibility Assessment |
| :--- | :---: | :--- |
| IMD Gridded Rainfall ($0.25^\circ$, 1901-2023) | **AVAILABLE** | Readily accessible in binary/NetCDF format from IMD Pune data portal. Essential for model training. |
| NOAA GFS Operational Forecast Archive ($0.25^\circ$) | **AVAILABLE** | Openly accessible via NOAA AWS Open Data S3 bucket (2015-present). Reliable public access. |
| NASA SRTM 90m / Copernicus DEM | **AVAILABLE** | Global open access. One-time offline download and raster processing to $0.25^\circ$. |
| Indian District Boundaries GeoJSON | **AVAILABLE** | Available via DataMeet open-source geospatial repository (standard WGS84). |
| NCMRWF NCUM Regional Forecast Data | **REQUIRES VERIFICATION** | Research portal access requires institutional credentials. GFS serves as reliable primary/fallback NWP source. |
| Real-time IMD AWS/ARG Station Telemetry | **REQUIRES VERIFICATION** | API access for real-time station feeds requires MoU or portal scraping. Handled via satellite proxy/lagged ground truth. |
| NASA GPM IMERG Final/Late Run | **AVAILABLE** | Free access via NASA Earthdata GES DISC. Serves as near-real-time observation fallback. |

### MVP Dataset vs. Advanced Dataset Partitioning

```
+----------------------------------------------------------------------------------------------------+
|                                    MVP DATASET (STRICTLY REQUIRED)                                 |
| 1. NOAA GFS 0.25° Daily NWP Forecasts (Precipitation, T2m, MSLP, U850, V850, RH850)               |
| 2. IMD 0.25° Daily Gridded Rainfall Observations (Target labels & Historical memory)               |
| 3. SRTM DEM Elevation, Slope, and Aspect (Aggregated to 0.25°)                                     |
| 4. Climatological Day-of-Year 30-Year Mean & 90th Percentile Grids                                 |
| 5. India District Administrative Boundary GeoJSON (EPSG:4326)                                      |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                  ADVANCED DATASET (ENHANCEMENT LAYER)                              |
| 1. NCMRWF High-Resolution Regional Model Forecasts (NCUM-R 4km)                                    |
| 2. Upper-Tropospheric Jet Fields: U200, V200, Z500 (Enhanced Western Disturbance skill)           |
| 3. High-Resolution Multi-Satellite Precipitation (NASA GPM IMERG 0.1° / INSAT-3D)                 |
| 4. Real-time Automatic Weather Station (AWS) Telemetry CSV feeds                                    |
| 5. River Basin & Hydrological Catchment Shapefiles (CWC Basins)                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 7. Feature Specification

### Feature Catalogue

| Group | Feature Identifier | Source | Meaning / Physical Formulation | Why Needed | Potential Leakage Risk | Consuming Model |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **A. NWP Features** | `nwp_precip_raw` | NWP model | Raw model predicted 24h precipitation (mm) | Core baseline predictor | None | Quantile Regressor, Prob Classifiers |
| | `nwp_precip_log` | Derived | $\ln(1 + \text{nwp\_precip\_raw})$ | Compresses heavy skewness for linear/tree splits | None | Quantile Regressor |
| | `nwp_t2m` | NWP model | 2-meter air temperature (K) | Surface thermal energy | None | Quantile Regressor |
| | `nwp_mslp` | NWP model | Mean sea-level pressure (hPa) | Identifies low-pressure systems | None | Regime Classifier, Quantile Regressor |
| **B. Atmospheric Features** | `u850`, `v850` | NWP model | Zonal & Meridional wind at 850 hPa (m/s) | Tracks monsoon westerly jet and cyclonic vorticity | None | Regime Classifier, Feature Core |
| | `wind_speed_850` | Derived | $\sqrt{u_{850}^2 + v_{850}^2}$ | Low-level jet kinetic intensity | None | Regime Classifier, Quantile Regressor |
| | `rh850` | NWP model | Relative humidity at 850 hPa (%) | Ambient moisture saturation level | None | Quantile Regressor, Prob Classifiers |
| | `u200`, `v200` | NWP model | Wind vectors at 200 hPa (m/s) | Upper-tropospheric easterly jet / WD troughs | None | Regime Classifier (WD detection) |
| **C. Physics-Guided Features** | `moisture_flux_conv` | Derived | $-\nabla \cdot (q_{850} \vec{V}_{850})$ | Strong physical precursor to deep convection | Spatial boundary edge artifacts | Quantile Regressor, Extreme Classifier |
| | `orographic_uplift` | Derived | $\vec{V}_{850} \cdot \nabla h$ | Direct mechanical lifting rate along mountain barriers | None | Quantile Regressor (Ghats/Himalayas) |
| | `bulk_wind_shear` | Derived | $|\vec{V}_{200} - \vec{V}_{850}|$ | Deep convective storm organization potential | None | Prob Extreme Classifier |
| | `vorticity_850` | Derived | $\frac{\partial v_{850}}{\partial x} - \frac{\partial u_{850}}{\partial y}$ | Cyclonic vortex rotation (Monsoon Depressions) | Edge gradient padding | Regime Classifier, Low/Depression |
| **D. Terrain Features** | `dem_elevation` | SRTM DEM | Terrain elevation above sea level ($h$, meters) | Altitude conditioning | None | Quantile Regressor |
| | `dem_slope` | SRTM DEM | Local terrain slope angle ($\alpha$, degrees) | Steepness of orographic barrier | None | Quantile Regressor |
| | `dem_aspect` | SRTM DEM | Terrain slope orientation azimuth ($\theta$, degrees) | Identifies windward vs. leeward rain-shadow faces | None | Quantile Regressor |
| | `dist_to_coast` | Vector GIS | Euclidean distance to closest coastline (km) | Coastal squall line moderation | None | Quantile Regressor |
| **E. Spatial-Neighbour** | `nwp_precip_neighbor_mean` | Derived | $3 \times 3$ grid cell spatial mean precipitation | Captures mesoscale spatial scale and buffer | Kernel edge leaks | Quantile Regressor |
| | `nwp_precip_neighbor_max` | Derived | $3 \times 3$ grid cell spatial maximum precipitation | Detects adjacent intense convective cells | Kernel edge leaks | Prob Heavy Rain Classifier |
| | `nwp_precip_gradient` | Derived | Spatial gradient magnitude $|\nabla \text{NWP}|$ | Identifies sharp frontal/convective boundaries | None | Quantile Regressor |
| **F. Climatology** | `clim_mean_doy` | IMD archive | 30-year historical mean rainfall for day-of-year | Grounding predictions in climatological reality | **HIGH:** Must compute on training years only | Quantile Regressor |
| | `clim_p90_doy` | IMD archive | 30-year 90th percentile rainfall for day-of-year | Identifies anomalies relative to local history | **HIGH:** Must compute on training years only | Prob Heavy/Very Heavy Classifiers |
| **G. Lead-Time** | `lead_time_hours` | NWP metadata | Forecast lead time ($24, 48, 72, 96, 120$) | Accounts for NWP forecast error growth with time | None | All Models |
| | `day_of_year_sin` | Derived | $\sin(2\pi \cdot \text{DOY} / 365.25)$ | Smooth seasonal progression cycle | None | All Models |
| | `day_of_year_cos` | Derived | $\cos(2\pi \cdot \text{DOY} / 365.25)$ | Smooth seasonal progression cycle | None | All Models |
| **H. Recent-Error Memory** | `error_memory_lag1` | Internal store | $\text{NWP}_{t-1} - \text{Obs}_{t-1}$ at grid cell | Captures persistent 24h model bias | **CRITICAL:** Observation from $t-1$ only | Quantile Regressor, Error DNA |
| | `error_memory_lag3_mean` | Internal store | Trailing 3-day average bias ($NWP - Obs$) | Short-term multi-day model drift | **CRITICAL:** Trailing window only | Quantile Regressor |
| | `error_memory_lag7_mean` | Internal store | Trailing 7-day average bias ($NWP - Obs$) | Synoptic-scale weekly model bias | **CRITICAL:** Trailing window only | Quantile Regressor |
| **I. Historical Analogs** | `analog_error_mean` | Feature Store | Average error of top-5 historical matching states | Historical error pattern recurrence | **CRITICAL:** Search historical training pool only | Quantile Regressor |
| **J. Regime Features** | `regime_prob_active` | Regime Model | Estimated soft probability of Active Monsoon | Dynamic model routing | None | Quantile Regressor, Soft Routing |
| | `regime_prob_break` | Regime Model | Estimated soft probability of Break Monsoon | Suppresses false peninsular rain | None | Quantile Regressor, Soft Routing |
| | `regime_prob_depression` | Regime Model | Estimated soft probability of Monsoon Depression | Amplifies central India rain shield | None | Quantile Regressor, Soft Routing |
| | `regime_prob_wd` | Regime Model | Estimated soft probability of Western Disturbance | Corrects northern India winter/spring storm | None | Quantile Regressor, Soft Routing |
| **K. Regime Transition** | `regime_transition_score` | Derived | $|\vec{\pi}_t - \vec{\pi}_{t-24h}|$ | Detects abrupt synoptic phase shifts | None | Quantile Regressor |

---

## 8. Temporal Leakage Design

### 8.1 Chronological Split Architecture
Meteorological data possesses strong temporal autocorrelation and seasonal cycles. Random $k$-fold cross-validation is **strictly prohibited** as it leaks future synoptic patterns into past predictions.

```
+───────────────────────────────────+──────────────────────+──────────────────────+
|          TRAIN SET                |   VALIDATION SET     |      TEST SET        |
|  Monsoon Seasons: 2018, 2019,     |  Monsoon Season:     |  Monsoon Season:     |
|         2020, 2021, 2022          |        2023          |        2024          |
|      (June 1 - September 30)      |  (June 1 - Sept 30)  |  (June 1 - Sept 30)  |
+───────────────────────────────────+──────────────────────+──────────────────────+
                     ▲                                      ▲                      ▲
                     │                                      │                      │
             Fit Transformers,                      Tune Hyperparameters,    Final Unseen
             Scalers & Models                       Select Thresholds        Evaluation
```

### 8.2 Strict Anti-Leakage Guardrails in Code

1. **Transformer & Scaler Isolation:**
   - Any normalization scaler (`StandardScaler`, `MinMaxScaler`) or target encodings must be fitted **exclusively on the training partition**:
     ```python
     scaler.fit(X_train)
     X_val_scaled = scaler.transform(X_val)
     X_test_scaled = scaler.transform(X_test)
     ```
   - Fitting scalers on the entire dataset prior to splitting will immediately fail automated QA inspection.

2. **Causal Rolling Error Memory Windows:**
   - At forecast initialization time $T_0$ (e.g., 2024-07-15 00:00 UTC), ground truth observations are only verified up to $T_0 - 24\text{ hours}$ (the 24-hour accumulation ending 03:00 UTC on 2024-07-14).
   - Error memory buffers:
     $$\text{Memory}_{\tau} = \text{NWP}_{T_0 - \tau} - \text{Obs}_{T_0 - \tau} \quad \text{for } \tau \ge 1\text{ day}$$
   - Any access to observation $t \ge T_0$ raises a fatal `DataLeakageViolationError`.

3. **Analog Search Corpus Restriction:**
   - The historical analog search database must contain candidate dates **strictly prior to the start of the validation partition**. Test year dates are never present in the search index.

4. **Purged / Embargo Cross-Validation for Hyperparameter Tuning:**
   - When tuning models across the training set, use **Time-Series Block Cross-Validation** with a 7-day embargo buffer between folds to eliminate memory buffer autocorrelation leakage.

---

## 9. Output Requirements

### 9.1 Grid-Level Output Schema (`0.25°` Resolution)

Every spatial grid cell $(lat, lon)$ at lead time $L$ outputs:

```json
{
  "grid_id": "G_1850_07375",
  "coordinates": { "latitude": 18.50, "longitude": 73.75 },
  "forecast_cycle": "2026-07-15T00:00:00Z",
  "lead_time_hours": 24,
  "valid_time_window": {
    "start_utc": "2026-07-15T03:00:00Z",
    "end_utc": "2026-07-16T03:00:00Z"
  },
  "raw_nwp_precip_mm": 48.2,
  "corrected_deterministic_p50_mm": 56.4,
  "quantiles": {
    "p10_mm": 39.1,
    "p50_mm": 56.4,
    "p75_mm": 67.8,
    "p90_mm": 78.2,
    "p95_mm": 86.5
  },
  "threshold_probabilities": {
    "prob_rain_ge_64_5mm": 0.42,
    "prob_rain_ge_115_6mm": 0.11,
    "prob_rain_ge_204_5mm": 0.015
  },
  "regime": {
    "dominant_regime": "ACTIVE_MONSOON",
    "regime_probabilities": {
      "ACTIVE_MONSOON": 0.82,
      "BREAK_MONSOON": 0.02,
      "MONSOON_DEPRESSION": 0.10,
      "WESTERN_DISTURBANCE": 0.01,
      "OFFSHORE_TROUGH": 0.04,
      "NORMAL_TRANSITIONAL": 0.01
    },
    "transition_flag": false
  },
  "uncertainty": {
    "iqr_mm": 28.7,
    "p90_p10_spread_mm": 39.1,
    "confidence_score": 0.85
  },
  "explanation": {
    "top_contributing_features": [
      { "name": "orographic_uplift", "shap_value": 5.4, "direction": "increase" },
      { "name": "error_memory_lag1", "shap_value": 3.8, "direction": "increase" },
      { "name": "rh850", "shap_value": 1.9, "direction": "increase" },
      { "name": "nwp_precip_raw", "shap_value": -2.1, "direction": "decrease" }
    ]
  }
}
```

### 9.2 District-Level Output Schema

Every administrative district outputs a structured operational record:

```json
{
  "district_id": "MH_PUNE",
  "district_name": "Pune",
  "state_name": "Maharashtra",
  "lead_time_hours": 24,
  "valid_date": "2026-07-16",
  "aggregation_metrics": {
    "area_weighted_mean_rainfall_mm": 42.6,
    "peak_district_rainfall_mm": 118.2,
    "raw_nwp_mean_rainfall_mm": 33.1,
    "bias_correction_delta_mm": 9.5
  },
  "risk_assessment": {
    "prob_heavy_rain_anywhere": 0.68,
    "prob_very_heavy_rain_anywhere": 0.24,
    "prob_extreme_rain_anywhere": 0.04,
    "imd_warning_level": "ORANGE",
    "recommended_action": "Be prepared: Intense localized rainfall likely along Western Ghats catchment zones."
  },
  "synoptic_context": {
    "dominant_regime": "ACTIVE_MONSOON",
    "regime_confidence": 0.82
  }
}
```

---

## 10. Validation Requirements

### Evaluation Framework Matrix

| Subsystem | Evaluation Metric | Mathematical Formulation | Evaluation Purpose | Benchmark / Target |
| :--- | :--- | :--- | :--- | :--- |
| **Regime Classifier** | Multi-class Macro F1 | $\frac{1}{K} \sum_{k=1}^K F1_k$ | Evaluates balanced regime detection across frequent and rare regimes | Macro $F1 \ge 0.75$ |
| | Confusion Matrix | $N \times N$ contingency | Analyzes confusion between adjacent synoptic states | Zero Active $\leftrightarrow$ Break misclassifications |
| **Deterministic Rainfall** | Root Mean Square Error (RMSE) | $\sqrt{\frac{1}{N} \sum (y - \hat{y})^2}$ | Penalizes large quantitative rainfall errors | $\ge 20\%$ reduction vs. Raw NWP |
| | Mean Absolute Error (MAE) | $\frac{1}{N} \sum \|y - \hat{y}\|$ | Measures average forecast error magnitude | $\ge 20\%$ reduction vs. Raw NWP |
| | Mean Bias Error (MBE) | $\frac{1}{N} \sum (\hat{y} - y)$ | Verifies elimination of systematic over/under-prediction | MBE within $\pm 1.0\text{ mm}$ |
| **Categorical Extreme Events** | Probability of Detection (POD) | $\frac{\text{Hits}}{\text{Hits} + \text{Misses}}$ | Hit rate for events $\ge 64.5$ mm and $\ge 115.6$ mm | POD $\ge 0.70$ for Heavy Rain |
| | False Alarm Ratio (FAR) | $\frac{\text{False Alarms}}{\text{Hits} + \text{False Alarms}}$ | Rate of unverified extreme warnings | FAR reduction $\ge 25\%$ vs. NWP |
| | Critical Success Index (CSI) | $\frac{\text{Hits}}{\text{Hits} + \text{Misses} + \text{False Alarms}}$ | Balanced metric for rare severe events | Statistically significant improvement over NWP |
| | Equitable Threat Score (ETS) | $\frac{\text{Hits} - \text{Hits}_{random}}{\text{Hits} + \text{Misses} + \text{FA} - \text{Hits}_{random}}$ | CSI penalized for random chance hits | ETS $> \text{ETS}_{raw}$ across all lead times |
| **Spatial Verification** | Fractions Skill Score (FSS) | $1 - \frac{\text{MSE}_{(spatial)}}{\text{MSE}_{(worst)}}$ | Evaluates spatial displacement over spatial neighborhood windows ($25\text{km} - 100\text{km}$) | FSS $\ge 0.85$ at $50\text{km}$ scale |
| **Probabilistic Quality** | Brier Score (BS) | $\frac{1}{N} \sum (p_i - o_i)^2$ | Evaluates calibrated probability of threshold exceedance | BS $\le 0.15$ for Heavy Rain |
| | Continuous Ranked Probability Score (CRPS) | $\int_{-\infty}^{\infty} (F(y) - H(y - y_{obs}))^2 dy$ | Measures overall quality of continuous probabilistic forecast | Lower than Raw NWP ensemble |
| | Quantile Calibration / Coverage | $\frac{1}{N} \sum \mathbb{I}(y_{obs} \le \hat{q}_\tau) \approx \tau$ | Verifies nominal vs. observed coverage for $P_{10}, P_{50}, P_{90}$ | Within $\pm 3\%$ of nominal rate |

### Stratified Evaluation Mandate
Every metric must be calculated and reported across distinct strata:
1. **Regime-Stratified:** Separate scores for Active Monsoon, Break Monsoon, Low/Depression, Western Disturbance, and Normal.
2. **Terrain-Stratified:** Separate scores for Western Ghats (Orographic), Indo-Gangetic Plains, Coastal Plains, and Deccan Plateau.
3. **Threshold-Stratified:** Light ($< 7.5\text{ mm}$), Moderate ($7.5 - 64.5\text{ mm}$), Heavy ($64.5 - 115.6\text{ mm}$), and Very Heavy ($\ge 115.6\text{ mm}$).

---

## 11. Baseline Experiment Plan

To rigorously demonstrate scientific and engineering value, RAIN-REPAIR X establishes a 6-tier comparative experiment plan:

```
[EXPERIMENT 1: Raw NWP Baseline]
  - Raw uncorrected precipitation direct from Numerical Weather Prediction model.
  - Establishes the ground-floor baseline.
        ↓
[EXPERIMENT 2: Traditional Statistical Bias Correction]
  - Standard Quantile Mapping (QM) and linear scaling applied per grid cell.
  - Tests whether complex ML is even justified over classical climatological mapping.
        ↓
[EXPERIMENT 3: Generic ML Post-Processing (Ablation)]
  - GBDT (LightGBM/XGBoost) trained on raw NWP + standard geographical coordinates.
  - No regime features, no physics features, no rolling error memory.
        ↓
[EXPERIMENT 4: Regime-Aware Post-Processing (Ablation)]
  - GBDT with soft weather-regime classification features added.
  - Quantifies the isolated performance contribution of regime conditioning.
        ↓
[EXPERIMENT 5: Full RAIN-REPAIR X (Candidate Core)]
  - Physics features + Soft Regimes + 3/7/14-day Error Memory + Historical Analogs + Quantiles.
  - Primary production candidate system.
        ↓
[EXPERIMENT 6: RAIN-REPAIR X + EVT/GPD (Advanced Extreme Tail)]
  - Core system supplemented with Generalized Pareto Distribution for tail residuals.
  - Tests whether extreme tail extrapolation improves Very Heavy / Extreme rainfall skill.
```

---

## 12. MVP vs. Advanced Scope

### Scope Boundary Matrix

| System Component | MVP Implementation (Must Build for SIH) | Advanced Module (Should Build if Time Permits) | Future Research (Post-SIH) |
| :--- | :--- | :--- | :--- |
| **NWP Input Source** | NOAA GFS $0.25^\circ$ open data archive & daily pipeline | Multi-model ensemble (GFS + NCUM + ECMWF IFS) | Radar assimilation & Nowcasting feed |
| **Observation Target** | IMD $0.25^\circ$ Gridded Rainfall (NetCDF) | NASA GPM IMERG $0.1^\circ$ near-real-time satellite feed | Automated Weather Station (AWS) crowd-sourced telemetry |
| **Terrain Features** | SRTM DEM: Elevation, Slope, Aspect ($0.25^\circ$) | High-resolution Topographic Position Index (TPI) & roughness | Land-use / land-cover (LULC) urbanization indices |
| **Physics Features** | Orographic uplift index, moisture convergence proxy | Complete 3D thermodynamic CAPE/CIN/divergence profiling | Cloud microphysics parameterizations |
| **Weather Regimes** | 5 discrete monsoon regimes via objective indices + GBDT classifier | Soft probabilistic mixture routing + transition alerts | Unsupervised continuous latent regime autoencoder |
| **Error Memory** | Rolling 1, 3, and 7-day spatial error memory buffers | Trailing 14-day memory + Top-5 historical analogs | Dynamic neural memory networks |
| **Quantile Engine** | LightGBM pinball loss ($P_{10}, P_{50}, P_{75}, P_{90}, P_{95}$) with monotonic sorting | Non-crossing Quantile Neural Networks / Conformal prediction | Diffusion probabilistic weather generation |
| **Extreme Tail** | Calibrated binary classification for $\ge 64.5\text{ mm}$ and $\ge 115.6\text{ mm}$ | **EVT / Generalized Pareto Distribution (GPD)** module | Non-stationary spatial extreme value processes |
| **GIS Aggregation** | Zonal mean/max on official District GeoJSON | Sub-district (Tehsil/Taluka) aggregation | Hydrological catchment / dam reservoir inflow boundaries |
| **Explainability** | Tree SHAP feature attributions on point queries | Global SHAP summary dashboards and regime interaction plots | Counterfactual meteorological explanations |
| **Web Dashboard** | React + Leaflet dual map comparison, meteograms, alerts | Animated wind streamline overlays and raster sliders | Full 3D WebGL atmospheric rendering |

---

## 13. Risk & Mitigation Table

| Risk Identifier | Risk Description | Prob. | Impact | Early Detection Mechanism | Mitigation Strategy | Fallback Plan |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **RSK-01** | **IMD Observation Release Lag**<br>Recent ground truth observations delayed by days/weeks. | High | High | Ingestion pipeline checks date gap of latest observed netCDF. | Implement dual-mode: use IMD gridded for historical training; use NASA GPM IMERG or lagged NWP error for near-real-time operations. | Fall back to trailing 7-day memory or static climatological error profile. |
| **RSK-02** | **Inconsistent NWP Variable Availability**<br>Missing isobaric levels (e.g. 200 hPa winds missing from portal). | Med | High | Schema validation checks required variables during raw ingestion. | Train feature extractors with feature-dropout robustness; treat upper levels as optional. | System gracefully drops upper-level features and executes using surface + 850 hPa inputs. |
| **RSK-03** | **Extreme Class Imbalance**<br>Rainfall $\ge 115.6\text{ mm}$ accounts for $< 0.5\%$ of historical grid points. | High | High | Label distribution check in dataset pipeline. | Use focal loss, cost-sensitive learning, and sample re-weighting for extreme convective bins. | Rely on quantile regression ($P_{90}, P_{95}$) rather than standalone binary threshold classifiers. |
| **RSK-04** | **Regime Label Ambiguity**<br>Transitional synoptic states do not fit clean Active/Break definitions. | High | Med | Soft regime entropy check ($\sum -\pi_k \log \pi_k$). | Use soft probabilistic vectors $\vec{\pi}$ rather than hard discrete regime assignment. | Default to `NORMAL_TRANSITIONAL` regime when entropy exceeds confidence threshold. |
| **RSK-05** | **Quantile Crossing**<br>Independent quantile models yield $P_{90} < P_{50}$ in high-uncertainty regions. | Med | High | Automated invariant test in `src/postprocessing/constraints.py`. | Apply isotonic rearrangement/sorting on output quantile vectors: $\tilde{Q} = \text{sort}(Q)$. | Invariant enforcement guarantees valid outputs before database or API ingestion. |
| **RSK-06** | **Spatial Regridding Artifacts**<br>Interpolation across coastal or mountain boundaries causes smearing. | Low | Med | Visual inspection of gradient maps across Western Ghats transects. | Use conservative or bilinear area-weighted regridding rather than simple nearest-neighbor. | Mask water cells using official high-resolution land-sea mask. |
| **RSK-07** | **District Polygon Geometry Mismatch**<br>Bifurcated or newly formed districts missing from shapefile. | Med | Low | Pre-flight validation against official Local Government Directory (LGD). | Use 2024 standardized Survey of India / Datameet district boundary GeoJSON with LGD codes. | Map unrecognized polygons to parent administrative division. |
| **RSK-08** | **API Latency on Real-Time Queries**<br>On-the-fly spatial aggregation chokes under load. | Med | Med | Latency monitoring in `/api/v1/forecast/*`. | Pre-compute all district zonal metrics during the batch post-processing pipeline; API only reads indexed store. | Return cached static GeoJSON if live calculation exceeds 500 ms. |

---

## 14. Senior Developer Review

As Senior Technical Lead and System Architect, I have reviewed this detailed requirement and data specification against production engineering standards and competition constraints:

1. **Clarity & Completeness:** The progression from meteorological phenomena (monsoon regimes, orographic lift) to mathematical formulations and Pydantic schemas is comprehensive and unambiguous.
2. **Avoidance of Unnecessary Complexity:** 
   - We have resisted the urge to prematurely mandate distributed Kafka queues or multi-region Kubernetes clusters.
   - We have kept **EVT/GPD** appropriately scoped as an *advanced modular plug-in* (Model 6), ensuring the core system (Models 1-5) can be completed, trained, and verified independently.
3. **Data Feasibility:** The reliance on open NOAA GFS and historical IMD gridded archives guarantees that the team is not blocked by external private database approvals during SIH development.
4. **Leakage & Physical Integrity:** The temporal split architecture (Years 2018-2022 train, 2023 validation, 2024 test) and explicit anti-leakage coding rules eliminate the single most common failure mode in academic ML weather projects.
5. **Architectural Coherence:** Every functional requirement directly maps to files already scaffolded in PART 0 (`src/data`, `src/features`, `src/regime`, `src/models`, `src/postprocessing`, `backend`, `frontend`, `tests`).

---

## 15. APPROVED REQUIREMENTS FOR PART 2

The technical requirements, data inventory, feature definitions, and anti-leakage constraints specified herein are formally approved. The project is completely aligned to proceed to **PART 2: DATA ENGINEERING, PIPELINE IMPLEMENTATION & SYNTHETIC BENCHMARK FIXTURES**.
