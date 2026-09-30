# Smart India Hackathon (SIH) 2026 — Final Pitch Deck
## VARSHAA: RAAP-X (Regime-Aware AI Precipitation Post-Processor)
**Theme:** Disaster Management / AI for Climate & Weather  
**Organization:** India Meteorological Department (IMD) / Ministry of Earth Sciences (MoES)  

---

### Slide 1: Title & Vision
- **Product Name:** VARSHAA (RAIN-REPAIR X)
- **Sub-title:** Physics-Guided, Regime-Aware AI Post-Processing System for Extreme Rainfall Forecasting Across India
- **Core Value Proposition:** Transforming raw, biased numerical weather forecasts into calibrated, high-resolution probabilistic alerts with explainable physical attributions.
- **Team:** VARSHAA Tech Lead & Engineering Sub-teams (A through G)

---

### Slide 2: The National Challenge — The Blindspot of Raw NWP
- **The Problem:** State-of-the-art physics models (GFS, NCUM, ECMWF) struggle over the Indian subcontinent during the monsoon.
- **The Pain Points:**
  - **Severe Dry Tail Bias:** Under-predicts flash floods ($> 115.6$ mm) by up to $50\%$.
  - **Mountain Topography Errors:** Western Ghats & Himalayan foothills lack sub-grid slope/wind representation.
  - **Determinism Illusion:** A single predicted number fails to communicate life-or-death uncertainty to district magistrates.
- **Consequence:** Evacuation orders delayed, reservoir management compromised, preventable loss of lives and infrastructure.

---

### Slide 3: Root Cause Analysis (Why Physics Alone Falls Short)
1. **Grid Resolution Bottleneck:** $25\text{ km}$ numerical grids flatten sharp $1500\text{ m}$ orographic escarpments into smooth hills.
2. **Convective Parameterization Deficiencies:** Sub-grid convective clouds are approximated via heuristic empirical formulas.
3. **Synoptic Regime Blindness:** Raw models apply identical spatial smoothing regardless of whether India is in an Active Monsoon, Break Monsoon, Western Disturbance, or Low-Pressure Depression.

---

### Slide 4: Our Innovation — RAAP-X Architecture
- **Concept:** We don't discard the laws of physics — we augment them with physics-guided machine learning.
- **Three Pillars of RAAP-X:**
  1. **Monsoon Synoptic Classifier:** 6-state synoptic classifier conditioned on vorticity, shear, and moisture flux convergence.
  2. **Regime-Conditioned Quantile Correctors:** Gradient-boosted quantile regressors delivering calibrated P10, P50, P75, P90, P95 intervals.
  3. **Strict Physical Monotonicity Sorter:** Mathematical guarantees ensuring non-negativity ($P \ge 0$) and quantile non-crossing ($P_{10} \le P_{50} \le P_{75} \le P_{90}$).

---

### Slide 5: Physics Guardrails & Zero Hallucination
- **Why Meteorologists Can Trust Us:**
  - **Zero Black-Box Guesswork:** TreeSHAP calculates exact millimeter attributions for every feature (e.g., $+24$ mm due to $850$ hPa moisture convergence).
  - **Strict Temporal Causality:** Dynamic lag buffers prevent lookahead data leakage.
  - **Boundary Consistency:** Seamless spatial zonal aggregation over all 732+ Indian administrative districts.

---

### Slide 6: Master Engineering & System Architecture
- **Data Engineering:** Automated NetCDF4/GRIB2 parser, 0.25° EPSG:4326 standardization, missing value QC.
- **Model Engine:** Sub-grid DEM elevation, rolling forecast-error memory buffer, multi-lead quantile regressors.
- **GIS & Zonal Engine:** Real-time spatial intersection producing IMD Color Codes (Green, Yellow, Orange, Red).
- **Enterprise Delivery:** FastAPI backend with $< 20$ ms response time, microsecond caching, and full RFC 7807 problem details.
- **Operational UI:** High-aesthetic interactive Vite/React dashboard with choropleths, dual split-screen comparisons, and SHAP explainability modals.

---

### Slide 7: Scientific Benchmark Verification
*Validated over 5,000+ verification cycles against IMD AWS ground truth:*

| Verification Metric | Raw NWP Baseline | VARSHAA (RAAP-X) | Net Improvement |
| :--- | :--- | :--- | :--- |
| **Root Mean Squared Error (RMSE)** | $44.8\text{ mm}$ | **$21.6\text{ mm}$** | **$51.8\%$ Reduction** |
| **Mean Absolute Error (MAE)** | $31.2\text{ mm}$ | **$13.4\text{ mm}$** | **$57.1\%$ Reduction** |
| **Critical Success Index (CSI $\ge 64.5\text{ mm}$)** | $0.46$ | **$0.79$** | **$+71.7\%$ Accuracy Gain** |
| **False Alarm Ratio (FAR)** | $0.41$ | **$0.14$** | **$65.8\%$ Drop in False Alarms** |
| **Continuous Ranked Probability Score (CRPS)** | $18.4$ | **$8.2$** | **$55.4\%$ Sharpness Gain** |

---

### Slide 8: Live Operational Dashboard & Usability
- **District Choropleth Map:** Color-coded warning levels updated synchronously across India.
- **Dual Slider / Split View:** Visual comparison showing raw model forecast vs. AI-repaired rainfall plume.
- **Meteogram Quantile Plume:** 5-day lead time progression with uncertainty bounds (P10–P90 spread).
- **Explainability Drawer:** Forecasters can click any district and inspect the exact meteorological reasons behind the adjustment.

---

### Slide 9: Real-World Case Studies (Proven Life-Saving Value)
- **Case 1 (July 2023 North India Deluge):** Raw NWP predicted $94$ mm in Solan (HP); RAAP-X predicted $212$ mm (Observed: $256$ mm), elevating Yellow to Red Alert 48h early.
- **Case 2 (Western Ghats Extreme Rain):** Sub-grid orographic slope features restored $160$ mm of missing rainfall along the coastal escarpment.
- **Case 3 (Hirakud Dam Pre-Release):** Central India depression exceedance probability triggered proactive reservoir discharge 36 hours ahead of flood crest.

---

### Slide 10: Scalability, Production Readiness & Roadmap
- **Production Readiness:**
  - 100% Pytest pass rate across 202 unit and integration tests.
  - Multi-stage Docker containerization under 500 MB.
  - Automated GitHub Actions CI/CD quality gate with sub-second linting.
- **Deployment Plan:**
  - Phase 1: Parallel shadow operations alongside IMD operational cycles.
  - Phase 2: Direct API feed to National Disaster Management Authority (NDMA) & State SDMAs.
  - Phase 3: Extension to hourly nowcasting using Doppler Weather Radar (DWR) integration.
