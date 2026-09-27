# Frontend Dashboard — RAIN-REPAIR X (VARSHAA)

## Overview
Operational meteorologist and disaster management decision support dashboard.

## Planned Capabilities:
1. **Interactive Dual-Map Canvas:** Side-by-side or slider comparison of Raw NWP vs. RAIN-REPAIR X Post-Processed forecasts.
2. **Administrative District Warning Matrix:** Color-coded alert map (Green / Yellow / Orange / Red) based on probability of heavy rainfall exceedance.
3. **Meteogram Station Drill-Down:** Quantile forecast plumes (P10, P50, P90) over lead times (Day 1 to Day 5).
4. **Model Transparency Card:** Local feature attribution (SHAP) and active synoptic regime tags.

## Architecture & Integration:
- Consumes RESTful JSON/GeoJSON from the backend API (`/api/v1/`).
- Fully decoupled from ML models and raw data formats.
- Framework: Modern React + Vite / Leaflet / MapLibre.
