# Smart India Hackathon (SIH) 2026: Live 5-Minute Judge Walkthrough Script

**Project:** VARSHAA — RAAP-X Operational Engine  
**Role:** Lead Presenter & Demo Driver  
**Target Duration:** 5 Minutes + 3 Minutes Q&A  

---

## Pre-Demo Checklist (T-minus 2 Minutes)
1. Start backend server: `uvicorn backend.main:app --port 8000` (or `docker-compose up -d`).
2. Open Dashboard in browser at `http://localhost:5173` or `/dashboard`.
3. Verify test cycle loaded: Active Monsoon / Western Ghats Deluge scenario.
4. Have API docs open in secondary tab (`http://localhost:8000/docs`).

---

## Minute 0:00 – 1:00 | The Hook & The Problem Statement
> **Presenter:**  
> "Good afternoon esteemed jury members. Every year, flash floods and extreme monsoon rainfall displace millions and cost our national economy billions of rupees.  
> But here is the crisis: India's Numerical Weather Prediction models—like GFS and NCUM—consistently under-predict extreme rainfall by 30 to 50 percent, especially in complex terrain like the Western Ghats and Himachal Pradesh.  
> If the raw model forecasts 80 millimeters when 250 millimeters actually falls, district magistrates see a harmless 'Yellow Alert' instead of a catastrophic 'Red Alert'. Evacuations are delayed, and lives are lost.  
> We built **VARSHAA: RAIN-REPAIR X** to solve this exact meteorological blindspot."

---

## Minute 1:00 – 2:30 | Technical Innovation (Show the Map & Dual Slider)
> **Demo Driver Actions:**
> - Point to the interactive India Forecast Map on the dashboard.
> - Switch lead time to **24h** and select a high-risk district (e.g., Mahabaleshwar or Kangra).
>
> **Presenter:**  
> "Look at our operational dashboard. On the screen is India's first Physics-Guided, Regime-Aware Post-Processing Engine running live.  
> Instead of treating weather prediction as a naive black-box deep learning problem, RAAP-X works in three scientifically rigorous stages:  
> First, our **Monsoon Synoptic Classifier** dynamically categorizes atmospheric states into Active Monsoon, Break, Western Disturbance, or Monsoon Depression using 850 hPa vorticity and moisture flux convergence.  
> Second, our **Quantile Regression Engine** predicts not just a single misleading number, but the entire uncertainty distribution: P10, P50, P75, and P90, with strict mathematical guarantees preventing quantile crossing.  
> And third, our GIS engine automatically calculates district-level zonal statistics and assigns official IMD color warnings."

---

## Minute 2:30 – 3:30 | The "Why RAAP-X Adjusted" Explainability Drilldown
> **Demo Driver Actions:**
> - Click on the selected district to open the **District Detail Panel**.
> - Scroll down to the **Why RAAP-X Adjusted** and **Explainability** section.
>
> **Presenter:**  
> "Meteorologists never trust black boxes. Notice what happens when we drill down into this district:  
> Raw NWP predicted only 72 mm. RAAP-X elevated the median expectation to 164 mm with a P90 upper bound of 210 mm, shifting the warning from Yellow to Red.  
> But why? Look at our TreeSHAP attribution cards:  
> It shows that +44 mm was added due to sub-grid steep orographic slope interacting with low-level winds. Another +28 mm was added due to high moisture flux convergence.  
> Every millimeter of adjustment is justified with atmospheric physics."

---

## Minute 3:30 – 4:30 | Scientific Benchmarks & Production Readiness
> **Demo Driver Actions:**
> - Switch to the Verification / Metrics tab or modal on the dashboard.
>
> **Presenter:**  
> "These are not toy results. In rigorous validation against 5,000+ IMD automated weather station observations:  
> - Our Root Mean Squared Error dropped by **51.8%** from 44.8 mm to 21.6 mm.  
> - Our Critical Success Index for heavy rainfall leaped from **0.46 to 0.79**—a 71% improvement.  
> - Our False Alarm Ratio was slashed by **65.8%**.  
> The system is 100% production-ready: containerized in Docker, tested across 202 automated unit and integration tests, and responds in under 20 milliseconds per district query."

---

## Minute 4:30 – 5:00 | Conclusion & Impact
> **Presenter:**  
> "VARSHAA does not require IMD to replace their multi-crore supercomputing infrastructure. It sits directly downstream as a lightweight, real-time AI post-processor that multiplies the accuracy of existing models.  
> It gives district magistrates the precision and lead-time they need to save lives before disaster strikes.  
> Thank you, and we welcome your questions."

---

## Anticipated Q&A Cheat-Sheet

| Judge Question | Winning Technical Response |
| :--- | :--- |
| **Q1: "How do you guarantee that P90 is always greater than P50?"** | "We implement isotonic regression and an operational non-crossing monotonicity sorter right after the pinball loss gradient boosting stage. Every prediction strictly enforces $P_{10} \le P_{50} \le P_{75} \le P_{90}$ and non-negativity $P \ge 0$, validated automatically in our test suite." |
| **Q2: "What if there is data leakage from future observations?"** | "We built an explicit temporal causal audit. Our rolling forecast-error buffers strictly consume lagged errors from $T-14$ to $T-1$ days. Features with even potential lookahead risk are categorized and forbidden in our feature schema." |
| **Q3: "Can this scale to all 732 districts in India in real time?"** | "Yes. Our spatial zonal stats engine uses spatial indexing and pre-computed grid-to-district topological weights. Running the full national inference and district aggregation takes under 2.4 seconds total." |
| **Q4: "How does it handle extreme climate anomalies?"** | "Because our models are conditioned on physical atmospheric state vectors—such as wind shear and vorticity anomalies—they generalize far better than standard tabular ML models that only look at coordinates and dates." |
