# RAIN-REPAIR X: PART 7 — Probabilistic Post-Processing, Quantile Plumes, Extreme Value Theory (EVT-GPD) & Calibrated Risk

**Status:** IMPLEMENTED  
**Layer:** Probabilistic Post-Processing & Extreme Risk (`src/postprocessing/`)  
**Interface Contract:** CONTRACT-ML-002  

---

## 1. Executive Summary & Flow

PART 7 transitions RAIN-REPAIR X from deterministic error repair to a fully calibrated probabilistic risk forecasting system:

```
                  PART 6: Deterministic Corrected Rainfall (P50)
                                      ↓
                  PART 7: Probabilistic & Extreme Value Engine
                                      ↓
       ┌──────────────────────────────┼──────────────────────────────┐
       ↓                              ↓                              ↓
Multi-Quantile Regression    Calibrated Exceedance        Extreme Value Theory
(P10 / P50 / P75 / P90 / P95)  (>=64.5, 115.6, 204.5mm)   (EVT-GPD Tail Survival)
       │                              │                              │
       └──────────────────────────────┼──────────────────────────────┘
                                      ↓
                      Physical Realizability Invariants
                  (Non-Negativity + Monotonic Rearrangement)
                                      ↓
                      Calibrated IMD Warning Risk Output
                     (Green, Yellow, Orange, Red Alerts)
```

---

## 2. Multi-Quantile Regression Plumes (P10, P50, P75, P90, P95)

Implemented in `src/postprocessing/quantiles.py` via `MultiQuantileRegressor`:
- **Optimization:** LightGBM regressors minimizing asymmetric Pinball Loss per confidence bound:
  $$\mathcal{L}_\alpha(y, q) = \max\left(\alpha (y - q), \; (1 - \alpha)(q - y)\right)$$
- **Canonical Quantiles:**
  - $\mathbf{P_{10}}$: Conservative lower bound (optimistic scenario).
  - $\mathbf{P_{50}}$: Median expectation (calibrated deterministic reference).
  - $\mathbf{P_{75}}$: Moderate heavy rain risk ceiling.
  - $\mathbf{P_{90}}$: High-convection alert threshold.
  - $\mathbf{P_{95}}$: Extreme convective flood envelope.

---

## 3. Physical Invariants & Monotonic Rearrangement

Implemented in `src/postprocessing/constraints.py`:
- **Quantile Crossing Elimination:** Applies the Rearrangement Operator (Chernozhukov et al.) to ensure non-crossing:
  $$0.0 \le Q_{10} \le Q_{50} \le Q_{75} \le Q_{90} \le Q_{95}$$
- **Threshold Probability Monotonicity:** Enforces cumulative inclusion constraints:
  $$0.0 \le P(\ge 204.5\text{ mm}) \le P(\ge 115.6\text{ mm}) \le P(\ge 64.5\text{ mm}) \le 1.0$$
- **Physical Realizability Validator:** Automated check ensuring zero negative precipitation and zero probability out-of-bound errors.

---

## 4. Extreme Threshold Exceedance Classifiers

Implemented in `src/postprocessing/thresholds.py` via `ExtremeThresholdClassifier`:
- **IMD Operational Categories:**
  - Heavy Rainfall: $P(\text{Rain} \ge 64.5\text{ mm/24h})$
  - Very Heavy Rainfall: $P(\text{Rain} \ge 115.6\text{ mm/24h})$
  - Extremely Heavy Rainfall: $P(\text{Rain} \ge 204.5\text{ mm/24h})$
- **Imbalance Handling:** Balanced cost-sensitive weighting mitigating rare-event class distortion.
- **Calibration:** Platt scaling (sigmoid mapping) to align raw logits with empirical verification frequencies.

---

## 5. Extreme Value Theory (EVT) & Generalized Pareto Distribution (GPD)

Implemented in `src/postprocessing/evt.py` via `EVTGPDTailModel`:
- **Peaks-Over-Threshold (POT):** Models asymptotic survival probability of extreme rainfall excesses beyond high thresholds ($u = 64.5\text{ mm}$):
  $$P(Y > y \mid Y > u) = \left[1 + \frac{\xi (y - u)}{\sigma}\right]^{-1/\xi}$$
  - $\xi$ (shape parameter / tail heaviness): Constrained within $[-0.10, 0.50]$ for meteorological realism. Positive $\xi > 0$ models heavy convective tails (Fréchet type).
  - $\sigma$ (scale parameter): Dispersion of extreme surges.
- **Catastrophic Extrapolation:** Computes rare return levels ($P_{99}$, $P_{99.5}$, 100-year events) well beyond the empirical maximum of short training datasets.

---

## 6. IMD 4-Tier Operational Alert Assignment

Implemented in `src/postprocessing/risk.py`:
- **GREEN (Normal):** $P(\ge 64.5\text{ mm}) < 0.25$ and $P_{90} < 64.5\text{ mm}$. No advisory needed.
- **YELLOW (Watch):** $P(\ge 64.5\text{ mm}) \ge 0.25$ or $P_{90} \ge 64.5\text{ mm}$. Be updated.
- **ORANGE (Alert):** $P(\ge 64.5\text{ mm}) \ge 0.65$ or $P(\ge 115.6\text{ mm}) \ge 0.35$ or $P_{90} \ge 115.6\text{ mm}$. Be prepared for localized inundation.
- **RED (Warning):** $P(\ge 115.6\text{ mm}) \ge 0.60$ or $P(\ge 204.5\text{ mm}) \ge 0.35$ or $P_{90} \ge 204.5\text{ mm}$. Take defensive action.

---

## 7. Probabilistic Verification Suite

Implemented in `src/postprocessing/probabilistic_eval.py`:
- **Continuous Ranked Probability Score (CRPS):** Quantile-integral approximation measuring overall distributional quality.
- **Brier Score (BS) & Brier Skill Score (BSS):** Threshold-specific probability skill evaluated against climatology.
- **Quantile Coverage & Reliability:** Comparison between nominal confidence levels and observed empirical coverage rates.
