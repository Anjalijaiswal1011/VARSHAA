import React from 'react';
import { ArrowDown } from 'lucide-react';
import type { CanonicalDistrictForecastRecord } from '../types/forecast';

interface WhyRaapxStoryFlowProps {
  district: CanonicalDistrictForecastRecord | null;
  leadTime: number;
}

export const WhyRaapxStoryFlow: React.FC<WhyRaapxStoryFlowProps> = ({ district, leadTime }) => {
  if (!district) return null;

  const rawNwp = district.raw_nwp_rainfall !== undefined && district.raw_nwp_rainfall !== null ? district.raw_nwp_rainfall : null;
  const p50 = district.corrected_p50 !== undefined && district.corrected_p50 !== null ? district.corrected_p50 : null;

  const hasRawNwp = rawNwp !== null && !isNaN(rawNwp);
  const hasP50 = p50 !== null && !isNaN(p50);

  const diff = hasRawNwp && hasP50
    ? (district.rainfall?.difference !== undefined ? district.rainfall.difference : Math.round((p50 - rawNwp) * 100) / 100)
    : null;

  // Soft regime probabilities from backend (normalize keys case-insensitively)
  const rawRegProbs = district.regime?.probabilities || district.regime_probabilities || {};
  const regProbsNormalized = new Map<string, number>();
  for (const [k, v] of Object.entries(rawRegProbs)) {
    if (typeof v === 'number') {
      regProbsNormalized.set(k.toUpperCase(), v);
    }
  }

  const dominantRegime = district.dominant_regime || 'ACTIVE_MONSOON';
  const dominantProb = regProbsNormalized.get(dominantRegime.toUpperCase()) ?? district.regime?.dominant_probability ?? null;

  // Recent NWP Error Memory directly from real backend output
  const errorMem = district.recent_error_memory || district.explanation?.recent_error_memory || null;
  const err3d = errorMem?.error_3day_mm !== undefined && errorMem?.error_3day_mm !== null ? errorMem.error_3day_mm : null;
  const err7d = errorMem?.error_7day_mm !== undefined && errorMem?.error_7day_mm !== null ? errorMem.error_7day_mm : null;
  const err14d = errorMem?.error_14day_mm !== undefined && errorMem?.error_14day_mm !== null ? errorMem.error_14day_mm : null;

  const canonicalRegimes = [
    { key: 'ACTIVE_MONSOON', label: 'Active Monsoon', color: '#8ECDF3' },
    { key: 'BREAK_MONSOON', label: 'Break Monsoon', color: '#BFEAF2' },
    { key: 'MONSOON_DEPRESSION', label: 'Monsoon Low / Depression', color: '#DCDDF8' },
    { key: 'OFFSHORE_TROUGH', label: 'Offshore Trough / Coastal', color: '#C5D6F2' },
    { key: 'WESTERN_DISTURBANCE', label: 'Western Disturbance', color: '#A8E0DB' },
    { key: 'NORMAL_TRANSITIONAL', label: 'Normal / Transitional', color: '#9EB4C7' },
  ];

  return (
    <div
      style={{
        background: '#FFFFFF',
        borderRadius: 'var(--radius-lg)',
        border: '1.5px solid #D0E5F5',
        boxShadow: 'var(--shadow-md)',
        padding: '1.4rem 1.6rem',
      }}
    >
      {/* Title & Core Differentiator Banner */}
      <div style={{ marginBottom: '1.25rem' }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', background: '#EAF6FD', color: 'var(--deep-blue)', padding: '0.2rem 0.65rem', borderRadius: '9999px', fontSize: '0.72rem', fontWeight: 700, marginBottom: '0.35rem' }}>
          <span>CORE TECHNICAL DIFFERENTIATOR</span>
        </div>
        <h1 style={{ fontSize: '1.55rem', fontWeight: 900, color: 'var(--deep-blue)', margin: 0, letterSpacing: '-0.01em' }}>
          Why RAAP-X Changed the Forecast
        </h1>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.35rem', maxWidth: '850px', lineHeight: 1.45 }}>
          RAAP-X does not simply predict rainfall. It learns when and how NWP forecasts fail under different weather regimes,
          applies recent NWP error memory, routes through regime-specific experts, and combines them using probability-weighted fusion.
        </p>
      </div>

      {/* Visual Processing Flow Pipeline */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
        {/* STAGE 1: RAW NWP FORECAST */}
        <div
          style={{
            background: 'var(--cloud-white)',
            border: '1px solid #D7E7F0',
            borderRadius: 'var(--radius-md)',
            padding: '0.9rem 1.15rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
            <div
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                background: '#B9DFF7',
                color: 'var(--deep-blue)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 800,
                fontSize: '0.85rem',
                flexShrink: 0,
              }}
            >
              1
            </div>
            <div>
              <div style={{ fontSize: '0.74rem', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Stage 1 — Raw NWP Baseline Forecast
              </div>
              <div style={{ fontSize: '0.82rem', color: 'var(--neutral-text)', marginTop: '0.15rem' }}>
                Uncorrected numerical model forecast before regime-aware bias repair
              </div>
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            {hasRawNwp ? (
              <div style={{ fontSize: '1.45rem', fontWeight: 900, color: 'var(--neutral-text)' }}>
                {rawNwp.toFixed(1)} <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>mm</span>
              </div>
            ) : (
              <div style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-muted)' }}>
                NWP baseline unavailable
              </div>
            )}
            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Lead Time: T+{leadTime}h</div>
          </div>
        </div>

        {/* Down Arrow */}
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: 'var(--sky-blue)' }}>
            <ArrowDown size={18} />
          </div>
        </div>

        {/* STAGE 2: WEATHER REGIME PROBABILITIES (SOFT REGIME ROUTING) */}
        <div
          style={{
            background: '#FFFFFF',
            border: '1.5px solid #8ECDF3',
            borderRadius: 'var(--radius-md)',
            padding: '1rem 1.25rem',
            boxShadow: '0 2px 6px rgba(142, 205, 243, 0.15)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.65rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
              <div
                style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  background: 'var(--deep-blue)',
                  color: '#FFFFFF',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 800,
                  fontSize: '0.85rem',
                  flexShrink: 0,
                }}
              >
                2
              </div>
              <div>
                <div style={{ fontSize: '0.74rem', fontWeight: 800, color: 'var(--deep-blue)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  Stage 2 — Weather Regime Probabilities (Soft Regime Routing)
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--neutral-text)', marginTop: '0.15rem' }}>
                  The model preserves the full continuous synoptic probability vector instead of collapsing to a single hard class.
                </div>
              </div>
            </div>
            <div style={{ background: '#EAF6FD', border: '1px solid #D0E5F5', padding: '0.2rem 0.55rem', borderRadius: '9999px', fontSize: '0.72rem', fontWeight: 700, color: 'var(--deep-blue)' }}>
              Dominant: {(dominantRegime || 'NORMAL_TRANSITIONAL').replace(/_/g, ' ')} {dominantProb !== null ? `(${(dominantProb * 100).toFixed(0)}%)` : ''}
            </div>
          </div>

          {/* Soft Regime Horizontal Bars (All 6 canonical regimes with real backend probabilities) */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.55rem', marginTop: '0.65rem' }}>
            {canonicalRegimes.map((reg) => {
              const probVal = regProbsNormalized.get(reg.key);
              const hasProb = probVal !== undefined && probVal !== null;
              const pct = hasProb ? (probVal * 100).toFixed(1) : null;
              const isDominant = reg.key.toUpperCase() === dominantRegime.toUpperCase();

              return (
                <div
                  key={reg.key}
                  style={{
                    background: isDominant ? '#F1F8FD' : 'var(--cloud-white)',
                    border: isDominant ? '1.5px solid var(--sky-blue)' : '1px solid #E8F1F7',
                    borderRadius: 'var(--radius-sm)',
                    padding: '0.45rem 0.65rem',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                    <span style={{ fontSize: '0.74rem', fontWeight: isDominant ? 800 : 600, color: isDominant ? 'var(--deep-blue)' : 'var(--neutral-text)' }}>
                      {reg.label} {isDominant && '★'}
                    </span>
                    <strong style={{ fontSize: '0.78rem', color: isDominant ? 'var(--deep-blue)' : 'var(--neutral-text)' }}>
                      {pct !== null ? `${pct}%` : 'Not available'}
                    </strong>
                  </div>
                  <div style={{ width: '100%', height: '8px', background: '#E8F1F7', borderRadius: '4px', overflow: 'hidden' }}>
                    <div
                      style={{
                        width: pct !== null ? `${Math.max(2, parseFloat(pct))}%` : '0%',
                        height: '100%',
                        background: reg.color,
                        borderRadius: '4px',
                        transition: 'width 0.4s ease',
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ marginTop: '0.55rem', fontSize: '0.7rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
            All 6 canonical meteorological synoptic regimes evaluated concurrently through soft continuous probability weighting.
          </div>
        </div>

        {/* Down Arrow */}
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: 'var(--sky-blue)' }}>
            <ArrowDown size={18} />
          </div>
        </div>

        {/* STAGE 3: RECENT NWP ERROR MEMORY */}
        <div
          style={{
            background: '#FFFFFF',
            border: '1.5px solid #D0E5F5',
            borderRadius: 'var(--radius-md)',
            padding: '1rem 1.25rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem', marginBottom: '0.65rem' }}>
            <div
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                background: '#5A9EC8',
                color: '#FFFFFF',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 800,
                fontSize: '0.85rem',
                flexShrink: 0,
              }}
            >
              3
            </div>
            <div>
              <div style={{ fontSize: '0.74rem', fontWeight: 800, color: 'var(--deep-blue)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Stage 3 — Recent NWP Error Memory
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--neutral-text)', marginTop: '0.15rem' }}>
                Trailing multi-scale error memory captures persistent NWP biases over the past 3, 7, and 14 days without temporal leakage.
              </div>
            </div>
          </div>

          {/* 3-Day, 7-Day, 14-Day Error Memory Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem', marginTop: '0.5rem' }}>
            <div style={{ background: 'var(--cloud-white)', border: '1px solid #D7E7F0', borderRadius: 'var(--radius-sm)', padding: '0.65rem', textAlign: 'center' }}>
              <div style={{ fontSize: '0.68rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                3-Day Error Memory
              </div>
              <div style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--deep-blue)', marginTop: '0.2rem' }}>
                {err3d !== null ? `${err3d >= 0 ? `+${err3d.toFixed(1)}` : err3d.toFixed(1)} mm` : 'Not available'}
              </div>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
                Immediate multi-cycle bias lag
              </div>
            </div>

            <div style={{ background: 'var(--cloud-white)', border: '1px solid #D7E7F0', borderRadius: 'var(--radius-sm)', padding: '0.65rem', textAlign: 'center' }}>
              <div style={{ fontSize: '0.68rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                7-Day Error Memory
              </div>
              <div style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--deep-blue)', marginTop: '0.2rem' }}>
                {err7d !== null ? `${err7d >= 0 ? `+${err7d.toFixed(1)}` : err7d.toFixed(1)} mm` : 'Not available'}
              </div>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
                Synoptic weekly persistence
              </div>
            </div>

            <div style={{ background: 'var(--cloud-white)', border: '1px solid #D7E7F0', borderRadius: 'var(--radius-sm)', padding: '0.65rem', textAlign: 'center' }}>
              <div style={{ fontSize: '0.68rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                14-Day Error Memory
              </div>
              <div style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--deep-blue)', marginTop: '0.2rem' }}>
                {err14d !== null ? `${err14d >= 0 ? `+${err14d.toFixed(1)}` : err14d.toFixed(1)} mm` : 'Not available'}
              </div>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
                Fortnightly systematic offset
              </div>
            </div>
          </div>
        </div>

        {/* Down Arrow */}
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: 'var(--sky-blue)' }}>
            <ArrowDown size={18} />
          </div>
        </div>

        {/* STAGE 4: REGIME-SPECIFIC EXPERTS (Visual Architecture without Fake Numerical Attribution) */}
        <div
          style={{
            background: '#FFFFFF',
            border: '1.5px solid #D0E5F5',
            borderRadius: 'var(--radius-md)',
            padding: '1rem 1.25rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem', marginBottom: '0.65rem' }}>
            <div
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                background: '#285A7A',
                color: '#FFFFFF',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 800,
                fontSize: '0.85rem',
                flexShrink: 0,
              }}
            >
              4
            </div>
            <div>
              <div style={{ fontSize: '0.74rem', fontWeight: 800, color: 'var(--deep-blue)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Stage 4 — Regime-Specific Experts (Parallel Routing Architecture)
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--neutral-text)', marginTop: '0.15rem' }}>
                Specialized sub-models conditioned on distinct meteorological circulation dynamics.
              </div>
            </div>
          </div>

          {/* Architecture Tree Diagram without fake numerical attribution */}
          <div
            style={{
              background: '#F8FCFF',
              border: '1px solid #E2EDF5',
              borderRadius: 'var(--radius-md)',
              padding: '0.85rem 1rem',
              textAlign: 'center',
            }}
          >
            <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--deep-blue)', marginBottom: '0.5rem' }}>
              Regime-Conditioned Error Correction Architecture
            </div>

            {/* Tree Branch Visual */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.55rem' }}>
              <div style={{ background: '#FFFFFF', border: '1.5px solid #8ECDF3', borderRadius: 'var(--radius-sm)', padding: '0.5rem' }}>
                <div style={{ fontSize: '0.72rem', fontWeight: 800, color: 'var(--deep-blue)' }}>Active Monsoon Expert</div>
                <div style={{ fontSize: '0.64rem', color: 'var(--neutral-text)', marginTop: '0.15rem' }}>Convective & low-level jet bias repair</div>
              </div>

              <div style={{ background: '#FFFFFF', border: '1px solid #D0E5F5', borderRadius: 'var(--radius-sm)', padding: '0.5rem' }}>
                <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--neutral-text)' }}>Break Monsoon Expert</div>
                <div style={{ fontSize: '0.64rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>Central plain wet-bias damping</div>
              </div>

              <div style={{ background: '#FFFFFF', border: '1px solid #D0E5F5', borderRadius: 'var(--radius-sm)', padding: '0.5rem' }}>
                <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--neutral-text)' }}>Depression Expert</div>
                <div style={{ fontSize: '0.64rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>Synoptic cyclonic vortex track scaling</div>
              </div>

              <div style={{ background: '#FFFFFF', border: '1px solid #D0E5F5', borderRadius: 'var(--radius-sm)', padding: '0.5rem' }}>
                <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--neutral-text)' }}>Offshore Trough Expert</div>
                <div style={{ fontSize: '0.64rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>Coastal orographic rain-band adjustment</div>
              </div>

              <div style={{ background: '#FFFFFF', border: '1px solid #D0E5F5', borderRadius: 'var(--radius-sm)', padding: '0.5rem' }}>
                <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--neutral-text)' }}>Western Disturbance Expert</div>
                <div style={{ fontSize: '0.64rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>Mid-latitude extratropical wave scaling</div>
              </div>

              <div style={{ background: '#FFFFFF', border: '1px solid #D0E5F5', borderRadius: 'var(--radius-sm)', padding: '0.5rem' }}>
                <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--neutral-text)' }}>Transitional State Expert</div>
                <div style={{ fontSize: '0.64rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>Climatological baseline regression</div>
              </div>
            </div>
            
            <div style={{ fontSize: '0.67rem', color: 'var(--text-muted)', marginTop: '0.45rem' }}>
              Note: Expert-level numerical attribution is blended via probability-weighted fusion without artificial single-expert assignment.
            </div>
          </div>
        </div>

        {/* Down Arrow */}
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: 'var(--sky-blue)' }}>
            <ArrowDown size={18} />
          </div>
        </div>

        {/* STAGE 5: PROBABILITY-WEIGHTED FUSION */}
        <div
          style={{
            background: 'linear-gradient(180deg, #FFFFFF 0%, #F1F8FD 100%)',
            border: '1.5px solid #8ECDF3',
            borderRadius: 'var(--radius-md)',
            padding: '1rem 1.25rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
            <div
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                background: '#1B4965',
                color: '#FFFFFF',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 800,
                fontSize: '0.85rem',
                flexShrink: 0,
              }}
            >
              5
            </div>
            <div>
              <div style={{ fontSize: '0.74rem', fontWeight: 800, color: 'var(--deep-blue)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Stage 5 — Probability-Weighted Fusion
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--neutral-text)', marginTop: '0.15rem' }}>
                Soft combination across all regime-conditioned experts weighted continuously by the active synoptic regime probability vector.
              </div>
            </div>
          </div>
          <div style={{ background: '#FFFFFF', border: '1px solid #D0E5F5', borderRadius: 'var(--radius-sm)', padding: '0.4rem 0.75rem', fontSize: '0.72rem', color: 'var(--deep-blue)', fontWeight: 700 }}>
            Correction = Σ (P_regime × Expert_Residual)
          </div>
        </div>

        {/* Down Arrow */}
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: 'var(--sky-blue)' }}>
            <ArrowDown size={18} />
          </div>
        </div>

        {/* STAGE 6: RAAP-X CORRECTED FORECAST */}
        <div
          style={{
            background: '#FFFFFF',
            border: '2px solid #285A7A',
            borderRadius: 'var(--radius-md)',
            padding: '1rem 1.25rem',
            boxShadow: '0 4px 14px rgba(40, 90, 122, 0.12)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
            <div
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                background: '#2E8B80',
                color: '#FFFFFF',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 800,
                fontSize: '0.85rem',
                flexShrink: 0,
              }}
            >
              6
            </div>
            <div>
              <div style={{ fontSize: '0.74rem', fontWeight: 800, color: '#2E8B80', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Stage 6 — RAAP-X Corrected Forecast
              </div>
              <div style={{ fontSize: '0.82rem', color: 'var(--deep-blue)', fontWeight: 600, marginTop: '0.15rem' }}>
                Final regime-repaired expectation with calibrated error memory integration
              </div>
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            {hasP50 ? (
              <div style={{ fontSize: '1.75rem', fontWeight: 900, color: 'var(--deep-blue)', lineHeight: 1 }}>
                {p50.toFixed(1)} <span style={{ fontSize: '0.9rem', fontWeight: 700 }}>mm</span>
              </div>
            ) : (
              <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--deep-blue)' }}>
                Output unavailable
              </div>
            )}
            <div style={{ fontSize: '0.72rem', fontWeight: 700, color: diff !== null && diff >= 0 ? '#2E8B80' : '#A34848', marginTop: '0.2rem' }}>
              Correction: {diff !== null ? (diff >= 0 ? `+${diff.toFixed(1)} mm` : `${diff.toFixed(1)} mm`) : 'Not available'} vs. NWP
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
