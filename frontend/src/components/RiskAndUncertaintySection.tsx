import React from 'react';
import { ShieldAlert, Info } from 'lucide-react';
import type { CanonicalDistrictForecastRecord } from '../types/forecast';

interface RiskAndUncertaintySectionProps {
  district: CanonicalDistrictForecastRecord | null;
  leadTime: number;
}

export const RiskAndUncertaintySection: React.FC<RiskAndUncertaintySectionProps> = ({
  district,
  leadTime,
}) => {
  if (!district) return null;

  const p50 = district.corrected_p50;
  const p75 = district.corrected_p75;
  const p90 = district.corrected_p90;
  const spread = district.spread_p90_p50 || Math.round((p90 - p50) * 100) / 100;

  const heavyProb = district.heavy_rainfall_probability;
  const extremeProb = district.extreme_rainfall_probability;
  const heavyPct = Math.round(heavyProb * 100);
  const extremePct = Math.round(extremeProb * 100);

  const maxVal = Math.max(10, p90) * 1.2;
  const p50Pct = Math.min(100, (p50 / maxVal) * 100);
  const p75Pct = Math.min(100, (p75 / maxVal) * 100);
  const p90Pct = Math.min(100, (p90 / maxVal) * 100);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* SECTION 8: RAINFALL UNCERTAINTY (RAINFALL QUANTILES) */}
      <div
        style={{
          background: '#FFFFFF',
          borderRadius: 'var(--radius-lg)',
          border: '1.5px solid #D0E5F5',
          boxShadow: 'var(--shadow-md)',
          padding: '1.25rem 1.4rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.85rem' }}>
          <div>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', background: '#EAF6FD', color: 'var(--deep-blue)', padding: '0.2rem 0.6rem', borderRadius: '9999px', fontSize: '0.72rem', fontWeight: 700, marginBottom: '0.35rem' }}>
              <span>PROBABILISTIC MULTI-QUANTILE PLUME</span>
            </div>
            <h2 style={{ fontSize: '1.35rem', fontWeight: 800, color: 'var(--deep-blue)', margin: 0 }}>
              Rainfall Uncertainty
            </h2>
            <div style={{ fontSize: '0.8rem', color: 'var(--neutral-text)', marginTop: '0.15rem' }}>
              Forecast for <strong>{district.district_name} District</strong> (T+{leadTime}h) • Label: <strong>RAINFALL QUANTILES</strong>
            </div>
          </div>
          <div style={{ background: '#F1F7FB', border: '1px solid #D0E5F5', padding: '0.3rem 0.65rem', borderRadius: 'var(--radius-sm)', textAlign: 'right' }}>
            <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Uncertainty Spread:</span>
            <div style={{ fontSize: '0.85rem', fontWeight: 800, color: 'var(--deep-blue)' }}>
              +{spread.toFixed(1)} mm
            </div>
          </div>
        </div>

        {/* 3 Connected Quantile Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.85rem' }}>
          {/* P50 Central Quantile */}
          <div
            style={{
              background: 'var(--cloud-white)',
              border: '1.5px solid #8ECDF3',
              borderRadius: 'var(--radius-md)',
              padding: '0.85rem 1rem',
              textAlign: 'center',
            }}
          >
            <div style={{ display: 'inline-block', background: '#8ECDF3', color: '#1B4965', padding: '0.15rem 0.5rem', borderRadius: '4px', fontSize: '0.72rem', fontWeight: 800, marginBottom: '0.35rem' }}>
              P50 Rainfall Quantile
            </div>
            <div style={{ fontSize: '1.75rem', fontWeight: 900, color: 'var(--deep-blue)', lineHeight: 1.1 }}>
              {p50.toFixed(1)} <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>mm</span>
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--neutral-text)', marginTop: '0.3rem', fontWeight: 600 }}>
              Central rainfall quantile
            </div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
              Median expected accumulation
            </div>
          </div>

          {/* P75 Higher Quantile */}
          <div
            style={{
              background: 'var(--cloud-white)',
              border: '1.5px solid #5A9EC8',
              borderRadius: 'var(--radius-md)',
              padding: '0.85rem 1rem',
              textAlign: 'center',
            }}
          >
            <div style={{ display: 'inline-block', background: '#5A9EC8', color: '#FFFFFF', padding: '0.15rem 0.5rem', borderRadius: '4px', fontSize: '0.72rem', fontWeight: 800, marginBottom: '0.35rem' }}>
              P75 Rainfall Quantile
            </div>
            <div style={{ fontSize: '1.75rem', fontWeight: 900, color: 'var(--deep-blue)', lineHeight: 1.1 }}>
              {p75.toFixed(1)} <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>mm</span>
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--neutral-text)', marginTop: '0.3rem', fontWeight: 600 }}>
              Higher rainfall quantile
            </div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
              75th percentile envelope
            </div>
          </div>

          {/* P90 Upper Quantile */}
          <div
            style={{
              background: 'var(--cloud-white)',
              border: '1.5px solid #285A7A',
              borderRadius: 'var(--radius-md)',
              padding: '0.85rem 1rem',
              textAlign: 'center',
            }}
          >
            <div style={{ display: 'inline-block', background: '#285A7A', color: '#FFFFFF', padding: '0.15rem 0.5rem', borderRadius: '4px', fontSize: '0.72rem', fontWeight: 800, marginBottom: '0.35rem' }}>
              P90 Rainfall Quantile
            </div>
            <div style={{ fontSize: '1.75rem', fontWeight: 900, color: 'var(--deep-blue)', lineHeight: 1.1 }}>
              {p90.toFixed(1)} <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>mm</span>
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--neutral-text)', marginTop: '0.3rem', fontWeight: 600 }}>
              Upper rainfall quantile
            </div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
              90th percentile high scenario
            </div>
          </div>
        </div>

        {/* Quantile Multi-Scale Spread Visualization */}
        <div style={{ marginTop: '1rem', padding: '0.75rem 1rem', background: '#F8FCFF', border: '1px solid #E8F1F7', borderRadius: 'var(--radius-md)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--neutral-text)', marginBottom: '0.4rem' }}>
            <span>0 mm</span>
            <span style={{ fontWeight: 700, color: 'var(--deep-blue)' }}>
              Quantile Spread Envelope: P50 ({p50.toFixed(1)} mm) → P90 ({p90.toFixed(1)} mm)
            </span>
            <span>{Math.round(maxVal)} mm</span>
          </div>

          <div style={{ position: 'relative', width: '100%', height: '14px', background: '#EEF5F9', borderRadius: '4px', overflow: 'hidden' }}>
            {/* P90 Bar */}
            <div
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                height: '100%',
                width: `${p90Pct}%`,
                background: '#285A7A',
                opacity: 0.35,
                borderRadius: '4px',
              }}
            />
            {/* P75 Bar */}
            <div
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                height: '100%',
                width: `${p75Pct}%`,
                background: '#5A9EC8',
                opacity: 0.65,
                borderRadius: '4px',
              }}
            />
            {/* P50 Bar */}
            <div
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                height: '100%',
                width: `${p50Pct}%`,
                background: '#8ECDF3',
                borderRadius: '4px',
              }}
            />
          </div>
        </div>

        {/* Terminology Notice */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', marginTop: '0.75rem', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          <Info size={13} color="var(--deep-blue)" />
          <span>
            <strong>Terminology Notice:</strong> P50, P75, and P90 are <strong>RAINFALL QUANTILES</strong> in millimeters. P90 is a rainfall quantile, not a risk score or probability.
          </span>
        </div>
      </div>

      {/* SECTION 9: RAINFALL RISK (CALIBRATED EVENT PROBABILITIES) */}
      <div
        style={{
          background: '#FFFFFF',
          borderRadius: 'var(--radius-lg)',
          border: '1.5px solid #D0E5F5',
          boxShadow: 'var(--shadow-md)',
          padding: '1.25rem 1.4rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.85rem' }}>
          <div>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', background: '#FDF4E5', color: '#9C6C1B', padding: '0.2rem 0.6rem', borderRadius: '9999px', fontSize: '0.72rem', fontWeight: 700, marginBottom: '0.35rem' }}>
              <span>CALIBRATED CLASSIFICATION MODELS</span>
            </div>
            <h2 style={{ fontSize: '1.35rem', fontWeight: 800, color: 'var(--deep-blue)', margin: 0 }}>
              Rainfall Risk
            </h2>
            <div style={{ fontSize: '0.8rem', color: 'var(--neutral-text)', marginTop: '0.15rem' }}>
              Official IMD Thresholds: Heavy (≥ 64.5 mm / 24h) & Extreme (≥ 204.5 mm / 24h)
            </div>
          </div>
          <ShieldAlert size={26} color="var(--deep-blue)" />
        </div>

        {/* 2 Calibrated Risk Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
          {/* Heavy Rainfall Probability */}
          <div
            style={{
              background: '#F8FCFF',
              border: '1.5px solid #D7E7F0',
              borderRadius: 'var(--radius-md)',
              padding: '1rem',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.35rem' }}>
              <div>
                <div style={{ fontSize: '0.8rem', fontWeight: 800, color: 'var(--deep-blue)' }}>
                  Heavy Rainfall Probability
                </div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                  Threshold: ≥ 64.5 mm / 24h
                </div>
              </div>
              <div style={{ fontSize: '1.65rem', fontWeight: 900, color: '#3B769E' }}>
                {heavyPct}%
              </div>
            </div>

            <div style={{ width: '100%', height: '10px', background: '#EEF5F9', borderRadius: '4px', overflow: 'hidden', marginTop: '0.45rem' }}>
              <div
                style={{
                  width: `${Math.max(2, heavyPct)}%`,
                  height: '100%',
                  background: '#3B769E',
                  borderRadius: '4px',
                  transition: 'width 0.4s ease',
                }}
              />
            </div>
            <div style={{ fontSize: '0.68rem', color: 'var(--neutral-text)', marginTop: '0.35rem' }}>
              IMD Heavy Rainfall criterion exceedance likelihood
            </div>
          </div>

          {/* Extreme Rainfall Probability */}
          <div
            style={{
              background: '#F8FCFF',
              border: '1.5px solid #D7E7F0',
              borderRadius: 'var(--radius-md)',
              padding: '1rem',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.35rem' }}>
              <div>
                <div style={{ fontSize: '0.8rem', fontWeight: 800, color: 'var(--deep-blue)' }}>
                  Extreme Rainfall Probability
                </div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                  Threshold: ≥ 204.5 mm / 24h
                </div>
              </div>
              <div style={{ fontSize: '1.65rem', fontWeight: 900, color: '#6B6CA8' }}>
                {extremePct}%
              </div>
            </div>

            <div style={{ width: '100%', height: '10px', background: '#EEF5F9', borderRadius: '4px', overflow: 'hidden', marginTop: '0.45rem' }}>
              <div
                style={{
                  width: `${Math.max(2, extremePct)}%`,
                  height: '100%',
                  background: '#6B6CA8',
                  borderRadius: '4px',
                  transition: 'width 0.4s ease',
                }}
              />
            </div>
            <div style={{ fontSize: '0.68rem', color: 'var(--neutral-text)', marginTop: '0.35rem' }}>
              IMD Extreme Rainfall criterion exceedance likelihood
            </div>
          </div>
        </div>

        {/* Distinct Models Callout Note */}
        <div
          style={{
            marginTop: '0.9rem',
            padding: '0.65rem 0.85rem',
            background: '#F1F7FB',
            border: '1px solid #D0E5F5',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.74rem',
            color: 'var(--deep-blue)',
            lineHeight: 1.45,
          }}
        >
          🛡️ <strong>Methodological Independence:</strong> Generated by calibrated heavy/extreme rainfall probability models.
          The system strictly distinguishes <strong>RAINFALL QUANTILES</strong> (P50/P75/P90 in mm) from <strong>RAINFALL RISK PROBABILITIES</strong> (exceedance % likelihood).
        </div>
      </div>
    </div>
  );
};
