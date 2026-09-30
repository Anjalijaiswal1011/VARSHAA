import React from 'react';
import { HelpCircle, ArrowDown, CheckCircle2 } from 'lucide-react';
import type { CanonicalDistrictForecastRecord, RainfallQuantiles } from '../types/forecast';

interface WhyRaapxAdjustedCardProps {
  district: CanonicalDistrictForecastRecord;
  rainfall: RainfallQuantiles;
}

export const WhyRaapxAdjustedCard: React.FC<WhyRaapxAdjustedCardProps> = ({ district, rainfall }) => {
  const explanation = district.explanation;
  const rawNwp = rainfall.raw_nwp !== undefined && rainfall.raw_nwp !== null ? rainfall.raw_nwp : null;
  const p50 = rainfall.p50 !== undefined && rainfall.p50 !== null ? rainfall.p50 : null;
  const hasRawNwp = rawNwp !== null && !isNaN(rawNwp);
  const hasP50 = p50 !== null && !isNaN(p50);
  const diff = hasRawNwp && hasP50 ? (rainfall.difference !== undefined ? rainfall.difference : p50 - rawNwp) : null;
  const regimeName = district.dominant_regime || explanation?.dominant_regime || 'ACTIVE_MONSOON';
  const regimeConf = explanation?.regime_confidence
    ? Math.round(explanation.regime_confidence * 100)
    : district.regime?.dominant_probability
    ? Math.round(district.regime.dominant_probability * 100)
    : null;

  // Extract recent error driver from actual model top_features
  const errorDriver = explanation?.top_features?.find(
    (f) => f.category === 'recent_error' || f.feature.includes('error_memory')
  );
  const errorContribution = errorDriver ? errorDriver.contribution_mm : diff;

  // Extract terrain/orography driver from actual model top_features
  const terrainDriver = explanation?.top_features?.find(
    (f) => f.category === 'terrain' || f.category === 'orography' || f.feature.includes('slope') || f.feature.includes('elevation')
  );
  const terrainText = terrainDriver
    ? `${terrainDriver.contribution_mm >= 0 ? '+' : ''}${terrainDriver.contribution_mm.toFixed(1)} mm`
    : 'Not available';

  return (
    <div className="card why-raapx-card">
      <div className="card-header" style={{ marginBottom: '0.65rem' }}>
        <div className="card-title">
          <HelpCircle size={16} className="card-title-icon" />
          WHY RAAP-X ADJUSTED THE FORECAST
        </div>
        <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
          Atmospheric Post-Processing Logic
        </span>
      </div>

      {/* Structural Progression Flow (Requirement 4) */}
      <div className="adjustment-flow-container">
        {/* Step 1: NWP Baseline */}
        <div className="flow-step">
          <div className="flow-badge baseline">1. NWP BASELINE</div>
          <div className="flow-val">{hasRawNwp ? `${rawNwp.toFixed(1)} mm` : 'NWP baseline unavailable'}</div>
          <div className="flow-desc">Uncorrected Raw Numerical Weather Prediction (GFS)</div>
        </div>

        <div className="flow-arrow"><ArrowDown size={14} /></div>

        {/* Step 2: Weather Regime */}
        <div className="flow-step">
          <div className="flow-badge regime">2. WEATHER REGIME</div>
          <div className="flow-val">{regimeName.replace('_', ' ')} {regimeConf !== null ? `(${regimeConf}%)` : ''}</div>
          <div className="flow-desc">Synoptic regime classified from physical flow dynamics</div>
        </div>

        <div className="flow-arrow"><ArrowDown size={14} /></div>

        {/* Step 3: Recent NWP Error Memory */}
        <div className="flow-step">
          <div className="flow-badge error-memory">3. RECENT NWP ERROR MEMORY</div>
          <div className="flow-val" style={{ color: errorContribution !== null && errorContribution >= 0 ? '#285A7A' : '#526775' }}>
            {errorContribution !== null ? `${errorContribution >= 0 ? '+' : ''}${errorContribution.toFixed(1)} mm` : 'Not available'}
          </div>
          <div className="flow-desc">
            {errorDriver?.description || 'Multi-scale error memory lag adjustment applied to damp NWP bias.'}
          </div>
        </div>

        <div className="flow-arrow"><ArrowDown size={14} /></div>

        {/* Step 4: Local / Terrain Modifier */}
        <div className="flow-step">
          <div className="flow-badge terrain">4. LOCAL / TERRAIN MODIFIER</div>
          <div className="flow-val" style={{ fontSize: terrainDriver ? '0.9rem' : '0.78rem', color: 'var(--neutral-text)' }}>
            {terrainText}
          </div>
          <div className="flow-desc">
            {terrainDriver?.description || 'Orographic slope modulation and elevation gradient effects'}
          </div>
        </div>

        <div className="flow-arrow"><ArrowDown size={14} /></div>

        {/* Step 5: RAAP-X Corrected Forecast */}
        <div className="flow-step final">
          <div className="flow-badge raapx">5. RAAP-X CORRECTED FORECAST (P50)</div>
          <div className="flow-val-final">{hasP50 ? `${p50.toFixed(1)} mm` : 'Output unavailable'}</div>
          <div className="flow-desc">
            Net adjustment: <strong>{diff !== null ? `${diff >= 0 ? '+' : ''}${diff.toFixed(1)} mm` : 'Not available'}</strong> vs raw NWP
          </div>
        </div>
      </div>

      {/* Section 8: Structured Breakdown Grid (WHY THIS FORECAST?) */}
      <div
        style={{
          marginTop: '0.65rem',
          padding: '0.55rem 0.75rem',
          background: '#F8FCFF',
          border: '1px solid var(--soft-border)',
          borderRadius: 'var(--radius-md)',
          fontSize: '0.72rem',
        }}
      >
        <div style={{ fontWeight: 700, color: 'var(--deep-blue)', marginBottom: '0.4rem', fontSize: '0.75rem' }}>
          WHY THIS FORECAST? (PHYSICAL ATTRIBUTION)
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.4rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.25rem 0.4rem', background: '#FFFFFF', borderRadius: '4px', border: '1px solid #E8F1F7' }}>
            <span style={{ color: 'var(--text-muted)' }}>Weather Regime:</span>
            <strong style={{ color: 'var(--deep-blue)' }}>{regimeName.replace('_', ' ')} {regimeConf !== null ? `(${regimeConf}%)` : ''}</strong>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.25rem 0.4rem', background: '#FFFFFF', borderRadius: '4px', border: '1px solid #E8F1F7' }}>
            <span style={{ color: 'var(--text-muted)' }}>Recent NWP Error:</span>
            <strong style={{ color: errorContribution !== null && errorContribution >= 0 ? '#285A7A' : '#526775' }}>
              {errorContribution !== null ? `${errorContribution >= 0 ? '+' : ''}${errorContribution.toFixed(1)} mm` : 'Not available'}
            </strong>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.25rem 0.4rem', background: '#FFFFFF', borderRadius: '4px', border: '1px solid #E8F1F7' }}>
            <span style={{ color: 'var(--text-muted)' }}>Terrain Effect:</span>
            <span style={{ color: terrainDriver ? 'var(--deep-blue)' : 'var(--text-muted)', fontStyle: terrainDriver ? 'normal' : 'italic' }}>
              {terrainText}
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.25rem 0.4rem', background: '#FFFFFF', borderRadius: '4px', border: '1px solid #E8F1F7' }}>
            <span style={{ color: 'var(--text-muted)' }}>Final RAAP-X Adjustment:</span>
            <strong style={{ color: 'var(--deep-blue)' }}>
              {diff !== null && hasRawNwp && hasP50
                ? `${diff >= 0 ? '+' : ''}${diff.toFixed(1)} mm (${rawNwp.toFixed(1)} → ${p50.toFixed(1)} mm)`
                : 'Not available'}
            </strong>
          </div>
        </div>
      </div>

      {/* Actual Evidence-Grounded Model Summary */}
      {explanation?.user_friendly_summary && (
        <div className="explanation-summary-box">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', marginBottom: '0.2rem' }}>
            <CheckCircle2 size={13} color="#2E8B80" />
            <strong style={{ fontSize: '0.73rem', color: 'var(--deep-blue)' }}>Model Synthesis:</strong>
          </div>
          <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
            {explanation.user_friendly_summary}
          </div>
        </div>
      )}
    </div>
  );
};

