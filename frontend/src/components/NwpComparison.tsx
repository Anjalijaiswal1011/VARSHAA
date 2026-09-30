import { ArrowRight, SlidersHorizontal } from 'lucide-react';
import type { RainfallQuantiles } from '../types/forecast';

interface NwpComparisonProps {
  rainfall: RainfallQuantiles;
}

export const NwpComparison: React.FC<NwpComparisonProps> = ({ rainfall }) => {
  const diff = rainfall.difference !== undefined ? rainfall.difference : rainfall.p50 - rainfall.raw_nwp;
  const isPositive = diff > 0;
  const isZero = Math.abs(diff) < 0.05;

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <SlidersHorizontal size={16} className="card-title-icon" />
          Raw NWP vs. RAAP-X Post-Processing
        </div>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          Atmospheric Correction
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.65rem 0', gap: '0.5rem', flexWrap: 'wrap' }}>
        {/* Step 1: Raw NWP */}
        <div style={{ textAlign: 'center', flex: 1, minWidth: '70px', padding: '0.4rem', background: '#F8FCFF', borderRadius: '6px', border: '1px solid var(--soft-border)' }}>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Raw NWP</div>
          <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--neutral-text)' }}>
            {rainfall.raw_nwp.toFixed(1)} <span style={{ fontSize: '0.7rem' }}>mm</span>
          </div>
        </div>

        <ArrowRight size={14} color="var(--soft-blue)" />

        {/* Step 2: RAAP-X P50 */}
        <div style={{ textAlign: 'center', flex: 1, minWidth: '85px', background: 'var(--very-light-sky)', padding: '0.45rem', borderRadius: '6px', border: '1px solid var(--soft-blue)' }}>
          <div style={{ fontSize: '0.65rem', color: 'var(--deep-blue)', fontWeight: 700, textTransform: 'uppercase' }}>RAAP-X P50</div>
          <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--deep-blue)' }}>
            {rainfall.p50.toFixed(1)} <span style={{ fontSize: '0.7rem' }}>mm</span>
          </div>
        </div>

        <ArrowRight size={14} color="var(--soft-blue)" />

        {/* Step 3: Upper Quantiles */}
        <div style={{ textAlign: 'center', flex: 1, minWidth: '70px', padding: '0.4rem', background: '#F8FCFF', borderRadius: '6px', border: '1px solid var(--soft-border)' }}>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>P75 / P90</div>
          <div style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--deep-blue)' }}>
            {rainfall.p75.toFixed(1)} / {rainfall.p90.toFixed(1)} <span style={{ fontSize: '0.65rem' }}>mm</span>
          </div>
        </div>
      </div>

      {/* Delta Badge */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid #F1F7FB', fontSize: '0.75rem' }}>
        <span style={{ color: 'var(--neutral-text)' }}>Correction Delta:</span>
        <span className={`delta-badge ${isZero ? '' : isPositive ? 'positive' : 'negative'}`}>
          {isZero ? '0.0 mm (Unchanged)' : `${isPositive ? '+' : ''}${diff.toFixed(1)} mm`}
          {isPositive ? ' (Orographic Offset)' : ' (Wet Bias Damping)'}
        </span>
      </div>
    </div>
  );
};
