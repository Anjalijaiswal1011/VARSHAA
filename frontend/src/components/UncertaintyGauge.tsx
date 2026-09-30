import { AlertCircle, TrendingUp } from 'lucide-react';
import type { RainfallQuantiles } from '../types/forecast';

interface UncertaintyGaugeProps {
  rainfall: RainfallQuantiles;
  spread: number;
}

export const UncertaintyGauge: React.FC<UncertaintyGaugeProps> = ({ rainfall, spread }) => {
  const maxVal = Math.max(100, rainfall.p90 * 1.15);
  const p50Pct = Math.min(100, (rainfall.p50 / maxVal) * 100);
  const p75Pct = Math.min(100, (rainfall.p75 / maxVal) * 100);
  const p90Pct = Math.min(100, (rainfall.p90 / maxVal) * 100);

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <TrendingUp size={16} className="card-title-icon" />
          Forecast Uncertainty Quantiles
        </div>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          Atmospheric Ensembles
        </span>
      </div>

      <div className="uncertainty-grid">
        {/* P50 Central Estimate */}
        <div className="quantile-box p50">
          <div className="quantile-tag">P50 — Rainfall Quantile</div>
          <div className="quantile-val">{rainfall.p50.toFixed(1)} <span style={{ fontSize: '0.75rem', fontWeight: 500 }}>mm</span></div>
          <div className="quantile-desc">Central Rainfall Quantile (50th Percentile)</div>
        </div>

        {/* P75 Higher Uncertainty */}
        <div className="quantile-box p75">
          <div className="quantile-tag">P75 — Rainfall Quantile</div>
          <div className="quantile-val">{rainfall.p75.toFixed(1)} <span style={{ fontSize: '0.75rem', fontWeight: 500 }}>mm</span></div>
          <div className="quantile-desc">Higher Rainfall Quantile (75th Percentile)</div>
        </div>

        {/* P90 High Risk Scenario */}
        <div className="quantile-box p90">
          <div className="quantile-tag">P90 — Rainfall Quantile</div>
          <div className="quantile-val">{rainfall.p90.toFixed(1)} <span style={{ fontSize: '0.75rem', fontWeight: 500 }}>mm</span></div>
          <div className="quantile-desc">Upper Rainfall Quantile (90th Percentile)</div>
        </div>
      </div>

      {/* Uncertainty Range Bar Visualizer */}
      <div style={{ marginTop: '0.85rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--neutral-text)', marginBottom: '0.35rem' }}>
          <span>0 mm</span>
          <span>Uncertainty Spread (P90 - P50): <strong style={{ color: 'var(--deep-blue)' }}>+{spread.toFixed(1)} mm</strong></span>
          <span>{Math.round(maxVal)} mm</span>
        </div>
        <div style={{ position: 'relative', width: '100%', height: '8px', background: '#E8F1F7', borderRadius: '4px', overflow: 'hidden' }}>
          {/* P90 bar - Deep Atmospheric Blue tint */}
          <div
            style={{
              position: 'absolute',
              left: 0,
              top: 0,
              height: '100%',
              width: `${p90Pct}%`,
              background: '#285A7A',
              opacity: 0.3,
            }}
          />
          {/* P75 bar - Medium Sky Blue */}
          <div
            style={{
              position: 'absolute',
              left: 0,
              top: 0,
              height: '100%',
              width: `${p75Pct}%`,
              background: '#5A9EC8',
              opacity: 0.55,
            }}
          />
          {/* P50 bar - Soft Sky Blue */}
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

      <div className="uncertainty-spread-bar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--neutral-text)' }}>
          <AlertCircle size={13} color="var(--deep-blue)" />
          <span>Spread Indicator:</span>
        </div>
        <strong style={{ color: 'var(--deep-blue)' }}>
          {spread <= 10 ? 'Low Spread (High Confidence)' : spread <= 25 ? 'Moderate Spread' : 'Elevated Spread (Convective)'}
        </strong>
      </div>
    </div>
  );
};
