import { Compass, Sparkles } from 'lucide-react';
import type { RegimeState } from '../types/forecast';

interface RegimeIntelligenceProps {
  regime: RegimeState;
}

export const RegimeIntelligence: React.FC<RegimeIntelligenceProps> = ({ regime }) => {
  const regimeDescriptions: Record<string, string> = {
    ACTIVE_MONSOON: 'Vigorous monsoon trough with persistent low-level westerly jet and intense orographic precipitation.',
    BREAK_MONSOON: 'Convection suppressed over central India; heavy precipitation anchored along Himalayan foothills.',
    MONSOON_DEPRESSION: 'Low-pressure cyclonic vortex propagating from Bay of Bengal across central India.',
    WESTERN_DISTURBANCE: 'Upper-tropospheric extratropical wave causing orographic precipitation in northwest Himalayas.',
    OFFSHORE_TROUGH: 'Shallow convective pressure trough along Western Ghats Arabian Sea coastline.',
    NORMAL_TRANSITIONAL: 'Quasi-steady climatological monsoon flow without prominent synoptic vortex forcing.',
  };

  // Distinct soft colors for regimes
  const regimeColors: Record<string, string> = {
    ACTIVE_MONSOON: '#8ECDF3',     // Sky blue
    BREAK_MONSOON: '#BFEAF2',      // Soft cyan
    MONSOON_DEPRESSION: '#DCDDF8', // Soft lavender
    WESTERN_DISTURBANCE: '#A8E0DB',// Soft teal
    OFFSHORE_TROUGH: '#C5D6F2',    // Soft periwinkle
    NORMAL_TRANSITIONAL: '#9EB4C7',// Soft slate blue
  };

  const sortedRegimes = Object.entries(regime.probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <Compass size={16} className="card-title-icon" />
          Synoptic Regime Intelligence
        </div>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          Atmospheric State Probabilities
        </span>
      </div>

      {/* Dominant Active Regime Banner */}
      <div className="regime-active-pill">
        <div>
          <div style={{ fontSize: '0.7rem', color: 'var(--neutral-text)', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <Sparkles size={12} color="var(--deep-blue)" />
            Dominant Synoptic State
          </div>
          <div style={{ fontSize: '1.05rem', fontWeight: 800, color: 'var(--deep-blue)' }}>
            {regime.dominant.replace('_', ' ')}
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '1.3rem', fontWeight: 800, color: 'var(--deep-blue)' }}>
            {(regime.dominant_probability * 100).toFixed(0)}%
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--neutral-text)' }}>Probability</div>
        </div>
      </div>

      <p style={{ fontSize: '0.75rem', color: 'var(--neutral-text)', marginBottom: '0.85rem', lineHeight: 1.4 }}>
        {regimeDescriptions[regime.dominant] || 'Meteorological regime dynamics governing regional precipitation bias.'}
      </p>

      {/* Full 6-Regime Probability Distribution */}
      <div className="regime-prob-list">
        {sortedRegimes.map(([rName, prob]) => {
          const isDominant = rName === regime.dominant;
          const pct = (prob * 100).toFixed(1);
          const barColor = regimeColors[rName] || '#B9DFF7';
          return (
            <div key={rName} className="regime-item">
              <div className="regime-header-row">
                <span style={{ color: isDominant ? 'var(--deep-blue)' : 'var(--neutral-text)', fontWeight: isDominant ? 700 : 500 }}>
                  {rName.replace('_', ' ')}
                </span>
                <span style={{ color: isDominant ? 'var(--deep-blue)' : 'var(--neutral-text)', fontWeight: isDominant ? 700 : 500 }}>
                  {pct}%
                </span>
              </div>
              <div className="regime-bar-bg">
                <div
                  className="regime-bar-fill"
                  style={{
                    width: `${Math.max(3, prob * 100)}%`,
                    backgroundColor: barColor,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
