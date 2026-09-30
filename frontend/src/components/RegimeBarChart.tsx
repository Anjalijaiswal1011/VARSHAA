import React, { useState } from 'react';
import { Compass, Sparkles } from 'lucide-react';
import type { RegimeState } from '../types/forecast';

interface RegimeBarChartProps {
  regime: RegimeState;
}

export const RegimeBarChart: React.FC<RegimeBarChartProps> = ({ regime }) => {
  const [hoveredRegime, setHoveredRegime] = useState<string | null>(null);

  // Soft distinct colors conforming to Section 9
  const regimeColors: Record<string, string> = {
    ACTIVE_MONSOON: '#8ECDF3',      // Sky blue
    BREAK_MONSOON: '#BFEAF2',       // Soft cyan
    MONSOON_DEPRESSION: '#DCDDF8',  // Soft lavender
    WESTERN_DISTURBANCE: '#A8E0DB', // Soft teal
    OFFSHORE_TROUGH: '#C5D6F2',     // Soft periwinkle
    NORMAL_TRANSITIONAL: '#9EB4C7', // Soft slate blue
  };

  const regimeDescriptions: Record<string, string> = {
    ACTIVE_MONSOON: 'Vigorous monsoon trough with persistent low-level westerly jet and intense orographic precipitation.',
    BREAK_MONSOON: 'Convection suppressed over central India; heavy precipitation anchored along Himalayan foothills.',
    MONSOON_DEPRESSION: 'Low-pressure cyclonic vortex propagating from Bay of Bengal across central India.',
    WESTERN_DISTURBANCE: 'Upper-tropospheric extratropical wave causing orographic precipitation in northwest Himalayas.',
    OFFSHORE_TROUGH: 'Shallow convective pressure trough along Western Ghats Arabian Sea coastline.',
    NORMAL_TRANSITIONAL: 'Quasi-steady climatological monsoon flow without prominent synoptic vortex forcing.',
  };

  // Canonical regime order preserved
  const canonicalOrder = [
    'ACTIVE_MONSOON',
    'BREAK_MONSOON',
    'MONSOON_DEPRESSION',
    'WESTERN_DISTURBANCE',
    'OFFSHORE_TROUGH',
    'NORMAL_TRANSITIONAL',
  ];

  // Map probabilities ensuring all 6 regimes exist
  const regimeEntries = canonicalOrder.map((name) => {
    const prob = regime.probabilities[name] || 0.0;
    return {
      name,
      prob,
      color: regimeColors[name] || '#B9DFF7',
      isDominant: name === regime.dominant,
      desc: regimeDescriptions[name] || 'Synoptic state circulation pattern',
    };
  });

  return (
    <div className="card">
      <div className="card-header" style={{ marginBottom: '0.4rem' }}>
        <div className="card-title">
          <Compass size={15} className="card-title-icon" />
          Synoptic Regime Probabilities
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.7rem' }}>
          <span style={{ color: 'var(--text-muted)' }}>Dominant:</span>
          <span
            style={{
              padding: '0.15rem 0.45rem',
              borderRadius: '9999px',
              background: 'var(--very-light-sky)',
              border: '1px solid var(--soft-blue)',
              color: 'var(--deep-blue)',
              fontWeight: 700,
              fontSize: '0.68rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.25rem',
            }}
          >
            <Sparkles size={10} color="var(--deep-blue)" />
            {(regime.dominant || 'NORMAL_TRANSITIONAL').replace(/_/g, ' ')} ({((regime.dominant_probability || 0.65) * 100).toFixed(0)}%)
          </span>
        </div>
      </div>

      {/* Horizontal Bars for 6 Regimes */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem', marginTop: '0.3rem' }}>
        {regimeEntries.map((item) => {
          const pct = (item.prob * 100).toFixed(1);
          const isHovered = hoveredRegime === item.name;

          return (
            <div
              key={item.name}
              style={{ position: 'relative', cursor: 'pointer' }}
              onMouseEnter={() => setHoveredRegime(item.name)}
              onMouseLeave={() => setHoveredRegime(null)}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.72rem', marginBottom: '0.15rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                  <div style={{ width: '8px', height: '8px', borderRadius: '2px', backgroundColor: item.color }} />
                  <span
                    style={{
                      color: item.isDominant ? 'var(--deep-blue)' : 'var(--neutral-text)',
                      fontWeight: item.isDominant ? 700 : 500,
                    }}
                  >
                    {(item.name || '').replace(/_/g, ' ')}
                  </span>
                  {item.isDominant && (
                    <span style={{ fontSize: '0.6rem', color: 'var(--deep-blue)', fontWeight: 600 }}>• Active</span>
                  )}
                </div>
                <strong
                  style={{
                    color: item.isDominant ? 'var(--deep-blue)' : 'var(--neutral-text)',
                    fontWeight: item.isDominant ? 700 : 500,
                    fontSize: '0.75rem',
                  }}
                >
                  {pct}%
                </strong>
              </div>

              {/* Bar Track */}
              <div
                style={{
                  width: '100%',
                  height: '8px',
                  background: '#E8F1F7',
                  borderRadius: '3px',
                  overflow: 'hidden',
                  position: 'relative',
                }}
              >
                <div
                  style={{
                    height: '100%',
                    width: `${Math.max(1, parseFloat(pct))}%`,
                    backgroundColor: item.color,
                    borderRadius: '0 3px 3px 0',
                    transition: 'width 0.4s ease',
                    opacity: isHovered ? 1 : 0.85,
                  }}
                />
              </div>

              {/* Hover Tooltip */}
              {isHovered && (
                <div
                  style={{
                    position: 'absolute',
                    top: '-30px',
                    right: '0',
                    background: 'rgba(40, 90, 122, 0.95)',
                    color: '#FFFFFF',
                    padding: '0.25rem 0.5rem',
                    borderRadius: '4px',
                    fontSize: '0.65rem',
                    boxShadow: 'var(--shadow-md)',
                    zIndex: 100,
                    pointerEvents: 'none',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {item.name}: <strong>{pct}%</strong> ({item.desc.substring(0, 48)}...)
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
