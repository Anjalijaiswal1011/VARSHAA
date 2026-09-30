import React, { useState } from 'react';
import { SlidersHorizontal } from 'lucide-react';
import type { RainfallQuantiles } from '../types/forecast';

interface NwpVsRaapxChartProps {
  rainfall: RainfallQuantiles;
  leadTime: number;
}

export const NwpVsRaapxChart: React.FC<NwpVsRaapxChartProps> = ({ rainfall, leadTime }) => {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const rawNwp = rainfall.raw_nwp;
  const p50 = rainfall.p50;
  const p75 = rainfall.p75;
  const p90 = rainfall.p90;

  const diff = rainfall.difference !== undefined ? rainfall.difference : p50 - rawNwp;
  const isPositive = diff > 0;
  const isZero = Math.abs(diff) < 0.05;

  const series = [
    {
      id: 'nwp',
      label: 'Raw NWP',
      shortLabel: 'NWP',
      value: rawNwp,
      color: '#B9DFF7', // Soft Blue
      desc: 'Uncorrected NWP baseline',
    },
    {
      id: 'p50',
      label: 'P50 — Rainfall Quantile',
      shortLabel: 'P50',
      value: p50,
      color: '#8ECDF3', // Soft Sky Blue
      desc: 'Median rainfall quantile',
    },
    {
      id: 'p75',
      label: 'P75 — Rainfall Quantile',
      shortLabel: 'P75',
      value: p75,
      color: '#5A9EC8', // Medium Sky Blue
      desc: '75th percentile rainfall quantile',
    },
    {
      id: 'p90',
      label: 'P90 — Rainfall Quantile',
      shortLabel: 'P90',
      value: p90,
      color: '#285A7A', // Deep Atmospheric Blue
      desc: 'Upper rainfall quantile (90th percentile)',
    },
  ];

  // Dynamic Y-axis scale with padding
  const maxVal = Math.max(10, ...series.map((s) => s.value));
  const yMax = Math.ceil((maxVal * 1.25) / 10) * 10;
  const gridTicks = [0, Math.round(yMax * 0.33), Math.round(yMax * 0.66), yMax];

  return (
    <div className="card">
      <div className="card-header" style={{ marginBottom: '0.5rem' }}>
        <div className="card-title">
          <SlidersHorizontal size={16} className="card-title-icon" />
          NWP vs. RAAP-X Rainfall Prediction (T+{leadTime}h)
        </div>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          Atmospheric Post-Processing Comparison
        </span>
      </div>

      {/* Vertical Bar Chart Canvas */}
      <div style={{ position: 'relative', height: '170px', marginTop: '0.5rem', marginBottom: '0.25rem' }}>
        {/* Horizontal Background Grid Lines */}
        <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', pointerEvents: 'none' }}>
          {gridTicks.slice().reverse().map((tick) => (
            <div
              key={tick}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                borderBottom: '1px dashed #E8F1F7',
                fontSize: '0.62rem',
                color: 'var(--text-muted)',
              }}
            >
              <span style={{ minWidth: '32px' }}>{tick} mm</span>
            </div>
          ))}
        </div>

        {/* Bars Container */}
        <div
          style={{
            position: 'absolute',
            left: '36px',
            right: '8px',
            bottom: '0',
            top: '0',
            display: 'flex',
            alignItems: 'flex-end',
            justifyContent: 'space-around',
            gap: '12px',
          }}
        >
          {series.map((item, idx) => {
            const heightPct = Math.min(100, Math.max(4, (item.value / yMax) * 100));
            const isHovered = hoveredIdx === idx;

            return (
              <div
                key={item.id}
                style={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  height: '100%',
                  justifyContent: 'flex-end',
                  cursor: 'pointer',
                  position: 'relative',
                }}
                onMouseEnter={() => setHoveredIdx(idx)}
                onMouseLeave={() => setHoveredIdx(null)}
              >
                {/* Numeric Value Label Above Bar */}
                <span
                  style={{
                    fontSize: '0.72rem',
                    fontWeight: 700,
                    color: item.id === 'p90' ? '#285A7A' : 'var(--deep-blue)',
                    marginBottom: '4px',
                    transition: 'transform 0.15s ease',
                    transform: isHovered ? 'scale(1.1)' : 'scale(1)',
                  }}
                >
                  {item.value.toFixed(1)}
                  <span style={{ fontSize: '0.62rem', fontWeight: 500, color: 'var(--text-muted)', marginLeft: '1px' }}>
                    mm
                  </span>
                </span>

                {/* Flat Bar */}
                <div
                  style={{
                    width: '100%',
                    maxWidth: '48px',
                    height: `${heightPct}%`,
                    backgroundColor: item.color,
                    borderRadius: '4px 4px 0 0',
                    transition: 'all 0.2s ease',
                    boxShadow: isHovered ? '0 2px 8px rgba(40, 90, 122, 0.2)' : 'none',
                    opacity: hoveredIdx !== null && !isHovered ? 0.65 : 1,
                  }}
                />

                {/* Hover Tooltip */}
                {isHovered && (
                  <div
                    style={{
                      position: 'absolute',
                      bottom: `${heightPct + 18}%`,
                      left: '50%',
                      transform: 'translateX(-50%)',
                      background: 'rgba(40, 90, 122, 0.95)',
                      color: '#FFFFFF',
                      padding: '0.35rem 0.6rem',
                      borderRadius: '6px',
                      fontSize: '0.68rem',
                      whiteSpace: 'nowrap',
                      zIndex: 100,
                      boxShadow: 'var(--shadow-md)',
                      pointerEvents: 'none',
                    }}
                  >
                    <div style={{ fontWeight: 700 }}>{item.label}: {item.value.toFixed(1)} mm</div>
                    <div style={{ fontSize: '0.62rem', opacity: 0.85 }}>{item.desc} (T+{leadTime}h)</div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Bottom Axis Labels */}
      <div style={{ display: 'flex', marginLeft: '36px', marginRight: '8px', borderTop: '1px solid #D7E7F0', paddingTop: '4px' }}>
        {series.map((item) => (
          <div key={item.id} style={{ flex: 1, textAlign: 'center', fontSize: '0.7rem', color: 'var(--neutral-text)', fontWeight: 600 }}>
            {item.shortLabel}
          </div>
        ))}
      </div>

      {/* Delta Post-Processing Annotation */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '0.65rem', paddingTop: '0.45rem', borderTop: '1px solid #F1F7FB', fontSize: '0.72rem' }}>
        <span style={{ color: 'var(--neutral-text)' }}>Post-Processing Delta:</span>
        <span className={`delta-badge ${isZero ? '' : isPositive ? 'positive' : 'negative'}`}>
          {isZero ? '0.0 mm (Identical)' : `${isPositive ? '+' : ''}${diff.toFixed(1)} mm`}
          {isPositive ? ' (Orographic Offset)' : ' (Wet Bias Damping)'}
        </span>
      </div>
    </div>
  );
};
