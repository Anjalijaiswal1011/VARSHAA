import React, { useState } from 'react';
import { AlertTriangle } from 'lucide-react';

interface ProbabilityBarChartProps {
  heavyProb: number;   // 0.0 to 1.0
  extremeProb: number; // 0.0 to 1.0
  leadTime: number;
}

export const ProbabilityBarChart: React.FC<ProbabilityBarChartProps> = ({
  heavyProb,
  extremeProb,
  leadTime,
}) => {
  const [hoveredMetric, setHoveredMetric] = useState<string | null>(null);

  const items = [
    {
      id: 'heavy',
      label: 'Heavy Rainfall',
      threshold: '≥ 64.5 mm / 24h',
      prob: Math.min(1, Math.max(0, heavyProb)),
      color: '#3B769E', // Deep Sky Blue
      desc: 'IMD Heavy Rainfall criterion',
    },
    {
      id: 'extreme',
      label: 'Extreme Rainfall',
      threshold: '≥ 115.5 mm / 24h',
      prob: Math.min(1, Math.max(0, extremeProb)),
      color: '#6B6CA8', // Controlled Violet/Indigo Accent
      desc: 'IMD Very Heavy/Extreme Rainfall criterion',
    },
  ];

  return (
    <div className="card">
      <div className="card-header" style={{ marginBottom: '0.4rem' }}>
        <div className="card-title">
          <AlertTriangle size={15} className="card-title-icon" />
          Rainfall Risk Probabilities (T+{leadTime}h)
        </div>
        <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
          Exceedance Probability (%)
        </span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.4rem' }}>
        {items.map((item) => {
          const pct = Math.round(item.prob * 100);
          const isHovered = hoveredMetric === item.id;

          return (
            <div
              key={item.id}
              style={{ position: 'relative', cursor: 'pointer' }}
              onMouseEnter={() => setHoveredMetric(item.id)}
              onMouseLeave={() => setHoveredMetric(null)}
            >
              {/* Header Label Row */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.25rem' }}>
                <div>
                  <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--deep-blue)' }}>
                    {item.label}
                  </span>
                  <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginLeft: '0.4rem' }}>
                    ({item.threshold})
                  </span>
                </div>
                <strong style={{ fontSize: '0.85rem', color: item.color, fontWeight: 700 }}>
                  {pct}%
                </strong>
              </div>

              {/* Horizontal Bar Track */}
              <div
                style={{
                  width: '100%',
                  height: '14px',
                  background: '#E8F1F7',
                  borderRadius: '4px',
                  overflow: 'hidden',
                  position: 'relative',
                  border: '1px solid #D7E7F0',
                }}
              >
                <div
                  style={{
                    height: '100%',
                    width: `${Math.max(2, pct)}%`,
                    backgroundColor: item.color,
                    borderRadius: '0 4px 4px 0',
                    transition: 'width 0.4s ease, opacity 0.15s ease',
                    opacity: isHovered ? 0.9 : 0.85,
                  }}
                />
              </div>

              {/* Hover Tooltip */}
              {isHovered && (
                <div
                  style={{
                    position: 'absolute',
                    top: '-32px',
                    right: '0',
                    background: 'rgba(40, 90, 122, 0.95)',
                    color: '#FFFFFF',
                    padding: '0.3rem 0.6rem',
                    borderRadius: '4px',
                    fontSize: '0.65rem',
                    boxShadow: 'var(--shadow-md)',
                    zIndex: 100,
                    pointerEvents: 'none',
                  }}
                >
                  {item.label} Probability: <strong>{pct}%</strong> ({item.threshold})
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Axis Scale Marker */}
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.62rem', color: 'var(--text-muted)', marginTop: '0.5rem', paddingTop: '0.3rem', borderTop: '1px dashed #E8F1F7' }}>
        <span>0%</span>
        <span>25%</span>
        <span>50%</span>
        <span>75%</span>
        <span>100%</span>
      </div>
    </div>
  );
};
