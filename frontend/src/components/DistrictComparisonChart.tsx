import React, { useState } from 'react';
import { BarChart2 } from 'lucide-react';
import type { CanonicalDistrictForecastRecord } from '../types/forecast';

interface DistrictComparisonChartProps {
  districts: CanonicalDistrictForecastRecord[];
  selectedDistrictId?: string;
  onSelectDistrict: (id: string) => void;
  leadTime: number;
}

export const DistrictComparisonChart: React.FC<DistrictComparisonChartProps> = ({
  districts,
  selectedDistrictId,
  onSelectDistrict,
  leadTime,
}) => {
  const [metric, setMetric] = useState<'p50' | 'p90' | 'heavy'>('p50');

  if (!districts || districts.length === 0) {
    return null;
  }

  // Pick top 6 districts ranked by active metric for clean visualization
  const sorted = [...districts].sort((a, b) => {
    if (metric === 'p50') return b.corrected_p50 - a.corrected_p50;
    if (metric === 'p90') return b.corrected_p90 - a.corrected_p90;
    return b.heavy_rainfall_probability - a.heavy_rainfall_probability;
  }).slice(0, 6);

  const maxVal = metric === 'heavy'
    ? 1.0
    : Math.max(10, ...sorted.map((d) => (metric === 'p50' ? d.corrected_p50 : d.corrected_p90)));

  return (
    <div className="card">
      <div className="card-header" style={{ marginBottom: '0.4rem' }}>
        <div className="card-title">
          <BarChart2 size={15} className="card-title-icon" />
          Regional District Comparison (T+{leadTime}h)
        </div>

        {/* Metric Selector Tabs */}
        <div style={{ display: 'flex', gap: '0.2rem', background: 'var(--very-light-sky)', padding: '0.15rem', borderRadius: '4px', border: '1px solid var(--soft-border)' }}>
          {(['p50', 'p90', 'heavy'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMetric(m)}
              style={{
                border: 'none',
                background: metric === m ? '#FFFFFF' : 'transparent',
                color: metric === m ? 'var(--deep-blue)' : 'var(--neutral-text)',
                fontWeight: metric === m ? 700 : 500,
                fontSize: '0.65rem',
                padding: '0.18rem 0.45rem',
                borderRadius: '3px',
                cursor: 'pointer',
              }}
            >
              {m === 'heavy' ? 'HEAVY %' : m.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', marginTop: '0.35rem' }}>
        {sorted.map((d) => {
          const isSelected = selectedDistrictId && d.district_id.toUpperCase() === selectedDistrictId.toUpperCase();
          const val = metric === 'p50' ? d.corrected_p50 : metric === 'p90' ? d.corrected_p90 : d.heavy_rainfall_probability * 100;
          const pct = metric === 'heavy' ? val : (val / maxVal) * 100;

          return (
            <div
              key={d.district_id}
              onClick={() => onSelectDistrict(d.district_id)}
              style={{
                cursor: 'pointer',
                padding: '0.25rem 0.4rem',
                borderRadius: '4px',
                background: isSelected ? 'var(--very-light-sky)' : 'transparent',
                borderLeft: isSelected ? '3px solid var(--deep-blue)' : '3px solid transparent',
                transition: 'background 0.15s ease',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', marginBottom: '2px' }}>
                <span style={{ color: isSelected ? 'var(--deep-blue)' : 'var(--neutral-text)', fontWeight: isSelected ? 700 : 500 }}>
                  {d.district_name}
                </span>
                <strong style={{ color: 'var(--deep-blue)', fontSize: '0.75rem' }}>
                  {val.toFixed(1)} {metric === 'heavy' ? '%' : 'mm'}
                </strong>
              </div>

              {/* Horizontal Bar */}
              <div style={{ width: '100%', height: '6px', background: '#E8F1F7', borderRadius: '3px', overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    width: `${Math.max(2, pct)}%`,
                    backgroundColor: isSelected ? '#285A7A' : '#8ECDF3',
                    borderRadius: '0 3px 3px 0',
                    transition: 'width 0.3s ease',
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
