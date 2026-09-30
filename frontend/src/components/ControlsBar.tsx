import { Calendar, Gauge } from 'lucide-react';
import type { MetricType } from '../types/forecast';

interface ControlsBarProps {
  leadTime: number;
  onSelectLeadTime: (lt: number) => void;
  activeMetric: MetricType;
  onSelectMetric: (metric: MetricType) => void;
  leadTimes?: number[];
}

export const ControlsBar: React.FC<ControlsBarProps> = ({
  leadTime,
  onSelectLeadTime,
  activeMetric,
  onSelectMetric,
  leadTimes = [24, 48, 72, 96, 120],
}) => {
  const metricOptions: { id: MetricType; label: string }[] = [
    { id: 'p50', label: 'P50 (Median)' },
    { id: 'p75', label: 'P75 (Upper)' },
    { id: 'p90', label: 'P90 Quantile' },
    { id: 'raw_nwp', label: 'Raw NWP' },
    { id: 'heavy_prob', label: 'Heavy Prob (≥64.5mm)' },
    { id: 'extreme_prob', label: 'Extreme Prob (≥204.5mm)' },
  ];

  return (
    <div className="controls-bar">
      {/* Lead Time Timeline Chips */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span style={{ fontSize: '0.74rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
          <Calendar size={13} />
          Lead Time:
        </span>
        <div className="timeline-chips">
          {leadTimes.map((lt) => {
            const dayNum = Math.floor(lt / 24);
            return (
              <button
                key={lt}
                className={`timeline-btn ${leadTime === lt ? 'active' : ''}`}
                onClick={() => onSelectLeadTime(lt)}
              >
                T+{lt}h (Day {dayNum})
              </button>
            );
          })}
        </div>
      </div>

      {/* Metric Selector */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <span style={{ fontSize: '0.74rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
          <Gauge size={13} />
          Layer:
        </span>
        <div className="metric-selector">
          {metricOptions.map((opt) => (
            <button
              key={opt.id}
              className={`metric-btn ${activeMetric === opt.id ? 'active' : ''}`}
              onClick={() => onSelectMetric(opt.id)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
