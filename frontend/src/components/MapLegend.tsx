import type { MetricType } from '../types/forecast';

interface MapLegendProps {
  activeMetric: MetricType;
}

export const MapLegend: React.FC<MapLegendProps> = ({ activeMetric }) => {
  const isProbability = activeMetric === 'heavy_prob' || activeMetric === 'extreme_prob';

  // Smooth sky progression: pale blue -> soft cyan -> medium sky -> deeper blue -> subtle violet
  const rainfallSteps = [
    { label: '<15', color: '#EAF6FD' },
    { label: '15-35', color: '#BFEAF2' },
    { label: '35-65', color: '#8ECDF3' },
    { label: '65-115', color: '#5198C5' },
    { label: '115+', color: '#8285D6' },
  ];

  // Soft probability progression: pale sky -> soft cyan -> sky blue -> deeper violet
  const probabilitySteps = [
    { label: '<25%', color: '#EAF6FD' },
    { label: '25-50%', color: '#BFEAF2' },
    { label: '50-75%', color: '#8ECDF3' },
    { label: '≥75%', color: '#6B6CA8' },
  ];

  return (
    <div className="map-legend-card">
      <div className="legend-title">
        {isProbability ? 'Exceedance Probability (%)' : 'Rainfall Accumulation (mm/24h)'}
      </div>
      <div className="legend-bar-container">
        {isProbability
          ? probabilitySteps.map((step, idx) => (
              <div key={idx} className="legend-step">
                <div className="legend-color-box" style={{ backgroundColor: step.color }} />
                <span className="legend-label">{step.label}</span>
              </div>
            ))
          : rainfallSteps.map((step, idx) => (
              <div key={idx} className="legend-step">
                <div className="legend-color-box" style={{ backgroundColor: step.color }} />
                <span className="legend-label">{step.label}</span>
              </div>
            ))}
      </div>
    </div>
  );
};
