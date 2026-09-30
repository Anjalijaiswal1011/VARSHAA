import { HelpCircle, BarChart3, Info } from 'lucide-react';
import type { DistrictExplanationMetadata } from '../types/forecast';

interface ExplainabilityPanelProps {
  explanation?: DistrictExplanationMetadata;
}

export const ExplainabilityPanel: React.FC<ExplainabilityPanelProps> = ({ explanation }) => {
  if (!explanation || !explanation.top_features || explanation.top_features.length === 0) {
    return (
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <HelpCircle size={16} className="card-title-icon" />
            Model Feature Contribution (SHAP)
          </div>
        </div>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Detailed feature attribution metadata is loading or unavailable for this district.
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <BarChart3 size={16} className="card-title-icon" />
          Model Feature Contribution
        </div>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          SHAP Attribution ({explanation.explanation_version})
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.73rem', color: 'var(--neutral-text)', marginBottom: '0.65rem' }}>
        <Info size={13} color="var(--deep-blue)" />
        <span>Statistical feature contributions to model prediction (non-causal framing).</span>
      </div>

      {/* Top Drivers List */}
      <div className="driver-list">
        {explanation.top_features.map((driver, idx) => {
          const isPositive = driver.contribution_mm >= 0;
          return (
            <div key={idx} className="driver-item">
              <div className="driver-name">
                <span>{driver.feature}</span>
                <span style={{ color: isPositive ? 'var(--deep-blue)' : 'var(--neutral-text)', fontWeight: 700 }}>
                  {isPositive ? '+' : ''}{driver.contribution_mm.toFixed(1)} mm
                </span>
              </div>
              <div className="driver-desc">{driver.description}</div>
            </div>
          );
        })}
      </div>

      {explanation.user_friendly_summary && (
        <div style={{ marginTop: '0.75rem', padding: '0.6rem 0.75rem', background: 'var(--very-light-sky)', borderRadius: '6px', fontSize: '0.75rem', color: 'var(--deep-blue)', borderLeft: '3px solid var(--sky-blue)' }}>
          <strong>Summary:</strong> {explanation.user_friendly_summary}
        </div>
      )}
    </div>
  );
};
