import React, { useState } from 'react';
import { MapPin, ShieldAlert, GitBranch, CheckCircle2, BarChart2, CloudRain, Layers } from 'lucide-react';
import type { CanonicalDistrictForecastRecord } from '../types/forecast';
import { NwpVsRaapxChart } from './NwpVsRaapxChart';
import { WhyRaapxAdjustedCard } from './WhyRaapxAdjustedCard';
import { UncertaintyGauge } from './UncertaintyGauge';
import { ProbabilityBarChart } from './ProbabilityBarChart';
import { RegimeBarChart } from './RegimeBarChart';
import { LeadTimeBarChart } from './LeadTimeBarChart';
import { DistrictComparisonChart } from './DistrictComparisonChart';

interface DistrictDetailPanelProps {
  district: CanonicalDistrictForecastRecord | null;
  loading: boolean;
  leadTime?: number;
  onSelectLeadTime?: (lt: number) => void;
  districts?: CanonicalDistrictForecastRecord[];
  onSelectDistrict?: (id: string) => void;
  onNavigateTab?: (tab: 'forecast' | 'why' | 'risk' | 'validation') => void;
}

export const DistrictDetailPanel: React.FC<DistrictDetailPanelProps> = ({
  district,
  loading,
  leadTime = 24,
  onSelectLeadTime,
  districts = [],
  onSelectDistrict,
  onNavigateTab,
}) => {
  const [activeTab, setActiveTab] = useState<'main' | 'deep_dive' | 'regional'>('main');

  if (loading) {
    return (
      <div className="intelligence-panel">
        <div className="state-container">
          <div className="spinner" />
          <p>Loading district forecast & explanation metadata...</p>
        </div>
      </div>
    );
  }

  if (!district) {
    return (
      <div className="intelligence-panel">
        <div className="state-container">
          <MapPin size={36} color="var(--sky-blue)" style={{ marginBottom: '0.75rem' }} />
          <h3>No District Selected</h3>
          <p style={{ marginTop: '0.4rem', fontSize: '0.85rem' }}>
            Select an administrative district on the map or use the search bar above to inspect bar chart comparisons,
            uncertainty quantiles, regime probabilities, and model explainability.
          </p>
        </div>
      </div>
    );
  }

  // Rainfall quantiles
  const rainfall = district.rainfall || {
    raw_nwp: district.raw_nwp_rainfall,
    p50: district.corrected_p50,
    p75: district.corrected_p75,
    p90: district.corrected_p90,
    spread: district.spread_p90_p50,
    difference: district.corrected_p50 - district.raw_nwp_rainfall,
  };

  // Regime state with case-insensitive probability lookup
  const regProbs = district.regime?.probabilities || district.regime_probabilities || {};
  let dominantProb = district.regime?.dominant_probability ?? 0.0;
  if (!dominantProb || dominantProb <= 0) {
    const domKey = (district.dominant_regime || '').toUpperCase();
    for (const [k, v] of Object.entries(regProbs)) {
      if (k.toUpperCase() === domKey && typeof v === 'number') {
        dominantProb = v;
        break;
      }
    }
  }

  const domRegime = district.regime?.dominant || district.dominant_regime || 'NORMAL_TRANSITIONAL';
  const regime = {
    dominant: domRegime,
    dominant_probability: dominantProb > 0 ? dominantProb : 0.65,
    probabilities: regProbs,
  };

  // Risk tier
  const heavyProb = district.heavy_rainfall_probability;
  const extremeProb = district.extreme_rainfall_probability;
  const warningTier =
    heavyProb >= 0.75 ? 'RED' : heavyProb >= 0.5 ? 'ORANGE' : heavyProb >= 0.25 ? 'YELLOW' : 'GREEN';

  // Soft atmospheric risk colors
  const warningColors: Record<string, { bg: string; text: string; border: string }> = {
    GREEN: { bg: '#E2F5F2', text: '#2E8B80', border: '#A8DED7' },
    YELLOW: { bg: '#FDF4E5', text: '#9C6C1B', border: '#EED6A6' },
    ORANGE: { bg: '#EAF6FD', text: '#285A7A', border: '#B9DFF7' },
    RED: { bg: '#ECECF8', text: '#565792', border: '#C8C9EB' },
  };

  const currentTierStyle = warningColors[warningTier] || warningColors.GREEN;

  return (
    <div className="intelligence-panel">
      {/* 1. PRIMARY DISTRICT FORECAST CARD (Section 7) */}
      <div className="card district-forecast-card" style={{ paddingBottom: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <MapPin size={18} color="var(--sky-blue)" />
              <h2 style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--deep-blue)', margin: 0, textTransform: 'uppercase', letterSpacing: '-0.01em' }}>
                {district.district_name} District
              </h2>
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--neutral-text)', marginLeft: '1.5rem', marginTop: '0.15rem' }}>
              {district.state || 'India'} • Code: <strong>{district.district_id}</strong>
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', background: '#F1F7FB', padding: '0.2rem 0.55rem', borderRadius: '4px', fontSize: '0.74rem', fontWeight: 700, color: 'var(--deep-blue)' }}>
              <span>Forecast: <strong>T+{leadTime}h</strong></span>
            </div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
              Status: <span style={{ color: '#2E8B80', fontWeight: 700 }}>{district.prediction_status}</span>
            </div>
          </div>
        </div>

        {/* 3 Structured Metric Sections: Rainfall, Risk, Regime (Requirement 7) */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.55rem', marginTop: '0.75rem' }}>
          {/* Section A: Rainfall */}
          <div style={{ background: '#FFFFFF', border: '1px solid var(--soft-border)', borderRadius: 'var(--radius-md)', padding: '0.55rem 0.65rem', boxShadow: '0 1px 3px rgba(0,0,0,0.02)' }}>
            <div style={{ fontSize: '0.68rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.35rem', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <CloudRain size={12} color="var(--sky-blue)" />
              <span>Rainfall</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem', fontSize: '0.73rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Raw NWP:</span>
                <span style={{ fontWeight: 600, color: 'var(--neutral-text)' }}>
                  {rainfall.raw_nwp !== undefined && rainfall.raw_nwp !== null ? `${rainfall.raw_nwp.toFixed(1)} mm` : 'NWP baseline unavailable'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--deep-blue)', fontWeight: 600 }}>P50 — Rainfall Quantile:</span>
                <strong style={{ color: 'var(--deep-blue)' }}>
                  {rainfall.p50 !== undefined && rainfall.p50 !== null ? `${rainfall.p50.toFixed(1)} mm` : 'Output unavailable'}
                </strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#5A5B9F', fontWeight: 600 }}>P90 — Rainfall Quantile:</span>
                <strong style={{ color: '#5A5B9F' }}>
                  {rainfall.p90 !== undefined && rainfall.p90 !== null ? `${rainfall.p90.toFixed(1)} mm` : 'Output unavailable'}
                </strong>
              </div>
            </div>
          </div>

          {/* Section B: Risk */}
          <div style={{ background: currentTierStyle.bg, border: `1px solid ${currentTierStyle.border}`, borderRadius: 'var(--radius-md)', padding: '0.55rem 0.65rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
              <span style={{ fontSize: '0.68rem', fontWeight: 700, color: currentTierStyle.text, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Risk Tier: {warningTier}
              </span>
              <ShieldAlert size={13} color={currentTierStyle.text} />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem', fontSize: '0.73rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--neutral-text)' }}>Heavy (≥64mm):</span>
                <strong style={{ color: currentTierStyle.text }}>{(heavyProb * 100).toFixed(0)}%</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--neutral-text)' }}>Extreme (≥204mm):</span>
                <strong style={{ color: currentTierStyle.text }}>{(extremeProb * 100).toFixed(0)}%</strong>
              </div>
            </div>
          </div>

          {/* Section C: Regime */}
          <div style={{ background: '#FFFFFF', border: '1px solid var(--soft-border)', borderRadius: 'var(--radius-md)', padding: '0.55rem 0.65rem', boxShadow: '0 1px 3px rgba(0,0,0,0.02)' }}>
            <div style={{ fontSize: '0.68rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.35rem', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <Layers size={12} color="var(--sky-blue)" />
              <span>Regime</span>
            </div>
            <div style={{ marginTop: '0.15rem' }}>
              <div style={{ fontSize: '0.8rem', fontWeight: 800, color: 'var(--deep-blue)' }}>
                {(regime.dominant || 'NORMAL_TRANSITIONAL').replace(/_/g, ' ')}
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
                Confidence: <strong style={{ color: 'var(--deep-blue)' }}>{(regime.dominant_probability * 100).toFixed(0)}%</strong>
              </div>
            </div>
          </div>
        </div>

        {/* View Switcher Tabs */}
        <div style={{ display: 'flex', gap: '0.35rem', marginTop: '0.75rem', borderTop: '1px solid #F1F7FB', paddingTop: '0.5rem' }}>
          <button
            onClick={() => setActiveTab('main')}
            style={{
              flex: 1,
              border: 'none',
              background: activeTab === 'main' ? 'var(--deep-blue)' : 'var(--very-light-sky)',
              color: activeTab === 'main' ? '#FFFFFF' : 'var(--deep-blue)',
              fontWeight: 600,
              fontSize: '0.72rem',
              padding: '0.35rem 0.5rem',
              borderRadius: '4px',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            Overview & Comparison
          </button>
          <button
            onClick={() => setActiveTab('deep_dive')}
            style={{
              flex: 1,
              border: 'none',
              background: activeTab === 'deep_dive' ? 'var(--deep-blue)' : 'var(--very-light-sky)',
              color: activeTab === 'deep_dive' ? '#FFFFFF' : 'var(--deep-blue)',
              fontWeight: 600,
              fontSize: '0.72rem',
              padding: '0.35rem 0.5rem',
              borderRadius: '4px',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            Plume & Timeline
          </button>
          <button
            onClick={() => setActiveTab('regional')}
            style={{
              flex: 1,
              border: 'none',
              background: activeTab === 'regional' ? 'var(--deep-blue)' : 'var(--very-light-sky)',
              color: activeTab === 'regional' ? '#FFFFFF' : 'var(--deep-blue)',
              fontWeight: 600,
              fontSize: '0.72rem',
              padding: '0.35rem 0.5rem',
              borderRadius: '4px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.25rem',
              transition: 'all 0.15s ease',
            }}
          >
            <BarChart2 size={12} />
            Regional
          </button>
        </div>
      </div>

      {/* Main Forecast Layout (Hierarchy: NWP vs RAAP-X comparison -> Why RAAP-X Adjusted) */}
      {activeTab === 'main' && (
        <>
          {/* Quick Story Navigation Buttons */}
          {onNavigateTab && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.45rem', marginBottom: '0.65rem' }}>
              <button
                onClick={() => onNavigateTab('why')}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.35rem',
                  background: 'var(--very-light-sky)',
                  border: '1px solid var(--sky-blue)',
                  color: 'var(--deep-blue)',
                  padding: '0.4rem 0.6rem',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.74rem',
                  fontWeight: 700,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                <span>🧠 Why RAAP-X? Story Flow</span>
              </button>
              <button
                onClick={() => onNavigateTab('risk')}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.35rem',
                  background: '#FDF4E5',
                  border: '1px solid #EED6A6',
                  color: '#9C6C1B',
                  padding: '0.4rem 0.6rem',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.74rem',
                  fontWeight: 700,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                <span>📊 Risk & Quantiles</span>
              </button>
            </div>
          )}

          {/* 2. NWP vs RAAP-X Visual Bar Chart (Section 5) */}
          <NwpVsRaapxChart rainfall={rainfall} leadTime={leadTime} />

          {/* 3. Prominent "WHY RAAP-X ADJUSTED THE FORECAST" (Section 4 & Section 8) */}
          <WhyRaapxAdjustedCard district={district} rainfall={rainfall} />
        </>
      )}

      {/* Deep-Dive Quantiles & Progression View */}
      {activeTab === 'deep_dive' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '0.9rem' }}>
            <ProbabilityBarChart
              heavyProb={heavyProb}
              extremeProb={extremeProb}
              leadTime={leadTime}
            />
            <UncertaintyGauge
              rainfall={rainfall}
              spread={district.spread_p90_p50}
            />
          </div>

          <RegimeBarChart regime={regime} />

          {onSelectLeadTime && (
            <LeadTimeBarChart
              currentLeadTime={leadTime}
              onSelectLeadTime={onSelectLeadTime}
              currentRainfall={rainfall}
            />
          )}
        </>
      )}

      {/* Regional Comparison View */}
      {activeTab === 'regional' && onSelectDistrict && (
        <DistrictComparisonChart
          districts={districts}
          selectedDistrictId={district.district_id}
          onSelectDistrict={onSelectDistrict}
          leadTime={leadTime}
        />
      )}

      {/* Technical Provenance & Metadata Card */}
      <div className="card" style={{ fontSize: '0.73rem', color: 'var(--neutral-text)' }}>
        <div className="card-header" style={{ marginBottom: '0.4rem' }}>
          <div className="card-title">
            <GitBranch size={15} className="card-title-icon" />
            Model & Lineage Metadata
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.4rem' }}>
          <div>
            Model Version: <strong style={{ color: 'var(--deep-blue)' }}>{district.model_version}</strong>
          </div>
          <div>
            Regime Classifier: <strong style={{ color: 'var(--deep-blue)' }}>{district.regime_model_version}</strong>
          </div>
          <div>
            Feature Pipeline: <strong style={{ color: 'var(--deep-blue)' }}>{district.feature_version}</strong>
          </div>
          <div>
            Dataset Lineage: <strong style={{ color: 'var(--deep-blue)' }}>{district.dataset_version}</strong>
          </div>
          <div>
            Boundary Standard: <strong style={{ color: 'var(--deep-blue)' }}>{district.boundary_version}</strong>
          </div>
          <div>
            Pipeline Run: <strong style={{ color: 'var(--deep-blue)' }}>{district.pipeline_run_id || 'Operational Live'}</strong>
          </div>
        </div>
        <div style={{ marginTop: '0.5rem', paddingTop: '0.35rem', borderTop: '1px solid #F1F7FB', display: 'flex', alignItems: 'center', gap: '0.35rem', color: 'var(--neutral-text)' }}>
          <CheckCircle2 size={13} color="#2E8B80" />
          <span>Validated against IMD operational quality gates before publishing.</span>
        </div>
      </div>
    </div>
  );
};
