import React, { useState } from 'react';
import { ArrowRight, TrendingUp, CheckCircle, HelpCircle, Sliders, SplitSquareVertical } from 'lucide-react';
import type { CanonicalDistrictForecastRecord } from '../types/forecast';

interface WhatRaapxPredictsHeroProps {
  district: CanonicalDistrictForecastRecord | null;
  leadTime: number;
  onExploreWhy?: () => void;
}

export const WhatRaapxPredictsHero: React.FC<WhatRaapxPredictsHeroProps> = ({
  district,
  leadTime,
  onExploreWhy,
}) => {
  const [sliderPos, setSliderPos] = useState<number>(50); // 0 = 100% Raw NWP, 100 = 100% RAAP-X P50

  if (!district) return null;

  const rawNwp = district.raw_nwp_rainfall !== undefined && district.raw_nwp_rainfall !== null ? district.raw_nwp_rainfall : null;
  const p50 = district.corrected_p50 !== undefined && district.corrected_p50 !== null ? district.corrected_p50 : null;
  
  const hasRawNwp = rawNwp !== null && !isNaN(rawNwp);
  const hasP50 = p50 !== null && !isNaN(p50);
  
  const diff = hasRawNwp && hasP50
    ? (district.rainfall?.difference !== undefined ? district.rainfall.difference : Math.round((p50 - rawNwp) * 100) / 100)
    : null;
    
  const pctChange = hasRawNwp && hasP50 && rawNwp > 0 && diff !== null
    ? ((diff / rawNwp) * 100).toFixed(1)
    : null;

  // Calculate dynamic blended inspection value based on slider
  const inspectedVal = hasRawNwp && hasP50
    ? rawNwp + (p50 - rawNwp) * (sliderPos / 100)
    : (hasP50 ? p50 : (hasRawNwp ? rawNwp : 0));

  const districtName = district.district_name;
  const stateName = district.state || 'Madhya Pradesh';

  return (
    <div
      style={{
        background: '#FFFFFF',
        borderRadius: 'var(--radius-lg)',
        border: '1.5px solid #D0E5F5',
        boxShadow: 'var(--shadow-md)',
        padding: '1.15rem 1.4rem',
        marginBottom: '1rem',
      }}
    >
      {/* Header Row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.85rem' }}>
        <div>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', background: '#EAF6FD', color: 'var(--deep-blue)', padding: '0.2rem 0.6rem', borderRadius: '9999px', fontSize: '0.72rem', fontWeight: 700, marginBottom: '0.35rem' }}>
            <span>PRIMARY RESULT — WHAT DOES RAAP-X PREDICT?</span>
          </div>
          <h2 style={{ fontSize: '1.35rem', fontWeight: 800, color: 'var(--deep-blue)', margin: 0, letterSpacing: '-0.01em' }}>
            What Does RAAP-X Predict?
          </h2>
          <div style={{ fontSize: '0.82rem', color: 'var(--neutral-text)', marginTop: '0.15rem' }}>
            Forecast for <strong>{districtName} District</strong>, {stateName} • Valid Lead Time: <strong>T+{leadTime}h</strong>
          </div>
        </div>

        {onExploreWhy && (
          <button
            onClick={onExploreWhy}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              background: 'var(--very-light-sky)',
              border: '1.5px solid var(--sky-blue)',
              color: 'var(--deep-blue)',
              fontWeight: 700,
              fontSize: '0.78rem',
              padding: '0.45rem 0.85rem',
              borderRadius: 'var(--radius-md)',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'var(--sky-blue)';
              e.currentTarget.style.color = '#FFFFFF';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'var(--very-light-sky)';
              e.currentTarget.style.color = 'var(--deep-blue)';
            }}
          >
            <HelpCircle size={15} />
            <span>Why RAAP-X Changed It</span>
            <ArrowRight size={14} />
          </button>
        )}
      </div>

      {/* Primary Comparison Cards Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr auto 1fr',
          alignItems: 'center',
          gap: '1rem',
          background: 'var(--cloud-white)',
          border: '1px solid #E2EDF5',
          borderRadius: 'var(--radius-md)',
          padding: '1rem 1.25rem',
        }}
      >
        {/* Card 1: Raw NWP Baseline */}
        <div
          style={{
            background: sliderPos < 50 ? '#F0F8FF' : '#FFFFFF',
            border: sliderPos < 50 ? '2px solid var(--accent-blue)' : '1px solid #D7E7F0',
            borderRadius: 'var(--radius-md)',
            padding: '0.85rem 1rem',
            textAlign: 'center',
            boxShadow: '0 1px 3px rgba(0,0,0,0.02)',
            transition: 'all 0.2s ease',
          }}
        >
          <div style={{ fontSize: '0.74rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '0.25rem' }}>
            Raw NWP Baseline
          </div>
          {hasRawNwp ? (
            <div style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--neutral-text)', lineHeight: 1.1 }}>
              {rawNwp.toFixed(1)} <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>mm</span>
            </div>
          ) : (
            <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-muted)', padding: '0.35rem 0' }}>
              NWP baseline unavailable
            </div>
          )}
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>
            Uncorrected numerical model forecast
          </div>
        </div>

        {/* Center Connection Arrow & Correction Badge */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.35rem' }}>
          <div
            style={{
              background: diff !== null ? (diff >= 0 ? '#E2F5F2' : '#FBEFEF') : '#F1F7FB',
              border: `1.5px solid ${diff !== null ? (diff >= 0 ? '#A8DED7' : '#E9BDBD') : '#D0E5F5'}`,
              color: diff !== null ? (diff >= 0 ? '#2E8B80' : '#A34848') : 'var(--text-muted)',
              borderRadius: '9999px',
              padding: '0.3rem 0.75rem',
              fontWeight: 800,
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
              boxShadow: '0 1px 4px rgba(0,0,0,0.05)',
            }}
          >
            <TrendingUp size={15} />
            <span>
              {diff !== null ? (diff >= 0 ? `+${diff.toFixed(1)} mm` : `${diff.toFixed(1)} mm`) : 'Not available'}
            </span>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 600 }}>
            {pctChange !== null
              ? (diff !== null && diff >= 0 ? `+${pctChange}% AI Repair` : `${pctChange}% Damping`)
              : 'Correction'}
          </div>
          <ArrowRight size={20} color="var(--sky-blue)" />
        </div>

        {/* Card 2: RAAP-X Corrected Forecast (P50) */}
        <div
          style={{
            background: sliderPos >= 50 ? 'linear-gradient(180deg, #FFFFFF 0%, #EBF6FD 100%)' : '#FFFFFF',
            border: sliderPos >= 50 ? '2px solid var(--sky-blue)' : '1px solid #D7E7F0',
            borderRadius: 'var(--radius-md)',
            padding: '0.85rem 1rem',
            textAlign: 'center',
            boxShadow: sliderPos >= 50 ? '0 2px 8px rgba(142, 205, 243, 0.25)' : 'none',
            transition: 'all 0.2s ease',
          }}
        >
          <div style={{ fontSize: '0.74rem', fontWeight: 800, color: 'var(--deep-blue)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '0.25rem' }}>
            RAAP-X Corrected (P50)
          </div>
          {hasP50 ? (
            <div style={{ fontSize: '1.95rem', fontWeight: 900, color: 'var(--deep-blue)', lineHeight: 1.1 }}>
              {p50.toFixed(1)} <span style={{ fontSize: '1rem', fontWeight: 700 }}>mm</span>
            </div>
          ) : (
            <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--deep-blue)', padding: '0.35rem 0' }}>
              Output unavailable
            </div>
          )}
          <div style={{ fontSize: '0.7rem', color: '#2E8B80', fontWeight: 700, marginTop: '0.3rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.25rem' }}>
            <CheckCircle size={12} />
            <span>Regime-Aware Error Repaired</span>
          </div>
        </div>
      </div>

      {/* ============================================================== */}
      {/* INTERACTIVE COMPARISON SLIDER (RAW NWP vs RAAP-X AI REPAIR)    */}
      {/* ============================================================== */}
      <div
        style={{
          marginTop: '0.9rem',
          padding: '0.85rem 1rem',
          background: 'linear-gradient(180deg, #F9FCFE 0%, #FFFFFF 100%)',
          border: '1.5px solid #D5E7F3',
          borderRadius: 'var(--radius-md)',
          boxShadow: '0 2px 6px rgba(27, 73, 101, 0.04)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.82rem', fontWeight: 800, color: 'var(--deep-blue)' }}>
            <Sliders size={16} color="var(--sky-blue)" />
            <span>Interactive Comparison Slider: Raw NWP ⟷ RAAP-X AI</span>
          </div>

          {/* Quick preset buttons */}
          <div style={{ display: 'flex', gap: '0.35rem' }}>
            <button
              onClick={() => setSliderPos(0)}
              style={{
                fontSize: '0.68rem',
                fontWeight: sliderPos === 0 ? 800 : 600,
                padding: '0.2rem 0.55rem',
                borderRadius: '4px',
                border: sliderPos === 0 ? '1.5px solid var(--accent-blue)' : '1px solid #D0E1ED',
                background: sliderPos === 0 ? '#E6F4FC' : '#FFFFFF',
                color: sliderPos === 0 ? 'var(--deep-blue)' : 'var(--text-muted)',
                cursor: 'pointer',
              }}
            >
              Raw NWP (0%)
            </button>
            <button
              onClick={() => setSliderPos(50)}
              style={{
                fontSize: '0.68rem',
                fontWeight: sliderPos === 50 ? 800 : 600,
                padding: '0.2rem 0.55rem',
                borderRadius: '4px',
                border: sliderPos === 50 ? '1.5px solid var(--sky-blue)' : '1px solid #D0E1ED',
                background: sliderPos === 50 ? '#EAF6FD' : '#FFFFFF',
                color: sliderPos === 50 ? 'var(--deep-blue)' : 'var(--text-muted)',
                cursor: 'pointer',
              }}
            >
              50/50 Split
            </button>
            <button
              onClick={() => setSliderPos(100)}
              style={{
                fontSize: '0.68rem',
                fontWeight: sliderPos === 100 ? 800 : 600,
                padding: '0.2rem 0.55rem',
                borderRadius: '4px',
                border: sliderPos === 100 ? '1.5px solid #285A7A' : '1px solid #D0E1ED',
                background: sliderPos === 100 ? '#EAF2F7' : '#FFFFFF',
                color: sliderPos === 100 ? '#1B4965' : 'var(--text-muted)',
                cursor: 'pointer',
              }}
            >
              RAAP-X (100%)
            </button>
          </div>
        </div>

        {/* Dynamic Dual Comparison Bar with Draggable Slider Handle */}
        <div style={{ position: 'relative', margin: '0.75rem 0' }}>
          {/* Visual Track */}
          <div
            style={{
              height: '24px',
              borderRadius: '12px',
              background: 'linear-gradient(90deg, #B9DFF7 0%, #5A9EC8 50%, #1B4965 100%)',
              position: 'relative',
              boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.15)',
              overflow: 'hidden',
            }}
          >
            {/* Left side label */}
            <div
              style={{
                position: 'absolute',
                left: '12px',
                top: '50%',
                transform: 'translateY(-50%)',
                fontSize: '0.68rem',
                fontWeight: 800,
                color: '#1B4965',
                pointerEvents: 'none',
                letterSpacing: '0.02em',
              }}
            >
              ◀ RAW NWP ({hasRawNwp ? `${rawNwp.toFixed(1)} mm` : '—'})
            </div>

            {/* Right side label */}
            <div
              style={{
                position: 'absolute',
                right: '12px',
                top: '50%',
                transform: 'translateY(-50%)',
                fontSize: '0.68rem',
                fontWeight: 800,
                color: '#FFFFFF',
                pointerEvents: 'none',
                letterSpacing: '0.02em',
              }}
            >
              RAAP-X AI ({hasP50 ? `${p50.toFixed(1)} mm` : '—'}) ▶
            </div>
          </div>

          {/* Interactive Range Input Element overlay */}
          <input
            type="range"
            min="0"
            max="100"
            value={sliderPos}
            onChange={(e) => setSliderPos(Number(e.target.value))}
            style={{
              position: 'absolute',
              top: '-4px',
              left: '0px',
              width: '100%',
              height: '32px',
              opacity: 0,
              cursor: 'ew-resize',
              margin: 0,
              zIndex: 10,
            }}
            aria-label="Comparison slider between Raw NWP and RAAP-X AI forecast"
          />

          {/* Custom Visual Slider Thumb Cursor */}
          <div
            style={{
              position: 'absolute',
              top: '-3px',
              left: `calc(${sliderPos}% - 14px)`,
              width: '28px',
              height: '30px',
              background: '#FFFFFF',
              border: '2.5px solid var(--deep-blue)',
              borderRadius: '6px',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.25)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              pointerEvents: 'none',
              transition: 'left 0.05s ease-out',
              zIndex: 5,
            }}
          >
            <SplitSquareVertical size={16} color="var(--deep-blue)" />
          </div>
        </div>

        {/* Real-time Slider Inspector Status */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.74rem', color: 'var(--neutral-text)', marginTop: '0.45rem' }}>
          <div>
            <span>Inspected Focus: </span>
            <strong style={{ color: sliderPos < 40 ? '#1B4965' : sliderPos > 60 ? 'var(--deep-blue)' : '#2E8B80' }}>
              {sliderPos === 0
                ? '100% Raw NWP Baseline'
                : sliderPos === 100
                ? '100% RAAP-X Post-Processed (P50)'
                : `${100 - sliderPos}% Raw NWP / ${sliderPos}% RAAP-X Blend`}
            </strong>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span>Effective Rainfall: </span>
            <span
              style={{
                fontFamily: 'monospace',
                fontSize: '0.95rem',
                fontWeight: 900,
                color: 'var(--deep-blue)',
                background: '#EAF6FD',
                padding: '0.15rem 0.5rem',
                borderRadius: '4px',
                border: '1px solid #B9DFF7',
              }}
            >
              {inspectedVal.toFixed(1)} mm
            </span>
          </div>
        </div>
      </div>

      {/* Immediate User Understanding Sentence */}
      <div style={{ marginTop: '0.65rem', fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
        💡 <strong>Key Takeaway:</strong> RAAP-X corrected the raw NWP forecast by <strong>{diff !== null ? (diff >= 0 ? `+${diff.toFixed(1)} mm` : `${diff.toFixed(1)} mm`) : 'calibrated adjustment'}</strong> for {districtName} based on multi-scale error memory under the prevailing <strong>{(district.dominant_regime || district.regime?.dominant || 'ACTIVE_MONSOON').replace(/_/g, ' ')}</strong> regime.
      </div>
    </div>
  );
};
