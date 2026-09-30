import React, { useState, useMemo } from 'react';
import { Search, X, MapPin, Calendar, Compass, ShieldAlert, Award, CloudRain } from 'lucide-react';
import type { CanonicalDistrictForecastRecord } from '../types/forecast';

export type StoryTab = 'forecast' | 'why' | 'risk' | 'validation';

interface StoryBarProps {
  districts: CanonicalDistrictForecastRecord[];
  selectedDistrict: CanonicalDistrictForecastRecord | null;
  selectedDistrictId: string;
  onSelectDistrict: (districtId: string) => void;
  leadTime: number;
  onSelectLeadTime: (lt: number) => void;
  activeStoryTab: StoryTab;
  onSelectStoryTab: (tab: StoryTab) => void;
}

export const StoryBar: React.FC<StoryBarProps> = ({
  districts,
  selectedDistrict,
  selectedDistrictId,
  onSelectDistrict,
  leadTime,
  onSelectLeadTime,
  activeStoryTab,
  onSelectStoryTab,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [isOpen, setIsOpen] = useState(false);

  // Quick benchmark districts for one-click demo evaluation
  const benchmarkDistricts = [
    { id: 'MP_DHAR', label: 'Dhar', state: 'Madhya Pradesh' },
    { id: 'MH_PUNE', label: 'Pune', state: 'Maharashtra' },
    { id: 'MH_MUMBAI', label: 'Mumbai', state: 'Maharashtra' },
    { id: 'KL_WAYANAD', label: 'Wayanad', state: 'Kerala' },
    { id: 'UK_DEHRADUN', label: 'Dehradun', state: 'Uttarakhand' },
  ];

  const leadTimes = [
    { hours: 24, day: 'Day 1', date: '15 Jul 2026' },
    { hours: 48, day: 'Day 2', date: '16 Jul 2026' },
    { hours: 72, day: 'Day 3', date: '17 Jul 2026' },
    { hours: 96, day: 'Day 4', date: '18 Jul 2026' },
    { hours: 120, day: 'Day 5', date: '19 Jul 2026' },
  ];

  const filteredDistricts = useMemo(() => {
    if (!searchTerm.trim()) {
      return districts.slice(0, 8);
    }
    const term = searchTerm.toLowerCase().trim();
    return districts
      .filter(
        (d) =>
          d.district_name.toLowerCase().includes(term) ||
          d.district_id.toLowerCase().includes(term) ||
          (d.state && d.state.toLowerCase().includes(term))
      )
      .slice(0, 8);
  }, [districts, searchTerm]);

  const handleSelect = (id: string) => {
    onSelectDistrict(id);
    setSearchTerm('');
    setIsOpen(false);
  };

  const districtDisplayName = selectedDistrict
    ? `${selectedDistrict.district_name} District, ${selectedDistrict.state || 'India'}`
    : selectedDistrictId === 'MP_DHAR'
    ? 'Dhar District, Madhya Pradesh'
    : selectedDistrictId;

  return (
    <div
      style={{
        background: '#FFFFFF',
        borderBottom: '1px solid var(--soft-border)',
        boxShadow: '0 2px 8px rgba(40, 90, 122, 0.05)',
        position: 'sticky',
        top: 0,
        zIndex: 50,
      }}
    >
      {/* Row 1: WHERE? Location Selection & Forecast Day Selection */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0.65rem 1.25rem',
          gap: '0.75rem',
          borderBottom: '1px solid #EEF5F9',
        }}
      >
        {/* Left: District Selection (WHERE?) */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <MapPin size={17} color="var(--deep-blue)" style={{ flexShrink: 0 }} />
            <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--deep-blue)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Forecast Location:
            </span>
          </div>

          {/* Search Input Box */}
          <div style={{ position: 'relative', width: '220px' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                background: 'var(--cloud-white)',
                border: '1px solid var(--soft-border)',
                borderRadius: 'var(--radius-sm)',
                padding: '0.28rem 0.55rem',
              }}
            >
              <Search size={14} color="var(--text-muted)" style={{ marginRight: '0.35rem' }} />
              <input
                type="text"
                placeholder="Search district..."
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.target.value);
                  setIsOpen(true);
                }}
                onFocus={() => setIsOpen(true)}
                style={{
                  border: 'none',
                  outline: 'none',
                  background: 'transparent',
                  fontSize: '0.76rem',
                  color: 'var(--deep-blue)',
                  width: '100%',
                }}
              />
              {searchTerm && (
                <button
                  onClick={() => {
                    setSearchTerm('');
                    setIsOpen(false);
                  }}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}
                >
                  <X size={12} />
                </button>
              )}
            </div>

            {/* Dropdown Menu */}
            {isOpen && (
              <div
                style={{
                  position: 'absolute',
                  top: '100%',
                  left: 0,
                  right: 0,
                  marginTop: '0.25rem',
                  background: '#FFFFFF',
                  border: '1px solid var(--soft-border)',
                  borderRadius: 'var(--radius-sm)',
                  boxShadow: 'var(--shadow-md)',
                  maxHeight: '220px',
                  overflowY: 'auto',
                  zIndex: 100,
                }}
              >
                {filteredDistricts.length === 0 ? (
                  <div style={{ padding: '0.5rem', fontSize: '0.74rem', color: 'var(--text-muted)', textAlign: 'center' }}>
                    No districts found
                  </div>
                ) : (
                  filteredDistricts.map((d) => (
                    <div
                      key={d.district_id}
                      onClick={() => handleSelect(d.district_id)}
                      style={{
                        padding: '0.45rem 0.65rem',
                        fontSize: '0.75rem',
                        cursor: 'pointer',
                        borderBottom: '1px solid #F1F7FB',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        background:
                          selectedDistrictId.toUpperCase() === d.district_id.toUpperCase()
                            ? 'var(--very-light-sky)'
                            : '#FFFFFF',
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--very-light-sky)')}
                      onMouseLeave={(e) =>
                        (e.currentTarget.style.background =
                          selectedDistrictId.toUpperCase() === d.district_id.toUpperCase()
                            ? 'var(--very-light-sky)'
                            : '#FFFFFF')
                      }
                    >
                      <span style={{ fontWeight: 600, color: 'var(--deep-blue)' }}>{d.district_name}</span>
                      <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{d.state || d.district_id}</span>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>

          {/* Quick Selection Benchmark Pills */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', flexWrap: 'wrap' }}>
            {benchmarkDistricts.map((item) => {
              const isSelected = selectedDistrictId.toUpperCase() === item.id.toUpperCase();
              return (
                <button
                  key={item.id}
                  onClick={() => handleSelect(item.id)}
                  style={{
                    border: isSelected ? '1.5px solid var(--deep-blue)' : '1px solid var(--soft-border)',
                    background: isSelected ? 'var(--deep-blue)' : '#FFFFFF',
                    color: isSelected ? '#FFFFFF' : 'var(--deep-blue)',
                    padding: '0.22rem 0.55rem',
                    borderRadius: '9999px',
                    fontSize: '0.72rem',
                    fontWeight: isSelected ? 700 : 500,
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                  }}
                >
                  {item.label}
                </button>
              );
            })}
          </div>

          {/* Active District Callout Badge */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              background: '#F1F7FB',
              border: '1px solid #D0E5F5',
              borderRadius: 'var(--radius-sm)',
              padding: '0.22rem 0.65rem',
              fontSize: '0.76rem',
              color: 'var(--deep-blue)',
            }}
          >
            <span>Selected:</span>
            <strong style={{ color: 'var(--deep-blue)', letterSpacing: '0.01em' }}>
              {districtDisplayName}
            </strong>
          </div>
        </div>

        {/* Right: Forecast Day Selector (FORECAST DAY) */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.78rem', color: 'var(--deep-blue)', fontWeight: 700 }}>
            <Calendar size={15} color="var(--deep-blue)" />
            <span>Forecast Day:</span>
          </div>
          <div style={{ display: 'flex', gap: '0.25rem' }}>
            {leadTimes.map((lt) => {
              const isActive = leadTime === lt.hours;
              return (
                <button
                  key={lt.hours}
                  onClick={() => onSelectLeadTime(lt.hours)}
                  style={{
                    border: isActive ? '1.5px solid var(--deep-blue)' : '1px solid var(--soft-border)',
                    background: isActive ? 'var(--deep-blue)' : '#FFFFFF',
                    color: isActive ? '#FFFFFF' : 'var(--deep-blue)',
                    padding: '0.25rem 0.55rem',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: '0.72rem',
                    fontWeight: isActive ? 700 : 500,
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    lineHeight: 1.15,
                    transition: 'all 0.15s ease',
                  }}
                >
                  <span>T+{lt.hours}h</span>
                  <span style={{ fontSize: '0.62rem', opacity: isActive ? 0.9 : 0.7 }}>{lt.date}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Row 2: Secondary Navigation Tabs (Section 12: [ Forecast ] [ Why RAAP-X? ] [ Risk & Uncertainty ] [ Validation ]) */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '0.35rem 1.25rem',
          gap: '0.5rem',
          background: 'var(--very-light-sky)',
        }}
      >
        <span style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--neutral-text)', textTransform: 'uppercase', letterSpacing: '0.04em', marginRight: '0.25rem' }}>
          Story Journey:
        </span>

        <button
          onClick={() => onSelectStoryTab('forecast')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.35rem',
            padding: '0.35rem 0.85rem',
            borderRadius: '9999px',
            border: activeStoryTab === 'forecast' ? '1.5px solid var(--deep-blue)' : '1px solid #D0E5F5',
            background: activeStoryTab === 'forecast' ? 'var(--deep-blue)' : '#FFFFFF',
            color: activeStoryTab === 'forecast' ? '#FFFFFF' : 'var(--deep-blue)',
            fontWeight: 700,
            fontSize: '0.76rem',
            cursor: 'pointer',
            boxShadow: activeStoryTab === 'forecast' ? '0 2px 4px rgba(40, 90, 122, 0.15)' : 'none',
            transition: 'all 0.15s ease',
          }}
        >
          <CloudRain size={14} />
          <span>Forecast (What?)</span>
        </button>

        <button
          onClick={() => onSelectStoryTab('why')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.35rem',
            padding: '0.35rem 0.85rem',
            borderRadius: '9999px',
            border: activeStoryTab === 'why' ? '1.5px solid var(--deep-blue)' : '1px solid #D0E5F5',
            background: activeStoryTab === 'why' ? 'var(--deep-blue)' : '#FFFFFF',
            color: activeStoryTab === 'why' ? '#FFFFFF' : 'var(--deep-blue)',
            fontWeight: 700,
            fontSize: '0.76rem',
            cursor: 'pointer',
            boxShadow: activeStoryTab === 'why' ? '0 2px 4px rgba(40, 90, 122, 0.15)' : 'none',
            transition: 'all 0.15s ease',
          }}
        >
          <Compass size={14} />
          <span>Why RAAP-X?</span>
        </button>

        <button
          onClick={() => onSelectStoryTab('risk')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.35rem',
            padding: '0.35rem 0.85rem',
            borderRadius: '9999px',
            border: activeStoryTab === 'risk' ? '1.5px solid var(--deep-blue)' : '1px solid #D0E5F5',
            background: activeStoryTab === 'risk' ? 'var(--deep-blue)' : '#FFFFFF',
            color: activeStoryTab === 'risk' ? '#FFFFFF' : 'var(--deep-blue)',
            fontWeight: 700,
            fontSize: '0.76rem',
            cursor: 'pointer',
            boxShadow: activeStoryTab === 'risk' ? '0 2px 4px rgba(40, 90, 122, 0.15)' : 'none',
            transition: 'all 0.15s ease',
          }}
        >
          <ShieldAlert size={14} />
          <span>Risk & Uncertainty</span>
        </button>

        <button
          onClick={() => onSelectStoryTab('validation')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.35rem',
            padding: '0.35rem 0.85rem',
            borderRadius: '9999px',
            border: activeStoryTab === 'validation' ? '1.5px solid var(--deep-blue)' : '1px solid #D0E5F5',
            background: activeStoryTab === 'validation' ? 'var(--deep-blue)' : '#FFFFFF',
            color: activeStoryTab === 'validation' ? '#FFFFFF' : 'var(--deep-blue)',
            fontWeight: 700,
            fontSize: '0.76rem',
            cursor: 'pointer',
            boxShadow: activeStoryTab === 'validation' ? '0 2px 4px rgba(40, 90, 122, 0.15)' : 'none',
            transition: 'all 0.15s ease',
          }}
        >
          <Award size={14} />
          <span>Validation (Proof)</span>
        </button>
      </div>
    </div>
  );
};
