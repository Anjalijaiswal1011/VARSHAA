import { useState, useMemo } from 'react';
import { Search, X, MapPin } from 'lucide-react';
import type { CanonicalDistrictForecastRecord } from '../types/forecast';

interface DistrictSearchProps {
  districts: CanonicalDistrictForecastRecord[];
  onSelectDistrict: (districtId: string) => void;
  selectedDistrictId?: string;
}

export const DistrictSearch: React.FC<DistrictSearchProps> = ({
  districts,
  onSelectDistrict,
  selectedDistrictId,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [isOpen, setIsOpen] = useState(false);

  // Quick benchmark districts for one-click selection
  const quickDistricts = [
    { id: 'MP_DHAR', label: 'Dhar', state: 'Madhya Pradesh' },
    { id: 'MH_PUNE', label: 'Pune', state: 'Maharashtra' },
    { id: 'MH_MUMBAI', label: 'Mumbai', state: 'Maharashtra' },
    { id: 'KL_WAYANAD', label: 'Wayanad', state: 'Kerala' },
    { id: 'UK_DEHRADUN', label: 'Dehradun', state: 'Uttarakhand' },
  ];

  const filteredDistricts = useMemo(() => {
    if (!searchTerm.trim()) {
      // If search is empty but focused, show top benchmark districts
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

  return (
    <div className="map-search-bar">
      <div className="search-input-wrapper">
        <Search size={15} className="search-icon-left" />
        <input
          type="text"
          className="search-input"
          placeholder="🔍 Search district (e.g. Dhar)..."
          value={searchTerm}
          onChange={(e) => {
            setSearchTerm(e.target.value);
            setIsOpen(true);
          }}
          onFocus={() => setIsOpen(true)}
        />
        {searchTerm && (
          <button
            onClick={() => {
              setSearchTerm('');
              setIsOpen(false);
            }}
            style={{
              position: 'absolute',
              right: '0.6rem',
              background: 'transparent',
              border: 'none',
              color: 'var(--neutral-text)',
              cursor: 'pointer',
            }}
            aria-label="Clear search"
          >
            <X size={14} />
          </button>
        )}
      </div>

      {/* Quick Benchmark Pills for Instant One-Click Evaluation */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.25rem',
          marginTop: '0.35rem',
        }}
      >
        {quickDistricts.map((qd) => {
          const isSelected = selectedDistrictId && selectedDistrictId.toUpperCase() === qd.id.toUpperCase();
          return (
            <button
              key={qd.id}
              onClick={() => handleSelect(qd.id)}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.2rem',
                fontSize: '0.68rem',
                fontWeight: isSelected ? 700 : 500,
                padding: '0.18rem 0.45rem',
                borderRadius: '9999px',
                border: isSelected ? '1px solid var(--deep-blue)' : '1px solid rgba(40, 90, 122, 0.2)',
                background: isSelected ? 'var(--deep-blue)' : 'rgba(255, 255, 255, 0.92)',
                color: isSelected ? '#FFFFFF' : 'var(--deep-blue)',
                cursor: 'pointer',
                boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
                backdropFilter: 'blur(4px)',
                transition: 'all 0.15s ease',
              }}
            >
              <MapPin size={10} />
              <span>{qd.label}</span>
            </button>
          );
        })}
      </div>

      {isOpen && filteredDistricts.length > 0 && (
        <div className="search-dropdown">
          <div style={{ padding: '0.3rem 0.6rem', fontSize: '0.65rem', color: 'var(--text-muted)', borderBottom: '1px solid #F1F7FB' }}>
            {searchTerm ? `Search results for "${searchTerm}"` : 'Benchmark Operational Districts'}
          </div>
          {filteredDistricts.map((d) => {
            const isSelected = selectedDistrictId && d.district_id.toUpperCase() === selectedDistrictId.toUpperCase();
            return (
              <div
                key={d.district_id}
                className="search-item"
                style={isSelected ? { background: 'var(--very-light-sky)', borderLeft: '3px solid var(--sky-blue)' } : undefined}
                onClick={() => handleSelect(d.district_id)}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                    <MapPin size={12} color="var(--sky-blue)" />
                    <strong style={{ color: 'var(--deep-blue)', fontSize: '0.78rem' }}>{d.district_name} District</strong>
                  </div>
                  <div style={{ fontSize: '0.68rem', color: 'var(--neutral-text)', marginLeft: '1rem' }}>
                    {d.state || 'India'}
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--deep-blue)' }}>
                    {d.corrected_p50.toFixed(1)} mm
                  </div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                    NWP: {d.raw_nwp_rainfall.toFixed(1)} mm
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

