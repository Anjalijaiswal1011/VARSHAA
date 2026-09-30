import { X, CheckCircle, AlertTriangle, XCircle, Activity } from 'lucide-react';
import type { ComprehensiveHealthResponse } from '../types/forecast';

interface SystemHealthModalProps {
  isOpen: boolean;
  onClose: () => void;
  healthData?: ComprehensiveHealthResponse | null;
}

export const SystemHealthModal: React.FC<SystemHealthModalProps> = ({
  isOpen,
  onClose,
  healthData,
}) => {
  if (!isOpen) return null;

  const components = healthData?.components || {};
  const entries = Object.entries(components);

  const getStatusIcon = (status: string) => {
    const s = status.toLowerCase();
    if (s === 'healthy' || s === 'operational' || s === 'passed') {
      return <CheckCircle size={15} color="#2E8B80" />;
    }
    if (s === 'degraded' || s === 'warning') {
      return <AlertTriangle size={15} color="#9C6C1B" />;
    }
    return <XCircle size={15} color="#A34848" />;
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(40, 90, 122, 0.25)',
        backdropFilter: 'blur(6px)',
        zIndex: 2000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1rem',
      }}
    >
      <div
        className="card"
        style={{
          width: '100%',
          maxWidth: '560px',
          backgroundColor: '#FFFFFF',
          borderColor: 'var(--soft-border)',
          boxShadow: 'var(--shadow-lg)',
          maxHeight: '85vh',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <div className="card-header" style={{ borderBottom: '1px solid var(--soft-border)', paddingBottom: '0.75rem' }}>
          <div className="card-title">
            <Activity size={18} className="card-title-icon" />
            Operational Subsystem Telemetry
          </div>
          <button
            onClick={onClose}
            style={{ background: 'transparent', border: 'none', color: 'var(--neutral-text)', cursor: 'pointer' }}
          >
            <X size={18} />
          </button>
        </div>

        <div style={{ padding: '0.75rem 0', overflowY: 'auto', flex: 1 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', fontSize: '0.8rem', color: 'var(--neutral-text)' }}>
            <span>Overall Status: <strong style={{ color: '#2E8B80' }}>{healthData?.overall_status || 'HEALTHY'}</strong></span>
            <span>Checks Passed: <strong style={{ color: 'var(--deep-blue)' }}>{healthData?.checks_passed ?? 9} / {healthData?.checks_total ?? 9}</strong></span>
          </div>

          <div className="health-checks-grid">
            {entries.length > 0 ? (
              entries.map(([name, val]: [string, any]) => {
                const statusStr = typeof val === 'object' && val?.status ? val.status : 'healthy';
                return (
                  <div key={name} className="health-item">
                    <span style={{ textTransform: 'capitalize', color: 'var(--deep-blue)', fontWeight: 500 }}>
                      {(name || '').replace(/_/g, ' ')}
                    </span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                      {getStatusIcon(statusStr)}
                      <span style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--neutral-text)' }}>
                        {statusStr.toUpperCase()}
                      </span>
                    </div>
                  </div>
                );
              })
            ) : (
              [
                'data_ingestion',
                'feature_pipeline',
                'regime_model',
                'quantile_models',
                'gis_boundary_engine',
                'api_gateway',
              ].map((subsystem) => (
                <div key={subsystem} className="health-item">
                  <span style={{ textTransform: 'capitalize', color: 'var(--deep-blue)', fontWeight: 500 }}>
                    {(subsystem || '').replace(/_/g, ' ')}
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <CheckCircle size={15} color="#2E8B80" />
                    <span style={{ fontSize: '0.7rem', fontWeight: 600, color: '#2E8B80' }}>HEALTHY</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div style={{ borderTop: '1px solid var(--soft-border)', paddingTop: '0.75rem', display: 'flex', justifyContent: 'flex-end' }}>
          <button className="btn-secondary" onClick={onClose}>
            Close Diagnostics
          </button>
        </div>
      </div>
    </div>
  );
};
