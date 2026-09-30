import { CloudRain, Activity, Database, Layers, Clock } from 'lucide-react';
import type { ComprehensiveHealthResponse, ForecastProductStatus } from '../types/forecast';


interface HeaderProps {
  forecastCycle: string;
  dataFreshnessStatus: ForecastProductStatus;
  modelVersion: string;
  boundaryVersion: string;
  healthData?: ComprehensiveHealthResponse | null;
  onOpenHealth: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  forecastCycle,
  dataFreshnessStatus,
  modelVersion,
  boundaryVersion,
  healthData,
  onOpenHealth,
}) => {
  const isHealthy = healthData ? healthData.overall_status === 'healthy' || healthData.status === 'healthy' : true;

  return (
    <header className="app-header">
      <div className="brand-section">
        <div className="brand-icon">
          <CloudRain size={24} color="#ffffff" />
        </div>
        <div>
          <div className="brand-title">RAAP-X</div>
          <div className="brand-subtitle">Regime-Aware AI Rainfall Forecasting</div>
        </div>
      </div>

      <div className="header-meta-group">
        <div className="meta-pill">
          <Clock size={13} />
          <span>Cycle:</span>
          <strong>{forecastCycle}</strong>
        </div>

        <div className={`status-badge ${dataFreshnessStatus.toLowerCase()}`}>
          <span className="status-dot"></span>
          <span>{dataFreshnessStatus}</span>
        </div>

        <div className="meta-pill">
          <Database size={13} />
          <span>Model:</span>
          <strong>{modelVersion}</strong>
        </div>

        <div className="meta-pill">
          <Layers size={13} />
          <span>GIS:</span>
          <strong>{boundaryVersion}</strong>
        </div>

        <button
          className={`status-badge ${isHealthy ? 'healthy' : 'degraded'}`}
          onClick={onOpenHealth}
          title="Click to view detailed operational subsystem diagnostics"
          style={{ cursor: 'pointer', border: 'none' }}
        >
          <Activity size={12} />
          <span>{isHealthy ? 'SYSTEM OPERATIONAL' : 'SYSTEM DEGRADED'}</span>
        </button>
      </div>
    </header>
  );
};
