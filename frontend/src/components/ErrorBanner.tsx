import { AlertTriangle, AlertCircle, Info, RefreshCw, X } from 'lucide-react';
import type { ForecastProductStatus, ProblemDetails } from '../types/forecast';

interface ErrorBannerProps {
  status?: ForecastProductStatus;
  apiError?: ProblemDetails | null;
  onDismiss?: () => void;
  onRetry?: () => void;
}

export const ErrorBanner: React.FC<ErrorBannerProps> = ({
  status,
  apiError,
  onDismiss,
  onRetry,
}) => {
  if (apiError) {
    return (
      <div className="alert-banner error">
        <AlertCircle size={18} color="#A34848" />
        <div style={{ flex: 1 }}>
          <strong>Operational Error ({apiError.error_code || 'API_FAILURE'}):</strong> {apiError.message}
        </div>
        <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
          {onRetry && (
            <button
              onClick={onRetry}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.3rem',
                padding: '0.25rem 0.55rem',
                borderRadius: '4px',
                border: '1px solid #E9BDBD',
                background: '#FFFFFF',
                color: '#A34848',
                fontSize: '0.75rem',
                cursor: 'pointer',
              }}
            >
              <RefreshCw size={12} />
              Retry
            </button>
          )}
          {onDismiss && (
            <button
              onClick={onDismiss}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#A34848',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
              }}
              title="Dismiss"
            >
              <X size={16} />
            </button>
          )}
        </div>
      </div>
    );
  }

  if (status === 'STALE') {
    return (
      <div className="alert-banner stale">
        <AlertTriangle size={18} color="#4A687A" />
        <div style={{ flex: 1 }}>
          <strong>Notice — Stale Cycle Data:</strong> The displayed forecast is from an earlier forecast cycle. Newer model run synchronization in progress.
        </div>
      </div>
    );
  }

  if (status === 'PARTIAL') {
    return (
      <div className="alert-banner partial">
        <Info size={18} color="#9C6C1B" />
        <div style={{ flex: 1 }}>
          <strong>Partial Domain Coverage:</strong> Some peripheral grid cells or sub-districts have missing observations. Published values passed strict partial verification.
        </div>
      </div>
    );
  }

  return null;
};
