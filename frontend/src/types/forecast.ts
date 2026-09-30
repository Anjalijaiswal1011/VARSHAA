/**
 * TypeScript Type Definitions for RAAP-X Frontend Dashboard (Phase 9).
 * Strictly mirrors backend schemas from backend/schemas.py (CONTRACT-API-003 & Phase 8).
 */

export type ForecastProductStatus = 'VALID' | 'PARTIAL' | 'STALE' | 'INVALID' | 'UNAVAILABLE';

export type MetricType = 'p50' | 'p75' | 'p90' | 'raw_nwp' | 'heavy_prob' | 'extreme_prob';

export interface ProblemDetails {
  status: number;
  error_code: string;
  message: string;
  timestamp: string;
}

export interface SpatialGridPoint {
  lat: number;
  lon: number;
  raw_nwp: number;
  p50: number;
  p75: number;
  p90: number;
  spread: number;
  prob_heavy: number;
  prob_extreme: number;
  regime: string;
  warning_level: string;
}

export interface SpatialRainfallGridResponse {
  cycle_date: string;
  forecast_time: string;
  lead_time: number;
  grid_resolution_deg: number;
  total_points: number;
  points: SpatialGridPoint[];
}

export interface DistrictIdentity {
  id: string;
  name: string;
  state?: string;
}

export interface RainfallQuantiles {
  raw_nwp: number;
  p50: number;
  p75: number;
  p90: number;
  spread?: number;
  difference?: number;
}

export interface RegimeState {
  dominant: string;
  dominant_probability: number;
  probabilities: Record<string, number>;
}

export interface RiskAssessment {
  heavy_rainfall_probability: number;
  extreme_rainfall_probability: number;
  warning_level?: 'GREEN' | 'YELLOW' | 'ORANGE' | 'RED' | string;
}

export interface ForecastMetadata {
  model_version: string;
  regime_model_version: string;
  feature_version: string;
  dataset_version: string;
  boundary_version: string;
  pipeline_run_id?: string | null;
}

export interface ExplanationDriver {
  feature: string;
  contribution_mm: number;
  category: string;
  description: string;
}

export interface DistrictExplanationMetadata {
  dominant_regime: string;
  regime_confidence: number;
  forecast_spread: number;
  top_features: ExplanationDriver[];
  feature_contributions: Record<string, number>;
  recent_error_memory?: {
    error_3day_mm: number;
    error_7day_mm: number;
    error_14day_mm: number;
  };
  user_friendly_summary: string;
  model_version: string;
  explanation_version: string;
}

export interface PaginationMeta {
  total_count: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

export interface CanonicalDistrictForecastRecord {
  district_id: string;
  district_name: string;
  state?: string;
  forecast_time: string;
  initialization_time: string;
  lead_time: number;

  raw_nwp_rainfall: number;
  corrected_p50: number;
  corrected_p75: number;
  corrected_p90: number;
  spread_p90_p50: number;

  heavy_rainfall_probability: number;
  extreme_rainfall_probability: number;

  dominant_regime: string;
  regime_probabilities: Record<string, number>;

  model_version: string;
  regime_model_version: string;
  feature_version: string;
  dataset_version: string;
  boundary_version: string;
  pipeline_run_id?: string | null;

  prediction_status: ForecastProductStatus;
  created_at: string;
  status?: string;

  // Nested structures
  district?: DistrictIdentity;
  rainfall?: RainfallQuantiles;
  regime?: RegimeState;
  risk?: RiskAssessment;
  recent_error_memory?: {
    error_3day_mm: number;
    error_7day_mm: number;
    error_14day_mm: number;
  };
  explanation?: DistrictExplanationMetadata;
  metadata?: ForecastMetadata;
}

export interface DistrictForecastProductResponse {
  pagination: PaginationMeta;
  forecast_time: string;
  lead_time: number;
  boundary_version: string;
  model_version: string;
  records: CanonicalDistrictForecastRecord[];
}

export interface CanonicalGridForecastRecord {
  prediction_id: string;
  forecast_time: string;
  initialization_time: string;
  lead_time: number;
  latitude: number;
  longitude: number;
  grid_id?: string;
  district_id?: string;
  district_name?: string;

  raw_nwp_rainfall: number;
  corrected_p50: number;
  corrected_p75: number;
  corrected_p90: number;
  spread_p90_p50: number;

  regime_probabilities: Record<string, number>;
  dominant_regime: string;
  dominant_probability: number;

  heavy_rainfall_probability: number;
  extreme_rainfall_probability: number;

  model_version: string;
  regime_model_version: string;
  feature_version: string;
  dataset_version: string;
  boundary_version: string;
  pipeline_run_id?: string | null;

  prediction_status: ForecastProductStatus;
  created_at: string;

  rainfall?: RainfallQuantiles;
  regime?: RegimeState;
  risk?: RiskAssessment;
  metadata?: ForecastMetadata;
}

export interface GridForecastProductResponse {
  pagination: PaginationMeta;
  forecast_time: string;
  lead_time: number;
  boundary_version: string;
  model_version: string;
  records: CanonicalGridForecastRecord[];
}

export interface GeoJSONFeature {
  type: 'Feature';
  id: string;
  properties: {
    district_id: string;
    district_name: string;
    state_name: string;
    lead_time: number;
    mean_rainfall_mm: number;
    max_rainfall_mm: number;
    p90_rainfall_mm?: number;
    prob_heavy_rain: number;
    prob_very_heavy_rain: number;
    prob_extreme_rain?: number;
    warning_level: 'GREEN' | 'YELLOW' | 'ORANGE' | 'RED';
    color_code?: string;
    action_required?: string;
    active_regime: string;
    uncertainty_range_mm?: number;
    forecast_spread?: number;
    boundary_version?: string;
    status?: string;
  };
  geometry: {
    type: 'Polygon' | 'MultiPolygon';
    coordinates: any;
  };
}

export interface DistrictGeoJSONResponse {
  type: 'FeatureCollection';
  features: GeoJSONFeature[];
}

export interface SubsystemHealth {
  status: 'healthy' | 'degraded' | 'down' | 'operational' | string;
  detail?: string;
  last_check?: string;
}

export interface ComprehensiveHealthResponse {
  status: string;
  overall_status: string;
  version: string;
  timestamp: string;
  checks_total: number;
  checks_passed: number;
  checks_failed: number;
  components: Record<string, SubsystemHealth | any>;
}

export interface VerificationMetrics {
  raw_nwp_rmse: number;
  corrected_rmse: number;
  rmse_improvement_pct: number;
  raw_nwp_mae: number;
  corrected_mae: number;
  heavy_rain_csi_raw: number;
  heavy_rain_csi_corrected: number;
  crps_raw: number;
  crps_corrected: number;
}

export interface VerificationSummaryResponse {
  evaluation_period: string;
  metrics: VerificationMetrics;
}
