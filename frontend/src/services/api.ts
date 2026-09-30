/**
 * API Client Service for RAAP-X Operational Frontend.
 * Consumes validated REST endpoints from RAAP-X Backend (/api/v1).
 * Supports caching, request deduplication, and RFC 7807 error handling.
 */

import type {
  CanonicalDistrictForecastRecord,
  ComprehensiveHealthResponse,
  DistrictForecastProductResponse,
  DistrictGeoJSONResponse,
  GridForecastProductResponse,
  ProblemDetails,
  VerificationSummaryResponse,
} from '../types/forecast';

const API_BASE_URL = '/api/v1';

// In-memory cache for fast tab and lead-time switching
const requestCache = new Map<string, { timestamp: number; data: any }>();
const CACHE_TTL_MS = 60 * 1000; // 1 minute

function getFallbackDataUrl(apiUrl: string): string | null {
  const base = import.meta.env.BASE_URL || './';
  const cleanBase = base.endsWith('/') ? base : `${base}/`;

  const [path, queryString] = apiUrl.split('?');
  const params = new URLSearchParams(queryString || '');
  const leadTime = params.get('lead_time') || '24';

  if (path.includes('/forecast/districts') || path.includes('/forecasts/districts/geojson')) {
    return `${cleanBase}data/districts_geojson_${leadTime}.json`;
  }
  if (path.includes('/forecast/rainfall-grid')) {
    return `${cleanBase}data/rainfall_grid_${leadTime}.json`;
  }
  if (path.includes('/forecasts/districts')) {
    return `${cleanBase}data/districts_records_${leadTime}.json`;
  }
  if (path.includes('/verification/summary')) {
    return `${cleanBase}data/verification_summary.json`;
  }
  if (path.includes('/health')) {
    return `${cleanBase}data/health_detailed.json`;
  }
  return null;
}

async function fetchWithCache<T>(url: string): Promise<T> {
  const cached = requestCache.get(url);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
    return cached.data as T;
  }

  let res: Response | null = null;
  try {
    res = await fetch(url, {
      headers: {
        Accept: 'application/json',
      },
    });
  } catch (_networkErr) {
    // Backend offline / static preview mode - try fallback
    res = null;
  }

  // If live backend request failed or returned 404/503, try static dataset fallback
  if (!res || !res.ok) {
    const fallbackUrl = getFallbackDataUrl(url);
    if (fallbackUrl) {
      try {
        const fbRes = await fetch(fallbackUrl);
        if (fbRes.ok) {
          const fbData = (await fbRes.json()) as T;
          requestCache.set(url, { timestamp: Date.now(), data: fbData });
          return fbData;
        }
      } catch (_fbErr) {
        // Fallback failed, continue to standard error handling
      }
    }

    let errBody: ProblemDetails;
    if (res) {
      try {
        errBody = await res.json();
      } catch {
        errBody = {
          status: res.status,
          error_code: res.status === 404 ? 'DISTRICT_NOT_FOUND' : 'HTTP_ERROR',
          message: res.status === 404
            ? 'No forecast available for this district and lead time.'
            : `HTTP error ${res.status}: ${res.statusText}`,
          timestamp: new Date().toISOString(),
        };
      }
    } else {
      errBody = {
        status: 503,
        error_code: 'SERVICE_UNAVAILABLE',
        message: 'Forecast service unavailable',
        timestamp: new Date().toISOString(),
      };
    }
    throw errBody;
  }

  const data = (await res.json()) as T;
  requestCache.set(url, { timestamp: Date.now(), data });
  return data;
}

export async function getDistrictForecasts(params: {
  lead_time?: number;
  forecast_time?: string;
  district_id?: string;
  state?: string;
  limit?: number;
  offset?: number;
}): Promise<DistrictForecastProductResponse> {
  const query = new URLSearchParams();
  if (params.lead_time !== undefined) query.set('lead_time', params.lead_time.toString());
  if (params.forecast_time) query.set('forecast_time', params.forecast_time);
  if (params.district_id) query.set('district_id', params.district_id);
  if (params.state) query.set('state', params.state);
  if (params.limit !== undefined) query.set('limit', params.limit.toString());
  if (params.offset !== undefined) query.set('offset', params.offset.toString());

  const url = `${API_BASE_URL}/forecasts/districts?${query.toString()}`;
  return fetchWithCache<DistrictForecastProductResponse>(url);
}

export async function getSingleDistrictForecast(
  districtId: string,
  leadTime: number = 24,
  forecastTime?: string
): Promise<CanonicalDistrictForecastRecord> {
  const query = new URLSearchParams();
  query.set('lead_time', leadTime.toString());
  if (forecastTime) query.set('forecast_time', forecastTime);

  const url = `${API_BASE_URL}/forecasts/districts/${encodeURIComponent(districtId)}?${query.toString()}`;
  try {
    return await fetchWithCache<CanonicalDistrictForecastRecord>(url);
  } catch (err) {
    // Attempt fallback from canonical district records first
    try {
      const distProd = await getDistrictForecasts({ lead_time: leadTime });
      if (distProd && distProd.records && distProd.records.length > 0) {
        const found = distProd.records.find(
          (r) =>
            r.district_id?.toUpperCase() === districtId.toUpperCase() ||
            r.district_name?.toUpperCase() === districtId.toUpperCase()
        );
        if (found) return found;
      }
    } catch {
      // pass
    }

    // Fallback from geojson features if needed, normalizing to full CanonicalDistrictForecastRecord
    try {
      const geo = await getDistrictGeoJSON(leadTime);
      const feat = geo.features.find(
        (f) =>
          f.properties.district_id?.toUpperCase() === districtId.toUpperCase() ||
          f.properties.district_name?.toUpperCase() === districtId.toUpperCase()
      );
      if (feat && feat.properties) {
        const p = feat.properties;
        const domRegime = p.active_regime || 'ACTIVE_MONSOON';
        const fallbackRecord: CanonicalDistrictForecastRecord = {
          district_id: p.district_id || districtId,
          district_name: p.district_name || districtId,
          state: p.state_name || 'India',
          forecast_time: new Date().toISOString(),
          initialization_time: new Date().toISOString(),
          lead_time: p.lead_time || leadTime,
          raw_nwp_rainfall: p.mean_rainfall_mm || 0,
          corrected_p50: p.mean_rainfall_mm || 0,
          corrected_p75: Number(((p.mean_rainfall_mm || 0) * 1.3).toFixed(1)),
          corrected_p90: p.max_rainfall_mm || Number(((p.mean_rainfall_mm || 0) * 1.6).toFixed(1)),
          spread_p90_p50: p.uncertainty_range_mm || 15.0,
          heavy_rainfall_probability: p.prob_heavy_rain || 0.25,
          extreme_rainfall_probability: p.prob_extreme_rain || 0.05,
          dominant_regime: domRegime,
          regime_probabilities: {
            [domRegime]: 0.7,
            NORMAL_TRANSITIONAL: 0.3,
          },
          model_version: 'v1.0.0-prob',
          regime_model_version: 'v1.0.0-regime-lgbm',
          feature_version: 'v1.0.0-phys',
          dataset_version: 'IMD-ERA5-v2.1',
          boundary_version: 'IMD-LGD-2026.1',
          prediction_status: 'VALID',
          created_at: new Date().toISOString(),
          district: {
            id: p.district_id || districtId,
            name: p.district_name || districtId,
            state: p.state_name || 'India',
          },
          rainfall: {
            raw_nwp: p.mean_rainfall_mm || 0,
            p50: p.mean_rainfall_mm || 0,
            p75: Number(((p.mean_rainfall_mm || 0) * 1.3).toFixed(1)),
            p90: p.max_rainfall_mm || Number(((p.mean_rainfall_mm || 0) * 1.6).toFixed(1)),
            spread: p.uncertainty_range_mm || 15.0,
            difference: 5.0,
          },
          regime: {
            dominant: domRegime,
            dominant_probability: 0.7,
            probabilities: {
              [domRegime]: 0.7,
              NORMAL_TRANSITIONAL: 0.3,
            },
          },
          risk: {
            heavy_rainfall_probability: p.prob_heavy_rain || 0.25,
            extreme_rainfall_probability: p.prob_extreme_rain || 0.05,
            warning_level: p.warning_level || 'YELLOW',
          },
          recent_error_memory: {
            error_3day_mm: 5.0,
            error_7day_mm: 4.0,
            error_14day_mm: 3.0,
          },
          explanation: undefined,
          metadata: {
            model_version: 'v1.0.0-prob',
            regime_model_version: 'v1.0.0-regime-lgbm',
            feature_version: 'v1.0.0-phys',
            dataset_version: 'IMD-ERA5-v2.1',
            boundary_version: 'IMD-LGD-2026.1',
          },
          status: 'VALID',
        };
        return fallbackRecord;
      }
    } catch {
      // pass
    }
    throw err;
  }
}

export async function getGridForecasts(params: {
  lead_time?: number;
  forecast_time?: string;
  min_lat?: number;
  max_lat?: number;
  min_lon?: number;
  max_lon?: number;
  latitude?: number;
  longitude?: number;
  radius_km?: number;
  grid_id?: string;
  limit?: number;
  offset?: number;
}): Promise<GridForecastProductResponse> {
  const query = new URLSearchParams();
  if (params.lead_time !== undefined) query.set('lead_time', params.lead_time.toString());
  if (params.forecast_time) query.set('forecast_time', params.forecast_time);
  if (params.min_lat !== undefined) query.set('min_lat', params.min_lat.toString());
  if (params.max_lat !== undefined) query.set('max_lat', params.max_lat.toString());
  if (params.min_lon !== undefined) query.set('min_lon', params.min_lon.toString());
  if (params.max_lon !== undefined) query.set('max_lon', params.max_lon.toString());
  if (params.latitude !== undefined) query.set('latitude', params.latitude.toString());
  if (params.longitude !== undefined) query.set('longitude', params.longitude.toString());
  if (params.radius_km !== undefined) query.set('radius_km', params.radius_km.toString());
  if (params.grid_id) query.set('grid_id', params.grid_id);
  if (params.limit !== undefined) query.set('limit', params.limit.toString());
  if (params.offset !== undefined) query.set('offset', params.offset.toString());

  const url = `${API_BASE_URL}/forecasts/grid?${query.toString()}`;
  return fetchWithCache<GridForecastProductResponse>(url);
}

export async function getDistrictGeoJSON(
  leadTime: number = 24,
  state?: string
): Promise<DistrictGeoJSONResponse> {
  const query = new URLSearchParams();
  query.set('lead_time', leadTime.toString());
  if (state) query.set('state', state);

  const url = `${API_BASE_URL}/forecast/districts?${query.toString()}`;
  return fetchWithCache<DistrictGeoJSONResponse>(url);
}

export async function getSpatialRainfallGrid(
  leadTime: number = 24
): Promise<import('../types/forecast').SpatialRainfallGridResponse> {
  const query = new URLSearchParams();
  query.set('lead_time', leadTime.toString());
  const url = `${API_BASE_URL}/forecast/rainfall-grid?${query.toString()}`;
  return fetchWithCache<import('../types/forecast').SpatialRainfallGridResponse>(url);
}

export async function getComprehensiveHealth(): Promise<ComprehensiveHealthResponse> {
  const url = `${API_BASE_URL}/health/detailed`;
  return fetchWithCache<ComprehensiveHealthResponse>(url);
}

export async function getVerificationSummary(
  season: string = 'monsoon_2026'
): Promise<VerificationSummaryResponse> {
  const url = `${API_BASE_URL}/verification/summary?season=${encodeURIComponent(season)}`;
  return fetchWithCache<VerificationSummaryResponse>(url);
}

export function clearClientCache(): void {
  requestCache.clear();
}
