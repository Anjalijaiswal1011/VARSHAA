import React, { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { StoryBar, type StoryTab } from './components/StoryBar';
import { WhatRaapxPredictsHero } from './components/WhatRaapxPredictsHero';
import { WhyRaapxStoryFlow } from './components/WhyRaapxStoryFlow';
import { RiskAndUncertaintySection } from './components/RiskAndUncertaintySection';
import { VerificationBarChart } from './components/VerificationBarChart';
import { IndiaForecastMap } from './components/IndiaForecastMap';
import { DistrictDetailPanel } from './components/DistrictDetailPanel';
import { SystemHealthModal } from './components/SystemHealthModal';
import { ErrorBanner } from './components/ErrorBanner';
import {
  getDistrictForecasts,
  getSingleDistrictForecast,
  getDistrictGeoJSON,
  getSpatialRainfallGrid,
  getComprehensiveHealth,
} from './services/api';
import type {
  CanonicalDistrictForecastRecord,
  ComprehensiveHealthResponse,
  DistrictGeoJSONResponse,
  ForecastProductStatus,
  MetricType,
  ProblemDetails,
  SpatialRainfallGridResponse,
} from './types/forecast';

export const App: React.FC = () => {
  const [leadTime, setLeadTime] = useState<number>(24);
  const [activeMetric] = useState<MetricType>('p50');
  const [selectedDistrictId, setSelectedDistrictId] = useState<string>('MP_DHAR');
  const [storyTab, setStoryTab] = useState<StoryTab>('forecast');

  const [districts, setDistricts] = useState<CanonicalDistrictForecastRecord[]>([]);
  const [selectedDistrict, setSelectedDistrict] = useState<CanonicalDistrictForecastRecord | null>(null);
  const [geoJsonData, setGeoJsonData] = useState<DistrictGeoJSONResponse | null>(null);
  const [spatialGrid, setSpatialGrid] = useState<SpatialRainfallGridResponse | null>(null);
  const [healthData, setHealthData] = useState<ComprehensiveHealthResponse | null>(null);

  const [loadingDistricts, setLoadingDistricts] = useState<boolean>(true);
  const [loadingDetail, setLoadingDetail] = useState<boolean>(false);
  const [apiError, setApiError] = useState<ProblemDetails | null>(null);
  const [isHealthOpen, setIsHealthOpen] = useState<boolean>(false);

  const [forecastCycle, setForecastCycle] = useState<string>('2026-07-15');
  const [dataFreshnessStatus, setDataFreshnessStatus] = useState<ForecastProductStatus>('VALID');

  // 1. Initial Load and Lead Time Sync
  const loadForecastData = useCallback(async (lt: number) => {
    setLoadingDistricts(true);
    setApiError(null);

    try {
      // Parallel fetch of GeoJSON polygons, district forecasts, and high-resolution spatial rainfall grid
      const [geoRes, distRes, gridRes] = await Promise.all([
        getDistrictGeoJSON(lt),
        getDistrictForecasts({ lead_time: lt, limit: 100 }),
        getSpatialRainfallGrid(lt).catch((e) => {
          console.warn('Spatial grid fetch failed, fallback to GeoJSON:', e);
          return null;
        }),
      ]);

      setGeoJsonData(geoRes);
      setDistricts(distRes.records || []);
      if (gridRes) {
        setSpatialGrid(gridRes);
      }

      if (distRes.records && distRes.records.length > 0) {
        // Derive cycle date and status from actual records
        const firstRec = distRes.records[0];
        if (firstRec.initialization_time) {
          setForecastCycle(firstRec.initialization_time.split('T')[0]);
        }
        setDataFreshnessStatus(firstRec.prediction_status || 'VALID');
      }
    } catch (err: any) {
      console.error('Failed to load forecast layers:', err);
      setApiError(err as ProblemDetails);
    } finally {
      setLoadingDistricts(false);
    }
  }, []);

  // 2. Fetch Single District Detail when selection or lead time changes (One Forecast Context)
  const loadDistrictDetail = useCallback(async (dId: string, lt: number) => {
    if (!dId) return;
    setLoadingDetail(true);
    setApiError(null);
    try {
      const detail = await getSingleDistrictForecast(dId, lt);
      setSelectedDistrict(detail);
      if (detail.prediction_status) {
        setDataFreshnessStatus(detail.prediction_status);
      }
      
      // Development logging strictly verifying same forecast context
      console.log(
        `%c[RAAP-X FORECAST CONTEXT] District: ${detail.district_name} | ID: ${detail.district_id} | Forecast Date: ${detail.forecast_time} | Lead Time: T+${detail.lead_time}h | Run ID: ${detail.pipeline_run_id || 'operational_run_current'} | Model Version: ${detail.model_version}`,
        'color: #0284c7; font-weight: bold; background: #e0f2fe; padding: 3px 8px; border-radius: 4px;'
      );
    } catch (err: any) {
      console.error(`[RAAP-X API ERROR] Failed to load forecast for ${dId} (T+${lt}h):`, err);
      // Check if record exists in active districts list for that lead time
      const fallback = districts.find((d) => d.district_id.toUpperCase() === dId.toUpperCase());
      if (fallback) {
        setSelectedDistrict(fallback);
      } else {
        setSelectedDistrict(null);
        setApiError(err as ProblemDetails);
      }
    } finally {
      setLoadingDetail(false);
    }
  }, [districts]);

  // 3. Load System Health Probe
  const loadHealthProbe = useCallback(async () => {
    try {
      const hData = await getComprehensiveHealth();
      setHealthData(hData);
    } catch (err) {
      console.warn('System health probe failed (non-critical):', err);
    }
  }, []);

  // Effect: On Mount & Lead Time Change
  useEffect(() => {
    loadForecastData(leadTime);
    loadHealthProbe();
  }, [leadTime, loadForecastData, loadHealthProbe]);

  // Effect: Load Detail when selection or lead time changes
  useEffect(() => {
    if (selectedDistrictId) {
      loadDistrictDetail(selectedDistrictId, leadTime);
    }
  }, [selectedDistrictId, leadTime, loadDistrictDetail]);

  const handleSelectDistrict = (dId: string) => {
    setSelectedDistrictId(dId);
  };

  return (
    <div className="dashboard-app" style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* 1. App Header */}
      <Header
        forecastCycle={forecastCycle}
        dataFreshnessStatus={dataFreshnessStatus}
        modelVersion="v1.0.0-prob"
        boundaryVersion="IMD-LGD-2026.1"
        healthData={healthData}
        onOpenHealth={() => setIsHealthOpen(true)}
      />

      {/* 2. Story-Driven Navigation Bar: WHERE? Location + FORECAST DAY + Secondary Navigation Tabs */}
      <StoryBar
        districts={districts}
        selectedDistrict={selectedDistrict}
        selectedDistrictId={selectedDistrictId}
        onSelectDistrict={handleSelectDistrict}
        leadTime={leadTime}
        onSelectLeadTime={(lt) => setLeadTime(lt)}
        activeStoryTab={storyTab}
        onSelectStoryTab={(tab) => setStoryTab(tab)}
      />

      {/* 3. Operational Banners for STALE, PARTIAL, or API Errors */}
      <ErrorBanner
        status={dataFreshnessStatus}
        apiError={apiError}
        onDismiss={() => setApiError(null)}
        onRetry={() => loadForecastData(leadTime)}
      />

      {/* 4. Story Stage Content Container */}
      <div style={{ flex: 1, padding: '1rem 1.25rem', maxWidth: '1600px', width: '100%', margin: '0 auto' }}>
        {/* TAB 1: FORECAST (WHAT?) */}
        {storyTab === 'forecast' && (
          <div>
            {/* Primary Result — What Does RAAP-X Predict? */}
            <WhatRaapxPredictsHero
              district={selectedDistrict}
              leadTime={leadTime}
              onExploreWhy={() => setStoryTab('why')}
            />

            {/* Main Two-Column Operational Canvas */}
            <main className="dashboard-grid">
              {/* Left Column: Interactive GIS Map Canvas */}
              <IndiaForecastMap
                geoJsonData={geoJsonData}
                spatialGrid={spatialGrid}
                districts={districts}
                selectedDistrictId={selectedDistrictId}
                onSelectDistrict={handleSelectDistrict}
                activeMetric={activeMetric}
                loading={loadingDistricts}
              />

              {/* Right Column: Meteorological Decision Support & Intelligence */}
              <DistrictDetailPanel
                district={selectedDistrict}
                loading={loadingDetail}
                leadTime={leadTime}
                onSelectLeadTime={(lt) => setLeadTime(lt)}
                districts={districts}
                onSelectDistrict={handleSelectDistrict}
                onNavigateTab={(tab) => setStoryTab(tab)}
              />
            </main>
          </div>
        )}

        {/* TAB 2: WHY RAAP-X? (The Core Differentiator) */}
        {storyTab === 'why' && (
          <WhyRaapxStoryFlow district={selectedDistrict} leadTime={leadTime} />
        )}

        {/* TAB 3: RISK & UNCERTAINTY (Quantiles vs Probabilities) */}
        {storyTab === 'risk' && (
          <RiskAndUncertaintySection district={selectedDistrict} leadTime={leadTime} />
        )}

        {/* TAB 4: VALIDATION (Model Verification & Proof) */}
        {storyTab === 'validation' && (
          <div
            style={{
              background: '#FFFFFF',
              borderRadius: 'var(--radius-lg)',
              border: '1.5px solid #D0E5F5',
              boxShadow: 'var(--shadow-md)',
              padding: '1.25rem 1.5rem',
            }}
          >
            <VerificationBarChart />
          </div>
        )}
      </div>

      {/* 5. Subsystem Health Diagnostics Modal */}
      <SystemHealthModal
        isOpen={isHealthOpen}
        onClose={() => setIsHealthOpen(false)}
        healthData={healthData}
      />
    </div>
  );
};

export default App;
