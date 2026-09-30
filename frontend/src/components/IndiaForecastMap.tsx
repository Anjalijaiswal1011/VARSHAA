import React, { useEffect, useRef, useMemo, useCallback, useState } from 'react';
import L from 'leaflet';
import { RotateCcw, Sliders } from 'lucide-react';
import type {
  DistrictGeoJSONResponse,
  GeoJSONFeature,
  MetricType,
  CanonicalDistrictForecastRecord,
  SpatialRainfallGridResponse,
  SpatialGridPoint,
} from '../types/forecast';
import { MapLegend } from './MapLegend';
import { DistrictSearch } from './DistrictSearch';

interface IndiaForecastMapProps {
  geoJsonData: DistrictGeoJSONResponse | null;
  spatialGrid?: SpatialRainfallGridResponse | null;
  districts: CanonicalDistrictForecastRecord[];
  selectedDistrictId?: string;
  onSelectDistrict: (districtId: string) => void;
  activeMetric: MetricType;
  loading: boolean;
}

export const IndiaForecastMap: React.FC<IndiaForecastMapProps> = ({
  geoJsonData,
  spatialGrid,
  districts,
  selectedDistrictId,
  onSelectDistrict,
  activeMetric,
  loading,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const geoJsonLayerRef = useRef<L.GeoJSON | null>(null);

  // Dual View / Split Comparison Slider states (0% = 100% Raw NWP, 100% = 100% RAAP-X)
  const [splitSliderActive, setSplitSliderActive] = useState<boolean>(false);
  const [splitPosition, setSplitPosition] = useState<number>(50);

  // Visualization display settings
  const renderMode = 'smooth';
  const showDistricts = true;
  const showRainfallLayer = true;
  const layerOpacity = 0.8;

  // Color mapping functions
  const getRainfallRgba = useCallback((val: number, opacity: number): string => {
    if (val <= 25.0) return `rgba(248, 252, 255, 0)`; // Baseline / dry background is transparent
    if (val < 35.0) return `rgba(185, 223, 247, ${0.45 * opacity})`; // Soft sky blue (light-moderate)
    if (val < 55.0) return `rgba(142, 205, 243, ${0.75 * opacity})`; // Medium sky blue (moderate)
    if (val < 85.0) return `rgba(81, 152, 197, ${0.88 * opacity})`; // Deeper blue (heavy)
    return `rgba(130, 133, 214, ${0.95 * opacity})`; // Subtle violet (extreme)
  }, []);

  const getProbabilityRgba = useCallback((prob: number, opacity: number): string => {
    if (prob < 0.15) return `rgba(234, 246, 253, 0)`; // Minimal risk is transparent
    if (prob < 0.35) return `rgba(234, 246, 253, ${0.6 * opacity})`;
    if (prob < 0.55) return `rgba(191, 234, 242, ${0.75 * opacity})`;
    if (prob < 0.75) return `rgba(142, 205, 243, ${0.85 * opacity})`;
    return `rgba(107, 108, 168, ${0.92 * opacity})`;
  }, []);

  const getMetricValue = useCallback((p: SpatialGridPoint, metric: MetricType): number => {
    switch (metric) {
      case 'p50':
        return p.p50;
      case 'p75':
        return p.p75;
      case 'p90':
        return p.p90;
      case 'raw_nwp':
        return p.raw_nwp;
      case 'heavy_prob':
        return p.prob_heavy;
      case 'extreme_prob':
        return p.prob_extreme;
      default:
        return p.p50;
    }
  }, []);

  const getColorForPoint = useCallback((p: SpatialGridPoint, metric: MetricType, opacity: number): string => {
    const val = getMetricValue(p, metric);
    if (metric === 'heavy_prob' || metric === 'extreme_prob') {
      return getProbabilityRgba(val, opacity);
    }
    return getRainfallRgba(val, opacity);
  }, [getMetricValue, getProbabilityRgba, getRainfallRgba]);

  // Index districts by ID for instant lookup
  const districtMap = useMemo(() => {
    const m = new Map<string, CanonicalDistrictForecastRecord>();
    for (const d of districts) {
      m.set(d.district_id.toUpperCase(), d);
    }
    return m;
  }, [districts]);

  // Supported Indian forecast domain bounds (from src/gis/grid_schema.py: MIN_LAT: 6.0, MAX_LAT: 38.5, MIN_LON: 68.0, MAX_LON: 98.0)
  const INDIA_BOUNDS = useMemo<L.LatLngBoundsLiteral>(
    () => [
      [8.0, 68.5],  // Southwest: Kanyakumari / Lakshadweep
      [36.5, 96.0], // Northeast: Ladakh / Arunachal Pradesh
    ],
    []
  );

  const INDIA_MAX_BOUNDS = useMemo<L.LatLngBoundsLiteral>(
    () => [
      [5.0, 60.0],
      [39.0, 102.0],
    ],
    []
  );

  // 1. Initialize Leaflet map with clean light geographic basemap focused on India
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: [22.0, 80.5], // Geographic centroid of Indian forecast domain
      zoom: 5,
      minZoom: 4,
      maxZoom: 10,
      maxBounds: INDIA_MAX_BOUNDS,
      maxBoundsViscosity: 0.9,
      zoomControl: false,
      attributionControl: false,
    });

    L.control.zoom({ position: 'bottomright' }).addTo(map);

    const cartoApiKey = import.meta.env.VITE_CARTO_API_KEY?.trim();
    if (cartoApiKey) {
      L.tileLayer(
        `https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png?api_key=${encodeURIComponent(cartoApiKey)}`,
        {
          maxZoom: 19,
          subdomains: 'abcd',
          className: 'soft-sky-basemap-tiles',
        }
      ).addTo(map);
    } else {
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        subdomains: 'abc',
        className: 'soft-sky-basemap-tiles',
      }).addTo(map);
    }

    // Explicitly fit bounds so India occupies the entire map viewport
    map.fitBounds(INDIA_BOUNDS, {
      padding: [15, 15],
      maxZoom: 6,
    });

    // Create Canvas Layer in overlay pane for continuous rainfall field
    const overlayPane = map.getPanes().overlayPane;
    const canvas = document.createElement('canvas');
    canvas.style.position = 'absolute';
    canvas.style.top = '0px';
    canvas.style.left = '0px';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '350';
    overlayPane.appendChild(canvas);
    canvasRef.current = canvas;

    mapInstanceRef.current = map;

    // Invalidate size and re-fit bounds once DOM layout stabilizes
    const resizeTimer = setTimeout(() => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.invalidateSize();
        mapInstanceRef.current.fitBounds(INDIA_BOUNDS, { padding: [15, 15], maxZoom: 6 });
      }
    }, 150);

    const handleResize = () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.invalidateSize();
      }
    };
    window.addEventListener('resize', handleResize);

    let ro: ResizeObserver | null = null;
    if (mapContainerRef.current && typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(() => {
        if (mapInstanceRef.current) {
          mapInstanceRef.current.invalidateSize();
        }
      });
      ro.observe(mapContainerRef.current);
    }

    return () => {
      clearTimeout(resizeTimer);
      window.removeEventListener('resize', handleResize);
      if (ro) ro.disconnect();
      if (canvasRef.current && canvasRef.current.parentNode) {
        canvasRef.current.parentNode.removeChild(canvasRef.current);
      }
      map.remove();
      mapInstanceRef.current = null;
    };
  }, [INDIA_BOUNDS, INDIA_MAX_BOUNDS]);

  // 2. Draw Continuous Spatial Rainfall Layer on HTML5 Canvas
  const redrawCanvas = useCallback(() => {
    const map = mapInstanceRef.current;
    const canvas = canvasRef.current;
    if (!map || !canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (!showRainfallLayer || !spatialGrid || !spatialGrid.points || spatialGrid.points.length === 0) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const size = map.getSize();
    const bounds = map.getBounds();
    const topLeft = map.latLngToLayerPoint(bounds.getNorthWest());

    // Sync canvas resolution and coordinate space to map viewport
    if (canvas.width !== size.x || canvas.height !== size.y) {
      canvas.width = size.x;
      canvas.height = size.y;
    }

    L.DomUtil.setPosition(canvas, topLeft);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const points = spatialGrid.points;
    const resDeg = spatialGrid.grid_resolution_deg || 0.5;

    // Calculate dynamic pixel dimensions for 0.5° grid cells at current zoom
    const centerLatLng = map.getCenter();
    const ptCenter = map.latLngToContainerPoint(centerLatLng);
    const ptOffset = map.latLngToContainerPoint([centerLatLng.lat + resDeg, centerLatLng.lng + resDeg]);
    const cellW = Math.max(10, Math.abs(ptOffset.x - ptCenter.x));
    const cellH = Math.max(10, Math.abs(ptOffset.y - ptCenter.y));
    const radius = Math.max(cellW, cellH) * 1.5;

    const isProbability = activeMetric === 'heavy_prob' || activeMetric === 'extreme_prob';
    const splitX = (splitPosition / 100) * canvas.width;

    if (renderMode === 'smooth') {
      // Smooth Continuous Spatial Field:
      // Overlapping radial gradients create a contiguous, organic rainfall field across India
      for (let i = 0; i < points.length; i++) {
        const p = points[i];
        if (
          p.lat < bounds.getSouth() - resDeg ||
          p.lat > bounds.getNorth() + resDeg ||
          p.lon < bounds.getWest() - resDeg ||
          p.lon > bounds.getEast() + resDeg
        ) {
          continue;
        }

        const cp = map.latLngToContainerPoint([p.lat, p.lon]);
        
        // If split slider active, left of divider renders Raw NWP, right renders RAAP-X AI metric
        const effectiveMetric: MetricType = splitSliderActive
          ? (cp.x < splitX ? 'raw_nwp' : activeMetric)
          : activeMetric;

        const val = getMetricValue(p, effectiveMetric);
        // Skip dry / baseline background points so active rainfall regions stand out
        if ((!isProbability && val <= 25.0) || (isProbability && val < 0.20)) {
          continue;
        }

        const grad = ctx.createRadialGradient(cp.x, cp.y, 0, cp.x, cp.y, radius);

        const baseColor = getColorForPoint(p, effectiveMetric, layerOpacity * 0.45);
        grad.addColorStop(0, baseColor);
        grad.addColorStop(0.5, baseColor.replace(/[\d.]+\)$/, `${0.25 * layerOpacity})`));
        grad.addColorStop(1, 'rgba(255, 255, 255, 0)');

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cp.x, cp.y, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    } else {
      // Contiguous Meteorological Mesh:
      // Renders contiguous adjacent grid cells without gaps
      for (let i = 0; i < points.length; i++) {
        const p = points[i];
        if (
          p.lat < bounds.getSouth() - resDeg ||
          p.lat > bounds.getNorth() + resDeg ||
          p.lon < bounds.getWest() - resDeg ||
          p.lon > bounds.getEast() + resDeg
        ) {
          continue;
        }

        const cp = map.latLngToContainerPoint([p.lat, p.lon]);
        const effectiveMetric: MetricType = splitSliderActive
          ? (cp.x < splitX ? 'raw_nwp' : activeMetric)
          : activeMetric;

        const val = getMetricValue(p, effectiveMetric);
        if ((!isProbability && val <= 25.0) || (isProbability && val < 0.20)) {
          continue;
        }

        ctx.fillStyle = getColorForPoint(p, effectiveMetric, layerOpacity * 0.55);
        ctx.fillRect(cp.x - cellW / 2, cp.y - cellH / 2, cellW + 0.5, cellH + 0.5);

        // Subtle cell grid divider
        ctx.strokeStyle = `rgba(40, 90, 122, ${0.1 * layerOpacity})`;
        ctx.lineWidth = 0.5;
        ctx.strokeRect(cp.x - cellW / 2, cp.y - cellH / 2, cellW + 0.5, cellH + 0.5);
      }
    }
  }, [spatialGrid, activeMetric, renderMode, showRainfallLayer, layerOpacity, getMetricValue, getColorForPoint, splitSliderActive, splitPosition]);

  // Hook map events to canvas redraw
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    redrawCanvas();

    map.on('move', redrawCanvas);
    map.on('moveend', redrawCanvas);
    map.on('zoom', redrawCanvas);
    map.on('zoomend', redrawCanvas);
    map.on('resize', redrawCanvas);

    return () => {
      map.off('move', redrawCanvas);
      map.off('moveend', redrawCanvas);
      map.off('zoom', redrawCanvas);
      map.off('zoomend', redrawCanvas);
      map.off('resize', redrawCanvas);
    };
  }, [redrawCanvas]);

  // 4. District Administrative Boundaries Layer
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !geoJsonData || !geoJsonData.features) return;

    if (geoJsonLayerRef.current) {
      map.removeLayer(geoJsonLayerRef.current);
      geoJsonLayerRef.current = null;
    }

    if (!showDistricts) return;

    const geoJsonLayer = L.geoJSON(geoJsonData as any, {
      style: (feature) => {
        if (!feature || !feature.properties) return {};
        const dId = (feature.properties.district_id || feature.id || '').toUpperCase();
        const isSelected = selectedDistrictId && dId === selectedDistrictId.toUpperCase();

        return {
          fillColor: isSelected ? 'rgba(142, 205, 243, 0.55)' : 'rgba(255, 255, 255, 0.05)',
          weight: isSelected ? 3.5 : 1.2,
          opacity: 1,
          color: isSelected ? '#0284C7' : '#285A7A', // Bold sky-blue boundary for selected district
          dashArray: isSelected ? undefined : '2, 3',
          fillOpacity: isSelected ? 0.55 : 0.05,
        };
      },
      onEachFeature: (feature: GeoJSONFeature, layer) => {
        const props = feature.properties;
        const dId = props.district_id || feature.id;
        const dRecord = districtMap.get(dId.toUpperCase());

        const dName = props.district_name || dRecord?.district_name || dId;
        const dState = dRecord?.state || props.state_name || 'India';
        const rawNwp = dRecord ? dRecord.raw_nwp_rainfall.toFixed(1) : ((props as any).raw_nwp || '—');
        const p50 = dRecord ? dRecord.corrected_p50.toFixed(1) : (props.mean_rainfall_mm || 0).toFixed(1);
        const p90 = dRecord ? dRecord.corrected_p90.toFixed(1) : (props.p90_rainfall_mm || 0).toFixed(1);
        const heavyProb = dRecord
          ? (dRecord.heavy_rainfall_probability * 100).toFixed(0)
          : ((props.prob_heavy_rain || 0) * 100).toFixed(0);
        const regime = dRecord ? dRecord.dominant_regime : props.active_regime || 'NORMAL_TRANSITIONAL';

        layer.bindTooltip(
          `<div style="font-family: Outfit, sans-serif; min-width: 175px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 2px;">
              <strong style="color: #1B4965; font-size: 0.85rem;">📍 ${dName} District</strong>
              <span style="font-size: 0.65rem; color: #526775; font-weight: 600;">${dId}</span>
            </div>
            <div style="font-size: 0.7rem; color: #526775; margin-bottom: 4px;">
              ${dState}
            </div>
            <div style="font-size: 0.74rem; color: #1B4965; background: #EAF4FB; padding: 4px 6px; border-radius: 4px; margin-bottom: 4px; display: flex; justify-content: space-between;">
              <span>Raw NWP: <strong>${rawNwp} mm</strong></span>
              <span>RAAP-X P50: <strong>${p50} mm</strong></span>
            </div>
            <div style="font-size: 0.7rem; color: #526775; display: flex; justify-content: space-between;">
              <span>P90 Quantile: <strong>${p90} mm</strong></span>
              <span>Heavy Risk: <strong>${heavyProb}%</strong></span>
            </div>
            <div style="font-size: 0.67rem; color: #7E93A2; margin-top: 3px; border-top: 1px dashed #D0E5F5; padding-top: 2px;">
              ${regime.replace('_', ' ')}
            </div>
          </div>`,
          { sticky: true, className: 'leaflet-tooltip-sky' }
        );

        layer.on({
          click: () => {
            onSelectDistrict(dId);
          },
          mouseover: (e) => {
            const target = e.target;
            target.setStyle({
              weight: 2.5,
              color: '#1B4965',
              fillOpacity: 0.35,
            });
          },
          mouseout: (e) => {
            geoJsonLayer.resetStyle(e.target);
          },
        });
      },
    }).addTo(map);

    geoJsonLayerRef.current = geoJsonLayer;
  }, [geoJsonData, districtMap, selectedDistrictId, showDistricts, onSelectDistrict]);

  // Pan to selected district smoothly while maintaining geographic context
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !selectedDistrictId || !geoJsonLayerRef.current) return;

    geoJsonLayerRef.current.eachLayer((layer: any) => {
      const feat = layer.feature;
      const dId = (feat?.properties?.district_id || feat?.id || '').toUpperCase();
      if (dId === selectedDistrictId.toUpperCase() && layer.getBounds) {
        const bounds = layer.getBounds();
        if (bounds && bounds.isValid()) {
          map.panTo(bounds.getCenter(), { animate: true, duration: 0.5 });
        }
      }
    });
  }, [selectedDistrictId]);

  const handleResetDomainView = useCallback(() => {
    const map = mapInstanceRef.current;
    if (!map) return;
    map.fitBounds(INDIA_BOUNDS, { padding: [15, 15], maxZoom: 6 });
  }, [INDIA_BOUNDS]);

  return (
    <div className="map-canvas-panel" style={{ position: 'relative' }}>
      {/* Search Bar Overlay - Top Left */}
      <DistrictSearch
        districts={districts}
        onSelectDistrict={onSelectDistrict}
        selectedDistrictId={selectedDistrictId}
      />

      {/* Map Reset Control - Top Right */}
      <div
        style={{
          position: 'absolute',
          top: 14,
          right: 14,
          zIndex: 850,
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
        }}
      >
        <button
          onClick={() => setSplitSliderActive(!splitSliderActive)}
          title="Toggle split-screen slider between Raw NWP and AI-Corrected forecast"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.35rem',
            background: splitSliderActive ? 'var(--deep-blue)' : 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(8px)',
            color: splitSliderActive ? '#FFFFFF' : 'var(--deep-blue)',
            border: splitSliderActive ? '1.5px solid var(--sky-blue)' : '1px solid var(--soft-border)',
            borderRadius: '9999px',
            padding: '0.3rem 0.75rem',
            cursor: 'pointer',
            fontWeight: 700,
            fontSize: '0.72rem',
            boxShadow: 'var(--shadow-sm)',
            transition: 'all 0.15s ease',
          }}
        >
          <Sliders size={12} color={splitSliderActive ? '#FFFFFF' : 'var(--sky-blue)'} />
          <span>{splitSliderActive ? 'Exit Split Slider' : 'Dual View Slider'}</span>
        </button>

        <button
          onClick={handleResetDomainView}
          title="Reset map view to full Indian forecast domain"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.3rem',
            background: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(8px)',
            color: 'var(--deep-blue)',
            border: '1px solid var(--soft-border)',
            borderRadius: '9999px',
            padding: '0.3rem 0.75rem',
            cursor: 'pointer',
            fontWeight: 700,
            fontSize: '0.72rem',
            boxShadow: 'var(--shadow-sm)',
            transition: 'all 0.15s ease',
          }}
        >
          <RotateCcw size={12} />
          <span>Reset India View</span>
        </button>
      </div>

      {/* Map Container */}
      <div ref={mapContainerRef} className="map-container" />

      {/* Interactive Map Split Slider Overlay */}
      {splitSliderActive && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            pointerEvents: 'none',
            zIndex: 450,
          }}
        >
          {/* Left Side Label (Raw NWP) */}
          <div
            style={{
              position: 'absolute',
              top: 54,
              left: 16,
              background: 'rgba(27, 73, 101, 0.92)',
              color: '#FFFFFF',
              padding: '0.35rem 0.75rem',
              borderRadius: '6px',
              fontSize: '0.74rem',
              fontWeight: 800,
              boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
              letterSpacing: '0.03em',
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
            }}
          >
            <span>◀ RAW NWP (UNCORRECTED)</span>
          </div>

          {/* Right Side Label (RAAP-X Corrected) */}
          <div
            style={{
              position: 'absolute',
              top: 54,
              right: 16,
              background: 'rgba(46, 139, 128, 0.95)',
              color: '#FFFFFF',
              padding: '0.35rem 0.75rem',
              borderRadius: '6px',
              fontSize: '0.74rem',
              fontWeight: 800,
              boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
              letterSpacing: '0.03em',
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
            }}
          >
            <span>RAAP-X AI CORRECTED ▶</span>
          </div>

          {/* Draggable Vertical Divider Bar */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: `${splitPosition}%`,
              width: '3px',
              background: '#FFFFFF',
              boxShadow: '0 0 10px rgba(0,0,0,0.5)',
              transform: 'translateX(-50%)',
            }}
          >
            {/* Center Draggable Slider Handle */}
            <div
              style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                width: '34px',
                height: '34px',
                borderRadius: '50%',
                background: '#FFFFFF',
                border: '3px solid var(--deep-blue)',
                boxShadow: '0 4px 12px rgba(0,0,0,0.35)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                pointerEvents: 'auto',
                cursor: 'ew-resize',
              }}
            >
              <span style={{ fontSize: '13px', fontWeight: 900, color: 'var(--deep-blue)' }}>⬌</span>
            </div>
          </div>

          {/* Invisible interactive range input for seamless drag navigation */}
          <input
            type="range"
            min="5"
            max="95"
            value={splitPosition}
            onChange={(e) => setSplitPosition(Number(e.target.value))}
            style={{
              position: 'absolute',
              top: '40%',
              left: 0,
              width: '100%',
              height: '90px',
              opacity: 0,
              cursor: 'ew-resize',
              pointerEvents: 'auto',
              margin: 0,
              zIndex: 500,
            }}
            aria-label="Map split comparison slider"
          />
        </div>
      )}

      {/* Dynamic Metric Legend */}
      <MapLegend activeMetric={activeMetric} />

      {/* Loading Overlay */}
      {loading && (
        <div
          style={{
            position: 'absolute',
            top: 16,
            right: 16,
            zIndex: 900,
            background: 'rgba(255, 255, 255, 0.95)',
            padding: '0.4rem 0.8rem',
            borderRadius: '9999px',
            fontSize: '0.75rem',
            color: 'var(--deep-blue)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            border: '1px solid var(--soft-border)',
            boxShadow: 'var(--shadow-sm)',
          }}
        >
          <div className="status-dot" style={{ backgroundColor: 'var(--sky-blue)' }} />
          <span>Synchronizing GIS layers...</span>
        </div>
      )}
    </div>
  );
};
