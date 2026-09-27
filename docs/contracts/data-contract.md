# Interface Contract: Data Layer (DATA -> ML)

**Contract ID:** CONTRACT-DATA-001  
**Version:** 1.0.0  
**Owner:** Team A (Data Engineering)  
**Consumer:** Team B (Machine Learning & Features)  
**Status:** Frozen for Foundation Phase  

---

## 1. Overview & Purpose
This contract establishes the schema, spatial-temporal coordinate system, missing data conventions, and file formats that Team A must provide to Team B. Team B models must consume only data meeting these exact specifications.

---

## 2. Spatial & Temporal Standards

| Property | Standard / Value | Notes |
| :--- | :--- | :--- |
| **Coordinate Reference System** | `EPSG:4326` (WGS 84) | Geographic latitude / longitude in decimal degrees |
| **Spatial Bounds (India Domain)** | Latitude: `6.00°N` to `38.00°N`<br>Longitude: `68.00°E` to `98.00°E` | Sub-domains (e.g., Western Ghats) must align to this grid |
| **Grid Resolution** | `0.25°` ($\approx 27.5\text{ km}$) | Primary common grid (129 lat points, 121 lon points) |
| **Timestamp Format** | ISO 8601 string (`YYYY-MM-DDTHH:MM:SSZ`) | UTC standard reference |
| **Rainfall Accumulation Window** | 24-hour accumulation (`03:00 UTC` to `03:00 UTC` next day) | Matches standard IMD observation day (08:30 IST) |
| **Lead Times** | `24h, 48h, 72h, 96h, 120h` (Day 1 to Day 5) | Lead time measured from NWP model initialization |

---

## 3. Storage Formats

1. **Gridded Tensors (Primary):** NetCDF4 (`.nc`) or Zarr format with standardized dimensions: `(time, lead_time, lat, lon)`.
2. **Tabular Feature Matrix (Secondary):** Apache Parquet (`.parquet`) for tabular ML model training.

---

## 4. Field Specification (Gridded Schema)

| Field Name | Type | Required? | Units | Range / Valid Values | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `time` | `datetime64[ns]` | **Required** | UTC | Valid ISO date | Model cycle initialization date/time |
| `lead_time` | `int32` | **Required** | hours | `[24, 48, 72, 96, 120]` | Forecast lead time |
| `lat` | `float32` | **Required** | degrees_north | `[6.00, 38.00]` | Latitude coordinate (0.25° increments) |
| `lon` | `float32` | **Required** | degrees_east | `[68.00, 98.00]` | Longitude coordinate (0.25° increments) |
| `nwp_precip` | `float32` | **Required** | $\text{mm/24h}$ | $[0.0, 1500.0]$ | Raw NWP 24-hour total precipitation |
| `obs_precip` | `float32` | Optional (Training only) | $\text{mm/24h}$ | $[0.0, 1500.0]$ | Observed ground-truth rainfall (IMD / GPM) |
| `t2m` | `float32` | Optional | $\text{Kelvin}$ | $[220.0, 330.0]$ | 2-meter air temperature |
| `rh850` | `float32` | Optional | $\%$ | $[0.0, 100.0]$ | Relative humidity at 850 hPa level |
| `u850` | `float32` | Optional | $\text{m/s}$ | $[-100.0, 100.0]$ | Zonal wind at 850 hPa |
| `v850` | `float32` | Optional | $\text{m/s}$ | $[-100.0, 100.0]$ | Meridional wind at 850 hPa |
| `mslp` | `float32` | Optional | $\text{hPa}$ | $[900.0, 1050.0]$ | Mean sea-level pressure |
| `elevation` | `float32` | **Required** | meters | $[-50.0, 8848.0]$ | Static DEM topography elevation |
| `dem_slope` | `float32` | Optional | degrees | $[0.0, 90.0]$ | Topographic terrain slope |
| `error_memory_lag1` | `float32` | Optional | $\text{mm/24h}$ | $[-500.0, 500.0]$ | Raw NWP minus Obs from yesterday (Day -1) |
| `climatology_mean` | `float32` | Optional | $\text{mm/24h}$ | $[0.0, 500.0]$ | Historical day-of-year mean rainfall baseline |

*Note: Optional variables marked as `TBD / REQUIRES VALIDATION` during initial feature experiments.*

---

## 5. Missing Data & Quality Control Conventions

- **Fill Value:** Standard IEEE `NaN` (`np.nan`). Sentinel values such as `-9999.0` or `999.0` are strictly forbidden and must be masked by Team A during ingestion.
- **Missing Coordinate Policy:** If $> 5\%$ of grid points within the land boundary are missing from raw NWP, the ingestion stage must raise a `DataQualityError` rather than silently propagating invalid data.
- **Ocean / Land Mask:** An explicit land-sea binary mask (`land_mask`, boolean: `1 = land`, `0 = ocean`) must be provided. For land-only evaluations, oceanic cells may be masked.

---

## 6. Error Handling Contract

When validation fails in `src/data/validate.py`:
- Raise `src.utils.exceptions.InvalidCoordinateError` if grid resolution, bounding box, or CRS does not match `EPSG:4326` at 0.25°.
- Raise `src.utils.exceptions.TemporalAlignmentError` if timestamps or lead times are inconsistent.
- Raise `src.utils.exceptions.DataQualityError` if values are out of physical bounds (e.g., negative precipitation).
