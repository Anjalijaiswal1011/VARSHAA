"""
Formal Feature Schema & Metadata Registry for RAIN-REPAIR X (VARSHAA).
Defines every feature, its mathematical formulation, physical meaning,
units, data types, leakage risk classification, and consuming downstream models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import pandas as pd


class FeatureGroup(str, Enum):
    NWP = "NWP"
    ATMOSPHERIC = "ATMOSPHERIC"
    PHYSICS = "PHYSICS"
    TERRAIN = "TERRAIN"
    SPATIAL = "SPATIAL"
    CLIMATOLOGY = "CLIMATOLOGY"
    TEMPORAL = "TEMPORAL"
    ERROR_MEMORY = "ERROR_MEMORY"
    REGIME_SUPPORT = "REGIME_SUPPORT"


class LeakageRisk(str, Enum):
    SAFE = "SAFE"
    POTENTIAL_LEAKAGE = "POTENTIAL_LEAKAGE"
    FUTURE_DEPENDENT = "FUTURE_DEPENDENT"


@dataclass(frozen=True)
class FeatureMetadata:
    feature_name: str
    feature_group: FeatureGroup
    source: str
    data_type: str
    unit: str
    description: str
    physical_meaning: str
    leakage_risk: LeakageRisk
    required_for: List[str] = field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["feature_group"] = self.feature_group.value
        d["leakage_risk"] = self.leakage_risk.value
        return d


# ==============================================================================
# Master Feature Catalogue Registry
# ==============================================================================

FEATURE_REGISTRY: Dict[str, FeatureMetadata] = {
    # --------------------------------------------------------------------------
    # Group A: NWP Features
    # --------------------------------------------------------------------------
    "nwp_precip": FeatureMetadata(
        feature_name="nwp_precip",
        feature_group=FeatureGroup.NWP,
        source="NWP Model (GFS/NCUM)",
        data_type="float32",
        unit="mm/24h",
        description="Raw NWP 24-hour accumulated total precipitation forecast",
        physical_meaning="Primary uncorrected precipitation baseline forecast",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["quantile_model", "probability_model", "error_model", "regime_classifier"],
        min_value=0.0,
        max_value=1500.0,
    ),
    "nwp_precip_log": FeatureMetadata(
        feature_name="nwp_precip_log",
        feature_group=FeatureGroup.NWP,
        source="Derived from NWP",
        data_type="float32",
        unit="log(mm+1)",
        description="Log-transformed raw NWP precipitation: ln(1 + nwp_precip)",
        physical_meaning="Stabilizes heavy-tail right skewness for tree splits",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["quantile_model", "probability_model"],
        min_value=0.0,
        max_value=10.0,
    ),
    "lead_time": FeatureMetadata(
        feature_name="lead_time",
        feature_group=FeatureGroup.NWP,
        source="NWP Forecast Metadata",
        data_type="int32",
        unit="hours",
        description="Forecast lead time from cycle initialization (24, 48, 72, 96, 120)",
        physical_meaning="Controls forecast error growth with temporal horizon",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["quantile_model", "probability_model", "regime_classifier"],
        min_value=24.0,
        max_value=120.0,
    ),
    "lead_time_days": FeatureMetadata(
        feature_name="lead_time_days",
        feature_group=FeatureGroup.NWP,
        source="Derived from lead_time",
        data_type="float32",
        unit="days",
        description="Forecast lead time in fractional days (lead_time / 24.0)",
        physical_meaning="Linear scaling of forecast horizon",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["quantile_model"],
        min_value=1.0,
        max_value=5.0,
    ),
    "t2m": FeatureMetadata(
        feature_name="t2m",
        feature_group=FeatureGroup.ATMOSPHERIC,
        source="NWP Model",
        data_type="float32",
        unit="Kelvin",
        description="2-meter air temperature",
        physical_meaning="Surface thermal energy state and boundary layer heating",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["regime_classifier", "quantile_model"],
        min_value=220.0,
        max_value=335.0,
    ),
    "rh850": FeatureMetadata(
        feature_name="rh850",
        feature_group=FeatureGroup.ATMOSPHERIC,
        source="NWP Model",
        data_type="float32",
        unit="%",
        description="Relative humidity at 850 hPa pressure level",
        physical_meaning="Lower tropospheric moisture saturation level",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["regime_classifier", "quantile_model", "probability_model"],
        min_value=0.0,
        max_value=100.0,
    ),
    "mslp": FeatureMetadata(
        feature_name="mslp",
        feature_group=FeatureGroup.ATMOSPHERIC,
        source="NWP Model",
        data_type="float32",
        unit="hPa",
        description="Mean sea-level pressure",
        physical_meaning="Synoptic pressure gradients, troughs, and cyclonic low centers",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["regime_classifier", "transition_engine"],
        min_value=900.0,
        max_value=1050.0,
    ),
    "u850": FeatureMetadata(
        feature_name="u850",
        feature_group=FeatureGroup.ATMOSPHERIC,
        source="NWP Model",
        data_type="float32",
        unit="m/s",
        description="Zonal (east-west) wind component at 850 hPa",
        physical_meaning="Primary monsoon westerly jet strength across the Arabian Sea & peninsula",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["regime_classifier", "quantile_model"],
        min_value=-100.0,
        max_value=100.0,
    ),
    "v850": FeatureMetadata(
        feature_name="v850",
        feature_group=FeatureGroup.ATMOSPHERIC,
        source="NWP Model",
        data_type="float32",
        unit="m/s",
        description="Meridional (north-south) wind component at 850 hPa",
        physical_meaning="Cross-equatorial flow and cyclonic vorticity transport",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["regime_classifier", "quantile_model"],
        min_value=-100.0,
        max_value=100.0,
    ),
    "wind_speed_850": FeatureMetadata(
        feature_name="wind_speed_850",
        feature_group=FeatureGroup.ATMOSPHERIC,
        source="Derived from u850, v850",
        data_type="float32",
        unit="m/s",
        description="Wind speed magnitude at 850 hPa: sqrt(u^2 + v^2)",
        physical_meaning="Kinetic intensity of the low-level monsoon jet stream",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["regime_classifier", "quantile_model"],
        min_value=0.0,
        max_value=120.0,
    ),
    "wind_direction_850": FeatureMetadata(
        feature_name="wind_direction_850",
        feature_group=FeatureGroup.ATMOSPHERIC,
        source="Derived from u850, v850",
        data_type="float32",
        unit="degrees",
        description="Meteorological wind direction from which wind blows (0-360 deg)",
        physical_meaning="Identifies southwesterly monsoon flow vs northeasterly retreat",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["regime_classifier"],
        min_value=0.0,
        max_value=360.0,
    ),
    "vapor_pressure_proxy": FeatureMetadata(
        feature_name="vapor_pressure_proxy",
        feature_group=FeatureGroup.ATMOSPHERIC,
        source="Derived from T2m, RH850",
        data_type="float32",
        unit="hPa",
        description="Estimated actual vapor pressure via Tetens formula: RH * e_sat(T2m)",
        physical_meaning="Absolute atmospheric moisture mass availability in the air column",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["probability_model", "quantile_model"],
        min_value=0.0,
        max_value=70.0,
    ),
    # --------------------------------------------------------------------------
    # Group C: Physics-Guided Features
    # --------------------------------------------------------------------------
    "orographic_uplift": FeatureMetadata(
        feature_name="orographic_uplift",
        feature_group=FeatureGroup.PHYSICS,
        source="Derived from V_850 and DEM gradient",
        data_type="float32",
        unit="m/s",
        description="Mechanical orographic ascent: w_orog = u*(dh/dx) + v*(dh/dy)",
        physical_meaning="Forced windward mechanical uplift and leeward rain-shadow subsidence",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["quantile_model", "probability_model", "regime_classifier"],
        min_value=-50.0,
        max_value=50.0,
    ),
    "moisture_flux_conv": FeatureMetadata(
        feature_name="moisture_flux_conv",
        feature_group=FeatureGroup.PHYSICS,
        source="Derived from V_850 and RH850",
        data_type="float32",
        unit="1/s",
        description="2D horizontal moisture flux convergence: - div(RH * V_850)",
        physical_meaning="Dynamical convergence of moist air parcels fueling deep convection",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["quantile_model", "probability_model"],
        min_value=-0.05,
        max_value=0.05,
    ),
    "vorticity_850": FeatureMetadata(
        feature_name="vorticity_850",
        feature_group=FeatureGroup.PHYSICS,
        source="Derived from u850, v850",
        data_type="float32",
        unit="1/s",
        description="Relative vertical vorticity at 850 hPa: (dv/dx) - (du/dy)",
        physical_meaning="Tracks cyclonic vortex centers during Bay of Bengal depressions",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["regime_classifier", "transition_engine"],
        min_value=-0.005,
        max_value=0.005,
    ),
    "bulk_wind_shear": FeatureMetadata(
        feature_name="bulk_wind_shear",
        feature_group=FeatureGroup.PHYSICS,
        source="Derived from 850hPa and 200hPa winds",
        data_type="float32",
        unit="m/s",
        description="Deep-layer vertical wind shear magnitude: ||V_200 - V_850||",
        physical_meaning="Controls tilt, severity, and organizational longevity of convective clusters",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["probability_model", "regime_classifier"],
        min_value=0.0,
        max_value=120.0,
    ),
    "windward_leeward_index": FeatureMetadata(
        feature_name="windward_leeward_index",
        feature_group=FeatureGroup.PHYSICS,
        source="Derived from wind direction and aspect",
        data_type="float32",
        unit="index [-1, 1]",
        description="Cosine of angle between incoming 850hPa wind and terrain aspect vector",
        physical_meaning="+1 indicates pure windward face; -1 indicates pure leeward shadow",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["quantile_model"],
        min_value=-1.0,
        max_value=1.0,
    ),
    # --------------------------------------------------------------------------
    # Group D: Terrain Features
    # --------------------------------------------------------------------------
    "elevation": FeatureMetadata(
        feature_name="elevation",
        feature_group=FeatureGroup.TERRAIN,
        source="SRTM / Copernicus DEM",
        data_type="float32",
        unit="meters",
        description="Surface terrain elevation above sea level",
        physical_meaning="Altitudinal cooling and condensation height",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["quantile_model", "regime_classifier"],
        min_value=-50.0,
        max_value=8848.0,
    ),
    "dem_slope": FeatureMetadata(
        feature_name="dem_slope",
        feature_group=FeatureGroup.TERRAIN,
        source="Derived from DEM",
        data_type="float32",
        unit="degrees",
        description="Local terrain slope gradient magnitude",
        physical_meaning="Steepness of topographical barriers causing acute mechanical lift",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["quantile_model"],
        min_value=0.0,
        max_value=90.0,
    ),
    "dem_aspect": FeatureMetadata(
        feature_name="dem_aspect",
        feature_group=FeatureGroup.TERRAIN,
        source="Derived from DEM",
        data_type="float32",
        unit="degrees",
        description="Azimuth direction that terrain slope faces (0-360 deg)",
        physical_meaning="Identifies barrier exposure relative to prevailing winds",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["quantile_model"],
        min_value=0.0,
        max_value=360.0,
    ),
    "dist_to_coast": FeatureMetadata(
        feature_name="dist_to_coast",
        feature_group=FeatureGroup.TERRAIN,
        source="Vector GIS / Coastline geometry",
        data_type="float32",
        unit="km",
        description="Euclidean distance to Indian coastline (Arabian Sea / Bay of Bengal)",
        physical_meaning="Maritime boundary-layer moderation and coastal squall line frequency",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["regime_classifier", "quantile_model"],
        min_value=0.0,
        max_value=2500.0,
    ),
    "relative_elevation_exposure": FeatureMetadata(
        feature_name="relative_elevation_exposure",
        feature_group=FeatureGroup.TERRAIN,
        source="Derived from DEM neighborhood",
        data_type="float32",
        unit="meters",
        description="Elevation minus local 3x3 neighborhood mean elevation: h - h_mean",
        physical_meaning="Distinguishes exposed mountain ridges (>0) from sheltered valleys (<0)",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["quantile_model"],
        min_value=-2000.0,
        max_value=2000.0,
    ),
    # --------------------------------------------------------------------------
    # Group E: Spatial Context Features
    # --------------------------------------------------------------------------
    "nwp_precip_spatial_mean_3x3": FeatureMetadata(
        feature_name="nwp_precip_spatial_mean_3x3",
        feature_group=FeatureGroup.SPATIAL,
        source="Derived from NWP neighborhood",
        data_type="float32",
        unit="mm/24h",
        description="Mean NWP precipitation in a 3x3 grid window (~75km x 75km area)",
        physical_meaning="Mesoscale surrounding precipitation context",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["quantile_model", "probability_model"],
        min_value=0.0,
        max_value=1500.0,
    ),
    "nwp_precip_spatial_max_3x3": FeatureMetadata(
        feature_name="nwp_precip_spatial_max_3x3",
        feature_group=FeatureGroup.SPATIAL,
        source="Derived from NWP neighborhood",
        data_type="float32",
        unit="mm/24h",
        description="Maximum NWP precipitation in a 3x3 grid window",
        physical_meaning="Captures presence of severe convective cores in adjacent cells",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["probability_model"],
        min_value=0.0,
        max_value=1500.0,
    ),
    "nwp_precip_spatial_std_3x3": FeatureMetadata(
        feature_name="nwp_precip_spatial_std_3x3",
        feature_group=FeatureGroup.SPATIAL,
        source="Derived from NWP neighborhood",
        data_type="float32",
        unit="mm/24h",
        description="Standard deviation of NWP precipitation in 3x3 grid window",
        physical_meaning="Spatial heterogeneity and convective patchiness of precipitation",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["probability_model"],
        min_value=0.0,
        max_value=500.0,
    ),
    "nwp_precip_spatial_gradient": FeatureMetadata(
        feature_name="nwp_precip_spatial_gradient",
        feature_group=FeatureGroup.SPATIAL,
        source="Derived from NWP",
        data_type="float32",
        unit="mm/km",
        description="Magnitude of 2D spatial gradient: ||grad(NWP)||",
        physical_meaning="Identifies sharp convective rain edges and frontal boundaries",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["quantile_model"],
        min_value=0.0,
        max_value=100.0,
    ),
    # --------------------------------------------------------------------------
    # Group F: Climatology Features
    # --------------------------------------------------------------------------
    "clim_mean_doy": FeatureMetadata(
        feature_name="clim_mean_doy",
        feature_group=FeatureGroup.CLIMATOLOGY,
        source="Historical 30-Year Observation Archive",
        data_type="float32",
        unit="mm/24h",
        description="Climatological historical mean precipitation for the calendar day-of-year",
        physical_meaning="Long-term historical baseline rainfall expectation",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["quantile_model", "regime_classifier"],
        min_value=0.0,
        max_value=500.0,
    ),
    "nwp_clim_anomaly": FeatureMetadata(
        feature_name="nwp_clim_anomaly",
        feature_group=FeatureGroup.CLIMATOLOGY,
        source="Derived from NWP and Climatology",
        data_type="float32",
        unit="mm/24h",
        description="NWP precipitation anomaly relative to climatology: NWP - clim_mean_doy",
        physical_meaning="Identifies whether forecast is unusually wet or dry relative to normal",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["quantile_model", "probability_model"],
        min_value=-500.0,
        max_value=1500.0,
    ),
    # --------------------------------------------------------------------------
    # Group G: Temporal Features
    # --------------------------------------------------------------------------
    "doy_sin": FeatureMetadata(
        feature_name="doy_sin",
        feature_group=FeatureGroup.TEMPORAL,
        source="Calendar Cycle Date",
        data_type="float32",
        unit="unitless [-1, 1]",
        description="Sine harmonic of Day of Year: sin(2 * pi * DOY / 365.25)",
        physical_meaning="Smooth seasonal progression of annual solar insolation",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["regime_classifier", "quantile_model"],
        min_value=-1.0,
        max_value=1.0,
    ),
    "doy_cos": FeatureMetadata(
        feature_name="doy_cos",
        feature_group=FeatureGroup.TEMPORAL,
        source="Calendar Cycle Date",
        data_type="float32",
        unit="unitless [-1, 1]",
        description="Cosine harmonic of Day of Year: cos(2 * pi * DOY / 365.25)",
        physical_meaning="Smooth seasonal progression of annual solar insolation",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["regime_classifier", "quantile_model"],
        min_value=-1.0,
        max_value=1.0,
    ),
    "month_sin": FeatureMetadata(
        feature_name="month_sin",
        feature_group=FeatureGroup.TEMPORAL,
        source="Calendar Cycle Date",
        data_type="float32",
        unit="unitless [-1, 1]",
        description="Sine harmonic of Month: sin(2 * pi * Month / 12)",
        physical_meaning="Monsoon seasonal phase representation",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["regime_classifier"],
        min_value=-1.0,
        max_value=1.0,
    ),
    "month_cos": FeatureMetadata(
        feature_name="month_cos",
        feature_group=FeatureGroup.TEMPORAL,
        source="Calendar Cycle Date",
        data_type="float32",
        unit="unitless [-1, 1]",
        description="Cosine harmonic of Month: cos(2 * pi * Month / 12)",
        physical_meaning="Monsoon seasonal phase representation",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["regime_classifier"],
        min_value=-1.0,
        max_value=1.0,
    ),
    # --------------------------------------------------------------------------
    # Group H: Multi-Scale Error Memory Features
    # --------------------------------------------------------------------------
    "error_memory_lag1": FeatureMetadata(
        feature_name="error_memory_lag1",
        feature_group=FeatureGroup.ERROR_MEMORY,
        source="Historical Error Store (T - 1 day)",
        data_type="float32",
        unit="mm/24h",
        description="NWP minus Observation error from yesterday (cycle T - 1)",
        physical_meaning="Immediate 24-hour model persistence bias",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["error_model", "quantile_model"],
        min_value=-500.0,
        max_value=500.0,
    ),
    "error_memory_lag3_mean": FeatureMetadata(
        feature_name="error_memory_lag3_mean",
        feature_group=FeatureGroup.ERROR_MEMORY,
        source="Historical Error Store (Trailing 3 days)",
        data_type="float32",
        unit="mm/24h",
        description="Trailing 3-day average NWP minus Observation bias",
        physical_meaning="Short-term synoptic error trend",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["error_model", "quantile_model"],
        min_value=-500.0,
        max_value=500.0,
    ),
    "error_memory_lag7_mean": FeatureMetadata(
        feature_name="error_memory_lag7_mean",
        feature_group=FeatureGroup.ERROR_MEMORY,
        source="Historical Error Store (Trailing 7 days)",
        data_type="float32",
        unit="mm/24h",
        description="Trailing 7-day average NWP minus Observation bias",
        physical_meaning="Weekly synoptic-scale model drift",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["error_model", "quantile_model"],
        min_value=-500.0,
        max_value=500.0,
    ),
    "error_memory_lag14_mean": FeatureMetadata(
        feature_name="error_memory_lag14_mean",
        feature_group=FeatureGroup.ERROR_MEMORY,
        source="Historical Error Store (Trailing 14 days)",
        data_type="float32",
        unit="mm/24h",
        description="Trailing 14-day average NWP minus Observation bias",
        physical_meaning="Intra-seasonal persistent bias memory",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["error_model", "quantile_model"],
        min_value=-500.0,
        max_value=500.0,
    ),
    # --------------------------------------------------------------------------
    # Group I: Regime Support Diagnostics (Features for Future Regime Classifier)
    # --------------------------------------------------------------------------
    "wd_shear_proxy": FeatureMetadata(
        feature_name="wd_shear_proxy",
        feature_group=FeatureGroup.REGIME_SUPPORT,
        source="Derived from U200, U850 and Latitude",
        data_type="float32",
        unit="m/s",
        description="Subtropical westerly jet indicator: max(u200, 0) * (lat >= 24.0)",
        physical_meaning="Upper-tropospheric jet dynamics indicating incoming Western Disturbances",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["regime_classifier"],
        min_value=0.0,
        max_value=150.0,
    ),
    "monsoon_trough_mslp_gradient": FeatureMetadata(
        feature_name="monsoon_trough_mslp_gradient",
        feature_group=FeatureGroup.REGIME_SUPPORT,
        source="Derived from MSLP grid",
        data_type="float32",
        unit="hPa/deg",
        description="North-South MSLP gradient across Central India: d(MSLP)/d(Lat)",
        physical_meaning="Determines Monsoon Trough axis position (Active vs Break phases)",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["regime_classifier", "transition_engine"],
        min_value=-10.0,
        max_value=10.0,
    ),
    "offshore_trough_coastal_shear": FeatureMetadata(
        feature_name="offshore_trough_coastal_shear",
        feature_group=FeatureGroup.REGIME_SUPPORT,
        source="Derived from V_850 and dist_to_coast",
        data_type="float32",
        unit="m/s",
        description="Low-level coastal wind convergence along Arabian Sea coastline (lat 10-18N)",
        physical_meaning="Identifies Arabian Sea offshore troughs triggering intense coastal deluge",
        leakage_risk=LeakageRisk.SAFE,
        required_for=["regime_classifier"],
        min_value=-50.0,
        max_value=50.0,
    ),
}


def get_feature_names_for_model(model_name: str) -> List[str]:
    """Retrieve all feature names required for a specific downstream model."""
    return [name for name, meta in FEATURE_REGISTRY.items() if model_name in meta.required_for]


def get_feature_group_names(group: FeatureGroup) -> List[str]:
    """Retrieve all feature names belonging to a specific feature group."""
    return [name for name, meta in FEATURE_REGISTRY.items() if meta.feature_group == group]


def validate_dataframe_against_schema(
    df: pd.DataFrame,
    required_features: Optional[List[str]] = None,
    allow_missing_cols: bool = False,
) -> Dict[str, Any]:
    """
    Validate that a feature DataFrame satisfies registry types, bounds, and column requirements.

    Returns:
        Validation report dictionary with 'is_valid', 'missing_features', and 'out_of_bounds'.
    """
    if required_features is None:
        required_features = list(FEATURE_REGISTRY.keys())

    missing = [f for f in required_features if f not in df.columns]
    out_of_bounds: Dict[str, Dict[str, float]] = {}

    for f in required_features:
        if f in df.columns:
            meta = FEATURE_REGISTRY.get(f)
            if meta is not None:
                series = df[f].dropna()
                if not series.empty:
                    min_val = float(series.min())
                    max_val = float(series.max())
                    if meta.min_value is not None and min_val < meta.min_value - 1e-4:
                        out_of_bounds[f] = {"min_observed": min_val, "min_allowed": meta.min_value}
                    if meta.max_value is not None and max_val > meta.max_value + 1e-4:
                        out_of_bounds[f] = {"max_observed": max_val, "max_allowed": meta.max_value}

    is_valid = (len(missing) == 0 or allow_missing_cols) and len(out_of_bounds) == 0

    return {
        "is_valid": is_valid,
        "missing_features": missing,
        "out_of_bounds": out_of_bounds,
        "total_columns_checked": len(required_features),
    }
