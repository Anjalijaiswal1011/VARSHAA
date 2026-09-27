"""
Feature Engineering Package for RAIN-REPAIR X (VARSHAA).
Provides physics-guided, terrain, spatial, climatological, and multi-timescale error memory features.
"""

from src.features.climatology import ClimatologyEngine
from src.features.diagnostics import generate_feature_diagnostics_report
from src.features.error_memory import ErrorMemoryBuffer
from src.features.leakage_audit import audit_feature_dataframe_leakage
from src.features.physics import (
    compute_deep_layer_shear,
    compute_moisture_convergence_proxy,
    compute_orographic_uplift,
    compute_regime_support_diagnostics,
    compute_relative_vorticity,
    compute_vapor_pressure_proxy,
    compute_wind_speed_and_direction,
    compute_windward_leeward_index,
)
from src.features.pipeline import FeatureEngineeringPipeline
from src.features.scaling import RobustFeatureScaler, handle_missing_features, sanitize_domain_outliers
from src.features.schema import (
    FEATURE_REGISTRY,
    FeatureGroup,
    FeatureMetadata,
    LeakageRisk,
    get_feature_group_names,
    get_feature_names_for_model,
    validate_dataframe_against_schema,
)
from src.features.spatial import compute_spatial_neighborhood_context
from src.features.temporal import compute_temporal_features
from src.features.terrain import compute_distance_to_coast, compute_terrain_derivatives

__all__ = [
    "FeatureEngineeringPipeline",
    "FEATURE_REGISTRY",
    "FeatureGroup",
    "FeatureMetadata",
    "LeakageRisk",
    "ClimatologyEngine",
    "ErrorMemoryBuffer",
    "RobustFeatureScaler",
    "compute_orographic_uplift",
    "compute_moisture_convergence_proxy",
    "compute_relative_vorticity",
    "compute_deep_layer_shear",
    "compute_wind_speed_and_direction",
    "compute_vapor_pressure_proxy",
    "compute_windward_leeward_index",
    "compute_regime_support_diagnostics",
    "compute_terrain_derivatives",
    "compute_distance_to_coast",
    "compute_spatial_neighborhood_context",
    "compute_temporal_features",
    "sanitize_domain_outliers",
    "handle_missing_features",
    "generate_feature_diagnostics_report",
    "audit_feature_dataframe_leakage",
    "validate_dataframe_against_schema",
    "get_feature_names_for_model",
    "get_feature_group_names",
]
