# Notebooks Directory Guidelines

This directory is designated for exploratory analysis, prototype visualization, and experimental feature diagnostics.

## Notebook Rules:
1. **Naming Convention:**
   - `01_eda_nwp_bias.ipynb`
   - `02_monsoon_regimes_clustering.ipynb`
   - `03_orographic_topography_features.ipynb`
   - `04_model_prototype_quantile_lgb.ipynb`
2. **Never hardcode paths:** Import configuration from `src.utils.config` using relative paths or environment variables.
3. **Keep notebooks clean:** Clear heavy output cells (especially large image / raster maps) before committing to reduce repository bloat.
4. **Transition to production:** Reusable logic developed in notebooks must be refactored into `src/` modules with accompanying tests.
