# Contributing Guidelines & Engineering Standards — RAIN-REPAIR X

**Audience:** All SIH 2026 Team Members & Contributors  
**Enforcement:** Automated via linters + Senior Developer PR Code Reviews  

---

## 1. General Engineering Principles

1. **Clarity Over Cleverness:** Write code that any team member can read, understand, and debug during high-pressure competition.
2. **Strict Module Boundaries:** Never import internal functions across unrelated domains (e.g., frontend code never executes Python ML logic; backend does not re-implement data validation).
3. **Zero Secrets in Code:** Never commit passwords, tokens, API keys, or private endpoints. Always retrieve credentials via `src.utils.config.load_config()` or environment variables.
4. **No Magic Numbers:** All meteorological thresholds (e.g. `64.5` mm for heavy rain, `0.25` for grid resolution) must be defined in `configs/base_config.yaml` or as explicit uppercase constants with units commented.
5. **No Dead Code:** Remove commented-out blocks, orphaned debug functions, and prototype scratch code before opening a PR.

---

## 2. Python Coding Standards

### 2.1 Style & Formatting
- **PEP 8 Compliance:** Strictly follow PEP 8.
- **Formatter & Linter:** Code must pass `ruff check .` and `black --check .` (line length: 100 characters max).
- **Imports Sorting:** Run `isort .` with standard library first, third-party second, local package (`src`) last.

### 2.2 Typing & Docstrings
- **Type Annotations:** All public functions, class methods, and dataclasses must include type hints:
  ```python
  def regrid_forecast(
      raw_grid: np.ndarray,
      source_lats: np.ndarray,
      source_lons: np.ndarray,
      target_resolution_deg: float = 0.25,
  ) -> np.ndarray:
      """
      Interpolate raw NWP precipitation grid to standard target resolution.

      Args:
          raw_grid: 2D array of precipitation in mm.
          source_lats: 1D array of latitude coordinates.
          source_lons: 1D array of longitude coordinates.
          target_resolution_deg: Target grid cell size in degrees (default 0.25).

      Returns:
          Regridded 2D precipitation array.

      Raises:
          InvalidCoordinateError: If coordinate arrays are malformed.
      """
  ```
- **Error Handling:** Use domain-specific exceptions from `src.utils.exceptions`. Never use bare `except:` or `except Exception: pass`.
- **Logging:** Use `src.utils.logging.get_logger(__name__)`. Do not use raw `print()` statements in production code.

---

## 3. JavaScript / TypeScript & Frontend Standards

1. **Component Modularity:** One component per file in `frontend/src/components/`. Components must not exceed 250 lines.
2. **State Decoupling:** Keep UI state (e.g., active map layer, selected date) separated from API query hooks.
3. **Graceful Loading & Error States:** Every map, chart, and card must render:
   - Skeleton/Spinner state while fetching.
   - User-friendly error message if API fails.
   - Clean empty state if no data exists.
4. **Accessibility & Responsive Design:** Ensure color-blind accessible alert palettes (e.g., supplemented with distinct icons/labels alongside colors).

---

## 4. Git & Commit Guidelines

- **Format:** `type(scope): concise subject in imperative mood`
- **Allowed Types:**
  - `feat`: A new feature (e.g., `feat(features): add rolling 3-day error memory calculation`)
  - `fix`: A bug fix (e.g., `fix(postproc): prevent negative rainfall values in p10 quantile`)
  - `docs`: Documentation only changes
  - `test`: Adding or correcting tests
  - `refactor`: Code change that neither fixes a bug nor adds a feature
  - `chore`: Build or dependency updates
- **Branch Naming:** `feature/<topic>`, `bugfix/<topic>`, `docs/<topic>`.
