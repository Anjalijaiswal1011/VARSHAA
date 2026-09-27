# Security Baseline Policy — RAIN-REPAIR X (VARSHAA)

**Status:** Approved  
**Version:** 1.0.0  
**Scope:** SIH 2026 Competition & Demonstration Security Baseline  

---

## 1. Secrets Management Policy

1. **Zero Hardcoded Credentials:** Under no circumstances shall database connection strings, cloud storage tokens, or external API keys be committed to source code or git history.
2. **Environment Variable Injection:** All sensitive configurations must be injected at runtime via environment variables loaded from `.env` (which is strictly git-ignored).
3. **Log Sanitization:** The logging layer (`src/utils/logging.py`) automatically executes regex masking on all outbound log messages to prevent accidental credential leakage in stdout or log files.

---

## 2. API & Network Security

1. **Input Validation:** All API endpoints must parse request bodies and query parameters using strict Pydantic models. Coordinates must validate within numerical ranges:
   $$\text{Latitude} \in [6.0, 38.0], \quad \text{Longitude} \in [68.0, 98.0]$$
2. **CORS Restrictions:** Cross-Origin Resource Sharing (CORS) is configured explicitly via `CORS_ORIGINS` in `configs/base_config.yaml`. Wildcard (`"*"`) origins are prohibited in production mode.
3. **Payload Size Limits:** File uploads (e.g., test NetCDF/GRIB files) are constrained to a maximum of 50 MB to prevent resource exhaustion / Denial of Service.
4. **Rate Limiting:** Public API endpoints should implement token-bucket or windowed rate limiting in production to guard against automated scraping.

---

## 3. File System & Ingestion Security

1. **Path Traversal Prevention:** All file path references from external API inputs or cycle dates must be sanitized through `pathlib.Path.resolve()` to prevent directory traversal attacks (`../`).
2. **Safe Serialization:** Avoid insecure Python `pickle` loading on untrusted files from network sources. Model weights are serialized through audited libraries (`joblib`, `safetensors`, or `ONNX`).
3. **Read-Only Raw Archives:** `data/raw/` is treated as an immutable append-only archive. The processing pipeline runs with non-root permissions.
