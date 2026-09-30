"""
FastAPI Application Entry Point & API Gateway for RAIN-REPAIR X (PART 9).
Operational Extreme Weather & NWP Error Repair Intelligence Backend.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.db.init_db import init_db
from backend.gateway import (
    RateLimiterMiddleware,
    RequestTracingMiddleware,
    SecurityHeadersMiddleware,
)
from backend.routes import router as api_router
from src.utils.logging import get_logger

logger = get_logger("rain_repair.backend.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for application startup and shutdown."""
    logger.info("Initializing database schema and seed data...")
    init_db()
    logger.info("Starting up RAIN-REPAIR X Backend API Gateway...")
    yield
    logger.info("Shutting down RAIN-REPAIR X Backend API Service...")


def create_app() -> FastAPI:
    """Application factory for RAIN-REPAIR X Backend & API Gateway."""
    app = FastAPI(
        title="RAIN-REPAIR X Production Backend API",
        description=(
            "Operational AI Decision Support & NWP Error Repair Intelligence Platform. "
            "Delivers probabilistic multi-quantile rainfall plumes, calibrated exceedance probabilities, "
            "EVT extreme tail risk, and IMD-compliant district warning layers with RBAC, caching, and rate limiting."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # 1. Security Headers Middleware (OWASP)
    app.add_middleware(SecurityHeadersMiddleware)

    # 2. Request Tracing & Audit Log Middleware
    app.add_middleware(RequestTracingMiddleware)

    # 3. Rate Limiting Middleware (1000 req/min for smooth spatial tile rendering)
    app.add_middleware(RateLimiterMiddleware, requests_limit=1000, window_seconds=60)

    # 4. GZip Compression Middleware (for large GeoJSON FeatureCollections)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # 5. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Permits local Vite dev server
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API Router
    app.include_router(api_router)

    # RFC 7807 Exception Handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        error_code = "HTTP_ERROR"
        if exc.status_code == 400:
            error_code = "BAD_REQUEST"
        elif exc.status_code == 401:
            error_code = "UNAUTHORIZED"
        elif exc.status_code == 403:
            error_code = "FORBIDDEN"
        elif exc.status_code == 404:
            error_code = "RESOURCE_NOT_FOUND"
        elif exc.status_code == 422:
            error_code = "VALIDATION_ERROR"
        elif exc.status_code == 429:
            error_code = "RATE_LIMIT_EXCEEDED"

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": exc.status_code,
                "error_code": error_code,
                "message": str(exc.detail),
                "timestamp": now_utc,
            },
            headers=dict(getattr(exc, "headers", {}) or {}),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.error("Unhandled server exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": 500,
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred while processing the weather forecast request.",
                "timestamp": now_utc,
            },
        )

    # Mount Frontend Dashboard if built
    from pathlib import Path
    from starlette.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    dist_dir = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if dist_dir.exists() and (dist_dir / "index.html").exists():
        assets_dir = dist_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend_assets")

        @app.get("/dashboard", tags=["Dashboard"])
        async def serve_dashboard():
            return FileResponse(str(dist_dir / "index.html"))

        @app.get("/favicon.svg", include_in_schema=False)
        async def favicon():
            return FileResponse(str(dist_dir / "favicon.svg"))

        @app.get("/icons.svg", include_in_schema=False)
        async def icons():
            return FileResponse(str(dist_dir / "icons.svg"))

    @app.get("/health", tags=["Health"])
    def health_check():
        return {
            "status": "healthy",
            "service": "RAIN-REPAIR X",
            "version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/", tags=["Root"])
    def root(request: Request):
        accept = request.headers.get("accept", "")
        if "text/html" in accept and dist_dir.exists() and (dist_dir / "index.html").exists():
            return FileResponse(str(dist_dir / "index.html"))
        return {
            "name": "RAIN-REPAIR X Production Backend API",
            "version": "1.0.0",
            "documentation": "/docs",
            "dashboard": "/dashboard",
            "health": "/api/v1/health",
            "gateway_status": "operational",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
