"""
API Gateway Middleware Stack for RAIN-REPAIR X (PART 9).
Provides:
    - Request Tracing (X-Request-ID, execution timing)
    - OWASP Security Headers (X-Content-Type-Options, X-Frame-Options, HSTS)
    - Sliding Window Rate Limiting (per-IP / API key with 429 Too Many Requests)
    - Access Audit Logging
"""

from __future__ import annotations

from datetime import datetime, timezone
import time
import uuid
from typing import Callable, Dict, List
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from backend.db.models import AuditLogModel
from backend.db.session import SessionLocal
from src.utils.logging import get_logger

logger = get_logger("rain_repair.backend.gateway")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Enforces OWASP recommended HTTP security headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Assigns unique X-Request-ID and measures request latency, logging slow requests.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.perf_counter()

        # Attach request_id to state
        request.state.request_id = request_id

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        # Asynchronously log access to database
        try:
            client_ip = request.client.host if request.client else "unknown"
            db = SessionLocal()
            log_entry = AuditLogModel(
                request_id=request_id,
                client_ip=client_ip,
                endpoint=str(request.url.path),
                method=request.method,
                status_code=response.status_code,
                response_time_ms=round(duration_ms, 2),
            )
            db.add(log_entry)
            db.commit()
            db.close()
        except Exception:
            # Audit logging should never block or fail request responses
            pass

        return response


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter per client IP / API token.
    Default limit: 120 requests per 60-second window.
    """

    def __init__(self, app, requests_limit: int = 120, window_seconds: int = 60):
        super().__init__(app)
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        self._clients: Dict[str, List[float]] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Exempt health and docs from rate limiting
        path = request.url.path
        if path in ["/docs", "/redoc", "/openapi.json", "/api/v1/health", "/"]:
            return await call_next(request)

        client_key = request.headers.get("X-API-Key") or (request.client.host if request.client else "global")
        now = time.time()
        window_start = now - self.window_seconds

        # Clean timestamps older than window
        timestamps = self._clients.get(client_key, [])
        valid_timestamps = [t for t in timestamps if t > window_start]

        if len(valid_timestamps) >= self.requests_limit:
            retry_after = int(self.window_seconds - (now - valid_timestamps[0]))
            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            logger.warning("Rate limit exceeded for client '%s'.", client_key)
            return JSONResponse(
                status_code=429,
                content={
                    "status": 429,
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Too many requests. Limit is {self.requests_limit} requests per {self.window_seconds}s.",
                    "timestamp": now_utc,
                },
                headers={
                    "Retry-After": str(max(1, retry_after)),
                    "X-RateLimit-Limit": str(self.requests_limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        valid_timestamps.append(now)
        self._clients[client_key] = valid_timestamps

        response = await call_next(request)
        remaining = max(0, self.requests_limit - len(valid_timestamps))
        response.headers["X-RateLimit-Limit"] = str(self.requests_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
