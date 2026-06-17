"""Rate limiting for the public query endpoints (ADR-002).

Requests are keyed by the ``X-API-Key`` header when present, otherwise by the
client IP. The limit is configurable via ``RATE_LIMIT_RPM`` and is applied only
to ``/query`` and ``/query/stream`` — health and metrics stay exempt.
"""

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import settings


def _rate_limit_key(request: Request) -> str:
    """Prefer the API key as the bucket key, falling back to the client IP."""
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key}"
    return get_remote_address(request)


def rate_limit_value(*args: object, **kwargs: object) -> str:
    """Dynamic limit string, re-read from settings on every request."""
    return f"{settings.RATE_LIMIT_RPM}/minute"


# headers_enabled stays False: injecting X-RateLimit-* headers would require
# every limited endpoint to accept a Response param. The Retry-After header is
# added explicitly by the 429 handler below instead.
limiter = Limiter(key_func=_rate_limit_key)


def rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Return a 429 with a ``Retry-After`` header when the limit is exceeded."""
    response = JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please slow down and retry."},
    )
    # Per-minute window, so the client may retry within 60 seconds.
    response.headers["Retry-After"] = "60"
    return response
