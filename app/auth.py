"""API key authentication for protected endpoints.

Authentication is enforced only when ``settings.API_KEY`` is set. When it is
empty (the default), the dependency is a no-op so local development and the test
suite can run without credentials.
"""

import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import settings

_API_KEY_HEADER = "X-API-Key"

# ``auto_error=False`` lets us craft our own 401 and skip auth entirely when no
# key is configured.
_api_key_header = APIKeyHeader(name=_API_KEY_HEADER, auto_error=False)


def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """FastAPI dependency that validates the ``X-API-Key`` header.

    No-op when ``API_KEY`` is unset. Otherwise requires a constant-time match.
    """
    if not settings.API_KEY:
        return

    if not api_key or not secrets.compare_digest(api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": _API_KEY_HEADER},
        )
