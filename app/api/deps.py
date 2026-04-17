from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from app.core.config import get_settings


def require_service_auth(
    authorization: str | None = Header(default=None),
    x_client_id: str | None = Header(default=None),
) -> str:
    settings = get_settings()

    if not settings.require_service_auth:
        return x_client_id or "anonymous"

    if not x_client_id or x_client_id not in settings.allowed_client_ids:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client identity.",
        )

    expected_api_key = settings.prompt_transformer_api_key
    provided_api_key = _read_bearer_token(authorization)
    if not expected_api_key or provided_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing service credentials.",
        )

    if not hmac.compare_digest(provided_api_key, expected_api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid service credentials.",
        )

    return x_client_id


def _read_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None

    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None

    return value.strip()
