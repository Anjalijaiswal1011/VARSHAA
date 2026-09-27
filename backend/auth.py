"""
Authentication, Authorization & Role-Based Access Control (RBAC) Module (PART 9).
Provides JWT Bearer authentication, API Key validation, and role enforcement.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import os
import secrets
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import jwt
from fastapi import Depends, Header, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, SecurityScopes
from pydantic import BaseModel, Field

from src.utils.logging import get_logger

logger = get_logger("rain_repair.backend.auth")

# Configuration constants (can be overridden via environment variables)
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "rain-repair-x-secret-key-sih2026-production-token")
JWT_ALGORITHM: str = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 120

RoleType = Literal["public", "forecaster", "admin"]

ROLE_HIERARCHY: Dict[str, int] = {
    "public": 1,
    "forecaster": 2,
    "admin": 3,
}

security_bearer = HTTPBearer(auto_error=False)


class UserCredentials(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    user_id: str
    username: str
    role: str


class UserPayload(BaseModel):
    user_id: str
    username: str
    role: RoleType = "public"
    is_active: bool = True


# Pre-configured default users for demonstration and testing
DEFAULT_USERS: Dict[str, Dict[str, Any]] = {
    "admin": {
        "user_id": "usr_admin_001",
        "username": "admin",
        "salt": "static_salt_admin",
        "password_hash": hashlib.pbkdf2_hmac(
            "sha256", "Admin@Varshaa2026".encode(), "static_salt_admin".encode(), 100000
        ).hex(),
        "role": "admin",
        "api_key": "rrx_live_admin_secret_key_999",
        "is_active": True,
    },
    "forecaster": {
        "user_id": "usr_forecaster_001",
        "username": "forecaster",
        "salt": "static_salt_forecaster",
        "password_hash": hashlib.pbkdf2_hmac(
            "sha256", "Forecaster@2026".encode(), "static_salt_forecaster".encode(), 100000
        ).hex(),
        "role": "forecaster",
        "api_key": "rrx_live_forecaster_key_555",
        "is_active": True,
    },
    "public_user": {
        "user_id": "usr_public_001",
        "username": "public_user",
        "salt": "static_salt_public",
        "password_hash": hashlib.pbkdf2_hmac(
            "sha256", "Public@2026".encode(), "static_salt_public".encode(), 100000
        ).hex(),
        "role": "public",
        "api_key": "rrx_live_public_key_111",
        "is_active": True,
    },
}


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hashes password with PBKDF2-HMAC-SHA256 and salt."""
    if not salt:
        salt = secrets.token_hex(16)
    p_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()
    return p_hash, salt


def verify_password(plain_password: str, password_hash: str, salt: str) -> bool:
    """Verifies plain password against salt and expected hash."""
    expected_hash = hashlib.pbkdf2_hmac("sha256", plain_password.encode(), salt.encode(), 100000).hex()
    return secrets.compare_digest(expected_hash, password_hash)


def create_access_token(
    user_id: str,
    username: str,
    role: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Generates standard RFC 7519 compliant signed JWT access token."""
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "iss": "rain-repair-x-api-gateway",
    }
    encoded_jwt = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decodes and validates JWT token signature and expiration."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM], issuer="rain-repair-x-api-gateway")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired. Please re-authenticate.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user_optional(
    auth_header: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> UserPayload:
    """
    Optional authentication: returns authenticated UserPayload if valid JWT or API Key is provided,
    otherwise defaults to anonymous public user.
    """
    # 1. Check API Key
    if x_api_key:
        for u in DEFAULT_USERS.values():
            if u["api_key"] == x_api_key and u["is_active"]:
                return UserPayload(user_id=u["user_id"], username=u["username"], role=u["role"], is_active=True)

    # 2. Check JWT Bearer token
    if auth_header and auth_header.credentials:
        payload = decode_access_token(auth_header.credentials)
        return UserPayload(
            user_id=payload.get("sub", "anon"),
            username=payload.get("username", "anonymous"),
            role=payload.get("role", "public"),
            is_active=True,
        )

    # Default to public guest user
    return UserPayload(user_id="anon_guest", username="guest", role="public", is_active=True)


def get_current_user(
    auth_header: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> UserPayload:
    """
    Strict authentication dependency requiring either a valid Bearer token or X-API-Key.
    """
    user = get_current_user_optional(auth_header=auth_header, x_api_key=x_api_key)
    if user.user_id == "anon_guest":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a valid Bearer token or X-API-Key header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_role(min_role: RoleType):
    """
    Role-Based Access Control (RBAC) dependency factory enforcing role hierarchy:
    public (1) < forecaster (2) < admin (3).
    """
    def role_checker(user: UserPayload = Depends(get_current_user)) -> UserPayload:
        user_level = ROLE_HIERARCHY.get(user.role, 0)
        required_level = ROLE_HIERARCHY.get(min_role, 0)
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: '{min_role}', current role: '{user.role}'.",
            )
        return user
    return role_checker
