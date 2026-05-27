"""
Authentication and authorization dependencies.

Two auth methods are supported:
  1. JWT Bearer token  → used by the frontend (cookie-based storage)
  2. API key (X-API-Key or Authorization: ApiKey <key>) → used by external integrations

Both resolve to a Principal that carries the user id and role.
API keys get role "user" by default (they can only do what a user can do).
"""
import hashlib
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import UserRoleEnum
from app.services.auth_service import decode_access_token

_bearer = HTTPBearer(auto_error=False)


@dataclass
class Principal:
    user_id: Optional[int]
    role: UserRoleEnum
    email: Optional[str] = None
    name: Optional[str] = None
    is_api_key: bool = False


async def _get_session():
    from app.main import async_session_maker
    async with async_session_maker() as session:
        yield session


async def get_current_principal(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: AsyncSession = Depends(_get_session),
) -> Principal:
    """Try JWT bearer → API key → 401."""

    # 1. JWT Bearer (Authorization header or ?token= query param for SSE)
    token_str: Optional[str] = None
    if credentials and credentials.scheme.lower() == "bearer":
        token_str = credentials.credentials
    if not token_str:
        token_str = request.query_params.get("token")

    if token_str:
        payload = decode_access_token(token_str)
        if payload:
            user_id = int(payload["sub"])
            role = UserRoleEnum(payload["role"])
            row = (await db.execute(
                text("SELECT email, name, is_active FROM users WHERE id = :id"),
                {"id": user_id},
            )).fetchone()
            if row and row.is_active:
                return Principal(user_id=user_id, role=role, email=row.email, name=row.name)

    # 2. API Key (X-API-Key header or Authorization: ApiKey <key>)
    api_key_value: Optional[str] = request.headers.get("X-API-Key")
    if not api_key_value and credentials and credentials.scheme.lower() == "apikey":
        api_key_value = credentials.credentials

    if api_key_value:
        key_hash = hashlib.sha256(api_key_value.encode()).hexdigest()
        row = (await db.execute(
            text("""SELECT id, user_id, scopes, is_active FROM api_keys
                    WHERE client_secret_hash = :h AND is_active = 1
                    AND (expires_at IS NULL OR expires_at > NOW())"""),
            {"h": key_hash},
        )).fetchone()
        if row:
            await db.execute(
                text("UPDATE api_keys SET last_used_at = NOW() WHERE id = :id"),
                {"id": row.id},
            )
            await db.commit()
            return Principal(user_id=row.user_id, role=UserRoleEnum.user, is_api_key=True)

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")


def require_user(principal: Principal = Depends(get_current_principal)) -> Principal:
    return principal


def require_admin(principal: Principal = Depends(get_current_principal)) -> Principal:
    if principal.role not in (UserRoleEnum.admin, UserRoleEnum.super_admin):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin required")
    return principal


def require_super_admin(principal: Principal = Depends(get_current_principal)) -> Principal:
    if principal.role != UserRoleEnum.super_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Super admin required")
    return principal
