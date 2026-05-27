"""API Key management endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import Optional, List
from datetime import datetime, timezone
import secrets
import hashlib
from pydantic import BaseModel

from app.dependencies.auth import get_current_principal, Principal, UserRoleEnum

router = APIRouter()


class ApiKeyCreate(BaseModel):
    name: str
    scopes: Optional[List[str]] = None
    expires_days: Optional[int] = None


class ApiKeyResponse(BaseModel):
    id: int
    user_id: Optional[int]
    name: str
    client_id: str
    scopes: Optional[List[str]]
    is_active: bool
    last_used_at: Optional[str]
    expires_at: Optional[str]
    created_at: str


class ApiKeyCreatedResponse(ApiKeyResponse):
    client_secret: str


class ApiKeyUpdate(BaseModel):
    name: Optional[str] = None
    scopes: Optional[List[str]] = None
    is_active: Optional[bool] = None


def generate_client_id() -> str:
    return secrets.token_hex(16)


def generate_client_secret() -> str:
    return secrets.token_hex(32)


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def format_datetime(dt) -> Optional[str]:
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    if hasattr(dt, 'isoformat'):
        return dt.isoformat()
    return str(dt)


def _is_admin(principal: Principal) -> bool:
    return principal.role in (UserRoleEnum.admin, UserRoleEnum.super_admin)


@router.get("/api-keys", response_model=List[ApiKeyResponse])
async def list_api_keys(principal: Principal = Depends(get_current_principal)):
    """List API keys. Admins see all; regular users see only their own."""
    from app.main import async_session_maker
    import json

    async with async_session_maker() as session:
        if _is_admin(principal):
            result = await session.execute(
                text("""
                    SELECT id, user_id, name, client_id, scopes, is_active,
                           last_used_at, expires_at, created_at
                    FROM api_keys
                    ORDER BY created_at DESC
                """)
            )
        else:
            result = await session.execute(
                text("""
                    SELECT id, user_id, name, client_id, scopes, is_active,
                           last_used_at, expires_at, created_at
                    FROM api_keys
                    WHERE user_id = :uid
                    ORDER BY created_at DESC
                """),
                {"uid": principal.user_id},
            )
        keys = result.fetchall()

        result_list = []
        for row in keys:
            scopes = None
            if row.scopes:
                try:
                    scopes = json.loads(row.scopes) if isinstance(row.scopes, str) else row.scopes
                except (json.JSONDecodeError, TypeError):
                    scopes = None

            result_list.append(
                ApiKeyResponse(
                    id=row.id,
                    user_id=row.user_id,
                    name=row.name,
                    client_id=row.client_id,
                    scopes=scopes,
                    is_active=bool(row.is_active),
                    last_used_at=format_datetime(row.last_used_at),
                    expires_at=format_datetime(row.expires_at),
                    created_at=format_datetime(row.created_at),
                )
            )
        return result_list


@router.post("/api-keys", response_model=ApiKeyCreatedResponse)
async def create_api_key(
    data: ApiKeyCreate,
    principal: Principal = Depends(get_current_principal),
):
    """Create a new API key bound to the requesting user. Secret shown only once."""
    from app.main import async_session_maker
    import json
    import logging

    logger = logging.getLogger(__name__)

    client_id = generate_client_id()
    client_secret = generate_client_secret()
    secret_hash = hash_secret(client_secret)

    expires_at = None
    if data.expires_days:
        from datetime import timedelta
        expires_at = datetime.now(timezone.utc) + timedelta(days=data.expires_days)

    try:
        async with async_session_maker() as session:
            result = await session.execute(
                text("""
                    INSERT INTO api_keys (user_id, name, client_id, client_secret_hash, scopes, is_active, expires_at, created_at, updated_at)
                    VALUES (:user_id, :name, :client_id, :secret_hash, :scopes, :is_active, :expires_at, :created_at, :updated_at)
                """),
                {
                    "user_id": principal.user_id,
                    "name": data.name,
                    "client_id": client_id,
                    "secret_hash": secret_hash,
                    "scopes": json.dumps(data.scopes) if data.scopes else None,
                    "is_active": True,
                    "expires_at": expires_at,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            await session.commit()

            result = await session.execute(
                text("SELECT id, created_at FROM api_keys WHERE client_id = :client_id"),
                {"client_id": client_id}
            )
            row = result.fetchone()

            if not row:
                raise HTTPException(status_code=500, detail="Failed to retrieve created API key")

            return ApiKeyCreatedResponse(
                id=row.id,
                user_id=principal.user_id,
                name=data.name,
                client_id=client_id,
                client_secret=client_secret,
                scopes=data.scopes,
                is_active=True,
                last_used_at=None,
                expires_at=expires_at.isoformat() if expires_at else None,
                created_at=format_datetime(row.created_at),
            )
    except Exception as e:
        logger.error(f"Error creating API key: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create API key: {str(e)}")


def _assert_key_access(row, principal: Principal):
    """Raises 403 if non-admin tries to touch another user's key."""
    if not _is_admin(principal) and row.user_id != principal.user_id:
        raise HTTPException(status_code=403, detail="Not your API key")


@router.put("/api-keys/{key_id}", response_model=ApiKeyResponse)
async def update_api_key(
    key_id: int,
    data: ApiKeyUpdate,
    principal: Principal = Depends(get_current_principal),
):
    from app.main import async_session_maker
    import json

    async with async_session_maker() as session:
        result = await session.execute(
            text("SELECT id, user_id FROM api_keys WHERE id = :id"),
            {"id": key_id}
        )
        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="API key not found")
        _assert_key_access(row, principal)

        updates = []
        params = {"id": key_id, "updated_at": datetime.now(timezone.utc)}

        if data.name is not None:
            updates.append("name = :name")
            params["name"] = data.name
        if data.scopes is not None:
            updates.append("scopes = :scopes")
            params["scopes"] = json.dumps(data.scopes)
        if data.is_active is not None:
            updates.append("is_active = :is_active")
            params["is_active"] = data.is_active

        updates.append("updated_at = :updated_at")

        if updates:
            await session.execute(
                text(f"UPDATE api_keys SET {', '.join(updates)} WHERE id = :id"),
                params
            )
            await session.commit()

        result = await session.execute(
            text("""
                SELECT id, user_id, name, client_id, scopes, is_active,
                       last_used_at, expires_at, created_at
                FROM api_keys WHERE id = :id
            """),
            {"id": key_id}
        )
        row = result.fetchone()

        scopes = None
        if row.scopes:
            try:
                scopes = json.loads(row.scopes) if isinstance(row.scopes, str) else row.scopes
            except (json.JSONDecodeError, TypeError):
                scopes = None

        return ApiKeyResponse(
            id=row.id,
            user_id=row.user_id,
            name=row.name,
            client_id=row.client_id,
            scopes=scopes,
            is_active=bool(row.is_active),
            last_used_at=format_datetime(row.last_used_at),
            expires_at=format_datetime(row.expires_at),
            created_at=format_datetime(row.created_at),
        )


@router.delete("/api-keys/{key_id}")
async def delete_api_key(
    key_id: int,
    principal: Principal = Depends(get_current_principal),
):
    from app.main import async_session_maker

    async with async_session_maker() as session:
        result = await session.execute(
            text("SELECT id, user_id FROM api_keys WHERE id = :id"),
            {"id": key_id}
        )
        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="API key not found")
        _assert_key_access(row, principal)

        await session.execute(
            text("DELETE FROM api_keys WHERE id = :id"),
            {"id": key_id}
        )
        await session.commit()

        return {"ok": True, "message": "API key deleted"}


@router.post("/api-keys/{key_id}/regenerate", response_model=ApiKeyCreatedResponse)
async def regenerate_api_key(
    key_id: int,
    principal: Principal = Depends(get_current_principal),
):
    """Regenerate secret for an API key. New secret shown only once."""
    from app.main import async_session_maker

    async with async_session_maker() as session:
        result = await session.execute(
            text("SELECT * FROM api_keys WHERE id = :id"),
            {"id": key_id}
        )
        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="API key not found")
        _assert_key_access(row, principal)

        client_secret = generate_client_secret()
        secret_hash = hash_secret(client_secret)

        await session.execute(
            text("""
                UPDATE api_keys
                SET client_secret_hash = :secret_hash, updated_at = :updated_at
                WHERE id = :id
            """),
            {
                "id": key_id,
                "secret_hash": secret_hash,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        await session.commit()

        import json
        scopes = None
        if row.scopes:
            try:
                scopes = json.loads(row.scopes) if isinstance(row.scopes, str) else row.scopes
            except (json.JSONDecodeError, TypeError):
                scopes = None

        return ApiKeyCreatedResponse(
            id=row.id,
            user_id=row.user_id,
            name=row.name,
            client_id=row.client_id,
            client_secret=client_secret,
            scopes=scopes,
            is_active=bool(row.is_active),
            last_used_at=format_datetime(row.last_used_at),
            expires_at=format_datetime(row.expires_at),
            created_at=format_datetime(row.created_at),
        )
