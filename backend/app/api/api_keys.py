"""API Key management endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import Optional, List
from datetime import datetime, timezone
import secrets
import hashlib
from pydantic import BaseModel

router = APIRouter()


class ApiKeyCreate(BaseModel):
    name: str
    scopes: Optional[List[str]] = None
    expires_days: Optional[int] = None  # None = never expires


class ApiKeyResponse(BaseModel):
    id: int
    name: str
    client_id: str
    scopes: Optional[List[str]]
    is_active: bool
    last_used_at: Optional[str]
    expires_at: Optional[str]
    created_at: str


class ApiKeyCreatedResponse(ApiKeyResponse):
    """Response when a new key is created - includes the secret (only shown once)."""
    client_secret: str


class ApiKeyUpdate(BaseModel):
    name: Optional[str] = None
    scopes: Optional[List[str]] = None
    is_active: Optional[bool] = None


def generate_client_id() -> str:
    """Generate a random client ID (16 bytes hex = 32 chars)."""
    return secrets.token_hex(16)


def generate_client_secret() -> str:
    """Generate a random client secret (32 bytes hex = 64 chars)."""
    return secrets.token_hex(32)


def hash_secret(secret: str) -> str:
    """Hash the client secret using SHA256."""
    return hashlib.sha256(secret.encode()).hexdigest()


def format_datetime(dt) -> Optional[str]:
    """Format datetime to ISO string, handling both datetime objects and strings."""
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    if hasattr(dt, 'isoformat'):
        return dt.isoformat()
    return str(dt)


@router.get("/api-keys", response_model=List[ApiKeyResponse])
async def list_api_keys():
    """List all API keys (without secrets)."""
    from app.main import async_session_maker
    import json
    
    async with async_session_maker() as session:
        result = await session.execute(
            text("""
                SELECT id, name, client_id, scopes, is_active, 
                       last_used_at, expires_at, created_at
                FROM api_keys
                ORDER BY created_at DESC
            """)
        )
        keys = result.fetchall()
        
        result_list = []
        for row in keys:
            # Parse scopes JSON if present
            scopes = None
            if row.scopes:
                try:
                    scopes = json.loads(row.scopes) if isinstance(row.scopes, str) else row.scopes
                except (json.JSONDecodeError, TypeError):
                    scopes = None
            
            result_list.append(
                ApiKeyResponse(
                    id=row.id,
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
async def create_api_key(data: ApiKeyCreate):
    """Create a new API key. The secret is only returned once!"""
    from app.main import async_session_maker
    import json
    import logging
    
    logger = logging.getLogger(__name__)
    
    client_id = generate_client_id()
    client_secret = generate_client_secret()
    secret_hash = hash_secret(client_secret)
    
    # Calculate expiration if specified
    expires_at = None
    if data.expires_days:
        from datetime import timedelta
        expires_at = datetime.now(timezone.utc) + timedelta(days=data.expires_days)
    
    try:
        async with async_session_maker() as session:
            result = await session.execute(
                text("""
                    INSERT INTO api_keys (name, client_id, client_secret_hash, scopes, is_active, expires_at, created_at, updated_at)
                    VALUES (:name, :client_id, :secret_hash, :scopes, :is_active, :expires_at, :created_at, :updated_at)
                """),
                {
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
            
            # Get the created key
            result = await session.execute(
                text("SELECT id, created_at FROM api_keys WHERE client_id = :client_id"),
                {"client_id": client_id}
            )
            row = result.fetchone()
            
            if not row:
                raise HTTPException(status_code=500, detail="Failed to retrieve created API key")
            
            return ApiKeyCreatedResponse(
                id=row.id,
                name=data.name,
                client_id=client_id,
                client_secret=client_secret,  # Only returned once!
                scopes=data.scopes,
                is_active=True,
                last_used_at=None,
                expires_at=expires_at.isoformat() if expires_at else None,
                created_at=format_datetime(row.created_at),
            )
    except Exception as e:
        logger.error(f"Error creating API key: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create API key: {str(e)}")


@router.put("/api-keys/{key_id}", response_model=ApiKeyResponse)
async def update_api_key(key_id: int, data: ApiKeyUpdate):
    """Update an API key (name, scopes, or active status)."""
    from app.main import async_session_maker
    import json
    
    async with async_session_maker() as session:
        # Check if key exists
        result = await session.execute(
            text("SELECT id FROM api_keys WHERE id = :id"),
            {"id": key_id}
        )
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="API key not found")
        
        # Build update query
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
        
        # Return updated key
        result = await session.execute(
            text("""
                SELECT id, name, client_id, scopes, is_active, 
                       last_used_at, expires_at, created_at
                FROM api_keys WHERE id = :id
            """),
            {"id": key_id}
        )
        row = result.fetchone()
        
        # Parse scopes JSON if present
        scopes = None
        if row.scopes:
            try:
                scopes = json.loads(row.scopes) if isinstance(row.scopes, str) else row.scopes
            except (json.JSONDecodeError, TypeError):
                scopes = None
        
        return ApiKeyResponse(
            id=row.id,
            name=row.name,
            client_id=row.client_id,
            scopes=scopes,
            is_active=bool(row.is_active),
            last_used_at=format_datetime(row.last_used_at),
            expires_at=format_datetime(row.expires_at),
            created_at=format_datetime(row.created_at),
        )


@router.delete("/api-keys/{key_id}")
async def delete_api_key(key_id: int):
    """Delete an API key."""
    from app.main import async_session_maker
    
    async with async_session_maker() as session:
        result = await session.execute(
            text("SELECT id FROM api_keys WHERE id = :id"),
            {"id": key_id}
        )
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="API key not found")
        
        await session.execute(
            text("DELETE FROM api_keys WHERE id = :id"),
            {"id": key_id}
        )
        await session.commit()
        
        return {"ok": True, "message": "API key deleted"}


@router.post("/api-keys/{key_id}/regenerate", response_model=ApiKeyCreatedResponse)
async def regenerate_api_key(key_id: int):
    """Regenerate the secret for an API key. The new secret is only returned once!"""
    from app.main import async_session_maker
    
    async with async_session_maker() as session:
        # Check if key exists
        result = await session.execute(
            text("SELECT * FROM api_keys WHERE id = :id"),
            {"id": key_id}
        )
        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="API key not found")
        
        # Generate new secret
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
        # Parse scopes JSON if present
        scopes = None
        if row.scopes:
            try:
                scopes = json.loads(row.scopes) if isinstance(row.scopes, str) else row.scopes
            except (json.JSONDecodeError, TypeError):
                scopes = None
        
        return ApiKeyCreatedResponse(
            id=row.id,
            name=row.name,
            client_id=row.client_id,
            client_secret=client_secret,  # Only returned once!
            scopes=scopes,
            is_active=bool(row.is_active),
            last_used_at=format_datetime(row.last_used_at),
            expires_at=format_datetime(row.expires_at),
            created_at=format_datetime(row.created_at),
        )
