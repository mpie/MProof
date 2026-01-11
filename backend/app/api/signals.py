"""
Classification Signals API.

CRUD for user-defined signals + read-only for built-in signals.
"""
from __future__ import annotations

import json as json_module
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.services.signal_engine import Signal, compute_all_signals

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Request/Response Schemas
# =============================================================================

class SignalCreate(BaseModel):
    """Schema for creating a user signal."""

    key: str = Field(min_length=1, max_length=100, pattern=r"^[a-z_][a-z0-9_]*$")
    label: str = Field(min_length=1, max_length=255)
    description: Optional[str] = None
    signal_type: str = Field(pattern=r"^(boolean|count)$")
    compute_kind: str = Field(pattern=r"^(keyword_set|regex_set)$")
    config_json: dict = Field(description="Config with keywords/patterns and match_mode")


class SignalUpdate(BaseModel):
    """Schema for updating a user signal."""

    label: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    config_json: Optional[dict] = None


class SignalResponse(BaseModel):
    """Signal response."""

    id: int
    key: str
    label: str
    description: Optional[str]
    signal_type: str
    source: str  # 'builtin' or 'user'
    compute_kind: str  # 'builtin', 'keyword_set', 'regex_set'
    config_json: Optional[dict]
    created_at: str
    updated_at: str


class SignalListResponse(BaseModel):
    """List of signals."""

    signals: list[SignalResponse]
    builtin_count: int
    user_count: int


class SignalTestRequest(BaseModel):
    """Request for testing signal computation."""

    text: str = Field(min_length=1)


class SignalTestResponse(BaseModel):
    """Response for signal test."""

    signals: dict[str, object]
    text_length: int
    line_count: int


# =============================================================================
# Helper Functions
# =============================================================================

def _format_datetime(dt) -> str:
    """Format a datetime value that might be a string or datetime object."""
    if not dt:
        return ""
    if hasattr(dt, 'isoformat'):
        return dt.isoformat()
    return str(dt)


def _db_row_to_response(row) -> SignalResponse:
    """Convert a database row to SignalResponse, handling schema differences."""
    # Handle both old schema (is_system, compute_method) and new schema (source, compute_kind)
    source = "builtin" if getattr(row, 'is_system', False) else "user"
    if hasattr(row, 'source'):
        source = row.source
    
    compute_kind = getattr(row, 'compute_method', 'builtin')
    if hasattr(row, 'compute_kind'):
        compute_kind = row.compute_kind
    
    # Parse config_json if it's a string
    config = row.config_json
    if isinstance(config, str):
        try:
            config = json_module.loads(config)
        except (json_module.JSONDecodeError, TypeError):
            config = None
    
    return SignalResponse(
        id=row.id,
        key=row.key,
        label=row.label,
        description=row.description,
        signal_type=row.signal_type,
        source=source,
        compute_kind=compute_kind,
        config_json=config,
        created_at=_format_datetime(row.created_at),
        updated_at=_format_datetime(row.updated_at),
    )


async def load_all_signals_from_db() -> list[Signal]:
    """Load all signal definitions from database."""
    from app.main import async_session_maker

    async with async_session_maker() as session:
        result = await session.execute(
            text("""
                SELECT key, label, description, signal_type, 
                       COALESCE(compute_method, 'builtin') as compute_kind,
                       CASE WHEN is_system = 1 THEN 'builtin' ELSE 'user' END as source,
                       config_json
                FROM classification_signals
                ORDER BY is_system DESC, key ASC
            """)
        )
        rows = result.fetchall()

        signals = []
        for row in rows:
            # Parse config_json if it's a string
            config = row.config_json
            if isinstance(config, str):
                try:
                    config = json_module.loads(config)
                except (json_module.JSONDecodeError, TypeError):
                    config = None

            signals.append(Signal(
                key=row.key,
                label=row.label,
                description=row.description or "",
                signal_type=row.signal_type,
                source=row.source,
                compute_kind=row.compute_kind,
                config_json=config,
            ))
        
        return signals


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/signals", response_model=SignalListResponse)
async def list_signals():
    """List all signals (built-in + user-defined)."""
    from app.main import async_session_maker

    async with async_session_maker() as session:
        result = await session.execute(
            text("""
                SELECT id, key, label, description, signal_type, 
                       COALESCE(compute_method, 'builtin') as compute_method,
                       is_system, config_json, created_at, updated_at
                FROM classification_signals
                ORDER BY is_system DESC, key ASC
            """)
        )
        rows = result.fetchall()

        signals = []
        builtin_count = 0
        user_count = 0

        for row in rows:
            is_system = bool(row.is_system) if row.is_system is not None else False
            if is_system:
                builtin_count += 1
            else:
                user_count += 1

            # Parse config_json if it's a string
            config = row.config_json
            if isinstance(config, str):
                try:
                    config = json_module.loads(config)
                except (json_module.JSONDecodeError, TypeError):
                    config = None

            signals.append(SignalResponse(
                id=row.id,
                key=row.key,
                label=row.label,
                description=row.description,
                signal_type=row.signal_type,
                source="builtin" if is_system else "user",
                compute_kind=row.compute_method or "builtin",
                config_json=config,
                created_at=_format_datetime(row.created_at),
                updated_at=_format_datetime(row.updated_at),
            ))

        return SignalListResponse(
            signals=signals,
            builtin_count=builtin_count,
            user_count=user_count,
        )


@router.get("/signals/{key}", response_model=SignalResponse)
async def get_signal(key: str):
    """Get a specific signal by key."""
    from app.main import async_session_maker

    async with async_session_maker() as session:
        result = await session.execute(
            text("""
                SELECT id, key, label, description, signal_type,
                       COALESCE(compute_method, 'builtin') as compute_method,
                       is_system, config_json, created_at, updated_at
                FROM classification_signals
                WHERE key = :key
            """),
            {"key": key}
        )
        row = result.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"Signal '{key}' not found")

        is_system = bool(row.is_system) if row.is_system is not None else False
        
        config = row.config_json
        if isinstance(config, str):
            try:
                config = json_module.loads(config)
            except (json_module.JSONDecodeError, TypeError):
                config = None

        return SignalResponse(
            id=row.id,
            key=row.key,
            label=row.label,
            description=row.description,
            signal_type=row.signal_type,
            source="builtin" if is_system else "user",
            compute_kind=row.compute_method or "builtin",
            config_json=config,
            created_at=_format_datetime(row.created_at),
            updated_at=_format_datetime(row.updated_at),
        )


@router.post("/signals", response_model=SignalResponse, status_code=201)
async def create_signal(signal: SignalCreate):
    """Create a new user-defined signal."""
    from app.main import async_session_maker

    # Validate config
    if signal.compute_kind == "keyword_set":
        if "keywords" not in signal.config_json or not signal.config_json["keywords"]:
            raise HTTPException(
                status_code=400,
                detail="keyword_set signalen vereisen 'keywords' array in config_json"
            )
    elif signal.compute_kind == "regex_set":
        if "patterns" not in signal.config_json or not signal.config_json["patterns"]:
            raise HTTPException(
                status_code=400,
                detail="regex_set signalen vereisen 'patterns' array in config_json"
            )

    async with async_session_maker() as session:
        # Check if key exists
        result = await session.execute(
            text("SELECT id FROM classification_signals WHERE key = :key"),
            {"key": signal.key}
        )
        if result.fetchone():
            raise HTTPException(
                status_code=409,
                detail=f"Signal with key '{signal.key}' already exists"
            )

        # Insert (using old schema column names for compatibility)
        config_str = json_module.dumps(signal.config_json)
        await session.execute(
            text("""
                INSERT INTO classification_signals
                (key, label, description, signal_type, compute_method, config_json, is_system, created_at, updated_at)
                VALUES (:key, :label, :description, :signal_type, :compute_kind, :config_json, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """),
            {
                "key": signal.key,
                "label": signal.label,
                "description": signal.description,
                "signal_type": signal.signal_type,
                "compute_kind": signal.compute_kind,
                "config_json": config_str,
            }
        )
        await session.commit()

        # Fetch created
        result = await session.execute(
            text("""
                SELECT id, key, label, description, signal_type,
                       COALESCE(compute_method, 'builtin') as compute_method,
                       is_system, config_json, created_at, updated_at
                FROM classification_signals
                WHERE key = :key
            """),
            {"key": signal.key}
        )
        row = result.fetchone()

        config = row.config_json
        if isinstance(config, str):
            try:
                config = json_module.loads(config)
            except (json_module.JSONDecodeError, TypeError):
                config = None

        return SignalResponse(
            id=row.id,
            key=row.key,
            label=row.label,
            description=row.description,
            signal_type=row.signal_type,
            source="user",
            compute_kind=row.compute_method or "builtin",
            config_json=config,
            created_at=_format_datetime(row.created_at),
            updated_at=_format_datetime(row.updated_at),
        )


@router.put("/signals/{key}", response_model=SignalResponse)
async def update_signal(key: str, update: SignalUpdate):
    """Update a user-defined signal."""
    from app.main import async_session_maker

    async with async_session_maker() as session:
        # Check if exists and is user signal
        result = await session.execute(
            text("SELECT id, is_system, compute_method FROM classification_signals WHERE key = :key"),
            {"key": key}
        )
        row = result.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"Signal '{key}' not found")

        if row.is_system:
            raise HTTPException(
                status_code=403,
                detail="Built-in signals cannot be modified"
            )

        # Build update
        updates = []
        params = {"key": key}

        if update.label is not None:
            updates.append("label = :label")
            params["label"] = update.label

        if update.description is not None:
            updates.append("description = :description")
            params["description"] = update.description

        if update.config_json is not None:
            # Validate config
            compute_kind = row.compute_method or "keyword_set"
            if compute_kind == "keyword_set" and "keywords" not in update.config_json:
                raise HTTPException(
                    status_code=400,
                    detail="keyword_set signalen vereisen 'keywords' array"
                )
            elif compute_kind == "regex_set" and "patterns" not in update.config_json:
                raise HTTPException(
                    status_code=400,
                    detail="regex_set signalen vereisen 'patterns' array"
                )

            updates.append("config_json = :config_json")
            params["config_json"] = json_module.dumps(update.config_json)

        if not updates:
            raise HTTPException(status_code=400, detail="Geen velden om bij te werken")

        updates.append("updated_at = CURRENT_TIMESTAMP")

        await session.execute(
            text(f"UPDATE classification_signals SET {', '.join(updates)} WHERE key = :key"),
            params
        )
        await session.commit()

        # Fetch updated
        result = await session.execute(
            text("""
                SELECT id, key, label, description, signal_type,
                       COALESCE(compute_method, 'builtin') as compute_method,
                       is_system, config_json, created_at, updated_at
                FROM classification_signals
                WHERE key = :key
            """),
            {"key": key}
        )
        row = result.fetchone()

        config = row.config_json
        if isinstance(config, str):
            try:
                config = json_module.loads(config)
            except (json_module.JSONDecodeError, TypeError):
                config = None

        return SignalResponse(
            id=row.id,
            key=row.key,
            label=row.label,
            description=row.description,
            signal_type=row.signal_type,
            source="user",
            compute_kind=row.compute_method or "builtin",
            config_json=config,
            created_at=_format_datetime(row.created_at),
            updated_at=_format_datetime(row.updated_at),
        )


@router.delete("/signals/{key}", status_code=204)
async def delete_signal(key: str):
    """Delete a user-defined signal."""
    from app.main import async_session_maker

    async with async_session_maker() as session:
        result = await session.execute(
            text("SELECT id, is_system FROM classification_signals WHERE key = :key"),
            {"key": key}
        )
        row = result.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"Signal '{key}' not found")

        if row.is_system:
            raise HTTPException(
                status_code=403,
                detail="Built-in signals cannot be deleted"
            )

        await session.execute(
            text("DELETE FROM classification_signals WHERE key = :key"),
            {"key": key}
        )
        await session.commit()


@router.post("/signals/test", response_model=SignalTestResponse)
async def test_signals(request: SignalTestRequest):
    """Test signal computation on sample text."""
    signals = await load_all_signals_from_db()
    
    # If no signals in DB, use builtin defaults
    if not signals:
        from app.services.signal_engine import get_builtin_signals
        signals = get_builtin_signals()
    
    computed = compute_all_signals(request.text, signals)

    return SignalTestResponse(
        signals=computed.values,
        text_length=computed.text_length,
        line_count=computed.line_count,
    )
