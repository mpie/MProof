"""API endpoints for managing skip markers."""
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from datetime import datetime, timezone
import re
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/skip-markers", tags=["skip-markers"])

# Regex compile cache
_regex_cache: dict[str, re.Pattern] = {}


class SkipMarkerBase(BaseModel):
    pattern: str
    description: Optional[str] = None
    is_regex: bool = False
    is_active: bool = True


class SkipMarkerCreate(SkipMarkerBase):
    pass


class SkipMarkerUpdate(BaseModel):
    pattern: Optional[str] = None
    description: Optional[str] = None
    is_regex: Optional[bool] = None
    is_active: Optional[bool] = None


class SkipMarker(SkipMarkerBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


@router.get("", response_model=List[SkipMarker])
async def list_skip_markers(active_only: bool = False):
    """List all skip markers."""
    from app.main import async_session_maker
    
    async with async_session_maker() as db:
        if active_only:
            query = text("SELECT * FROM skip_markers WHERE is_active = 1 ORDER BY id")
        else:
            query = text("SELECT * FROM skip_markers ORDER BY id")
        
        result = await db.execute(query)
        rows = result.fetchall()
        
        return [
            SkipMarker(
                id=row[0],
                pattern=row[1],
                description=row[2],
                is_regex=bool(row[3]),
                is_active=bool(row[4]),
                created_at=row[5] if row[5] else datetime.now(timezone.utc),
                updated_at=row[6] if row[6] else datetime.now(timezone.utc)
            )
            for row in rows
        ]


@router.post("", response_model=SkipMarker)
async def create_skip_marker(marker: SkipMarkerCreate):
    """Create a new skip marker."""
    from app.main import async_session_maker
    
    # Validate regex if applicable
    if marker.is_regex:
        try:
            if marker.pattern not in _regex_cache:
                _regex_cache[marker.pattern] = re.compile(marker.pattern)
        except re.error as e:
            logger.debug(f"Invalid regex pattern: {marker.pattern} - {e}")
            raise HTTPException(status_code=400, detail=f"Invalid regex pattern: {e}")
    
    async with async_session_maker() as db:
        now = datetime.now(timezone.utc)
        result = await db.execute(
            text("""
                INSERT INTO skip_markers (pattern, description, is_regex, is_active, created_at, updated_at)
                VALUES (:pattern, :description, :is_regex, :is_active, :created_at, :updated_at)
            """),
            {
                "pattern": marker.pattern,
                "description": marker.description,
                "is_regex": marker.is_regex,
                "is_active": marker.is_active,
                "created_at": now,
                "updated_at": now
            }
        )
        await db.commit()
        
        # Get the inserted row
        marker_id = result.lastrowid
        return SkipMarker(
            id=marker_id,
            pattern=marker.pattern,
            description=marker.description,
            is_regex=marker.is_regex,
            is_active=marker.is_active,
            created_at=now,
            updated_at=now
        )


@router.put("/{marker_id}", response_model=SkipMarker)
async def update_skip_marker(marker_id: int, update: SkipMarkerUpdate):
    """Update a skip marker."""
    from app.main import async_session_maker
    
    async with async_session_maker() as db:
        # Check if marker exists
        result = await db.execute(
            text("SELECT * FROM skip_markers WHERE id = :id"),
            {"id": marker_id}
        )
        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Skip marker not found")
        
        # Validate regex if updating pattern
        if update.is_regex or (update.pattern and row[3]):
            pattern = update.pattern or row[1]
            is_regex = update.is_regex if update.is_regex is not None else row[3]
            if is_regex:
                try:
                    if pattern not in _regex_cache:
                        _regex_cache[pattern] = re.compile(pattern)
                except re.error as e:
                    logger.debug(f"Invalid regex pattern: {pattern} - {e}")
                    raise HTTPException(status_code=400, detail=f"Invalid regex pattern: {e}")
        
        # Build update query
        updates = []
        params = {"id": marker_id, "updated_at": datetime.utcnow()}
        
        if update.pattern is not None:
            updates.append("pattern = :pattern")
            params["pattern"] = update.pattern
        if update.description is not None:
            updates.append("description = :description")
            params["description"] = update.description
        if update.is_regex is not None:
            updates.append("is_regex = :is_regex")
            params["is_regex"] = update.is_regex
        if update.is_active is not None:
            updates.append("is_active = :is_active")
            params["is_active"] = update.is_active
        
        updates.append("updated_at = :updated_at")
        
        if updates:
            query = text(f"UPDATE skip_markers SET {', '.join(updates)} WHERE id = :id")
            await db.execute(query, params)
            await db.commit()
        
        # Return updated marker
        result = await db.execute(
            text("SELECT * FROM skip_markers WHERE id = :id"),
            {"id": marker_id}
        )
        row = result.fetchone()
        
        return SkipMarker(
            id=row[0],
            pattern=row[1],
            description=row[2],
            is_regex=bool(row[3]),
            is_active=bool(row[4]),
            created_at=row[5] if row[5] else datetime.now(timezone.utc),
            updated_at=row[6] if row[6] else datetime.now(timezone.utc)
        )


@router.delete("/{marker_id}")
async def delete_skip_marker(marker_id: int):
    """Delete a skip marker."""
    from app.main import async_session_maker
    
    async with async_session_maker() as db:
        # Check if marker exists
        result = await db.execute(
            text("SELECT id FROM skip_markers WHERE id = :id"),
            {"id": marker_id}
        )
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="Skip marker not found")
        
        await db.execute(
            text("DELETE FROM skip_markers WHERE id = :id"),
            {"id": marker_id}
        )
        await db.commit()
        
        return {"ok": True, "deleted_id": marker_id}
