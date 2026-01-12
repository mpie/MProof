"""API endpoints for application settings."""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from datetime import datetime, timezone

from app.models.database import AppSetting

router = APIRouter()


class AppSettingResponse(BaseModel):
    key: str
    value: str
    description: Optional[str] = None


class AppSettingUpdate(BaseModel):
    value: str
    description: Optional[str] = None


async def get_db():
    from app.main import async_session_maker
    async with async_session_maker() as session:
        yield session


@router.get("/app-settings", response_model=list[AppSettingResponse])
async def get_app_settings(db: AsyncSession = Depends(get_db)):
    """Get all application settings."""
    result = await db.execute(select(AppSetting))
    settings = result.scalars().all()
    return [
        AppSettingResponse(
            key=s.key,
            value=s.value,
            description=s.description
        )
        for s in settings
    ]


@router.get("/app-settings/{key}", response_model=AppSettingResponse)
async def get_app_setting(key: str, db: AsyncSession = Depends(get_db)):
    """Get a specific application setting."""
    result = await db.execute(select(AppSetting).where(AppSetting.key == key))
    setting = result.scalar_one_or_none()
    if not setting:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
    return AppSettingResponse(
        key=setting.key,
        value=setting.value,
        description=setting.description
    )


@router.put("/app-settings/{key}", response_model=AppSettingResponse)
async def update_app_setting(
    key: str,
    update: AppSettingUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update an application setting."""
    result = await db.execute(select(AppSetting).where(AppSetting.key == key))
    setting = result.scalar_one_or_none()
    
    now = datetime.now(timezone.utc)
    
    if setting:
        setting.value = update.value
        if update.description is not None:
            setting.description = update.description
        setting.updated_at = now
    else:
        setting = AppSetting(
            key=key,
            value=update.value,
            description=update.description,
            created_at=now,
            updated_at=now
        )
        db.add(setting)
    
    await db.commit()
    await db.refresh(setting)
    
    return AppSettingResponse(
        key=setting.key,
        value=setting.value,
        description=setting.description
    )
