"""API endpoints for LLM settings management."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Literal, Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import httpx

from app.config import settings
from app.models.database import AppSetting

router = APIRouter()


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""
    provider: str
    base_url: str
    model: str
    timeout: float
    max_retries: int
    max_tokens: int
    is_active: bool
    is_reachable: Optional[bool] = None


class LLMSettingsResponse(BaseModel):
    """Response containing all LLM settings."""
    active_provider: str
    providers: Dict[str, LLMProviderConfig]


class SwitchProviderRequest(BaseModel):
    """Request to switch the active LLM provider."""
    provider: Literal["ollama", "vllm"]


class SwitchProviderResponse(BaseModel):
    """Response after switching provider."""
    success: bool
    active_provider: str
    message: str


async def get_db():
    """Get database session."""
    import app.main as app_main
    async with app_main.async_session_maker() as session:
        yield session


async def get_active_provider_from_db(db: AsyncSession) -> str:
    """Get the active LLM provider from database."""
    result = await db.execute(
        select(AppSetting).where(AppSetting.key == "llm_provider")
    )
    setting = result.scalar_one_or_none()
    if setting:
        return setting.value
    return "ollama"  # Default


async def set_active_provider_in_db(db: AsyncSession, provider: str) -> None:
    """Set the active LLM provider in database."""
    if provider not in ("ollama", "vllm"):
        raise ValueError(f"Invalid provider: {provider}. Must be 'ollama' or 'vllm'")
    
    result = await db.execute(
        select(AppSetting).where(AppSetting.key == "llm_provider")
    )
    setting = result.scalar_one_or_none()
    
    if setting:
        setting.value = provider
    else:
        from datetime import datetime
        setting = AppSetting(
            key="llm_provider",
            value=provider,
            description="Active LLM provider (ollama or vllm)",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(setting)
    
    await db.commit()


@router.get("/llm/settings", response_model=LLMSettingsResponse)
async def get_llm_settings(db: AsyncSession = Depends(get_db)):
    """Get current LLM settings for all providers."""
    active = await get_active_provider_from_db(db)
    
    # Get config for both providers
    ollama_config = settings.get_llm_config("ollama")
    vllm_config = settings.get_llm_config("vllm")
    
    return LLMSettingsResponse(
        active_provider=active,
        providers={
            "ollama": LLMProviderConfig(
                provider="ollama",
                base_url=ollama_config["base_url"],
                model=ollama_config["model"],
                timeout=ollama_config["timeout"],
                max_retries=ollama_config["max_retries"],
                max_tokens=ollama_config["max_tokens"],
                is_active=(active == "ollama"),
            ),
            "vllm": LLMProviderConfig(
                provider="vllm",
                base_url=vllm_config["base_url"],
                model=vllm_config["model"],
                timeout=vllm_config["timeout"],
                max_retries=vllm_config["max_retries"],
                max_tokens=vllm_config["max_tokens"],
                is_active=(active == "vllm"),
            ),
        }
    )


@router.get("/llm/health")
async def check_llm_health(db: AsyncSession = Depends(get_db), check_all: bool = False):
    """
    Check health of LLM providers.
    
    Args:
        check_all: If True, check both providers. If False (default), only check the active provider.
    """
    active = await get_active_provider_from_db(db)
    results = {}
    
    # Only check active provider by default, unless explicitly requested
    providers_to_check = ["ollama", "vllm"] if check_all else [active]
    
    for provider in providers_to_check:
        config = settings.get_llm_config(provider)
        base_url = config["base_url"].rstrip('/')
        
        try:
            if provider == "vllm":
                url = f"{base_url}/v1/models"
            else:
                url = f"{base_url}/api/tags"
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                
                # Check if model is available
                model = config["model"]
                if provider == "vllm":
                    models = [m.get("id") for m in data.get("data", [])]
                else:
                    models = [m.get("name") for m in data.get("models", [])]
                
                model_available = model in models
                
                results[provider] = {
                    "reachable": True,
                    "model_available": model_available,
                    "available_models": models,
                    "configured_model": model,
                    "base_url": base_url,
                    "is_active": (provider == active),
                }
        except Exception as e:
            results[provider] = {
                "reachable": False,
                "error": str(e),
                "base_url": base_url,
                "configured_model": config["model"],
                "is_active": (provider == active),
            }
    
    # If not checking all, still return info about the inactive provider (without checking it)
    if not check_all:
        inactive_provider = "vllm" if active == "ollama" else "ollama"
        inactive_config = settings.get_llm_config(inactive_provider)
        results[inactive_provider] = {
            "reachable": None,  # Not checked
            "model_available": None,
            "available_models": None,
            "configured_model": inactive_config["model"],
            "base_url": inactive_config["base_url"],
            "is_active": False,
        }
    
    return {
        "active_provider": active,
        "providers": results,
    }


@router.post("/llm/switch", response_model=SwitchProviderResponse)
async def switch_provider(request: SwitchProviderRequest, db: AsyncSession = Depends(get_db)):
    """Switch the active LLM provider (persisted in database)."""
    try:
        old_provider = await get_active_provider_from_db(db)
        await set_active_provider_in_db(db, request.provider)
        
        # Reinitialize the global LLM client (import here to avoid circular import)
        import app.main as app_main
        app_main.llm_client._refresh_config(request.provider)
        
        return SwitchProviderResponse(
            success=True,
            active_provider=request.provider,
            message=f"Switched from {old_provider} to {request.provider}",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch provider: {e}")
