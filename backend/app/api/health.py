from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.llm_client import LLMClient
from app.models.schemas import HealthResponse, OllamaHealth, LLMHealth
from app.api.llm_settings import get_active_provider_from_db

router = APIRouter()


async def get_db():
    """Get database session."""
    import app.main as app_main
    async with app_main.async_session_maker() as session:
        yield session


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check endpoint that verifies LLM provider connectivity."""
    # Use the active provider from database, not default
    active_provider = await get_active_provider_from_db(db)
    llm_client = LLMClient(provider=active_provider)
    llm_reachable = await llm_client.check_health()

    return HealthResponse(
        ok=True,
        # Keep ollama field for backward compatibility
        ollama=OllamaHealth(
            reachable=llm_reachable,
            base_url=llm_client.base_url,
            model=llm_client.model
        ),
        # New llm field with provider info
        llm=LLMHealth(
            provider=llm_client.provider,
            reachable=llm_reachable,
            base_url=llm_client.base_url,
            model=llm_client.model
        )
    )