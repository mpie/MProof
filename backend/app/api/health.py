from fastapi import APIRouter, Depends
from app.services.llm_client import LLMClient
from app.models.schemas import HealthResponse, OllamaHealth

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint that verifies Ollama connectivity."""
    llm_client = LLMClient()
    ollama_reachable = await llm_client.check_health()

    return HealthResponse(
        ok=True,
        ollama=OllamaHealth(
            reachable=ollama_reachable,
            base_url=llm_client.base_url,
            model=llm_client.model
        )
    )