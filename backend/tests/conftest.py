import pytest
import asyncio
from unittest.mock import AsyncMock
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.config import settings
from app.models.database import Base
from app.services.llm_client import LLMClient
from app.services.sse_service import SSEService


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(settings.database_url, echo=False)

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def db_session(test_engine):
    """Create test database session."""
    async_session = sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        yield session
        # Rollback any changes
        await session.rollback()


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    client = AsyncMock(spec=LLMClient)
    client.generate_json.return_value = {"test": "mocked_response"}
    client.check_health.return_value = True
    return client


@pytest.fixture
def mock_sse_service():
    """Mock SSE service for testing."""
    service = AsyncMock(spec=SSEService)
    return service