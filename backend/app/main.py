import warnings
# Suppress Pydantic warning about "model_name" conflicting with protected namespace
# This must be done BEFORE importing FastAPI/Pydantic
warnings.filterwarnings("ignore", message=".*Field.*has conflict with protected namespace.*model_.*", category=UserWarning)

# Suppress urllib3 OpenSSL warning on macOS (LibreSSL vs OpenSSL compatibility)
# This is a known issue: urllib3 v2 requires OpenSSL 1.1.1+, but macOS uses LibreSSL
# The warning is harmless - httpx/urllib3 still works correctly with LibreSSL
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    # urllib3 might not be installed or version doesn't have this warning
    pass

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import logging
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models.database import Base
from app.services.llm_client import LLMClient
from app.services.sse_service import SSEService
from app.services.job_queue import JobQueue
from app.api import (
    health, subjects, documents, document_types,
    upload, sse, queue, classifier, api_keys, skip_markers, mcp,
    classification_policy, signals, llm_settings, app_settings
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services
llm_client = LLMClient()
sse_service = SSEService()
job_queue = None

# Database setup
engine = create_async_engine(settings.database_url, echo=False)
async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global job_queue

    # Note: Database tables are created via Alembic migrations
    # Run 'alembic upgrade head' to create/update tables
    # This startup code does NOT create tables - migrations handle that

    # Load LLM provider from database
    await load_llm_provider_from_db()

    # Initialize job queue
    global job_queue
    job_queue = JobQueue(async_session_maker, llm_client, sse_service)
    await job_queue.start()
    logger.info(f"Job queue initialized: {job_queue}")

    # Seed initial data
    await seed_initial_data()

    logger.info("Application started")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    
    # Cancel training tasks gracefully
    try:
        from app.services.doc_type_classifier import classifier_service
        from app.services.bert_classifier import bert_classifier_service
        
        classifier = classifier_service()
        if hasattr(classifier, '_train_task') and classifier._train_task and not classifier._train_task.done():
            logger.info("Cancelling classifier training task...")
            classifier._cancelled.set()  # Signal sync thread to stop first
            classifier._train_task.cancel()
            try:
                await asyncio.wait_for(classifier._train_task, timeout=3.0)
                logger.info("Classifier training task cancelled successfully")
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.warning("Classifier training task cancellation timed out or was cancelled")
        
        bert_classifier = bert_classifier_service()
        if hasattr(bert_classifier, '_train_task') and bert_classifier._train_task and not bert_classifier._train_task.done():
            logger.info("Cancelling BERT training task...")
            if hasattr(bert_classifier, '_cancelled'):
                bert_classifier._cancelled.set()  # Signal sync thread to stop first
            bert_classifier._train_task.cancel()
            try:
                await asyncio.wait_for(bert_classifier._train_task, timeout=3.0)
                logger.info("BERT training task cancelled successfully")
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.warning("BERT training task cancellation timed out or was cancelled")
    except Exception as e:
        logger.warning(f"Error during training task cancellation: {e}")
    
    if job_queue:
        await job_queue.stop()

    await engine.dispose()
    logger.info("Application shutdown")


async def load_llm_provider_from_db():
    """Load the active LLM provider from database and configure LLM client."""
    try:
        from sqlalchemy import select
        from app.models.database import AppSetting
        
        async with async_session_maker() as session:
            result = await session.execute(
                select(AppSetting).where(AppSetting.key == "llm_provider")
            )
            setting = result.scalar_one_or_none()
            
            if setting and setting.value in ("ollama", "vllm"):
                provider = setting.value
                logger.info(f"Loaded LLM provider from database: {provider}")
                llm_client._refresh_config(provider)
            else:
                logger.info("No LLM provider setting in database, using default: ollama")
    except Exception as e:
        logger.warning(f"Failed to load LLM provider from database: {e}. Using default.")


async def seed_initial_data():
    """Seed initial document types and fields."""
    try:
        async with async_session_maker() as session:
            from sqlalchemy import text
            from datetime import datetime, timezone
            
            # Check if already seeded
            result = await session.execute(text("SELECT COUNT(*) FROM document_types"))
            count = result.scalar()
            if count > 0:
                logger.info("Database already seeded")
                return

            logger.info("Seeding initial document types...")
            now = datetime.now(timezone.utc)

            # Insert document types
            doc_type_count = 0
            field_count = 0

            # Insert invoice type
            result = await session.execute(
                text("SELECT id FROM document_types WHERE slug = :slug"),
                {"slug": "invoice"}
            )
            if not result.fetchone():
                await session.execute(
                    text("""
                        INSERT INTO document_types (name, slug, description, created_at, updated_at)
                        VALUES (:name, :slug, :description, :created_at, :updated_at)
                    """),
                    {
                        "name": "Invoice",
                        "slug": "invoice",
                        "description": "Invoice document",
                        "created_at": now,
                        "updated_at": now
                    }
                )
                doc_type_count += 1

            # Insert bank_statement type
            result = await session.execute(
                text("SELECT id FROM document_types WHERE slug = :slug"),
                {"slug": "bank_statement"}
            )
            if not result.fetchone():
                await session.execute(
                    text("""
                        INSERT INTO document_types (name, slug, description, created_at, updated_at)
                        VALUES (:name, :slug, :description, :created_at, :updated_at)
                    """),
                    {
                        "name": "Bank Statement",
                        "slug": "bank_statement",
                        "description": "Bank statement document",
                        "created_at": now,
                        "updated_at": now
                    }
                )
                doc_type_count += 1

            await session.commit()

            # Get document type IDs for fields
            result = await session.execute(
                text("SELECT id FROM document_types WHERE slug = :slug"),
                {"slug": "invoice"}
            )
            invoice_row = result.fetchone()
            invoice_id = invoice_row[0] if invoice_row else None

            result = await session.execute(
                text("SELECT id FROM document_types WHERE slug = :slug"),
                {"slug": "bank_statement"}
            )
            bank_row = result.fetchone()
            bank_id = bank_row[0] if bank_row else None

            # Insert fields for invoice
            if invoice_id:
                fields = [
                    ("invoice_number", "Invoice Number", "string", True),
                    ("date", "Date", "date", True),
                    ("amount", "Amount", "money", False),
                    ("vat_amount", "VAT Amount", "money", False),
                ]
                for key, label, field_type, required in fields:
                    result = await session.execute(
                        text("SELECT id FROM document_type_fields WHERE document_type_id = :doc_type_id AND key = :key"),
                        {"doc_type_id": invoice_id, "key": key}
                    )
                    if not result.fetchone():
                        await session.execute(
                            text("""
                                INSERT INTO document_type_fields 
                                (document_type_id, key, label, field_type, required, created_at, updated_at)
                                VALUES (:document_type_id, :key, :label, :field_type, :required, :created_at, :updated_at)
                            """),
                            {
                                "document_type_id": invoice_id,
                                "key": key,
                                "label": label,
                                "field_type": field_type,
                                "required": required,
                                "created_at": now,
                                "updated_at": now
                            }
                        )
                        field_count += 1

            # Insert fields for bank_statement
            if bank_id:
                fields = [
                    ("account_number", "Account Number", "iban", True),
                    ("statement_date", "Statement Date", "date", True),
                    ("balance", "Balance", "money", False),
                    ("transaction_count", "Transaction Count", "number", False),
                ]
                for key, label, field_type, required in fields:
                    result = await session.execute(
                        text("SELECT id FROM document_type_fields WHERE document_type_id = :doc_type_id AND key = :key"),
                        {"doc_type_id": bank_id, "key": key}
                    )
                    if not result.fetchone():
                        await session.execute(
                            text("""
                                INSERT INTO document_type_fields 
                                (document_type_id, key, label, field_type, required, created_at, updated_at)
                                VALUES (:document_type_id, :key, :label, :field_type, :required, :created_at, :updated_at)
                            """),
                            {
                                "document_type_id": bank_id,
                                "key": key,
                                "label": label,
                                "field_type": field_type,
                                "required": required,
                                "created_at": now,
                                "updated_at": now
                            }
                        )
                        field_count += 1

            await session.commit()
            logger.info(f"Seed completed: {doc_type_count} document types, {field_count} fields")

    except Exception as e:
        logger.error(f"Failed to seed initial data: {e}")
        # Don't re-raise - allow app to start even if seeding fails


# Create FastAPI app
app = FastAPI(
    title="MProof API",
    description="MProof document analysis platform - OCR, classification, and metadata extraction",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler to ensure CORS headers on errors
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
        headers={
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
        headers={
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Dependency injection
async def get_db():
    async with async_session_maker() as session:
        yield session

def get_llm_client():
    return llm_client

def get_sse_service():
    return sse_service

def get_job_queue():
    return job_queue

# Include routers
app.include_router(
    health.router,
    prefix="/api",
    tags=["health"]
)

app.include_router(
    subjects.router,
    prefix="/api",
    tags=["subjects"]
)

app.include_router(
    documents.router,
    prefix="/api",
    tags=["documents"]
)

# Register classification_policy before document_types so /document-types/{slug}/policy is matched first
app.include_router(
    classification_policy.router,
    prefix="/api",
    tags=["classification-policy"]
)

app.include_router(
    signals.router,
    prefix="/api",
    tags=["signals"]
)

app.include_router(
    document_types.router,
    prefix="/api",
    tags=["document-types"]
)

app.include_router(
    upload.router,
    prefix="/api",
    tags=["upload"]
)

app.include_router(
    sse.router,
    prefix="/api",
    tags=["sse"]
)

app.include_router(
    queue.router,
    prefix="/api",
    tags=["queue"]
)

app.include_router(
    classifier.router,
    prefix="/api",
    tags=["classifier"]
)

app.include_router(
    api_keys.router,
    prefix="/api",
    tags=["api-keys"]
)

app.include_router(
    skip_markers.router,
    prefix="/api",
    tags=["skip-markers"]
)

app.include_router(
    llm_settings.router,
    prefix="/api",
    tags=["llm-settings"]
)

app.include_router(
    app_settings.router,
    prefix="/api",
    tags=["app-settings"]
)

# Register MCP router at root level (only /mcp, not /api/mcp)
app.include_router(
    mcp.router,
    prefix="",
    tags=["mcp"]
)


@app.get("/")
async def root():
    return {"message": "MProof API", "version": "1.0.0"}

@app.get("/test")
async def test():
    return {"status": "ok", "message": "Test endpoint works"}