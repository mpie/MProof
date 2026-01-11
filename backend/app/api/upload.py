from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import shutil
import hashlib
from pathlib import Path
import re
import unicodedata
from app.models.schemas import DocumentUploadResponse
from app.config import settings

router = APIRouter()

def _sanitize_filename(filename: str) -> str:
    # Remove null bytes first (prevents "embedded null byte" crashes).
    filename = (filename or "").replace("\x00", "")
    filename = unicodedata.normalize("NFKC", filename)

    # Drop any path components (defense-in-depth).
    filename = filename.split("/")[-1].split("\\")[-1]

    # Remove control characters and trim.
    filename = "".join(ch for ch in filename if ch.isprintable() and ch not in "\r\n\t")
    filename = re.sub(r"\s+", " ", filename).strip()

    # Replace problematic characters while keeping common filename chars.
    filename = re.sub(r"[^A-Za-z0-9.\- _()]+", "_", filename)

    # Avoid empty/hidden names.
    if not filename or filename in {".", ".."}:
        return "upload"

    # Keep it within a reasonable length (preserve extension).
    if len(filename) > 180:
        parts = filename.rsplit(".", 1)
        if len(parts) == 2 and parts[1]:
            base, ext = parts
            base = base[:160].rstrip(" ._-")
            ext = ext[:16]
            return f"{base}.{ext}" if base else f"upload.{ext}"
        return filename[:180]

    return filename


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    subject_id: int = Form(..., description="Subject ID to associate the document with"),
    file: UploadFile = File(..., description="Document file to upload"),
    model_name: str = Form(None, description="Optional model name for classification"),
    db: AsyncSession = Depends(lambda: None)  # Will be injected
):
    """Upload a document and initiate processing."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Upload request: subject_id={subject_id}, filename={file.filename}, content_type={file.content_type}")

    safe_filename = _sanitize_filename(file.filename or "")
    if not safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Validate file type
    allowed_mime_types = [
        'application/pdf',
        'image/jpeg', 'image/png',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # DOCX
        'application/msword',  # DOC
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # XLSX
        'application/vnd.ms-excel'  # XLS
    ]

    logger.info(f"File content_type: {file.content_type}, allowed: {allowed_mime_types}")

    if file.content_type not in allowed_mime_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: PDF, JPG, PNG, DOCX, XLSX"
        )

    # Validate file size (max 50MB)
    max_size = 50 * 1024 * 1024  # 50MB
    file_content = await file.read()
    if len(file_content) > max_size:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    # Test database connection
    from app.main import async_session_maker
    async with async_session_maker() as session:
        # Verify subject exists
        result = await session.execute(
            text("SELECT id FROM subjects WHERE id = :subject_id"),
            {"subject_id": subject_id}
        )
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="Subject not found")

        # Compute SHA256 first
        sha256 = hashlib.sha256(file_content).hexdigest()

        # Create document record FIRST to get the real ID (atomic)
        from datetime import datetime
        now = datetime.now()
        result = await session.execute(
            text("""INSERT INTO documents
                   (subject_id, original_filename, mime_type, size_bytes, sha256, status, progress, ocr_used, created_at, updated_at)
                   VALUES (:subject_id, :filename, :mime_type, :size, :sha256, 'queued', 0, false, :created_at, :updated_at) RETURNING id"""),
            {
                "subject_id": subject_id,
                "filename": safe_filename,
                "mime_type": file.content_type,
                "size": len(file_content),
                "sha256": sha256,
                "created_at": now,
                "updated_at": now
            }
        )
        inserted_id = result.scalar()
        await session.commit()

        logger.info(f"Document inserted with ID: {inserted_id}")

        # NOW create directory with the real document ID
        doc_dir = Path(settings.data_dir) / "subjects" / str(subject_id) / "documents"
        document_dir = doc_dir / str(inserted_id)
        original_dir = document_dir / "original"
        original_dir.mkdir(parents=True, exist_ok=True)

        # Save original file
        file_path = original_dir / safe_filename

        with open(file_path, "wb") as f:
            f.write(file_content)

        logger.info(f"File saved to {file_path}")

        # Enqueue for processing
        from app import main as app_main
        from app.services.job_queue import JobQueue
        logger.info(f"Job queue object: {app_main.job_queue}")

        if app_main.job_queue is None:
            # Fallback: if lifespan didn't initialize the queue, create it now.
            app_main.job_queue = JobQueue(app_main.async_session_maker, app_main.llm_client, app_main.sse_service)
            await app_main.job_queue.start()
            logger.info("Job queue initialized on-demand")

        # Ensure the worker is running, then enqueue
        await app_main.job_queue.start()
        await app_main.job_queue.enqueue_document_processing(inserted_id, model_name=model_name)
        logger.info(f"Document {inserted_id} enqueued for processing with model: {model_name or 'default'}")

    # Return the actual document ID
    return DocumentUploadResponse(document_id=inserted_id)