from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import shutil
import hashlib
from pathlib import Path
from app.models.schemas import DocumentUploadResponse
from app.config import settings

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    subject_id: int = Form(..., description="Subject ID to associate the document with"),
    file: UploadFile = File(..., description="Document file to upload"),
    db: AsyncSession = Depends(lambda: None)  # Will be injected
):
    """Upload a document and initiate processing."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Upload request: subject_id={subject_id}, filename={file.filename}, content_type={file.content_type}")

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

        # Create document directory
        doc_dir = Path(settings.data_dir) / "subjects" / str(subject_id) / "documents"
        doc_dir.mkdir(parents=True, exist_ok=True)

        # Get next document ID
        result = await session.execute(text("SELECT COALESCE(MAX(id), 0) + 1 FROM documents"))
        document_id = result.scalar()

        document_dir = doc_dir / str(document_id)
        document_dir.mkdir(exist_ok=True)

        # Save original file
        original_dir = document_dir / "original"
        original_dir.mkdir(exist_ok=True)
        file_path = original_dir / file.filename

        with open(file_path, "wb") as f:
            f.write(file_content)

        logger.info(f"File saved to {file_path}")

        # Compute SHA256
        sha256 = hashlib.sha256(file_content).hexdigest()

        # Create document record
        from datetime import datetime
        now = datetime.now()
        result = await session.execute(
            text("""INSERT INTO documents
                   (subject_id, original_filename, mime_type, size_bytes, sha256, status, progress, ocr_used, created_at, updated_at)
                   VALUES (:subject_id, :filename, :mime_type, :size, :sha256, 'queued', 0, false, :created_at, :updated_at) RETURNING id"""),
            {
                "subject_id": subject_id,
                "filename": file.filename,
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

        # Enqueue for processing
        from app.main import job_queue
        logger.info(f"Job queue object: {job_queue}")
        if job_queue is not None:
            await job_queue.enqueue_document_processing(inserted_id)
            logger.info(f"Document {inserted_id} enqueued for processing")
        else:
            logger.warning("Job queue not available")

    # Return the actual document ID
    return DocumentUploadResponse(document_id=inserted_id)