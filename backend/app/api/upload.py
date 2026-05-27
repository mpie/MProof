from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from datetime import datetime, timezone
import shutil
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import List, Optional
from app.models.schemas import DocumentUploadResponse, BatchUploadResponse
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
    external_reference: str = Form(None, description="Caller's own reference ID, returned unchanged in the response"),
    callback_url: str = Form(None, description="URL to POST results to when processing completes"),
    force_doc_type: str = Form(None, description="Skip classification and use this document type directly"),
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

        # Check for duplicates across all documents (warn, don't block)
        dup_result = await session.execute(
            text("SELECT id FROM documents WHERE sha256 = :sha256 ORDER BY id ASC LIMIT 1"),
            {"sha256": sha256}
        )
        duplicate_of = None
        dup_row = dup_result.fetchone()
        if dup_row:
            duplicate_of = dup_row.id
            logger.info(f"Duplicate document detected: sha256 matches existing document {duplicate_of}")

        # Create document record FIRST to get the real ID (atomic)
        now = datetime.now(timezone.utc)
        await session.execute(
            text("""INSERT INTO documents
                   (subject_id, original_filename, mime_type, size_bytes, sha256, external_reference, callback_url, status, progress, ocr_used, created_at, updated_at)
                   VALUES (:subject_id, :filename, :mime_type, :size, :sha256, :external_reference, :callback_url, 'queued', 0, false, :created_at, :updated_at)"""),
            {
                "subject_id": subject_id,
                "filename": safe_filename,
                "mime_type": file.content_type,
                "size": len(file_content),
                "sha256": sha256,
                "external_reference": external_reference,
                "callback_url": callback_url,
                "created_at": now,
                "updated_at": now
            }
        )
        await session.commit()

        # Fetch the inserted document using sha256 (unique)
        result = await session.execute(
            text("SELECT id FROM documents WHERE sha256 = :sha256 AND subject_id = :subject_id ORDER BY id DESC LIMIT 1"),
            {"sha256": sha256, "subject_id": subject_id}
        )
        row = result.fetchone()
        inserted_id = row.id if row else None

        if not inserted_id:
            raise HTTPException(status_code=500, detail="Failed to retrieve inserted document ID")

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
        await app_main.job_queue.enqueue_document_processing(inserted_id, model_name=model_name, force_doc_type=force_doc_type)
        logger.info(f"Document {inserted_id} enqueued for processing with model: {model_name or 'default'}")

    # Return the actual document ID (+ duplicate warning if sha256 matched existing)
    return DocumentUploadResponse(document_id=inserted_id, duplicate_of=duplicate_of)


async def _insert_and_enqueue(
    session,
    subject_id: int,
    file_content: bytes,
    safe_filename: str,
    mime_type: str,
    external_reference: Optional[str],
    callback_url: Optional[str],
    model_name: Optional[str],
    force_doc_type: Optional[str],
) -> int:
    """Insert one document record, save file, return document ID."""
    import logging as _logging
    _log = _logging.getLogger(__name__)

    sha256 = hashlib.sha256(file_content).hexdigest()
    now = datetime.now(timezone.utc)
    await session.execute(
        text("""INSERT INTO documents
               (subject_id, original_filename, mime_type, size_bytes, sha256,
                external_reference, callback_url, status, progress, ocr_used,
                created_at, updated_at)
               VALUES (:subject_id, :filename, :mime_type, :size, :sha256,
                       :external_reference, :callback_url, 'queued', 0, false,
                       :created_at, :updated_at)"""),
        {
            "subject_id": subject_id,
            "filename": safe_filename,
            "mime_type": mime_type,
            "size": len(file_content),
            "sha256": sha256,
            "external_reference": external_reference,
            "callback_url": callback_url,
            "created_at": now,
            "updated_at": now,
        },
    )
    await session.commit()

    row = (await session.execute(
        text("SELECT id FROM documents WHERE sha256 = :sha256 AND subject_id = :subject_id ORDER BY id DESC LIMIT 1"),
        {"sha256": sha256, "subject_id": subject_id},
    )).fetchone()
    if not row:
        raise RuntimeError(f"Failed to retrieve inserted document for {safe_filename}")
    doc_id = int(row.id)

    doc_dir = Path(settings.data_dir) / "subjects" / str(subject_id) / "documents" / str(doc_id) / "original"
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / safe_filename).write_bytes(file_content)
    _log.info(f"Batch: saved {safe_filename} as document {doc_id}")
    return doc_id


@router.post("/upload/batch", response_model=BatchUploadResponse)
async def upload_documents_batch(
    subject_id: int = Form(..., description="Subject ID to associate documents with"),
    files: List[UploadFile] = File(..., description="One or more document files"),
    model_name: str = Form(None, description="Optional model name for classification"),
    callback_url: str = Form(None, description="URL to POST results to for each document"),
    force_doc_type: str = Form(None, description="Skip classification and use this document type for all files"),
    external_references: str = Form(None, description="JSON array of external references, one per file (optional)"),
):
    """Upload multiple documents in one request; each is processed independently in parallel."""
    import logging as _logging
    _log = _logging.getLogger(__name__)

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Max 50 files per batch")

    # Parse per-file external references
    ext_refs: List[Optional[str]] = []
    if external_references:
        try:
            parsed = json.loads(external_references)
            if not isinstance(parsed, list):
                raise ValueError
            ext_refs = [str(r) if r is not None else None for r in parsed]
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="external_references must be a JSON array")
        if len(ext_refs) != len(files):
            raise HTTPException(
                status_code=400,
                detail=f"external_references length ({len(ext_refs)}) must match files count ({len(files)})",
            )
    else:
        ext_refs = [None] * len(files)

    allowed_mime_types = {
        'application/pdf',
        'image/jpeg', 'image/png',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel',
    }
    max_size = 50 * 1024 * 1024

    from app.main import async_session_maker
    from app import main as app_main
    from app.services.job_queue import JobQueue

    document_ids: List[int] = []

    async with async_session_maker() as session:
        # Verify subject once
        if not (await session.execute(
            text("SELECT id FROM subjects WHERE id = :subject_id"),
            {"subject_id": subject_id},
        )).fetchone():
            raise HTTPException(status_code=404, detail="Subject not found")

        for i, file in enumerate(files):
            safe_filename = _sanitize_filename(file.filename or "")
            if not safe_filename:
                raise HTTPException(status_code=400, detail=f"File {i}: invalid filename")

            if file.content_type not in allowed_mime_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {i} ({file.filename}): unsupported type {file.content_type}",
                )

            content = await file.read()
            if len(content) > max_size:
                raise HTTPException(status_code=400, detail=f"File {i} ({file.filename}): exceeds 50MB limit")

            doc_id = await _insert_and_enqueue(
                session,
                subject_id=subject_id,
                file_content=content,
                safe_filename=safe_filename,
                mime_type=file.content_type,
                external_reference=ext_refs[i],
                callback_url=callback_url,
                model_name=model_name,
                force_doc_type=force_doc_type,
            )
            document_ids.append(doc_id)

    # Ensure queue running, enqueue all (parallel processing by existing workers)
    if app_main.job_queue is None:
        app_main.job_queue = JobQueue(app_main.async_session_maker, app_main.llm_client, app_main.sse_service)
        await app_main.job_queue.start()

    await app_main.job_queue.start()
    for doc_id in document_ids:
        await app_main.job_queue.enqueue_document_processing(doc_id, model_name=model_name, force_doc_type=force_doc_type)

    _log.info(f"Batch upload: enqueued {len(document_ids)} documents for subject {subject_id}")
    return BatchUploadResponse(document_ids=document_ids, count=len(document_ids))