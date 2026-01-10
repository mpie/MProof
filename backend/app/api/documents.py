from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import Optional
import os
from pathlib import Path
from app.models.schemas import DocumentResponse, DocumentListResponse
from app.config import settings

router = APIRouter()


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    subject_id: Optional[int] = Query(None, description="Filter by subject ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(lambda: None)  # Will be injected
):
    """List documents with optional filtering."""
    from app.main import async_session_maker
    async with async_session_maker() as session:
        conditions = []
        params = {}

        if subject_id is not None:
            conditions.append("subject_id = :subject_id")
            params["subject_id"] = subject_id

        if status:
            conditions.append("status = :status")
            params["status"] = status

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""SELECT d.*, s.name as subject_name, s.context as subject_context
                  FROM documents d
                  LEFT JOIN subjects s ON d.subject_id = s.id
                  WHERE {where_clause}
                  ORDER BY d.created_at DESC
                  LIMIT :limit OFFSET :offset"""
        params["limit"] = limit
        params["offset"] = offset

        result = await session.execute(text(sql), params)
        documents = result.fetchall()

        # Get total count
        count_sql = f"SELECT COUNT(*) FROM documents WHERE {where_clause}"
        count_params = params.copy()
        del count_params["limit"]
        del count_params["offset"]
        count_result = await session.execute(text(count_sql), count_params)
        total = count_result.scalar()

        # Convert documents to proper format with conversions
        formatted_documents = []
        for doc in documents:
            doc_dict = dict(doc._mapping)
            doc_dict['ocr_used'] = bool(doc_dict['ocr_used'])
            # Parse JSON fields
            import json
            if doc_dict.get('risk_signals_json'):
                doc_dict['risk_signals_json'] = json.loads(doc_dict['risk_signals_json'])
            if doc_dict.get('metadata_json'):
                doc_dict['metadata_json'] = json.loads(doc_dict['metadata_json'])
            if doc_dict.get('metadata_validation_json'):
                doc_dict['metadata_validation_json'] = json.loads(doc_dict['metadata_validation_json'])
            if doc_dict.get('metadata_evidence_json'):
                doc_dict['metadata_evidence_json'] = json.loads(doc_dict['metadata_evidence_json'])
            formatted_documents.append(DocumentResponse(**doc_dict))

        return DocumentListResponse(
            documents=formatted_documents,
            total=total
        )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(lambda: None)  # Will be injected
):
    """Get a specific document."""
    from app.main import async_session_maker
    async with async_session_maker() as session:
        result = await session.execute(
            text("SELECT * FROM documents WHERE id = :document_id"),
            {"document_id": document_id}
        )
        document = result.fetchone()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Convert to dict and fix conversions
        doc_dict = dict(document._mapping)
        doc_dict['ocr_used'] = bool(doc_dict['ocr_used'])
        # Parse JSON fields
        import json
        if doc_dict.get('risk_signals_json'):
            doc_dict['risk_signals_json'] = json.loads(doc_dict['risk_signals_json'])
        if doc_dict.get('metadata_json'):
            doc_dict['metadata_json'] = json.loads(doc_dict['metadata_json'])
        if doc_dict.get('metadata_validation_json'):
            doc_dict['metadata_validation_json'] = json.loads(doc_dict['metadata_validation_json'])
        if doc_dict.get('metadata_evidence_json'):
            doc_dict['metadata_evidence_json'] = json.loads(doc_dict['metadata_evidence_json'])

        return DocumentResponse(**doc_dict)


@router.post("/documents/{document_id}/analyze")
async def analyze_document(
    document_id: int
):
    """Trigger analysis for a document."""
    from app.main import async_session_maker, job_queue

    async with async_session_maker() as session:
        # Check if document exists
        result = await session.execute(
            text("SELECT id, status FROM documents WHERE id = :document_id"),
            {"document_id": document_id}
        )
        document = result.fetchone()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Reset document status for re-analysis
        from datetime import datetime
        await session.execute(
            text("""UPDATE documents SET
                   status = 'queued', progress = 0, stage = NULL,
                   error_message = NULL, doc_type_slug = NULL,
                   doc_type_confidence = NULL, doc_type_rationale = NULL,
                   metadata_json = NULL, metadata_validation_json = NULL,
                   metadata_evidence_json = NULL, risk_score = NULL,
                   risk_signals_json = NULL, updated_at = :updated_at
                   WHERE id = :document_id"""),
            {"document_id": document_id, "updated_at": datetime.now()}
        )
        await session.commit()

        # Enqueue for processing
        await job_queue.enqueue_document_processing(document_id)

        return {"ok": True, "message": "Analysis started"}


@router.get("/documents/{document_id}/artifact")
async def get_document_artifact(
    document_id: int,
    path: str = Query(..., description="Artifact path (e.g., 'text/extracted.txt')"),
    db: AsyncSession = Depends(lambda: None)  # Will be injected
):
    """Serve document artifacts safely."""
    from app.main import async_session_maker

    # Validate path to prevent directory traversal
    if ".." in path or path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid path")

    # Only allow specific artifact types
    allowed_paths = [
        "original/", "text/extracted.json", "text/extracted.txt",
        "metadata/result.json", "metadata/validation.json", "metadata/evidence.json",
        "risk/result.json",
        "llm/",
    ]

    if not any(path.startswith(allowed) for allowed in allowed_paths):
        raise HTTPException(status_code=400, detail="Artifact path not allowed")

    async with async_session_maker() as session:
        # Get document and verify ownership
        result = await session.execute(
            text("SELECT subject_id, original_filename FROM documents WHERE id = :document_id"),
            {"document_id": document_id}
        )
        document = result.fetchone()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Construct safe path
        subject_id = document.subject_id
        artifact_path = Path(settings.data_dir) / "subjects" / str(subject_id) / "documents" / str(document_id) / path

        # Verify file exists and is within allowed directory
        if not artifact_path.exists():
            raise HTTPException(status_code=404, detail="Artifact not found")

        # Get the base document directory
        doc_dir = Path(settings.data_dir) / "subjects" / str(subject_id) / "documents" / str(document_id)
        try:
            # Ensure the artifact is within the document directory
            artifact_path.resolve().relative_to(doc_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")

        # Determine content type
        if path.endswith('.json'):
            media_type = 'application/json'
        elif path.endswith('.txt'):
            media_type = 'text/plain'
        elif path.startswith('original/'):
            # For original files, we'd need to detect MIME type
            # For now, return as binary
            media_type = 'application/octet-stream'
        else:
            media_type = 'application/octet-stream'

        return FileResponse(
            path=str(artifact_path),
            media_type=media_type,
            filename=artifact_path.name if not path.startswith('original/') else document.original_filename
        )


@router.delete("/documents/{document_id}")
async def delete_document(document_id: int, db: AsyncSession = Depends(lambda: None)):
    """Delete a document and all its artifacts."""
    from app.main import async_session_maker
    async with async_session_maker() as session:
        # Check if document exists
        result = await session.execute(
            text("SELECT id, subject_id FROM documents WHERE id = :document_id"),
            {"document_id": document_id}
        )
        document = result.fetchone()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete document record (cascade will handle related records)
        await session.execute(
            text("DELETE FROM documents WHERE id = :document_id"),
            {"document_id": document_id}
        )

        await session.commit()

        return {"ok": True, "message": "Document deleted"}