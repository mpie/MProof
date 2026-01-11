from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import Optional
import json
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
    from app import main as app_main
    from app.services.job_queue import JobQueue

    async with app_main.async_session_maker() as session:
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

        # Ensure queue exists + is running, then enqueue
        if app_main.job_queue is None:
            app_main.job_queue = JobQueue(app_main.async_session_maker, app_main.llm_client, app_main.sse_service)
            await app_main.job_queue.start()

        await app_main.job_queue.start()
        await app_main.job_queue.enqueue_document_processing(document_id)

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
    if "\x00" in path or ".." in path or path.startswith("/"):
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
            # Detect MIME type for original files
            import mimetypes
            mime_type, _ = mimetypes.guess_type(str(artifact_path))
            if mime_type:
                media_type = mime_type
            else:
                # Fallback based on extension
                if artifact_path.suffix.lower() == '.pdf':
                    media_type = 'application/pdf'
                elif artifact_path.suffix.lower() in ['.jpg', '.jpeg']:
                    media_type = 'image/jpeg'
                elif artifact_path.suffix.lower() == '.png':
                    media_type = 'image/png'
                else:
                    media_type = 'application/octet-stream'
        else:
            media_type = 'application/octet-stream'

        # Set headers for inline display (especially for PDFs and images)
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, OPTIONS',
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Expose-Headers': 'Content-Disposition, Content-Type, Content-Length',
            'X-Content-Type-Options': 'nosniff',
            'Cache-Control': 'public, max-age=3600',
        }
        
        # Get file size for Content-Length header
        file_size = os.path.getsize(artifact_path)
        headers['Content-Length'] = str(file_size)
        
        if media_type == 'application/pdf' or media_type.startswith('image/'):
            # For PDFs and images, set inline disposition to display in browser
            headers['Content-Disposition'] = 'inline'
        else:
            # For other files, allow download
            filename = artifact_path.name if not path.startswith('original/') else document.original_filename
            headers['Content-Disposition'] = f'attachment; filename="{filename}"'

        # Use streaming for efficient file serving
        def iter_file():
            with open(artifact_path, 'rb') as f:
                while chunk := f.read(65536):  # 64KB chunks
                    yield chunk
        
        return StreamingResponse(
            iter_file(),
            media_type=media_type,
            headers=headers
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


@router.get("/documents/{document_id}/fraud-analysis")
async def get_fraud_analysis(document_id: int, db: AsyncSession = Depends(lambda: None)):
    """
    Get comprehensive fraud analysis for a document.
    
    Analyzes:
    - PDF metadata (creator software, timestamps)
    - Image forensics (ELA manipulation detection)
    - Text anomalies (unicode, repetition)
    - Classification confidence
    """
    from app.main import async_session_maker
    from app.services.fraud_detector import fraud_detector
    
    async with async_session_maker() as session:
        # Get document
        result = await session.execute(
            text("""
                SELECT d.*, s.name as subject_name
                FROM documents d
                LEFT JOIN subjects s ON d.subject_id = s.id
                WHERE d.id = :document_id
            """),
            {"document_id": document_id}
        )
        document = result.fetchone()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Load document file
        doc_dir = Path(settings.data_dir) / "subjects" / str(document.subject_id) / "documents" / str(document.id)
        original_path = doc_dir / "original" / document.original_filename
        
        file_bytes = None
        if original_path.exists():
            file_bytes = original_path.read_bytes()
        
        # Load extracted text
        extracted_text = None
        text_path = doc_dir / "text" / "extracted.txt"
        if text_path.exists():
            extracted_text = text_path.read_text(encoding='utf-8', errors='ignore')
        
        # Check for cached fraud analysis
        fraud_cache_path = doc_dir / "fraud_analysis.json"
        if fraud_cache_path.exists():
            try:
                cached_data = json.loads(fraud_cache_path.read_text())
                # Verify cache is still valid (check if document hasn't been reprocessed)
                # For now, we'll trust the cache if it exists
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cached fraud analysis: {e}")
        
        # Load existing fraud signals from processing
        existing_signals = {}
        llm_dir = doc_dir / "llm"
        
        # Check for fpdf flag
        if llm_dir.exists():
            for json_file in llm_dir.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text())
                    if "fpdf" in data:
                        existing_signals["fpdf"] = data.get("fpdf")
                    if "fraud_signals" in data:
                        existing_signals.update(data["fraud_signals"])
                except Exception:
                    pass
        
        # Run fraud analysis
        report = fraud_detector().analyze_document(
            file_bytes=file_bytes,
            filename=document.original_filename,
            document_id=document.id,
            extracted_text=extracted_text,
            classification_confidence=document.doc_type_confidence,
            existing_signals=existing_signals,
        )
        
        # Cache the result
        report_dict = report.to_dict()
        try:
            fraud_cache_path.parent.mkdir(parents=True, exist_ok=True)
            fraud_cache_path.write_text(json.dumps(report_dict, indent=2))
        except Exception as e:
            logger.warning(f"Failed to cache fraud analysis: {e}")
        
        return report_dict


@router.post("/documents/analyze-fraud")
async def analyze_fraud_upload(
    file_bytes: bytes,
    filename: str,
):
    """
    Analyze an uploaded file for fraud without saving it.
    Useful for quick checks before full processing.
    """
    from app.services.fraud_detector import fraud_detector
    
    report = fraud_detector().analyze_document(
        file_bytes=file_bytes,
        filename=filename,
    )
    
    return report.to_dict()