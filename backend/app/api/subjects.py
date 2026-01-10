from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, or_, and_
from typing import List, Optional
import unicodedata
import re
from app.models.schemas import (
    SubjectCreate, SubjectResponse, SubjectSearchResponse,
    SubjectGroupResponse, ContextEnum
)
from app.models.database import Subject

router = APIRouter()


def normalize_text(text: str) -> str:
    """Normalize text for case-insensitive search."""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').lower()


@router.get("/subjects", response_model=SubjectSearchResponse)
async def search_subjects(
    query: Optional[str] = Query(None, description="Search query"),
    context: Optional[ContextEnum] = Query(None, description="Filter by context"),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(lambda: None)  # Will be injected
):
    """Search subjects with optional filtering."""
    # This is a simplified version - in practice you'd use dependency injection
    from app.main import async_session_maker
    async with async_session_maker() as session:
        conditions = []
        params = {}

        if query:
            normalized_query = normalize_text(query)
            conditions.append("name_normalized LIKE :query")
            params["query"] = f"%{normalized_query}%"

        if context:
            conditions.append("context = :context")
            params["context"] = context.value

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM subjects WHERE {where_clause} ORDER BY name LIMIT :limit"
        params["limit"] = limit

        result = await session.execute(text(sql), params)
        subjects = result.fetchall()

        return SubjectSearchResponse(
            subjects=[SubjectResponse(**subject._mapping) for subject in subjects],
            total=len(subjects)
        )


@router.post("/subjects", response_model=SubjectResponse)
async def create_subject(
    subject: SubjectCreate,
    db: AsyncSession = Depends(lambda: None)  # Will be injected
):
    """Create a new subject."""
    from app.main import async_session_maker
    async with async_session_maker() as session:
        # Check if subject already exists
        normalized_name = normalize_text(subject.name)
        result = await session.execute(
            text("SELECT * FROM subjects WHERE name_normalized = :name_normalized AND context = :context"),
            {"name_normalized": normalized_name, "context": subject.context.value}
        )
        existing = result.fetchone()

        if existing:
            raise HTTPException(status_code=409, detail="Subject already exists")

        # Create new subject
        from datetime import datetime
        now = datetime.now()
        result = await session.execute(
            text("""INSERT INTO subjects (name, name_normalized, context, created_at, updated_at)
                   VALUES (:name, :name_normalized, :context, :created_at, :updated_at) RETURNING id"""),
            {
                "name": subject.name,
                "name_normalized": normalized_name,
                "context": subject.context.value,
                "created_at": now,
                "updated_at": now
            }
        )
        subject_id = result.scalar()
        await session.commit()

        # Fetch the created subject
        result = await session.execute(
            text("SELECT * FROM subjects WHERE id = :subject_id"),
            {"subject_id": subject_id}
        )
        new_subject = result.fetchone()

        return SubjectResponse(**new_subject._mapping)


@router.get("/subjects/by-name", response_model=SubjectGroupResponse)
async def search_subjects_by_name_grouped(
    name: str = Query(..., min_length=1),
    db: AsyncSession = Depends(lambda: None)  # Will be injected
):
    """Search subjects by name and group results by context."""
    from app.main import async_session_maker
    async with async_session_maker() as session:
        normalized_name = normalize_text(name)

        result = await session.execute(
            text("SELECT * FROM subjects WHERE name_normalized LIKE :name ORDER BY context, name"),
            {"name": f"%{normalized_name}%"}
        )
        subjects = result.fetchall()

        # Group by context
        groups = {}
        for subject in subjects:
            context = subject.context
            if context not in groups:
                groups[context] = []
            groups[context].append(SubjectResponse(**subject._mapping))

        return SubjectGroupResponse(name=name, groups=groups)


@router.get("/subjects/{subject_id}", response_model=SubjectResponse)
async def get_subject(
    subject_id: int,
    db: AsyncSession = Depends(lambda: None)  # Will be injected
):
    """Get a specific subject."""
    from app.main import async_session_maker
    async with async_session_maker() as session:
        result = await session.execute(
            text("SELECT * FROM subjects WHERE id = :subject_id"),
            {"subject_id": subject_id}
        )
        subject = result.fetchone()

        if not subject:
            raise HTTPException(status_code=404, detail="Subject not found")

        return SubjectResponse(**subject._mapping)


@router.get("/subjects/{subject_id}/documents")
async def get_subject_documents(
    subject_id: int,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(lambda: None)  # Will be injected
):
    """Get documents for a specific subject."""
    from app.main import async_session_maker
    async with async_session_maker() as session:
        # Verify subject exists
        result = await session.execute(
            text("SELECT id FROM subjects WHERE id = :subject_id"),
            {"subject_id": subject_id}
        )
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="Subject not found")

        # Get documents with subject info
        result = await session.execute(
            text("""SELECT d.*, s.name as subject_name, s.context as subject_context
                    FROM documents d
                    LEFT JOIN subjects s ON d.subject_id = s.id
                    WHERE d.subject_id = :subject_id
                    ORDER BY d.created_at DESC
                    LIMIT :limit OFFSET :offset"""),
            {"subject_id": subject_id, "limit": limit, "offset": offset}
        )
        documents = result.fetchall()

        # Get total count
        count_result = await session.execute(
            text("SELECT COUNT(*) FROM documents WHERE subject_id = :subject_id"),
            {"subject_id": subject_id}
        )
        total = count_result.scalar()

        from app.models.schemas import DocumentListResponse, DocumentResponse
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