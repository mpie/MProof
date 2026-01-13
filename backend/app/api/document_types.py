import json
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List
from pydantic import BaseModel
from app.models.schemas import (
    DocumentTypeResponse, DocumentTypeCreate, DocumentTypeUpdate,
    DocumentTypeFieldResponse, DocumentTypeFieldCreate, DocumentTypeFieldUpdate
)
from app.services.policy_loader import validate_policy_json

router = APIRouter()


class PrefillRequest(BaseModel):
    name: str
    keywords: str


class PrefillResponse(BaseModel):
    description: str
    extraction_prompt_preamble: str


@router.get("/document-types", response_model=List[DocumentTypeResponse])
async def list_document_types(db: AsyncSession = Depends(lambda: None)):
    """List all document types with their fields."""
    from app.main import async_session_maker
    async with async_session_maker() as session:
        result = await session.execute(
            text("SELECT * FROM document_types ORDER BY name")
        )
        doc_types = result.fetchall()

        response = []
        for doc_type in doc_types:
            # Get fields for this document type
            fields_result = await session.execute(
                text("SELECT * FROM document_type_fields WHERE document_type_id = :document_type_id ORDER BY key"),
                {"document_type_id": doc_type.id}
            )
            fields = fields_result.fetchall()

            doc_type_dict = dict(doc_type._mapping)
            doc_type_dict["fields"] = [DocumentTypeFieldResponse(**field._mapping) for field in fields]
            response.append(DocumentTypeResponse(**doc_type_dict))

        return response


@router.get("/document-types/check-name")
async def check_document_type_name_unique(
    name: str,
    exclude_slug: str = None,
    db: AsyncSession = Depends(lambda: None)
):
    """Check if a document type name is unique."""
    from app.main import async_session_maker
    async with async_session_maker() as session:
        conditions = ["LOWER(name) = LOWER(:name)"]
        params = {"name": name}

        if exclude_slug:
            conditions.append("slug != :exclude_slug")
            params["exclude_slug"] = exclude_slug

        where_clause = " AND ".join(conditions)
        result = await session.execute(
            text(f"SELECT COUNT(*) FROM document_types WHERE {where_clause}"),
            params
        )
        count = result.scalar()
        return {"is_unique": count == 0}


@router.get("/document-types/check-slug")
async def check_document_type_slug_unique(
    slug: str,
    exclude_slug: str = None,
    db: AsyncSession = Depends(lambda: None)
):
    """Check if a document type slug is unique."""
    from app.main import async_session_maker
    async with async_session_maker() as session:
        conditions = ["slug = :slug"]
        params = {"slug": slug}

        if exclude_slug:
            conditions.append("slug != :exclude_slug")
            params["exclude_slug"] = exclude_slug

        where_clause = " AND ".join(conditions)
        result = await session.execute(
            text(f"SELECT COUNT(*) FROM document_types WHERE {where_clause}"),
            params
        )
        count = result.scalar()
        return {"is_unique": count == 0}


@router.post("/document-types", response_model=DocumentTypeResponse)
async def create_document_type(
    doc_type: DocumentTypeCreate,
    db: AsyncSession = Depends(lambda: None)
):
    """Create a new document type."""
    from app.main import async_session_maker
    async with async_session_maker() as session:
        # Check if slug already exists
        result = await session.execute(
            text("SELECT id FROM document_types WHERE slug = :slug"),
            {"slug": doc_type.slug}
        )
        if result.fetchone():
            raise HTTPException(status_code=409, detail="Document type slug already exists")

        # Validate policy JSON if provided
        policy_json_str = None
        if doc_type.classification_policy_json:
            is_valid, error = validate_policy_json(doc_type.classification_policy_json)
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid classification policy: {error}")
            policy_json_str = json.dumps(doc_type.classification_policy_json)

        # Create document type
        now = datetime.now(timezone.utc)
        await session.execute(
            text("""INSERT INTO document_types (name, slug, description, classification_hints, extraction_prompt_preamble, classification_policy_json, created_at, updated_at)
                   VALUES (:name, :slug, :description, :classification_hints, :extraction_prompt_preamble, :classification_policy_json, :created_at, :updated_at)"""),
            {
                "name": doc_type.name,
                "slug": doc_type.slug,
                "description": doc_type.description,
                "classification_hints": doc_type.classification_hints,
                "extraction_prompt_preamble": doc_type.extraction_prompt_preamble,
                "classification_policy_json": policy_json_str,
                "created_at": now,
                "updated_at": now
            }
        )
        await session.commit()

        # Fetch the created document type using slug (unique)
        result = await session.execute(
            text("SELECT * FROM document_types WHERE slug = :slug"),
            {"slug": doc_type.slug}
        )
        new_doc_type = result.fetchone()
        doc_type_dict = dict(new_doc_type._mapping)
        doc_type_dict["fields"] = []
        return DocumentTypeResponse(**doc_type_dict)


@router.get("/document-types/{slug}", response_model=DocumentTypeResponse)
async def get_document_type(slug: str, db: AsyncSession = Depends(lambda: None)):
    """Get a specific document type with its fields."""
    from app.main import async_session_maker
    async with async_session_maker() as session:
        result = await session.execute(
            text("SELECT * FROM document_types WHERE slug = :slug"),
            {"slug": slug}
        )
        doc_type = result.fetchone()

        if not doc_type:
            raise HTTPException(status_code=404, detail="Document type not found")

        # Get fields
        fields_result = await session.execute(
            text("SELECT * FROM document_type_fields WHERE document_type_id = :document_type_id ORDER BY key"),
            {"document_type_id": doc_type.id}
        )
        fields = fields_result.fetchall()

        doc_type_dict = dict(doc_type._mapping)
        doc_type_dict["fields"] = [DocumentTypeFieldResponse(**field._mapping) for field in fields]
        return DocumentTypeResponse(**doc_type_dict)


@router.put("/document-types/{slug}", response_model=DocumentTypeResponse)
async def update_document_type(
    slug: str,
    doc_type_update: DocumentTypeUpdate,
    db: AsyncSession = Depends(lambda: None)
):
    """Update a document type."""
    from app.main import async_session_maker
    async with async_session_maker() as session:
        # Get current document type
        result = await session.execute(
            text("SELECT * FROM document_types WHERE slug = :slug"),
            {"slug": slug}
        )
        doc_type = result.fetchone()

        if not doc_type:
            raise HTTPException(status_code=404, detail="Document type not found")

        # Check slug conflict if changing slug
        if doc_type_update.slug and doc_type_update.slug != slug:
            conflict_result = await session.execute(
                text("SELECT id FROM document_types WHERE slug = :slug"),
                {"slug": doc_type_update.slug}
            )
            if conflict_result.fetchone():
                raise HTTPException(status_code=409, detail="Document type slug already exists")

        # Build update query with named parameters
        update_fields = []
        params = {}

        update_data = doc_type_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            if value is not None:
                # Special handling for classification_policy_json
                if field == "classification_policy_json":
                    is_valid, error = validate_policy_json(value)
                    if not is_valid:
                        raise HTTPException(status_code=400, detail=f"Invalid classification policy: {error}")
                    update_fields.append(f"{field} = :{field}")
                    params[field] = json.dumps(value)
                else:
                    update_fields.append(f"{field} = :{field}")
                    params[field] = value

        if not update_fields:
            # No changes, return current
            fields_result = await session.execute(
                text("SELECT * FROM document_type_fields WHERE document_type_id = :document_type_id ORDER BY key"),
                {"document_type_id": doc_type.id}
            )
            fields = fields_result.fetchall()
            doc_type_dict = dict(doc_type._mapping)
            doc_type_dict["fields"] = [DocumentTypeFieldResponse(**field._mapping) for field in fields]
            return DocumentTypeResponse(**doc_type_dict)

        sql = f"UPDATE document_types SET {', '.join(update_fields)}, updated_at = :updated_at WHERE slug = :old_slug"
        params["updated_at"] = datetime.now(timezone.utc)
        params["old_slug"] = slug

        await session.execute(text(sql), params)
        await session.commit()

        # Return updated document type
        return await get_document_type(doc_type_update.slug if doc_type_update.slug else slug)


@router.delete("/document-types/{slug}")
async def delete_document_type(slug: str, db: AsyncSession = Depends(lambda: None)):
    """Delete a document type."""
    from app.main import async_session_maker
    async with async_session_maker() as session:
        # Check if document type exists
        result = await session.execute(
            text("SELECT id FROM document_types WHERE slug = :slug"),
            {"slug": slug}
        )
        doc_type = result.fetchone()

        if not doc_type:
            raise HTTPException(status_code=404, detail="Document type not found")

        # Check if there are any documents using this type
        docs_result = await session.execute(
            text("SELECT COUNT(*) FROM documents WHERE doc_type_slug = :doc_type_slug"),
            {"doc_type_slug": slug}
        )
        doc_count = docs_result.scalar()

        if doc_count > 0:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot delete document type: {doc_count} documents are using it"
            )

        # Delete fields first
        await session.execute(
            text("DELETE FROM document_type_fields WHERE document_type_id = :document_type_id"),
            {"document_type_id": doc_type.id}
        )

        # Delete document type
        await session.execute(
            text("DELETE FROM document_types WHERE id = :id"),
            {"id": doc_type.id}
        )

        await session.commit()

        return {"ok": True, "message": "Document type deleted"}


# Field CRUD endpoints
@router.get("/document-types/{slug}/fields", response_model=List[DocumentTypeFieldResponse])
async def list_document_type_fields(slug: str, db: AsyncSession = Depends(lambda: None)):
    """List fields for a document type."""
    from app.main import async_session_maker
    async with async_session_maker() as session:
        # Get document type ID
        result = await session.execute(
            text("SELECT id FROM document_types WHERE slug = :slug"),
            {"slug": slug}
        )
        doc_type = result.fetchone()

        if not doc_type:
            raise HTTPException(status_code=404, detail="Document type not found")

        # Get fields
        fields_result = await session.execute(
            text("SELECT * FROM document_type_fields WHERE document_type_id = :document_type_id ORDER BY key"),
            {"document_type_id": doc_type.id}
        )
        fields = fields_result.fetchall()

        return [DocumentTypeFieldResponse(**field._mapping) for field in fields]


@router.post("/document-types/{slug}/fields", response_model=DocumentTypeFieldResponse)
async def create_document_type_field(
    slug: str,
    field: DocumentTypeFieldCreate,
    db: AsyncSession = Depends(lambda: None)
):
    """Create a new field for a document type."""
    from app.main import async_session_maker
    async with async_session_maker() as session:
        # Get document type ID
        result = await session.execute(
            text("SELECT id FROM document_types WHERE slug = :slug"),
            {"slug": slug}
        )
        doc_type = result.fetchone()

        if not doc_type:
            raise HTTPException(status_code=404, detail="Document type not found")

        # Check if key already exists
        key_result = await session.execute(
            text("SELECT id FROM document_type_fields WHERE document_type_id = :document_type_id AND key = :key"),
            {"document_type_id": doc_type.id, "key": field.key}
        )
        if key_result.fetchone():
            raise HTTPException(status_code=409, detail="Field key already exists")

        # Create field
        now = datetime.now(timezone.utc)
        await session.execute(
            text("""INSERT INTO document_type_fields
                   (document_type_id, key, label, field_type, required, enum_values, regex, description, examples, created_at, updated_at)
                   VALUES (:document_type_id, :key, :label, :field_type, :required, :enum_values, :regex, :description, :examples, :created_at, :updated_at)"""),
            {
                "document_type_id": doc_type.id,
                "key": field.key,
                "label": field.label,
                "field_type": field.field_type.value,
                "required": field.required,
                "enum_values": field.enum_values,
                "regex": field.regex,
                "description": field.description,
                "examples": field.examples,
                "created_at": now,
                "updated_at": now
            }
        )
        await session.commit()

        # Fetch the created field using document_type_id + key (unique combination)
        result = await session.execute(
            text("SELECT * FROM document_type_fields WHERE document_type_id = :document_type_id AND key = :key"),
            {"document_type_id": doc_type.id, "key": field.key}
        )
        new_field = result.fetchone()

        return DocumentTypeFieldResponse(**new_field._mapping)


@router.put("/document-types/{slug}/fields/{field_id}", response_model=DocumentTypeFieldResponse)
async def update_document_type_field(
    slug: str,
    field_id: int,
    field_update: DocumentTypeFieldUpdate,
    db: AsyncSession = Depends(lambda: None)
):
    """Update a field for a document type."""
    from app.main import async_session_maker
    async with async_session_maker() as session:
        # Get document type ID and verify ownership
        result = await session.execute(
            text("""SELECT dt.id, dtf.id as field_id FROM document_types dt
                   JOIN document_type_fields dtf ON dt.id = dtf.document_type_id
                   WHERE dt.slug = :slug AND dtf.id = :field_id"""),
            {"slug": slug, "field_id": field_id}
        )
        row = result.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Document type or field not found")

        # Build update query with named parameters
        update_fields = []
        params = {}

        update_data = field_update.dict(exclude_unset=True)
        if update_data:
            for field, value in update_data.items():
                if field == "field_type" and value is not None:
                    update_fields.append(f"{field} = :{field}")
                    params[field] = value.value
                else:
                    update_fields.append(f"{field} = :{field}")
                    params[field] = value

        if not update_fields:
            # No changes, return current field
            field_result = await session.execute(
                text("SELECT * FROM document_type_fields WHERE id = :field_id"),
                {"field_id": field_id}
            )
            current_field = field_result.fetchone()
            return DocumentTypeFieldResponse(**current_field._mapping)

        sql = f"UPDATE document_type_fields SET {', '.join(update_fields)}, updated_at = :updated_at WHERE id = :field_id"
        params["updated_at"] = datetime.now(timezone.utc)
        params["field_id"] = field_id

        await session.execute(text(sql), params)
        await session.commit()

        # Return updated field
        field_result = await session.execute(
            text("SELECT * FROM document_type_fields WHERE id = :field_id"),
            {"field_id": field_id}
        )
        updated_field = field_result.fetchone()
        return DocumentTypeFieldResponse(**updated_field._mapping)


@router.delete("/document-types/{slug}/fields/{field_id}")
async def delete_document_type_field(
    slug: str,
    field_id: int,
    db: AsyncSession = Depends(lambda: None)
):
    """Delete a field from a document type."""
    from app.main import async_session_maker
    async with async_session_maker() as session:
        # Verify ownership
        result = await session.execute(
            text("""SELECT dtf.id FROM document_types dt
                   JOIN document_type_fields dtf ON dt.id = dtf.document_type_id
                   WHERE dt.slug = :slug AND dtf.id = :field_id"""),
            {"slug": slug, "field_id": field_id}
        )

        if not result.fetchone():
            raise HTTPException(status_code=404, detail="Document type or field not found")

        # Delete field
        await session.execute(
            text("DELETE FROM document_type_fields WHERE id = :field_id"),
            {"field_id": field_id}
        )
        await session.commit()

        return {"ok": True, "message": "Field deleted"}


@router.post("/document-types/generate-prefill", response_model=PrefillResponse)
async def generate_prefill(request: PrefillRequest):
    """Generate description and extraction prompt preamble using LLM."""
    from app.services.llm_client import LLMClient
    from app.models.database import AppSetting
    from app.main import async_session_maker
    from sqlalchemy import select
    
    try:
        # Get active LLM provider
        async with async_session_maker() as session:
            result = await session.execute(
                select(AppSetting).where(AppSetting.key == "llm_provider")
            )
            setting = result.scalar_one_or_none()
            provider = setting.value if setting else "ollama"
        
        llm_client = LLMClient(provider=provider)
        
        # Create prompt for LLM
        prompt = f"""Je bent een assistent die helpt bij het opstellen van document type definities.

Document type naam: {request.name}
Belangrijke keywords: {request.keywords}

Genereer:
1. Een korte, professionele beschrijving (1-2 zinnen) van wat dit document type is
2. Een extractie prompt preamble die instructies geeft aan een LLM voor het extraheren van metadata uit dit type document

Antwoord in JSON formaat:
{{
  "description": "korte beschrijving hier",
  "extraction_prompt_preamble": "instructies voor metadata extractie hier"
}}

De extraction_prompt_preamble moet specifieke instructies bevatten over welke velden belangrijk zijn voor dit document type en hoe ze geëxtraheerd moeten worden."""
        
        response = await llm_client.generate_json(prompt)
        
        return PrefillResponse(
            description=response.get("description", ""),
            extraction_prompt_preamble=response.get("extraction_prompt_preamble", "")
        )
    except Exception as e:
        # Return empty strings if LLM fails
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to generate prefill via LLM: {e}")
        return PrefillResponse(
            description="",
            extraction_prompt_preamble=""
        )