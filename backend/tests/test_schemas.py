"""Tests for Pydantic schemas and data validation."""

import pytest
from datetime import datetime
from pydantic import ValidationError
from app.models.schemas import (
    SubjectCreate, SubjectResponse, DocumentStatusEnum,
    DocumentTypeCreate, DocumentTypeFieldCreate, EvidenceSpan
)


class TestSubjectSchemas:
    def test_subject_create_valid(self):
        """Test valid subject creation."""
        data = {"name": "Jan Jansen", "context": "person"}
        subject = SubjectCreate(**data)
        assert subject.name == "Jan Jansen"
        assert subject.context == "person"

    def test_subject_create_all_contexts(self):
        """Test all valid context types."""
        contexts = ["person", "company", "dossier", "other"]
        for ctx in contexts:
            subject = SubjectCreate(name="Test", context=ctx)
            assert subject.context == ctx

    def test_subject_create_invalid_context(self):
        """Test invalid context type raises error."""
        with pytest.raises(ValidationError):
            SubjectCreate(name="Test", context="invalid_context")

    def test_subject_create_empty_name_rejected(self):
        """Test empty name is rejected by validation."""
        with pytest.raises(ValidationError):
            SubjectCreate(name="", context="person")


class TestDocumentStatusEnum:
    def test_all_statuses(self):
        """Test all document statuses are defined."""
        statuses = ["pending", "queued", "processing", "done", "error"]
        for status in statuses:
            assert status in [e.value for e in DocumentStatusEnum]

    def test_status_values(self):
        """Test status enum values."""
        assert DocumentStatusEnum.pending.value == "pending"
        assert DocumentStatusEnum.queued.value == "queued"
        assert DocumentStatusEnum.processing.value == "processing"
        assert DocumentStatusEnum.done.value == "done"
        assert DocumentStatusEnum.error.value == "error"


class TestDocumentTypeSchemas:
    def test_document_type_create_minimal(self):
        """Test minimal document type creation."""
        data = {"name": "Test Type", "slug": "test-type"}
        doc_type = DocumentTypeCreate(**data)
        assert doc_type.name == "Test Type"
        assert doc_type.slug == "test-type"
        assert doc_type.description is None

    def test_document_type_create_full(self):
        """Test full document type creation."""
        data = {
            "name": "Invoice",
            "slug": "invoice",
            "description": "An invoice document",
            "classification_hints": "kw:factuur\nkw:invoice",
            "extraction_prompt_preamble": "Extract invoice fields"
        }
        doc_type = DocumentTypeCreate(**data)
        assert doc_type.name == "Invoice"
        assert doc_type.classification_hints == "kw:factuur\nkw:invoice"


class TestDocumentTypeFieldSchemas:
    def test_field_create_string(self):
        """Test string field creation."""
        data = {
            "key": "supplier_name",
            "label": "Supplier Name",
            "field_type": "string",
            "required": True
        }
        field = DocumentTypeFieldCreate(**data)
        assert field.key == "supplier_name"
        assert field.field_type == "string"
        assert field.required is True

    def test_field_create_enum(self):
        """Test enum field with values."""
        data = {
            "key": "status",
            "label": "Status",
            "field_type": "enum",
            "required": True,
            "enum_values": ["paid", "unpaid", "partial"]
        }
        field = DocumentTypeFieldCreate(**data)
        assert field.field_type == "enum"
        assert field.enum_values == ["paid", "unpaid", "partial"]

    def test_field_create_all_types(self):
        """Test all field types are valid."""
        types = ["string", "date", "number", "money", "currency", "iban", "enum"]
        for field_type in types:
            field = DocumentTypeFieldCreate(
                key="test_field",
                label="Test",
                field_type=field_type,
                required=False
            )
            assert field.field_type == field_type


class TestEvidenceSpan:
    def test_evidence_span_valid(self):
        """Test valid evidence span."""
        data = {"page": 0, "start": 10, "end": 20, "quote": "test value"}
        span = EvidenceSpan(**data)
        assert span.page == 0
        assert span.start == 10
        assert span.end == 20
        assert span.quote == "test value"

    def test_evidence_span_without_quote(self):
        """Test evidence span without quote (optional)."""
        data = {"page": 0, "start": 10, "end": 20}
        span = EvidenceSpan(**data)
        assert span.quote == ""  # Default empty string

    def test_evidence_span_negative_page(self):
        """Test negative page number raises error."""
        with pytest.raises(ValidationError):
            EvidenceSpan(page=-1, start=10, end=20)

    def test_evidence_span_negative_positions(self):
        """Test negative positions raise error."""
        with pytest.raises(ValidationError):
            EvidenceSpan(page=0, start=-1, end=20)
        
        with pytest.raises(ValidationError):
            EvidenceSpan(page=0, start=10, end=-1)
