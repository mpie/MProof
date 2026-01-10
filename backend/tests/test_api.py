import pytest
from httpx import AsyncClient
from fastapi import FastAPI
from app.main import app
from app.models.database import Subject, DocumentType


class TestHealthAPI:
    @pytest.mark.asyncio
    async def test_health_check(self, mock_llm_client):
        """Test health check endpoint."""
        # Mock the LLM client in the app
        app.dependency_overrides = {}

        async with AsyncClient(app=app, base_url="http://testserver") as client:
            response = await client.get("/api/health")

            assert response.status_code == 200
            data = response.json()
            assert data["ok"] is True
            assert "ollama" in data
            assert "reachable" in data["ollama"]
            assert "base_url" in data["ollama"]
            assert "model" in data["ollama"]


class TestSubjectsAPI:
    @pytest.mark.asyncio
    async def test_create_subject(self, db_session):
        """Test subject creation."""
        subject_data = {
            "name": "Test Subject",
            "context": "person"
        }

        async with AsyncClient(app=app, base_url="http://testserver") as client:
            response = await client.post("/api/subjects", json=subject_data)

            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Test Subject"
            assert data["context"] == "person"
            assert "id" in data

    @pytest.mark.asyncio
    async def test_create_duplicate_subject(self, db_session):
        """Test creating duplicate subject fails."""
        # First create a subject
        subject_data = {
            "name": "Duplicate Subject",
            "context": "company"
        }

        async with AsyncClient(app=app, base_url="http://testserver") as client:
            # Create first subject
            response1 = await client.post("/api/subjects", json=subject_data)
            assert response1.status_code == 200

            # Try to create duplicate
            response2 = await client.post("/api/subjects", json=subject_data)
            assert response2.status_code == 409

    @pytest.mark.asyncio
    async def test_search_subjects(self, db_session):
        """Test subject search."""
        # Create test subjects
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            await client.post("/api/subjects", json={"name": "John Doe", "context": "person"})
            await client.post("/api/subjects", json={"name": "Jane Smith", "context": "person"})

            # Search for "John"
            response = await client.get("/api/subjects?query=John&context=person")
            assert response.status_code == 200
            data = response.json()
            assert len(data["subjects"]) == 1
            assert data["subjects"][0]["name"] == "John Doe"


class TestDocumentTypesAPI:
    @pytest.mark.asyncio
    async def test_list_document_types(self):
        """Test listing document types."""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            response = await client.get("/api/document-types")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            # Should have seeded document types
            assert len(data) >= 3  # invoice, bank_statement, contract

    @pytest.mark.asyncio
    async def test_get_document_type(self):
        """Test getting specific document type."""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            response = await client.get("/api/document-types/invoice")

            assert response.status_code == 200
            data = response.json()
            assert data["slug"] == "invoice"
            assert data["name"] == "Invoice"
            assert "fields" in data
            assert len(data["fields"]) > 0

    @pytest.mark.asyncio
    async def test_get_nonexistent_document_type(self):
        """Test getting non-existent document type."""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            response = await client.get("/api/document-types/nonexistent")

            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_create_document_type(self):
        """Test creating new document type."""
        doc_type_data = {
            "name": "Test Document Type",
            "slug": "test_doc_type",
            "description": "A test document type",
            "classification_hints": "test keywords",
            "extraction_prompt_preamble": "Test preamble"
        }

        async with AsyncClient(app=app, base_url="http://testserver") as client:
            response = await client.post("/api/document-types", json=doc_type_data)

            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Test Document Type"
            assert data["slug"] == "test_doc_type"
            assert data["description"] == "A test document type"


class TestDocumentsAPI:
    @pytest.mark.asyncio
    async def test_list_documents_empty(self):
        """Test listing documents when none exist."""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            response = await client.get("/api/documents")

            assert response.status_code == 200
            data = response.json()
            assert data["documents"] == []
            assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self):
        """Test getting non-existent document."""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            response = await client.get("/api/documents/99999")

            assert response.status_code == 404


class TestUploadAPI:
    @pytest.mark.asyncio
    async def test_upload_without_subject(self):
        """Test upload without specifying subject."""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            # Create a dummy file
            files = {"file": ("test.pdf", b"dummy pdf content", "application/pdf")}

            response = await client.post("/api/upload", files=files)

            # Should fail because no subject_id provided
            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_upload_with_invalid_subject(self):
        """Test upload with invalid subject ID."""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            files = {"file": ("test.pdf", b"dummy pdf content", "application/pdf")}
            data = {"subject_id": "99999"}

            response = await client.post("/api/upload", files=files, data=data)

            assert response.status_code == 404