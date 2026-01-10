"""Tests for SSE (Server-Sent Events) service."""

import pytest
import asyncio
from unittest.mock import AsyncMock
from app.services.sse_service import SSEService


class TestSSEService:
    @pytest.fixture
    def sse_service(self):
        """Create SSE service instance."""
        return SSEService()

    @pytest.mark.asyncio
    async def test_subscribe_creates_connection(self, sse_service):
        """Test that subscribing creates a connection."""
        mock_send = AsyncMock()
        document_id = 1

        unsubscribe = await sse_service.subscribe(document_id, mock_send)

        # Check connection was created
        count = await sse_service.get_active_connections_count(document_id)
        assert count == 1

        # Cleanup
        await unsubscribe()

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_connection(self, sse_service):
        """Test that unsubscribing removes the connection."""
        mock_send = AsyncMock()
        document_id = 1

        unsubscribe = await sse_service.subscribe(document_id, mock_send)
        await unsubscribe()

        count = await sse_service.get_active_connections_count(document_id)
        assert count == 0

    @pytest.mark.asyncio
    async def test_publish_status_update(self, sse_service):
        """Test publishing status update."""
        mock_send = AsyncMock()
        document_id = 1

        await sse_service.subscribe(document_id, mock_send)
        await sse_service.publish_status_update(
            document_id,
            status="processing",
            stage="extracting_text",
            progress=50,
            updated_at="2024-01-01T00:00:00Z"
        )

        # Check that send was called
        mock_send.assert_called_once()
        call_args = mock_send.call_args[0][0]
        assert "event: document-update" in call_args
        assert "processing" in call_args
        assert "extracting_text" in call_args

    @pytest.mark.asyncio
    async def test_publish_result_update(self, sse_service):
        """Test publishing result update."""
        mock_send = AsyncMock()
        document_id = 1

        await sse_service.subscribe(document_id, mock_send)
        await sse_service.publish_result_update(
            document_id,
            doc_type_slug="invoice",
            confidence=0.95,
            metadata={"supplier": "ACME"},
            risk_score=25
        )

        mock_send.assert_called_once()
        call_args = mock_send.call_args[0][0]
        assert "invoice" in call_args
        assert "0.95" in call_args

    @pytest.mark.asyncio
    async def test_publish_error(self, sse_service):
        """Test publishing error event."""
        mock_send = AsyncMock()
        document_id = 1

        await sse_service.subscribe(document_id, mock_send)
        await sse_service.publish_error(document_id, "LLM request timed out")

        mock_send.assert_called_once()
        call_args = mock_send.call_args[0][0]
        assert "error" in call_args
        assert "LLM request timed out" in call_args

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, sse_service):
        """Test multiple subscribers receive updates."""
        mock_send1 = AsyncMock()
        mock_send2 = AsyncMock()
        document_id = 1

        await sse_service.subscribe(document_id, mock_send1)
        await sse_service.subscribe(document_id, mock_send2)

        await sse_service.publish_status_update(
            document_id,
            status="done",
            stage="completed",
            progress=100,
            updated_at="2024-01-01T00:00:00Z"
        )

        # Both should be called
        mock_send1.assert_called_once()
        mock_send2.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_to_nonexistent_document(self, sse_service):
        """Test publishing to document with no subscribers."""
        # Should not raise an error
        await sse_service.publish_status_update(
            document_id=999,
            status="processing",
            stage="test",
            progress=0,
            updated_at="2024-01-01T00:00:00Z"
        )

    @pytest.mark.asyncio
    async def test_disconnected_subscriber_removed(self, sse_service):
        """Test that disconnected subscribers are removed."""
        mock_send = AsyncMock()
        mock_send.side_effect = Exception("Connection closed")
        document_id = 1

        await sse_service.subscribe(document_id, mock_send)

        # Publish should handle the exception
        await sse_service.publish_status_update(
            document_id,
            status="processing",
            stage="test",
            progress=0,
            updated_at="2024-01-01T00:00:00Z"
        )

        # Subscriber should be removed
        count = await sse_service.get_active_connections_count(document_id)
        assert count == 0
