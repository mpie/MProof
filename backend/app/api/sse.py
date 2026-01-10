from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
from typing import Callable
from app.services.sse_service import SSEService

router = APIRouter()


@router.get("/documents/{document_id}/events")
async def document_events(document_id: int):
    """SSE endpoint for real-time document processing events."""
    from app.main import sse_service

    # Verify document exists (simplified check)
    from app.main import async_session_maker
    async with async_session_maker() as session:
        from sqlalchemy import text
        result = await session.execute(
            text("SELECT id FROM documents WHERE id = :document_id"),
            {"document_id": document_id}
        )
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="Document not found")

    async def event_generator():
        """Generate SSE events for the document."""
        messages = []

        async def send_message(message: str):
            messages.append(message)

        # Subscribe to document events
        unsubscribe = await sse_service.subscribe(document_id, send_message)

        try:
            # Send initial connection event
            yield "event: connected\ndata: {}\n\n"

            # Process queued messages and new events
            while True:
                # Send any queued messages
                while messages:
                    yield messages.pop(0)

                # Wait a bit before checking again
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            # Client disconnected
            pass
        finally:
            # Clean up subscription
            await unsubscribe()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        }
    )