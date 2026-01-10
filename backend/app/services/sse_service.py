import json
import asyncio
from typing import Dict, Set, Callable
import logging

logger = logging.getLogger(__name__)


class SSEService:
    def __init__(self):
        self._connections: Dict[int, Set[Callable]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, document_id: int, send_func: Callable[[str], None]) -> Callable:
        """Subscribe to document events. Returns unsubscribe function."""
        async with self._lock:
            if document_id not in self._connections:
                self._connections[document_id] = set()

            self._connections[document_id].add(send_func)

        # Return unsubscribe function
        async def unsubscribe():
            async with self._lock:
                if document_id in self._connections:
                    self._connections[document_id].discard(send_func)
                    if not self._connections[document_id]:
                        del self._connections[document_id]

        return unsubscribe

    async def publish_status_update(self, document_id: int, status: str, stage: str,
                                  progress: int, updated_at: str) -> None:
        """Publish status update event."""
        event_data = {
            "type": "status",
            "status": status,
            "stage": stage,
            "progress": progress,
            "updated_at": updated_at
        }

        await self._publish_event(document_id, event_data)

    async def publish_result_update(self, document_id: int, doc_type_slug: str = None,
                                  confidence: float = None, metadata: dict = None,
                                  risk_score: int = None, **kwargs) -> None:
        """Publish result update event."""
        event_data = {
            "type": "result",
            "doc_type_slug": doc_type_slug,
            "confidence": confidence,
            "metadata": metadata,
            "risk_score": risk_score,
            **kwargs
        }

        await self._publish_event(document_id, event_data)

    async def publish_error(self, document_id: int, error_message: str) -> None:
        """Publish error event."""
        event_data = {
            "type": "error",
            "error_message": error_message
        }

        await self._publish_event(document_id, event_data)

    async def _publish_event(self, document_id: int, event_data: dict) -> None:
        """Publish event to all subscribers of a document."""
        async with self._lock:
            if document_id not in self._connections:
                return

            # Convert to SSE format
            event_lines = [
                "event: document-update",
                f"data: {json.dumps(event_data)}",
                ""  # Empty line to end the event
            ]
            # Important: SSE events must end with a blank line (\n\n)
            message = "\n".join(event_lines) + "\n"

            # Send to all subscribers
            disconnected = set()
            for send_func in self._connections[document_id]:
                try:
                    await send_func(message)
                except Exception as e:
                    logger.warning(f"Failed to send SSE to subscriber: {e}")
                    disconnected.add(send_func)

            # Remove disconnected subscribers
            for send_func in disconnected:
                self._connections[document_id].discard(send_func)

            if not self._connections[document_id]:
                del self._connections[document_id]

    async def get_active_connections_count(self, document_id: int) -> int:
        """Get number of active connections for a document."""
        async with self._lock:
            return len(self._connections.get(document_id, set()))