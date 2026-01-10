import asyncio
import logging
from typing import Callable, Awaitable
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.document_processor import DocumentProcessor
from app.services.llm_client import LLMClient
from app.services.sse_service import SSEService

logger = logging.getLogger(__name__)


class JobQueue:
    def __init__(self, db_session_factory: Callable[[], AsyncSession],
                 llm_client: LLMClient, sse_service: SSEService):
        self.db_session_factory = db_session_factory
        self.llm_client = llm_client
        self.sse_service = sse_service
        self._queue = asyncio.Queue()
        self._running = False
        self._worker_task: asyncio.Task = None

    async def start(self) -> None:
        """Start the job queue worker."""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Job queue worker started")

    async def stop(self) -> None:
        """Stop the job queue worker."""
        if not self._running:
            return

        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        logger.info("Job queue worker stopped")

    async def enqueue_document_processing(self, document_id: int) -> None:
        """Enqueue a document for processing."""
        await self._queue.put(("process_document", document_id))
        logger.info(f"Enqueued document {document_id} for processing")

    async def _worker_loop(self) -> None:
        """Main worker loop that processes jobs sequentially."""
        while self._running:
            try:
                # Wait for a job with timeout
                try:
                    job_type, document_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                logger.info(f"Processing job: {job_type} for document {document_id}")

                # Process the job
                if job_type == "process_document":
                    await self._process_document_job(document_id)

                # Mark job as done
                self._queue.task_done()

            except Exception as e:
                logger.error(f"Job processing error: {e}")
                # Continue processing other jobs

    async def _process_document_job(self, document_id: int) -> None:
        """Process a document processing job."""
        db_session = self.db_session_factory()

        try:
            processor = DocumentProcessor(db_session, self.llm_client)

            # Progress callback for SSE updates
            async def progress_callback(doc_id: int, progress: int, stage: str) -> None:
                from datetime import datetime
                updated_at = datetime.utcnow().isoformat()
                await self.sse_service.publish_status_update(
                    doc_id, "processing", stage, progress, updated_at
                )

            # Process the document
            await processor.process_document(document_id, progress_callback)

            # Publish completion event
            await self.sse_service.publish_result_update(
                document_id,
                doc_type_slug=None,  # Will be fetched from DB
                confidence=None,
                metadata=None,
                risk_score=None
            )

            logger.info(f"Document {document_id} processing completed")

        except Exception as e:
            logger.error(f"Document {document_id} processing failed: {e}")

            # Publish error event
            await self.sse_service.publish_error(document_id, str(e))

        finally:
            await db_session.close()

    async def get_queue_size(self) -> int:
        """Get the current queue size."""
        return self._queue.qsize()

    async def get_queue_status(self) -> dict:
        """Get detailed queue status."""
        return {
            "queue_size": self._queue.qsize(),
            "is_running": self._running,
        }

    async def is_processing(self, document_id: int) -> bool:
        """Check if a document is currently being processed."""
        # This is a simple implementation - in production you'd track active jobs
        return False  # TODO: Implement proper tracking