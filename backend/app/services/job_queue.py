import asyncio
import json
import logging
from typing import Callable, Awaitable, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.services.document_processor import DocumentProcessor
from app.services.llm_client import LLMClient
from app.services.sse_service import SSEService

logger = logging.getLogger(__name__)


class JobQueue:
    def __init__(self, db_session_factory: Callable[[], AsyncSession],
                 llm_client: LLMClient, sse_service: SSEService, max_workers: int = 3):
        self.db_session_factory = db_session_factory
        self.llm_client = llm_client
        self.sse_service = sse_service
        self._queue = asyncio.Queue()
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
        self._max_workers = max_workers
        self._enqueued: set[int] = set()
        self._active: set[str] = set()

    async def start(self) -> None:
        """Start the job queue workers."""
        if self._running:
            return

        self._running = True
        # Start multiple workers for parallel processing
        self._worker_tasks = [
            asyncio.create_task(self._worker_loop())
            for _ in range(self._max_workers)
        ]
        logger.info(f"Job queue started with {self._max_workers} workers for parallel processing")

        # Best-effort: requeue any documents that are still marked as queued in DB
        # (e.g. after a restart or when the queue wasn't initialized earlier).
        try:
            await self._requeue_queued_documents()
        except Exception as e:
            logger.warning(f"Failed to requeue queued documents: {e}")

    async def stop(self) -> None:
        """Stop the job queue workers."""
        if not self._running:
            return

        self._running = False
        # Cancel all worker tasks
        for task in self._worker_tasks:
            if task:
                task.cancel()
        
        # Wait for all tasks to complete cancellation
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        self._worker_tasks = []
        logger.info("Job queue workers stopped")

    async def enqueue_document_processing(self, document_id: int, model_name: str = None) -> None:
        """Enqueue a document for processing."""
        if document_id in self._enqueued:
            return
        self._enqueued.add(document_id)
        await self._queue.put(("process_document", document_id, model_name))
        logger.info(f"Enqueued document {document_id} for processing with model: {model_name or 'default'}")

    async def _worker_loop(self) -> None:
        """Main worker loop that processes jobs in parallel with other workers."""
        worker_id = id(asyncio.current_task())
        logger.debug(f"Worker {worker_id} started")
        while self._running:
            try:
                # Wait for a job with timeout
                try:
                    job_data = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Unpack job data (supports both old and new format)
                if len(job_data) == 3:
                    job_type, document_id, model_name = job_data
                else:
                    job_type, document_id = job_data
                    model_name = None

                logger.info(f"Processing job: {job_type} for document {document_id} with model: {model_name or 'default'}")

                try:
                    # Process the job
                    if job_type == "process_document":
                        await self._process_document_job(document_id, model_name)
                finally:
                    # Mark job as done + allow re-enqueue
                    self._enqueued.discard(document_id)
                    self._queue.task_done()

            except Exception as e:
                logger.error(f"Job processing error: {e}")
                # Continue processing other jobs

    async def _process_document_job(self, document_id: int, model_name: str = None) -> None:
        """Process a document processing job."""
        doc_id_str = str(document_id)
        self._active.add(doc_id_str)
        db_session = self.db_session_factory()

        try:
            processor = DocumentProcessor(db_session, self.llm_client, model_name=model_name)

            # Progress callback for SSE updates
            async def progress_callback(doc_id: int, progress: int, stage: str,
                                        doc_type_slug: str = None, doc_type_confidence: float = None) -> None:
                from datetime import datetime, timezone
                updated_at = datetime.now(timezone.utc).isoformat()
                await self.sse_service.publish_status_update(
                    doc_id, "processing", stage, progress, updated_at,
                    doc_type_slug=doc_type_slug, doc_type_confidence=doc_type_confidence
                )

            # Process the document
            await processor.process_document(document_id, progress_callback)

            # Publish completion event with final results (so the UI can update immediately)
            row = (await db_session.execute(
                text("""
                    SELECT doc_type_slug, doc_type_confidence, metadata_json, risk_score
                    FROM documents
                    WHERE id = :document_id
                """),
                {"document_id": document_id}
            )).fetchone()

            doc_type_slug = row.doc_type_slug if row else None
            confidence = row.doc_type_confidence if row else None
            risk_score = row.risk_score if row else None
            metadata = None
            if row and row.metadata_json:
                try:
                    metadata = json.loads(row.metadata_json) if isinstance(row.metadata_json, str) else row.metadata_json
                except Exception:
                    metadata = None

            await self.sse_service.publish_result_update(
                document_id,
                doc_type_slug=doc_type_slug,
                confidence=confidence,
                metadata=metadata,
                risk_score=risk_score,
            )

            logger.info(f"Document {document_id} processing completed")

        except Exception as e:
            logger.error(f"Document {document_id} processing failed: {e}")

            # Publish error event
            await self.sse_service.publish_error(document_id, str(e))

        finally:
            self._active.discard(doc_id_str)
            await db_session.close()

    async def _requeue_queued_documents(self) -> None:
        session = self.db_session_factory()
        try:
            rows = (await session.execute(
                text("SELECT id FROM documents WHERE status = 'queued' ORDER BY id ASC")
            )).fetchall()
            for row in rows:
                doc_id = int(row.id)
                await self.enqueue_document_processing(doc_id)
        finally:
            await session.close()

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
        return str(document_id) in self._active