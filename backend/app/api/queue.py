from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

router = APIRouter()


@router.get("/queue/status")
async def get_queue_status():
    """Get current queue status."""
    from app.main import job_queue
    if job_queue is None:
        return {"queue_size": 0, "is_running": False, "error": "Job queue not initialized"}
    return await job_queue.get_queue_status()


@router.post("/queue/reprocess/{document_id}")
async def reprocess_document(document_id: int, model_name: str = None, force_doc_type: str = None):
    """Re-enqueue a document for processing (resets status to queued)."""
    from app.main import job_queue, async_session_maker
    from app.services.job_queue import JobQueue

    if job_queue is None:
        raise HTTPException(status_code=503, detail="Job queue not initialized")

    async with async_session_maker() as session:
        row = (await session.execute(
            text("SELECT id, status FROM documents WHERE id = :id"),
            {"id": document_id}
        )).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        # Reset to queued so it can be re-processed
        await session.execute(
            text("UPDATE documents SET status = 'queued', progress = 0, stage = NULL, error_message = NULL, updated_at = NOW() WHERE id = :id"),
            {"id": document_id}
        )
        await session.commit()

    # Remove from enqueued set so it can be re-added
    job_queue._enqueued.discard(document_id)
    await job_queue.enqueue_document_processing(document_id, model_name=model_name, force_doc_type=force_doc_type)
    return {"status": "enqueued", "document_id": document_id}


@router.post("/queue/cancel/{document_id}")
async def cancel_document(document_id: int):
    """Cancel a queued or processing document — sets status to error without deleting."""
    from app.main import job_queue, async_session_maker

    if job_queue is None:
        raise HTTPException(status_code=503, detail="Job queue not initialized")

    async with async_session_maker() as session:
        row = (await session.execute(
            text("SELECT id, status FROM documents WHERE id = :id"),
            {"id": document_id}
        )).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        await session.execute(
            text("""UPDATE documents SET status = 'error', error_message = 'Geannuleerd door gebruiker',
                    updated_at = :now WHERE id = :id"""),
            {"id": document_id, "now": datetime.now(timezone.utc)},
        )
        await session.commit()

    # Remove from queue so it won't start if still waiting
    job_queue._enqueued.discard(document_id)
    job_queue._cancelled.add(document_id)
    return {"status": "cancelled", "document_id": document_id}