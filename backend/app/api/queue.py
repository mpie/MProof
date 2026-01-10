from fastapi import APIRouter

router = APIRouter()


@router.get("/queue/status")
async def get_queue_status():
    """Get current queue status."""
    from app.main import job_queue
    if job_queue is None:
        return {"queue_size": 0, "is_running": False, "error": "Job queue not initialized"}
    return await job_queue.get_queue_status()