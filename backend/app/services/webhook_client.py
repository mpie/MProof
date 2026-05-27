import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_BACKOFF = [0, 5, 30]  # seconds before each attempt


async def fire_webhook(callback_url: str, payload: Dict[str, Any]) -> None:
    """POST payload to callback_url with retries. Non-blocking — caller does not await result."""
    for attempt in range(_MAX_ATTEMPTS):
        delay = _BACKOFF[attempt]
        if delay:
            await asyncio.sleep(delay)
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    callback_url,
                    json=payload,
                    headers={"Content-Type": "application/json", "X-MProof-Event": "document.done"},
                )
                if resp.status_code < 300:
                    logger.info(f"Webhook delivered to {callback_url} (attempt {attempt + 1}, status {resp.status_code})")
                    return
                logger.warning(
                    f"Webhook {callback_url} returned {resp.status_code} (attempt {attempt + 1})"
                )
        except Exception as exc:
            logger.warning(f"Webhook {callback_url} failed (attempt {attempt + 1}): {exc}")

    logger.error(f"Webhook {callback_url} failed after {_MAX_ATTEMPTS} attempts — giving up")


def build_webhook_payload(
    document_id: int,
    status: str,
    external_reference: Optional[str],
    doc_type_slug: Optional[str],
    doc_type_confidence: Optional[float],
    risk_score: Optional[int],
    metadata_json: Optional[Dict[str, Any]],
    missing_required_fields: Optional[list],
    error_message: Optional[str],
) -> Dict[str, Any]:
    return {
        "event": "document.done" if status == "done" else "document.error",
        "document_id": document_id,
        "external_reference": external_reference,
        "status": status,
        "doc_type_slug": doc_type_slug,
        "doc_type_confidence": round(doc_type_confidence, 4) if doc_type_confidence is not None else None,
        "risk_score": risk_score,
        "extracted_fields": metadata_json or {},
        "missing_required_fields": missing_required_fields or [],
        "error_message": error_message,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
