import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, NamedTuple, Set
from datetime import datetime, timezone
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
from pypdf import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract
from docx import Document as DocxDocument
from openpyxl import load_workbook
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.models.database import Document, DocumentType, DocumentTypeField
from app.models.schemas import (
    OCRResult, ClassificationResult, ExtractionEvidence,
    EvidenceSpan, RiskAnalysis, RiskSignal
)
from app.services.llm_client import LLMClient
from app.config import settings

logger = logging.getLogger(__name__)

# Global semaphore for heavy OCR/PDF operations
OCR_SEMAPHORE = asyncio.Semaphore(2)


class TextPrepareResult(NamedTuple):
    """Result of text preparation with skip marker info."""
    text: str
    skip_marker_used: Optional[str] = None
    skip_marker_position: Optional[int] = None


from app.services.text_extractor import TextExtractorMixin
from app.services.classification_mixin import ClassificationMixin
from app.services.metadata_extractor import MetadataExtractorMixin
from app.services.extraction_utils import ExtractionUtilsMixin
from app.services.fraud_stage import FraudStageMixin


class DocumentProcessor(TextExtractorMixin, ClassificationMixin, MetadataExtractorMixin, ExtractionUtilsMixin, FraudStageMixin):
    def __init__(self, db_session: AsyncSession, llm_client: LLMClient, model_name: str = None, force_doc_type: str = None):
        self.db = db_session
        self.llm = llm_client
        self.data_dir = Path(settings.data_dir)
        self.model_name = model_name
        self.force_doc_type = force_doc_type  # Skip classification, use this type directly
        self._skip_marker_used: Optional[str] = None
        self._skip_marker_position: Optional[int] = None

    def _json_serialize(self, obj: Any) -> Any:
        """Recursively convert Pydantic models and other objects to JSON-serializable dicts."""
        if hasattr(obj, 'model_dump'):
            # Pydantic v2
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            # Pydantic v1
            return obj.dict()
        elif isinstance(obj, dict):
            return {key: self._json_serialize(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._json_serialize(item) for item in obj]
        else:
            return obj

    async def process_document(self, document_id: int, progress_callback: callable = None) -> None:
        """Main processing pipeline for a document."""
        # Reset skip marker tracking for this document
        self._skip_marker_used = None
        self._skip_marker_position = None

        try:
            document = await self._get_document(document_id)
            if not document:
                raise ValueError(f"Document {document_id} not found")

            document_dir = self.data_dir / "subjects" / str(document.subject_id) / "documents" / str(document_id)
            document_dir.mkdir(parents=True, exist_ok=True)

            # Clean old LLM artifacts before re-analysis to prevent confusion
            llm_dir = document_dir / "llm"
            if llm_dir.exists():
                logger.info(f"Cleaning old LLM artifacts for document {document_id}")
                for artifact in llm_dir.glob("*"):
                    try:
                        artifact.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete artifact {artifact}: {e}")
            # Stage 1: Sniffing (0-10%)
            await self._update_progress(document_id, 0, "sniffing", progress_callback)
            await self._stage_sniffing(document, document_dir)
            await self._update_progress(document_id, 10, "sniffing", progress_callback)

            # Stage 2: Text Extraction (10-45%)
            await self._update_progress(document_id, 10, "extracting_text", progress_callback)
            ocr_result = await self._stage_text_extraction(document, document_dir, progress_callback=progress_callback)
            await self._update_progress(document_id, 45, "extracting_text", progress_callback)

            # Stage 3: Classification (45-60%)
            await self._update_progress(document_id, 45, "classifying", progress_callback)
            classification = await self._with_progress_tick(
                self._stage_classification(document, document_dir, ocr_result),
                document_id, 45, 60, "classifying", progress_callback, interval=1.5,
            )
            # Send classification result immediately so UI can show the type
            await self._update_progress(
                document_id, 60, "extracting_metadata", progress_callback,
                doc_type_slug=classification.doc_type_slug,
                doc_type_confidence=classification.confidence
            )

            # Stage 4: Metadata Extraction (60-85%)
            logger.info(f"Document {document.id}: Classification result - doc_type_slug='{classification.doc_type_slug}', confidence={classification.confidence:.2f}, rationale='{classification.rationale}'")

            # Start metadata extraction
            extraction_result = None
            if classification.doc_type_slug not in ["other", "unknown"]:
                logger.info(f"Document {document.id}: Starting LLM metadata extraction for type '{classification.doc_type_slug}'")
                extraction_result = await self._stage_metadata_extraction(
                    document, document_dir, classification, ocr_result, progress_callback
                )
                if extraction_result:
                    logger.info(f"Document {document.id}: LLM metadata extraction completed successfully")
                else:
                    logger.warning(f"Document {document.id}: LLM metadata extraction returned None (may have been skipped - check extraction_skipped.json)")
            else:
                logger.info(f"Document {document.id}: Skipping metadata extraction for type '{classification.doc_type_slug}' (other/unknown)")

            await self._update_progress(document_id, 85, "extracting_metadata", progress_callback)

            # Stage 5: Unified fraud analysis (85-100%)
            await self._update_progress(document_id, 85, "risk_signals", progress_callback)
            risk_analysis = await self._with_progress_tick(
                self._stage_unified_fraud_analysis(
                    document, document_dir, ocr_result, classification, extraction_result,
                ),
                document_id, 85, 100, "risk_signals", progress_callback, interval=2.0,
            )

            await self._update_progress(document_id, 100, "completed", progress_callback)

            # Final update
            await self._finalize_document(document_id, classification, extraction_result, risk_analysis, ocr_result, document_dir)

        except Exception as e:
            logger.error(f"Document processing failed for {document_id}: {e}")
            await self._update_document_error(document_id, str(e))
            raise

    async def _stage_sniffing(self, document: Document, document_dir: Path) -> None:
        """Stage 1: File type detection and SHA256 computation."""
        if "\x00" in (document.original_filename or ""):
            raise ValueError("Invalid filename (embedded null byte)")
        file_path = document_dir / "original" / document.original_filename

        # Compute SHA256
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)

        # Detect MIME type
        if HAS_MAGIC:
            mime_type = magic.from_file(str(file_path), mime=True)
        else:
            # Fallback to extension-based detection
            ext = file_path.suffix.lower()
            mime_types = {
                '.pdf': 'application/pdf',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.doc': 'application/msword',
                '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                '.xls': 'application/vnd.ms-excel',
            }
            mime_type = mime_types.get(ext, 'application/octet-stream')

        # Update document
        await self.db.execute(
            text("UPDATE documents SET sha256 = :sha256, mime_type = :mime_type WHERE id = :document_id"),
            {"sha256": sha256.hexdigest(), "mime_type": mime_type, "document_id": document.id}
        )
        await self.db.commit()

    async def _with_progress_tick(self, coro, document_id: int, start_pct: int, end_pct: int,
                                  stage: str, progress_callback, interval: float = 2.5):
        """Run coro while smoothly ticking progress from start_pct toward end_pct."""
        if not progress_callback:
            return await coro

        steps = max(1, end_pct - start_pct - 1)
        step_size = max(1, steps // max(1, int((end_pct - start_pct) / interval * 0.8)))
        current = [start_pct]

        async def ticker():
            while True:
                await asyncio.sleep(interval)
                if current[0] < end_pct - 1:
                    current[0] = min(current[0] + step_size, end_pct - 1)
                    try:
                        await progress_callback(document_id, current[0], stage)
                    except Exception:
                        pass

        ticker_task = asyncio.create_task(ticker())
        try:
            return await coro
        finally:
            ticker_task.cancel()
            try:
                await ticker_task
            except asyncio.CancelledError:
                pass

    async def _update_progress(self, document_id: int, progress: int, stage: str,
                             callback: callable = None,
                             doc_type_slug: str = None, doc_type_confidence: float = None) -> None:
        """Update document progress and call callback if provided."""
        await self.db.execute(
            text("UPDATE documents SET status = 'processing', progress = :progress, stage = :stage, updated_at = :updated_at WHERE id = :document_id"),
            {"progress": progress, "stage": stage, "document_id": document_id, "updated_at": datetime.now(timezone.utc)}
        )
        await self.db.commit()

        if callback:
            await callback(document_id, progress, stage, doc_type_slug, doc_type_confidence)

    async def _finalize_document(self, document_id: int, classification: ClassificationResult,
                               extraction_result: Optional[ExtractionEvidence],
                               risk_analysis: RiskAnalysis, ocr_result: OCRResult, document_dir: Path) -> None:
        """Finalize document processing."""
        # Try to load classification scores from classification_local.json
        classification_scores = None
        classification_file = document_dir / "llm" / "classification_local.json"
        if classification_file.exists():
            try:
                with open(classification_file, "r", encoding="utf-8") as f:
                    classification_data = json.load(f)
                    # Extract NB and BERT scores if available
                    classification_scores = {}
                    if "naive_bayes" in classification_data:
                        classification_scores["naive_bayes"] = classification_data["naive_bayes"]
                    if "bert" in classification_data:
                        classification_scores["bert"] = classification_data["bert"]
            except Exception as e:
                logger.warning(f"Failed to load classification scores: {e}")

        update_data = {
            "status": "done",
            "progress": 100,
            "stage": "completed",
            "doc_type_slug": classification.doc_type_slug,
            "doc_type_confidence": classification.confidence,
            "doc_type_rationale": classification.rationale,
            "risk_score": risk_analysis.risk_score,
            "risk_signals_json": json.dumps([s.dict() for s in risk_analysis.signals]),
            "ocr_used": ocr_result.ocr_used,
            "ocr_quality": ocr_result.ocr_quality,
            "skip_marker_used": self._skip_marker_used,
            "skip_marker_position": self._skip_marker_position,
            "updated_at": datetime.now(timezone.utc)
        }

        if extraction_result:
            update_data["metadata_json"] = json.dumps(extraction_result.data)
            # Load merged evidence from file (includes auto-found evidence)
            evidence_file = document_dir / "metadata" / "evidence.json"
            if evidence_file.exists():
                try:
                    with open(evidence_file, "r", encoding="utf-8") as f:
                        merged_evidence = json.load(f)
                    update_data["metadata_evidence_json"] = json.dumps(merged_evidence)
                except Exception as e:
                    logger.warning(f"Failed to load merged evidence from file: {e}")
                    update_data["metadata_evidence_json"] = json.dumps(self._json_serialize(extraction_result.evidence))
            else:
                update_data["metadata_evidence_json"] = json.dumps(self._json_serialize(extraction_result.evidence))

        semantic_context = self._load_semantic_context(document_dir)

        # Store classification scores and semantic context in metadata_validation_json (reusing existing JSON field)
        if classification_scores:
            validation_payload = {"classification_scores": classification_scores}
            if semantic_context:
                validation_payload["semantic_context"] = semantic_context
            update_data["metadata_validation_json"] = json.dumps(validation_payload)
        elif semantic_context:
            update_data["metadata_validation_json"] = json.dumps({"semantic_context": semantic_context})

        # Build query with named parameters
        set_parts = [f"{k} = :{k}" for k in update_data.keys()]
        query = f"UPDATE documents SET {', '.join(set_parts)} WHERE id = :document_id"
        update_data["document_id"] = document_id

        await self.db.execute(text(query), update_data)
        await self.db.commit()

    async def _update_document_error(self, document_id: int, error_message: str) -> None:
        """Mark document as error."""
        await self.db.execute(
            text("UPDATE documents SET status = 'error', error_message = :error_message, updated_at = :updated_at WHERE id = :document_id"),
            {"error_message": error_message, "document_id": document_id, "updated_at": datetime.now(timezone.utc)}
        )
        await self.db.commit()

    async def _get_document(self, document_id: int) -> Optional[Document]:
        """Get document by ID."""
        result = await self.db.execute(text("SELECT * FROM documents WHERE id = :document_id"), {"document_id": document_id})
        row = result.fetchone()
        if row:
            return Document(**row._mapping)
        return None
