import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
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


class TextPrepareResult(NamedTuple):
    """Result of text preparation with skip marker info."""
    text: str
    skip_marker_used: Optional[str] = None
    skip_marker_position: Optional[int] = None


class DocumentProcessor:
    def __init__(self, db_session: AsyncSession, llm_client: LLMClient, model_name: str = None):
        self.db = db_session
        self.llm = llm_client
        self.data_dir = Path(settings.data_dir)
        self.model_name = model_name  # Model to use for classification
        # Track skip marker usage during processing
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

        try:
            # Stage 1: Sniffing (0-10%)
            await self._update_progress(document_id, 0, "sniffing", progress_callback)
            await self._stage_sniffing(document, document_dir)
            await self._update_progress(document_id, 10, "sniffing", progress_callback)

            # Stage 2: Text Extraction (10-45%)
            await self._update_progress(document_id, 10, "extracting_text", progress_callback)
            ocr_result = await self._stage_text_extraction(document, document_dir)
            await self._update_progress(document_id, 45, "extracting_text", progress_callback)

            # Stage 3: Classification (45-60%)
            await self._update_progress(document_id, 45, "classifying", progress_callback)
            classification = await self._stage_classification(document, document_dir, ocr_result)
            # Send classification result immediately so UI can show the type
            await self._update_progress(
                document_id, 60, "extracting_metadata", progress_callback,
                doc_type_slug=classification.doc_type_slug,
                doc_type_confidence=classification.confidence
            )

            # Stage 4: Metadata Extraction (60-85%)
            if classification.doc_type_slug not in ["other", "unknown"]:
                extraction_result = await self._stage_metadata_extraction(
                    document, document_dir, classification, ocr_result, progress_callback
                )
            else:
                extraction_result = None
                logger.info(f"Document {document.id}: Skipping metadata extraction for type '{classification.doc_type_slug}'")
            await self._update_progress(document_id, 85, "completed", progress_callback)
            await self._update_progress(document_id, 85, "risk_signals", progress_callback)

            # Stage 5: Risk Analysis (85-100%)
            risk_analysis = await self._stage_risk_analysis(document, document_dir, ocr_result, extraction_result)
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

    async def _stage_text_extraction(self, document: Document, document_dir: Path) -> OCRResult:
        """Stage 2: Extract text from document using OCR if needed."""
        if "\x00" in (document.original_filename or ""):
            raise ValueError("Invalid filename (embedded null byte)")
        file_path = document_dir / "original" / document.original_filename
        text_dir = document_dir / "text"
        text_dir.mkdir(exist_ok=True)

        mime_type = document.mime_type
        pages = []
        combined_text = ""
        ocr_used = False

        try:
            if mime_type == "application/pdf":
                pages, combined_text, ocr_used = await self._extract_pdf_text(file_path)
            elif mime_type.startswith("image/"):
                pages, combined_text, ocr_used = await self._extract_image_text(file_path)
            elif mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                              "application/msword"]:
                pages, combined_text, ocr_used = await self._extract_docx_text(file_path)
            elif mime_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                              "application/vnd.ms-excel"]:
                pages, combined_text, ocr_used = await self._extract_xlsx_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {mime_type}")

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise

        # Determine OCR quality
        ocr_quality = self._assess_ocr_quality(combined_text, ocr_used)

        result = OCRResult(
            pages=pages,
            combined_text=combined_text,
            ocr_used=ocr_used,
            ocr_quality=ocr_quality
        )

        # Save artifacts
        with open(text_dir / "extracted.json", "w") as f:
            json.dump(result.dict(), f, indent=2)

        with open(text_dir / "extracted.txt", "w") as f:
            f.write(combined_text)

        return result

    def _ocr_with_rotation_detection(self, img: Image.Image) -> str:
        """Perform OCR on image, trying different rotations (0, 90, 180, 270) and return best result.
        
        Args:
            img: PIL Image to perform OCR on
            
        Returns:
            Best OCR text result from all rotations
        """
        results = []
        
        # Try all rotations: 0, 90, 180, 270 degrees
        for angle in [0, 90, 180, 270]:
            try:
                # Rotate image
                if angle == 0:
                    rotated_img = img
                else:
                    rotated_img = img.rotate(-angle, expand=True)  # Negative for counter-clockwise
                
                # Perform OCR
                text = pytesseract.image_to_string(rotated_img, config=settings.tesseract_config)
                
                # Score the result: count alphanumeric characters and words
                alnum_count = sum(1 for c in text if c.isalnum())
                word_count = len(text.split())
                
                # Prefer results with more alphanumeric characters and words
                score = alnum_count * 2 + word_count
                
                results.append({
                    'angle': angle,
                    'text': text,
                    'score': score,
                    'alnum_count': alnum_count,
                    'word_count': word_count
                })
                
                logger.debug(f"OCR rotation {angle}°: {alnum_count} alnum chars, {word_count} words, score: {score}")
                
            except Exception as e:
                logger.warning(f"OCR failed for rotation {angle}°: {e}")
                continue
        
        if not results:
            logger.warning("All OCR rotations failed, returning empty string")
            return ""
        
        # Sort by score (descending) and return best result
        results.sort(key=lambda x: x['score'], reverse=True)
        best = results[0]
        
        if best['angle'] != 0:
            logger.info(f"Best OCR result found at {best['angle']}° rotation ({best['alnum_count']} alnum chars, {best['word_count']} words)")
        
        return best['text']

    async def _extract_pdf_text(self, file_path: Path) -> Tuple[List[Dict[str, Any]], str, bool]:
        """Extract text from PDF, using OCR if text extraction yields poor results."""
        pages = []
        combined_text = ""
        ocr_used = False

        # Try text extraction first
        try:
            doc = fitz.open(str(file_path))
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)

                # Try to extract text
                text = page.get_text()
                source = "text-layer"

                # Check if text extraction is poor (less than 200 chars or mostly empty)
                if len(text.strip()) < 200 or self._is_mostly_empty(text):
                    # Fall back to OCR with rotation detection
                    pix = page.get_pixmap(dpi=250)
                    img = Image.open(BytesIO(pix.tobytes("png")))
                    text = self._ocr_with_rotation_detection(img)
                    source = "ocr"
                    ocr_used = True

                pages.append({
                    "page": page_num,
                    "source": source,
                    "text": text
                })
                combined_text += text + "\n"

            doc.close()

        except Exception as e:
            logger.warning(f"PDF text extraction failed, trying OCR: {e}")
            # Full OCR fallback with rotation detection
            doc = fitz.open(str(file_path))
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=250)
                img = Image.open(BytesIO(pix.tobytes("png")))
                text = self._ocr_with_rotation_detection(img)

                pages.append({
                    "page": page_num,
                    "source": "ocr",
                    "text": text
                })
                combined_text += text + "\n"
                ocr_used = True

            doc.close()

        return pages, combined_text, ocr_used

    async def _extract_image_text(self, file_path: Path) -> Tuple[List[Dict[str, Any]], str, bool]:
        """Extract text from image using OCR with rotation detection."""
        img = Image.open(file_path)
        text = self._ocr_with_rotation_detection(img)

        pages = [{
            "page": 0,
            "source": "ocr",
            "text": text
        }]

        return pages, text, True

    async def _extract_docx_text(self, file_path: Path) -> Tuple[List[Dict[str, Any]], str, bool]:
        """Extract text from DOCX file."""
        doc = DocxDocument(file_path)
        text = ""

        for para in doc.paragraphs:
            text += para.text + "\n"

        pages = [{
            "page": 0,
            "source": "docx",
            "text": text
        }]

        return pages, text, False

    async def _extract_xlsx_text(self, file_path: Path) -> Tuple[List[Dict[str, Any]], str, bool]:
        """Extract text from XLSX file."""
        wb = load_workbook(file_path, read_only=True)
        text = ""

        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            text += f"Sheet: {sheet_name}\n"

            for row in sheet.iter_rows(values_only=True):
                row_text = "\t".join(str(cell) if cell is not None else "" for cell in row)
                text += row_text + "\n"

            text += "\n"

        pages = [{
            "page": 0,
            "source": "xlsx",
            "text": text
        }]

        return pages, text, False

    def _is_mostly_empty(self, text: str) -> bool:
        """Check if text is mostly empty or contains mostly symbols."""
        # Remove whitespace and check ratio of alphanumeric characters
        cleaned = re.sub(r'\s+', '', text)
        if len(cleaned) == 0:
            return True

        alpha_numeric = len(re.findall(r'[a-zA-Z0-9]', cleaned))
        return (alpha_numeric / len(cleaned)) < 0.1

    def _assess_ocr_quality(self, text: str, ocr_used: bool) -> str:
        """Assess OCR quality based on text characteristics."""
        if not ocr_used:
            return "high"

        cleaned = re.sub(r'\s+', '', text)
        if len(cleaned) < 100:
            return "low"

        # Check for common OCR errors
        error_indicators = [
            r'[^\x00-\x7F]',  # Non-ASCII characters (could be OCR artifacts)
            r'\d{10,}',      # Very long numbers (potential OCR errors)
            r'[|@#$%^&*]{3,}',  # Multiple special characters together
        ]

        error_score = 0
        for pattern in error_indicators:
            error_score += len(re.findall(pattern, text))

        if error_score > len(cleaned) * 0.05:  # More than 5% potential errors
            return "low"
        elif error_score > len(cleaned) * 0.02:  # More than 2% potential errors
            return "medium"
        else:
            return "high"

    def classify_deterministic(self, text: str, available_types: List[Tuple[str, str]]) -> Optional[str]:
        """Deterministic pre-classifier that only returns a type if there's strong evidence.

        Args:
            text: Document text to analyze
            available_types: List of (slug, classification_hints) tuples

        Returns:
            Document type slug if strong evidence exists, None otherwise
        """
        text_lower = text.lower()
        scores = {}

        for slug, hints in available_types:
            if not hints:
                continue

            score = 0
            disqualified = False

            # Parse hints - supports both structured (kw:, re:, not:) and simple keyword format
            for hint_line in hints.strip().split('\n'):
                hint_line = hint_line.strip()
                if not hint_line:
                    continue

                if hint_line.startswith('kw:'):
                    # Structured: Required keywords (case-insensitive)
                    keyword = hint_line[3:].strip().lower()
                    if keyword in text_lower:
                        score += 1
                elif hint_line.startswith('re:'):
                    # Structured: Regex patterns
                    try:
                        pattern = hint_line[3:].strip()
                        if re.search(pattern, text, re.IGNORECASE):
                            score += 3  # Regex matches are worth more
                    except re.error:
                        logger.warning(f"Invalid regex pattern in hints for {slug}: {pattern}")
                elif hint_line.startswith('not:'):
                    # Structured: Negative keywords (must NOT appear)
                    negative_word = hint_line[4:].strip().lower()
                    if negative_word in text_lower:
                        disqualified = True
                        break
                else:
                    # Legacy format: treat as simple keyword (case-insensitive)
                    keyword = hint_line.lower()
                    if keyword in text_lower:
                        score += 1
                        logger.debug(f"Legacy keyword match '{keyword}' for {slug}")

            if disqualified:
                continue

            if score > 0:
                scores[slug] = score
                logger.debug(f"Document type '{slug}' scored {score}")

        if not scores:
            logger.debug("No deterministic matches found")
            return None

        # Find the highest score
        max_score = max(scores.values())
        best_candidates = [slug for slug, score in scores.items() if score == max_score]

        # Must have score >= 1 AND be the unique top scorer (for strong evidence)
        if max_score >= 1 and len(best_candidates) == 1:
            logger.info(f"Deterministic match: {best_candidates[0]} (score: {max_score}, unique top scorer)")
            return best_candidates[0]

        logger.debug(f"No deterministic match: max_score={max_score}, candidates={best_candidates}")
        return None

    async def _stage_combined_analysis(self, document: Document, document_dir: Path,
                                       ocr_result: OCRResult) -> Tuple[ClassificationResult, Optional[ExtractionEvidence]]:
        """Combined classification and metadata extraction in single LLM call."""
        # Get available document types
        result = await self.db.execute(text("SELECT slug, classification_hints FROM document_types"))
        available_types = result.fetchall()

        # Load skip markers and prepare text sample
        skip_markers = await self._load_skip_markers()
        text_result = self._prepare_text_sample(ocr_result.combined_text, skip_markers=skip_markers)
        sample_text = text_result.text
        
        # Track skip marker usage (first call during processing captures it)
        if text_result.skip_marker_used and self._skip_marker_used is None:
            self._skip_marker_used = text_result.skip_marker_used
            self._skip_marker_position = text_result.skip_marker_position
            logger.info(f"Document {document.id}: Skip marker '{text_result.skip_marker_used}' applied at position {text_result.skip_marker_position}")
        
        allowed_slugs = [slug for slug, _ in available_types]

        # Step 1: Try local (trained) Naive Bayes classifier first - PRIORITY over deterministic
        nb_pred = None
        bert_pred = None
        classifier_result = None
        classifier_confidence = 0.0
        
        try:
            from app.services.doc_type_classifier import classifier_service
            pred = classifier_service().predict(sample_text, allowed_labels=allowed_slugs, model_name=self.model_name)
            if pred:
                nb_pred = pred
                classifier_result = pred.label
                classifier_confidence = pred.confidence
                logger.info(f"Document {document.id} classified as '{pred.label}' via Naive Bayes (p={pred.confidence:.2f}, model={self.model_name or 'default'})")
        except Exception as e:
            logger.warning(f"Naive Bayes classifier failed or unavailable: {e}")

        # Step 1.5: Try BERT classifier (always run to get score, even if NB is good)
        try:
            from app.services.bert_classifier import bert_classifier_service
            bert_result = bert_classifier_service().predict(sample_text, model_name=self.model_name, allowed_labels=allowed_slugs)
            if bert_result:
                bert_pred = bert_result
                logger.info(f"Document {document.id} BERT classification: '{bert_result.label}' (p={bert_result.confidence:.2f})")
                # Use BERT result if: no NB result, or BERT confidence is significantly higher
                if not classifier_result or bert_result.confidence > classifier_confidence + 0.1:
                    classifier_result = bert_result.label
                    classifier_confidence = bert_result.confidence
                    logger.info(f"Document {document.id} classified as '{bert_result.label}' via BERT (p={bert_result.confidence:.2f}, model={self.model_name or 'default'})")
        except Exception as e:
            logger.debug(f"BERT classifier not available: {e}")

        # Step 2: Fall back to deterministic if trained models didn't match
        if not classifier_result:
            classifier_result = self.classify_deterministic(sample_text, available_types)
            if classifier_result:
                logger.info(f"Document {document.id} classified as '{classifier_result}' via deterministic matching (fallback)")

        if classifier_result:
            # Get the document type for metadata extraction
            doc_type_result = await self.db.execute(
                text("SELECT * FROM document_types WHERE slug = :slug"),
                {"slug": classifier_result}
            )
            doc_type_row = doc_type_result.fetchone()

            # Check if this type has fields configured
            fields_result = await self.db.execute(
                text("SELECT * FROM document_type_fields WHERE document_type_id = :doc_type_id"),
                {"doc_type_id": doc_type_row.id}
            )
            fields = fields_result.fetchall()

            if fields:
                # Do metadata extraction for classified result
                classification = ClassificationResult(
                    doc_type_slug=classifier_result,
                    confidence=0.95,
                    rationale=f"Local classifier match"
                )

                extraction_result = await self._stage_metadata_extraction(
                    document, document_dir, classification, ocr_result, None
                )

                # Save classification artifacts
                llm_dir = document_dir / "llm"
                llm_dir.mkdir(exist_ok=True)
                classification_data = {
                    "method": "local_classifier",
                    "doc_type_slug": classifier_result,
                    "confidence": classification.confidence,
                    "rationale": classification.rationale,
                }
                # Add both classifier scores
                if nb_pred:
                    classification_data["naive_bayes"] = {
                        "label": nb_pred.label,
                        "confidence": float(nb_pred.confidence)
                    }
                if bert_pred:
                    classification_data["bert"] = {
                        "label": bert_pred.label,
                        "confidence": float(bert_pred.confidence)
                    }
                with open(llm_dir / "classification_local.json", "w") as f:
                    json.dump(classification_data, f, indent=2)

                return classification, extraction_result
            else:
                # No fields configured, return unknown
                classification = ClassificationResult(
                    doc_type_slug="unknown",
                    confidence=0.0,
                    rationale=f"Document type '{classifier_result}' has no fields configured"
                )
                return classification, None

        # Step 3: Combined LLM analysis as last resort
        return await self._llm_combined_analysis(document_dir, sample_text, available_types)

    async def _llm_combined_analysis(self, document_dir: Path, sample_text: str, available_types: List[Tuple[str, str]]) -> Tuple[ClassificationResult, Optional[ExtractionEvidence]]:
        """Combined classification and extraction in single LLM call."""
        # Build comprehensive prompt with all document types and their fields
        types_info = []
        available_slugs = []

        for slug, hints in available_types:
            if slug == "unknown":
                continue

            available_slugs.append(slug)

            # Get document type info and fields
            doc_type_result = await self.db.execute(
                text("SELECT name, description, extraction_prompt_preamble FROM document_types WHERE slug = :slug"),
                {"slug": slug}
            )
            doc_type_row = doc_type_result.fetchone()

            fields_result = await self.db.execute(
                text("""
                    SELECT key, label, field_type, required, enum_values, regex, description
                    FROM document_type_fields
                    WHERE document_type_id = (SELECT id FROM document_types WHERE slug = :slug)
                    ORDER BY key
                """),
                {"slug": slug}
            )
            fields = fields_result.fetchall()

            if fields:
                type_info = f"## {slug.upper()}: {doc_type_row.name}\n"
                if doc_type_row.description:
                    type_info += f"Description: {doc_type_row.description}\n"
                if hints:
                    type_info += f"\n⚠️ DISTINCTIVE CLASSIFICATION HINTS (use these to differentiate from similar types):\n{hints}\n"
                if doc_type_row.extraction_prompt_preamble:
                    type_info += f"\nExtraction context: {doc_type_row.extraction_prompt_preamble}\n"

                type_info += "\nFields to extract:\n"
                for field in fields:
                    key, label, field_type, required, enum_values, regex, description = field
                    field_desc = f"- {key} ({field_type}): {label}"
                    if enum_values:
                        field_desc += f" - Values: {enum_values}"
                    if regex:
                        field_desc += f" - Pattern: {regex}"
                    if required:
                        field_desc += " (required)"
                    if description:
                        field_desc += f" - {description}"
                    type_info += field_desc + "\n"

                types_info.append(type_info)

        if not types_info:
            # No document types with fields configured
            classification = ClassificationResult(
                doc_type_slug="unknown",
                confidence=0.0,
                rationale="No document types with fields configured"
            )
            return classification, None

        # Build the combined prompt
        types_text = "\n\n".join(types_info)

        prompt = f"""Analyze this document and perform both classification AND metadata extraction in a single response.

First, classify the document into one of these types based on the content:
{', '.join(available_slugs)}, or 'unknown' if it doesn't match any type.

CRITICAL CLASSIFICATION RULES:
- If multiple document types share common keywords (e.g., both contain "iban"), you MUST look for DISTINCTIVE features that differentiate them
- Pay close attention to the document's PURPOSE and STRUCTURE, not just individual keywords
- Use the classification hints provided for each type to identify unique characteristics
- Consider the CONTEXT: what is the document trying to accomplish? What is its primary function?
- If two types seem similar, choose based on the MOST SPECIFIC and DISTINCTIVE features
- When in doubt, prefer the type with more specific matching characteristics

Then, if you classified it as a specific type (not 'unknown'), extract the metadata according to that type's field definitions.

IMPORTANT:
- If you cannot confidently classify the document, return 'unknown' and empty metadata
- For classification, you MUST provide exact evidence from the text that shows DISTINCTIVE features
- For extraction, follow the field definitions exactly
- Return both classification and extraction results in one JSON response

Document types and their field definitions:
{types_text}

Document text sample:
{sample_text}

Respond with JSON:
{{
  "classification": {{
    "doc_type_slug": "one of the types or 'unknown'",
    "confidence": 0.0-1.0,
    "rationale": "brief explanation",
    "evidence": "exact quote from text or empty"
  }},
  "extraction": {{
    "data": {{
      "field_key": "extracted_value",
      ...
    }},
    "evidence": {{
      "field_key": [
        {{"page": 0, "start": 10, "end": 30, "quote": "exact text span"}},
        ...
      ],
      ...
    }}
  }}
}}"""

        schema = {
            "type": "object",
            "required": ["classification", "extraction"],
            "properties": {
                "classification": {
                    "type": "object",
                    "required": ["doc_type_slug", "confidence", "rationale", "evidence"],
                    "properties": {
                        "doc_type_slug": {"type": "string", "enum": available_slugs + ["unknown"]},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"},
                        "evidence": {"type": "string"}
                    }
                },
                "extraction": {
                    "type": "object",
                    "required": ["data", "evidence"],
                    "properties": {
                        "data": {"type": "object"},
                        "evidence": {"type": "object"}
                    }
                }
            }
        }

        llm_dir = document_dir / "llm"
        llm_dir.mkdir(exist_ok=True)

        with open(llm_dir / "combined_analysis_prompt.txt", "w") as f:
            f.write(prompt)

        with open(llm_dir / "combined_analysis_schema.json", "w") as f:
            json.dump(schema, f, indent=2)

        curl_command = None
        try:
            logger.info(f"Starting combined LLM analysis for document {document_dir.parent.name}")
            result, response_text, curl_command = await self.llm.generate_json_with_raw(prompt, schema)
            logger.info(f"Combined LLM analysis completed for document {document_dir.parent.name}")
            
            # Save response and curl command immediately after successful request
            with open(llm_dir / "combined_analysis_response.txt", "w") as f:
                f.write(response_text)
            
            if curl_command:
                with open(llm_dir / "combined_analysis_curl.txt", "w") as f:
                    f.write(curl_command)
        except Exception as e:
            logger.error(f"Combined LLM analysis failed for document {document_dir.parent.name}: {e}")
            with open(llm_dir / "combined_analysis_error.txt", "w") as f:
                f.write(str(e))
            # Save curl command even if there was an error (if we got that far)
            if curl_command:
                with open(llm_dir / "combined_analysis_curl.txt", "w") as f:
                    f.write(curl_command)
            raise

        # Parse the combined result
        classification_data = result.get("classification", {})
        extraction_data = result.get("extraction", {})

        # Validate classification (returns dict)
        validated_classification_data = self._validate_llm_classification(classification_data, sample_text)
        # Convert to ClassificationResult object
        classification = ClassificationResult(**validated_classification_data)

        # Handle extraction result
        extraction_result = None
        if classification.doc_type_slug != "unknown" and extraction_data:
            try:
                # Build schema for the specific document type
                doc_type_result = await self.db.execute(
                    text("SELECT id FROM document_types WHERE slug = :slug"),
                    {"slug": classification.doc_type_slug}
                )
                doc_type_row = doc_type_result.fetchone()

                if doc_type_row:
                    fields_result = await self.db.execute(
                        text("SELECT * FROM document_type_fields WHERE document_type_id = :doc_type_id"),
                        {"doc_type_id": doc_type_row.id}
                    )
                    fields = fields_result.fetchall()

                    if fields:
                        # Create extraction schema
                        extraction_schema = self._build_extraction_schema(fields)

                        # Validate extraction data against schema
                        if self._validate_extraction_data(extraction_data, extraction_schema):
                            # Normalize extraction data to ensure correct structure
                            normalized_data = self._normalize_extraction_data(extraction_data)
                            extraction_result = ExtractionEvidence(**normalized_data)

                            # Save extraction artifacts
                            metadata_dir = document_dir / "metadata"
                            metadata_dir.mkdir(exist_ok=True)

                            with open(metadata_dir / "result.json", "w") as f:
                                json.dump(extraction_result.data, f, indent=2)

                            with open(metadata_dir / "evidence.json", "w") as f:
                                json.dump(self._json_serialize(extraction_result.evidence), f, indent=2)

                            # Validate evidence spans
                            pages = ocr_result.pages
                            validation_errors = self._validate_evidence(extraction_result, pages)

                            with open(metadata_dir / "validation.json", "w") as f:
                                json.dump({"errors": validation_errors}, f, indent=2)
            except Exception as e:
                logger.warning(f"Extraction result processing failed: {e}")

        with open(llm_dir / "combined_analysis_result.json", "w") as f:
            json.dump(result, f, indent=2)

        return classification, extraction_result

    def _normalize_extraction_data(self, extraction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize extraction data to ensure correct structure for ExtractionEvidence."""
        data = extraction_data.get("data", {})
        evidence = extraction_data.get("evidence", {})
        
        # Handle case where evidence is an array instead of an object
        if isinstance(evidence, list):
            # Try to match evidence spans to data fields based on quote content
            normalized_evidence = {}
            for field_key in data.keys():
                normalized_evidence[field_key] = []
            
            # Try to match evidence spans to fields
            for evidence_item in evidence:
                if not isinstance(evidence_item, dict):
                    continue
                
                quote = evidence_item.get("quote", "").lower()
                # Try to find matching field by checking if quote contains field value
                matched = False
                for field_key, field_value in data.items():
                    if field_value is None:
                        continue
                    field_value_str = str(field_value).lower()
                    # Check if quote contains the field value or vice versa
                    if field_value_str in quote or quote in field_value_str or any(
                        word in quote for word in field_value_str.split() if len(word) > 3
                    ):
                        if field_key not in normalized_evidence:
                            normalized_evidence[field_key] = []
                        normalized_evidence[field_key].append(evidence_item)
                        matched = True
                        break
                
                # If no match found, add to first field or create a generic entry
                if not matched and data:
                    first_key = list(data.keys())[0]
                    if first_key not in normalized_evidence:
                        normalized_evidence[first_key] = []
                    normalized_evidence[first_key].append(evidence_item)
            
            # Ensure all data fields have evidence entries (even if empty)
            for field_key in data.keys():
                if field_key not in normalized_evidence:
                    normalized_evidence[field_key] = []
            
            # Normalize all evidence spans to EvidenceSpan objects
            for field_key in normalized_evidence.keys():
                normalized_evidence[field_key] = self._normalize_evidence_list(normalized_evidence[field_key])
        else:
            # Normal case: evidence is already an object
            normalized_evidence = {}
            
            for key, value in evidence.items():
                if value is None:
                    normalized_evidence[key] = []
                elif isinstance(value, str):
                    # Convert string to EvidenceSpan with default values
                    normalized_evidence[key] = [
                        EvidenceSpan(page=0, start=0, end=len(value), quote=value)
                    ]
                elif isinstance(value, dict):
                    # If it's a dict, try to extract spans from it or convert to list
                    # Check if it looks like a nested structure (e.g., {'street': [...]})
                    if any(isinstance(v, list) for v in value.values()):
                        # Flatten nested structure - take first list found
                        for nested_value in value.values():
                            if isinstance(nested_value, list):
                                normalized_evidence[key] = self._normalize_evidence_list(nested_value)
                                break
                        else:
                            # No list found, create empty list
                            normalized_evidence[key] = []
                    else:
                        # Try to convert dict to EvidenceSpan if it has the right keys
                        try:
                            normalized_evidence[key] = [EvidenceSpan(**value)]
                        except Exception:
                            # If conversion fails, create empty list
                            normalized_evidence[key] = []
                elif isinstance(value, list):
                    normalized_evidence[key] = self._normalize_evidence_list(value)
                else:
                    # Unknown type, create empty list
                    normalized_evidence[key] = []
        
        return {
            "data": data,
            "evidence": normalized_evidence
        }
    
    def _normalize_evidence_list(self, value_list: List[Any]) -> List[EvidenceSpan]:
        """Normalize a list of evidence values to EvidenceSpan objects."""
        normalized = []
        for item in value_list:
            if item is None:
                continue
            elif isinstance(item, str):
                # Convert string to EvidenceSpan
                normalized.append(EvidenceSpan(page=0, start=0, end=len(item), quote=item))
            elif isinstance(item, dict):
                # Try to convert dict to EvidenceSpan
                try:
                    # Ensure required fields are present
                    span_dict = {
                        "page": item.get("page", 0),
                        "start": item.get("start", 0),
                        "end": item.get("end", item.get("start", 0) + len(item.get("quote", ""))),
                        "quote": item.get("quote", "")
                    }
                    normalized.append(EvidenceSpan(**span_dict))
                except Exception:
                    # If conversion fails, skip this item
                    continue
            else:
                # Unknown type, skip
                continue
        
        return normalized

    def _validate_extraction_data(self, extraction_data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Basic validation of extraction data against schema."""
        if not isinstance(extraction_data, dict):
            return False

        data = extraction_data.get("data", {})
        evidence = extraction_data.get("evidence", {})

        if not isinstance(data, dict) or not isinstance(evidence, dict):
            return False

        # Check that evidence keys match data keys
        if set(data.keys()) != set(evidence.keys()):
            return False

        return True

    async def _stage_classification(self, document: Document, document_dir: Path,
                                  ocr_result: OCRResult) -> ClassificationResult:
        """Stage 3: Classify document type using trained model first, then deterministic, then LLM."""
        # Get available document types with hints
        result = await self.db.execute(text("SELECT slug, classification_hints FROM document_types"))
        available_types = result.fetchall()

        # Load skip markers and prepare text sample (first 6000 chars, normalized)
        skip_markers = await self._load_skip_markers()
        text_result = self._prepare_text_sample(ocr_result.combined_text, skip_markers=skip_markers)
        sample_text = text_result.text
        
        # Track skip marker usage (first call during processing captures it)
        if text_result.skip_marker_used and self._skip_marker_used is None:
            self._skip_marker_used = text_result.skip_marker_used
            self._skip_marker_position = text_result.skip_marker_position
            logger.info(f"Document {document.id}: Skip marker '{text_result.skip_marker_used}' applied at position {text_result.skip_marker_position}")
        
        allowed_slugs = [slug for slug, _ in available_types]

        # Step 1: Local (trained) Naive Bayes classifier - PRIORITY over deterministic
        nb_pred = None
        bert_pred = None
        best_pred = None
        best_method = None
        
        try:
            from app.services.doc_type_classifier import classifier_service
            pred = classifier_service().predict(sample_text, allowed_labels=allowed_slugs, model_name=self.model_name)
            if pred:
                nb_pred = pred
                best_pred = pred
                best_method = "naive_bayes"
                logger.info(f"Document {document.id} NB classification: '{pred.label}' (p={pred.confidence:.2f})")
        except Exception as e:
            logger.warning(f"Naive Bayes classifier failed or unavailable: {e}")

        # Step 1.5: Try BERT classifier (always run to get score, even if NB is good)
        try:
            from app.services.bert_classifier import bert_classifier_service
            bert_result = bert_classifier_service().predict(sample_text, model_name=self.model_name, allowed_labels=allowed_slugs)
            if bert_result:
                bert_pred = bert_result
                logger.info(f"Document {document.id} BERT classification: '{bert_result.label}' (p={bert_result.confidence:.2f})")
                # Use BERT if no NB result or if BERT is significantly better
                if not best_pred or bert_result.confidence > best_pred.confidence + 0.1:
                    best_pred = bert_result
                    best_method = "bert"
        except Exception as e:
            logger.debug(f"BERT classifier not available: {e}")

        if best_pred:
            # Build rationale with both scores if available
            rationale_parts = [f"{best_method.upper()} classifier (p={best_pred.confidence:.2f})"]
            if nb_pred and bert_pred:
                rationale_parts.append(f"NB: {nb_pred.label} ({nb_pred.confidence:.2f}), BERT: {bert_pred.label} ({bert_pred.confidence:.2f})")
            elif nb_pred:
                rationale_parts.append(f"NB: {nb_pred.label} ({nb_pred.confidence:.2f})")
            elif bert_pred:
                rationale_parts.append(f"BERT: {bert_pred.label} ({bert_pred.confidence:.2f})")
            
            classification = ClassificationResult(
                doc_type_slug=best_pred.label,
                confidence=float(best_pred.confidence),
                rationale=f"{' | '.join(rationale_parts)} (model={self.model_name or 'default'})",
                evidence=""
            )

            llm_dir = document_dir / "llm"
            llm_dir.mkdir(exist_ok=True)
            
            # Save both scores
            classification_data = {
                "method": best_method,
                "doc_type_slug": best_pred.label,
                "confidence": float(best_pred.confidence),
            }
            
            # Add both classifier scores
            if nb_pred:
                classification_data["naive_bayes"] = {
                    "label": nb_pred.label,
                    "confidence": float(nb_pred.confidence)
                }
            if bert_pred:
                classification_data["bert"] = {
                    "label": bert_pred.label,
                    "confidence": float(bert_pred.confidence)
                }
            
            with open(llm_dir / "classification_local.json", "w") as f:
                json.dump(classification_data, f, indent=2)

            logger.info(f"Document {document.id} classified as '{best_pred.label}' via {best_method} (p={best_pred.confidence:.2f})")
            return classification

        # Step 2: Deterministic classification (fallback when trained models don't match)
        deterministic_result = self.classify_deterministic(sample_text, available_types)
        logger.info(f"Document {document.id} deterministic classification: {deterministic_result} (sample_text_length: {len(sample_text)})")

        if deterministic_result:
            # Deterministic match found (as fallback)
            classification = ClassificationResult(
                doc_type_slug=deterministic_result,
                confidence=0.95,  # High confidence for deterministic matches
                rationale=f"Deterministic match (fallback)",
                evidence=""
            )

            # Save classification artifacts
            llm_dir = document_dir / "llm"
            llm_dir.mkdir(exist_ok=True)
            with open(llm_dir / "classification_deterministic.json", "w") as f:
                json.dump({
                    "method": "deterministic",
                    "doc_type_slug": deterministic_result,
                    "confidence": classification.confidence,
                    "rationale": classification.rationale,
                }, f, indent=2)

            logger.info(f"Document {document.id} classified as '{deterministic_result}' via deterministic matching (fallback)")
            return classification

        # Step 3: LLM classification as last resort
        logger.info(f"Document {document.id} falling back to LLM classification")
        available_slugs_with_unknown = allowed_slugs + ["unknown"]
        llm_result = await self._llm_classify_document(document_dir, sample_text, available_slugs_with_unknown)

        logger.info(f"Document {document.id} classified as '{llm_result.doc_type_slug}' via LLM (confidence: {llm_result.confidence})")
        return llm_result

    def _prepare_text_sample(self, text: str, max_chars: int = 6000, skip_markers: List[Tuple[str, bool]] = None) -> TextPrepareResult:
        """Prepare text sample for classification by normalizing whitespace, applying skip markers, and including header.
        
        Args:
            text: Raw text to process
            max_chars: Maximum characters to include
            skip_markers: List of (pattern, is_regex) tuples - text after first match is skipped
            
        Returns:
            TextPrepareResult with prepared text and skip marker info
        """
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', text.strip())
        
        # Track skip marker info
        matched_marker = None
        matched_position = None
        
        # Apply skip markers - truncate text at first match
        if skip_markers:
            earliest_skip_pos = len(normalized)
            
            for pattern, is_regex in skip_markers:
                try:
                    if is_regex:
                        match = re.search(pattern, normalized, re.IGNORECASE)
                        if match and match.start() < earliest_skip_pos:
                            earliest_skip_pos = match.start()
                            matched_marker = pattern
                    else:
                        # Case-insensitive plain text search
                        pos = normalized.lower().find(pattern.lower())
                        if pos != -1 and pos < earliest_skip_pos:
                            earliest_skip_pos = pos
                            matched_marker = pattern
                except re.error as e:
                    logger.warning(f"Invalid skip marker regex '{pattern}': {e}")
            
            if matched_marker and earliest_skip_pos < len(normalized):
                logger.info(f"Skip marker '{matched_marker}' found at position {earliest_skip_pos}, truncating text (was {len(normalized)} chars)")
                matched_position = earliest_skip_pos
                normalized = normalized[:earliest_skip_pos].strip()

        # Always include first 2-3 lines if available (header area)
        lines = normalized.split('\n')
        header_lines = []
        char_count = 0

        for line in lines[:3]:  # First 3 lines
            if char_count + len(line) > max_chars:
                break
            header_lines.append(line)
            char_count += len(line) + 1  # +1 for newline

        # If header is too short, add more content
        if char_count < max_chars:
            remaining_chars = max_chars - char_count
            remaining_text = normalized[char_count:char_count + remaining_chars]
            header_lines.append(remaining_text)

        result_text = '\n'.join(header_lines)
        return TextPrepareResult(text=result_text, skip_marker_used=matched_marker, skip_marker_position=matched_position)
    
    async def _load_skip_markers(self) -> List[Tuple[str, bool]]:
        """Load active skip markers from database."""
        result = await self.db.execute(
            text("SELECT pattern, is_regex FROM skip_markers WHERE is_active = 1")
        )
        rows = result.fetchall()
        return [(row[0], bool(row[1])) for row in rows]
    
    def _split_text_into_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into chunks with overlap to avoid missing data at boundaries.
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this is not the last chunk, try to break at a word boundary
            if end < len(text):
                # Look for a good break point (newline, period, or space) within the last 200 chars
                break_search_start = max(start, end - 200)
                for i in range(end - 1, break_search_start, -1):
                    if text[i] in '\n. ':
                        end = i + 1
                        break
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start forward, accounting for overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks

    async def _llm_classify_document(self, document_dir: Path, sample_text: str, available_types: List[str]) -> ClassificationResult:
        """Use LLM to classify document type with evidence-driven validation."""
        types_str = ", ".join(available_types)

        # Get classification hints from configured document types (exclude "unknown")
        hints = []
        configured_types = [t for t in available_types if t != "unknown"]
        for doc_type_slug in configured_types:
            hint_result = await self.db.execute(
                text("SELECT classification_hints FROM document_types WHERE slug = :slug"),
                {"slug": doc_type_slug}
            )
            hint_row = hint_result.fetchone()
            if hint_row and hint_row[0]:
                hints.append(f"{doc_type_slug}: {hint_row[0]}")

        hints_text = "\n".join(f"- {hint}" for hint in hints) if hints else ""

        prompt = f"""Classify this document into one of these types: {types_str}

CRITICAL INSTRUCTIONS:
- Choose a type ONLY if you can quote exact evidence from the text.
- If you cannot prove any specific type, return 'unknown'.
- NEVER guess or assume - only classify based on concrete evidence.
- For 'unknown', confidence should be 0.0 and evidence should be empty.
- Keep evidence SHORT (max 50 characters) - just enough to prove the type.

Available types and their hints:
{hints_text}

Document text sample:
{sample_text}

Respond with JSON only:
{{
  "doc_type_slug": "one of the available types or 'unknown'",
  "confidence": 0.0-1.0,
  "rationale": "brief explanation",
  "evidence": "short exact quote (max 50 chars)"
}}"""

        schema = {
            "type": "object",
            "required": ["doc_type_slug", "confidence", "rationale", "evidence"],
            "properties": {
                "doc_type_slug": {"type": "string", "enum": available_types},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "rationale": {"type": "string"},
                "evidence": {"type": "string"}
            }
        }

        llm_dir = document_dir / "llm"
        llm_dir.mkdir(exist_ok=True)

        with open(llm_dir / "classification_prompt.txt", "w") as f:
            f.write(prompt)

        with open(llm_dir / "classification_schema.json", "w") as f:
            json.dump(schema, f, indent=2)

        curl_command = None
        response_text = None
        result = None
        try:
            logger.info(f"Starting LLM classification request for document {document_dir.parent.name}")
            result, response_text, curl_command = await self.llm.generate_json_with_raw(prompt, schema)
            logger.info(f"LLM classification completed for document {document_dir.parent.name}")

            # Save response and curl command immediately after successful request
            with open(llm_dir / "classification_response.txt", "w") as f:
                f.write(response_text)
            
            if curl_command:
                with open(llm_dir / "classification_curl.txt", "w") as f:
                    f.write(curl_command)
        except Exception as e:
            logger.error(f"LLM classification failed for document {document_dir.parent.name}: {e}")
            error_msg = str(e)
            with open(llm_dir / "classification_error.txt", "w") as f:
                f.write(error_msg)
            
            # Save curl command even if there was an error (if we got that far)
            if curl_command:
                with open(llm_dir / "classification_curl.txt", "w") as f:
                    f.write(curl_command)
            
            # If we have response_text but parsing failed, try to repair it
            if response_text and "Failed to parse JSON" in error_msg:
                logger.warning("Attempting to repair JSON from response_text after parsing failure")
                try:
                    repaired_result = self.llm._repair_json(response_text)
                    if repaired_result:
                        logger.info("Successfully repaired JSON from response_text")
                        result = repaired_result
                    else:
                        logger.error("Failed to repair JSON even from response_text")
                        raise
                except Exception as repair_error:
                    logger.error(f"JSON repair attempt also failed: {repair_error}")
                    raise e  # Re-raise original error
            else:
                raise

        # Validate LLM response
        if result:
            validated_result = self._validate_llm_classification(result, sample_text)

            with open(llm_dir / "classification_result.json", "w") as f:
                json.dump(validated_result, f, indent=2)

            return ClassificationResult(**validated_result)
        else:
            # Fallback to unknown if we couldn't get a result
            logger.warning("No classification result available, falling back to 'unknown'")
            return ClassificationResult(
                doc_type_slug="unknown",
                confidence=0.0,
                rationale="Classification failed - could not parse LLM response"
            )

    def _validate_llm_classification(self, result: Dict[str, Any], sample_text: str) -> Dict[str, Any]:
        """Validate LLM classification result and force 'unknown' if invalid."""
        doc_type_slug = result.get("doc_type_slug", "unknown")
        evidence = result.get("evidence", "").strip()
        confidence = result.get("confidence", 0.0)

        validation_errors = []

        # If claiming a specific type (not unknown), must have evidence
        if doc_type_slug != "unknown":
            if not evidence:
                validation_errors.append("No evidence provided for non-unknown classification")
            elif not self._evidence_supported_by_text(evidence, sample_text):
                validation_errors.append(f"Evidence not sufficiently supported by document text")
            elif confidence < 0.5:
                validation_errors.append(f"Low confidence ({confidence}) for specific classification")

        if validation_errors:
            logger.warning(f"LLM classification validation failed: {validation_errors}. Forcing 'unknown'.")
            logger.warning(f"Original result: {result}")
            logger.warning(f"Sample text length: {len(sample_text)} chars")

            return {
                "doc_type_slug": "unknown",
                "confidence": 0.0,
                "rationale": f"Validation failed: {'; '.join(validation_errors)}",
                "evidence": ""
            }

        return result

    def _evidence_supported_by_text(self, evidence: str, sample_text: str) -> bool:
        """Check if the evidence is sufficiently supported by the document text."""
        if not evidence or not sample_text:
            return False

        # Normalize both texts for comparison
        evidence_norm = evidence.lower().strip()
        sample_norm = sample_text.lower()
        
        # Remove leading/trailing quotes that LLM sometimes adds
        if evidence_norm.startswith('"') and evidence_norm.endswith('"'):
            evidence_norm = evidence_norm[1:-1]
        elif evidence_norm.startswith('"'):
            evidence_norm = evidence_norm[1:]

        # If the full evidence is found, it's definitely supported
        if evidence_norm in sample_norm:
            return True

        # OCR often introduces/loses whitespace or punctuation. Try a compact match.
        evidence_compact = re.sub(r'[^a-z0-9]', '', evidence_norm)
        sample_compact = re.sub(r'[^a-z0-9]', '', sample_norm)
        if len(evidence_compact) >= 8 and evidence_compact in sample_compact:
            return True

        # Otherwise, check if significant keywords from evidence are present
        # Split evidence into words and check if at least 50% of significant words are found
        evidence_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', evidence_norm))  # Words of 3+ chars
        sample_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', sample_norm))

        if not evidence_words:
            # No significant words in evidence, check for numbers/codes
            evidence_codes = set(re.findall(r'\b[A-Z0-9]{4,}\b', evidence))
            sample_codes = set(re.findall(r'\b[A-Z0-9]{4,}\b', sample_text))
            if evidence_codes:
                found_codes = evidence_codes.intersection(sample_codes)
                return len(found_codes) / len(evidence_codes) >= 0.5
            return False

        # Count how many evidence words are found in sample
        found_words = evidence_words.intersection(sample_words)
        support_ratio = len(found_words) / len(evidence_words)

        # Require at least 50% of significant words to be present (lowered from 60%)
        return support_ratio >= 0.5

    async def _stage_metadata_extraction(self, document: Document, document_dir: Path,
                                       classification: ClassificationResult,
                                       ocr_result: OCRResult,
                                       progress_callback: callable = None) -> Optional[ExtractionEvidence]:
        """Stage 4: Extract metadata using LLM based on document type schema."""
        # Get document type fields and preamble
        result = await self.db.execute(
            text("SELECT key, label, field_type, required, enum_values, regex FROM document_type_fields WHERE document_type_id = (SELECT id FROM document_types WHERE slug = :slug)"),
            {"slug": classification.doc_type_slug}
        )
        fields = result.fetchall()

        # Get preamble
        preamble_result = await self.db.execute(
            text("SELECT extraction_prompt_preamble FROM document_types WHERE slug = :slug"),
            {"slug": classification.doc_type_slug}
        )
        preamble_row = preamble_result.fetchone()
        preamble = preamble_row[0] if preamble_row else ""

        if not fields:
            llm_dir = document_dir / "llm"
            llm_dir.mkdir(exist_ok=True)
            with open(llm_dir / "extraction_skipped.json", "w") as f:
                json.dump(
                    {
                        "used_llm": False,
                        "reason": "No document type fields configured for this type",
                        "doc_type_slug": classification.doc_type_slug,
                    },
                    f,
                    indent=2,
                )
            return None

        # Load skip markers and prepare text (apply same filtering as classification)
        skip_markers = await self._load_skip_markers()
        # For metadata extraction, use full text (not limited to 8000 chars like classification)
        # But still apply skip markers
        text_result = self._prepare_text_sample(ocr_result.combined_text, max_chars=999999, skip_markers=skip_markers)
        filtered_text = text_result.text
        
        # Track skip marker usage (if not already set during classification)
        if text_result.skip_marker_used and self._skip_marker_used is None:
            self._skip_marker_used = text_result.skip_marker_used
            self._skip_marker_position = text_result.skip_marker_position

        # Build schema
        schema = self._build_extraction_schema(fields)
        
        llm_dir = document_dir / "llm"
        llm_dir.mkdir(exist_ok=True)
        
        with open(llm_dir / "extraction_schema.json", "w") as f:
            json.dump(schema, f, indent=2)
        
        # Split into chunks if text is too large (> 6000 chars to leave room for prompt overhead)
        CHUNK_SIZE = 6000
        OVERLAP = 500  # Overlap between chunks to avoid missing data at boundaries
        
        curl_command = None
        response_text = None
        result = None
        total_chunks = None  # Track if we're using chunks
        
        if len(filtered_text) > CHUNK_SIZE:
            logger.info(f"Document text is large ({len(filtered_text)} chars), splitting into chunks for extraction")
            chunks = self._split_text_into_chunks(filtered_text, CHUNK_SIZE, OVERLAP)
            logger.info(f"Split into {len(chunks)} chunks")
            
            # Process each chunk
            all_results = []
            chunk_responses = []
            total_chunks = len(chunks)  # Store for post-processing stages
            
            # Metadata extraction runs from 60-85%, so we use 60-80% for chunks, 80-85% for merging
            EXTRACTION_START = 60
            EXTRACTION_END = 80
            MERGE_START = 80
            MERGE_END = 85
            
            for i, chunk in enumerate(chunks):
                chunk_num = i + 1
                logger.info(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} chars)")
                
                # Update progress: 60% + (chunk_num/total_chunks * 20%)
                chunk_progress = int(EXTRACTION_START + (chunk_num / total_chunks * (EXTRACTION_END - EXTRACTION_START)))
                chunk_stage = f"extracting_metadata_chunk_{chunk_num}_{total_chunks}"
                await self._update_progress(document.id, chunk_progress, chunk_stage, progress_callback)
                
                prompt = self._build_extraction_prompt(fields, chunk, classification.doc_type_slug, preamble, chunk_num=chunk_num, total_chunks=total_chunks)
                
                # Save chunk prompt
                with open(llm_dir / f"extraction_prompt_chunk_{chunk_num}.txt", "w") as f:
                    f.write(prompt)
                
                try:
                    chunk_result, chunk_response_text, chunk_curl_command = await self.llm.generate_json_with_raw(prompt, schema)
                    all_results.append(chunk_result)
                    chunk_responses.append(chunk_response_text)
                    
                    # Save chunk response
                    with open(llm_dir / f"extraction_response_chunk_{chunk_num}.txt", "w") as f:
                        f.write(chunk_response_text)
                    if chunk_curl_command:
                        with open(llm_dir / f"extraction_curl_chunk_{chunk_num}.txt", "w") as f:
                            f.write(chunk_curl_command)
                except Exception as e:
                    logger.warning(f"Chunk {chunk_num} extraction failed: {e}, continuing with other chunks")
                    continue
            
            # Update progress for merging
            await self._update_progress(document.id, MERGE_START, "extracting_metadata_merging", progress_callback)
            
            # Merge all chunk results
            if all_results:
                logger.info(f"Merging {len(all_results)} chunk results")
                result = self.llm._merge_json_objects(all_results)
                if result is None:
                    logger.warning("Failed to merge chunk results, using first chunk")
                    result = all_results[0]
                
                # Update progress after merging - keep chunk info in stage for visibility
                await self._update_progress(document.id, MERGE_END, f"extracting_metadata_chunk_done_{total_chunks}", progress_callback)
                
                # Combine all response texts for logging
                response_text = "\n\n--- CHUNK MERGE ---\n\n".join([
                    f"Chunk {i+1}:\n{r}" for i, r in enumerate(chunk_responses)
                ])
                
                # Save merged prompt and response
                with open(llm_dir / "extraction_prompt.txt", "w") as f:
                    f.write(f"# Multi-chunk extraction ({len(chunks)} chunks)\n")
                    for i in range(len(chunks)):
                        if (llm_dir / f"extraction_prompt_chunk_{i+1}.txt").exists():
                            f.write(f"\n--- Chunk {i+1} ---\n")
                            f.write(open(llm_dir / f"extraction_prompt_chunk_{i+1}.txt").read())
                
                with open(llm_dir / "extraction_response.txt", "w") as f:
                    f.write(response_text)
                
                logger.info(f"LLM metadata extraction completed for document {document_dir.parent.name} ({len(chunks)} chunks merged)")
                
                # Save merged result
                with open(llm_dir / "extraction_result.json", "w") as f:
                    json.dump(self._json_serialize(result), f, indent=2)
            else:
                raise Exception("All chunk extractions failed")
        else:
            # Single chunk - normal processing
            prompt = self._build_extraction_prompt(fields, filtered_text, classification.doc_type_slug, preamble)
            
            with open(llm_dir / "extraction_prompt.txt", "w") as f:
                f.write(prompt)
            
                try:
                    logger.info(f"Starting LLM metadata extraction for document {document_dir.parent.name}")
                    result, response_text, curl_command = await self.llm.generate_json_with_raw(prompt, schema)
                    logger.info(f"LLM metadata extraction completed for document {document_dir.parent.name}")
                    
                    # Save response and curl command immediately after successful request
                    with open(llm_dir / "extraction_response.txt", "w") as f:
                        f.write(response_text)
                    
                    if curl_command:
                        with open(llm_dir / "extraction_curl.txt", "w") as f:
                            f.write(curl_command)
                    
                    with open(llm_dir / "extraction_result.json", "w") as f:
                        json.dump(self._json_serialize(result), f, indent=2)
                except Exception as e:
                    logger.error(f"LLM metadata extraction failed for document {document_dir.parent.name}: {e}")
                    error_msg = str(e)
                    with open(llm_dir / "extraction_error.txt", "w") as f:
                        f.write(error_msg)
                    
                    # Save curl command even if there was an error (if we got that far)
                    if curl_command:
                        with open(llm_dir / "extraction_curl.txt", "w") as f:
                            f.write(curl_command)
                    
                    # If we have response_text but parsing failed, try to repair it
                    if response_text and "Failed to parse JSON" in error_msg:
                        logger.warning("Attempting to repair JSON from response_text after parsing failure")
                        try:
                            repaired_result = self.llm._repair_json(response_text)
                            if repaired_result:
                                logger.info("Successfully repaired JSON from response_text")
                                result = repaired_result
                                # Save the repaired result
                                with open(llm_dir / "extraction_response.txt", "w") as f:
                                    f.write(response_text)
                                with open(llm_dir / "extraction_result.json", "w") as f:
                                    json.dump(self._json_serialize(result), f, indent=2)
                            else:
                                logger.error("Failed to repair JSON even from response_text")
                                raise
                        except Exception as repair_error:
                            logger.error(f"JSON repair attempt also failed: {repair_error}")
                            raise e  # Re-raise original error
                    else:
                        raise

        # Post-processing steps - include chunk info if we used chunks
        post_stage_suffix = f"_chunks_{total_chunks}" if total_chunks else ""
        await self._update_progress(document.id, 78, f"extracting_metadata_post_processing{post_stage_suffix}", progress_callback)
        # Fill in missing quotes from document text
        result = self._fill_missing_quotes(result, ocr_result.pages)
        
        # Apply regex post-processing to clean extracted values
        result = self._apply_regex_filters(result, fields, llm_dir)
        
        # Validate evidence spans
        await self._update_progress(document.id, 80, f"extracting_metadata_validating{post_stage_suffix}", progress_callback)
        # Normalize extraction data to ensure correct structure
        normalized_result = self._normalize_extraction_data(result)
        evidence_data = ExtractionEvidence(**normalized_result)
        validation_errors = self._validate_evidence(evidence_data, ocr_result.pages)

        # Save results
        await self._update_progress(document.id, 82, f"extracting_metadata_saving{post_stage_suffix}", progress_callback)
        metadata_dir = document_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        with open(metadata_dir / "result.json", "w") as f:
            json.dump(evidence_data.data, f, indent=2)

        with open(metadata_dir / "validation.json", "w") as f:
            json.dump({"errors": validation_errors}, f, indent=2)

        with open(metadata_dir / "evidence.json", "w") as f:
            json.dump(self._json_serialize(evidence_data.evidence), f, indent=2)

        await self._update_progress(document.id, 85, "extracting_metadata_complete", progress_callback)
        return evidence_data

    def _build_extraction_schema(self, fields: List[Tuple]) -> Dict[str, Any]:
        """Build JSON schema for metadata extraction."""
        properties = {}
        required = []

        for key, label, field_type, is_required, enum_values, regex in fields:
            prop = {"type": "string", "description": label}

            if field_type == "number":
                prop["type"] = "number"
            elif field_type == "date":
                prop["format"] = "date"
            elif field_type == "enum" and enum_values:
                prop["enum"] = enum_values

            properties[key] = prop

            if is_required:
                required.append(key)

        evidence_properties = {}
        for key, _, _, _, _, _ in fields:
            evidence_properties[key] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer", "minimum": 0},
                        "start": {"type": "integer", "minimum": 0},
                        "end": {"type": "integer", "minimum": 0},
                        "quote": {"type": "string"}
                    },
                    "required": ["page", "start", "end", "quote"]
                }
            }

        return {
            "type": "object",
            "required": ["data", "evidence"],
            "properties": {
                "data": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                },
                "evidence": {
                    "type": "object",
                    "properties": evidence_properties,
                    "required": list(evidence_properties.keys())
                }
            }
        }

    def _build_extraction_prompt(self, fields: List[Tuple], text: str, doc_type: str, preamble: str = "", chunk_num: int = None, total_chunks: int = None) -> str:
        """Build extraction prompt for LLM.
        
        Args:
            fields: Document type fields to extract
            text: Text chunk to extract from
            doc_type: Document type slug
            preamble: Optional preamble text
            chunk_num: Optional chunk number (if multi-chunk extraction)
            total_chunks: Optional total number of chunks
        """
        field_descriptions = []
        has_address_field = False
        has_iban_field = False
        for key, label, field_type, is_required, enum_values, regex in fields:
            key_lower = str(key).lower()
            label_lower = str(label).lower()
            if "adres" in key_lower or "adres" in label_lower or "address" in key_lower or "address" in label_lower:
                has_address_field = True
            if field_type == "iban" or "iban" in key_lower or "iban" in label_lower:
                has_iban_field = True

            desc = f"- {key} ({field_type}): {label}"
            if enum_values:
                desc += f" - Values: {', '.join(enum_values)}"
            if regex:
                desc += f" - Pattern: {regex}"
            if is_required:
                desc += " (required)"
            field_descriptions.append(desc)

        fields_str = "\n".join(field_descriptions)

        preamble_text = f"\n\n{preamble}" if preamble else ""
        notes = []
        if has_address_field:
            notes.append("- For address fields: extract a postal address (street, house number, postal code, city). Do not return only a person/company name.")
        if has_iban_field:
            notes.append("- For IBAN fields: return only the IBAN (e.g., NL..), no extra words or currency.")
        notes_text = "\n".join(notes)
        notes_block = f"\nSpecial notes:\n{notes_text}\n" if notes_text else ""
        
        # Add chunk info if multi-chunk
        chunk_info = ""
        if chunk_num and total_chunks:
            chunk_info = f"""
NOTE: This is chunk {chunk_num} of {total_chunks} of a large document. 
- Extract ONLY the metadata you can find in THIS chunk of text.
- If a field is not present in this chunk, return null for that field.
- Do NOT make up values for fields not found in this chunk.
- The results from all chunks will be merged automatically."""

        return f"""Extract metadata from this {doc_type} document. Find ALL instances and provide them in a SINGLE JSON response.

IMPORTANT: Respond with exactly ONE JSON object. Return null for fields not found in the text.{chunk_info}
{notes_block}{preamble_text}

Fields to extract:
{fields_str}

Document text:
{text}

Respond with a SINGLE JSON object:
{{
  "data": {{
    "field_key": "extracted_value",
    ...
  }},
  "evidence": {{
    "field_key": [
      {{"page": 0, "start": 10, "end": 30, "quote": "exact text span"}},
      {{"page": 1, "start": 20, "end": 40, "quote": "another text span"}}
    ],
    ...
  }}
}}

CRITICAL: The "evidence" field MUST be an OBJECT (not an array), where each key is a field name from "data" and the value is an array of evidence objects for that field. For example:
- If you extract "iban": "NL35...", then evidence should have "iban": [{{"page": 0, "start": 20, "end": 40, "quote": "NL35 INGB..."}}]
- If you extract "naam": "Company Name", then evidence should have "naam": [{{"page": 0, "start": 10, "end": 30, "quote": "Company Name"}}]

Each field in "data" should have a corresponding key in "evidence" with an array of evidence objects."""

    def _fill_missing_quotes(self, result: Dict[str, Any], pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fill in missing quote fields in evidence from document text."""
        if "evidence" not in result or not isinstance(result["evidence"], dict):
            return result
        
        for field_key, spans in result["evidence"].items():
            if not isinstance(spans, list):
                continue
            
            for span in spans:
                if not isinstance(span, dict):
                    continue
                
                # If quote is missing or empty, try to extract it from the text
                if not span.get("quote"):
                    page_num = span.get("page", 0)
                    start = span.get("start", 0)
                    end = span.get("end", 0)
                    
                    if page_num < len(pages) and start < end:
                        page_text = pages[page_num].get("text", "")
                        if end <= len(page_text):
                            span["quote"] = page_text[start:end]
                            logger.debug(f"Filled missing quote for {field_key}: '{span['quote']}'")
                        else:
                            # End is beyond text length, take what we can
                            span["quote"] = page_text[start:] if start < len(page_text) else ""
                            logger.warning(f"Quote range {start}:{end} exceeds page text length {len(page_text)}")
        
        return result

    def _apply_regex_filters(self, result: Dict[str, Any], fields: List[Tuple], llm_dir: Path) -> Dict[str, Any]:
        """Apply regex post-processing to extracted values.
        
        For fields with a regex pattern defined, extract only the matching part
        from the LLM-extracted value. This cleans up values like 
        "NL59 RABO 0304 4232 11 EUR" to just the IBAN part.
        Also updates evidence quotes to match the filtered value.
        """
        if "data" not in result or not isinstance(result["data"], dict):
            return result
        
        # Build field regex lookup: {key: regex}
        field_regex = {}
        for field_tuple in fields:
            key, label, field_type, is_required, enum_values, regex = field_tuple
            if regex:
                field_regex[key] = regex
        
        if not field_regex:
            return result  # No regex patterns to apply
        
        regex_corrections = {}
        
        for field_key, pattern in field_regex.items():
            if field_key not in result["data"]:
                continue
            
            value = result["data"][field_key]
            if not value or not isinstance(value, str):
                continue
            
            original_value = value
            
            try:
                # Try to find a match in the value
                # First normalize whitespace in value for better matching
                normalized_value = ' '.join(value.split())  # Normalize whitespace
                
                match = None
                # If pattern ends with $, try to match without the $ first (to handle trailing text)
                # This helps with cases like "NL59 RABO 0304 4232 11 EUR" where EUR should be removed
                pattern_for_search = pattern
                if pattern.endswith('$') and not pattern.startswith('^'):
                    # Pattern like "pattern$" - remove $ to search within string
                    pattern_for_search = pattern[:-1]
                elif pattern.startswith('^') and pattern.endswith('$'):
                    # Full string match pattern - try exact match first
                    match = re.match(pattern, value, re.IGNORECASE) or re.match(pattern, normalized_value, re.IGNORECASE)
                    if not match:
                        # If exact match fails, try without anchors to find pattern within string
                        pattern_for_search = pattern[1:-1]  # Remove ^ and $
                
                if not match:
                    # Search for pattern within value
                    match = re.search(pattern_for_search, value, re.IGNORECASE) or re.search(pattern_for_search, normalized_value, re.IGNORECASE)
                
                if match:
                    # Use the matched group (group 0 = entire match)
                    matched_value = match.group(0).strip()
                    
                    # Only update if the match is different from original
                    if matched_value != original_value.strip():
                        result["data"][field_key] = matched_value
                        regex_corrections[field_key] = {
                            "original": original_value,
                            "corrected": matched_value,
                            "pattern": pattern
                        }
                        logger.info(f"Regex filter applied to {field_key}: '{original_value}' -> '{matched_value}'")
                        
                        # Update evidence quotes to match the filtered value
                        if "evidence" in result and isinstance(result["evidence"], dict):
                            if field_key in result["evidence"]:
                                spans = result["evidence"][field_key]
                                if isinstance(spans, list):
                                    for span in spans:
                                        if isinstance(span, dict) and "quote" in span:
                                            # Try to find the matched value in the original quote
                                            quote = span.get("quote", "")
                                            if matched_value in quote:
                                                # Update quote to show only the matched part
                                                # Try to find the exact position
                                                quote_lower = quote.lower()
                                                matched_lower = matched_value.lower()
                                                idx = quote_lower.find(matched_lower)
                                                if idx >= 0:
                                                    # Update quote to the matched portion
                                                    span["quote"] = quote[idx:idx+len(matched_value)]
                                                    logger.debug(f"Updated evidence quote for {field_key}: '{quote}' -> '{span['quote']}'")
                                                else:
                                                    # Fallback: use the matched value
                                                    span["quote"] = matched_value
                                            else:
                                                # Quote doesn't contain match, update to matched value
                                                span["quote"] = matched_value
                else:
                    # No match found - log warning but keep original value
                    logger.warning(f"Regex pattern '{pattern}' did not match value '{value}' for field {field_key}")
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern}' for field {field_key}: {e}")
        
        # Save regex corrections for debugging
        if regex_corrections:
            try:
                with open(llm_dir / "regex_corrections.json", "w") as f:
                    json.dump(regex_corrections, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save regex corrections: {e}")
        
        return result

    def _validate_evidence(self, evidence: ExtractionEvidence, pages: List[Dict[str, Any]]) -> List[str]:
        """Validate that evidence spans match the actual text."""
        errors = []

        for field_key, spans in evidence.evidence.items():
            if field_key not in evidence.data:
                continue

            value = evidence.data[field_key]
            if value is None:
                if spans:  # Should be empty array for null values
                    errors.append(f"{field_key}: Evidence provided for null value")
                continue

            if not spans:  # Should have evidence for non-null values
                errors.append(f"{field_key}: No evidence provided for value '{value}'")
                continue

            for span in spans:
                if span.page >= len(pages):
                    errors.append(f"{field_key}: Invalid page {span.page}")
                    continue

                page_text = pages[span.page]["text"]
                if span.start >= len(page_text) or span.end > len(page_text) or span.start >= span.end:
                    errors.append(f"{field_key}: Invalid span [{span.start}:{span.end}] for page {span.page}")
                    continue

                actual_quote = page_text[span.start:span.end]
                if actual_quote != span.quote:
                    errors.append(f"{field_key}: Quote mismatch - expected '{span.quote}', got '{actual_quote}'")

        return errors

    async def _stage_risk_analysis(self, document: Document, document_dir: Path,
                                 ocr_result: OCRResult,
                                 extraction_result: Optional[ExtractionEvidence]) -> RiskAnalysis:
        """Stage 5: Perform risk analysis and generate signals."""
        signals = []
        risk_score = 0

        # Signal 1: File metadata suspicious (PDF only)
        if document.mime_type == "application/pdf":
            try:
                reader = PdfReader(document_dir / "original" / document.original_filename)
                metadata = reader.metadata

                # Whitelist of known legitimate PDF producers from major software companies
                # Note: Only add producers that you KNOW are legitimate and commonly used
                legitimate_producers = [
                    "adobe", "microsoft", "google", "apple",  # Major software companies
                    "libreoffice", "openoffice", "inkscape",  # Open source tools
                    "ghostscript", "pdflib", "itext", "apache",  # Well-known PDF libraries
                ]
                
                # Suspicious patterns - these indicate potentially manipulated PDFs
                suspicious_patterns = [
                    "ilovepdf", "ilove pdf",  # Known suspicious online converter
                    "pdf converter", "pdfconverter",  # Generic converters (often used for manipulation)
                    "unknown", "unnamed",  # Missing metadata (could indicate manipulation)
                ]
                
                # Programming language PDF libraries - these CAN be used legitimately but also for fraud
                # We flag them but with a more informative message
                programmatic_libraries = [
                    "fpdf", "tcpdf", "dompdf", "mpdf",  # PHP PDF libraries
                    "python", "java", "php", "ruby",  # Programming languages
                ]
                
                producer = getattr(metadata, 'producer', '').lower() if metadata else ''
                
                # Check if it's a known legitimate producer
                if any(legit in producer for legit in legitimate_producers):
                    pass  # Legitimate producer, no risk signal
                # Check if it's a programmatic library (can be legitimate but also used for fraud)
                elif any(lib in producer for lib in programmatic_libraries):
                    # Find which library
                    detected_lib = next((lib for lib in programmatic_libraries if lib in producer), "programmatische PDF library")
                    signals.append(RiskSignal(
                        code="FILE_METADATA_PROGRAMMATIC",
                        severity="medium",
                        message=f"PDF gemaakt met {detected_lib}",
                        evidence=f"PDF metadata producer: {producer}. Dit kan legitiem zijn, maar wordt ook gebruikt om documenten te manipuleren."
                    ))
                    risk_score += 30
                # Check for suspicious patterns
                elif any(susp in producer for susp in suspicious_patterns):
                    signals.append(RiskSignal(
                        code="FILE_METADATA_SUSPICIOUS",
                        severity="medium",
                        message=f"Verdachte PDF maker: {producer}",
                        evidence=f"PDF metadata producer: {producer}. Dit wijst vaak op gebruik van online converters die gebruikt worden voor document manipulatie."
                    ))
                    risk_score += 30
            except Exception as e:
                logger.warning(f"PDF metadata analysis failed: {e}")

        # Signal 2: Text anomaly
        text = ocr_result.combined_text
        anomaly_score, metrics = self._analyze_text_anomalies(text)

        if anomaly_score > 0.3:
            severity = "high" if anomaly_score > 0.7 else "medium"
            # Build evidence string
            evidence_parts = [
                f"Unicode ratio: {metrics['unicode_ratio']:.2f}",
                f"Repetition ratio: {metrics['repetition_ratio']:.2f}"
            ]
            
            # Prepare examples for frontend
            examples_dict = {}
            if metrics.get('unicode_examples'):
                examples_dict['unicode_examples'] = metrics['unicode_examples']
            if metrics.get('repetition_examples'):
                examples_dict['repetition_examples'] = metrics['repetition_examples']
            
            signals.append(RiskSignal(
                code="TEXT_ANOMALY",
                severity=severity,
                message=f"Text anomalies detected (score: {anomaly_score:.2f})",
                evidence="; ".join(evidence_parts),
                examples=examples_dict if examples_dict else None
            ))
            risk_score += int(anomaly_score * 40)

        # Signal 3: Consistency check failed
        if extraction_result:
            consistency_errors = self._check_consistency(extraction_result.data)
            if consistency_errors:
                signals.append(RiskSignal(
                    code="CONSISTENCY_CHECK_FAILED",
                    severity="medium",
                    message="Inconsistent extracted data",
                    evidence="; ".join(consistency_errors)
                ))
                risk_score += 20

        # Signal 4: OCR low quality
        if ocr_result.ocr_used and ocr_result.ocr_quality == "low":
            signals.append(RiskSignal(
                code="OCR_LOW_QUALITY",
                severity="high",
                message="OCR quality is low, extraction may be unreliable",
                evidence=f"OCR quality assessment: {ocr_result.ocr_quality}"
            ))
            risk_score += 50

        # Cap risk score
        risk_score = min(100, risk_score)

        analysis = RiskAnalysis(risk_score=risk_score, signals=signals)

        # Save artifact
        risk_dir = document_dir / "risk"
        risk_dir.mkdir(exist_ok=True)

        with open(risk_dir / "result.json", "w") as f:
            json.dump(analysis.dict(), f, indent=2)

        return analysis

    def _analyze_text_anomalies(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """Analyze text for anomalies that might indicate manipulation."""
        if not text:
            return 1.0, {
                "unicode_ratio": 1.0,
                "repetition_ratio": 1.0,
                "unicode_examples": [],
                "repetition_examples": []
            }

        # Unicode anomaly ratio (non-ASCII characters)
        unicode_chars = len(re.findall(r'[^\x00-\x7F]', text))
        unicode_ratio = unicode_chars / len(text) if text else 0

        # Find unicode example sequences (first 3 unique sequences with unicode chars)
        unicode_examples = []
        if unicode_ratio > 0.1:  # Only collect examples if significant
            unicode_pattern = re.compile(r'[^\x00-\x7F]+')
            matches = unicode_pattern.findall(text)
            # Get unique sequences, max 3
            seen = set()
            for match in matches[:20]:  # Check first 20 matches
                if match not in seen and len(match.strip()) > 0:
                    seen.add(match)
                    unicode_examples.append(match.strip()[:50])  # Max 50 chars per example
                    if len(unicode_examples) >= 3:
                        break

        # Repetition ratio (repeated sequences)
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            repetition_ratio = 0
            repetition_examples = []
        else:
            unique_words = set(words)
            repetition_ratio = 1 - (len(unique_words) / len(words))
            
            # Find repetitive word sequences (words that appear 5+ times)
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Get most repeated words
            repeated_words = [(word, count) for word, count in word_counts.items() if count >= 5]
            repeated_words.sort(key=lambda x: x[1], reverse=True)
            
            # Find example sentences/phrases with these repeated words
            repetition_examples = []
            if repeated_words:
                # Find sentences containing the most repeated word
                sentences = re.split(r'[.!?]\s+', text)
                top_repeated_word = repeated_words[0][0]
                for sentence in sentences[:10]:  # Check first 10 sentences
                    if top_repeated_word.lower() in sentence.lower() and len(sentence.strip()) > 10:
                        # Count occurrences in this sentence
                        count_in_sentence = sentence.lower().count(top_repeated_word.lower())
                        if count_in_sentence >= 2:  # Word appears multiple times in sentence
                            repetition_examples.append(sentence.strip()[:100])  # Max 100 chars
                            if len(repetition_examples) >= 3:
                                break

        # Combined score
        anomaly_score = (unicode_ratio * 0.6) + (repetition_ratio * 0.4)

        return anomaly_score, {
            "unicode_ratio": unicode_ratio,
            "repetition_ratio": repetition_ratio,
            "unicode_examples": unicode_examples[:3],  # Max 3 examples
            "repetition_examples": repetition_examples[:3]  # Max 3 examples
        }

    def _check_consistency(self, data: Dict[str, Any]) -> List[str]:
        """Check for consistency in extracted data."""
        errors = []

        # Check amounts are non-negative
        for key, value in data.items():
            if "amount" in key.lower() and isinstance(value, (int, float)) and value < 0:
                errors.append(f"{key}: Negative amount {value}")

        # Check VAT <= total for invoices
        vat = data.get("vat_amount")
        total = data.get("total_amount")
        if vat is not None and total is not None and isinstance(vat, (int, float)) and isinstance(total, (int, float)):
            if vat > total:
                errors.append(f"VAT amount ({vat}) exceeds total amount ({total})")

        return errors

    async def _update_progress(self, document_id: int, progress: int, stage: str,
                             callback: callable = None,
                             doc_type_slug: str = None, doc_type_confidence: float = None) -> None:
        """Update document progress and call callback if provided."""
        from datetime import datetime
        await self.db.execute(
            text("UPDATE documents SET status = 'processing', progress = :progress, stage = :stage, updated_at = :updated_at WHERE id = :document_id"),
            {"progress": progress, "stage": stage, "document_id": document_id, "updated_at": datetime.now()}
        )
        await self.db.commit()

        if callback:
            await callback(document_id, progress, stage, doc_type_slug, doc_type_confidence)

    async def _finalize_document(self, document_id: int, classification: ClassificationResult,
                               extraction_result: Optional[ExtractionEvidence],
                               risk_analysis: RiskAnalysis, ocr_result: OCRResult, document_dir: Path) -> None:
        """Finalize document processing."""
        from datetime import datetime
        
        # Try to load classification scores from classification_local.json
        classification_scores = None
        classification_file = document_dir / "llm" / "classification_local.json"
        if classification_file.exists():
            try:
                with open(classification_file, "r") as f:
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
            "updated_at": datetime.now()
        }

        if extraction_result:
            update_data["metadata_json"] = json.dumps(extraction_result.data)
            update_data["metadata_evidence_json"] = json.dumps(self._json_serialize(extraction_result.evidence))
        
        # Store classification scores in metadata_validation_json (reusing existing JSON field)
        if classification_scores:
            update_data["metadata_validation_json"] = json.dumps({"classification_scores": classification_scores})

        # Build query with named parameters
        set_parts = [f"{k} = :{k}" for k in update_data.keys()]
        query = f"UPDATE documents SET {', '.join(set_parts)} WHERE id = :document_id"
        update_data["document_id"] = document_id

        await self.db.execute(text(query), update_data)
        await self.db.commit()

    async def _update_document_error(self, document_id: int, error_message: str) -> None:
        """Mark document as error."""
        from datetime import datetime
        await self.db.execute(
            text("UPDATE documents SET status = 'error', error_message = :error_message, updated_at = :updated_at WHERE id = :document_id"),
            {"error_message": error_message, "document_id": document_id, "updated_at": datetime.now()}
        )
        await self.db.commit()

    async def _get_document(self, document_id: int) -> Optional[Document]:
        """Get document by ID."""
        result = await self.db.execute(text("SELECT * FROM documents WHERE id = :document_id"), {"document_id": document_id})
        row = result.fetchone()
        if row:
            return Document(**row._mapping)
        return None