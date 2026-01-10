import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
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


class DocumentProcessor:
    def __init__(self, db_session: AsyncSession, llm_client: LLMClient):
        self.db = db_session
        self.llm = llm_client
        self.data_dir = Path(settings.data_dir)

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
            await self._update_progress(document_id, 60, "extracting_metadata", progress_callback)

            # Stage 4: Metadata Extraction (60-85%)
            if classification.doc_type_slug not in ["other", "unknown"]:
                extraction_result = await self._stage_metadata_extraction(
                    document, document_dir, classification, ocr_result
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
            await self._finalize_document(document_id, classification, extraction_result, risk_analysis, ocr_result)

        except Exception as e:
            logger.error(f"Document processing failed for {document_id}: {e}")
            await self._update_document_error(document_id, str(e))
            raise

    async def _stage_sniffing(self, document: Document, document_dir: Path) -> None:
        """Stage 1: File type detection and SHA256 computation."""
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
                    # Fall back to OCR
                    pix = page.get_pixmap(dpi=250)
                    img = Image.open(pix.tobytes("png"))
                    text = pytesseract.image_to_string(img, config=settings.tesseract_config)
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
            # Full OCR fallback
            doc = fitz.open(str(file_path))
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=250)
                img = Image.open(pix.tobytes("png"))
                text = pytesseract.image_to_string(img, config=settings.tesseract_config)

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
        """Extract text from image using OCR."""
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img, config=settings.tesseract_config)

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

        # Prepare text sample
        sample_text = self._prepare_text_sample(ocr_result.combined_text)

        # Step 1: Try deterministic classification first
        deterministic_result = self.classify_deterministic(sample_text, available_types)

        if deterministic_result:
            # Get the document type for metadata extraction
            doc_type_result = await self.db.execute(
                text("SELECT * FROM document_types WHERE slug = :slug"),
                {"slug": deterministic_result}
            )
            doc_type_row = doc_type_result.fetchone()

            # Check if this type has fields configured
            fields_result = await self.db.execute(
                text("SELECT * FROM document_type_fields WHERE document_type_id = :doc_type_id"),
                {"doc_type_id": doc_type_row.id}
            )
            fields = fields_result.fetchall()

            if fields:
                # Do metadata extraction for deterministic result
                classification = ClassificationResult(
                    doc_type_slug=deterministic_result,
                    confidence=0.95,
                    rationale=f"Deterministic match with strong evidence"
                )

                extraction_result = await self._stage_metadata_extraction(
                    document, document_dir, classification, ocr_result
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

                return classification, extraction_result
            else:
                # No fields configured, return unknown
                classification = ClassificationResult(
                    doc_type_slug="unknown",
                    confidence=0.0,
                    rationale=f"Document type '{deterministic_result}' has no fields configured"
                )
                return classification, None

        # Step 2: Combined LLM analysis
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
                    type_info += f"Classification hints: {hints}\n"
                if doc_type_row.extraction_prompt_preamble:
                    type_info += f"Extraction context: {doc_type_row.extraction_prompt_preamble}\n"

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

Then, if you classified it as a specific type (not 'unknown'), extract the metadata according to that type's field definitions.

IMPORTANT:
- If you cannot confidently classify the document, return 'unknown' and empty metadata
- For classification, you MUST provide exact evidence from the text
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
                            extraction_result = ExtractionEvidence(**extraction_data)

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
        """Stage 3: Classify document type using deterministic pre-classifier and LLM."""
        # Get available document types with hints
        result = await self.db.execute(text("SELECT slug, classification_hints FROM document_types"))
        available_types = result.fetchall()

        # Prepare text sample (first 6000 chars, normalized)
        sample_text = self._prepare_text_sample(ocr_result.combined_text)

        # Step 1: Deterministic pre-classification
        deterministic_result = self.classify_deterministic(sample_text, available_types)

        # Log deterministic result
        logger.info(f"Document {document.id} deterministic classification: {deterministic_result} (sample_text_length: {len(sample_text)})")

        if deterministic_result:
            # Strong deterministic match found
            classification = ClassificationResult(
                doc_type_slug=deterministic_result,
                confidence=0.95,  # High confidence for deterministic matches
                rationale=f"Deterministic match with strong evidence"
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

            logger.info(f"Document {document.id} classified as '{deterministic_result}' via deterministic matching")
            return classification

        # Step 2: LLM classification as fallback
        logger.info(f"Document {document.id} falling back to LLM classification")
        available_slugs = [slug for slug, _ in available_types] + ["unknown"]
        llm_result = await self._llm_classify_document(document_dir, sample_text, available_slugs)

        logger.info(f"Document {document.id} classified as '{llm_result.doc_type_slug}' via LLM (confidence: {llm_result.confidence})")
        return llm_result

    def _prepare_text_sample(self, text: str, max_chars: int = 6000) -> str:
        """Prepare text sample for classification by normalizing whitespace and including header."""
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', text.strip())

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

        return '\n'.join(header_lines)

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
                                       ocr_result: OCRResult) -> Optional[ExtractionEvidence]:
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

        # Build schema and prompt
        schema = self._build_extraction_schema(fields)
        prompt = self._build_extraction_prompt(fields, ocr_result.combined_text, classification.doc_type_slug, preamble)

        llm_dir = document_dir / "llm"
        llm_dir.mkdir(exist_ok=True)

        with open(llm_dir / "extraction_prompt.txt", "w") as f:
            f.write(prompt)

        with open(llm_dir / "extraction_schema.json", "w") as f:
            json.dump(schema, f, indent=2)

        curl_command = None
        response_text = None
        result = None
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

        # Fill in missing quotes from document text
        result = self._fill_missing_quotes(result, ocr_result.pages)
        
        # Apply regex post-processing to clean extracted values
        result = self._apply_regex_filters(result, fields, llm_dir)
        
        # Validate evidence spans
        evidence_data = ExtractionEvidence(**result)
        validation_errors = self._validate_evidence(evidence_data, ocr_result.pages)

        # Save results
        metadata_dir = document_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        with open(metadata_dir / "result.json", "w") as f:
            json.dump(evidence_data.data, f, indent=2)

        with open(metadata_dir / "validation.json", "w") as f:
            json.dump({"errors": validation_errors}, f, indent=2)

        with open(metadata_dir / "evidence.json", "w") as f:
            json.dump(self._json_serialize(evidence_data.evidence), f, indent=2)

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

    def _build_extraction_prompt(self, fields: List[Tuple], text: str, doc_type: str, preamble: str = "") -> str:
        """Build extraction prompt for LLM."""
        field_descriptions = []
        for key, label, field_type, is_required, enum_values, regex in fields:
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

        return f"""Extract metadata from this {doc_type} document. Find ALL instances across all pages and provide them in a SINGLE JSON response.

IMPORTANT: Respond with exactly ONE JSON object containing ALL found data. Do not return separate JSON objects for different pages.

Fields to extract:
{fields_str}

Document text:
{text[:8000]}

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
}}"""

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
                match = re.search(pattern, value, re.IGNORECASE)
                if match:
                    # Use the matched group (group 0 = entire match)
                    matched_value = match.group(0)
                    
                    # Only update if the match is different from original
                    if matched_value != original_value:
                        result["data"][field_key] = matched_value
                        regex_corrections[field_key] = {
                            "original": original_value,
                            "corrected": matched_value,
                            "pattern": pattern
                        }
                        logger.info(f"Regex filter applied to {field_key}: '{original_value}' -> '{matched_value}'")
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

                suspicious_producers = ["ilove", "pdf", "converter", "unknown"]
                producer = getattr(metadata, 'producer', '').lower() if metadata else ''

                if any(susp in producer for susp in suspicious_producers):
                    signals.append(RiskSignal(
                        code="FILE_METADATA_SUSPICIOUS",
                        severity="medium",
                        message=f"Suspicious PDF producer: {producer}",
                        evidence=f"PDF metadata producer field: {producer}"
                    ))
                    risk_score += 30
            except Exception as e:
                logger.warning(f"PDF metadata analysis failed: {e}")

        # Signal 2: Text anomaly
        text = ocr_result.combined_text
        anomaly_score, metrics = self._analyze_text_anomalies(text)

        if anomaly_score > 0.3:
            severity = "high" if anomaly_score > 0.7 else "medium"
            signals.append(RiskSignal(
                code="TEXT_ANOMALY",
                severity=severity,
                message=f"Text anomalies detected (score: {anomaly_score:.2f})",
                evidence=f"Unicode ratio: {metrics['unicode_ratio']:.2f}, repetition ratio: {metrics['repetition_ratio']:.2f}"
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

    def _analyze_text_anomalies(self, text: str) -> Tuple[float, Dict[str, float]]:
        """Analyze text for anomalies that might indicate manipulation."""
        if not text:
            return 1.0, {"unicode_ratio": 1.0, "repetition_ratio": 1.0}

        # Unicode anomaly ratio (non-ASCII characters)
        unicode_chars = len(re.findall(r'[^\x00-\x7F]', text))
        unicode_ratio = unicode_chars / len(text) if text else 0

        # Repetition ratio (repeated sequences)
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            repetition_ratio = 0
        else:
            unique_words = set(words)
            repetition_ratio = 1 - (len(unique_words) / len(words))

        # Combined score
        anomaly_score = (unicode_ratio * 0.6) + (repetition_ratio * 0.4)

        return anomaly_score, {
            "unicode_ratio": unicode_ratio,
            "repetition_ratio": repetition_ratio
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
                             callback: callable = None) -> None:
        """Update document progress and call callback if provided."""
        from datetime import datetime
        await self.db.execute(
            text("UPDATE documents SET status = 'processing', progress = :progress, stage = :stage, updated_at = :updated_at WHERE id = :document_id"),
            {"progress": progress, "stage": stage, "document_id": document_id, "updated_at": datetime.now()}
        )
        await self.db.commit()

        if callback:
            await callback(document_id, progress, stage)

    async def _finalize_document(self, document_id: int, classification: ClassificationResult,
                               extraction_result: Optional[ExtractionEvidence],
                               risk_analysis: RiskAnalysis, ocr_result: OCRResult) -> None:
        """Finalize document processing."""
        from datetime import datetime
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
            "updated_at": datetime.now()
        }

        if extraction_result:
            update_data["metadata_json"] = json.dumps(extraction_result.data)
            update_data["metadata_evidence_json"] = json.dumps(self._json_serialize(extraction_result.evidence))

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