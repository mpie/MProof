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
            risk_analysis = await self._stage_unified_fraud_analysis(
                document,
                document_dir,
                ocr_result,
                classification,
                extraction_result,
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
        with open(text_dir / "extracted.json", "w", encoding="utf-8") as f:
            json.dump(result.dict(), f, indent=2, ensure_ascii=False)

        with open(text_dir / "extracted.txt", "w", encoding="utf-8") as f:
            f.write(combined_text)

        return result

    async def _ocr_with_rotation_detection(self, img: Image.Image) -> str:
        """Perform OCR on image, trying different rotations (0, 90, 180, 270) and return best result.
        
        Optimized: First tries 0°, only tries other rotations if 0° doesn't produce good results.
        
        Args:
            img: PIL Image to perform OCR on
            
        Returns:
            Best OCR text result from all rotations
        """
        results = []
        
        # First try 0° (no rotation) - most common case
        try:
            async with OCR_SEMAPHORE:
                text_0 = await asyncio.to_thread(
                    pytesseract.image_to_string, img, config=settings.tesseract_config
                )
            
            alnum_count_0 = sum(1 for c in text_0 if c.isalnum())
            word_count_0 = len(text_0.split())
            score_0 = alnum_count_0 * 2 + word_count_0
            
            results.append({
                'angle': 0,
                'text': text_0,
                'score': score_0,
                'alnum_count': alnum_count_0,
                'word_count': word_count_0
            })
            
            logger.debug(f"OCR rotation 0°: {alnum_count_0} alnum chars, {word_count_0} words, score: {score_0}")
            
            # If 0° produces good results (reasonable amount of text), skip other rotations
            # Threshold: at least 50 alphanumeric characters and 10 words indicates readable text
            if alnum_count_0 >= 50 and word_count_0 >= 10:
                logger.debug(f"OCR 0° produces good results ({alnum_count_0} alnum, {word_count_0} words), skipping other rotations")
                return text_0
                
        except Exception as e:
            logger.warning(f"OCR failed for rotation 0°: {e}")
        
        # If 0° didn't produce good results, try other rotations
        for angle in [90, 180, 270]:
            try:
                # Rotate image
                rotated_img = img.rotate(-angle, expand=True)  # Negative for counter-clockwise
                
                # Perform OCR (wrapped in semaphore + to_thread)
                async with OCR_SEMAPHORE:
                    text = await asyncio.to_thread(
                        pytesseract.image_to_string, rotated_img, config=settings.tesseract_config
                    )
                
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
        """Extract text from PDF using multiple methods, falling back to OCR if needed."""
        pages = []
        combined_text = ""
        ocr_used = False

        # Try multiple extraction methods in order of preference
        extraction_methods = [
            ("pymupdf_text", lambda page: page.get_text("text")),  # Plain text extraction
            ("pymupdf_rawtext", lambda page: page.get_text("rawtext")),  # Raw text (preserves layout)
            ("pdfminer", None),  # Will be handled separately
        ]

        try:
            doc = fitz.open(str(file_path))
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = ""
                source = "text-layer"
                
                # Method 1: Try PyMuPDF text extraction (best quality)
                try:
                    text = page.get_text("text")
                    logger.debug(f"Page {page_num}: PyMuPDF text extraction: {len(text.strip())} chars")
                    if not text or len(text.strip()) < 50:
                        # Try rawtext as fallback
                        text = page.get_text("rawtext")
                        source = "text-layer-raw"
                        logger.debug(f"Page {page_num}: PyMuPDF rawtext extraction: {len(text.strip())} chars")
                except Exception as e:
                    logger.debug(f"PyMuPDF text extraction failed for page {page_num}: {e}")
                
                # Method 2: If PyMuPDF fails, yields poor results, or garbage text, try pypdf
                is_garbage = self._is_garbage_text(text)
                if len(text.strip()) < 200 or self._is_mostly_empty(text) or is_garbage:
                    logger.info(f"Page {page_num}: PyMuPDF text inadequate (len={len(text.strip())}, garbage={is_garbage}), trying pypdf")
                    try:
                        from pypdf import PdfReader
                        reader = PdfReader(str(file_path))
                        if page_num < len(reader.pages):
                            page_obj = reader.pages[page_num]
                            text = page_obj.extract_text()
                            source = "pypdf"
                            logger.info(f"Using pypdf for page {page_num} ({len(text.strip())} chars)")
                    except Exception as e:
                        logger.debug(f"pypdf extraction failed for page {page_num}: {e}")
                        
                    # If pypdf also failed, still poor, or garbage, try pdfminer as last resort
                    is_garbage = self._is_garbage_text(text)
                    if len(text.strip()) < 200 or self._is_mostly_empty(text) or is_garbage:
                        logger.info(f"Page {page_num}: pypdf text inadequate (len={len(text.strip())}, garbage={is_garbage}), trying pdfminer")
                        try:
                            from pdfminer.high_level import extract_text as pdfminer_extract
                            # pdfminer doesn't support per-page extraction well, so we extract all
                            # This is not ideal but better than nothing
                            full_text = pdfminer_extract(str(file_path))
                            # Try to split by page breaks (form feed characters or double newlines)
                            page_texts = full_text.split('\f')
                            if len(page_texts) > page_num:
                                text = page_texts[page_num]
                            elif len(page_texts) == 1 and len(doc) > 1:
                                # No page breaks found, try to split evenly
                                lines = full_text.split('\n')
                                lines_per_page = len(lines) // len(doc) if len(doc) > 0 else len(lines)
                                start_line = page_num * lines_per_page
                                end_line = (page_num + 1) * lines_per_page if page_num < len(doc) - 1 else len(lines)
                                text = '\n'.join(lines[start_line:end_line])
                            else:
                                text = full_text
                            source = "pdfminer"
                            logger.info(f"Using pdfminer for page {page_num} ({len(text.strip())} chars)")
                        except Exception as e:
                            logger.debug(f"pdfminer extraction failed for page {page_num}: {e}")
                
                # Method 3: If all text extraction fails or text is garbage, use OCR
                is_garbage = self._is_garbage_text(text)
                needs_ocr = (
                    len(text.strip()) < 200 or 
                    self._is_mostly_empty(text) or 
                    is_garbage
                )
                if needs_ocr:
                    if len(text.strip()) < 200:
                        reason = "too short"
                    elif self._is_mostly_empty(text):
                        reason = "mostly empty"
                    else:
                        reason = "garbage text"
                    logger.info(f"Page {page_num}: All extractors failed ({reason}, len={len(text.strip())}, garbage={is_garbage}), using OCR")
                    async def render_page():
                        async with OCR_SEMAPHORE:
                            return await asyncio.to_thread(page.get_pixmap, dpi=250)
                    pix = await render_page()
                    img = Image.open(BytesIO(pix.tobytes("png")))
                    text = await self._ocr_with_rotation_detection(img)
                    source = "ocr"
                    ocr_used = True

                pages.append({
                    "page": page_num,
                    "source": source,
                    "text": text
                })
                combined_text += text + "\n\n"  # Add extra newline between pages
                
                logger.info(f"Page {page_num}: Extracted {len(text.strip())} chars using {source}")

            doc.close()
            
            logger.info(f"PDF extraction complete: {len(pages)} pages, {len(combined_text.strip())} total chars, OCR used: {ocr_used}")

        except Exception as e:
            logger.warning(f"PDF text extraction failed, trying full OCR: {e}")
            # Full OCR fallback with rotation detection
            try:
                doc = fitz.open(str(file_path))
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    async def render_page():
                        async with OCR_SEMAPHORE:
                            return await asyncio.to_thread(page.get_pixmap, dpi=250)
                    pix = await render_page()
                    img = Image.open(BytesIO(pix.tobytes("png")))
                    text = await self._ocr_with_rotation_detection(img)

                    pages.append({
                        "page": page_num,
                        "source": "ocr",
                        "text": text
                    })
                    combined_text += text + "\n\n"
                    ocr_used = True

                doc.close()
            except Exception as ocr_error:
                logger.error(f"OCR fallback also failed: {ocr_error}")
                raise

        return pages, combined_text, ocr_used

    async def _extract_image_text(self, file_path: Path) -> Tuple[List[Dict[str, Any]], str, bool]:
        """Extract text from image using OCR with rotation detection."""
        img = Image.open(file_path)
        text = await self._ocr_with_rotation_detection(img)

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

    def _is_garbage_text(self, text: str) -> bool:
        """Detect garbage text from corrupted PDF encoding or wrong character mapping.
        
        This happens when PDF has embedded fonts with custom encoding that don't 
        map to standard characters, resulting in readable-looking but meaningless text.
        """
        if not text or len(text.strip()) < 50:
            logger.info(f"Garbage detection: text too short ({len(text.strip()) if text else 0} chars)")
            return True
        
        # Clean text for analysis
        cleaned = text.strip()
        
        # Check 1: Too many special/punctuation characters relative to letters
        # Garbage text often has patterns like &'/!%.&'1&+$
        letters = len(re.findall(r'[a-zA-Z]', cleaned))
        special_chars = len(re.findall(r'[&%$#@!^*+=<>|\\~`\'\"/]', cleaned))
        if letters > 0 and special_chars / letters > 0.5:
            logger.info(f"Garbage detection: high special char ratio ({special_chars}/{letters} = {special_chars/letters:.2f})")
            return True
        
        # Check 2: Very few recognizable words (3+ consecutive letters)
        words = re.findall(r'[a-zA-Z]{3,}', cleaned)
        if len(cleaned) > 100 and len(words) < 5:
            logger.info(f"Garbage detection: too few words ({len(words)} in {len(cleaned)} chars)")
            return True
        
        # Check 3: High ratio of control characters or weird unicode
        control_chars = len(re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', cleaned))
        if control_chars > len(cleaned) * 0.05:
            logger.info(f"Garbage detection: high control char ratio ({control_chars}/{len(cleaned)})")
            return True
        
        # Check 4: Check for common Dutch/English word patterns
        # If text has enough letters but no common word patterns, it's likely garbage
        common_patterns = [
            r'\b(de|het|een|en|van|in|is|dat|op|te|voor|met|aan|zijn|worden|door|als|naar|uit|over|ook|meer|tot|bij|nog|dan|wel|om|of|kan|dit|niet|maar|er|al|wat|hebben|was|jaar|zou|gaan|na|zo|ons|die|hier|wordt)\b',  # Dutch
            r'\b(the|a|an|is|are|was|were|be|been|have|has|had|do|does|did|will|would|could|should|may|might|must|shall|can|to|of|and|in|for|on|with|at|by|from|or|as|but|not|this|that|it|he|she|we|they|you|i)\b',  # English
        ]
        
        text_lower = cleaned.lower()
        common_word_count = 0
        for pattern in common_patterns:
            common_word_count += len(re.findall(pattern, text_lower))
        
        # If we have significant text but very few common words, likely garbage
        if len(words) > 10 and common_word_count < 3:
            logger.info(f"Garbage detection: no common words found ({common_word_count} in {len(words)} words)")
            return True
        
        # Check 5: Look for specific garbage patterns common in corrupted PDFs
        # Patterns like #+0#,1#.-#. or &'/!%.&'1&+$
        garbage_patterns = [
            r'[#&][^a-zA-Z\s]{3,}',  # # or & followed by 3+ non-letter chars
            r'[+\-*/]{2,}',  # Multiple math operators in a row
            r'\d[#&%]\d',  # Number-symbol-number patterns
        ]
        garbage_matches = 0
        for pattern in garbage_patterns:
            garbage_matches += len(re.findall(pattern, cleaned))
        
        if garbage_matches >= 3:
            logger.info(f"Garbage detection: found {garbage_matches} garbage patterns")
            return True
        
        logger.debug(f"Garbage detection: text appears valid ({len(cleaned)} chars, {len(words)} words, {common_word_count} common)")
        return False

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

    async def classify_deterministic_strong(self, text: str, available_types: List[Tuple[str, str]]) -> Tuple[Optional[str], Optional[List[str]]]:
        """Check for STRONG deterministic matches where ALL kw: rules match.
        
        This runs BEFORE trained models to ensure explicit keyword rules have priority.
        Only returns a match if a document type has kw: rules AND ALL of them match.
        
        Args:
            text: Document text to analyze
            available_types: List of (slug, classification_hints) tuples
            
        Returns:
            Tuple of (document type slug if ALL kw: rules match, list of matched keywords), or (None, None)
        """
        text_lower = text.lower()
        strong_matches = []
        
        for slug, hints in available_types:
            if not hints:
                continue
                
            required_keywords = []
            matched_keywords = []
            disqualified = False
            
            for hint_line in hints.strip().split('\n'):
                hint_line = hint_line.strip()
                if not hint_line:
                    continue
                    
                if hint_line.startswith('kw:'):
                    # Required keyword - ALL must match
                    keyword = hint_line[3:].strip().lower()
                    required_keywords.append(keyword)
                    if keyword in text_lower:
                        matched_keywords.append(keyword)
                elif hint_line.startswith('not:'):
                    # Negative keyword - must NOT appear
                    negative_word = hint_line[4:].strip().lower()
                    if negative_word in text_lower:
                        disqualified = True
                        break
            
            if disqualified:
                continue
                
            # Strong match = has kw: rules AND ALL matched
            if required_keywords and len(matched_keywords) == len(required_keywords):
                strong_matches.append((slug, len(required_keywords), matched_keywords))
                logger.info(f"Strong deterministic match: '{slug}' - all {len(required_keywords)} kw: rules matched: {matched_keywords}")
        
        if not strong_matches:
            return None, None
            
        # If multiple strong matches, pick the one with most keywords
        strong_matches.sort(key=lambda x: x[1], reverse=True)
        best_match, _, matched_keywords = strong_matches[0]
        
        if len(strong_matches) > 1:
            logger.warning(f"Multiple strong matches found: {[s[0] for s in strong_matches]}, using '{best_match}' (most keywords)")
        
        return best_match, matched_keywords

    async def _validate_classification_against_not_rules(
        self, predicted_slug: str, text: str, available_types: List[Tuple[str, str]], check_keywords: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """Validate that a classification doesn't violate any not: rules AND optionally has required kw: matches.
        
        Args:
            predicted_slug: The predicted document type slug
            text: Document text to check
            available_types: List of (slug, classification_hints) tuples
            check_keywords: If True, also check that at least one kw: rule matches (for Naive Bayes).
                          If False, only check not: rules (for BERT, which is semantic and doesn't need keywords).
            
        Returns:
            Tuple of (is_valid, reason) where reason explains why it was rejected (if not valid)
        """
        text_lower = text.lower()
        
        # Find the hints for the predicted type
        hints = None
        for slug, type_hints in available_types:
            if slug == predicted_slug:
                hints = type_hints
                break
        
        if not hints:
            return True, None  # No hints = no rules to check
        
        required_keywords = []
        matched_keywords = []
        violated_not_rule = None
        
        # Check all rules
        for hint_line in hints.strip().split('\n'):
            hint_line = hint_line.strip()
            if not hint_line:
                continue
                
            if hint_line.startswith('not:'):
                # not: rules - if found, reject
                negative_word = hint_line[4:].strip().lower()
                if negative_word in text_lower:
                    violated_not_rule = negative_word
                    logger.info(f"Classification '{predicted_slug}' rejected: not: rule '{negative_word}' found in text")
                    return False, f"not: rule '{negative_word}' gevonden in tekst (dit is de bedoeling - deze regel voorkomt verkeerde classificatie)"
            elif hint_line.startswith('kw:'):
                # kw: rules - track required keywords
                keyword = hint_line[3:].strip().lower()
                required_keywords.append(keyword)
                if keyword in text_lower:
                    matched_keywords.append(keyword)
        
        # If there are required keywords, at least one must match for Naive Bayes to be valid
        # BERT is semantic and doesn't need keyword matching, so we skip this check for BERT
        if check_keywords and required_keywords and len(matched_keywords) == 0:
            reason = f"geen kw: regels gematcht (vereist: {', '.join(required_keywords)}) - dit is de bedoeling om verkeerde classificaties te voorkomen"
            logger.info(f"Classification '{predicted_slug}' rejected: {reason}")
            return False, reason
        
        return True, None

    async def classify_deterministic(self, text: str, available_types: List[Tuple[str, str]]) -> Optional[str]:
        """Deterministic pre-classifier that only returns a type if there's strong evidence.
        
        Checks both classification_hints AND required fields from document_type_fields.
        If a document type has required fields, those must be detectable in the text.

        Args:
            text: Document text to analyze
            available_types: List of (slug, classification_hints) tuples

        Returns:
            Document type slug if strong evidence exists, None otherwise
        """
        text_lower = text.lower()
        scores = {}

        for slug, hints in available_types:
            score = 0
            disqualified = False
            required_keywords = []  # Track all required keywords (kw:)
            optional_keywords = []  # Track optional keywords (legacy format)
            matched_required = []  # Track which required keywords matched
            matched_optional = []  # Track which optional keywords matched

            # Parse hints - supports both structured (kw:, re:, not:) and simple keyword format
            if hints:
                for hint_line in hints.strip().split('\n'):
                    hint_line = hint_line.strip()
                    if not hint_line:
                        continue

                    if hint_line.startswith('kw:'):
                        # Structured: Required keywords (case-insensitive) - ALL must match
                        keyword = hint_line[3:].strip().lower()
                        required_keywords.append(keyword)
                        if keyword in text_lower:
                            matched_required.append(keyword)
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
                        # Legacy format: treat as optional keyword (case-insensitive)
                        keyword = hint_line.lower()
                        optional_keywords.append(keyword)
                        if keyword in text_lower:
                            matched_optional.append(keyword)
                            score += 1
                            logger.debug(f"Legacy keyword match '{keyword}' for {slug}")

            if disqualified:
                continue

            # Check required fields from document_type_fields
            required_fields = []
            matched_required_fields = []
            
            try:
                fields_result = await self.db.execute(
                    text("""
                        SELECT key, label, regex 
                        FROM document_type_fields 
                        WHERE document_type_id = (SELECT id FROM document_types WHERE slug = :slug)
                        AND required = 1
                    """),
                    {"slug": slug}
                )
                required_fields_rows = fields_result.fetchall()
                
                for field_key, field_label, field_regex in required_fields_rows:
                    required_fields.append((field_key, field_label, field_regex))
                    
                    # Check if required field is detectable in text
                    field_detected = False
                    
                    # Strategy 1: Check if field key or label appears in text
                    if field_key.lower() in text_lower or field_label.lower() in text_lower:
                        field_detected = True
                    
                    # Strategy 2: If field has regex, check if it matches
                    if not field_detected and field_regex:
                        try:
                            if re.search(field_regex, text, re.IGNORECASE):
                                field_detected = True
                        except re.error:
                            logger.warning(f"Invalid regex in required field '{field_key}' for {slug}: {field_regex}")
                    
                    # Strategy 3: Check for common patterns based on field key
                    if not field_detected:
                        key_lower = field_key.lower()
                        if 'iban' in key_lower:
                            # IBAN pattern: NL followed by 2 digits and 10+ alphanumeric
                            if re.search(r'\bNL\d{2}[A-Z0-9]{10,}\b', text, re.IGNORECASE):
                                field_detected = True
                        elif 'rekening' in key_lower or 'account' in key_lower:
                            # Account number patterns
                            if re.search(r'\b\d{8,}\b', text) or re.search(r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,}\b', text, re.IGNORECASE):
                                field_detected = True
                        elif 'datum' in key_lower or 'date' in key_lower:
                            # Date patterns
                            if re.search(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', text):
                                field_detected = True
                        elif 'bedrag' in key_lower or 'amount' in key_lower or 'saldo' in key_lower:
                            # Amount patterns (EUR, €, numbers with decimals)
                            if re.search(r'\b\d+[.,]\d{2}\b', text) or re.search(r'€\s*\d+', text, re.IGNORECASE) or re.search(r'EUR\s*\d+', text, re.IGNORECASE):
                                field_detected = True
                        elif 'naam' in key_lower or 'name' in key_lower:
                            # Name patterns (capitalized words)
                            if re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text):
                                field_detected = True
                        elif 'adres' in key_lower or 'address' in key_lower:
                            # Address patterns (street names, postal codes)
                            if re.search(r'\b\d{4}\s*[A-Z]{2}\b', text) or re.search(r'\b[A-Z][a-z]+\s+\d+[a-z]?\b', text):
                                field_detected = True
                    
                    if field_detected:
                        matched_required_fields.append(field_key)
                        score += 2  # Required fields are worth more
                    else:
                        # Required field not detected - this type cannot match
                        logger.debug(f"Document type '{slug}' skipped: required field '{field_key}' not detected in text")
                        disqualified = True
                        break
            except Exception as e:
                logger.warning(f"Failed to check required fields for {slug}: {e}")

            if disqualified:
                continue

            # If there are required keywords (kw:), ALL must match for this type to be eligible
            if required_keywords:
                if len(matched_required) != len(required_keywords):
                    # Not all required keywords matched - skip this type
                    logger.debug(f"Document type '{slug}' skipped: only {len(matched_required)}/{len(required_keywords)} required keywords matched (matched: {matched_required}, required: {required_keywords})")
                    continue

            # Only add to scores if we have matches (and all required keywords/fields matched)
            if score > 0:
                scores[slug] = score
                logger.debug(f"Document type '{slug}' scored {score} (required keywords: {len(matched_required)}/{len(required_keywords)}, required fields: {len(matched_required_fields)}/{len(required_fields)}, optional: {len(matched_optional)})")

        if not scores:
            logger.debug("No deterministic matches found (no scores after required keyword/field checks)")
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

        # Step 0: Check for STRONG deterministic matches first (all kw: rules match)
        # This ensures explicit keyword rules have priority over trained models
        strong_deterministic_result = await self.classify_deterministic_strong(sample_text, available_types)
        if strong_deterministic_result:
            logger.info(f"Document {document.id} classified as '{strong_deterministic_result}' via STRONG deterministic match (all kw: rules matched)")
            # Skip to using this result - still run NB/BERT for scores but don't override
            classifier_result = strong_deterministic_result
            classifier_confidence = 1.0  # Strong match = 100% confidence

        # Step 1: Try local (trained) Naive Bayes classifier
        nb_pred = None
        bert_pred = None
        if not strong_deterministic_result:
            classifier_result = None
            classifier_confidence = 0.0
        
        try:
            from app.services.doc_type_classifier import classifier_service
            pred = classifier_service().predict(sample_text, allowed_labels=allowed_slugs, model_name=self.model_name)
            if pred:
                nb_pred = pred
                # Only use NB result if no strong deterministic match
                if not strong_deterministic_result:
                    classifier_result = pred.label
                    classifier_confidence = pred.confidence
                logger.info(f"Document {document.id} classified as '{pred.label}' via Naive Bayes (p={pred.confidence:.2f}, model={self.model_name or 'default'})")
        except Exception as e:
            logger.warning(f"Naive Bayes classifier failed or unavailable: {e}")

        # Step 1.5: Try BERT classifier (always run to get score, even if NB is good)
        # Try selected model first, then fallback to other available models
        try:
            from app.services.bert_classifier import bert_classifier_service
            bert_svc = bert_classifier_service()
            
            # BERT is semantic context and last fallback. Do not silently borrow
            # unrelated model folders for hard classification.
            models_to_try = [self.model_name] if self.model_name else ["default"]
            
            bert_result = None
            bert_used_model = None
            
            for model_name in models_to_try:
                try:
                    result = bert_svc.predict(sample_text, model_name=model_name, allowed_labels=allowed_slugs)
                    if result:
                        bert_result = result
                        bert_used_model = model_name
                        logger.info(f"Document {document.id} BERT classification with model '{model_name}': '{result.label}' (p={result.confidence:.2f})")
                        break
                except Exception as e:
                    logger.debug(f"Document {document.id}: BERT model '{model_name}' failed: {e}")
                    continue
            
            if bert_result:
                bert_pred = bert_result
                # Use BERT result if: no strong deterministic match AND (no NB result, or BERT confidence is significantly higher)
                if not strong_deterministic_result and (not classifier_result or bert_result.confidence > classifier_confidence + 0.1):
                    classifier_result = bert_result.label
                    classifier_confidence = bert_result.confidence
                    logger.info(f"Document {document.id} classified as '{bert_result.label}' via BERT (p={bert_result.confidence:.2f}, model={bert_used_model or self.model_name or 'default'})")
                    if bert_used_model != self.model_name:
                        logger.info(f"Document {document.id}: Used BERT model '{bert_used_model}' (fallback from '{self.model_name or 'default'}')")
            else:
                # Log why BERT didn't return a result
                if self.model_name:
                    status = bert_svc.status(self.model_name)
                    if not status.get('model_exists'):
                        logger.warning(f"Document {document.id}: BERT model '{self.model_name}' not found, tried {len(models_to_try)} models")
                    else:
                        model_labels = status.get('labels', [])
                        missing_labels = [l for l in allowed_slugs if l not in model_labels]
                        if missing_labels:
                            logger.warning(f"Document {document.id}: BERT model '{self.model_name}' missing labels {missing_labels}, tried {len(models_to_try)} models")
                        else:
                            logger.info(f"Document {document.id}: BERT returned None for all {len(models_to_try)} models (score below threshold)")
                else:
                    logger.info(f"Document {document.id}: BERT returned None for all {len(models_to_try)} available models")
        except Exception as e:
            logger.warning(f"BERT classifier error: {e}")

        # Step 2: Fall back to deterministic if trained models didn't match
        if not classifier_result:
            classifier_result = await self.classify_deterministic(sample_text, available_types)
            if classifier_result:
                logger.info(f"Document {document.id} classified as '{classifier_result}' via deterministic matching (fallback)")

        if classifier_result and classifier_result != "unknown":
            # Get the document type for metadata extraction
            doc_type_result = await self.db.execute(
                text("SELECT * FROM document_types WHERE slug = :slug"),
                {"slug": classifier_result}
            )
            doc_type_row = doc_type_result.fetchone()

            if not doc_type_row:
                # Document type not found in database
                classification = ClassificationResult(
                    doc_type_slug="unknown",
                    confidence=0.0,
                    rationale=f"Document type '{classifier_result}' not found in database"
                )
                return classification, None

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
                    if bert_used_model:
                        classification_data["bert"]["model_used"] = bert_used_model
                with open(llm_dir / "classification_local.json", "w", encoding="utf-8") as f:
                    json.dump(classification_data, f, indent=2, ensure_ascii=False)

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
        """Combined classification and extraction in single LLM call (fallback when trained models fail)."""
        # For classification, only use hints - NOT all field definitions (too long)
        # Field definitions are only needed for extraction, which happens after classification
        available_slugs = [slug for slug, _ in available_types if slug != "unknown"]
        
        # Get classification hints only (much shorter than full field definitions)
        hints = []
        for slug, hint_text in available_types:
            if slug == "unknown" or not hint_text:
                continue
            hints.append(f"- {slug}: {hint_text}")

        hints_text = "\n".join(hints) if hints else "No classification hints available."

        prompt = f"""Classify this document into one of these types: {', '.join(available_slugs)}, or 'unknown' if it doesn't match any type.

CRITICAL CLASSIFICATION RULES:
- Choose a type ONLY if you can quote exact evidence from the text
- If you cannot prove any specific type, return 'unknown'
- NEVER guess or assume - only classify based on concrete evidence
- If multiple document types share common keywords, look for DISTINCTIVE features that differentiate them
- Pay close attention to the document's PURPOSE and STRUCTURE, not just individual keywords

Classification hints:
{hints_text}

Document text sample:
{sample_text}

Respond with JSON:
{{
  "doc_type_slug": "one of the types or 'unknown'",
  "confidence": 0.0-1.0,
  "rationale": "brief explanation",
  "evidence": "exact quote from text or empty (max 50 chars)"
}}"""

        schema = {
            "type": "object",
            "required": ["doc_type_slug", "confidence", "rationale", "evidence"],
            "properties": {
                "doc_type_slug": {"type": "string", "enum": available_slugs + ["unknown"]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "rationale": {"type": "string"},
                "evidence": {"type": "string"}
            }
        }

        llm_dir = document_dir / "llm"
        llm_dir.mkdir(exist_ok=True)

        with open(llm_dir / "combined_analysis_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        with open(llm_dir / "combined_analysis_schema.json", "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)

        curl_command = None
        try:
            logger.info(f"Starting combined LLM analysis for document {document_dir.parent.name}")
            result, response_text, curl_command, duration = await self.llm.generate_json_with_raw(prompt, schema)
            logger.info(f"Combined LLM analysis completed for document {document_dir.parent.name} in {duration:.2f}s")
            
            # Save response, curl command, and timing immediately after successful request
            with open(llm_dir / "combined_analysis_response.txt", "w", encoding="utf-8") as f:
                f.write(response_text)
            
            if curl_command:
                with open(llm_dir / "combined_analysis_curl.txt", "w", encoding="utf-8") as f:
                    f.write(curl_command)
            
            # Save timing metadata
            with open(llm_dir / "combined_analysis_timing.json", "w", encoding="utf-8") as f:
                json.dump({"duration_seconds": duration, "provider": self.llm.provider, "model": self.llm.model}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Combined LLM analysis failed for document {document_dir.parent.name}: {e}")
            with open(llm_dir / "combined_analysis_error.txt", "w", encoding="utf-8") as f:
                f.write(str(e))
            # Save curl command even if there was an error (if we got that far)
            if curl_command:
                with open(llm_dir / "combined_analysis_curl.txt", "w", encoding="utf-8") as f:
                    f.write(curl_command)
            raise

        # Parse classification result (now only classification, no extraction)
        # Validate classification (returns dict)
        validated_classification_data = self._validate_llm_classification(result, sample_text)
        # Convert to ClassificationResult object
        classification = ClassificationResult(**validated_classification_data)

        # Save classification result
        with open(llm_dir / "classification_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Return only classification - extraction will be done separately if needed
        # (via _stage_metadata_extraction which is called after classification)
        return classification, None

    def _normalize_extraction_data(self, extraction_data: Dict[str, Any], expected_fields: Optional[Set[str]] = None) -> Dict[str, Any]:
        """Normalize extraction data to ensure correct structure for ExtractionEvidence.
        
        Args:
            extraction_data: Raw extraction data from LLM
            expected_fields: Optional set of expected field keys to filter out unexpected fields
        """
        data = extraction_data.get("data", {})
        evidence = extraction_data.get("evidence", {})
        
        # Filter out unexpected fields if expected_fields is provided
        if expected_fields is not None:
            unexpected_in_data = [k for k in data.keys() if k not in expected_fields]
            if unexpected_in_data:
                logger.warning(f"Filtering out unexpected fields from data during normalization: {unexpected_in_data}")
                data = {k: v for k, v in data.items() if k in expected_fields}
                # Also filter evidence
                if isinstance(evidence, dict):
                    evidence = {k: v for k, v in evidence.items() if k in expected_fields}
        
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

        # Step 0: Check for STRONG deterministic matches first (all kw: rules match)
        # This ensures explicit keyword rules have priority over trained models
        strong_deterministic_result, strong_matched_keywords = await self.classify_deterministic_strong(sample_text, available_types)
        
        nb_pred = None
        bert_pred = None
        best_pred = None
        best_method = None
        
        if strong_deterministic_result:
            logger.info(f"Document {document.id} classified as '{strong_deterministic_result}' via STRONG deterministic match (all kw: rules matched: {strong_matched_keywords})")
            # Create a fake prediction with 100% confidence
            from dataclasses import dataclass
            @dataclass
            class StrongMatch:
                label: str
                confidence: float
                matched_keywords: Optional[List[str]] = None
            best_pred = StrongMatch(label=strong_deterministic_result, confidence=1.0, matched_keywords=strong_matched_keywords)
            best_method = "deterministic_strong"

        # Step 1: Local (trained) Naive Bayes classifier
        nb_error = None
        nb_below_threshold = None
        nb_threshold = None
        nb_all_scores = None  # Store all scores for all types
        try:
            from app.services.doc_type_classifier import classifier_service
            pred, threshold, raw_pred = classifier_service().predict_with_threshold_info(sample_text, allowed_labels=allowed_slugs, model_name=self.model_name)
            nb_threshold = threshold
            
            # Always get all scores, even if prediction fails
            try:
                nb_all_scores, _ = classifier_service().predict_all_scores_with_threshold(sample_text, allowed_labels=allowed_slugs, model_name=self.model_name)
                if nb_all_scores:
                    logger.info(f"Document {document.id} NB all scores: {[(k, f'{v:.2%}') for k, v in sorted(nb_all_scores.items(), key=lambda x: x[1], reverse=True)[:3]]}")
            except Exception as e:
                logger.debug(f"Could not get all NB scores: {e}")
            
            if pred:
                nb_pred = pred
                # Only use NB if no strong deterministic match
                if not strong_deterministic_result:
                    best_pred = pred
                    best_method = "naive_bayes"
                logger.info(f"Document {document.id} NB classification: '{pred.label}' (p={pred.confidence:.2f}, threshold={threshold:.2f})")
            elif raw_pred:
                # Prediction exists but below threshold
                nb_below_threshold = raw_pred
                logger.info(f"Document {document.id} NB classification: '{raw_pred.label}' (p={raw_pred.confidence:.2f}) BELOW threshold ({threshold:.2f})")
            else:
                logger.info(f"Document {document.id} NB classification: No result (no model or no prediction)")
        except Exception as e:
            nb_error = str(e)
            logger.warning(f"Naive Bayes classifier failed or unavailable: {e}")

        # Step 1.5: Try BERT classifier (always run to get score, even if NB is good)
        # Try selected model first, then fallback to other available models
        bert_error = None
        bert_validation_reason = None
        bert_models_tried = []
        try:
            from app.services.bert_classifier import bert_classifier_service
            bert_svc = bert_classifier_service()
            
            # Build list of models to try: selected model first, then all others
            models_to_try = []
            if self.model_name:
                models_to_try.append(self.model_name)
            
            # Add all other available models as fallback
            available_models = bert_svc.list_available_models()
            for model in available_models:
                if model != self.model_name:
                    models_to_try.append(model)
            
            # Also try "default" if not already in list
            if "default" not in models_to_try:
                models_to_try.append("default")
            
            bert_result = None
            bert_used_model = None
            
            for model_name in models_to_try:
                bert_models_tried.append(model_name)
                try:
                    result = bert_svc.predict(sample_text, model_name=model_name, allowed_labels=allowed_slugs)
                    if result:
                        bert_result = result
                        bert_used_model = model_name
                        logger.info(f"Document {document.id} BERT classification with model '{model_name}': '{result.label}' (p={result.confidence:.2f})")
                        break
                except Exception as e:
                    logger.debug(f"Document {document.id}: BERT model '{model_name}' failed: {e}")
                    continue
            
            if bert_result:
                bert_pred = bert_result
                # Use BERT only as last fallback when no stronger classifier produced a result.
                # It still gets persisted as semantic context even when it does not decide.
                MIN_CONFIDENCE_FOR_BERT = 0.5  # Minimum confidence to use BERT when NB has no result
                if not strong_deterministic_result:
                    if not best_pred:
                        # No NB result - use BERT only if confidence is high enough
                        if bert_result.confidence >= MIN_CONFIDENCE_FOR_BERT:
                            best_pred = bert_result
                            best_method = "bert"
                            logger.info(f"Document {document.id}: Using BERT result (NB had no result, confidence={bert_result.confidence:.2f})")
                        else:
                            logger.info(f"Document {document.id}: BERT result rejected (confidence {bert_result.confidence:.2f} < {MIN_CONFIDENCE_FOR_BERT}, will use unknown)")
                            bert_validation_reason = f"confidence {bert_result.confidence:.2f} < {MIN_CONFIDENCE_FOR_BERT}"
                    else:
                        logger.info(f"Document {document.id}: Keeping {best_method}; BERT is stored as semantic context only")
                    if best_method == "bert" and bert_used_model != self.model_name:
                        logger.info(f"Document {document.id}: Used BERT model '{bert_used_model}' (fallback from '{self.model_name or 'default'}')")
            else:
                # Log why BERT didn't return a result
                if self.model_name:
                    status = bert_svc.status(self.model_name)
                    if not status.get('model_exists'):
                        logger.warning(f"Document {document.id}: BERT model '{self.model_name}' not found, tried {len(models_to_try)} models")
                    else:
                        model_labels = status.get('labels', [])
                        missing_labels = [l for l in allowed_slugs if l not in model_labels]
                        if missing_labels:
                            logger.warning(f"Document {document.id}: BERT model '{self.model_name}' missing labels {missing_labels}, tried {len(models_to_try)} models")
                        else:
                            logger.info(f"Document {document.id}: BERT returned None for all {len(models_to_try)} models (score below threshold)")
                else:
                    logger.info(f"Document {document.id}: BERT returned None for all {len(models_to_try)} available models")
        except Exception as e:
            bert_error = str(e)
            logger.warning(f"BERT classifier error: {e}")

        # Step 1.9: Validate NB/BERT result against not: rules
        # Only use BERT if confidence is high enough (>= 0.5) when NB has no result
        # If confidence is too low, reject and use unknown instead
        nb_validation_reason = None
        if best_pred and best_method in ("naive_bayes", "bert"):
            # Check if the predicted type has not: rules that are violated
            # For Naive Bayes: check both not: and kw: rules
            # For BERT: only check not: rules (BERT is semantic and doesn't need keywords)
            check_keywords = (best_method == "naive_bayes")
            is_valid, reason = await self._validate_classification_against_not_rules(
                best_pred.label, sample_text, available_types, check_keywords=check_keywords
            )
            if not is_valid:
                # Reject if not: rules are violated (don't force a match)
                validation_reason = reason or "not: rule violation"
                if best_method == "naive_bayes":
                    nb_validation_reason = validation_reason
                else:
                    bert_validation_reason = validation_reason
                logger.warning(f"Document {document.id}: {best_method} result '{best_pred.label}' rejected: {validation_reason}")
                best_pred = None
                best_method = None
            elif best_method == "bert" and best_pred.confidence < 0.5:
                # Reject BERT if confidence is too low (even if not: rules pass)
                bert_validation_reason = f"confidence {best_pred.confidence:.2f} < 0.5"
                logger.info(f"Document {document.id}: BERT result '{best_pred.label}' rejected ({bert_validation_reason}, will use unknown)")
                best_pred = None
                best_method = None
        
        # Also validate NB and BERT separately to show why they were rejected
        if nb_pred and not best_pred:
            # NB had a result but was rejected - check why (check keywords for NB)
            is_valid, reason = await self._validate_classification_against_not_rules(
                nb_pred.label, sample_text, available_types, check_keywords=True
            )
            if not is_valid:
                nb_validation_reason = reason or "not: rule violation"
        
        if bert_pred and not best_pred:
            # BERT had a result but was rejected - check why (don't check keywords for BERT)
            is_valid, reason = await self._validate_classification_against_not_rules(
                bert_pred.label, sample_text, available_types, check_keywords=False
            )
            if not is_valid:
                bert_validation_reason = reason or "not: rule violation"

        if best_pred:
            # Build rationale with both scores if available
            if best_method == "deterministic_strong":
                # For strong deterministic matches, show matched keywords
                keywords_str = ", ".join(getattr(best_pred, 'matched_keywords', []) or [])
                rationale_parts = [f"STRONG keyword match (100% confidence) - matched keywords: {keywords_str}"]
            else:
                rationale_parts = [f"{best_method.upper()} classifier (p={best_pred.confidence:.2f})"]
            
            if nb_pred and bert_pred:
                rationale_parts.append(f"NB: {nb_pred.label} ({nb_pred.confidence:.2f}), BERT: {bert_pred.label} ({bert_pred.confidence:.2f})")
            elif nb_pred:
                rationale_parts.append(f"NB: {nb_pred.label} ({nb_pred.confidence:.2f})")
            elif bert_pred:
                rationale_parts.append(f"BERT: {bert_pred.label} ({bert_pred.confidence:.2f})")
            
            logger.info(f"Document {document.id}: Creating classification with doc_type_slug='{best_pred.label}' from {best_method}")
            classification = ClassificationResult(
                doc_type_slug=best_pred.label,
                confidence=float(best_pred.confidence),
                rationale=f"{' | '.join(rationale_parts)} (model={self.model_name or 'default'})",
                evidence=""
            )

            llm_dir = document_dir / "llm"
            llm_dir.mkdir(exist_ok=True)
            
            # Save both scores (always save NB and BERT attempts, even if they failed)
            classification_data = {
                "method": best_method,
                "doc_type_slug": best_pred.label,
                "confidence": float(best_pred.confidence),
            }
            
            # Add matched keywords for strong deterministic matches
            if best_method == "deterministic_strong" and hasattr(best_pred, 'matched_keywords'):
                classification_data["matched_keywords"] = getattr(best_pred, 'matched_keywords', []) or []
            
            # Add both classifier scores (or error info if they failed)
            if nb_pred:
                nb_data = {
                    "label": nb_pred.label,
                    "confidence": float(nb_pred.confidence),
                    "threshold": float(nb_threshold) if nb_threshold else None
                }
                if nb_validation_reason:
                    nb_data["status"] = "rejected"
                    nb_data["rejection_reason"] = nb_validation_reason
                # Add all scores if available
                if nb_all_scores:
                    nb_data["all_scores"] = {k: float(v) for k, v in nb_all_scores.items()}
                classification_data["naive_bayes"] = nb_data
            elif nb_below_threshold:
                classification_data["naive_bayes"] = {
                    "status": "below_threshold",
                    "label": nb_below_threshold.label,
                    "confidence": float(nb_below_threshold.confidence),
                    "threshold": float(nb_threshold) if nb_threshold else None,
                    "reason": f"Confidence {nb_below_threshold.confidence:.2f} < threshold {nb_threshold:.2f}"
                }
                # Add all scores if available
                if nb_all_scores:
                    classification_data["naive_bayes"]["all_scores"] = {k: float(v) for k, v in nb_all_scores.items()}
            elif nb_error:
                classification_data["naive_bayes"] = {
                    "error": nb_error,
                    "status": "failed"
                }
                # Add all scores if available (even if there was an error getting the best prediction)
                if nb_all_scores:
                    classification_data["naive_bayes"]["all_scores"] = {k: float(v) for k, v in nb_all_scores.items()}
            else:
                classification_data["naive_bayes"] = {
                    "status": "no_result",
                    "reason": "No model available or no prediction generated"
                }
                # Add all scores if available (even if no best prediction)
                if nb_all_scores:
                    classification_data["naive_bayes"]["all_scores"] = {k: float(v) for k, v in nb_all_scores.items()}
            
            if bert_pred:
                bert_data = {
                    "label": bert_pred.label,
                    "confidence": float(bert_pred.confidence)
                }
                if getattr(bert_pred, "all_scores", None):
                    all_scores = {k: float(v) for k, v in bert_pred.all_scores.items()}
                    sorted_scores = sorted(all_scores.values(), reverse=True)
                    bert_data["all_scores"] = all_scores
                    bert_data["margin"] = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
                if bert_used_model:
                    bert_data["model_used"] = bert_used_model
                if bert_validation_reason:
                    bert_data["status"] = "rejected"
                    bert_data["rejection_reason"] = bert_validation_reason
                classification_data["bert"] = bert_data
            elif bert_error:
                classification_data["bert"] = {
                    "error": bert_error,
                    "status": "failed",
                    "models_tried": bert_models_tried
                }
            else:
                classification_data["bert"] = {
                    "status": "no_result",
                    "reason": "Below threshold or no model available",
                    "models_tried": bert_models_tried
                }
            
            with open(llm_dir / "classification_local.json", "w", encoding="utf-8") as f:
                    json.dump(classification_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Document {document.id} classified as '{best_pred.label}' via {best_method} (p={best_pred.confidence:.2f})")
            return classification

        # Save classification attempts even if no best_pred was found
        if not best_pred:
            llm_dir = document_dir / "llm"
            llm_dir.mkdir(exist_ok=True)
            classification_data = {
                "method": None,
                "doc_type_slug": None,
                "confidence": 0.0,
            }
            
            # Add both classifier scores (or error info if they failed)
            if nb_pred:
                nb_data = {
                    "label": nb_pred.label,
                    "confidence": float(nb_pred.confidence),
                    "threshold": float(nb_threshold) if nb_threshold else None
                }
                if nb_validation_reason:
                    nb_data["status"] = "rejected"
                    nb_data["rejection_reason"] = nb_validation_reason
                # Add all scores if available
                if nb_all_scores:
                    nb_data["all_scores"] = {k: float(v) for k, v in nb_all_scores.items()}
                classification_data["naive_bayes"] = nb_data
            elif nb_below_threshold:
                classification_data["naive_bayes"] = {
                    "status": "below_threshold",
                    "label": nb_below_threshold.label,
                    "confidence": float(nb_below_threshold.confidence),
                    "threshold": float(nb_threshold) if nb_threshold else None,
                    "reason": f"Confidence {nb_below_threshold.confidence:.2f} < threshold {nb_threshold:.2f}"
                }
                # Add all scores if available
                if nb_all_scores:
                    classification_data["naive_bayes"]["all_scores"] = {k: float(v) for k, v in nb_all_scores.items()}
            elif nb_error:
                classification_data["naive_bayes"] = {
                    "error": nb_error,
                    "status": "failed"
                }
                # Add all scores if available (even if there was an error getting the best prediction)
                if nb_all_scores:
                    classification_data["naive_bayes"]["all_scores"] = {k: float(v) for k, v in nb_all_scores.items()}
            else:
                classification_data["naive_bayes"] = {
                    "status": "no_result",
                    "reason": "No model available or no prediction generated"
                }
                # Add all scores if available (even if no best prediction)
                if nb_all_scores:
                    classification_data["naive_bayes"]["all_scores"] = {k: float(v) for k, v in nb_all_scores.items()}
            
            if bert_pred:
                bert_data = {
                    "label": bert_pred.label,
                    "confidence": float(bert_pred.confidence)
                }
                if getattr(bert_pred, "all_scores", None):
                    all_scores = {k: float(v) for k, v in bert_pred.all_scores.items()}
                    sorted_scores = sorted(all_scores.values(), reverse=True)
                    bert_data["all_scores"] = all_scores
                    bert_data["margin"] = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
                if bert_used_model:
                    bert_data["model_used"] = bert_used_model
                if bert_validation_reason:
                    bert_data["status"] = "rejected"
                    bert_data["rejection_reason"] = bert_validation_reason
                classification_data["bert"] = bert_data
            elif bert_error:
                classification_data["bert"] = {
                    "error": bert_error,
                    "status": "failed",
                    "models_tried": bert_models_tried
                }
            else:
                classification_data["bert"] = {
                    "status": "no_result",
                    "reason": "Below threshold or no model available",
                    "models_tried": bert_models_tried
                }
            
            with open(llm_dir / "classification_local.json", "w", encoding="utf-8") as f:
                    json.dump(classification_data, f, indent=2, ensure_ascii=False)
        
        # Step 2: Deterministic classification (fallback when trained models don't match)
        # Only use deterministic if trained models had low confidence (< 0.7) or no result
        use_deterministic = not best_pred or (best_pred and best_pred.confidence < 0.7)
        
        deterministic_result = None
        if use_deterministic:
            deterministic_result = await self.classify_deterministic(sample_text, available_types)
            logger.info(f"Document {document.id} deterministic classification: {deterministic_result} (sample_text_length: {len(sample_text)}, trained_model_confidence: {best_pred.confidence if best_pred else 'none'})")
        else:
            logger.info(f"Document {document.id} skipping deterministic (trained model confidence {best_pred.confidence:.2f} >= 0.7)")

        if deterministic_result:
            # Deterministic match found (as fallback)
            # Find the matched hints for this document type
            matched_keywords = []
            matched_patterns = []
            text_lower = sample_text.lower()
            
            for slug, hints in available_types:
                if slug == deterministic_result and hints:
                    for hint_line in hints.strip().split('\n'):
                        hint_line = hint_line.strip()
                        if not hint_line:
                            continue
                        if hint_line.startswith('kw:'):
                            keyword = hint_line[3:].strip().lower()
                            if keyword in text_lower:
                                matched_keywords.append(keyword)
                        elif hint_line.startswith('re:'):
                            try:
                                pattern = hint_line[3:].strip()
                                if re.search(pattern, sample_text, re.IGNORECASE):
                                    matched_patterns.append(pattern)
                            except re.error:
                                pass
                        elif not hint_line.startswith('not:'):
                            # Legacy keyword
                            keyword = hint_line.lower()
                            if keyword in text_lower:
                                matched_keywords.append(keyword)
                    break
            
            rationale_parts = [f"Deterministic match (fallback)"]
            if matched_keywords:
                rationale_parts.append(f"keywords: {', '.join(matched_keywords[:5])}")
            if matched_patterns:
                rationale_parts.append(f"patterns: {len(matched_patterns)} matched")
            
            classification = ClassificationResult(
                doc_type_slug=deterministic_result,
                confidence=0.95,  # High confidence for deterministic matches
                rationale="; ".join(rationale_parts),
                evidence=""
            )

            # Save classification artifacts with matched keywords
            llm_dir = document_dir / "llm"
            llm_dir.mkdir(exist_ok=True)
            with open(llm_dir / "classification_deterministic.json", "w", encoding="utf-8") as f:
                json.dump({
                    "method": "deterministic",
                    "doc_type_slug": deterministic_result,
                    "confidence": classification.confidence,
                    "rationale": classification.rationale,
                    "matched_keywords": matched_keywords[:10],  # Limit to 10
                    "matched_patterns": matched_patterns[:5],   # Limit to 5
                }, f, indent=2)

            logger.info(f"Document {document.id} classified as '{deterministic_result}' via deterministic matching (fallback, keywords: {matched_keywords[:3]})")
            return classification

        # Step 3: LLM classification as last resort
        # Only use LLM if trained models failed
        # Get labels from trained models to limit LLM to only those types
        model_labels = set()
        try:
            from app.services.doc_type_classifier import classifier_service
            nb_svc = classifier_service()
            if self.model_name:
                nb_model = nb_svc._load_model_by_name(self.model_name)
            else:
                nb_model = nb_svc._load_classifier_if_changed()
            if nb_model and hasattr(nb_model, 'model') and nb_model.model.get("labels"):
                model_labels.update(nb_model.model.get("labels", []))
                logger.info(f"Document {document.id}: Found NB model with {len(model_labels)} labels: {sorted(model_labels)}")
        except Exception as e:
            logger.debug(f"Could not get NB model labels: {e}")
        
        try:
            from app.services.bert_classifier import bert_classifier_service
            bert_svc = bert_classifier_service()
            if self.model_name:
                bert_status = bert_svc.status(self.model_name)
                if bert_status.get("model_exists") and bert_status.get("labels"):
                    model_labels.update(bert_status.get("labels", []))
                    logger.info(f"Document {document.id}: Found BERT model with labels: {sorted(bert_status.get('labels', []))}")
        except Exception as e:
            logger.debug(f"Could not get BERT model labels: {e}")
        
        # If we have trained models, only classify among their labels
        # Otherwise, use all available types
        if model_labels:
            # Filter to only types that are in the trained model AND in allowed_slugs
            llm_types = [slug for slug in allowed_slugs if slug in model_labels]
            if not llm_types:
                # No overlap between model labels and allowed slugs - use all allowed slugs
                llm_types = allowed_slugs
                logger.warning(f"Document {document.id}: No overlap between model labels {sorted(model_labels)} and allowed slugs {allowed_slugs}, using all allowed types")
            else:
                logger.info(f"Document {document.id} falling back to LLM classification (limited to trained model types: {llm_types})")
        else:
            # No trained model, use all available types
            llm_types = allowed_slugs
            logger.info(f"Document {document.id} falling back to LLM classification (no trained model, using all types)")
        
        available_slugs_with_unknown = llm_types + ["unknown"]
        llm_result = await self._llm_classify_document(document_dir, sample_text, available_slugs_with_unknown)

        logger.info(f"Document {document.id} classified as '{llm_result.doc_type_slug}' via LLM (confidence: {llm_result.confidence})")
        return llm_result

    def _prepare_text_sample(self, text: str, max_chars: int = 6000, skip_markers: List[Tuple[str, bool]] = None) -> TextPrepareResult:
        # Truncate text for regex matching to prevent CPU issues
        text = text[:200_000]
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
            
            # Regex compile cache
            _skip_regex_cache: dict[str, re.Pattern] = {}
            for pattern, is_regex in skip_markers:
                try:
                    if is_regex:
                        if pattern not in _skip_regex_cache:
                            _skip_regex_cache[pattern] = re.compile(pattern, re.IGNORECASE)
                        compiled = _skip_regex_cache[pattern]
                        match = compiled.search(normalized)
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
                    logger.debug(f"Invalid skip marker regex '{pattern}': {e}")
            
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

    def _normalize_for_search(self, value: Any) -> str:
        """Normalize text for generic label/key based chunk selection."""
        normalized = str(value or "").replace("m²", "m2").replace("M²", "m2")
        normalized = unicodedata.normalize("NFKD", normalized)
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        normalized = normalized.lower()
        normalized = re.sub(r"[_\-]+", " ", normalized)
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    def _search_tokens(self, value: Any) -> List[str]:
        """Tokenize normalized search text into useful terms."""
        return [token for token in self._normalize_for_search(value).split() if len(token) >= 2]

    def _build_field_search_terms(self, fields: List[Tuple]) -> Dict[str, Dict[str, Any]]:
        """Build generic search terms from configured field keys and labels."""
        term_config: Dict[str, Dict[str, Any]] = {}
        for key, label, field_type, is_required, enum_values, regex in fields:
            field_key = self._canonical_field_key(key)
            key_term = self._normalize_for_search(field_key)
            label_term = self._normalize_for_search(label)
            label_without_units = self._normalize_for_search(re.sub(r"\([^)]*\)", " ", str(label or "")))
            terms = {term for term in [key_term, label_term, label_without_units] if term}

            combined = " ".join(terms)
            if "m2" in combined or "oppervlakte" in combined or "area" in combined:
                terms.update({
                    "m2",
                    "m 2",
                    "oppervlakte",
                    "vloeroppervlak",
                    "gebruiksoppervlakte",
                    "floor area",
                    "area",
                })
            if "bouwjaar" in combined or "construction year" in combined or "built year" in combined:
                terms.update({"bouw jaar", "bouwjaar", "construction year", "built", "built year"})

            term_config[field_key] = {
                "key": key_term,
                "label": label_term,
                "raw_labels": [str(label or ""), str(key or ""), str(label_without_units or "")],
                "terms": sorted(terms, key=lambda item: (-len(item), item)),
                "field_type": field_type,
                "required": bool(is_required),
                "enum_values": enum_values,
                "regex": regex,
            }

        return term_config

    def _build_text_chunks_from_pages(
        self,
        pages: List[Dict[str, Any]],
        fallback_text: str,
        chunk_size: int,
        overlap: int,
    ) -> List[Dict[str, Any]]:
        """Build ordered page/chunk records for relevance scoring."""
        chunks: List[Dict[str, Any]] = []
        if pages:
            for page_idx, page in enumerate(pages):
                page_text = str(page.get("text") or "")
                if not page_text.strip():
                    continue
                page_chunks = self._split_text_into_chunks(page_text, chunk_size, overlap)
                for part_idx, chunk_text in enumerate(page_chunks):
                    chunks.append({
                        "chunk_num": len(chunks) + 1,
                        "page": page_idx,
                        "part": part_idx,
                        "text": chunk_text,
                    })

        if chunks:
            return chunks

        return [
            {"chunk_num": index + 1, "page": index, "part": 0, "text": chunk}
            for index, chunk in enumerate(self._split_text_into_chunks(fallback_text, chunk_size, overlap))
        ]

    def _words_within_window(self, text_tokens: List[str], label_tokens: List[str], window: int = 10) -> bool:
        """Return whether all label words occur close together in a token stream."""
        if not label_tokens or len(label_tokens) < 2:
            return False

        positions: List[int] = []
        for label_token in label_tokens:
            try:
                positions.append(text_tokens.index(label_token))
            except ValueError:
                return False

        return max(positions) - min(positions) <= window

    def _find_field_label_positions(self, text_value: str, field_terms: Dict[str, Dict[str, Any]]) -> Dict[str, List[int]]:
        """Find raw text positions for configured field labels/keys."""
        positions_by_field: Dict[str, List[int]] = {}
        text_lower = text_value.lower()
        for field_key, config in field_terms.items():
            positions: List[int] = []
            labels = list(config.get("raw_labels", [])) + list(config.get("terms", []))
            for label in labels:
                label_text = str(label or "").strip()
                if len(self._normalize_for_search(label_text)) < 2:
                    continue
                pattern = r"\b" + r"\s+".join(
                    re.escape(part)
                    for part in re.split(r"[\s_\-]+", label_text)
                    if part
                ) + r"\b"
                try:
                    positions.extend(match.start() for match in re.finditer(pattern, text_value, re.IGNORECASE))
                except re.error:
                    label_pos = text_lower.find(label_text.lower())
                    if label_pos >= 0:
                        positions.append(label_pos)

            if positions:
                positions_by_field[field_key] = sorted(set(positions))

        return positions_by_field

    def _has_concrete_value_near_label(
        self,
        text_value: str,
        position: int,
        field_config: Dict[str, Any],
        before: int = 40,
        after: int = 150,
    ) -> bool:
        """Detect a concrete value near a label using field-type validation."""
        window = text_value[max(0, position - before):position + after]
        kind = self._field_kind(field_config)

        if kind == "enum" and field_config.get("enum_values"):
            window_norm = self._normalize_for_search(window)
            return any(
                self._normalize_for_search(enum_value) in window_norm
                for enum_value in field_config["enum_values"]
            )

        if kind == "boolean":
            return bool(re.search(r"\b(true|false|yes|no|ja|nee)\b", window, re.IGNORECASE))

        if kind == "string":
            return bool(re.search(r"[:\-]\s*[^\n:;\-]{1,80}", window))

        for number_match in re.finditer(r"\b\d{1,4}(?:[.,]\d{1,3})?\b", window):
            valid, _, _ = self._validate_candidate_value_for_field(
                number_match.group(0),
                field_config,
                evidence=window,
            )
            if valid:
                return True

        return False

    def _concrete_value_candidates_near_field(
        self,
        text_value: str,
        position: int,
        field_config: Dict[str, Any],
        before: int = 120,
        after: int = 300,
    ) -> List[Dict[str, Any]]:
        """Find generic concrete value candidates around a field label/key."""
        window_start = max(0, position - before)
        window_end = min(len(text_value), position + after)
        window = text_value[window_start:window_end]
        candidates: List[Dict[str, Any]] = []
        kind = self._field_kind(field_config)

        if kind == "enum" and field_config.get("enum_values"):
            window_norm = self._normalize_for_search(window)
            for enum_value in field_config["enum_values"]:
                enum_norm = self._normalize_for_search(enum_value)
                if enum_norm and enum_norm in window_norm:
                    candidates.append({"value": str(enum_value), "kind": "enum", "window": window})
            return candidates

        if kind == "boolean":
            for match in re.finditer(r"\b(true|false|yes|no|ja|nee)\b", window, re.IGNORECASE):
                candidates.append({"value": match.group(0), "kind": "boolean", "window": window})
            return candidates

        if kind == "string":
            match = re.search(r"[:\-]\s*([^\n:;\-]{1,80})", window)
            if match:
                value = match.group(1).strip()
                if value and len(value.split()) <= 8:
                    candidates.append({"value": value, "kind": "string", "window": window})
            return candidates

        for number_match in re.finditer(r"\b\d{1,5}(?:[.,]\d{1,3})?\b", window):
            value = number_match.group(0)
            valid, _, normalized = self._validate_candidate_value_for_field(
                value,
                field_config,
                evidence=window,
            )
            if valid:
                candidates.append({"value": normalized, "kind": kind, "window": window})

        return candidates

    def _is_table_like_text(self, text_value: str) -> bool:
        """Detect generic table-like structure without document-specific terms."""
        lines = [line for line in text_value.splitlines() if line.strip()]
        if len(lines) < 2:
            return False
        tableish_lines = 0
        for line in lines:
            has_columns = "\t" in line or "|" in line or bool(re.search(r"\S+\s{2,}\S+", line))
            has_multiple_tokens = len(line.split()) >= 3
            has_number = bool(re.search(r"\b\d+(?:[.,]\d+)?\b", line))
            if has_columns or (has_multiple_tokens and has_number):
                tableish_lines += 1
        return tableish_lines >= 2

    def _score_metadata_chunk(
        self,
        chunk: Dict[str, Any],
        field_terms: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Score one chunk using configured labels/keys and optional regex signals."""
        raw_text = str(chunk.get("text") or "")
        normalized_text = self._normalize_for_search(raw_text)
        text_tokens = normalized_text.split()
        field_matches: Dict[str, Dict[str, Any]] = {}
        score = 0
        matched_field_count = 0
        numeric_count = len(re.findall(r"\b\d+(?:[.,]\d+)?\b", raw_text))
        positions_by_field = self._find_field_label_positions(raw_text, field_terms)

        for field_key, config in field_terms.items():
            field_score = 0
            label_score = 0
            value_score = 0
            reasons: List[str] = []
            matched_terms: List[str] = []
            label_matched = False
            regex_only = False
            match_positions: List[int] = positions_by_field.get(field_key, [])
            concrete_candidates: List[Dict[str, Any]] = []

            label = config.get("label") or ""
            if label and label in normalized_text:
                label_score += 100
                label_matched = True
                matched_terms.append(label)
                reasons.append("exact_label:+100")
            else:
                best_term = next((term for term in config["terms"] if term and term in normalized_text), "")
                if best_term:
                    label_score += 30
                    label_matched = True
                    matched_terms.append(best_term)
                    reasons.append("partial_label:+30")

                label_tokens = self._search_tokens(label)
                if self._words_within_window(text_tokens, label_tokens):
                    label_score += 70
                    label_matched = True
                    reasons.append("label_words_near:+70")

            key = config.get("key") or ""
            key_tokens = self._search_tokens(key)
            if key and (key in normalized_text or self._words_within_window(text_tokens, key_tokens)):
                label_score += 50
                matched_terms.append(key)
                reasons.append("key_match:+50")

            regex = config.get("regex")
            if regex:
                try:
                    if re.search(str(regex), raw_text, re.IGNORECASE):
                        regex_score = 5
                        value_score += regex_score
                        regex_only = not label_matched and key not in normalized_text
                        reasons.append(f"regex_signal:+{regex_score}")
                except re.error:
                    logger.warning(f"Invalid regex for chunk scoring: {regex}")

            if label_matched:
                for position in match_positions or [0]:
                    concrete_candidates.extend(self._concrete_value_candidates_near_field(raw_text, position, config))
                if concrete_candidates:
                    value_score += 120
                    reasons.append("label_concrete_value_near:+120")
                else:
                    label_score += 20
                    reasons.append("label_without_concrete_value:+20")

            field_score = label_score + value_score

            if field_score > 0:
                matched_field_count += 1
                field_matches[field_key] = {
                    "score": field_score,
                    "label_score": label_score,
                    "value_score": value_score,
                    "matched_terms": sorted(set(matched_terms)),
                    "reasons": reasons,
                    "has_label_context": label_matched,
                    "regex_only": regex_only,
                    "match_positions": match_positions,
                    "concrete_value_candidates": concrete_candidates[:10],
                }
                score += field_score

        penalties: List[str] = []
        usable_length = len(normalized_text)
        if usable_length < 50:
            score -= 20
            penalties.append("short_text:-20")
        generic_section_terms = [
            "inhoudsopgave",
            "table of contents",
            "disclaimer",
            "definities",
            "definitions",
            "bijlage index",
            "appendix index",
        ]
        if any(term in normalized_text for term in generic_section_terms):
            score -= 50
            penalties.append("generic_section:-50")
        if numeric_count >= 12 and matched_field_count <= 1:
            score -= 40
            penalties.append("many_numbers_few_labels:-40")
        if numeric_count >= 20 and matched_field_count <= 2:
            score -= 40
            penalties.append("repeated_records_few_requested_labels:-40")
        explanation_like = bool(re.search(r"\b(is|betekent|wordt|hiermee|uitleg|explanation|means|defined|definition)\b", normalized_text))
        if explanation_like and numeric_count == 0 and field_matches:
            score -= 60
            penalties.append("explanation_without_concrete_value:-60")
        value_rich_matches = sum(
            1
            for match in field_matches.values()
            if match.get("value_score", 0) > 0
        )
        if len(field_matches) >= 2 and value_rich_matches >= 2:
            score += 150
            penalties.append("multiple_labels_values_cluster:+150")
        weak_match_count = sum(
            1
            for match in field_matches.values()
            if not match.get("has_label_context") or match.get("score", 0) < 70
        )
        if usable_length > 2500 and weak_match_count == 1 and len(field_matches) == 1:
            score -= 30
            penalties.append("long_one_weak_match:-30")
        if matched_field_count > 1:
            score += 20
            penalties.append("multiple_fields_bonus:+20")

        return {
            "chunk_num": chunk["chunk_num"],
            "page": chunk.get("page"),
            "part": chunk.get("part", 0),
            "score": score,
            "field_matches": field_matches,
            "penalties": penalties,
            "text_length": len(raw_text),
        }

    def _should_include_neighbor_chunk(self, chunk: Dict[str, Any], scored: Dict[str, Any]) -> bool:
        """Only include adjacent context when the selected chunk likely cuts through content."""
        text_value = str(chunk.get("text") or "").rstrip()
        if not text_value:
            return False

        ends_mid_sentence = text_value[-1] not in ".!?:;\n"
        table_like_end = bool(re.search(r"(\s{2,}|\t|\|)\S*$", text_value[-240:]))
        text_len = max(len(text_value), 1)
        near_edge = any(
            position <= 800 or text_len - position <= 150
            for match in scored.get("field_matches", {}).values()
            for position in match.get("match_positions", [])
        )
        return ends_mid_sentence or table_like_end or near_edge

    def _select_relevant_metadata_chunks(
        self,
        chunks: List[Dict[str, Any]],
        fields: List[Tuple],
        top_n: int = 0,
        per_field_top_n: int = 2,
        threshold: int = 70,
        max_context_chars: int = 3500,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Select relevant chunks before metadata LLM extraction."""
        field_terms = self._build_field_search_terms(fields)
        scored_chunks = [self._score_metadata_chunk(chunk, field_terms) for chunk in chunks]
        selected_nums: Set[int] = set()
        selected_reasons: Dict[int, List[str]] = {}
        rejected_reasons: Dict[int, List[str]] = {}

        top_chunks = sorted(scored_chunks, key=lambda item: (-item["score"], item["chunk_num"]))[:top_n]
        for scored in top_chunks:
            if scored["score"] >= threshold:
                selected_nums.add(scored["chunk_num"])
                selected_reasons.setdefault(scored["chunk_num"], []).append("top_overall")

        for field_key in field_terms.keys():
            field_scored = [
                scored for scored in scored_chunks
                if field_key in scored["field_matches"]
                and scored["field_matches"][field_key]["score"] >= threshold
                and scored["field_matches"][field_key].get("has_label_context")
                and scored["field_matches"][field_key].get("label_score", 0) > 0
                and scored["field_matches"][field_key].get("value_score", 0) > 0
            ]
            if field_scored:
                for scored in sorted(
                    field_scored,
                    key=lambda item: (-item["field_matches"][field_key]["score"], item["chunk_num"]),
                )[:per_field_top_n]:
                    selected_nums.add(scored["chunk_num"])
                    selected_reasons.setdefault(scored["chunk_num"], []).append(f"best_for_field:{field_key}")
                continue

            regex_fallback = [
                scored for scored in scored_chunks
                if field_key in scored["field_matches"]
                and scored["field_matches"][field_key].get("regex_only")
                and not any(
                    other.get("field_matches", {}).get(field_key, {}).get("value_score", 0) > 0
                    and other.get("field_matches", {}).get(field_key, {}).get("label_score", 0) > 0
                    for other in scored_chunks
                )
            ]
            if regex_fallback:
                best = max(regex_fallback, key=lambda item: (item["field_matches"][field_key]["score"], -item["chunk_num"]))
                selected_nums.add(best["chunk_num"])
                selected_reasons.setdefault(best["chunk_num"], []).append(f"regex_fallback_for_field:{field_key}")

        if not selected_nums and scored_chunks:
            concrete_scored = [
                scored for scored in scored_chunks
                if any(
                    match.get("label_score", 0) > 0 and match.get("value_score", 0) > 0
                    for match in scored.get("field_matches", {}).values()
                )
            ]
            best = max(concrete_scored or scored_chunks, key=lambda item: (item["score"], -item["chunk_num"]))
            selected_nums.add(best["chunk_num"])
            selected_reasons.setdefault(best["chunk_num"], []).append(
                "fallback_best_concrete_chunk" if concrete_scored else "fallback_no_concrete_chunk"
            )

        chunk_by_num = {chunk["chunk_num"]: chunk for chunk in chunks}
        scored_by_num = {scored["chunk_num"]: scored for scored in scored_chunks}
        total_chars = sum(len(chunk_by_num[num]["text"]) for num in selected_nums if num in chunk_by_num)
        for num in sorted(list(selected_nums)):
            scored = scored_by_num.get(num, {})
            if not self._should_include_neighbor_chunk(chunk_by_num[num], scored):
                continue
            for neighbor_num in (num - 1, num + 1):
                neighbor = chunk_by_num.get(neighbor_num)
                if not neighbor or neighbor_num in selected_nums:
                    continue
                neighbor_score = scored_by_num.get(neighbor_num, {})
                neighbor_matches = neighbor_score.get("field_matches", {})
                if neighbor_matches and all(
                    match.get("label_score", 0) > 0 and match.get("value_score", 0) <= 0
                    for match in neighbor_matches.values()
                ):
                    rejected_reasons.setdefault(neighbor_num, []).append(f"neighbor_explanation_only_for:{num}")
                    continue
                if total_chars + len(neighbor["text"]) > max_context_chars:
                    rejected_reasons.setdefault(neighbor_num, []).append(f"neighbor_budget_blocked_for:{num}")
                    continue
                selected_nums.add(neighbor_num)
                selected_reasons.setdefault(neighbor_num, []).append(f"context_neighbor_of:{num}")
                total_chars += len(neighbor["text"])

        selected_priority = {
            scored["chunk_num"]: scored["score"]
            for scored in scored_chunks
        }
        budgeted_nums: Set[int] = set()
        budgeted_total = 0
        for num in sorted(
            selected_nums,
            key=lambda item: (-selected_priority.get(item, 0), item),
        ):
            chunk = chunk_by_num.get(num)
            if not chunk:
                continue
            chunk_length = len(chunk["text"])
            if budgeted_nums and budgeted_total + chunk_length > max_context_chars:
                selected_reasons.setdefault(num, []).append("dropped_prompt_budget")
                rejected_reasons.setdefault(num, []).append("dropped_prompt_budget")
                continue
            budgeted_nums.add(num)
            budgeted_total += chunk_length

        selected_nums = budgeted_nums or selected_nums
        selected_chunks = [
            {
                **chunk,
                "matched_fields": scored_by_num.get(chunk["chunk_num"], {}).get("field_matches", {}),
                "selection_reasons": selected_reasons.get(chunk["chunk_num"], []),
            }
            for chunk in chunks
            if chunk["chunk_num"] in selected_nums
        ]
        debug = {
            "top_n": top_n,
            "per_field_top_n": per_field_top_n,
            "threshold": threshold,
            "max_context_chars": max_context_chars,
            "original_chunk_count": len(chunks),
            "selected_chunk_count": len(selected_chunks),
            "selected_text_chars": sum(len(chunk["text"]) for chunk in selected_chunks),
            "selected_chunks": [
                {
                    "chunk_num": chunk["chunk_num"],
                    "page": chunk.get("page"),
                    "part": chunk.get("part", 0),
                    "reasons": selected_reasons.get(chunk["chunk_num"], []),
                    "score": next((scored["score"] for scored in scored_chunks if scored["chunk_num"] == chunk["chunk_num"]), 0),
                    "label_score": sum(
                        match.get("label_score", 0)
                        for match in next((scored["field_matches"] for scored in scored_chunks if scored["chunk_num"] == chunk["chunk_num"]), {}).values()
                    ),
                    "value_score": sum(
                        match.get("value_score", 0)
                        for match in next((scored["field_matches"] for scored in scored_chunks if scored["chunk_num"] == chunk["chunk_num"]), {}).values()
                    ),
                    "field_matches": next((scored["field_matches"] for scored in scored_chunks if scored["chunk_num"] == chunk["chunk_num"]), {}),
                }
                for chunk in selected_chunks
            ],
            "skipped_chunks": [
                {
                    "chunk_num": scored["chunk_num"],
                    "page": scored.get("page"),
                    "score": scored["score"],
                    "reason": rejected_reasons.get(scored["chunk_num"], ["below_threshold_or_no_field_label_match"])[0],
                    "rejected_as_explanation_only": any(
                        match.get("label_score", 0) > 0 and match.get("value_score", 0) <= 0
                        for match in scored.get("field_matches", {}).values()
                    ),
                }
                for scored in scored_chunks
                if scored["chunk_num"] not in selected_nums
            ],
            "scores": scored_chunks,
        }
        return selected_chunks, debug

    def _deterministic_candidate_extraction(
        self,
        chunks: List[Dict[str, Any]],
        fields: List[Tuple],
    ) -> Dict[str, Any]:
        """Extract obvious same-line label/value candidates before using the LLM."""
        field_terms = self._build_field_search_terms(fields)
        candidates: Dict[str, List[Dict[str, Any]]] = {
            self._canonical_field_key(key): []
            for key, _, _, _, _, _ in fields
        }

        for chunk in chunks:
            chunk_index = max(int(chunk["chunk_num"]) - 1, 0)
            for line in str(chunk.get("text") or "").splitlines():
                line_text = line.strip()
                if len(line_text) < 3:
                    continue
                for field_key, config in field_terms.items():
                    raw_labels = [
                        label for label in config.get("raw_labels", [])
                        if label and len(self._normalize_for_search(label)) >= 2
                    ]
                    value = None
                    for raw_label in raw_labels:
                        label_pattern = r"\s+".join(
                            re.escape(part)
                            for part in str(raw_label).replace("_", " ").replace("-", " ").split()
                            if part
                        )
                        if not label_pattern:
                            continue
                        match = re.search(rf"\b{label_pattern}\b\s*[:\-]\s*(.+)$", line_text, re.IGNORECASE)
                        if match:
                            value = match.group(1).strip()
                            break
                    if value is None:
                        continue

                    other_label_patterns = []
                    for other_key, other_config in field_terms.items():
                        if other_key == field_key:
                            continue
                        for other_label in other_config.get("raw_labels", []):
                            normalized_other = self._normalize_for_search(other_label)
                            if len(normalized_other) < 2:
                                continue
                            other_label_patterns.append(r"\s+".join(
                                re.escape(part)
                                for part in str(other_label).replace("_", " ").replace("-", " ").split()
                                if part
                            ))
                    if other_label_patterns:
                        next_label = re.search(rf"\s+(?:{'|'.join(other_label_patterns)})\s*[:\-]", value, re.IGNORECASE)
                        if next_label:
                            value = value[:next_label.start()].strip()

                    value = value.strip(" :;-")
                    if not value or len(value) > 60 or len(value.split()) > 6:
                        continue

                    valid, _, normalized_value = self._validate_candidate_value_for_field(
                        value,
                        config,
                        evidence=line_text,
                    )
                    if not valid:
                        continue
                    regex = config.get("regex")
                    if regex:
                        try:
                            if re.search(str(regex), str(normalized_value), re.IGNORECASE) is None:
                                continue
                        except re.error:
                            logger.warning(f"Invalid regex for deterministic candidate validation: {regex}")
                    candidates[field_key].append({
                        "value": value,
                        "normalized_value": normalized_value,
                        "unit": None,
                        "evidence": line_text,
                        "chunk_index": chunk_index,
                        "confidence": 95,
                        "evidence_type": "exact_label",
                        "record_role": "unknown",
                    })

        return {"candidates": candidates}

    def _clip_chunk_text_for_llm(
        self,
        chunk: Dict[str, Any],
        fields: List[Tuple],
        max_chars: int = 2200,
    ) -> str:
        """Clip selected text around the strongest generic data anchor, not a loose label."""
        text = str(chunk.get("text") or "").strip()
        if len(text) <= min(3500, max_chars):
            return text

        field_terms = self._build_field_search_terms(fields)
        match_positions: List[int] = []
        for config in field_terms.values():
            raw_labels = [
                label for label in config.get("raw_labels", [])
                if label and len(self._normalize_for_search(label)) >= 2
            ]
            for raw_label in raw_labels:
                label_pattern = r"\s+".join(
                    re.escape(part)
                    for part in str(raw_label).replace("_", " ").replace("-", " ").split()
                    if part
                )
                if not label_pattern:
                    continue
                for match in re.finditer(rf"\b{label_pattern}\b", text, re.IGNORECASE):
                    match_positions.append(match.start())

        if not match_positions:
            return text[:max_chars].rstrip()

        windows = []
        for position in sorted(set(match_positions)):
            windows.append((max(0, position - 1500), min(len(text), position + 800)))

        merged_windows: List[Tuple[int, int]] = []
        for start, end in windows:
            if not merged_windows or start > merged_windows[-1][1]:
                merged_windows.append((start, end))
            else:
                previous_start, previous_end = merged_windows[-1]
                merged_windows[-1] = (previous_start, max(previous_end, end))

        def score_window(window: Tuple[int, int]) -> Tuple[int, int]:
            start, end = window
            window_text = text[start:end]
            normalized_window = self._normalize_for_search(window_text)
            label_count = 0
            concrete_count = 0
            for config in field_terms.values():
                has_label = any(term and term in normalized_window for term in config["terms"])
                if not has_label:
                    continue
                label_count += 1
                numbers = re.findall(r"\b\d+(?:[.,]\d+)?\b", window_text)
                if any(
                    self._validate_candidate_value_for_field(number, config, evidence=window_text)[0]
                    for number in numbers[:20]
                ):
                    concrete_count += 1

            table_like_lines = sum(
                1
                for line in window_text.splitlines()
                if len(re.findall(r"\S+", line)) >= 3 and len(re.findall(r"\d+", line)) >= 1
            )
            explanation_penalty = 2 if concrete_count == 0 and re.search(
                r"\b(is|betekent|wordt|hiermee|uitleg|explanation|means|defined|definition)\b",
                normalized_window,
            ) else 0
            score = (label_count * 10) + (concrete_count * 25) + (table_like_lines * 8) - (explanation_penalty * 20)
            return score, -start

        best_start, best_end = max(merged_windows, key=score_window)
        if best_end - best_start > max_chars:
            best_score_position = max(
                [position for position in match_positions if best_start <= position <= best_end],
                key=lambda position: (
                    score_window((max(best_start, position - 1500), min(best_end, position + 800)))[0],
                    -position,
                ),
            )
            pre_context = min(1500, max(200, max_chars // 2))
            best_start = max(0, best_score_position - pre_context)
            best_end = min(len(text), best_start + max_chars)
            if best_end - best_start < max_chars:
                best_start = max(0, best_end - max_chars)

        return text[best_start:best_end].strip()

    def _build_selected_chunks_text(
        self,
        selected_chunks: List[Dict[str, Any]],
        fields: List[Tuple],
        max_chars: int = 4200,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Build compact selected chunk text for a single LLM request."""
        parts: List[str] = []
        debug: List[Dict[str, Any]] = []
        used_chars = 0

        per_chunk_chars = min(
            max(600, max_chars - 300),
            max(1800, min(3000, max_chars // max(len(selected_chunks), 1))),
        )
        for chunk in selected_chunks:
            matched_fields = chunk.get("matched_fields", {})
            match_summary = []
            if isinstance(matched_fields, dict):
                for field_key, match in matched_fields.items():
                    terms = ", ".join(match.get("matched_terms", [])) if isinstance(match, dict) else ""
                    match_summary.append(f"{field_key}: {terms}".strip())
            header = (
                f"<!-- SYSTEM CHUNK METADATA, NOT DOCUMENT CONTENT: "
                f"chunk_num={chunk['chunk_num']}; page={int(chunk.get('page', 0)) + 1}; "
                f"matched_fields={' | '.join(match_summary) if match_summary else 'unknown'} -->\n"
                "DOCUMENT CHUNK TEXT:\n"
            )
            raw_text = str(chunk.get("text") or "")
            if len(raw_text) <= 3500 and used_chars + len(header) + len(raw_text) + 2 <= max_chars:
                clipped_text = raw_text.strip()
            else:
                clipped_text = self._clip_chunk_text_for_llm(chunk, fields, per_chunk_chars)
            part = f"{header}{clipped_text}"
            if parts and used_chars + len(part) + 2 > max_chars:
                debug.append({
                    "chunk_num": chunk["chunk_num"],
                    "original_chars": len(str(chunk.get("text") or "")),
                    "included_chars": 0,
                    "reason": "dropped_prompt_budget",
                })
                continue

            parts.append(part)
            used_chars += len(part) + 2
            debug.append({
                "chunk_num": chunk["chunk_num"],
                "original_chars": len(str(chunk.get("text") or "")),
                "included_chars": len(clipped_text),
                "reason": "included",
            })

        return "\n\n".join(parts), debug

    def _all_required_fields_resolved(self, resolved: Optional[Dict[str, Any]], fields: List[Tuple]) -> bool:
        """Return true when deterministic extraction found all required fields."""
        if not resolved or not isinstance(resolved.get("data"), dict):
            return False

        required_keys = [self._canonical_field_key(key) for key, _, _, is_required, _, _ in fields if is_required]
        if not required_keys:
            return False

        return all(not self._candidate_value_is_empty(resolved["data"].get(key)) for key in required_keys)

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

        with open(llm_dir / "classification_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        with open(llm_dir / "classification_schema.json", "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)

        curl_command = None
        response_text = None
        result = None
        duration = None
        try:
            logger.info(f"Starting LLM classification request for document {document_dir.parent.name}")
            result, response_text, curl_command, duration = await self.llm.generate_json_with_raw(prompt, schema)
            logger.info(f"LLM classification completed for document {document_dir.parent.name} in {duration:.2f}s")

            # Save response, curl command, and timing immediately after successful request
            with open(llm_dir / "classification_response.txt", "w", encoding="utf-8") as f:
                f.write(response_text)
            
            if curl_command:
                with open(llm_dir / "classification_curl.txt", "w", encoding="utf-8") as f:
                    f.write(curl_command)
            
            # Save timing metadata
            if duration is not None:
                with open(llm_dir / "classification_timing.json", "w", encoding="utf-8") as f:
                    json.dump({"duration_seconds": duration, "provider": self.llm.provider, "model": self.llm.model}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"LLM classification failed for document {document_dir.parent.name}: {e}")
            error_msg = str(e)
            with open(llm_dir / "classification_error.txt", "w", encoding="utf-8") as f:
                f.write(error_msg)
            
            # Save curl command even if there was an error (if we got that far)
            if curl_command:
                with open(llm_dir / "classification_curl.txt", "w", encoding="utf-8") as f:
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

            with open(llm_dir / "classification_result.json", "w", encoding="utf-8") as f:
                json.dump(validated_result, f, indent=2, ensure_ascii=False)

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
        logger.info(f"Starting metadata extraction for document {document.id}, doc_type: {classification.doc_type_slug}")
        
        # Get document type fields and preamble
        # ORDER BY id to ensure consistent ordering and get latest fields
        result = await self.db.execute(
            text("SELECT key, label, field_type, required, enum_values, regex FROM document_type_fields WHERE document_type_id = (SELECT id FROM document_types WHERE slug = :slug) ORDER BY id"),
            {"slug": classification.doc_type_slug}
        )
        fields = self._normalize_extraction_fields(result.fetchall())
        logger.info(f"Found {len(fields)} fields for document type {classification.doc_type_slug}")

        # Get preamble
        preamble_result = await self.db.execute(
            text("SELECT extraction_prompt_preamble FROM document_types WHERE slug = :slug"),
            {"slug": classification.doc_type_slug}
        )
        preamble_row = preamble_result.fetchone()
        preamble = preamble_row[0] if preamble_row else ""

        if not fields:
            logger.info(f"Skipping LLM extraction for document {document.id}: No fields configured for type '{classification.doc_type_slug}'")
            llm_dir = document_dir / "llm"
            llm_dir.mkdir(exist_ok=True)
            with open(llm_dir / "extraction_skipped.json", "w", encoding="utf-8") as f:
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
        chunk_schema = self._build_candidate_extraction_schema(fields)
        
        llm_dir = document_dir / "llm"
        llm_dir.mkdir(exist_ok=True)
        
        with open(llm_dir / "extraction_schema.json", "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        
        # Split into chunks if text is too large
        # For 4K context models with long preambles:
        # - Model context: 4096 tokens
        # - Response output: ~800 tokens needed
        # - Prompt overhead (instructions, preamble, fields): ~1500 tokens
        # - Available for document text: ~1800 tokens (~5000 chars)
        # Use 2500 chars to be safe with long preambles
        CHUNK_SIZE = 2500
        OVERLAP = 300  # Overlap between chunks to avoid missing data at boundaries
        
        curl_command = None
        response_text = None
        result = None
        total_chunks = None  # Track if we're using chunks
        
        if len(filtered_text) > CHUNK_SIZE:
            logger.info(f"Document text is large ({len(filtered_text)} chars), splitting into chunks for extraction")
            chunk_records = self._build_text_chunks_from_pages(ocr_result.pages, filtered_text, CHUNK_SIZE, OVERLAP)
            chunks = [chunk["text"] for chunk in chunk_records]
            total_chunks = len(chunks)
            selected_chunks, selection_debug = self._select_relevant_metadata_chunks(chunk_records, fields)
            processing_mode = "SELECTED_SINGLE_CALL"
            logger.info(
                "Split into %s chunks - selected %s relevant chunks for one LLM call",
                total_chunks,
                len(selected_chunks),
            )
            
            # Metadata extraction runs from 60-85%, so we use 60-80% for chunks, 80-85% for merging
            EXTRACTION_START = 60
            EXTRACTION_END = 80
            MERGE_START = 80
            MERGE_END = 85
            
            # Update progress to show multi-chunk extraction starting
            await self._update_progress(
                document.id,
                EXTRACTION_START,
                f"extracting_metadata_selecting_{len(selected_chunks)}_of_{total_chunks}_chunks",
                progress_callback,
            )
            
            with open(llm_dir / "extraction_schema_chunk.json", "w", encoding="utf-8") as f:
                json.dump(chunk_schema, f, indent=2, ensure_ascii=False)

            with open(llm_dir / "extraction_chunk_selection.json", "w", encoding="utf-8") as f:
                json.dump(selection_debug, f, indent=2, ensure_ascii=False)
            logger.info(
                "Metadata chunk selection: selected chunks %s, skipped %s chunks",
                [chunk["chunk_num"] for chunk in selected_chunks],
                len(selection_debug["skipped_chunks"]),
            )

            await self._update_progress(
                document.id,
                63,
                f"extracting_metadata_deterministic_{len(selected_chunks)}_of_{total_chunks}_chunks",
                progress_callback,
            )
            deterministic_candidates = self._deterministic_candidate_extraction(selected_chunks, fields)
            deterministic_result = self._resolve_chunk_candidate_results(
                [(1, deterministic_candidates)],
                fields,
                ocr_result.pages,
            )

            llm_was_skipped = self._all_required_fields_resolved(deterministic_result, fields)
            all_results = []
            chunk_responses = []
            chunk_durations = []

            if llm_was_skipped:
                logger.info("Skipping LLM metadata extraction: deterministic extraction found all required fields")
                result = deterministic_result
                response_text = json.dumps({"skipped_llm": True, "reason": "deterministic_required_fields_found"}, indent=2)
                with open(llm_dir / "extraction_response.txt", "w", encoding="utf-8") as f:
                    f.write(response_text)
                with open(llm_dir / "extraction_timing.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "duration_seconds": 0,
                        "provider": self.llm.provider,
                        "model": self.llm.model,
                        "original_chunk_count": total_chunks,
                        "selected_chunk_count": len(selected_chunks),
                        "llm_skipped": True,
                    }, f, indent=2, ensure_ascii=False)
            else:
                selected_text, selected_prompt_debug = self._build_selected_chunks_text(selected_chunks, fields)
                compact_preamble = preamble
                if compact_preamble and len(compact_preamble) > 1500:
                    compact_preamble = compact_preamble[:1500].rstrip()
                    selection_debug["preamble_truncated_chars"] = len(preamble) - len(compact_preamble)
                prompt = self._build_extraction_prompt(
                    fields,
                    selected_text,
                    classification.doc_type_slug,
                    compact_preamble,
                    chunk_num=1,
                    total_chunks=1,
                )
                for prompt_text_budget in (3000, 2200, 1600):
                    if len(prompt) <= 11000:
                        break
                    selected_text, selected_prompt_debug = self._build_selected_chunks_text(
                        selected_chunks,
                        fields,
                        max_chars=prompt_text_budget,
                    )
                    selection_debug["prompt_reduced_to_chars"] = prompt_text_budget
                    prompt = self._build_extraction_prompt(
                        fields,
                        selected_text,
                        classification.doc_type_slug,
                        compact_preamble,
                        chunk_num=1,
                        total_chunks=1,
                    )
                selection_debug["prompt_chunks"] = selected_prompt_debug
                selection_debug["prompt_text_chars"] = len(selected_text)
                selection_debug["prompt_chars"] = len(prompt)
                with open(llm_dir / "extraction_chunk_selection.json", "w", encoding="utf-8") as f:
                    json.dump(selection_debug, f, indent=2, ensure_ascii=False)

                with open(llm_dir / "extraction_prompt.txt", "w", encoding="utf-8") as f:
                    f.write(prompt)
                logger.info(
                    "Selected chunk prompt size: %s chars document text, %s chars full prompt",
                    len(selected_text),
                    len(prompt),
                )

                try:
                    await self._update_progress(
                        document.id,
                        68,
                        f"extracting_metadata_llm_selected_{len(selected_chunks)}_of_{total_chunks}_chunks",
                        progress_callback,
                    )
                    logger.info("Starting single-call selected-chunk metadata extraction")
                    try:
                        chunk_result, chunk_response_text, chunk_curl_command, chunk_duration = await self.llm.generate_json_with_raw(prompt, None)
                    except Exception as parse_error:
                        logger.warning(f"Selected-chunk candidate extraction failed: {parse_error}")
                        with open(llm_dir / "extraction_warning_selected_chunks.txt", "w", encoding="utf-8") as f:
                            f.write(f"Candidate JSON parse failed:\n{parse_error}\n")
                        context_error = "maximum context length" in str(parse_error).lower() or "reduce the length" in str(parse_error).lower()
                        if context_error:
                            selected_text, selected_prompt_debug = self._build_selected_chunks_text(
                                selected_chunks,
                                fields,
                                max_chars=1000,
                            )
                            selection_debug["prompt_chunks"] = selected_prompt_debug
                            selection_debug["prompt_text_chars"] = len(selected_text)
                            selection_debug["prompt_reduced_after_context_error"] = True
                            prompt = self._build_extraction_prompt(
                                fields,
                                selected_text,
                                classification.doc_type_slug,
                                compact_preamble[:800] if compact_preamble else "",
                                chunk_num=1,
                                total_chunks=1,
                            )
                            selection_debug["prompt_chars"] = len(prompt)
                            with open(llm_dir / "extraction_prompt.txt", "w", encoding="utf-8") as prompt_file:
                                prompt_file.write(prompt)
                            with open(llm_dir / "extraction_chunk_selection.json", "w", encoding="utf-8") as selection_file:
                                json.dump(selection_debug, selection_file, indent=2, ensure_ascii=False)
                            chunk_result, chunk_response_text, chunk_curl_command, chunk_duration = await self.llm.generate_json_with_raw(prompt, None)
                        else:
                            repair_prompt = self._build_candidate_json_repair_prompt(prompt)
                            chunk_result, chunk_response_text, chunk_curl_command, chunk_duration = await self.llm.generate_json_with_raw(repair_prompt, None)
                        with open(llm_dir / "extraction_warning_selected_chunks.txt", "a", encoding="utf-8") as f:
                            f.write("\nRetry with stricter compact JSON prompt succeeded.\n")

                    chunk_result = self._normalize_candidate_chunk_result(chunk_result, fields, 1, selected_text)
                    if deterministic_candidates.get("candidates"):
                        for field_key, candidates in deterministic_candidates["candidates"].items():
                            if candidates:
                                chunk_result.setdefault("candidates", {}).setdefault(field_key, []).extend(candidates)

                    all_results.append((1, chunk_result))
                    chunk_responses.append(chunk_response_text)
                    chunk_durations.append(chunk_duration)
                    response_text = chunk_response_text

                    with open(llm_dir / "extraction_response.txt", "w", encoding="utf-8") as f:
                        f.write(chunk_response_text)
                    if chunk_curl_command:
                        with open(llm_dir / "extraction_curl.txt", "w", encoding="utf-8") as f:
                            f.write(chunk_curl_command)
                except Exception as e:
                    logger.warning(f"Selected-chunk extraction failed: {e}")
                    with open(llm_dir / "extraction_error.txt", "w", encoding="utf-8") as f:
                        f.write(f"Selected-chunk extraction failed:\n{e}\n")
                    raise
            
            # Update progress for merging
            await self._update_progress(document.id, MERGE_START, "extracting_metadata_merging", progress_callback)
            
            # Merge all chunk results
            if result is not None:
                logger.info("Using deterministic metadata result")
            elif all_results:
                logger.info(f"Merging {len(all_results)} chunk results")
                result = self._resolve_chunk_candidate_results(all_results, fields, ocr_result.pages)
                if result is None:
                    logger.warning("Failed to merge chunk results, using first chunk")
                    result = self._empty_extraction_result(fields)
                else:
                    # Log merged result
                    if "data" in result:
                        merged_fields = [k for k, v in result["data"].items() if v is not None]
                        logger.info(f"Merged result: {len(merged_fields)} fields with values: {merged_fields}")
                    rejected_candidates = result.get("rejected_candidates", {})
                    if rejected_candidates:
                        rejected_count = sum(len(candidates) for candidates in rejected_candidates.values())
                        logger.info(f"Rejected {rejected_count} lower-priority chunk candidates during merge")
                
                # Ensure all expected fields are present in merged result
                expected_field_keys = [key for key, _, _, _, _, _ in fields]
                if "data" in result and isinstance(result["data"], dict):
                    for field_key in expected_field_keys:
                        if field_key not in result["data"]:
                            logger.warning(f"Field '{field_key}' missing from merged extraction result, adding as null")
                            result["data"][field_key] = None
                    
                    # Remove any unexpected fields
                    unexpected_fields = [k for k in result["data"].keys() if k not in expected_field_keys]
                    if unexpected_fields:
                        logger.warning(f"Removing unexpected fields from merged result: {unexpected_fields}")
                        for unexpected_field in unexpected_fields:
                            result["data"].pop(unexpected_field, None)
                            if "evidence" in result and isinstance(result["evidence"], dict):
                                result["evidence"].pop(unexpected_field, None)
                
                # Update progress after merging - keep chunk info in stage for visibility
                await self._update_progress(document.id, MERGE_END, f"extracting_metadata_chunk_done_{total_chunks}", progress_callback)
                
                # Combine all response texts for logging
                response_text = "\n\n--- SELECTED CHUNK MERGE ---\n\n".join(chunk_responses)
                
                # Save merged prompt and response
                with open(llm_dir / "extraction_selected_chunks.txt", "w", encoding="utf-8") as f:
                    f.write(f"# Selected chunks ({len(selected_chunks)} of {len(chunks)})\n")
                    for chunk in selected_chunks:
                        f.write(f"\n--- Chunk {chunk['chunk_num']} page {int(chunk.get('page', 0)) + 1} ---\n")
                        f.write(chunk["text"])
                
                with open(llm_dir / "extraction_response.txt", "w", encoding="utf-8") as f:
                    f.write(response_text)
                
                # Calculate and save total duration
                total_duration = sum(chunk_durations) if chunk_durations else 0
                with open(llm_dir / "extraction_timing.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "duration_seconds": total_duration,
                        "chunk_count": len(chunk_durations),
                        "original_chunk_count": total_chunks,
                        "selected_chunk_count": len(selected_chunks),
                        "chunk_durations": chunk_durations,
                        "provider": self.llm.provider,
                        "model": self.llm.model
                    }, f, indent=2)
                
                logger.info(
                    "LLM metadata extraction completed for document %s (%s selected of %s chunks) in %.2fs total",
                    document_dir.parent.name,
                    len(selected_chunks),
                    len(chunks),
                    total_duration,
                )
                
                # Save merged result
                with open(llm_dir / "extraction_result.json", "w", encoding="utf-8") as f:
                    json.dump(self._json_serialize(result), f, indent=2, ensure_ascii=False)
                if isinstance(result, dict) and result.get("rejected_candidates"):
                    with open(llm_dir / "extraction_rejected_candidates.json", "w", encoding="utf-8") as f:
                        json.dump(self._json_serialize(result["rejected_candidates"]), f, indent=2, ensure_ascii=False)
            else:
                raise Exception("All chunk extractions failed")

            if result is None:
                raise Exception("Selected chunk extraction failed: no result obtained")

            expected_field_keys = [self._canonical_field_key(key) for key, _, _, _, _, _ in fields]
            if "data" in result and isinstance(result["data"], dict):
                for field_key in expected_field_keys:
                    if field_key not in result["data"]:
                        logger.warning(f"Field '{field_key}' missing from selected-chunk result, adding as null")
                        result["data"][field_key] = None

                unexpected_fields = [k for k in result["data"].keys() if k not in expected_field_keys]
                if unexpected_fields:
                    logger.warning(f"Removing unexpected fields from selected-chunk result: {unexpected_fields}")
                    for unexpected_field in unexpected_fields:
                        result["data"].pop(unexpected_field, None)
                        if "evidence" in result and isinstance(result["evidence"], dict):
                            result["evidence"].pop(unexpected_field, None)

            await self._update_progress(document.id, MERGE_END, f"extracting_metadata_selected_chunks_done_{len(selected_chunks)}_of_{total_chunks}", progress_callback)
            with open(llm_dir / "extraction_selected_chunks.txt", "w", encoding="utf-8") as f:
                f.write(f"# Selected chunks ({len(selected_chunks)} of {len(chunks)})\n")
                for chunk in selected_chunks:
                    f.write(f"\n--- Chunk {chunk['chunk_num']} page {int(chunk.get('page', 0)) + 1} ---\n")
                    f.write(chunk["text"])

            with open(llm_dir / "extraction_result.json", "w", encoding="utf-8") as f:
                json.dump(self._json_serialize(result), f, indent=2, ensure_ascii=False)
            if isinstance(result, dict) and result.get("rejected_candidates"):
                with open(llm_dir / "extraction_rejected_candidates.json", "w", encoding="utf-8") as f:
                    json.dump(self._json_serialize(result["rejected_candidates"]), f, indent=2, ensure_ascii=False)
        else:
            # Single chunk - normal processing
            prompt = self._build_extraction_prompt(fields, filtered_text, classification.doc_type_slug, preamble)
            
            with open(llm_dir / "extraction_prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
            
            result = None
            response_text = None
            curl_command = None
            duration = None
            try:
                logger.info(f"Starting LLM metadata extraction for document {document_dir.parent.name}")
                result, response_text, curl_command, duration = await self.llm.generate_json_with_raw(prompt, schema)
                logger.info(f"LLM metadata extraction completed for document {document_dir.parent.name} in {duration:.2f}s")
                
                # Save response, curl command, and timing immediately after successful request
                with open(llm_dir / "extraction_response.txt", "w", encoding="utf-8") as f:
                    f.write(response_text)
                
                if curl_command:
                    with open(llm_dir / "extraction_curl.txt", "w", encoding="utf-8") as f:
                        f.write(curl_command)
                
                # Save timing metadata
                with open(llm_dir / "extraction_timing.json", "w", encoding="utf-8") as f:
                    json.dump({"duration_seconds": duration, "provider": self.llm.provider, "model": self.llm.model}, f, indent=2, ensure_ascii=False)
                
                with open(llm_dir / "extraction_result.json", "w", encoding="utf-8") as f:
                    json.dump(self._json_serialize(result), f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"LLM metadata extraction failed for document {document_dir.parent.name}: {e}")
                error_msg = str(e)
                with open(llm_dir / "extraction_error.txt", "w", encoding="utf-8") as f:
                    f.write(error_msg)
                
                # Save curl command even if there was an error (if we got that far)
                if curl_command:
                    with open(llm_dir / "extraction_curl.txt", "w", encoding="utf-8") as f:
                        f.write(curl_command)
                
                # If we have response_text but parsing failed, try to repair it
                if response_text:
                    logger.warning("Attempting to repair JSON from response_text after parsing failure")
                    try:
                        repaired_result = self.llm._repair_json(response_text)
                        if repaired_result:
                            logger.info("Successfully repaired JSON from response_text")
                            result = repaired_result
                            # Save the repaired result
                            with open(llm_dir / "extraction_response.txt", "w", encoding="utf-8") as f:
                                f.write(response_text)
                            with open(llm_dir / "extraction_result.json", "w", encoding="utf-8") as f:
                                json.dump(self._json_serialize(result), f, indent=2, ensure_ascii=False)
                        else:
                            logger.error("Failed to repair JSON even from response_text")
                            # Try to extract at least one object as fallback
                            json_objects = self.llm._extract_json_objects(response_text)
                            if json_objects:
                                logger.warning(f"Using first extracted object as fallback from {len(json_objects)} objects")
                                result = json_objects[0]
                                with open(llm_dir / "extraction_result.json", "w", encoding="utf-8") as f:
                                    json.dump(self._json_serialize(result), f, indent=2, ensure_ascii=False)
                            else:
                                raise
                    except Exception as repair_error:
                        logger.error(f"JSON repair attempt also failed: {repair_error}")
                        # Last resort: try to extract any JSON object
                        try:
                            json_objects = self.llm._extract_json_objects(response_text)
                            if json_objects:
                                logger.warning(f"Using first extracted object as last resort from {len(json_objects)} objects")
                                result = json_objects[0]
                                with open(llm_dir / "extraction_result.json", "w", encoding="utf-8") as f:
                                    json.dump(self._json_serialize(result), f, indent=2, ensure_ascii=False)
                            else:
                                raise e  # Re-raise original error
                        except Exception as extract_error:
                            logger.error(f"JSON extraction also failed: {extract_error}")
                            raise e  # Re-raise original error
                else:
                    # No response_text available - can't repair
                    raise
            
            # Ensure result is set before post-processing
            if result is None:
                raise Exception("LLM extraction failed: no result obtained")
            
            # Validate that result contains expected structure
            if "data" not in result or not isinstance(result["data"], dict):
                logger.error(f"Invalid extraction result structure: {result}")
                raise Exception(f"LLM extraction returned invalid structure. Expected {{'data': {{...}}, 'evidence': {{...}}}}, got: {list(result.keys())}")
            
            # Ensure all expected fields are present in result (add null if missing)
            expected_field_keys = [key for key, _, _, _, _, _ in fields]
            for field_key in expected_field_keys:
                if field_key not in result["data"]:
                    logger.warning(f"Field '{field_key}' missing from LLM extraction result, adding as null")
                    result["data"][field_key] = None
            
            # Remove any unexpected fields that are not in the schema
            unexpected_fields = [k for k in result["data"].keys() if k not in expected_field_keys]
            if unexpected_fields:
                logger.warning(f"Removing unexpected fields from extraction result: {unexpected_fields}")
                for unexpected_field in unexpected_fields:
                    result["data"].pop(unexpected_field, None)
                    # Also remove from evidence if present
                    if "evidence" in result and isinstance(result["evidence"], dict):
                        result["evidence"].pop(unexpected_field, None)

        # Post-processing steps - include chunk info if we used chunks
        post_stage_suffix = f"_chunks_{total_chunks}" if total_chunks else ""
        await self._update_progress(document.id, 78, f"extracting_metadata_post_processing{post_stage_suffix}", progress_callback)
        # Fill in missing quotes from document text
        result = self._fill_missing_quotes(result, ocr_result.pages)
        
        # Single-chunk legacy extraction may still need regex cleanup. Multi-chunk
        # candidate extraction uses regex only as candidate validation before resolve.
        if total_chunks:
            logger.info("Skipping regex post-processing for multi-chunk candidate extraction")
        else:
            result = self._apply_regex_filters(result, fields, llm_dir)
        
        # Validate evidence spans
        await self._update_progress(document.id, 80, f"extracting_metadata_validating{post_stage_suffix}", progress_callback)
        # Normalize extraction data to ensure correct structure
        # Pass expected fields to filter out unexpected fields during normalization
        expected_field_keys = {self._canonical_field_key(key) for key, _, _, _, _, _ in fields}
        logger.info(f"Expected fields for validation: {sorted(expected_field_keys)}")
        normalized_result = self._normalize_extraction_data(result, expected_fields=expected_field_keys)
        
        # Double-check: remove any unexpected fields that might have slipped through
        if "data" in normalized_result:
            unexpected_after_norm = [k for k in normalized_result["data"].keys() if k not in expected_field_keys]
            if unexpected_after_norm:
                logger.warning(f"Removing unexpected fields after normalization: {unexpected_after_norm}")
                for unexpected_field in unexpected_after_norm:
                    normalized_result["data"].pop(unexpected_field, None)
                    if "evidence" in normalized_result and isinstance(normalized_result["evidence"], dict):
                        normalized_result["evidence"].pop(unexpected_field, None)
        
        evidence_data = ExtractionEvidence(**normalized_result)
        validation_errors = self._validate_evidence(evidence_data, ocr_result.pages)
        
        # Log what fields are actually in evidence_data.data
        actual_fields = set(evidence_data.data.keys())
        logger.info(f"Fields in evidence_data.data: {sorted(actual_fields)}")
        unexpected_in_evidence = actual_fields - expected_field_keys
        if unexpected_in_evidence:
            logger.error(f"CRITICAL: Unexpected fields still in evidence_data.data after normalization: {sorted(unexpected_in_evidence)}")
        
        # Build verified.json - only for fields that are in the schema
        verified = {}
        for key in evidence_data.data.keys():
            # Skip fields that are not in the schema (e.g., "datasets" from LLM mistakes)
            if key not in expected_field_keys:
                logger.warning(f"Skipping verification for unexpected field '{key}' (not in schema)")
                continue
                
            value = evidence_data.data.get(key)
            evidence_spans = evidence_data.evidence.get(key, [])
            
            if value is not None and evidence_spans:
                # Validate first span
                first_span = evidence_spans[0]
                if first_span.page < len(ocr_result.pages):
                    page_text = ocr_result.pages[first_span.page]["text"]
                    if first_span.start < len(page_text) and first_span.end <= len(page_text):
                        snippet = page_text[first_span.start:first_span.end]
                        verified[key] = {
                            "verified": True,
                            "method": "evidence",
                            "page": first_span.page,
                            "snippet": snippet
                        }
                        continue
            elif value is not None and not evidence_spans:
                # Value exists but no evidence spans - try to find it in the document text
                value_str = str(value).strip()
                if value_str:
                    # Search for the value in all pages
                    found_evidence = False
                    for page_idx, page in enumerate(ocr_result.pages):
                        page_text = page.get("text", "")
                        # Normalize whitespace for comparison (OCR often adds extra whitespace/newlines)
                        page_text_normalized = ' '.join(page_text.split())
                        value_str_normalized = ' '.join(value_str.split())
                        
                        # Try exact match first
                        if value_str in page_text:
                            start_pos = page_text.find(value_str)
                            verified[key] = {
                                "verified": True,
                                "method": "auto_found",
                                "page": page_idx,
                                "snippet": value_str
                            }
                            found_evidence = True
                            logger.info(f"Auto-found evidence for field '{key}' in page {page_idx}")
                            break
                        # Try normalized whitespace match
                        elif value_str_normalized in page_text_normalized:
                            verified[key] = {
                                "verified": True,
                                "method": "auto_found_normalized",
                                "page": page_idx,
                                "snippet": value_str_normalized
                            }
                            found_evidence = True
                            logger.info(f"Auto-found evidence for field '{key}' in page {page_idx} (normalized whitespace)")
                            break
                        # Try case-insensitive match
                        elif value_str.lower() in page_text.lower():
                            start_pos = page_text.lower().find(value_str.lower())
                            verified[key] = {
                                "verified": True,
                                "method": "auto_found",
                                "page": page_idx,
                                "snippet": page_text[start_pos:start_pos + len(value_str)]
                            }
                            found_evidence = True
                            logger.info(f"Auto-found evidence for field '{key}' in page {page_idx} (case-insensitive)")
                            break
                        # Try normalized case-insensitive match
                        elif value_str_normalized.lower() in page_text_normalized.lower():
                            verified[key] = {
                                "verified": True,
                                "method": "auto_found_normalized",
                                "page": page_idx,
                                "snippet": value_str_normalized
                            }
                            found_evidence = True
                            logger.info(f"Auto-found evidence for field '{key}' in page {page_idx} (normalized, case-insensitive)")
                            break
                        # For numeric values, try different formats (100000 -> 100.000 or 100,000)
                        elif isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '').replace(',', '').isdigit()):
                            try:
                                num_value = float(str(value).replace(',', '.')) if isinstance(value, str) else float(value)
                                # Try European format: 100.000
                                european_format = f"{num_value:,.0f}".replace(',', '.')
                                # Try with euro sign
                                euro_format = f"€ {european_format}"
                                euro_format_alt = f"€{european_format}"
                                
                                for fmt in [european_format, euro_format, euro_format_alt, f"{int(num_value)}"]:
                                    if fmt in page_text or fmt in page_text_normalized:
                                        verified[key] = {
                                            "verified": True,
                                            "method": "auto_found_number_format",
                                            "page": page_idx,
                                            "snippet": fmt
                                        }
                                        found_evidence = True
                                        logger.info(f"Auto-found evidence for field '{key}' in page {page_idx} (number format: {fmt})")
                                        break
                                if found_evidence:
                                    break
                            except (ValueError, TypeError):
                                pass
                    
                    if found_evidence:
                        continue
            
            verified[key] = {
                "verified": False,
                "method": "none",
                "page": None,
                "snippet": None
            }
        
        # Apply hard validators and add to validation_errors
        # Filter evidence_data.data to only include expected fields before validation
        filtered_data = {k: v for k, v in evidence_data.data.items() if k in expected_field_keys}
        validation_errors.extend(self._validate_hard_validators(filtered_data, fields))
        
        # Check required fields - only check fields that are in the schema
        for key, label, field_type, is_required, enum_values, regex in fields:
            # Double-check: skip if key is not in expected fields (shouldn't happen, but safety check)
            if key not in expected_field_keys:
                logger.warning(f"Skipping validation for field '{key}' - not in expected fields list")
                continue
                
            if is_required:
                val = evidence_data.data.get(key)
                if val is None or val == "":
                    validation_errors.append(f"missing_required_field:{key}")
                else:
                    if not verified.get(key, {}).get("verified", False):
                        validation_errors.append(f"missing_verified_required_field:{key}")
        
        # Optional: RobBERT evidence retrieval for unverified required fields
        if os.getenv("MPROOF_ROBBERT_EVIDENCE") == "1":
            try:
                from sentence_transformers import SentenceTransformer, util
                robbert_model = SentenceTransformer("NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers")
                
                # Get required fields without evidence
                unverified_required = [
                    (key, label, evidence_data.data.get(key))
                    for key, label, field_type, is_required, enum_values, regex in fields
                    if is_required and not verified.get(key, {}).get("verified", False) and evidence_data.data.get(key) is not None
                ]
                
                if unverified_required:
                    # Split filtered_text into candidate sentences
                    sentences = [s.strip() for s in filtered_text.split('\n') if 20 <= len(s.strip()) <= 300][:500]
                    
                    if sentences:
                        # Embed sentences once
                        sentence_embeddings = robbert_model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)
                        
                        for key, label, value in unverified_required:
                            if value is None:
                                continue
                            
                            query_text = f"{label or key}: {value}"
                            query_embedding = robbert_model.encode([query_text], show_progress_bar=False, convert_to_numpy=True)[0]
                            
                            # Find best match
                            scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
                            best_idx = int(scores.argmax())
                            best_score = float(scores[best_idx])
                            
                            if best_score >= 0.45:
                                verified[key]["semantic_snippet"] = sentences[best_idx]
                                verified[key]["semantic_score"] = best_score
                                logger.debug(f"RobBERT found semantic match for {key}: score={best_score:.3f}")
            except ImportError:
                pass  # sentence-transformers not available, skip silently
            except Exception as e:
                logger.debug(f"RobBERT evidence retrieval failed: {e}")

        # Save results
        await self._update_progress(document.id, 82, f"extracting_metadata_saving{post_stage_suffix}", progress_callback)
        metadata_dir = document_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        with open(metadata_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(evidence_data.data, f, indent=2, ensure_ascii=False)

        with open(metadata_dir / "validation.json", "w", encoding="utf-8") as f:
            json.dump({"errors": validation_errors}, f, indent=2, ensure_ascii=False)
        
        with open(metadata_dir / "verified.json", "w", encoding="utf-8") as f:
            json.dump(verified, f, indent=2, ensure_ascii=False)

        # Build comprehensive evidence.json by searching ALL pages for ALL field values
        # This ensures the PDF viewer can show all occurrences of extracted values
        merged_evidence = {}
        
        # First, copy existing LLM evidence (convert to list of dicts)
        for key, spans in evidence_data.evidence.items():
            if spans:
                merged_evidence[key] = [
                    {"page": s.page, "start": s.start, "end": s.end, "quote": s.quote}
                    for s in spans
                ]
        
        # Then search ALL pages for ALL field values to find additional occurrences
        for key, value in evidence_data.data.items():
            if key not in expected_field_keys or value is None:
                continue
                
            value_str = str(value).strip()
            if not value_str:
                continue
            
            # Get existing evidence pages for this field
            existing_pages = set()
            if key in merged_evidence:
                existing_pages = {e.get("page") for e in merged_evidence[key]}
            else:
                merged_evidence[key] = []
            
            # Search all pages
            value_str_normalized = ' '.join(value_str.split())
            
            for page_idx, page in enumerate(ocr_result.pages):
                if page_idx in existing_pages:
                    continue  # Already have evidence from this page
                    
                page_text = page.get("text", "")
                page_text_normalized = ' '.join(page_text.split())
                found_snippet = None
                
                # Try various matching strategies
                if value_str in page_text:
                    found_snippet = value_str
                elif value_str_normalized in page_text_normalized:
                    found_snippet = value_str_normalized
                elif value_str.lower() in page_text.lower():
                    start_pos = page_text.lower().find(value_str.lower())
                    found_snippet = page_text[start_pos:start_pos + len(value_str)]
                else:
                    # Try numeric formats
                    if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '').replace(',', '').isdigit()):
                        try:
                            num_value = float(str(value).replace(',', '.')) if isinstance(value, str) else float(value)
                            european_format = f"{num_value:,.0f}".replace(',', '.')
                            euro_format = f"€ {european_format}"
                            euro_format_alt = f"€{european_format}"
                            
                            for fmt in [european_format, euro_format, euro_format_alt, f"{int(num_value)}"]:
                                if fmt in page_text or fmt in page_text_normalized:
                                    found_snippet = fmt
                                    break
                        except (ValueError, TypeError):
                            pass
                
                if found_snippet:
                    merged_evidence[key].append({
                        "page": page_idx,
                        "start": 0,
                        "end": len(found_snippet),
                        "quote": found_snippet
                    })
                    logger.info(f"Found additional evidence for field '{key}' on page {page_idx}: '{found_snippet[:30]}...'")

        with open(metadata_dir / "evidence.json", "w", encoding="utf-8") as f:
            json.dump(merged_evidence, f, indent=2, ensure_ascii=False)

        await self._update_progress(document.id, 85, "extracting_metadata_complete", progress_callback)
        return evidence_data

    def _canonical_field_key(self, field_key: Any) -> str:
        """Normalize known field key variants before prompting and resolving."""
        key = str(field_key)
        aliases = {
            "energylabel": "energielabel",
        }
        return aliases.get(key, key)

    def _normalize_extraction_fields(self, fields: List[Tuple]) -> List[Tuple]:
        """Normalize extraction field keys while preserving all field metadata."""
        normalized_fields = []
        for key, label, field_type, is_required, enum_values, regex in fields:
            normalized_fields.append((
                self._canonical_field_key(key),
                label,
                field_type,
                is_required,
                enum_values,
                regex,
            ))

        return normalized_fields

    def _empty_extraction_result(self, fields: List[Tuple]) -> Dict[str, Any]:
        """Build an empty extraction result for the configured field list."""
        field_keys = [self._canonical_field_key(key) for key, _, _, _, _, _ in fields]
        return {
            "data": {field_key: None for field_key in field_keys},
            "evidence": {field_key: [] for field_key in field_keys},
        }

    def _candidate_from_value(
        self,
        value: Any,
        evidence: Any,
        chunk_index: int,
        confidence: float = 60,
        evidence_type: str = "semantic",
        record_role: str = "unknown",
    ) -> Dict[str, Any]:
        """Create a candidate object from loose or legacy extraction output."""
        quote = ""
        if isinstance(evidence, str):
            quote = evidence
        elif isinstance(evidence, list) and evidence:
            first_evidence = evidence[0]
            if isinstance(first_evidence, dict):
                quote = str(first_evidence.get("quote") or "")
            else:
                quote = str(first_evidence or "")
        elif isinstance(evidence, dict):
            quote = str(evidence.get("quote") or "")

        if not quote and value is not None:
            quote = str(value)

        return {
            "value": "" if value is None else str(value),
            "normalized_value": "" if value is None else str(value),
            "unit": None,
            "evidence": quote,
            "chunk_index": chunk_index,
            "confidence": confidence,
            "evidence_type": evidence_type,
            "record_role": record_role,
        }

    def _strip_system_chunk_metadata(self, chunk_text: str) -> str:
        """Remove non-document chunk metadata comments before evidence repair."""
        text_value = str(chunk_text or "")
        text_value = re.sub(r"<!--\s*SYSTEM CHUNK METADATA.*?-->\s*", "", text_value, flags=re.DOTALL)
        text_value = text_value.replace("DOCUMENT CHUNK TEXT:\n", "")
        return text_value

    def _repair_candidate_evidence_from_chunk(
        self,
        candidate: Dict[str, Any],
        chunk_text: str,
    ) -> Dict[str, Any]:
        """Repair LLM evidence by expanding around the value in original chunk text."""
        if not chunk_text:
            return candidate
        if self._candidate_value_matches_evidence(
            candidate.get("value"),
            candidate.get("normalized_value"),
            candidate.get("evidence"),
        ):
            return candidate

        source_text = self._strip_system_chunk_metadata(chunk_text)
        value_candidates = [
            str(candidate.get("value") or "").strip(),
            str(candidate.get("normalized_value") or "").strip(),
        ]
        value_candidates = [value for value in value_candidates if value and not self._candidate_value_is_empty(value)]

        for value in value_candidates:
            match = re.search(re.escape(value), source_text, re.IGNORECASE)
            if not match:
                continue

            value_line_start = source_text.rfind("\n", 0, match.start()) + 1
            value_line_end = source_text.find("\n", match.end())
            if value_line_end < 0:
                value_line_end = len(source_text)

            all_lines = source_text.splitlines()
            char_cursor = 0
            value_line_index = 0
            for index, line in enumerate(all_lines):
                next_cursor = char_cursor + len(line) + 1
                if char_cursor <= match.start() < next_cursor:
                    value_line_index = index
                    break
                char_cursor = next_cursor

            start_line = max(0, value_line_index - 6)
            end_line = min(len(all_lines), value_line_index + 3)
            table_lines = [line.strip() for line in all_lines[start_line:end_line] if line.strip()]
            has_header_context = any(re.search(r"[A-Za-zÀ-ÿ]", line) for line in table_lines[: max(1, value_line_index - start_line)])
            if has_header_context and len(table_lines) >= 2:
                repaired_evidence = "\n".join(table_lines)
            else:
                start = max(0, match.start() - 300)
                end = min(len(source_text), match.end() + 300)
                repaired_evidence = source_text[start:end].strip()

            repaired = {
                **candidate,
                "evidence": repaired_evidence,
                "evidence_repair": {
                    "original_evidence": candidate.get("evidence"),
                    "repaired_evidence": repaired_evidence,
                    "repair_reason": "value_found_nearby_in_chunk",
                },
            }
            return repaired

        return candidate

    def _normalize_candidate_chunk_result(
        self,
        chunk_result: Dict[str, Any],
        fields: List[Tuple],
        chunk_num: int,
        chunk_text: str = "",
    ) -> Dict[str, Any]:
        """Coerce LLM chunk output into the candidate-only shape expected by the resolver."""
        field_keys = [self._canonical_field_key(key) for key, _, _, _, _, _ in fields]
        chunk_index = chunk_num - 1
        normalized_candidates = {field_key: [] for field_key in field_keys}

        if not isinstance(chunk_result, dict):
            return {"candidates": normalized_candidates}

        candidates = chunk_result.get("candidates")
        if isinstance(candidates, dict):
            for field_key in field_keys:
                raw_candidates = candidates.get(field_key, [])
                if field_key == "energielabel" and not raw_candidates:
                    raw_candidates = candidates.get("energylabel", [])
                if raw_candidates is None:
                    continue
                if not isinstance(raw_candidates, list):
                    raw_candidates = [raw_candidates]

                for raw_candidate in raw_candidates:
                    if isinstance(raw_candidate, dict):
                        candidate = {
                            "value": raw_candidate.get("value"),
                            "normalized_value": raw_candidate.get("normalized_value"),
                            "unit": raw_candidate.get("unit"),
                            "evidence": str(raw_candidate.get("evidence") or ""),
                            "chunk_index": chunk_index,
                            "confidence": raw_candidate.get("confidence", 0),
                            "evidence_type": raw_candidate.get("evidence_type", "ambiguous"),
                            "record_role": raw_candidate.get("record_role", "unknown"),
                        }
                    else:
                        candidate = self._candidate_from_value(raw_candidate, raw_candidate, chunk_index, confidence=50, evidence_type="ambiguous")

                    if not self._candidate_value_is_empty(candidate["value"]):
                        candidate = self._repair_candidate_evidence_from_chunk(candidate, chunk_text)
                        normalized_candidates[field_key].append(candidate)

            return {"candidates": normalized_candidates}

        data = chunk_result.get("data")
        if isinstance(data, dict):
            evidence = chunk_result.get("evidence", {})
            if not isinstance(evidence, dict):
                evidence = {}

            for field_key in field_keys:
                value = data.get(field_key)
                evidence_value = evidence.get(field_key)
                if field_key == "energielabel" and value is None:
                    value = data.get("energylabel")
                    evidence_value = evidence.get("energylabel")
                if self._candidate_value_is_empty(value):
                    continue

                normalized_candidates[field_key].append(
                    self._repair_candidate_evidence_from_chunk(
                        self._candidate_from_value(
                            value,
                            evidence_value,
                            chunk_index,
                            confidence=55,
                            evidence_type="semantic",
                            record_role="unknown",
                        ),
                        chunk_text,
                    )
                )

        return {"candidates": normalized_candidates}

    def _normalize_candidate_label(self, value: Any, allowed_values: Set[str], default: str) -> str:
        """Normalize a bounded candidate label returned by the LLM."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in allowed_values:
                return normalized

        return default

    def _candidate_value_is_empty(self, value: Any) -> bool:
        """Return whether a candidate value is empty or a placeholder."""
        if value is None:
            return True

        value_str = str(value).strip()
        if not value_str:
            return True

        placeholder_values = {
            "null",
            "none",
            "n/a",
            "na",
            "not found",
            "not_found",
            "niet gevonden",
            "niet opgenomen",
            "unknown",
            "onbekend",
            "-",
            "--",
        }
        return value_str.lower() in placeholder_values

    def _candidate_evidence_is_invalid(self, evidence: Any) -> bool:
        """Return whether evidence is empty or explicitly says no value was found."""
        if evidence is None:
            return True

        evidence_str = str(evidence).strip()
        if not evidence_str:
            return True

        invalid_fragments = [
            "niet opgenomen",
            "not found",
            "niet gevonden",
            "n/a",
        ]
        evidence_lower = evidence_str.lower()
        return any(fragment in evidence_lower for fragment in invalid_fragments)

    def _compact_for_candidate_match(self, value: Any) -> str:
        """Normalize text for clear value/evidence containment checks."""
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    def _candidate_value_matches_evidence(self, value: Any, normalized_value: Any, evidence: Any) -> bool:
        """Check whether a raw or normalized value is clearly supported by exact evidence."""
        evidence_str = str(evidence or "")
        evidence_lower = evidence_str.lower()
        for candidate_value in (value, normalized_value):
            if self._candidate_value_is_empty(candidate_value):
                continue

            value_str = str(candidate_value).strip()
            if value_str.lower() in evidence_lower:
                return True

            compact_value = self._compact_for_candidate_match(value_str)
            compact_evidence = self._compact_for_candidate_match(evidence_str)
            if compact_value and compact_value in compact_evidence:
                return True

        return False

    def _field_kind(self, field_config: Dict[str, Any]) -> str:
        """Infer a generic validation kind from configured field metadata."""
        field_type = str(field_config.get("field_type") or "text").lower()
        key_label = self._normalize_for_search(f"{field_config.get('key', '')} {field_config.get('label', '')}")

        if field_type in {"boolean", "bool"}:
            return "boolean"
        if field_type == "enum":
            return "enum"
        if field_type == "date":
            return "date"
        if any(term in key_label.split() for term in {"jaar", "year"}) or "bouwjaar" in key_label:
            return "year"
        if any(term in key_label for term in ["m2", "sqm", "area", "oppervlakte", "vloeroppervlak", "gebruiksoppervlakte"]):
            return "area"
        if field_type == "number":
            return "number"
        if field_type == "iban":
            return "iban"
        return "string"

    def _normalize_candidate_number(self, value: Any) -> Optional[str]:
        """Normalize a human numeric value without treating dates/codes as numbers."""
        value_str = str(value or "").strip()
        if not value_str:
            return None
        if re.search(r"[A-Za-z]", value_str):
            return None
        if re.search(r"\d+[/-]\d+", value_str):
            return None

        cleaned = re.sub(r"\s+", "", value_str)
        cleaned = re.sub(r"^[^\d+-]+|[^\d]+$", "", cleaned)
        if not cleaned:
            return None

        if re.fullmatch(r"[+-]?\d{1,3}(\.\d{3})+", cleaned):
            cleaned = cleaned.replace(".", "")
        elif "," in cleaned and "." in cleaned:
            cleaned = cleaned.replace(".", "").replace(",", ".")
        elif "," in cleaned:
            cleaned = cleaned.replace(",", ".")

        if not re.fullmatch(r"[+-]?\d+(\.\d+)?", cleaned):
            return None
        return cleaned

    def _evidence_has_label_context(self, field_config: Dict[str, Any], evidence: Any) -> bool:
        """Check whether evidence contains the configured key/label context."""
        evidence_norm = self._normalize_for_search(evidence)
        if not evidence_norm:
            return False

        label = self._normalize_for_search(field_config.get("label"))
        key = self._normalize_for_search(field_config.get("key"))
        if label and label in evidence_norm:
            return True
        if key and key in evidence_norm:
            return True

        label_tokens = self._search_tokens(label)
        evidence_tokens = evidence_norm.split()
        return self._words_within_window(evidence_tokens, label_tokens, window=10)

    def _validate_candidate_value_for_field(
        self,
        value: Any,
        field_config: Dict[str, Any],
        evidence: Any = "",
        unit: Any = None,
    ) -> Tuple[bool, Optional[str], Any]:
        """Hard-validate and optionally normalize a candidate before scoring."""
        if self._candidate_value_is_empty(value):
            return False, "empty_or_placeholder_value", value

        value_str = str(value).strip()
        evidence_str = str(evidence or "")
        evidence_norm = self._normalize_for_search(evidence_str)
        kind = self._field_kind(field_config)

        if kind == "year":
            if re.search(r"\d+[/-]\d+", value_str):
                return False, "year_value_looks_like_date", value
            match = re.fullmatch(r"\d{4}", value_str)
            if not match:
                return False, "year_must_be_four_digits", value
            year = int(value_str)
            if year < 1600 or year > datetime.now().year + 1:
                return False, "year_out_of_range", value
            value_pos = evidence_norm.find(value_str)
            date_label_positions = [
                evidence_norm.find(label)
                for label in ("datum", "date")
                if evidence_norm.find(label) >= 0
            ]
            if value_pos >= 0 and any(abs(value_pos - label_pos) <= 30 for label_pos in date_label_positions):
                return False, "year_evidence_has_date_label", value
            if re.search(r"\b\d{4}\s?[a-z]{2}\b", evidence_norm):
                return False, "year_evidence_has_postcode", value
            document_label_positions = [
                evidence_norm.find(label)
                for label in ("documentnummer", "document number", "nummer", "number", "reference", "referentie", "id")
                if evidence_norm.find(label) >= 0
            ]
            if value_pos >= 0 and any(abs(value_pos - label_pos) <= 35 for label_pos in document_label_positions):
                return False, "year_evidence_has_document_number_label", value
            return True, None, value_str

        if kind == "number":
            normalized_number = self._normalize_candidate_number(value_str)
            if normalized_number is None:
                return False, "number_invalid", value
            return True, None, normalized_number

        if kind == "area":
            normalized_number = self._normalize_candidate_number(value_str)
            if normalized_number is None:
                return False, "area_not_numeric", value
            if re.search(r"[€$£]|eur|euro|amount|bedrag|prijs|price", evidence_norm):
                return False, "area_looks_like_amount", value
            if re.search(r"\d+[/-]\d+", value_str):
                return False, "area_looks_like_date", value
            compact_digits = re.sub(r"\D", "", value_str)
            if len(compact_digits) >= 7:
                return False, "area_looks_like_phone_or_document_number", value
            if re.search(r"\b\d{4}\s?[a-z]{2}\b", evidence_norm):
                return False, "area_evidence_has_postcode", value

            unit_norm = self._normalize_for_search(unit)
            has_area_context = any(term in f"{evidence_norm} {unit_norm}" for term in [
                "m2",
                "sqm",
                "area",
                "oppervlakte",
                "vloeroppervlak",
                "gebruiksoppervlakte",
            ])
            if not has_area_context:
                return False, "area_missing_unit_or_context", value
            return True, None, normalized_number

        if kind == "enum":
            enum_values = field_config.get("enum_values")
            if enum_values:
                value_lower = value_str.lower()
                if not any(value_lower == str(enum_value).strip().lower() for enum_value in enum_values):
                    return False, "enum_value_not_allowed", value
            return True, None, value_str

        if kind == "boolean":
            bool_value = self._normalize_for_search(value_str)
            explicit_true = {"true", "yes", "ja", "waar", "1"}
            explicit_false = {"false", "no", "nee", "onwaar", "0"}
            if bool_value in explicit_true:
                return True, None, True
            if bool_value in explicit_false:
                return True, None, False
            return False, "boolean_not_explicit", value

        if kind == "date":
            if self._parse_date(value_str) is None:
                return False, "date_invalid", value
            return True, None, value_str

        if kind == "iban":
            if not self._validate_iban(value_str):
                return False, "iban_invalid", value
            return True, None, value_str

        if self._candidate_value_is_empty(value_str):
            return False, "string_empty_or_placeholder", value

        regex = field_config.get("regex")
        if regex:
            try:
                if re.search(str(regex), value_str, re.IGNORECASE) is None:
                    return False, "regex_mismatch", value
            except re.error:
                logger.warning(f"Invalid regex for candidate validation: {regex}")

        return True, None, value_str

    def _candidate_fits_field_type(self, value: Any, field_type: str, enum_values: Any, regex: Any = None) -> bool:
        """Validate a candidate value against the configured field type."""
        valid, _, _ = self._validate_candidate_value_for_field(
            value,
            {"field_type": field_type, "enum_values": enum_values, "regex": regex},
        )
        return valid

    def _candidate_final_value(self, candidate: Dict[str, Any]) -> Any:
        """Return the normalized value when available, otherwise the raw value."""
        normalized_value = candidate.get("normalized_value")
        if not self._candidate_value_is_empty(normalized_value):
            return normalized_value

        return candidate.get("value")

    def _find_evidence_span(self, evidence: str, pages: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Locate an exact evidence quote in OCR pages for downstream highlighting."""
        quote = evidence.strip()
        if not quote:
            return []

        if pages:
            for page_idx, page in enumerate(pages):
                page_text = page.get("text", "")
                start = page_text.find(quote)
                if start >= 0:
                    return [{
                        "page": page_idx,
                        "start": start,
                        "end": start + len(quote),
                        "quote": quote,
                    }]

            quote_lower = quote.lower()
            for page_idx, page in enumerate(pages):
                page_text = page.get("text", "")
                start = page_text.lower().find(quote_lower)
                if start >= 0:
                    matched_quote = page_text[start:start + len(quote)]
                    return [{
                        "page": page_idx,
                        "start": start,
                        "end": start + len(matched_quote),
                        "quote": matched_quote,
                    }]

        return [{
            "page": 0,
            "start": 0,
            "end": len(quote),
            "quote": quote,
        }]

    def _validate_candidate(self, candidate: Dict[str, Any], field_config: Dict[str, Any]) -> Optional[str]:
        """Validate one candidate after LLM output and normalization."""
        allowed_evidence_types = {"exact_label", "table_context", "nearby_label", "semantic", "ambiguous"}
        allowed_record_roles = {"primary", "secondary", "example", "background", "unknown"}

        if self._candidate_value_is_empty(candidate.get("value")):
            return "empty_value"
        if self._candidate_value_is_empty(candidate.get("normalized_value")):
            return "empty_normalized_value"
        if self._candidate_evidence_is_invalid(candidate.get("evidence")):
            return "invalid_evidence"
        if candidate.get("evidence_type") not in allowed_evidence_types:
            return "invalid_evidence_type"
        if candidate.get("record_role") not in allowed_record_roles:
            return "invalid_record_role"
        if candidate.get("confidence", 0) <= 0:
            return "non_positive_confidence"
        valid_value, invalid_reason, normalized_value = self._validate_candidate_value_for_field(
            candidate.get("selected_value"),
            field_config,
            evidence=candidate.get("evidence"),
            unit=candidate.get("unit"),
        )
        if not valid_value:
            return invalid_reason or "field_type_mismatch"
        regex = field_config.get("regex")
        if regex:
            try:
                if re.search(str(regex), str(candidate.get("selected_value")), re.IGNORECASE) is None:
                    return "regex_mismatch"
            except re.error:
                logger.warning(f"Invalid regex for candidate validation: {regex}")
        candidate["selected_value"] = normalized_value
        candidate["normalized_value"] = normalized_value
        if not self._candidate_value_matches_evidence(
            candidate.get("value"),
            candidate.get("normalized_value"),
            candidate.get("evidence"),
        ):
            return "value_not_in_evidence"

        return None

    def _score_candidate(self, candidate: Dict[str, Any], field_config: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Score a hard-validated candidate using evidence and field context first."""
        score = 60.0
        reasons = []
        reasons.append("valid_field_type:+60")

        evidence_type_scores = {
            "exact_label": 60,
            "table_context": 40,
            "nearby_label": 30,
            "semantic": 10,
            "ambiguous": 0,
        }
        record_role_scores = {
            "primary": 10,
            "unknown": 0,
            "secondary": -20,
            "example": -20,
            "background": -20,
        }

        evidence_type = candidate["evidence_type"]
        record_role = candidate["record_role"]
        evidence_type_score = evidence_type_scores[evidence_type]
        record_role_score = record_role_scores[record_role]
        score += evidence_type_score
        score += record_role_score
        reasons.extend([f"evidence_type:{evidence_type_score}", f"record_role:{record_role_score}"])

        if self._evidence_has_label_context(field_config, candidate.get("evidence")):
            score += 60
            reasons.append("exact_label_context:+60")

        confidence_score = min(20, max(0, float(candidate["confidence"]) / 5))
        score += confidence_score
        reasons.append(f"confidence_scaled:+{confidence_score:.1f}")

        chunk_index = candidate["chunk_index"]
        position_score = max(0, 5 - min(chunk_index, 5))
        score += position_score
        reasons.append(f"position:+{position_score}")

        return score, reasons

    def _resolve_chunk_candidate_results(
        self,
        chunk_results: List[Tuple[int, Dict[str, Any]]],
        fields: List[Tuple],
        pages: Optional[List[Dict[str, Any]]] = None,
        threshold: int = 100,
    ) -> Optional[Dict[str, Any]]:
        """Resolve candidate-only chunk extraction results into final field values."""
        if not chunk_results:
            return None

        field_config = {
            self._canonical_field_key(key): {
                "key": self._canonical_field_key(key),
                "label": label,
                "field_type": field_type,
                "enum_values": enum_values,
                "regex": regex,
            }
            for key, label, field_type, _, enum_values, regex in fields
        }
        field_keys = list(field_config.keys())
        candidates_by_field: Dict[str, List[Dict[str, Any]]] = {field_key: [] for field_key in field_keys}
        for chunk_num, chunk_result in chunk_results:
            if not isinstance(chunk_result, dict):
                continue

            candidates = chunk_result.get("candidates", {})
            if not isinstance(candidates, dict):
                continue

            chunk_index = chunk_num - 1
            for field_key in field_keys:
                raw_candidates = candidates.get(field_key, [])
                if field_key == "energielabel" and not raw_candidates:
                    raw_candidates = candidates.get("energylabel", [])
                if not isinstance(raw_candidates, list):
                    continue

                for raw_candidate in raw_candidates:
                    if not isinstance(raw_candidate, dict):
                        continue

                    confidence_raw = raw_candidate.get("confidence", 0)
                    try:
                        confidence = float(confidence_raw)
                    except (TypeError, ValueError):
                        confidence = 0.0

                    candidate = {
                        "field_key": field_key,
                        "value": raw_candidate.get("value"),
                        "normalized_value": raw_candidate.get("normalized_value"),
                        "unit": raw_candidate.get("unit"),
                        "evidence": str(raw_candidate.get("evidence") or ""),
                        "chunk_index": chunk_index,
                        "confidence": confidence,
                        "evidence_type": str(raw_candidate.get("evidence_type") or "").strip().lower(),
                        "record_role": str(raw_candidate.get("record_role") or "").strip().lower(),
                    }
                    candidate["selected_value"] = self._candidate_final_value(candidate)
                    candidates_by_field[field_key].append(candidate)

        resolved = self._empty_extraction_result(fields)
        rejected_candidates: Dict[str, List[Dict[str, Any]]] = {}

        for field_key, candidates in candidates_by_field.items():
            if not candidates:
                continue

            scored_candidates = []
            config = field_config[field_key]
            for candidate in candidates:
                reject_reason = self._validate_candidate(candidate, config)
                if reject_reason:
                    scored_candidates.append({
                        **candidate,
                        "score": None,
                        "rejection_reason": reject_reason,
                        "rejected_reason": reject_reason,
                    })
                    continue

                score, reasons = self._score_candidate(candidate, config)
                scored_candidate = {
                    **candidate,
                    "score": score,
                    "score_reasons": reasons,
                    "has_label_context": self._evidence_has_label_context(config, candidate.get("evidence")),
                }
                scored_candidates.append(scored_candidate)

            valid_candidates = [candidate for candidate in scored_candidates if candidate.get("score") is not None]
            if not valid_candidates:
                rejected_candidates[field_key] = scored_candidates
                continue

            valid_candidates.sort(
                key=lambda item: (
                    -item["score"],
                    not item.get("has_label_context", False),
                    item["chunk_index"],
                )
            )
            selected = valid_candidates[0]
            rejected = [candidate for candidate in scored_candidates if candidate is not selected]

            if selected["score"] >= threshold and not self._candidate_value_is_empty(selected["selected_value"]):
                resolved["data"][field_key] = selected["selected_value"]
                resolved["evidence"][field_key] = self._find_evidence_span(selected["evidence"], pages)
            else:
                rejected = scored_candidates

            rejected = [candidate for candidate in scored_candidates if candidate is not selected]
            if selected["score"] < threshold:
                rejected = scored_candidates

            if rejected:
                rejected_candidates[field_key] = sorted(
                    rejected,
                    key=lambda item: (
                        item.get("score") is None,
                        -(item.get("score") or -9999),
                        item["chunk_index"],
                    ),
                )

        if rejected_candidates:
            resolved["rejected_candidates"] = rejected_candidates

        return resolved

    def _build_candidate_extraction_schema(self, fields: List[Tuple]) -> Dict[str, Any]:
        """Build JSON schema for candidate-only chunk extraction."""
        candidate_properties = {
            "value": {"type": "string"},
            "normalized_value": {"type": "string"},
            "unit": {"type": ["string", "null"]},
            "evidence": {"type": "string"},
            "chunk_index": {"type": "integer"},
            "confidence": {"type": "number"},
            "evidence_type": {
                "type": "string",
                "enum": ["exact_label", "table_context", "nearby_label", "semantic", "ambiguous"],
            },
            "record_role": {
                "type": "string",
                "enum": ["primary", "secondary", "example", "background", "unknown"],
            },
        }
        candidates_properties = {}
        for key, _, _, _, _, _ in fields:
            field_key = self._canonical_field_key(key)
            candidates_properties[field_key] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "value",
                        "normalized_value",
                        "unit",
                        "evidence",
                        "chunk_index",
                        "confidence",
                        "evidence_type",
                        "record_role",
                    ],
                    "properties": candidate_properties,
                },
            }

        return {
            "type": "object",
            "required": ["candidates"],
            "properties": {
                "candidates": {
                    "type": "object",
                    "properties": candidates_properties,
                    "required": list(candidates_properties.keys()),
                }
            },
        }

    def _build_candidate_json_repair_prompt(self, original_prompt: str) -> str:
        """Build a stricter retry prompt for candidate extraction JSON parsing failures."""
        return f"""{original_prompt}

Your previous response was invalid JSON. Retry now.
Return exactly one compact JSON object.
No markdown. No comments. No trailing commas. No incomplete strings.
If uncertain, return empty arrays for the affected fields.
Do not start a candidate object unless you can complete every required property.
Every candidate object must be complete and use this exact property order:
value, normalized_value, unit, evidence, chunk_index, confidence, evidence_type, record_role."""

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
            if is_required:
                desc += " (required)"
            field_descriptions.append(desc)

        fields_str = "\n".join(field_descriptions)

        preamble_text = f"\n\n{preamble.strip()}" if preamble and preamble.strip() else ""
        notes = []
        if has_address_field:
            notes.append("- For address fields: extract a postal address (street, house number, postal code, city). Do not return only a person/company name.")
        if has_iban_field:
            notes.append("- For IBAN fields: return only the IBAN (e.g., NL..), no extra words or currency.")
        notes_text = "\n".join(notes)
        notes_block = f"\nSpecial notes:\n{notes_text}\n" if notes_text else ""
        
        # Add chunk info if multi-chunk
        chunk_info = ""
        is_chunk = bool(chunk_num and total_chunks)
        if chunk_num and total_chunks:
            chunk_index = chunk_num - 1
            chunk_info = f"""
NOTE: This is chunk {chunk_num} of {total_chunks} of a large document. Its zero-based chunk_index is {chunk_index}.

Return possible candidates per field only. Do not choose the final document-level value.
The application will collect all candidates from all chunks and resolve the best value per field later.
Use text, labels, nearby text, table headers, row labels and section structure.
Only return candidates that truly appear in this chunk's text or table structure.
For each candidate, include exact evidence, evidence_type, confidence and record_role.
For table_context candidates, evidence must include both the relevant headers and the row/value line. The candidate value must literally appear inside evidence.
If nothing is found for a field, return an empty array for that field.
Do not create placeholder candidates.
Do not use regex as the extraction method.

Determine record_role per candidate from document structure and relative position, not fixed document-type keywords:
- primary: the candidate appears to belong to the main subject that the document is primarily about.
- secondary: the candidate appears to belong to another record/entity/object/person/transaction than the main subject.
- example: the candidate appears to be part of an example, reference, comparison, illustration or demonstration.
- background: the candidate comes from general explanation, definitions, conditions or background text.
- unknown: insufficient context.

Structural guidance:
- Candidates near the document title, summary, first main section or first complete record are more often primary.
- Candidates in repeated records after the first complete record are more often secondary or example.
- Candidates in explanatory paragraphs are more often background.
- Candidates in tables may be primary when the table describes the first or central record."""

        # Build actual field keys for the JSON structure
        field_keys = [self._canonical_field_key(key) for key, _, _, _, _, _ in fields]
        
        # Build a concrete JSON template with the actual field names
        data_template = ", ".join([f'"{k}": null' for k in field_keys])
        field_list = ", ".join([f'"{k}"' for k in field_keys])
        json_template = f'{{"data": {{{data_template}}}, "evidence": {{}}}}'
        output_instruction = "Return JSON in this exact format (replace null with actual values, keep null if not found):"
        important_notes = [
            f"- Extract ALL fields listed above: {field_list}",
            "- Replace null with the actual extracted value for each field",
            "- If a field is not found in the document, keep it as null",
            "- Do NOT add fields that are not in the list above",
            '- Do NOT use placeholder values like "datasets" or other field names not in the schema',
        ]
        if is_chunk:
            candidate_template = ", ".join([f'"{k}": []' for k in field_keys])
            json_template = f'{{"candidates": {{{candidate_template}}}}}'
            output_instruction = "Return JSON in this exact candidate-only format:"
            important_notes = [
                f"- Return candidates for ALL fields listed above: {field_list}",
                "- Give possible candidates only. Never choose the definitive document value.",
                "- Use text, labels, nearby text and table structure to find candidates.",
                "- Only return candidates that really occur in this chunk.",
                "- Every candidate must include value, normalized_value, unit, evidence, chunk_index, confidence, evidence_type and record_role",
                "- evidence must be an exact quote from this chunk",
                "- For table_context candidates, evidence must include both the relevant headers and the row/value line. The candidate value must literally appear inside evidence.",
                "- Do not return candidates with an empty value; use an empty array for that field instead",
                "- Empty values, null-like values and placeholders are never valid candidates",
                "- Do not return background text as a candidate unless it contains a concrete value for the field",
                "- Return an empty array when no candidate is found. Do not create placeholder candidates.",
                "- Never return candidates with empty value. Never use evidence like 'not found' or 'niet opgenomen'.",
                "- evidence_type and record_role must use only the allowed enum values.",
                "- confidence is 0-100",
                "- evidence_type must be exactly one of: exact_label, table_context, nearby_label, semantic, ambiguous",
                "- record_role must be exactly one of: primary, secondary, example, background, unknown",
                "- The LLM must not choose final document-level values; the resolver will do that after all chunks",
                "- Do NOT add fields that are not in the list above",
                "- Return compact valid JSON only: no markdown, no comments, no trailing commas, no incomplete strings",
                "- If you cannot complete a candidate object, omit it and leave that field as []",
            ]
        extra_notes = ""
        if notes_block.strip():
            extra_notes = f"\n{notes_block.strip()}"
        important_notes_text = "\n".join(important_notes)
        
        return f"""Extract metadata from this {doc_type} document.{preamble_text}{chunk_info}

Fields to extract:
{fields_str}{extra_notes}

Document text:
{text}

{output_instruction}
{json_template}

IMPORTANT:
{important_notes_text}"""

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
                    page_num = span.get("page")
                    start = span.get("start")
                    end = span.get("end")
                    
                    # Skip if any required values are None
                    if page_num is None or start is None or end is None:
                        continue
                    
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
                with open(llm_dir / "regex_corrections.json", "w", encoding="utf-8") as f:
                    json.dump(regex_corrections, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Failed to save regex corrections: {e}")
        
        return result

    def _validate_iban(self, value: str) -> bool:
        """Validate IBAN checksum."""
        if not value:
            return False
        # Remove spaces/hyphens
        iban = re.sub(r'[\s\-]', '', value.upper())
        if len(iban) < 15 or len(iban) > 34:
            return False
        # Basic format check: 2 letters + 2 digits + alphanumeric
        if not re.match(r'^[A-Z]{2}\d{2}[A-Z0-9]+$', iban):
            return False
        # Simple checksum validation (mod 97)
        try:
            rearranged = iban[4:] + iban[:4]
            numeric = ''.join(str(ord(c) - ord('A') + 10) if c.isalpha() else c for c in rearranged)
            remainder = int(numeric) % 97
            return remainder == 1
        except:
            return False
    
    def _parse_date(self, value: Any) -> Optional[str]:
        """Parse date to YYYY-MM-DD format."""
        if not value:
            return None
        value_str = str(value).strip()
        # Try common formats
        patterns = [
            (r'(\d{4})-(\d{2})-(\d{2})', lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
            (r'(\d{2})-(\d{2})-(\d{4})', lambda m: f"{m.group(3)}-{m.group(2)}-{m.group(1)}"),
            (r'(\d{2})/(\d{2})/(\d{4})', lambda m: f"{m.group(3)}-{m.group(2)}-{m.group(1)}"),
        ]
        for pattern, formatter in patterns:
            match = re.match(pattern, value_str)
            if match:
                try:
                    result = formatter(match)
                    # Validate it's a real date
                    from datetime import datetime
                    datetime.strptime(result, "%Y-%m-%d")
                    return result
                except:
                    pass
        return None
    
    def _parse_amount(self, value: Any) -> Optional[str]:
        """Parse amount to Decimal string."""
        if not value:
            return None
        value_str = str(value).strip()
        # Remove currency symbols and spaces
        value_str = re.sub(r'[€$£\s]', '', value_str)
        # Replace comma with dot for decimal
        value_str = value_str.replace(',', '.')
        # Extract number
        match = re.search(r'(\d+\.?\d*)', value_str)
        if match:
            try:
                float_val = float(match.group(1))
                return str(float_val)
            except:
                pass
        return None
    
    def _validate_hard_validators(self, data: Dict[str, Any], fields: List[Tuple]) -> List[str]:
        """Apply hard validators (IBAN, date, amount) and return errors."""
        errors = []
        for key, label, field_type, is_required, enum_values, regex in fields:
            value = data.get(key)
            if value is None:
                continue
            
            key_lower = key.lower()
            label_lower = label.lower()
            
            # IBAN validation
            if "iban" in key_lower or field_type == "iban":
                if not self._validate_iban(str(value)):
                    errors.append(f"invalid_iban:{key}")
            
            # Date validation
            if "date" in key_lower or field_type == "date":
                if not self._parse_date(value):
                    errors.append(f"invalid_date:{key}")
            
            # Amount validation
            if "amount" in key_lower or field_type in ("money", "currency"):
                if not self._parse_amount(value):
                    errors.append(f"invalid_amount:{key}")
        
        return errors
    
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
                # Skip validation if page/start/end are None
                if span.page is None or span.start is None or span.end is None:
                    errors.append(f"{field_key}: Missing page/start/end in evidence span")
                    continue
                
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

    def _load_semantic_context(self, document_dir: Path) -> Optional[Dict[str, Any]]:
        """Load BERT classifier output as assistive semantic context."""
        classification_file = document_dir / "llm" / "classification_local.json"
        if not classification_file.exists():
            return None

        try:
            classification_data = json.loads(classification_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug(f"Could not load semantic context: {e}")
            return None

        bert_data = classification_data.get("bert")
        if not isinstance(bert_data, dict):
            return None

        all_scores = bert_data.get("all_scores") or {}
        if isinstance(all_scores, dict) and all_scores:
            sorted_scores = sorted(
                ((label, float(score)) for label, score in all_scores.items()),
                key=lambda item: item[1],
                reverse=True,
            )
        elif bert_data.get("label") and bert_data.get("confidence") is not None:
            sorted_scores = [(bert_data["label"], float(bert_data["confidence"]))]
        else:
            return None

        top_matches = [
            {"label": label, "confidence": score}
            for label, score in sorted_scores[:3]
        ]
        margin = 0.0
        if len(sorted_scores) > 1:
            margin = sorted_scores[0][1] - sorted_scores[1][1]

        return {
            "source": "bert_embeddings",
            "role": "semantic_context",
            "model_used": bert_data.get("model_used") or self.model_name or "default",
            "status": bert_data.get("status", "available"),
            "top_matches": top_matches,
            "confidence": top_matches[0]["confidence"],
            "margin": margin,
            "selected_for_classification": classification_data.get("method") == "bert",
            "summary": f"Inhoud lijkt het meest op '{top_matches[0]['label']}' ({top_matches[0]['confidence'] * 100:.1f}%).",
        }

    async def _stage_unified_fraud_analysis(
        self,
        document: Document,
        document_dir: Path,
        ocr_result: OCRResult,
        classification: ClassificationResult,
        extraction_result: Optional[ExtractionEvidence],
    ) -> RiskAnalysis:
        """Run the canonical fraud analyzer and persist compatible artifacts."""
        from app.services.fraud_detector import fraud_detector

        original_path = document_dir / "original" / document.original_filename
        file_bytes = original_path.read_bytes() if original_path.exists() else None
        semantic_context = self._load_semantic_context(document_dir)

        detector = fraud_detector(llm_client=self.llm)
        report = await detector.analyze_document(
            file_bytes=file_bytes,
            filename=document.original_filename,
            document_id=document.id,
            document_dir=document_dir,
            extracted_text=ocr_result.combined_text,
            classification_confidence=classification.confidence,
            classification_label=classification.doc_type_slug,
            semantic_context=semantic_context,
        )

        report_dict = report.to_dict()
        fraud_cache_path = document_dir / "fraud_analysis.json"
        fraud_cache_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")

        risk_dict = report.to_risk_analysis_dict()
        risk_dir = document_dir / "risk"
        risk_dir.mkdir(exist_ok=True)
        with open(risk_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(risk_dict, f, indent=2, ensure_ascii=False)

        risk_signals = [
            RiskSignal(
                code=signal["code"],
                severity=signal["severity"],
                message=signal["message"],
                evidence=signal["evidence"],
                examples={
                    "category": signal.get("category"),
                    "confidence": signal.get("confidence"),
                    "recommendation": signal.get("recommendation"),
                    "details": signal.get("details", {}),
                },
            )
            for signal in risk_dict["signals"]
        ]

        if risk_signals:
            logger.info(f"Document {document.id} fraud analysis: score={risk_dict['risk_score']}, {len(risk_signals)} signal(s) found")
        else:
            logger.info(f"Document {document.id} fraud analysis: score={risk_dict['risk_score']}, no signals found")

        return RiskAnalysis(risk_score=risk_dict["risk_score"], signals=risk_signals)

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
        
        # Log risk analysis summary
        if signals:
            logger.info(f"Document {document.id} risk analysis: score={risk_score}, {len(signals)} signal(s) found")
        else:
            logger.info(f"Document {document.id} risk analysis: score={risk_score}, no signals found (document appears clean)")

        # Save artifact
        risk_dir = document_dir / "risk"
        risk_dir.mkdir(exist_ok=True)

        with open(risk_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(analysis.dict(), f, indent=2, ensure_ascii=False)

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