"""
Fraud Detection Service

Consolidates all fraud detection signals from document analysis:
- PDF metadata analysis (creation software, timestamps)
- Image forensics (ELA, copy-move detection)
- Text anomalies (unicode, repetition patterns)
- Classification confidence analysis
"""

import io
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageChops, ImageDraw, ImageFont
import numpy as np

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FraudSignal:
    """A single fraud detection signal."""
    name: str
    description: str
    risk_level: RiskLevel
    confidence: float  # 0.0 - 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)


@dataclass
class FraudReport:
    """Complete fraud analysis report for a document."""
    document_id: Optional[int]
    filename: str
    overall_risk: RiskLevel
    risk_score: float  # 0.0 - 100.0
    signals: List[FraudSignal]
    summary: str
    analyzed_at: str
    
    def _convert_numpy_types(self, obj: Any) -> Any:
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "overall_risk": self.overall_risk.value,
            "risk_score": float(self.risk_score),
            "signals": [
                {
                    "name": s.name,
                    "description": s.description,
                    "risk_level": s.risk_level.value,
                    "confidence": float(s.confidence),
                    "details": self._convert_numpy_types(s.details),
                    "evidence": s.evidence,
                }
                for s in self.signals
            ],
            "summary": self.summary,
            "analyzed_at": self.analyzed_at,
        }


class FraudDetector:
    """
    Analyzes documents for potential fraud indicators.
    
    Combines multiple detection methods:
    1. PDF metadata analysis
    2. Image forensics (ELA)
    3. Text anomaly detection
    4. Classification confidence analysis
    """
    
    # Known suspicious PDF creators
    SUSPICIOUS_CREATORS = {
        "fpdf": "FPDF library - vaak gebruikt voor gegenereerde facturen",
        "tcpdf": "TCPDF library - vaak gebruikt voor gegenereerde documenten", 
        "wkhtmltopdf": "HTML naar PDF converter - kan legitiem zijn",
        "phantomjs": "Headless browser - vaak voor automatische generatie",
        "puppeteer": "Chrome automation - vaak voor automatische generatie",
    }
    
    # Suspicious Unicode ranges
    SUSPICIOUS_UNICODE_RANGES = [
        (0x200B, 0x200F, "Zero-width karakters"),
        (0x2028, 0x2029, "Line/paragraph separators"),
        (0xFEFF, 0xFEFF, "Byte order mark in tekst"),
        (0x00AD, 0x00AD, "Soft hyphen (onzichtbaar)"),
        (0x034F, 0x034F, "Combining grapheme joiner"),
        (0x2060, 0x2064, "Word joiners"),
        (0x180E, 0x180E, "Mongolian vowel separator"),
        (0x3164, 0x3164, "Hangul filler"),
    ]
    
    def __init__(self, llm_client=None, ela_min_size: Optional[int] = None, 
                 ela_allow_non_jpeg: Optional[bool] = None, 
                 ela_scale_for_heatmap: Optional[int] = None,
                 ela_quality: Optional[int] = None):
        self.llm_client = llm_client
        from app.config import settings
        self.ela_min_size = ela_min_size if ela_min_size is not None else settings.ela_min_size
        self.ela_allow_non_jpeg = ela_allow_non_jpeg if ela_allow_non_jpeg is not None else settings.ela_allow_non_jpeg
        self.ela_scale_for_heatmap = ela_scale_for_heatmap if ela_scale_for_heatmap is not None else settings.ela_scale_for_heatmap
        self.ela_quality = ela_quality if ela_quality is not None else settings.ela_quality
    
    async def analyze_document(
        self,
        file_path: Optional[Path] = None,
        file_bytes: Optional[bytes] = None,
        filename: str = "",
        document_id: Optional[int] = None,
        document_dir: Optional[Path] = None,
        extracted_text: Optional[str] = None,
        classification_confidence: Optional[float] = None,
        existing_signals: Optional[Dict[str, Any]] = None,
    ) -> FraudReport:
        """
        Analyze a document for fraud indicators.
        
        Args:
            file_path: Path to the document file
            file_bytes: Raw bytes of the document
            filename: Original filename
            document_id: Database ID if available
            extracted_text: Pre-extracted text content
            classification_confidence: Classification confidence score
            existing_signals: Existing fraud signals from processing
        
        Returns:
            FraudReport with all detected signals
        """
        signals: List[FraudSignal] = []
        
        # Load file if needed
        if file_bytes is None and file_path:
            file_bytes = file_path.read_bytes()
        
        if not filename and file_path:
            filename = file_path.name
        
        # Detect file type
        is_pdf = filename.lower().endswith('.pdf') or (file_bytes and file_bytes[:4] == b'%PDF')
        is_image = filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))
        
        # 1. PDF Metadata Analysis
        if is_pdf and file_bytes:
            pdf_signals = self._analyze_pdf_metadata(file_bytes)
            signals.extend(pdf_signals)
        
        # 2. Image Forensics (ELA)
        ela_heatmap_path = None
        if document_dir:
            ela_heatmap_path = document_dir / "risk" / "ela_heatmap.png"
            logger.info(f"ELA heatmap will be saved to: {ela_heatmap_path}")
        
        if is_image and file_bytes:
            image_signals = self._analyze_image_forensics(file_bytes, filename, ela_heatmap_path)
            signals.extend(image_signals)
            
            # If we have ELA signals but no heatmap was saved, try to regenerate
            ela_signals = [s for s in image_signals if s.name == "ela_manipulation_detected"]
            if ela_signals and ela_heatmap_path and not ela_heatmap_path.exists():
                logger.warning("ELA signal detected but heatmap not saved, attempting to regenerate...")
                try:
                    img = Image.open(io.BytesIO(file_bytes))
                    original_format = img.format or "UNKNOWN"
                    _, heatmap = self._perform_ela(
                        img,
                        original_format=original_format,
                        save_heatmap_path=ela_heatmap_path,
                        allow_non_jpeg=self.ela_allow_non_jpeg
                    )
                    if heatmap:
                        logger.info("Regenerated ELA heatmap after signal detection for image")
                except Exception as e:
                    logger.warning(f"Failed to regenerate ELA heatmap: {e}")
        
        # 3. Extract images from PDF for analysis
        if is_pdf and file_bytes:
            pdf_image_signals = self._analyze_pdf_images(file_bytes, ela_heatmap_path)
            signals.extend(pdf_image_signals)
            
            # If we have ELA signals but no heatmap was saved, try to generate one
            # Note: _analyze_pdf_images already handles heatmap generation, so this is a fallback
            ela_signals = [s for s in pdf_image_signals if s.name == "ela_manipulation_detected"]
            if ela_signals and ela_heatmap_path and not ela_heatmap_path.exists():
                logger.warning("ELA signal detected but heatmap not saved, attempting to regenerate...")
                try:
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    for page_num in range(min(5, len(doc))):
                        page = doc[page_num]
                        image_list = page.get_images()
                        if image_list:
                            xref = image_list[0][0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            pil_img = Image.open(io.BytesIO(image_bytes))
                            original_format = pil_img.format or base_image.get("ext", "UNKNOWN").lstrip('.').upper()
                            _, heatmap = self._perform_ela(
                                pil_img,
                                original_format=original_format,
                                save_heatmap_path=ela_heatmap_path,
                                allow_non_jpeg=self.ela_allow_non_jpeg
                            )
                            if heatmap:
                                logger.info("Regenerated ELA heatmap after signal detection for PDF")
                                break
                    doc.close()
                except Exception as e:
                    logger.warning(f"Failed to regenerate ELA heatmap: {e}")
        
        # 4. Text Anomaly Detection
        if extracted_text:
            text_signals = self._analyze_text_anomalies(extracted_text)
            signals.extend(text_signals)
        
        # 5. Classification Confidence Analysis - DISABLED
        # Low classification confidence is not considered a fraud indicator
        # if classification_confidence is not None:
        #     conf_signals = self._analyze_classification_confidence(classification_confidence)
        #     signals.extend(conf_signals)
        
        # 6. Include existing signals from document processing
        if existing_signals:
            proc_signals = self._parse_existing_signals(existing_signals)
            signals.extend(proc_signals)
        
        # 7. Enhance signals with LLM-generated user-friendly descriptions
        if self.llm_client and signals:
            try:
                signals = await self._enhance_signals_with_llm(signals, extracted_text or "")
            except Exception as e:
                logger.warning(f"Failed to enhance signals with LLM: {e}, using original signals")
        
        # Calculate overall risk
        risk_score, overall_risk = self._calculate_overall_risk(signals)
        
        # Generate summary
        summary = self._generate_summary(signals, overall_risk)
        
        return FraudReport(
            document_id=document_id,
            filename=filename,
            overall_risk=overall_risk,
            risk_score=risk_score,
            signals=signals,
            summary=summary,
            analyzed_at=datetime.utcnow().isoformat(),
        )
    
    def _analyze_pdf_metadata(self, pdf_bytes: bytes) -> List[FraudSignal]:
        """Analyze PDF metadata for suspicious indicators."""
        signals = []
        
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            metadata = doc.metadata
            
            # Check creator/producer
            creator = (metadata.get("creator") or "").lower()
            producer = (metadata.get("producer") or "").lower()
            
            for suspicious, description in self.SUSPICIOUS_CREATORS.items():
                if suspicious in creator or suspicious in producer:
                    signals.append(FraudSignal(
                        name="suspicious_pdf_creator",
                        description=f"PDF gemaakt met verdachte software: {description}",
                        risk_level=RiskLevel.MEDIUM,
                        confidence=0.8,
                        details={
                            "creator": metadata.get("creator"),
                            "producer": metadata.get("producer"),
                            "matched": suspicious,
                        },
                        evidence=[f"Creator: {metadata.get('creator')}", f"Producer: {metadata.get('producer')}"],
                    ))
                    break
            
            # Check creation/modification dates
            creation_date = metadata.get("creationDate")
            mod_date = metadata.get("modDate")
            
            if creation_date and mod_date:
                # Parse dates (format: D:YYYYMMDDHHmmSS)
                try:
                    if creation_date != mod_date:
                        signals.append(FraudSignal(
                            name="pdf_date_mismatch",
                            description="Creatiedatum en wijzigingsdatum verschillen",
                            risk_level=RiskLevel.LOW,
                            confidence=0.5,
                            details={
                                "creation_date": creation_date,
                                "modification_date": mod_date,
                            },
                            evidence=[f"Aangemaakt: {creation_date}", f"Gewijzigd: {mod_date}"],
                        ))
                except Exception:
                    pass
            
            # Check for missing metadata (often sign of generated PDFs)
            if not metadata.get("author") and not metadata.get("title"):
                signals.append(FraudSignal(
                    name="missing_metadata",
                    description="PDF mist auteur en titel metadata",
                    risk_level=RiskLevel.LOW,
                    confidence=0.4,
                    details={"metadata": metadata},
                    evidence=["Geen auteur", "Geen titel"],
                ))
            
            doc.close()
            
        except Exception as e:
            logger.warning(f"PDF metadata analysis failed: {e}")
        
        return signals
    
    def _analyze_image_forensics(self, image_bytes: bytes, filename: str, ela_heatmap_path: Optional[Path] = None) -> List[FraudSignal]:
        """Perform Error Level Analysis (ELA) on images."""
        signals = []
        
        try:
            # Load image
            img = Image.open(io.BytesIO(image_bytes))
            original_format = img.format or "UNKNOWN"
            
            # Perform ELA with format awareness
            ela_result, ela_heatmap = self._perform_ela(
                img, 
                original_format=original_format,
                save_heatmap_path=ela_heatmap_path,
                allow_non_jpeg=self.ela_allow_non_jpeg
            )
            
            if ela_result:
                signals.append(ela_result)
            
            # Check EXIF data
            exif_signals = self._analyze_exif(img)
            signals.extend(exif_signals)
            
        except Exception as e:
            logger.warning(f"Image forensics failed: {e}", exc_info=True)
        
        return signals
    
    def _perform_ela(self, img: Image.Image, original_format: Optional[str] = None, 
                     quality: Optional[int] = None, save_heatmap_path: Optional[Path] = None,
                     allow_non_jpeg: Optional[bool] = None) -> Tuple[Optional[FraudSignal], Optional[Image.Image]]:
        """
        Error Level Analysis - detects areas that have been modified.
        
        Refactored to:
        - Calculate stats on UN-SCALED diff (for reliable detection)
        - Create heatmap only for visualization (scaled+clipped)
        - Use numpy operations instead of pixel loops
        - Support format gating (JPEG only by default)
        
        Args:
            img: PIL Image to analyze
            original_format: Original image format (JPEG, PNG, etc.)
            quality: JPEG quality for re-saving (defaults to self.ela_quality)
            save_heatmap_path: Optional path to save heatmap PNG
            allow_non_jpeg: Override allow_non_jpeg setting (defaults to self.ela_allow_non_jpeg)
        
        Returns:
            Tuple of (FraudSignal if manipulation detected, ELA heatmap image)
        """
        heatmap = None
        signal = None
        skipped_reason = None
        
        try:
            width, height = img.size
            original_format = original_format or img.format or "UNKNOWN"
            quality = quality if quality is not None else self.ela_quality
            allow_non_jpeg = allow_non_jpeg if allow_non_jpeg is not None else self.ela_allow_non_jpeg
            
            # Size gating: skip small images
            if width < self.ela_min_size or height < self.ela_min_size:
                skipped_reason = f"small_image_{width}x{height}_below_min_{self.ela_min_size}"
                logger.debug(f"Skipping ELA for small image ({width}x{height} pixels, minimum is {self.ela_min_size}x{self.ela_min_size}) - likely a logo or icon")
                # Create placeholder heatmap if path provided (for UI compatibility)
                if save_heatmap_path:
                    self._create_placeholder_heatmap(save_heatmap_path, f"ELA skipped: image too small ({width}x{height} < {self.ela_min_size}x{self.ela_min_size})")
                return None, None
            
            # Format gating: only analyze JPEG unless explicitly allowed
            is_jpeg = original_format.upper() in ('JPEG', 'JPG')
            if not is_jpeg and not allow_non_jpeg:
                skipped_reason = f"non_jpeg_format_{original_format.lower()}"
                logger.debug(f"Skipping ELA for non-JPEG format: {original_format} (allow_non_jpeg={allow_non_jpeg})")
                # Create placeholder heatmap if path provided (for UI compatibility)
                if save_heatmap_path:
                    self._create_placeholder_heatmap(save_heatmap_path, f"ELA not applicable: {original_format} format (JPEG required)")
                return None, None
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save original to temporary buffer and reload (simulates re-compression)
            temp_buffer = io.BytesIO()
            img.save(temp_buffer, 'JPEG', quality=quality)
            temp_buffer.seek(0)
            resaved = Image.open(temp_buffer)
            
            # Calculate difference (UN-SCALED for stats)
            diff_unscaled = ImageChops.difference(img, resaved)
            
            # Convert to numpy array for analysis (UN-SCALED)
            diff_array_unscaled = np.array(diff_unscaled, dtype=np.float32)
            
            # Calculate stats on UN-SCALED diff (reliable, not affected by visualization scaling)
            mean_error = float(np.mean(diff_array_unscaled))
            max_error = float(np.max(diff_array_unscaled))
            std_error = float(np.std(diff_array_unscaled))
            
            # Create heatmap for visualization (SCALED + CLIPPED)
            # Use numpy for performance instead of pixel loop
            diff_array_scaled = diff_array_unscaled * self.ela_scale_for_heatmap
            diff_array_clipped = np.clip(diff_array_scaled, 0, 255).astype(np.uint8)
            heatmap = Image.fromarray(diff_array_clipped, mode='RGB')
            
            # Analyze heatmap for visible differences (bright pixels indicate manipulation)
            # This helps catch cases where std_error/max_error are low but visual differences are clear
            heatmap_array = np.array(heatmap)
            # Check for bright pixels in the heatmap (scaled values > 50 indicate visible differences)
            bright_pixels = np.sum(heatmap_array > 50)
            total_pixels = heatmap_array.size
            bright_ratio = bright_pixels / total_pixels if total_pixels > 0 else 0
            
            # Detection thresholds based on UN-SCALED std_error and max_error
            # These are stable and not affected by visualization scaling
            # Typical values for unscaled diff:
            # - Normal JPEG compression: 2-8
            # - Light manipulation: 8-15
            # - Clear manipulation: 15-30+
            # - Heavy manipulation: 30+
            # Also check max_error for localized manipulation (e.g., scanned documents)
            # Scanned documents may have lower std_error but high max_error in specific areas
            threshold_std = 2.0  # Very low threshold to catch subtle manipulation
            threshold_max = 10.0  # Lower threshold for localized manipulation
            threshold_bright_ratio = 0.01  # 1% of pixels should be bright to indicate manipulation
            
            # Generate signal if any threshold is exceeded OR if heatmap shows visible differences
            has_visible_differences = bright_ratio > threshold_bright_ratio
            exceeds_statistical_threshold = std_error > threshold_std or max_error > threshold_max
            
            if exceeds_statistical_threshold or has_visible_differences:
                # Determine confidence and risk level
                # Use both std_error and max_error for better detection
                std_factor = max(0, (std_error - threshold_std) / 20) if std_error > threshold_std else 0
                max_factor = max(0, (max_error - threshold_max) / 40) if max_error > threshold_max else 0
                # Also consider visible differences in heatmap
                bright_factor = min(1.0, bright_ratio / 0.1) if has_visible_differences else 0  # Scale to 0.1 (10% bright pixels = max)
                # Combine factors - visible differences are strong indicator
                confidence = float(max(0.5, min(0.95, max(std_factor, max_factor * 0.8, bright_factor * 0.7))))
                
                # Risk level based on severity - always at least MEDIUM if detected
                if std_error > 20 or max_error > 50:
                    risk_level = RiskLevel.HIGH
                elif std_error > 12 or max_error > 30:
                    risk_level = RiskLevel.MEDIUM
                else:
                    risk_level = RiskLevel.MEDIUM  # Even low values are suspicious if detected - never LOW
                
                logger.info(f"ELA manipulation detected: std_error={std_error:.2f}, max_error={max_error:.2f}, bright_ratio={bright_ratio:.4f}, risk={risk_level.value}, confidence={confidence:.2f}")
                
                # Context based on image size
                # Don't downgrade to LOW - even small manipulated images are suspicious
                is_small_image = width < 300 or height < 300
                if is_small_image:
                    confidence = confidence * 0.8  # Slightly reduce confidence but keep risk level
                    # Keep minimum MEDIUM risk level - manipulation is still suspicious even in small images
                    if risk_level == RiskLevel.LOW:
                        risk_level = RiskLevel.MEDIUM
                
                # Description will be enhanced by LLM later, but keep basic structure
                # Check for high max_error (localized manipulation, e.g., scanned documents)
                has_localized_manipulation = max_error > threshold_max
                
                if std_error > 20 or max_error > 50:
                    if is_small_image:
                        description = "Een afbeelding in dit document lijkt bewerkt. Let op: dit is een kleine afbeelding, mogelijk een logo."
                    else:
                        if has_localized_manipulation:
                            description = "Een afbeelding in dit document is waarschijnlijk bewerkt. Er zijn duidelijke compressieverschillen in specifieke gebieden."
                        else:
                            description = "Een afbeelding in dit document is waarschijnlijk bewerkt. Sommige delen zijn anders dan de rest."
                    evidence_text = "Er zijn duidelijke verschillen gevonden tussen delen van de afbeelding. Dit wijst op mogelijke bewerking of het plakken van elementen."
                elif std_error > 12 or max_error > 30:
                    if is_small_image:
                        description = "Een kleine afbeelding in dit document toont tekenen van bewerking. Dit kan een logo of icoon zijn."
                    else:
                        if has_localized_manipulation:
                            description = "Een afbeelding in dit document toont tekenen van mogelijke bewerking. Er zijn compressieverschillen in bepaalde gebieden."
                        else:
                            description = "Een afbeelding in dit document toont tekenen van mogelijke bewerking."
                    evidence_text = "Er zijn verschillen gevonden in de compressie van de afbeelding. Dit kan wijzen op bewerking, maar is niet altijd verdacht."
                else:
                    if is_small_image:
                        description = "Een kleine afbeelding (mogelijk een logo) toont lichte compressieverschillen."
                    else:
                        if has_localized_manipulation:
                            description = "Een afbeelding in dit document toont compressieverschillen in specifieke gebieden. Dit kan wijzen op bewerking of scanning."
                        else:
                            description = "Een afbeelding in dit document toont lichte compressieverschillen."
                    evidence_text = "Er zijn kleine technische verschillen gevonden. Dit is vaak normaal bij afbeeldingen die zijn bewerkt of meerdere keren opgeslagen."
                
                # Add size context
                if is_small_image:
                    size_info = f"Dit betreft een kleine afbeelding van {width}×{height} pixels (mogelijk een logo of icoon). ELA-resultaten zijn bij kleine afbeeldingen minder betrouwbaar."
                else:
                    size_info = f"Afbeeldingsgrootte: {width}×{height} pixels."
                
                # Add evidence about max_error if it triggered detection
                evidence_list = [evidence_text, size_info]
                if max_error > threshold_max and std_error <= threshold_std:
                    evidence_list.append(f"Lokale compressieverschillen gedetecteerd (max: {max_error:.1f}). Dit kan wijzen op bewerking in specifieke gebieden.")
                elif max_error > threshold_max:
                    evidence_list.append(f"Zowel algemene als lokale compressieverschillen gedetecteerd (std: {std_error:.1f}, max: {max_error:.1f}).")
                
                signal = FraudSignal(
                    name="ela_manipulation_detected",
                    description=description,
                    risk_level=risk_level,
                    confidence=confidence,
                    details={
                        "mean_error": mean_error,
                        "max_error": max_error,
                        "std_error": std_error,
                        "bright_pixel_ratio": bright_ratio,
                        "image_size": f"{width}x{height}",
                        "is_small_image": is_small_image,
                        "original_format": original_format,
                        "quality": quality,
                        "scale_used": self.ela_scale_for_heatmap,
                        "min_size_gate": self.ela_min_size,
                    },
                    evidence=evidence_list,
                )
                logger.info(f"ELA signal detected: std_error={std_error:.2f}, risk={risk_level.value}, confidence={confidence:.2f}, format={original_format}, size={width}x{height}")
            
            # Save heatmap if path provided
            if save_heatmap_path and heatmap:
                try:
                    save_heatmap_path.parent.mkdir(parents=True, exist_ok=True)
                    heatmap.save(save_heatmap_path, 'PNG')
                    signal_status = "signal_detected" if signal else "no_signal"
                    logger.info(f"ELA heatmap saved to {save_heatmap_path} (std_error={std_error:.2f}, format={original_format}, {signal_status})")
                except Exception as e:
                    logger.error(f"Failed to save ELA heatmap to {save_heatmap_path}: {e}", exc_info=True)
            
            # Log if no signal was generated (for debugging)
            if not signal and std_error > 0:
                logger.warning(f"ELA analysis completed but no signal generated: std_error={std_error:.2f} (threshold: {threshold_std}), max_error={max_error:.2f} (threshold: {threshold_max}), bright_ratio={bright_ratio:.4f} (threshold: {threshold_bright_ratio}), format={original_format}, size={width}x{height}")
                # If heatmap was created but no signal, this might indicate a false negative
                if save_heatmap_path and save_heatmap_path.exists():
                    logger.warning(f"ELA heatmap exists but no signal was generated - possible false negative (bright_ratio={bright_ratio:.4f})")
            
        except Exception as e:
            logger.warning(f"ELA analysis failed: {e}", exc_info=True)
            if save_heatmap_path:
                logger.warning(f"ELA heatmap could not be generated (path was: {save_heatmap_path})")
                try:
                    self._create_placeholder_heatmap(save_heatmap_path, f"ELA analysis failed: {str(e)[:50]}")
                except Exception as placeholder_error:
                    logger.error(f"Failed to create placeholder heatmap: {placeholder_error}")
        
        return signal, heatmap
    
    def _create_placeholder_heatmap(self, path: Path, message: str) -> None:
        """Create a placeholder PNG heatmap with text message for UI compatibility."""
        try:
            # Create a simple gray image with text
            img = Image.new('RGB', (400, 200), color=(64, 64, 64))
            draw = ImageDraw.Draw(img)
            
            # Try to use a default font, fallback to basic if not available
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            # Draw text (centered)
            text = message[:60]  # Truncate long messages
            bbox = draw.textbbox((0, 0), text, font=font) if font else (0, 0, len(text) * 6, 16)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((400 - text_width) // 2, (200 - text_height) // 2)
            
            draw.text(position, text, fill=(200, 200, 200), font=font)
            
            path.parent.mkdir(parents=True, exist_ok=True)
            img.save(path, 'PNG')
            logger.debug(f"Created placeholder heatmap at {path}")
        except Exception as e:
            logger.warning(f"Failed to create placeholder heatmap: {e}")
    
    def _analyze_exif(self, img: Image.Image) -> List[FraudSignal]:
        """Analyze EXIF metadata for suspicious indicators."""
        signals = []
        
        try:
            exif = img._getexif()
            if not exif:
                return signals
            
            # Check for software manipulation indicators
            software_tags = [305, 11]  # Software, ProcessingSoftware
            for tag in software_tags:
                if tag in exif:
                    software = str(exif[tag]).lower()
                    if any(s in software for s in ['photoshop', 'gimp', 'paint']):
                        signals.append(FraudSignal(
                            name="image_editing_software",
                            description=f"Afbeelding bewerkt met: {exif[tag]}",
                            risk_level=RiskLevel.MEDIUM,
                            confidence=0.7,
                            details={"software": exif[tag]},
                            evidence=[f"Bewerkt met: {exif[tag]}"],
                        ))
            
        except Exception as e:
            logger.debug(f"EXIF analysis failed: {e}")
        
        return signals
    
    def _analyze_pdf_images(self, pdf_bytes: bytes, ela_heatmap_path: Optional[Path] = None) -> List[FraudSignal]:
        """
        Extract and analyze images embedded in PDF.
        
        Refactored to:
        - Select largest JPEG image above min_size threshold
        - Fix ext-check (PyMuPDF gives "jpg"/"png" without dot)
        - Run ELA only on selected candidate
        - Add detailed metadata (page, image_index, xref, ext, size_px)
        """
        signals = []
        
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            # Collect all candidate images with metadata
            candidates = []
            
            for page_num in range(min(5, len(doc))):  # Check first 5 pages
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Load PIL image to get size
                        pil_img = Image.open(io.BytesIO(image_bytes))
                        width, height = pil_img.size
                        size_px = width * height
                        
                        # Fix ext-check: PyMuPDF gives "jpg"/"png" without dot
                        ext_raw = base_image.get("ext", "").lower()
                        ext = ext_raw if ext_raw.startswith('.') else f".{ext_raw}" if ext_raw else ""
                        
                        # Check format
                        is_jpeg = (pil_img.format == 'JPEG' or 
                                  ext in ('.jpg', '.jpeg') or
                                  ext_raw in ('jpg', 'jpeg'))
                        
                        # Only consider images above min_size
                        if width >= self.ela_min_size and height >= self.ela_min_size:
                            candidates.append({
                                'page': page_num + 1,
                                'image_index': img_index,
                                'xref': xref,
                                'ext': ext or ext_raw or 'unknown',
                                'size_px': size_px,
                                'width': width,
                                'height': height,
                                'is_jpeg': is_jpeg,
                                'pil_img': pil_img,
                                'image_bytes': image_bytes,
                            })
                        else:
                            logger.debug(f"Skipping PDF image {page_num + 1}/{img_index}: too small ({width}x{height} < {self.ela_min_size})")
                    except Exception as e:
                        logger.debug(f"Failed to extract PDF image metadata: {e}")
            
            doc.close()
            
            if not candidates:
                logger.debug("No PDF images found above minimum size threshold")
                if ela_heatmap_path:
                    self._create_placeholder_heatmap(ela_heatmap_path, "No suitable images found in PDF")
                return signals
            
            # Select best candidate: prefer largest JPEG, otherwise largest image
            jpeg_candidates = [c for c in candidates if c['is_jpeg']]
            if jpeg_candidates:
                selected = max(jpeg_candidates, key=lambda c: c['size_px'])
                logger.info(f"Selected JPEG image from PDF: page {selected['page']}, size {selected['width']}x{selected['height']} ({selected['size_px']} px)")
            else:
                selected = max(candidates, key=lambda c: c['size_px'])
                logger.info(f"Selected largest image from PDF (non-JPEG): page {selected['page']}, format {selected['ext']}, size {selected['width']}x{selected['height']} ({selected['size_px']} px)")
            
            # Perform ELA on selected candidate
            original_format = selected['pil_img'].format or selected['ext'].lstrip('.').upper() or "UNKNOWN"
            ela_signal, ela_heatmap = self._perform_ela(
                selected['pil_img'],
                original_format=original_format,
                save_heatmap_path=ela_heatmap_path,
                allow_non_jpeg=self.ela_allow_non_jpeg
            )
            
            if ela_signal:
                # Add PDF-specific metadata
                ela_signal.details.update({
                    "page": selected['page'],
                    "image_index": selected['image_index'],
                    "xref": selected['xref'],
                    "ext": selected['ext'],
                    "size_px": selected['size_px'],
                })
                signals.append(ela_signal)
                logger.info(f"ELA signal detected for PDF image: page {selected['page']}, std_error={ela_signal.details.get('std_error', 'N/A'):.2f}")
            
        except Exception as e:
            logger.warning(f"PDF image analysis failed: {e}", exc_info=True)
        
        return signals
    
    def _analyze_text_anomalies(self, text: str) -> List[FraudSignal]:
        """Detect suspicious text patterns and unicode anomalies."""
        signals = []
        
        # Check for suspicious unicode characters
        suspicious_chars = []
        for start, end, description in self.SUSPICIOUS_UNICODE_RANGES:
            for char in text:
                if start <= ord(char) <= end:
                    suspicious_chars.append((char, hex(ord(char)), description))
        
        if suspicious_chars:
            unique_types = set(desc for _, _, desc in suspicious_chars)
            signals.append(FraudSignal(
                name="unicode_anomalies",
                description=f"Verdachte unicode karakters gedetecteerd ({len(suspicious_chars)} stuks)",
                risk_level=RiskLevel.HIGH if len(suspicious_chars) > 5 else RiskLevel.MEDIUM,
                confidence=0.85,
                details={
                    "count": len(suspicious_chars),
                    "types": list(unique_types),
                    "examples": [(hex(ord(c)), d) for c, h, d in suspicious_chars[:5]],
                },
                evidence=[f"{len(suspicious_chars)} verdachte karakters", f"Types: {', '.join(unique_types)}"],
            ))
        
        # Check for excessive repetition
        words = re.findall(r'\b\w{4,}\b', text.lower())
        if words:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Find words repeated more than expected
            total_words = len(words)
            suspicious_repeats = [
                (word, count) for word, count in word_counts.items()
                if count > 10 and count / total_words > 0.05  # More than 5% of text
            ]
            
            if suspicious_repeats:
                signals.append(FraudSignal(
                    name="text_repetition",
                    description="Onnatuurlijke tekstherhaling gedetecteerd",
                    risk_level=RiskLevel.MEDIUM,
                    confidence=0.6,
                    details={
                        "repeated_words": suspicious_repeats[:10],
                        "total_words": total_words,
                    },
                    evidence=[f"'{word}' herhaald {count}x" for word, count in suspicious_repeats[:3]],
                ))
        
        return signals
    
    def _analyze_classification_confidence(self, confidence: float) -> List[FraudSignal]:
        """Analyze classification confidence as fraud indicator."""
        signals = []
        
        if confidence < 0.5:
            signals.append(FraudSignal(
                name="low_classification_confidence",
                description="Document past niet bij bekende document types",
                risk_level=RiskLevel.MEDIUM,
                confidence=1.0 - confidence,
                details={"classification_confidence": confidence},
                evidence=[f"Classificatie zekerheid: {confidence*100:.1f}%"],
            ))
        
        return signals
    
    def _parse_existing_signals(self, existing: Dict[str, Any]) -> List[FraudSignal]:
        """Parse existing fraud signals from document processing."""
        signals = []
        
        # Check for fpdf flag
        if existing.get("fpdf"):
            signals.append(FraudSignal(
                name="fpdf_detected",
                description="Document gegenereerd met FPDF library",
                risk_level=RiskLevel.MEDIUM,
                confidence=0.9,
                details=existing.get("fpdf_details", {}),
                evidence=["FPDF signature gedetecteerd"],
            ))
        
        # Check for existing anomalies
        if existing.get("unicode_anomalies"):
            anomalies = existing["unicode_anomalies"]
            signals.append(FraudSignal(
                name="unicode_anomalies_processed",
                description=f"Unicode afwijkingen uit verwerking: {len(anomalies)} stuks",
                risk_level=RiskLevel.MEDIUM,
                confidence=0.8,
                details={"anomalies": anomalies},
                evidence=[str(a) for a in anomalies[:3]],
            ))
        
        if existing.get("repetition_patterns"):
            patterns = existing["repetition_patterns"]
            signals.append(FraudSignal(
                name="repetition_patterns_processed",
                description=f"Herhalingspatronen uit verwerking: {len(patterns)} stuks",
                risk_level=RiskLevel.LOW,
                confidence=0.6,
                details={"patterns": patterns},
                evidence=[str(p)[:50] for p in patterns[:3]],
            ))
        
        return signals
    
    def _calculate_overall_risk(self, signals: List[FraudSignal]) -> Tuple[float, RiskLevel]:
        """Calculate overall risk score from all signals."""
        if not signals:
            return 0.0, RiskLevel.LOW
        
        # Weight by risk level and confidence
        weights = {
            RiskLevel.LOW: 10,
            RiskLevel.MEDIUM: 30,
            RiskLevel.HIGH: 60,
            RiskLevel.CRITICAL: 100,
        }
        
        total_score = sum(
            weights[s.risk_level] * s.confidence
            for s in signals
        )
        
        # Normalize to 0-100
        max_possible = len(signals) * 100
        risk_score = float(min(100, (total_score / max_possible) * 100 * 2))  # Scale up
        
        # Determine overall risk level
        if risk_score >= 70 or any(s.risk_level == RiskLevel.CRITICAL for s in signals):
            overall_risk = RiskLevel.CRITICAL
        elif risk_score >= 50 or any(s.risk_level == RiskLevel.HIGH for s in signals):
            overall_risk = RiskLevel.HIGH
        elif risk_score >= 25:
            overall_risk = RiskLevel.MEDIUM
        else:
            overall_risk = RiskLevel.LOW
        
        return risk_score, overall_risk
    
    def _generate_summary(self, signals: List[FraudSignal], overall_risk: RiskLevel) -> str:
        """Generate human-readable summary."""
        if not signals:
            return "Geen verdachte signalen gedetecteerd."
        
        risk_descriptions = {
            RiskLevel.LOW: "Laag risico",
            RiskLevel.MEDIUM: "Gemiddeld risico", 
            RiskLevel.HIGH: "Hoog risico",
            RiskLevel.CRITICAL: "Kritiek risico",
        }
        
        summary_parts = [f"{risk_descriptions[overall_risk]}: {len(signals)} signalen gedetecteerd."]
        
        # Add top signals
        high_signals = [s for s in signals if s.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)]
        if high_signals:
            summary_parts.append(f"Belangrijkste: {', '.join(s.name for s in high_signals[:3])}")
        
        return " ".join(summary_parts)
    
    async def _enhance_signals_with_llm(self, signals: List[FraudSignal], context_text: str) -> List[FraudSignal]:
        """Enhance fraud signals with LLM-generated user-friendly descriptions in simple language."""
        if not self.llm_client:
            return signals
        
        enhanced_signals = []
        
        for signal in signals:
            # Skip ELA signals - don't enhance with LLM
            if signal.name == "ela_manipulation_detected":
                enhanced_signals.append(signal)
                continue
            
            try:
                # Build prompt for LLM to generate simple language explanation
                evidence_text = ' | '.join(signal.evidence) if signal.evidence else 'Geen specifiek bewijs'
                details_text = ', '.join([f"{k}: {v}" for k, v in signal.details.items() if k not in ['original_description', 'llm_enhanced']])
                
                prompt = f"""Je bent een hulpmiddel dat technische fraud detection bevindingen omzet naar eenvoudige, begrijpelijke taal voor eindgebruikers.

Technische bevinding:
- Signaal: {signal.name}
- Huidige beschrijving: {signal.description}
- Risico niveau: {signal.risk_level.value}
- Zekerheid: {int(signal.confidence * 100)}%
- Details: {details_text}
- Bewijs: {evidence_text}

Context (eerste 500 tekens van document tekst):
{context_text[:500] if context_text else 'Geen tekst beschikbaar'}

Schrijf een korte, eenvoudige uitleg in "Jip en Janneke" taal (maximaal 2 zinnen) die uitlegt wat er is gevonden, zonder technische termen. 
Gebruik gewone woorden die iedereen begrijpt. Wees specifiek over wat er is gevonden (bijvoorbeeld: "een foto in dit document" in plaats van "een afbeelding").

Geef alleen de uitleg terug, geen extra tekst, geen markdown formatting, geen quotes."""

                # Call LLM to generate simple explanation
                response, duration = await self.llm_client.generate_text(
                    prompt,
                    max_tokens=150,
                    temperature=0.3,
                )
                
                if response and response.strip():
                    # Clean up response (remove quotes, markdown, etc.)
                    cleaned_response = response.strip()
                    # Remove leading/trailing quotes
                    if cleaned_response.startswith('"') and cleaned_response.endswith('"'):
                        cleaned_response = cleaned_response[1:-1]
                    elif cleaned_response.startswith("'") and cleaned_response.endswith("'"):
                        cleaned_response = cleaned_response[1:-1]
                    # Remove markdown code blocks
                    if cleaned_response.startswith('```') and cleaned_response.endswith('```'):
                        lines = cleaned_response.split('\n')
                        cleaned_response = '\n'.join(lines[1:-1]).strip()
                    
                    # Replace description with LLM-generated simple language
                    enhanced_signal = FraudSignal(
                        name=signal.name,
                        description=cleaned_response,
                        risk_level=signal.risk_level,
                        confidence=signal.confidence,
                        details={
                            **signal.details,
                            "original_description": signal.description,  # Keep original for reference
                            "llm_enhanced": True,
                        },
                        evidence=signal.evidence,
                    )
                    enhanced_signals.append(enhanced_signal)
                    logger.info(f"Enhanced signal '{signal.name}' with LLM-generated description")
                else:
                    # Fallback to original if LLM fails
                    enhanced_signals.append(signal)
                    logger.warning(f"LLM enhancement failed for signal '{signal.name}' (empty response), using original")
            except Exception as e:
                # Fallback to original if LLM fails
                logger.warning(f"Failed to enhance signal '{signal.name}' with LLM: {e}")
                enhanced_signals.append(signal)
        
        return enhanced_signals


# Singleton instance
_fraud_detector: Optional[FraudDetector] = None


def fraud_detector(llm_client=None) -> FraudDetector:
    """Get the singleton fraud detector instance."""
    global _fraud_detector
    if _fraud_detector is None:
        _fraud_detector = FraudDetector(llm_client=llm_client)
    elif llm_client and _fraud_detector.llm_client is None:
        # Update LLM client if not set
        _fraud_detector.llm_client = llm_client
    return _fraud_detector
