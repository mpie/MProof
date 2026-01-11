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
from PIL import Image
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "overall_risk": self.overall_risk.value,
            "risk_score": self.risk_score,
            "signals": [
                {
                    "name": s.name,
                    "description": s.description,
                    "risk_level": s.risk_level.value,
                    "confidence": s.confidence,
                    "details": s.details,
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
    
    def __init__(self):
        pass
    
    def analyze_document(
        self,
        file_path: Optional[Path] = None,
        file_bytes: Optional[bytes] = None,
        filename: str = "",
        document_id: Optional[int] = None,
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
        if is_image and file_bytes:
            image_signals = self._analyze_image_forensics(file_bytes, filename)
            signals.extend(image_signals)
        
        # 3. Extract images from PDF for analysis
        if is_pdf and file_bytes:
            pdf_image_signals = self._analyze_pdf_images(file_bytes)
            signals.extend(pdf_image_signals)
        
        # 4. Text Anomaly Detection
        if extracted_text:
            text_signals = self._analyze_text_anomalies(extracted_text)
            signals.extend(text_signals)
        
        # 5. Classification Confidence Analysis
        if classification_confidence is not None:
            conf_signals = self._analyze_classification_confidence(classification_confidence)
            signals.extend(conf_signals)
        
        # 6. Include existing signals from document processing
        if existing_signals:
            proc_signals = self._parse_existing_signals(existing_signals)
            signals.extend(proc_signals)
        
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
    
    def _analyze_image_forensics(self, image_bytes: bytes, filename: str) -> List[FraudSignal]:
        """Perform Error Level Analysis (ELA) on images."""
        signals = []
        
        try:
            # Load image
            img = Image.open(io.BytesIO(image_bytes))
            
            # Only ELA works well on JPEG
            if img.format == 'JPEG' or filename.lower().endswith(('.jpg', '.jpeg')):
                ela_result = self._perform_ela(img)
                if ela_result:
                    signals.append(ela_result)
            
            # Check EXIF data
            exif_signals = self._analyze_exif(img)
            signals.extend(exif_signals)
            
        except Exception as e:
            logger.warning(f"Image forensics failed: {e}")
        
        return signals
    
    def _perform_ela(self, img: Image.Image, quality: int = 90) -> Optional[FraudSignal]:
        """
        Error Level Analysis - detects areas that have been modified.
        
        When a JPEG is resaved, modified areas show different error levels
        compared to the original compression.
        """
        try:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resave at specified quality
            buffer = io.BytesIO()
            img.save(buffer, 'JPEG', quality=quality)
            buffer.seek(0)
            resaved = Image.open(buffer)
            
            # Calculate difference
            original_array = np.array(img, dtype=np.float32)
            resaved_array = np.array(resaved, dtype=np.float32)
            
            diff = np.abs(original_array - resaved_array)
            
            # Analyze error levels
            mean_error = np.mean(diff)
            max_error = np.max(diff)
            std_error = np.std(diff)
            
            # High standard deviation suggests manipulation
            # (different areas have very different error levels)
            if std_error > 15:  # Threshold based on testing
                return FraudSignal(
                    name="ela_manipulation_detected",
                    description="ELA detecteert mogelijke beeldmanipulatie",
                    risk_level=RiskLevel.HIGH,
                    confidence=min(0.9, std_error / 30),
                    details={
                        "mean_error": float(mean_error),
                        "max_error": float(max_error),
                        "std_error": float(std_error),
                    },
                    evidence=[
                        f"Standaarddeviatie error levels: {std_error:.2f}",
                        "Hoge variatie suggereert verschillende compressieniveaus",
                    ],
                )
            
        except Exception as e:
            logger.warning(f"ELA analysis failed: {e}")
        
        return None
    
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
    
    def _analyze_pdf_images(self, pdf_bytes: bytes) -> List[FraudSignal]:
        """Extract and analyze images embedded in PDF."""
        signals = []
        
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(min(5, len(doc))):  # Check first 5 pages
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Perform ELA on extracted image
                        pil_img = Image.open(io.BytesIO(image_bytes))
                        ela_signal = self._perform_ela(pil_img)
                        if ela_signal:
                            ela_signal.details["page"] = page_num + 1
                            ela_signal.details["image_index"] = img_index
                            signals.append(ela_signal)
                            
                    except Exception as e:
                        logger.debug(f"Failed to analyze PDF image: {e}")
            
            doc.close()
            
        except Exception as e:
            logger.warning(f"PDF image analysis failed: {e}")
        
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
        risk_score = min(100, (total_score / max_possible) * 100 * 2)  # Scale up
        
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


# Singleton instance
_fraud_detector: Optional[FraudDetector] = None


def fraud_detector() -> FraudDetector:
    """Get the singleton fraud detector instance."""
    global _fraud_detector
    if _fraud_detector is None:
        _fraud_detector = FraudDetector()
    return _fraud_detector
