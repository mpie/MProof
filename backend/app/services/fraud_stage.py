import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.models.schemas import (
    ExtractionEvidence,
    OCRResult,
    ClassificationResult,
    RiskAnalysis,
    RiskSignal,
)
from app.models.database import Document

logger = logging.getLogger(__name__)


class FraudStageMixin:
    """Mixin providing fraud/risk analysis stage methods."""

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
        from app.main import async_session_maker
        from sqlalchemy import text as sa_text

        original_path = document_dir / "original" / document.original_filename
        file_bytes = original_path.read_bytes() if original_path.exists() else None
        semantic_context = self._load_semantic_context(document_dir)

        # Load dynamic field validation rules for this document type
        field_rules = []
        if classification.doc_type_slug:
            try:
                async with async_session_maker() as session:
                    rows = await session.execute(
                        sa_text(
                            "SELECT dtf.key, dtf.validation_rules "
                            "FROM document_type_fields dtf "
                            "JOIN document_types dt ON dt.id = dtf.document_type_id "
                            "WHERE dt.slug = :slug AND dtf.validation_rules IS NOT NULL"
                        ),
                        {"slug": classification.doc_type_slug},
                    )
                    import json as _json
                    for row in rows.fetchall():
                        raw = row.validation_rules
                        try:
                            rules = _json.loads(raw) if isinstance(raw, str) else (raw or [])
                        except Exception as parse_exc:
                            logger.warning(f"Malformed validation_rules JSON for field '{row.key}': {parse_exc}")
                            continue
                        for rule in rules:
                            if isinstance(rule, dict) and "validation_type" in rule:
                                field_rules.append({"field_name": row.key, **rule})
            except Exception as exc:
                logger.warning(f"Could not load field validation rules for {classification.doc_type_slug}: {exc}")

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
            field_rules=field_rules or None,
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

    async def _stage_risk_analysis(
        self,
        document: Document,
        document_dir: Path,
        ocr_result: OCRResult,
        extraction_result: Optional[ExtractionEvidence],
    ) -> RiskAnalysis:
        """Stage 5: Perform risk analysis and generate signals."""
        from pypdf import PdfReader

        signals = []
        risk_score = 0

        # Signal 1: File metadata suspicious (PDF only)
        if document.mime_type == "application/pdf":
            try:
                reader = PdfReader(document_dir / "original" / document.original_filename)
                metadata = reader.metadata

                # Whitelist of known legitimate PDF producers from major software companies
                legitimate_producers = [
                    "adobe", "microsoft", "google", "apple",
                    "libreoffice", "openoffice", "inkscape",
                    "ghostscript", "pdflib", "itext", "apache",
                ]

                # Suspicious patterns - indicate potentially manipulated PDFs
                suspicious_patterns = [
                    "ilovepdf", "ilove pdf",
                    "pdf converter", "pdfconverter",
                    "unknown", "unnamed",
                ]

                # Programming language PDF libraries - can be legitimate but also used for fraud
                programmatic_libraries = [
                    "fpdf", "tcpdf", "dompdf", "mpdf",
                    "python", "java", "php", "ruby",
                ]

                producer = getattr(metadata, 'producer', '').lower() if metadata else ''

                if any(legit in producer for legit in legitimate_producers):
                    pass  # Legitimate producer, no risk signal
                elif any(lib in producer for lib in programmatic_libraries):
                    detected_lib = next((lib for lib in programmatic_libraries if lib in producer), "programmatische PDF library")
                    signals.append(RiskSignal(
                        code="FILE_METADATA_PROGRAMMATIC",
                        severity="medium",
                        message=f"PDF gemaakt met {detected_lib}",
                        evidence=f"PDF metadata producer: {producer}. Dit kan legitiem zijn, maar wordt ook gebruikt om documenten te manipuleren."
                    ))
                    risk_score += 30
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
            evidence_parts = [
                f"Unicode ratio: {metrics['unicode_ratio']:.2f}",
                f"Repetition ratio: {metrics['repetition_ratio']:.2f}"
            ]

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
        if unicode_ratio > 0.1:
            unicode_pattern = re.compile(r'[^\x00-\x7F]+')
            matches = unicode_pattern.findall(text)
            seen = set()
            for match in matches[:20]:
                if match not in seen and len(match.strip()) > 0:
                    seen.add(match)
                    unicode_examples.append(match.strip()[:50])
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

            word_counts: Dict[str, int] = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            repeated_words = [(word, count) for word, count in word_counts.items() if count >= 5]
            repeated_words.sort(key=lambda x: x[1], reverse=True)

            repetition_examples = []
            if repeated_words:
                sentences = re.split(r'[.!?]\s+', text)
                top_repeated_word = repeated_words[0][0]
                for sentence in sentences[:10]:
                    if top_repeated_word.lower() in sentence.lower() and len(sentence.strip()) > 10:
                        count_in_sentence = sentence.lower().count(top_repeated_word.lower())
                        if count_in_sentence >= 2:
                            repetition_examples.append(sentence.strip()[:100])
                            if len(repetition_examples) >= 3:
                                break

        # Combined score
        anomaly_score = (unicode_ratio * 0.6) + (repetition_ratio * 0.4)

        return anomaly_score, {
            "unicode_ratio": unicode_ratio,
            "repetition_ratio": repetition_ratio,
            "unicode_examples": unicode_examples[:3],
            "repetition_examples": repetition_examples[:3]
        }

    def _check_consistency(self, data: Dict[str, Any]) -> List[str]:
        """Check for consistency in extracted data."""
        errors = []

        for key, value in data.items():
            if "amount" in key.lower() and isinstance(value, (int, float)) and value < 0:
                errors.append(f"{key}: Negative amount {value}")

        vat = data.get("vat_amount")
        total = data.get("total_amount")
        if vat is not None and total is not None and isinstance(vat, (int, float)) and isinstance(total, (int, float)):
            if vat > total:
                errors.append(f"VAT amount ({vat}) exceeds total amount ({total})")

        return errors
