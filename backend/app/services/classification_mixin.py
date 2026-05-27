from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from app.models.schemas import ClassificationResult, ExtractionEvidence, EvidenceSpan

if TYPE_CHECKING:
    pass  # avoid circular imports

logger = logging.getLogger(__name__)


class ClassificationMixin:
    """Mixin: document classification methods."""

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
        from sqlalchemy import text as sa_text
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
                    sa_text("""
                        SELECT `key`, label, regex
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

    async def _stage_combined_analysis(self, document, document_dir: Path,
                                       ocr_result) -> Tuple[ClassificationResult, Optional[ExtractionEvidence]]:
        """Combined classification and metadata extraction in single LLM call."""
        from sqlalchemy import text as sa_text
        # Get available document types
        result = await self.db.execute(sa_text("SELECT slug, classification_hints FROM document_types"))
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
                sa_text("SELECT * FROM document_types WHERE slug = :slug"),
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
                sa_text("SELECT * FROM document_type_fields WHERE document_type_id = :doc_type_id"),
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

    def _normalize_extraction_data(self, extraction_data: Dict[str, Any], expected_fields=None) -> Dict[str, Any]:
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

    async def _stage_classification(self, document, document_dir: Path,
                                  ocr_result) -> ClassificationResult:
        """Stage 3: Classify document type using trained model first, then deterministic, then LLM."""
        from sqlalchemy import text as sa_text
        # Skip classification entirely if caller forced a doc type
        if self.force_doc_type:
            logger.info(f"Document {document.id}: force_doc_type='{self.force_doc_type}' — skipping classification")
            return ClassificationResult(
                doc_type_slug=self.force_doc_type,
                confidence=1.0,
                rationale="Forced by caller",
            )

        # Get available document types with hints
        result = await self.db.execute(sa_text("SELECT slug, classification_hints FROM document_types"))
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
                # BERT is purely informational — it provides semantic context shown in the UI.
                # It never decides the document type; only NB → deterministic → LLM do that.
                logger.info(f"Document {document.id}: BERT scored '{bert_result.label}' ({bert_result.confidence:.2f}) — stored as semantic context only, not used for classification")
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

    def _prepare_text_sample(self, text: str, max_chars: int = 6000, skip_markers=None):
        from app.services.document_processor import TextPrepareResult
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
            _skip_regex_cache: dict = {}
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

    async def _load_skip_markers(self):
        """Load active skip markers from database."""
        from sqlalchemy import text as sa_text
        result = await self.db.execute(
            sa_text("SELECT pattern, is_regex FROM skip_markers WHERE is_active = 1")
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
        from sqlalchemy import text as sa_text
        types_str = ", ".join(available_types)

        # Get classification hints from configured document types (exclude "unknown")
        hints = []
        configured_types = [t for t in available_types if t != "unknown"]
        for doc_type_slug in configured_types:
            hint_result = await self.db.execute(
                sa_text("SELECT classification_hints FROM document_types WHERE slug = :slug"),
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
