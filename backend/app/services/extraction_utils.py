from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from app.models.schemas import ExtractionEvidence, EvidenceSpan

if TYPE_CHECKING:
    pass  # avoid circular imports

logger = logging.getLogger(__name__)


class ExtractionUtilsMixin:
    """Mixin: candidate extraction utilities and schema building."""

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

        # Too short to highlight reliably — a bare number like "4" matches everywhere
        if len(quote) < 5:
            return []

        def _find_with_boundary(page_text: str, q: str) -> int:
            """Find q in page_text, requiring a word boundary so '4' doesn't match inside '4.000'."""
            start = 0
            while True:
                idx = page_text.find(q, start)
                if idx < 0:
                    return -1
                before_ok = idx == 0 or not page_text[idx - 1].isalnum()
                after_ok = idx + len(q) >= len(page_text) or not page_text[idx + len(q)].isalnum()
                if before_ok and after_ok:
                    return idx
                start = idx + 1

        if pages:
            for page_idx, page in enumerate(pages):
                page_text = page.get("text", "")
                start = _find_with_boundary(page_text, quote)
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
                start = _find_with_boundary(page_text.lower(), quote_lower)
                if start >= 0:
                    matched_quote = page_text[start:start + len(quote)]
                    return [{
                        "page": page_idx,
                        "start": start,
                        "end": start + len(matched_quote),
                        "quote": matched_quote,
                    }]

        # Quote not found in any page — return nothing rather than fake coordinates
        return []

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
        has_money_field = any(ft in ("money", "currency") for _, _, ft, _, _, _ in fields)
        notes = []
        if has_address_field:
            notes.append("- For address fields: extract a postal address (street, house number, postal code, city). Do not return only a person/company name.")
        if has_iban_field:
            notes.append("- For IBAN fields: return only the IBAN (e.g., NL..), no extra words or currency.")
        if has_money_field:
            notes.append("- For money/currency fields: preserve the original currency symbol exactly as it appears in the document. If no currency symbol appears near the value in the source text, default to € (euro) — this is a Dutch document. Return as a string such as '€ 39.212,40', not as a bare number. Never use US$ unless $ is explicitly printed in the source text.")
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

    def _correct_evidence_pages(self, result: Dict[str, Any], pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fix LLM-attributed page numbers by searching quote across all pages.

        Tries exact match first, then token-based scoring (for LLM-synthesised quotes
        that combine label + value from separate table rows so no verbatim match exists).
        """
        import re as _re

        def _page_score(page_text: str, quote: str) -> float:
            """Return 0-1 fraction of significant quote tokens present in page_text."""
            tokens = _re.findall(r"[A-Za-zÀ-ÿ0-9]{4,}", quote)
            if not tokens:
                return 0.0
            page_lower = page_text.lower()
            hits = sum(1 for t in tokens if t.lower() in page_lower)
            return hits / len(tokens)

        if "evidence" not in result or not isinstance(result["evidence"], dict):
            return result

        for field_key, spans in result["evidence"].items():
            if not isinstance(spans, list):
                continue
            for span in spans:
                if not isinstance(span, dict):
                    continue
                quote = (span.get("quote") or "").strip()
                if len(quote) < 8:
                    continue
                current_page = span.get("page")
                quote_lower = quote.lower()

                # 1. Exact match on attributed page — nothing to fix
                if current_page is not None and 0 <= current_page < len(pages):
                    if quote_lower in pages[current_page].get("text", "").lower():
                        continue

                # 2. Exact match on any other page
                best_exact = None
                for page_idx, page in enumerate(pages):
                    if page_idx == current_page:
                        continue
                    if quote_lower in page.get("text", "").lower():
                        best_exact = page_idx
                        break
                if best_exact is not None:
                    logger.info(f"Corrected evidence page for '{field_key}': {current_page} → {best_exact} (exact)")
                    span["page"] = best_exact
                    continue

                # 3. Token-based: find the page where most significant tokens appear
                best_page = None
                best_score = 0.0
                for page_idx, page in enumerate(pages):
                    score = _page_score(page.get("text", ""), quote)
                    if score > best_score:
                        best_score = score
                        best_page = page_idx
                if best_page is not None and best_score >= 0.6 and best_page != current_page:
                    logger.info(
                        f"Corrected evidence page for '{field_key}': {current_page} → {best_page} "
                        f"(token score {best_score:.2f})"
                    )
                    span["page"] = best_page

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
                    from datetime import datetime as dt
                    dt.strptime(result, "%Y-%m-%d")
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
