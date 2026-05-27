from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from app.models.schemas import ExtractionEvidence

if TYPE_CHECKING:
    pass  # avoid circular imports

logger = logging.getLogger(__name__)


class MetadataExtractorMixin:
    """Mixin: metadata chunk selection and extraction orchestration."""

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
            if "energielabel" in combined or "energie label" in combined:
                terms.update({"energielabel", "energie label", "energieklasse",
                               "energieprestatie", "energie certificaat", "epc"})
            if "exploitatielasten" in combined or "exploitatie lasten" in combined:
                terms.update({"exploitatielasten", "totale exploitatielasten",
                               "exploitatiekosten", "exploitatie kosten",
                               "kapitalisatiefactoren", "bar nar kapitalisatie",
                               "barinar kapitalisatie"})
            if "locatie_beoordeling" in combined or "locatie beoordeling" in combined:
                terms.update({"ligging", "locatie", "beoordeling locatie", "locatiebeoordeling",
                               "bereikbaarheid", "omgeving", "locatie beoordeling"})
            if "vastgoedtype" in combined or "type vastgoed" in combined or "vastgoed type" in combined:
                terms.update({"vastgoedtype", "type vastgoed", "bedrijfsruimte",
                               "gebruik", "bestemming", "object type", "soort vastgoed",
                               "bedrijfsverzamelgebouw", "appartementsrecht"})

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
        max_context_chars: int = 20000,
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

        fields_covered: Set[str] = set()
        for field_key in field_terms.keys():
            # Tier 1: strict — label AND concrete value found near label
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
                fields_covered.add(field_key)
                continue

            # Tier 2: regex fallback
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
                fields_covered.add(field_key)

        # Tier 3: label-only fallback — for fields not covered by tiers 1/2, pick the best
        # chunk that at least has the label, even without a confirmed concrete value nearby.
        for field_key in field_terms.keys():
            if field_key in fields_covered:
                continue
            label_only = [
                scored for scored in scored_chunks
                if field_key in scored["field_matches"]
                and scored["field_matches"][field_key].get("has_label_context")
                and scored["field_matches"][field_key].get("label_score", 0) > 0
            ]
            if label_only:
                best = max(label_only, key=lambda item: (-item["field_matches"][field_key]["score"], -item["chunk_num"]))
                selected_nums.add(best["chunk_num"])
                selected_reasons.setdefault(best["chunk_num"], []).append(f"label_only_fallback_for_field:{field_key}")

        # Tier 4: fill remaining budget with highest-scoring chunks not yet selected.
        # This ensures the LLM sees enough context even when keyword matching is sparse.
        current_chars = sum(len(str(c.get("text") or "")) for c in chunks if c["chunk_num"] in selected_nums)
        remaining_budget = max_context_chars - current_chars
        if remaining_budget > 800:
            unselected_by_score = sorted(
                [s for s in scored_chunks if s["chunk_num"] not in selected_nums],
                key=lambda s: (-s["score"], s["chunk_num"]),
            )
            for scored in unselected_by_score:
                chunk_text = str(next((c.get("text") or "" for c in chunks if c["chunk_num"] == scored["chunk_num"]), ""))
                chunk_len = len(chunk_text)
                if chunk_len == 0 or remaining_budget - chunk_len < 0:
                    continue
                selected_nums.add(scored["chunk_num"])
                selected_reasons.setdefault(scored["chunk_num"], []).append("fill_budget_by_score")
                remaining_budget -= chunk_len

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

    async def _stage_metadata_extraction(self, document, document_dir: Path,
                                       classification,
                                       ocr_result,
                                       progress_callback=None) -> Optional[ExtractionEvidence]:
        """Stage 4: Extract metadata using LLM based on document type schema."""
        from sqlalchemy import text as sa_text
        logger.info(f"Starting metadata extraction for document {document.id}, doc_type: {classification.doc_type_slug}")

        # Get document type fields and preamble
        # ORDER BY id to ensure consistent ordering and get latest fields
        result = await self.db.execute(
            sa_text("SELECT `key`, label, field_type, required, enum_values, regex FROM document_type_fields WHERE document_type_id = (SELECT id FROM document_types WHERE slug = :slug) ORDER BY id"),
            {"slug": classification.doc_type_slug}
        )
        fields = self._normalize_extraction_fields(result.fetchall())
        logger.info(f"Found {len(fields)} fields for document type {classification.doc_type_slug}")

        # Get preamble
        preamble_result = await self.db.execute(
            sa_text("SELECT extraction_prompt_preamble FROM document_types WHERE slug = :slug"),
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
                        # Tick progress 68→78 while LLM responds; simulate per-chunk progress
                        _tick_pct = [68]
                        _tick_count = [0]
                        _n_selected = len(selected_chunks)
                        _max_ticks = 5
                        async def _llm_ticker():
                            import asyncio as _aio
                            while True:
                                await _aio.sleep(2.5)
                                if _tick_pct[0] < 78:
                                    _tick_pct[0] += 2
                                    _tick_count[0] += 1
                                    simulated = min(_n_selected, max(1, round(_tick_count[0] / _max_ticks * _n_selected)))
                                    try:
                                        await progress_callback(document.id, _tick_pct[0], f"extracting_metadata_llm_chunk_{simulated}_of_{_n_selected}")
                                    except Exception:
                                        pass
                        _ticker = asyncio.create_task(_llm_ticker()) if progress_callback else None
                        try:
                            chunk_result, chunk_response_text, chunk_curl_command, chunk_duration = await self.llm.generate_json_with_raw(prompt, None)
                        finally:
                            if _ticker:
                                _ticker.cancel()
                                try:
                                    await _ticker
                                except asyncio.CancelledError:
                                    pass
                    except Exception as parse_error:
                        logger.warning(f"Selected-chunk candidate extraction failed: {parse_error}")
                        with open(llm_dir / "extraction_warning_selected_chunks.txt", "w", encoding="utf-8") as f:
                            f.write(f"Candidate JSON parse failed:\n{parse_error}\n")
                        err_str = str(parse_error).lower()
                        _resp_preview = str(parse_error)[:600]
                        _truncated = "Failed to parse JSON response" in str(parse_error) and _resp_preview.count('{') > _resp_preview.count('}')
                        context_error = "maximum context length" in err_str or "reduce the length" in err_str or _truncated
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
        # Correct LLM-attributed page numbers by verifying quote location
        result = self._correct_evidence_pages(result, ocr_result.pages)

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
