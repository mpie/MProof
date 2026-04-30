import pytest
import json
from app.services.document_processor import DocumentProcessor
from app.models.schemas import OCRResult, ExtractionEvidence, EvidenceSpan


class TestDocumentProcessor:
    @pytest.fixture
    def processor(self):
        # Mock dependencies for testing
        return DocumentProcessor(None, None)

    def test_assess_ocr_quality_high(self, processor):
        """Test OCR quality assessment for high quality."""
        text = "This is a normal text document with proper sentences and paragraphs. It contains many words and should be considered high quality OCR."

        quality = processor._assess_ocr_quality(text, True)
        assert quality == "high"

    def test_assess_ocr_quality_medium(self, processor):
        """Test OCR quality assessment for medium quality."""
        # Text with some potential OCR errors
        text = "This is a test document with some potential OCR errors like @#$%^&* and unicode characters àáâãäå"

        quality = processor._assess_ocr_quality(text, True)
        # Should be medium due to unicode characters
        assert quality in ["medium", "low"]

    def test_assess_ocr_quality_low(self, processor):
        """Test OCR quality assessment for low quality."""
        # Very short text with OCR artifacts
        text = "A B C @#$%"

        quality = processor._assess_ocr_quality(text, True)
        assert quality == "low"

    def test_assess_ocr_quality_no_ocr(self, processor):
        """Test OCR quality when OCR was not used."""
        text = "This text was extracted directly from PDF without OCR."

        quality = processor._assess_ocr_quality(text, False)
        assert quality == "high"

    def test_validate_evidence_success(self, processor):
        """Test successful evidence validation."""
        evidence = ExtractionEvidence(
            data={"field1": "test value", "field2": None},
            evidence={
                "field1": [
                    EvidenceSpan(page=0, start=10, end=20, quote="test value")
                ],
                "field2": []  # Empty for null values
            }
        )

        pages = [{"page": 0, "source": "text", "text": "This is a test value document"}]

        errors = processor._validate_evidence(evidence, pages)
        assert len(errors) == 0

    def test_validate_evidence_quote_mismatch(self, processor):
        """Test evidence validation with quote mismatch."""
        evidence = ExtractionEvidence(
            data={"field1": "wrong value"},
            evidence={
                "field1": [
                    EvidenceSpan(page=0, start=10, end=20, quote="test value")
                ]
            }
        )

        pages = [{"page": 0, "source": "text", "text": "This is a different value document"}]

        errors = processor._validate_evidence(evidence, pages)
        assert len(errors) == 1
        assert "Quote mismatch" in errors[0]

    def test_validate_evidence_invalid_page(self, processor):
        """Test evidence validation with invalid page number."""
        evidence = ExtractionEvidence(
            data={"field1": "test value"},
            evidence={
                "field1": [
                    EvidenceSpan(page=5, start=10, end=20, quote="test value")
                ]
            }
        )

        pages = [{"page": 0, "source": "text", "text": "This is a test value document"}]

        errors = processor._validate_evidence(evidence, pages)
        assert len(errors) == 1
        assert "Invalid page" in errors[0]

    def test_validate_evidence_invalid_span(self, processor):
        """Test evidence validation with invalid character span."""
        evidence = ExtractionEvidence(
            data={"field1": "test value"},
            evidence={
                "field1": [
                    EvidenceSpan(page=0, start=50, end=60, quote="test value")
                ]
            }
        )

        pages = [{"page": 0, "source": "text", "text": "This is a test value document"}]

        errors = processor._validate_evidence(evidence, pages)
        assert len(errors) == 1
        assert "Invalid span" in errors[0]

    def test_validate_evidence_missing_for_non_null(self, processor):
        """Test evidence validation when evidence is missing for non-null field."""
        evidence = ExtractionEvidence(
            data={"field1": "test value"},
            evidence={
                "field1": []  # Should have evidence for non-null value
            }
        )

        pages = [{"page": 0, "source": "text", "text": "This is a test value document"}]

        errors = processor._validate_evidence(evidence, pages)
        assert len(errors) == 1
        assert "No evidence provided" in errors[0]

    def test_analyze_text_anomalies_normal_text(self, processor):
        """Test text anomaly analysis for normal text."""
        text = "This is a normal business document with proper text content and no unusual characters."

        anomaly_score, metrics = processor._analyze_text_anomalies(text)

        assert anomaly_score < 0.3  # Should be low
        assert "unicode_ratio" in metrics
        assert "repetition_ratio" in metrics

    def test_analyze_text_anomalies_unicode_heavy(self, processor):
        """Test text anomaly analysis for unicode-heavy text."""
        text = "àáâãäåàáâãäåàáâãäåàáâãäåàáâãäå" * 10  # Lots of unicode

        anomaly_score, metrics = processor._analyze_text_anomalies(text)

        assert anomaly_score > 0.5  # Should be high due to unicode
        assert metrics["unicode_ratio"] > 0.8

    def test_analyze_text_anomalies_repetitive(self, processor):
        """Test text anomaly analysis for repetitive text."""
        text = "test word " * 100  # Very repetitive

        anomaly_score, metrics = processor._analyze_text_anomalies(text)

        assert anomaly_score > 0.5  # Should be high due to repetition
        assert metrics["repetition_ratio"] > 0.8

    def test_check_consistency_vat_higher_than_total(self, processor):
        """Test consistency check for VAT > total."""
        data = {
            "total_amount": 100.0,
            "vat_amount": 25.0
        }

        errors = processor._check_consistency(data)
        assert len(errors) == 0  # Should be fine

        # Now test invalid case
        data["vat_amount"] = 150.0  # VAT > total
        errors = processor._check_consistency(data)
        assert len(errors) == 1
        assert "VAT amount" in errors[0] and "exceeds total" in errors[0]

    def test_check_consistency_negative_amounts(self, processor):
        """Test consistency check for negative amounts."""
        data = {
            "total_amount": -100.0,
            "vat_amount": 25.0
        }

        errors = processor._check_consistency(data)
        assert len(errors) == 1
        assert "Negative amount" in errors[0]

    def test_classify_deterministic_strong_match(self, processor):
        """Test deterministic classification with strong keyword match."""
        available_types = [
            ("bankafschrift-abn-amro", "kw: ABN AMRO\nkw: ABNAMRO"),
            ("bankafschrift-knab", "kw: Knab"),
            ("factuur", "kw: Factuur")
        ]

        text = "Bankafschrift van Knab voor rekeningnummer NL12KNAB0123456789"
        result = processor.classify_deterministic(text, available_types)

        assert result == "bankafschrift-knab"

    def test_classify_deterministic_regex_match(self, processor):
        """Test deterministic classification with regex match."""
        available_types = [
            ("bankafschrift-abn-amro", "re: NL\\d{2}ABNA\\d{10}"),
            ("bankafschrift-knab", "re: NL\\d{2}KNAB\\d{10}"),
        ]

        text = "Bankafschrift NL12KNAB0123456789 van Knab"
        result = processor.classify_deterministic(text, available_types)

        assert result == "bankafschrift-knab"

    def test_classify_deterministic_negative_match(self, processor):
        """Test deterministic classification with negative keywords."""
        available_types = [
            ("bankafschrift-knab", "kw: Knab\nnot: ING\nnot: Rabobank"),
            ("bankafschrift-ing", "kw: ING"),
        ]

        text = "Bankafschrift van Knab met ING in de tekst"
        result = processor.classify_deterministic(text, available_types)

        assert result == "bankafschrift-ing"  # Knab disqualified due to "not: ING", ING matches

    def test_classify_deterministic_no_match(self, processor):
        """Test deterministic classification with no strong evidence."""
        available_types = [
            ("bankafschrift-abn-amro", "kw: ABN AMRO"),
            ("factuur", "kw: Factuur"),
        ]

        text = "Dit is een algemeen bank document zonder specifieke kenmerken"
        result = processor.classify_deterministic(text, available_types)

        assert result is None

    def test_classify_deterministic_tie_no_match(self, processor):
        """Test deterministic classification with tie (should return None)."""
        available_types = [
            ("type1", "kw: test"),
            ("type2", "kw: test"),
        ]

        text = "Dit document bevat het woord test"
        result = processor.classify_deterministic(text, available_types)

        assert result is None  # Tie should not be resolved

    def test_classify_deterministic_single_keyword_match(self, processor):
        """Test deterministic classification with single keyword match."""
        available_types = [
            ("bankafschrift", "kw: bank"),
        ]

        text = "Dit document bevat het woord bank"
        result = processor.classify_deterministic(text, available_types)

        assert result == "bankafschrift"  # Single keyword match is sufficient

    def test_validate_llm_classification_valid_unknown(self, processor):
        """Test LLM validation for valid 'unknown' classification."""
        result = {
            "doc_type_slug": "unknown",
            "confidence": 0.0,
            "rationale": "No evidence found",
            "evidence": ""
        }
        sample_text = "Some random text without specific patterns"

        validated = processor._validate_llm_classification(result, sample_text)
        assert validated["doc_type_slug"] == "unknown"
        assert validated["confidence"] == 0.0
        assert validated["evidence"] == ""

    def test_validate_llm_classification_valid_with_evidence(self, processor):
        """Test LLM validation for valid classification with evidence."""
        result = {
            "doc_type_slug": "bankafschrift-knab",
            "confidence": 0.8,
            "rationale": "Found Knab evidence",
            "evidence": "Knab bank"
        }
        sample_text = "Bankafschrift van Knab bank voor rekening NL12KNAB0123456789"

        validated = processor._validate_llm_classification(result, sample_text)
        assert validated["doc_type_slug"] == "bankafschrift-knab"
        assert validated["confidence"] == 0.8

    def test_validate_llm_classification_invalid_evidence_not_found(self, processor):
        """Test LLM validation when evidence is not found in text."""
        result = {
            "doc_type_slug": "bankafschrift-knab",
            "confidence": 0.8,
            "rationale": "Found evidence",
            "evidence": "nonexistent evidence"
        }
        sample_text = "Some text without the claimed evidence"

        validated = processor._validate_llm_classification(result, sample_text)
        assert validated["doc_type_slug"] == "unknown"
        assert "not found in document text" in validated["rationale"]

    def test_validate_llm_classification_missing_evidence(self, processor):
        """Test LLM validation when evidence is missing for non-unknown type."""
        result = {
            "doc_type_slug": "bankafschrift-knab",
            "confidence": 0.8,
            "rationale": "Found evidence",
            "evidence": ""
        }
        sample_text = "Some text"

        validated = processor._validate_llm_classification(result, sample_text)
        assert validated["doc_type_slug"] == "unknown"
        assert "No evidence provided" in validated["rationale"]

    def test_validate_llm_classification_low_confidence(self, processor):
        """Test LLM validation with low confidence for specific type."""
        result = {
            "doc_type_slug": "bankafschrift-knab",
            "confidence": 0.3,
            "rationale": "Uncertain match",
            "evidence": "Knab"
        }
        sample_text = "Bankafschrift van Knab"

        validated = processor._validate_llm_classification(result, sample_text)
        assert validated["doc_type_slug"] == "unknown"
        assert "Low confidence" in validated["rationale"]

    def test_prepare_text_sample_normalization(self, processor):
        """Test text sample preparation with whitespace normalization."""
        text = "Line 1\n\n  Line 2  \t\tLine 3\n\n\nLine 4"
        sample = processor._prepare_text_sample(text, 1000)

        # Should normalize whitespace
        assert "\n\n" not in sample or sample.count("\n\n") <= 1
        assert "\t" not in sample
        assert "  " not in sample

    def test_prepare_text_sample_header_priority(self, processor):
        """Test that header lines are prioritized in text sample."""
        text = "Header line 1\nHeader line 2\nHeader line 3\n" + "Body content " * 100
        sample = processor._prepare_text_sample(text, 200)

        # First lines should be included
        assert "Header line 1" in sample
        assert "Header line 2" in sample

    def test_normalize_for_search_handles_units_and_separators(self, processor):
        """Search normalization should make labels and OCR text comparable."""
        normalized = processor._normalize_for_search("Oppervlakte_m² - Bouw-Jaar")

        assert normalized == "oppervlakte m2 bouw jaar"

    def test_select_relevant_metadata_chunks_uses_field_labels(self, processor):
        """Large documents should select label-relevant chunks instead of all chunks."""
        fields = [
            ("bouwjaar", "Bouwjaar", "number", False, None, r"\b\d{4}\b"),
            ("oppervlakte_m2", "Oppervlakte (m2)", "number", False, None, None),
        ]
        chunks = [
            {"chunk_num": 1, "page": 0, "part": 0, "text": "Inhoudsopgave\nDisclaimer\nDefinities"},
            {"chunk_num": 2, "page": 1, "part": 0, "text": "Algemene uitleg zonder metadata labels. " * 80},
            {"chunk_num": 3, "page": 2, "part": 0, "text": "Objectgegevens\nBouwjaar: 1930\nOppervlakte (m2): 62"},
            {"chunk_num": 4, "page": 3, "part": 0, "text": "Bijlage index\nReferenties"},
        ]

        selected, debug = processor._select_relevant_metadata_chunks(
            chunks,
            fields,
            top_n=2,
            threshold=40,
            max_context_chars=2000,
        )

        selected_nums = [chunk["chunk_num"] for chunk in selected]
        assert 3 in selected_nums
        assert len(selected) < len(chunks)
        assert debug["original_chunk_count"] == 4
        assert debug["selected_chunk_count"] == len(selected)
        assert debug["scores"][2]["field_matches"]["bouwjaar"]["score"] >= 100

    def test_select_relevant_metadata_chunks_prefers_per_field_label_matches(self, processor):
        """Regex-only chunks should not be selected when field-label chunks exist."""
        fields = [("bouwjaar", "Bouwjaar", "number", False, None, r"\b\d{4}\b")]
        chunks = [
            {"chunk_num": 1, "page": 0, "part": 0, "text": "Documentnummer: 2024"},
            {"chunk_num": 2, "page": 1, "part": 0, "text": "Objectgegevens\nBouwjaar: 1930"},
        ]

        selected, debug = processor._select_relevant_metadata_chunks(chunks, fields)

        assert 2 in [chunk["chunk_num"] for chunk in selected]
        selected_reason_by_chunk = {
            chunk["chunk_num"]: chunk["reasons"]
            for chunk in debug["selected_chunks"]
        }
        assert selected_reason_by_chunk[2] == ["best_for_field:bouwjaar"]
        assert "regex_fallback_for_field:bouwjaar" not in selected_reason_by_chunk.get(1, [])

    def test_deterministic_candidate_extraction_same_line_labels(self, processor):
        """Obvious same-line labels should become candidates before the LLM."""
        fields = [("bouwjaar", "Bouwjaar", "number", True, None, r"\b\d{4}\b")]
        chunks = [
            {"chunk_num": 3, "page": 2, "part": 0, "text": "Objectgegevens\nBouwjaar: 1930\n"},
        ]

        candidates = processor._deterministic_candidate_extraction(chunks, fields)
        resolved = processor._resolve_chunk_candidate_results([(3, candidates)], fields)

        assert candidates["candidates"]["bouwjaar"][0]["value"] == "1930"
        assert resolved["data"]["bouwjaar"] == "1930"
        assert processor._all_required_fields_resolved(resolved, fields) is True

    def test_deterministic_candidate_rejects_date_as_year(self, processor):
        """A date value must not become a year candidate for a year-like field."""
        fields = [("bouwjaar", "Bouwjaar", "number", False, None, r"\b\d{4}\b")]
        chunks = [
            {"chunk_num": 1, "page": 0, "part": 0, "text": "Bouwjaar: 26/03/2026"},
        ]

        candidates = processor._deterministic_candidate_extraction(chunks, fields)

        assert candidates["candidates"]["bouwjaar"] == []

    def test_candidate_validation_rejects_document_number_as_year(self, processor):
        """Document/reference numbers should not validate as year fields."""
        field_config = {
            "key": "bouwjaar",
            "label": "Bouwjaar",
            "field_type": "number",
            "enum_values": None,
            "regex": r"\b\d{4}\b",
        }

        valid, reason, _ = processor._validate_candidate_value_for_field(
            "2024",
            field_config,
            evidence="Documentnummer: 2024",
        )

        assert valid is False
        assert reason == "year_evidence_has_document_number_label"

    def test_candidate_validation_rejects_postcode_as_area(self, processor):
        """Postcodes should not validate as area fields."""
        field_config = {
            "key": "oppervlakte_m2",
            "label": "Oppervlakte (m2)",
            "field_type": "number",
            "enum_values": None,
            "regex": None,
        }

        valid, reason, _ = processor._validate_candidate_value_for_field(
            "1234",
            field_config,
            evidence="Postcode: 1234 AB",
            unit="m2",
        )

        assert valid is False
        assert reason == "area_evidence_has_postcode"

    def test_build_selected_chunks_text_respects_prompt_budget(self, processor):
        """Selected chunk text should be clipped before creating the LLM prompt."""
        fields = [("bouwjaar", "Bouwjaar", "number", False, None, None)]
        chunks = [
            {
                "chunk_num": 1,
                "page": 0,
                "part": 0,
                "text": ("Algemene tekst " * 300) + "\nBouwjaar: 1930\n" + ("Meer tekst " * 300),
            },
            {
                "chunk_num": 2,
                "page": 1,
                "part": 0,
                "text": ("Andere tekst " * 500) + "\nBouwjaar: 1931\n",
            },
        ]

        selected_text, debug = processor._build_selected_chunks_text(chunks, fields, max_chars=1600)

        assert len(selected_text) <= 1600
        assert "Bouwjaar" in selected_text
        assert debug[0]["included_chars"] < debug[0]["original_chars"]

    def test_clip_prefers_metadata_table_block_over_later_explanation(self, processor):
        """Clipping should preserve the real data block before later label explanations."""
        fields = [
            ("bouwjaar", "Bouwjaar", "number", False, None, None),
            ("oppervlakte_m2", "Oppervlakte (m2)", "number", False, None, None),
        ]
        data_block = (
            "Objectgegevens:\n"
            "Type woning: Etage-portiekwoning\n"
            "Onderdelen Bouwjaar Oppervlakte (m²) Aantal\n"
            "Woning 1930 62\n"
            "Balkon/terras 1930 3\n"
        )
        text = (
            "Intro tekst\n" * 30
            + data_block
            + "Algemene toelichting\n" * 80
            + "Bouwjaar: Het bouwjaar van uw woning wordt gebruikt voor uitleg.\n"
            + "Oppervlakte: De oppervlakte is een definitie zonder concrete waarde.\n"
        )
        chunks = [{"chunk_num": 1, "page": 0, "part": 0, "text": text}]

        selected_text, _ = processor._build_selected_chunks_text(chunks, fields, max_chars=2200)

        assert data_block.strip() in selected_text
        assert "Bouwjaar: Het bouwjaar van uw woning" not in selected_text

    def test_score_metadata_chunk_prefers_concrete_value_context_over_explanation(self, processor):
        """A label with concrete values should beat a definition-only label chunk."""
        fields = [
            ("bouwjaar", "Bouwjaar", "number", False, None, None),
            ("oppervlakte_m2", "Oppervlakte (m2)", "number", False, None, None),
        ]
        field_terms = processor._build_field_search_terms(fields)

        data_score = processor._score_metadata_chunk({
            "chunk_num": 1,
            "page": 0,
            "part": 0,
            "text": "Onderdelen Bouwjaar Oppervlakte (m²) Aantal\nWoning 1930 62",
        }, field_terms)
        explanation_score = processor._score_metadata_chunk({
            "chunk_num": 2,
            "page": 1,
            "part": 0,
            "text": "Bouwjaar: Het bouwjaar van uw woning wordt gebruikt voor algemene uitleg.",
        }, field_terms)

        assert data_score["score"] > explanation_score["score"]
        assert data_score["score"] >= 70
        assert data_score["field_matches"]["bouwjaar"]["value_score"] > 0
        assert explanation_score["field_matches"]["bouwjaar"]["value_score"] == 0

    def test_select_relevant_metadata_chunks_rejects_explanation_only_labels(self, processor):
        """Definition-only label chunks should not beat concrete table value chunks."""
        fields = [
            ("bouwjaar", "Bouwjaar", "number", False, None, None),
            ("oppervlakte_m2", "Oppervlakte (m2)", "number", False, None, None),
        ]
        chunks = [
            {
                "chunk_num": 1,
                "page": 0,
                "part": 0,
                "text": (
                    "Objectgegevens:\n"
                    "Type woning: Etage-portiekwoning\n"
                    "Onderdelen Bouwjaar Oppervlakte (m²) Aantal\n"
                    "Woning 1930 62\n"
                    "Balkon/terras 1930 3\n"
                ),
            },
            {
                "chunk_num": 2,
                "page": 1,
                "part": 0,
                "text": (
                    "Bouwjaar: Het bouwjaar van uw woning is van belang voor de uitleg.\n"
                    "Oppervlakte: Bij het bepalen van de oppervlakte geldt algemene toelichting.\n"
                ),
            },
        ]

        selected, debug = processor._select_relevant_metadata_chunks(chunks, fields)

        selected_nums = [chunk["chunk_num"] for chunk in selected]
        assert 1 in selected_nums
        assert 2 not in selected_nums
        skipped_explanation = next(item for item in debug["skipped_chunks"] if item["chunk_num"] == 2)
        assert skipped_explanation["rejected_as_explanation_only"] is True

    def test_build_selected_chunks_text_keeps_short_full_page_with_objectgegevens(self, processor):
        """Short selected pages should be sent whole so table context before labels is preserved."""
        fields = [
            ("bouwjaar", "Bouwjaar", "number", False, None, None),
            ("oppervlakte_m2", "Oppervlakte (m2)", "number", False, None, None),
        ]
        expected_block = (
            "Objectgegevens:\n"
            "Type woning: Etage-portiekwoning\n"
            "Onderdelen Bouwjaar Oppervlakte (m²) Aantal\n"
            "Woning 1930 62\n"
            "Balkon/terras 1930 3\n"
        )
        chunks = [{"chunk_num": 1, "page": 0, "part": 0, "text": expected_block + "\nExtra korte context"}]

        selected_text, _ = processor._build_selected_chunks_text(chunks, fields, max_chars=3500)

        assert expected_block.strip() in selected_text
        assert "SYSTEM CHUNK METADATA, NOT DOCUMENT CONTENT" in selected_text
        assert "DOCUMENT CHUNK TEXT:" in selected_text

    def test_normalize_candidate_chunk_result_repairs_table_evidence(self, processor):
        """Header-only table evidence should be expanded around the value in chunk text."""
        fields = [
            ("bouwjaar", "Bouwjaar", "number", False, None, None),
            ("oppervlakte_m2", "Oppervlakte (m2)", "number", False, None, None),
        ]
        chunk_text = (
            "Objectgegevens:\n"
            "Type woning: Etage-portiekwoning\n"
            "Onderdelen\n"
            "Bouwjaar\n"
            "Oppervlakte (m²)\n"
            "Aantal\n"
            "Woning\n"
            "1930\n"
            "62\n"
            "Balkon/terras\n"
            "1930\n"
            "3\n"
        )
        chunk_result = {
            "candidates": {
                "bouwjaar": [{
                    "value": "1930",
                    "normalized_value": "1930",
                    "unit": None,
                    "evidence": "Onderdelen\nBouwjaar\nOppervlakte (m²)",
                    "chunk_index": 0,
                    "confidence": 90,
                    "evidence_type": "table_context",
                    "record_role": "unknown",
                }],
                "oppervlakte_m2": [{
                    "value": "62",
                    "normalized_value": "62",
                    "unit": "m2",
                    "evidence": "Onderdelen\nBouwjaar\nOppervlakte (m²)",
                    "chunk_index": 0,
                    "confidence": 90,
                    "evidence_type": "table_context",
                    "record_role": "unknown",
                }],
            },
        }

        normalized = processor._normalize_candidate_chunk_result(chunk_result, fields, 1, chunk_text)
        resolved = processor._resolve_chunk_candidate_results([(1, normalized)], fields)
        repaired_bouwjaar = normalized["candidates"]["bouwjaar"][0]
        repaired_oppervlakte = normalized["candidates"]["oppervlakte_m2"][0]

        assert "Woning" in repaired_bouwjaar["evidence"]
        assert "1930" in repaired_bouwjaar["evidence"]
        assert "62" in repaired_oppervlakte["evidence"]
        assert repaired_bouwjaar["evidence_repair"]["repair_reason"] == "value_found_nearby_in_chunk"
        assert resolved["data"]["bouwjaar"] == "1930"
        assert resolved["data"]["oppervlakte_m2"] == "62"

    def test_candidate_resolver_keeps_primary_table_values(self, processor):
        """Primary table candidates should beat later repeated records."""
        fields = [
            ("year", "Year", "number", False, None, None),
            ("size", "Size", "number", False, None, None),
        ]
        pages = [{
            "page": 0,
            "source": "text",
            "text": "Attributes: Year Size\nMain 1930 62\n\nReference records:\nRecord A 1934 61\nRecord B 1934 63",
        }]
        chunk_results = [
            (1, {
                "candidates": {
                    "year": [{
                        "value": "1930",
                        "normalized_value": "1930",
                        "unit": None,
                        "evidence": "Main 1930 62",
                        "chunk_index": 0,
                        "confidence": 90,
                        "evidence_type": "table_context",
                        "record_role": "primary",
                    }],
                    "size": [{
                        "value": "62",
                        "normalized_value": "62",
                        "unit": None,
                        "evidence": "Main 1930 62",
                        "chunk_index": 0,
                        "confidence": 90,
                        "evidence_type": "table_context",
                        "record_role": "primary",
                    }],
                },
            }),
            (3, {
                "candidates": {
                    "year": [
                        {
                            "value": "1934",
                            "normalized_value": "1934",
                            "unit": None,
                            "evidence": "Record A 1934 61",
                            "chunk_index": 2,
                            "confidence": 85,
                            "evidence_type": "table_context",
                            "record_role": "secondary",
                        },
                        {
                            "value": "1934",
                            "normalized_value": "1934",
                            "unit": None,
                            "evidence": "Record B 1934 63",
                            "chunk_index": 2,
                            "confidence": 85,
                            "evidence_type": "table_context",
                            "record_role": "example",
                        },
                    ],
                    "size": [
                        {
                            "value": "61",
                            "normalized_value": "61",
                            "unit": None,
                            "evidence": "Record A 1934 61",
                            "chunk_index": 2,
                            "confidence": 85,
                            "evidence_type": "table_context",
                            "record_role": "secondary",
                        },
                        {
                            "value": "63",
                            "normalized_value": "63",
                            "unit": None,
                            "evidence": "Record B 1934 63",
                            "chunk_index": 2,
                            "confidence": 85,
                            "evidence_type": "table_context",
                            "record_role": "example",
                        },
                    ],
                },
            }),
        ]

        resolved = processor._resolve_chunk_candidate_results(chunk_results, fields, pages)

        assert resolved["data"]["year"] == "1930"
        assert resolved["data"]["size"] == "62"
        assert "rejected_candidates" in resolved
        assert resolved["rejected_candidates"]["year"][0]["record_role"] in {"secondary", "example"}

    def test_candidate_resolver_uses_later_primary_after_background(self, processor):
        """Background-only chunks should not block a later primary record."""
        fields = [("amount", "Amount", "number", False, None, None)]
        pages = [{
            "page": 0,
            "source": "text",
            "text": "General explanation without a concrete value.\n\nPrimary summary\nAmount: 1250",
        }]
        chunk_results = [
            (1, {
                "candidates": {
                    "amount": [],
                },
            }),
            (2, {
                "candidates": {
                    "amount": [{
                        "value": "1250",
                        "normalized_value": "1250",
                        "unit": None,
                        "evidence": "Amount: 1250",
                        "chunk_index": 1,
                        "confidence": 92,
                        "evidence_type": "exact_label",
                        "record_role": "primary",
                    }],
                },
            }),
        ]

        resolved = processor._resolve_chunk_candidate_results(chunk_results, fields, pages)

        assert resolved["data"]["amount"] == "1250"
        assert resolved["evidence"]["amount"][0]["quote"] == "Amount: 1250"

    def test_candidate_resolver_prefers_exact_label_over_ambiguous_low_confidence(self, processor):
        """A strong exact-label candidate should beat an earlier ambiguous value."""
        fields = [("code", "Code", "text", False, None, None)]
        pages = [{
            "page": 0,
            "source": "text",
            "text": "Maybe X-1 applies.\n\nCode: A-42",
        }]
        chunk_results = [
            (1, {
                "candidates": {
                    "code": [{
                        "value": "X-1",
                        "normalized_value": "X-1",
                        "unit": None,
                        "evidence": "Maybe X-1 applies.",
                        "chunk_index": 0,
                        "confidence": 35,
                        "evidence_type": "ambiguous",
                        "record_role": "unknown",
                    }],
                },
            }),
            (2, {
                "candidates": {
                    "code": [{
                        "value": "A-42",
                        "normalized_value": "A-42",
                        "unit": None,
                        "evidence": "Code: A-42",
                        "chunk_index": 1,
                        "confidence": 95,
                        "evidence_type": "exact_label",
                        "record_role": "primary",
                    }],
                },
            }),
        ]

        resolved = processor._resolve_chunk_candidate_results(chunk_results, fields, pages)

        assert resolved["data"]["code"] == "A-42"
        assert resolved["rejected_candidates"]["code"][0]["value"] == "X-1"

    def test_candidate_resolver_rejects_invalid_high_confidence_year(self, processor):
        """High LLM confidence must not rescue an invalid field type."""
        fields = [("bouwjaar", "Bouwjaar", "number", False, None, r"\b\d{4}\b")]
        chunk_results = [
            (1, {
                "candidates": {
                    "bouwjaar": [
                        {
                            "value": "26/03/2026",
                            "normalized_value": "26/03/2026",
                            "unit": None,
                            "evidence": "Bouwjaar: 26/03/2026",
                            "chunk_index": 0,
                            "confidence": 99,
                            "evidence_type": "exact_label",
                            "record_role": "primary",
                        },
                        {
                            "value": "1930",
                            "normalized_value": "1930",
                            "unit": None,
                            "evidence": "Bouwjaar: 1930",
                            "chunk_index": 0,
                            "confidence": 70,
                            "evidence_type": "exact_label",
                            "record_role": "unknown",
                        },
                    ],
                },
            }),
        ]

        resolved = processor._resolve_chunk_candidate_results(chunk_results, fields)

        assert resolved["data"]["bouwjaar"] == "1930"
        assert resolved["rejected_candidates"]["bouwjaar"][0]["rejection_reason"] == "year_value_looks_like_date"

    def test_chunk_prompt_requests_candidates_without_regex_instruction(self, processor):
        """Chunk extraction prompt should request candidates and avoid regex instructions."""
        fields = [("code", "Code", "text", False, None, r"CODE-\d+")]

        prompt = processor._build_extraction_prompt(
            fields,
            "Code: CODE-123",
            "generic-document",
            chunk_num=1,
            total_chunks=2,
        )

        assert '"candidates": {"code": []}' in prompt
        assert "Return possible candidates per field only" in prompt
        assert "Give possible candidates only" in prompt
        assert "Only return candidates that really occur in this chunk" in prompt
        assert "Do not create placeholder candidates" in prompt
        assert "Do not use regex as the extraction method" in prompt
        assert "Pattern:" not in prompt

    def test_normalize_candidate_chunk_result_converts_legacy_data_output(self, processor):
        """Legacy data/evidence chunk output should become resolver candidates."""
        fields = [("code", "Code", "text", False, None, None)]
        chunk_result = {
            "data": {"code": "A-42"},
            "evidence": {"code": [{"page": 0, "start": 0, "end": 10, "quote": "Code: A-42"}]},
        }

        normalized = processor._normalize_candidate_chunk_result(chunk_result, fields, chunk_num=2)

        candidate = normalized["candidates"]["code"][0]
        assert candidate["value"] == "A-42"
        assert candidate["evidence"] == "Code: A-42"
        assert candidate["chunk_index"] == 1
        assert candidate["record_role"] == "unknown"

    def test_candidate_resolver_rejects_placeholder_candidates(self, processor):
        """Placeholder candidates should stay empty and resolve to null."""
        fields = [("energielabel", "Energielabel", "text", False, None, None)]
        chunk_results = [
            (1, {
                "candidates": {
                    "energylabel": [{
                        "value": "",
                        "normalized_value": "",
                        "unit": None,
                        "evidence": "niet opgenomen in chunk",
                        "chunk_index": 0,
                        "confidence": 50,
                        "evidence_type": "ambiguous",
                        "record_role": "background",
                    }],
                },
            }),
        ]

        normalized = processor._normalize_candidate_chunk_result(chunk_results[0][1], fields, chunk_num=1)
        resolved = processor._resolve_chunk_candidate_results([(1, normalized)], fields)

        assert normalized["candidates"]["energielabel"] == []
        assert resolved["data"]["energielabel"] is None

    def test_candidate_resolver_prefers_stronger_later_evidence(self, processor):
        """A later candidate can win when its evidence score is stronger."""
        fields = [("value", "Value", "text", False, None, None)]
        pages = [{
            "page": 0,
            "source": "text",
            "text": "Value: first\n\nValue: second",
        }]
        chunk_results = [
            (1, {
                "candidates": {
                    "value": [{
                        "value": "first",
                        "normalized_value": "first",
                        "unit": None,
                        "evidence": "Value: first",
                        "chunk_index": 0,
                        "confidence": 70,
                        "evidence_type": "nearby_label",
                        "record_role": "primary",
                    }],
                },
            }),
            (2, {
                "candidates": {
                    "value": [{
                        "value": "second",
                        "normalized_value": "second",
                        "unit": None,
                        "evidence": "Value: second",
                        "chunk_index": 1,
                        "confidence": 90,
                        "evidence_type": "exact_label",
                        "record_role": "primary",
                    }],
                },
            }),
        ]

        resolved = processor._resolve_chunk_candidate_results(chunk_results, fields, pages)

        assert resolved["data"]["value"] == "second"

    def test_candidate_resolver_expected_generic_document_shape(self, processor):
        """Resolver should produce the expected generic final data shape."""
        fields = [
            ("bouwjaar", "Bouwjaar", "number", False, None, None),
            ("energylabel", "Energielabel", "text", False, None, None),
            ("oppervlakte_m2", "Oppervlakte", "number", False, None, None),
            ("locatie_beoordeling", "Locatie", "text", False, None, None),
        ]
        pages = [{
            "page": 0,
            "source": "text",
            "text": "Attributes: bouwjaar oppervlakte_m2\nMain 1930 62\n\nniet opgenomen in chunk",
        }]
        chunk_results = [
            (1, {
                "candidates": {
                    "bouwjaar": [{
                        "value": "1930",
                        "normalized_value": "1930",
                        "unit": None,
                        "evidence": "Main 1930 62",
                        "chunk_index": 0,
                        "confidence": 90,
                        "evidence_type": "table_context",
                        "record_role": "primary",
                    }],
                    "energylabel": [{
                        "value": "",
                        "normalized_value": "",
                        "unit": None,
                        "evidence": "niet opgenomen in chunk",
                        "chunk_index": 0,
                        "confidence": 50,
                        "evidence_type": "ambiguous",
                        "record_role": "background",
                    }],
                    "oppervlakte_m2": [{
                        "value": "62",
                        "normalized_value": "62",
                        "unit": "m2",
                        "evidence": "Main 1930 62",
                        "chunk_index": 0,
                        "confidence": 90,
                        "evidence_type": "table_context",
                        "record_role": "primary",
                    }],
                    "locatie_beoordeling": [],
                },
            }),
        ]

        normalized = processor._normalize_candidate_chunk_result(chunk_results[0][1], fields, chunk_num=1)
        resolved = processor._resolve_chunk_candidate_results([(1, normalized)], fields, pages)

        assert resolved["data"] == {
            "bouwjaar": "1930",
            "energielabel": None,
            "oppervlakte_m2": "62",
            "locatie_beoordeling": None,
        }

    def test_candidate_resolver_uses_regex_only_for_validation(self, processor):
        """Regex should reject invalid candidates without being used in the prompt."""
        fields = [("code", "Code", "text", False, None, r"^CODE-\d+$")]
        pages = [{
            "page": 0,
            "source": "text",
            "text": "Code: OTHER-123\nCode: CODE-456",
        }]
        chunk_results = [
            (1, {
                "candidates": {
                    "code": [
                        {
                            "value": "OTHER-123",
                            "normalized_value": "OTHER-123",
                            "unit": None,
                            "evidence": "Code: OTHER-123",
                            "chunk_index": 0,
                            "confidence": 95,
                            "evidence_type": "exact_label",
                            "record_role": "primary",
                        },
                        {
                            "value": "CODE-456",
                            "normalized_value": "CODE-456",
                            "unit": None,
                            "evidence": "Code: CODE-456",
                            "chunk_index": 0,
                            "confidence": 80,
                            "evidence_type": "exact_label",
                            "record_role": "primary",
                        },
                    ],
                },
            }),
        ]

        resolved = processor._resolve_chunk_candidate_results(chunk_results, fields, pages)
        prompt = processor._build_extraction_prompt(
            fields,
            "Code: CODE-456",
            "generic-document",
            chunk_num=1,
            total_chunks=1,
        )

        assert resolved["data"]["code"] == "CODE-456"
        assert resolved["rejected_candidates"]["code"][0]["rejected_reason"] == "regex_mismatch"
        assert "Pattern:" not in prompt

    def test_evidence_supported_by_text_exact_match(self, processor):
        """Test evidence validation with exact match."""
        evidence = "Rabobank rekeningafschrift"
        sample_text = "Dit is een Rabobank rekeningafschrift document"

        assert processor._evidence_supported_by_text(evidence, sample_text) == True

    def test_evidence_supported_by_text_partial_match(self, processor):
        """Test evidence validation with partial keyword match."""
        evidence = "Rabobank Westland Bankcode 3436 Rekeningafschrift"
        sample_text = "Dit document bevat Rabobank en rekeningafschrift woorden"

        assert processor._evidence_supported_by_text(evidence, sample_text) == True

    def test_evidence_supported_by_text_insufficient_match(self, processor):
        """Test evidence validation with insufficient keyword match."""
        evidence = "ABN AMRO bank statement with transactions"
        sample_text = "Dit document heeft alleen ABN woorden maar niet de anderen"

        assert processor._evidence_supported_by_text(evidence, sample_text) == False

    def test_evidence_supported_by_text_empty_evidence(self, processor):
        """Test evidence validation with empty evidence."""
        evidence = ""
        sample_text = "Some text"

        assert processor._evidence_supported_by_text(evidence, sample_text) == False