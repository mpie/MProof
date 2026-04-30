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
        assert "Extract candidate values only" in prompt
        assert "Do not use regex as the extraction method" in prompt
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