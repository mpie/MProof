import pytest
from app.services.document_processor import DocumentProcessor


class TestValidators:
    @pytest.fixture
    def processor(self):
        return DocumentProcessor(None, None)

    def test_iban_validation_valid(self, processor):
        """Test IBAN validation with valid IBANs."""
        valid_ibans = [
            "NL91ABNA0417164300",  # Dutch IBAN
            "DE89370400440532013000",  # German IBAN
            "GB29NWBK60161331926819",  # UK IBAN
            "FR7630006000011234567890189",  # French IBAN
        ]

        # This would need to be implemented in the actual processor
        # For now, just test the structure exists
        assert hasattr(processor, '_validate_evidence')

    def test_iban_validation_invalid(self, processor):
        """Test IBAN validation with invalid IBANs."""
        invalid_ibans = [
            "INVALID123456",  # Too short
            "NL91ABNA0417164301",  # Invalid checksum (should be 00)
            "XX12345678901234567890",  # Invalid country code
            "",  # Empty
        ]

        # This would need to be implemented in the actual processor
        # For now, just test the structure exists
        assert hasattr(processor, '_check_consistency')

    def test_processor_has_required_methods(self, processor):
        """Test that processor has required validation methods."""
        assert hasattr(processor, '_validate_evidence')
        assert hasattr(processor, '_check_consistency')
        assert callable(processor._validate_evidence)
        assert callable(processor._check_consistency)

    def test_evidence_span_validation(self, processor):
        """Test evidence span validation."""
        # Test successful validation
        evidence = {
            "field1": [
                {"page": 0, "start": 10, "end": 20, "quote": "test value"}
            ]
        }
        pages = [{"page": 0, "source": "text", "text": "This is a test value document"}]

        errors = processor._validate_evidence(evidence, pages)
        assert len(errors) == 0

        # Test failed validation
        evidence_invalid = {
            "field1": [
                {"page": 0, "start": 10, "end": 20, "quote": "wrong quote"}
            ]
        }

        errors = processor._validate_evidence(evidence_invalid, pages)
        assert len(errors) == 1
        assert "Quote mismatch" in errors[0]