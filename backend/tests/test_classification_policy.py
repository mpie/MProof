"""
Tests for the classification policy system.

Tests:
- Built-in signal computations
- User-defined signal computations (keyword_set, regex_set)
- Requirement/exclusion evaluation
- Import/export roundtrip
- Unknown fallback when no eligible types
"""
import pytest
from app.models.classification_policy import (
    ClassificationPolicy,
    SignalRequirement,
    AcceptanceConfig,
    TrainedModelAcceptance,
    DEFAULT_POLICY,
)
from app.services.signal_engine import (
    Signal,
    compute_all_signals,
    compute_builtin_signal,
    compute_keyword_set_signal,
    compute_regex_set_signal,
    get_builtin_signals,
)
from app.services.classification_engine import (
    evaluate_eligibility,
    evaluate_requirement,
    filter_to_eligible_types,
    should_accept_trained_model,
    parse_policy,
)


class TestBuiltinSignals:
    """Test built-in signal computations."""

    def test_iban_present_detected(self):
        """Test IBAN detection."""
        text = "Betaling naar NL91ABNA0417164300"
        lines = text.split("\n")
        result = compute_builtin_signal("iban_present", text, lines)
        assert result is True

    def test_iban_present_not_detected(self):
        """Test IBAN not detected when absent."""
        text = "Geen IBAN hier"
        lines = text.split("\n")
        result = compute_builtin_signal("iban_present", text, lines)
        assert result is False

    def test_date_count(self):
        """Test date counting."""
        text = "Datum: 01-01-2025\nVolgende: 15-02-2025\nEnde: 31-12-2025"
        lines = text.split("\n")
        result = compute_builtin_signal("date_count", text, lines)
        assert result == 3

    def test_amount_count(self):
        """Test amount counting."""
        text = "Bedrag: €1.234,56\nTotaal: 500,00\nKorting: €25,00"
        lines = text.split("\n")
        result = compute_builtin_signal("amount_count", text, lines)
        assert result == 3

    def test_date_amount_row_count(self):
        """Test transaction row counting (lines with both date and amount)."""
        text = """
01-01-2025  Boodschappen  €45,00
02-01-2025  Benzine  €65,50
Totaal: €110,50
03-01-2025  Restaurant  €32,00
"""
        lines = text.strip().split("\n")
        result = compute_builtin_signal("date_amount_row_count", text, lines)
        assert result == 3  # 3 lines with both date AND amount

    def test_line_count(self):
        """Test non-empty line counting."""
        text = "Regel 1\n\nRegel 2\n  \nRegel 3"
        lines = text.split("\n")
        result = compute_builtin_signal("line_count", text, lines)
        assert result == 3

    def test_token_count(self):
        """Test word counting."""
        text = "Dit is een test met zes woorden"
        lines = text.split("\n")
        result = compute_builtin_signal("token_count", text, lines)
        assert result == 7


class TestUserDefinedSignals:
    """Test user-defined signal computations."""

    def test_keyword_set_any_match(self):
        """Test keyword_set with any match mode."""
        text = "Dit is een factuur voor services"
        config = {"keywords": ["factuur", "rekening", "invoice"], "match_mode": "any"}
        result = compute_keyword_set_signal(text, config)
        assert result is True

    def test_keyword_set_any_no_match(self):
        """Test keyword_set with no match."""
        text = "Dit is een contract document"
        config = {"keywords": ["factuur", "rekening", "invoice"], "match_mode": "any"}
        result = compute_keyword_set_signal(text, config)
        assert result is False

    def test_keyword_set_all_match(self):
        """Test keyword_set with all match mode."""
        text = "Deze factuur bevat het totaal bedrag inclusief BTW"
        config = {"keywords": ["factuur", "totaal", "btw"], "match_mode": "all"}
        result = compute_keyword_set_signal(text, config)
        assert result is True

    def test_keyword_set_all_partial_match(self):
        """Test keyword_set with all mode but only partial match."""
        text = "Deze factuur bevat het totaal"
        config = {"keywords": ["factuur", "totaal", "btw"], "match_mode": "all"}
        result = compute_keyword_set_signal(text, config)
        assert result is False  # BTW is missing

    def test_regex_set_any_match(self):
        """Test regex_set with any match mode."""
        text = "BTW nummer: NL123456789B01"
        config = {"patterns": [r"BTW.*nummer", r"VAT.*number"], "match_mode": "any"}
        result = compute_regex_set_signal(text, config)
        assert result is True

    def test_regex_set_all_match(self):
        """Test regex_set with all match mode."""
        text = "Factuurnummer: F2025-001, BTW: NL123456789B01"
        config = {"patterns": [r"Factuurnummer.*\d+", r"BTW.*NL\d+"], "match_mode": "all"}
        result = compute_regex_set_signal(text, config)
        assert result is True


class TestComputeAllSignals:
    """Test computing all signals at once."""

    def test_compute_all_builtin_signals(self):
        """Test computing all built-in signals."""
        text = """
Bankafschrift
01-01-2025  Saldo: €1.234,56
02-01-2025  Boodschappen  -€45,00
03-01-2025  Salaris  +€2.500,00
IBAN: NL91ABNA0417164300
"""
        signals = get_builtin_signals()
        result = compute_all_signals(text, signals)

        assert result.get("iban_present") is True
        assert result.get("date_count") == 3
        assert result.get("amount_count") >= 3
        assert result.get("date_amount_row_count") >= 2
        assert result.get("line_count") >= 4

    def test_compute_all_with_user_signal(self):
        """Test computing signals including user-defined ones."""
        text = "Factuur voor geleverde diensten"
        
        signals = [
            Signal(
                key="has_invoice_terms",
                label="Factuur termen",
                description="",
                signal_type="boolean",
                source="user",
                compute_kind="keyword_set",
                config_json={"keywords": ["factuur", "invoice"], "match_mode": "any"},
            ),
        ]
        
        result = compute_all_signals(text, signals)
        assert result.get("has_invoice_terms") is True


class TestRequirementEvaluation:
    """Test signal requirement evaluation."""

    def test_boolean_equals_true(self):
        """Test boolean == true requirement."""
        signals = get_builtin_signals()
        text = "IBAN: NL91ABNA0417164300"
        computed = compute_all_signals(text, signals)

        req = SignalRequirement(signal="iban_present", op="==", value=True)
        passed, reason = evaluate_requirement(req, computed)
        assert passed is True
        assert reason == ""

    def test_boolean_equals_false(self):
        """Test boolean == false fails when signal is true."""
        signals = get_builtin_signals()
        text = "IBAN: NL91ABNA0417164300"
        computed = compute_all_signals(text, signals)

        req = SignalRequirement(signal="iban_present", op="==", value=False)
        passed, reason = evaluate_requirement(req, computed)
        assert passed is False
        assert "iban_present" in reason

    def test_count_greater_or_equal(self):
        """Test count >= requirement."""
        signals = get_builtin_signals()
        text = "01-01-2025\n02-01-2025\n03-01-2025"
        computed = compute_all_signals(text, signals)

        req = SignalRequirement(signal="date_count", op=">=", value=3)
        passed, _ = evaluate_requirement(req, computed)
        assert passed is True

    def test_count_less_than_threshold(self):
        """Test count >= fails when below threshold."""
        signals = get_builtin_signals()
        text = "01-01-2025\n02-01-2025"
        computed = compute_all_signals(text, signals)

        req = SignalRequirement(signal="date_count", op=">=", value=5)
        passed, reason = evaluate_requirement(req, computed)
        assert passed is False
        assert "date_count" in reason


class TestEligibilityEvaluation:
    """Test full eligibility evaluation."""

    def test_eligible_with_no_requirements(self):
        """Document is eligible when there are no requirements."""
        policy = ClassificationPolicy()
        signals = get_builtin_signals()
        computed = compute_all_signals("Any text", signals)

        result = evaluate_eligibility("test_type", computed, policy)
        assert result.is_eligible is True
        assert len(result.failed_requirements) == 0
        assert len(result.triggered_exclusions) == 0

    def test_eligible_with_met_requirements(self):
        """Document is eligible when all requirements are met."""
        policy = ClassificationPolicy(
            requirements=[
                SignalRequirement(signal="iban_present", op="==", value=True),
                SignalRequirement(signal="date_count", op=">=", value=2),
            ]
        )
        signals = get_builtin_signals()
        text = "IBAN: NL91ABNA0417164300\n01-01-2025\n02-01-2025"
        computed = compute_all_signals(text, signals)

        result = evaluate_eligibility("test_type", computed, policy)
        assert result.is_eligible is True

    def test_not_eligible_with_failed_requirement(self):
        """Document is not eligible when a requirement fails."""
        policy = ClassificationPolicy(
            requirements=[
                SignalRequirement(signal="date_count", op=">=", value=10),
            ]
        )
        signals = get_builtin_signals()
        text = "01-01-2025\n02-01-2025"
        computed = compute_all_signals(text, signals)

        result = evaluate_eligibility("test_type", computed, policy)
        assert result.is_eligible is False
        assert len(result.failed_requirements) == 1

    def test_not_eligible_with_triggered_exclusion(self):
        """Document is not eligible when an exclusion is triggered."""
        policy = ClassificationPolicy(
            exclusions=[
                SignalRequirement(signal="iban_present", op="==", value=True),
            ]
        )
        signals = get_builtin_signals()
        text = "IBAN: NL91ABNA0417164300"
        computed = compute_all_signals(text, signals)

        result = evaluate_eligibility("test_type", computed, policy)
        assert result.is_eligible is False
        assert len(result.triggered_exclusions) == 1


class TestTrainedModelAcceptance:
    """Test trained model acceptance criteria."""

    def test_accepts_high_confidence(self):
        """Accept prediction with high confidence and margin."""
        policy = ClassificationPolicy(
            acceptance=AcceptanceConfig(
                trained_model=TrainedModelAcceptance(
                    enabled=True,
                    min_confidence=0.85,
                    min_margin=0.10,
                )
            )
        )
        assert should_accept_trained_model(0.90, 0.70, policy) is True

    def test_rejects_low_confidence(self):
        """Reject prediction with low confidence."""
        policy = ClassificationPolicy(
            acceptance=AcceptanceConfig(
                trained_model=TrainedModelAcceptance(
                    enabled=True,
                    min_confidence=0.85,
                    min_margin=0.10,
                )
            )
        )
        assert should_accept_trained_model(0.80, 0.50, policy) is False

    def test_rejects_low_margin(self):
        """Reject prediction with low margin."""
        policy = ClassificationPolicy(
            acceptance=AcceptanceConfig(
                trained_model=TrainedModelAcceptance(
                    enabled=True,
                    min_confidence=0.85,
                    min_margin=0.10,
                )
            )
        )
        assert should_accept_trained_model(0.90, 0.85, policy) is False

    def test_rejects_when_disabled(self):
        """Reject when trained model is disabled."""
        policy = ClassificationPolicy(
            acceptance=AcceptanceConfig(
                trained_model=TrainedModelAcceptance(enabled=False)
            )
        )
        assert should_accept_trained_model(0.99, 0.50, policy) is False


class TestFilterToEligibleTypes:
    """Test filtering to eligible types."""

    def test_filters_based_on_requirements(self):
        """Test that types are filtered based on their requirements."""
        # Type 1: requires IBAN
        policy1 = {
            "requirements": [{"signal": "iban_present", "op": "==", "value": True}],
            "exclusions": [],
            "acceptance": {},
        }
        
        # Type 2: no requirements
        policy2 = {
            "requirements": [],
            "exclusions": [],
            "acceptance": {},
        }

        available_types = [
            ("type_with_iban_req", policy1),
            ("type_no_req", policy2),
        ]

        # Text without IBAN
        text = "Document zonder IBAN"
        signals = get_builtin_signals()

        eligible, results, computed = filter_to_eligible_types(text, available_types, signals)

        assert "type_no_req" in eligible
        assert "type_with_iban_req" not in eligible
        assert results["type_with_iban_req"].is_eligible is False


class TestUnknownFallback:
    """Test unknown fallback behavior."""

    def test_no_eligible_types_returns_empty(self):
        """When no types are eligible, eligible list is empty."""
        policy = {
            "requirements": [{"signal": "date_count", "op": ">=", "value": 100}],
            "exclusions": [],
            "acceptance": {},
        }

        available_types = [("strict_type", policy)]
        text = "Simple text without many dates"
        signals = get_builtin_signals()

        eligible, results, _ = filter_to_eligible_types(text, available_types, signals)

        assert len(eligible) == 0
        assert results["strict_type"].is_eligible is False


class TestPolicyParsing:
    """Test policy parsing."""

    def test_parse_valid_policy(self):
        """Test parsing a valid policy."""
        policy_json = {
            "requirements": [{"signal": "iban_present", "op": "==", "value": True}],
            "exclusions": [],
            "acceptance": {"trained_model": {"enabled": True, "min_confidence": 0.9}},
        }
        
        policy = parse_policy(policy_json)
        assert len(policy.requirements) == 1
        assert policy.requirements[0].signal == "iban_present"
        assert policy.acceptance.trained_model.min_confidence == 0.9

    def test_parse_none_returns_default(self):
        """Test that None returns default policy."""
        policy = parse_policy(None)
        assert policy == DEFAULT_POLICY

    def test_parse_invalid_returns_default(self):
        """Test that invalid JSON returns default policy."""
        policy = parse_policy({"invalid": "structure", "requirements": "not_a_list"})
        assert policy == DEFAULT_POLICY


class TestImportExportRoundtrip:
    """Test import/export roundtrip."""

    def test_policy_roundtrip(self):
        """Test that policy survives JSON roundtrip."""
        import json

        original = ClassificationPolicy(
            requirements=[
                SignalRequirement(signal="iban_present", op="==", value=True),
                SignalRequirement(signal="date_count", op=">=", value=5),
            ],
            exclusions=[
                SignalRequirement(signal="line_count", op="<", value=10),
            ],
            acceptance=AcceptanceConfig(
                trained_model=TrainedModelAcceptance(
                    enabled=True,
                    min_confidence=0.90,
                    min_margin=0.15,
                )
            ),
        )

        # Serialize and deserialize
        json_str = json.dumps(original.model_dump())
        parsed_dict = json.loads(json_str)
        restored = ClassificationPolicy.model_validate(parsed_dict)

        assert len(restored.requirements) == 2
        assert len(restored.exclusions) == 1
        assert restored.acceptance.trained_model.min_confidence == 0.90
        assert restored.requirements[0].signal == "iban_present"
