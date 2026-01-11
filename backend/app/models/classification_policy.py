"""
Classification Policy Schema - Single Canonical Format.

NO VERSIONING. One schema for all policies.
All domain-specific logic comes from user-defined signals, not hardcoded.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# =============================================================================
# Signal Requirement
# =============================================================================

class SignalRequirement(BaseModel):
    """A signal-based condition for eligibility."""

    signal: str = Field(description="Signal key (e.g., 'iban_present', 'date_count')")
    op: Literal["==", "!=", ">=", "<=", ">", "<"] = Field(
        default="==",
        description="Comparison operator",
    )
    value: Union[bool, int, float] = Field(description="Value to compare against")

    def evaluate(self, computed_value: Union[bool, int, float]) -> bool:
        """Evaluate this requirement against a computed signal value."""
        if self.op == "==":
            return computed_value == self.value
        elif self.op == "!=":
            return computed_value != self.value
        elif self.op == ">=":
            return computed_value >= self.value
        elif self.op == "<=":
            return computed_value <= self.value
        elif self.op == ">":
            return computed_value > self.value
        elif self.op == "<":
            return computed_value < self.value
        return False


# =============================================================================
# Acceptance Configuration
# =============================================================================

class TrainedModelAcceptance(BaseModel):
    """Acceptance criteria for trained model (Naive Bayes / BERT)."""

    enabled: bool = Field(default=True, description="Whether to use trained model")
    min_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum probability threshold",
    )
    min_margin: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Minimum margin between top prediction and second-best",
    )


class DeterministicAcceptance(BaseModel):
    """Acceptance criteria for deterministic matching."""

    enabled: bool = Field(default=True, description="Whether to use deterministic matching")


class LLMAcceptance(BaseModel):
    """Acceptance criteria for LLM classification."""

    enabled: bool = Field(default=True, description="Whether to use LLM classification")
    require_evidence: bool = Field(
        default=True,
        description="Require evidence substring validation",
    )


class AcceptanceConfig(BaseModel):
    """Combined acceptance configuration for all classification methods."""

    trained_model: TrainedModelAcceptance = Field(default_factory=TrainedModelAcceptance)
    deterministic: DeterministicAcceptance = Field(default_factory=DeterministicAcceptance)
    llm: LLMAcceptance = Field(default_factory=LLMAcceptance)


# =============================================================================
# Canonical Classification Policy (SINGLE FORMAT)
# =============================================================================

class ClassificationPolicy(BaseModel):
    """
    Per-document-type classification policy.

    This is the ONLY policy format. No versioning.
    All behavioral differences come from this policy + signals.
    """

    requirements: list[SignalRequirement] = Field(
        default_factory=list,
        description="Signal requirements - ALL must be met for eligibility",
    )
    exclusions: list[SignalRequirement] = Field(
        default_factory=list,
        description="Signal exclusions - if ANY match, document is NOT eligible",
    )
    acceptance: AcceptanceConfig = Field(default_factory=AcceptanceConfig)


# =============================================================================
# Global Configuration
# =============================================================================

class GlobalClassificationConfig(BaseModel):
    """Global classification settings."""

    unknown_fallback: str = Field(
        default="unknown",
        description="Label to use when no type matches (hardcoded fallback, not a document type)",
    )
    normalize_pii_for_classification: bool = Field(
        default=True,
        description="Normalize PII (IBAN, dates, amounts) during classification/training",
    )
    default_acceptance: AcceptanceConfig = Field(
        default_factory=AcceptanceConfig,
        description="Default acceptance thresholds for types without specific policy",
    )


# =============================================================================
# Signal Definition
# =============================================================================

class SignalDefinition(BaseModel):
    """Definition of a classification signal."""

    key: str
    label: str
    description: Optional[str] = None
    signal_type: Literal["boolean", "count"]
    source: Literal["builtin", "user"]
    compute_kind: Literal["builtin", "keyword_set", "regex_set"]
    config_json: Optional[dict] = None


# =============================================================================
# Import/Export Schemas
# =============================================================================

class SignalExport(BaseModel):
    """Signal for export (user-defined only)."""

    key: str
    label: str
    description: Optional[str] = None
    signal_type: str
    compute_kind: str  # 'keyword_set' or 'regex_set'
    config_json: Optional[dict] = None


class PolicyExportEntry(BaseModel):
    """Document type policy for export."""

    slug: str
    policy: Optional[ClassificationPolicy] = None


class ConfigExportBundle(BaseModel):
    """Bundle for import/export of signals and policies."""

    signals: list[SignalExport] = Field(default_factory=list)
    policies: list[PolicyExportEntry] = Field(default_factory=list)


class ConfigImportResult(BaseModel):
    """Result of config import operation."""

    success: bool
    imported_signals: list[str] = Field(default_factory=list)
    imported_policies: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# =============================================================================
# Default instances
# =============================================================================

DEFAULT_POLICY = ClassificationPolicy()
DEFAULT_GLOBAL_CONFIG = GlobalClassificationConfig()
