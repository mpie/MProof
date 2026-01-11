"""
Classification Engine.

Policy-driven classification with signal-based eligibility.
Single canonical policy format. No versioning.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from app.models.classification_policy import (
    ClassificationPolicy,
    SignalRequirement,
    DEFAULT_POLICY,
)
from app.services.signal_engine import (
    Signal,
    ComputedSignals,
    compute_all_signals,
    get_builtin_signals,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class EligibilityResult:
    """Result of eligibility evaluation for a document type."""

    slug: str
    is_eligible: bool
    failed_requirements: list[str] = field(default_factory=list)
    triggered_exclusions: list[str] = field(default_factory=list)


@dataclass
class ClassificationScore:
    """Score from a classification method."""

    slug: str
    confidence: float
    method: str  # "trained_model", "deterministic", "llm"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationDecision:
    """Final classification decision with metadata."""

    slug: str
    confidence: float
    method: str
    rationale: str
    evidence: str = ""
    top_scores: list[ClassificationScore] = field(default_factory=list)
    eligibility_results: dict[str, EligibilityResult] = field(default_factory=dict)
    computed_signals: Optional[dict[str, Union[bool, int]]] = None


# =============================================================================
# Policy Parsing
# =============================================================================

def parse_policy(policy_json: Optional[dict]) -> ClassificationPolicy:
    """
    Parse a classification policy from JSON.

    Returns default policy if input is None or invalid.
    """
    if policy_json is None:
        return DEFAULT_POLICY

    try:
        return ClassificationPolicy.model_validate(policy_json)
    except Exception as e:
        logger.warning(f"Invalid policy JSON, using defaults: {e}")
        return DEFAULT_POLICY


# =============================================================================
# Eligibility Evaluation
# =============================================================================

def evaluate_requirement(
    req: SignalRequirement,
    signals: ComputedSignals,
) -> tuple[bool, str]:
    """
    Evaluate a single signal requirement.

    Returns (passed, failure_reason).
    """
    signal_value = signals.get(req.signal, 0)
    passed = req.evaluate(signal_value)

    if passed:
        return True, ""

    reason = f"{req.signal} {req.op} {req.value} (werkelijk: {signal_value})"
    return False, reason


def evaluate_eligibility(
    slug: str,
    signals: ComputedSignals,
    policy: ClassificationPolicy,
) -> EligibilityResult:
    """
    Evaluate if a document is eligible for a specific document type.

    Args:
        slug: Document type slug
        signals: Computed signal values
        policy: Classification policy

    Returns:
        EligibilityResult with pass/fail status and reasons
    """
    failed_requirements: list[str] = []
    triggered_exclusions: list[str] = []

    # Check all requirements (ALL must pass)
    for req in policy.requirements:
        passed, reason = evaluate_requirement(req, signals)
        if not passed:
            failed_requirements.append(reason)

    # Check exclusions (if ANY match, document is excluded)
    for excl in policy.exclusions:
        passed, _ = evaluate_requirement(excl, signals)
        if passed:  # Exclusion triggers if the condition is MET
            triggered_exclusions.append(f"{excl.signal} {excl.op} {excl.value}")

    is_eligible = len(failed_requirements) == 0 and len(triggered_exclusions) == 0

    return EligibilityResult(
        slug=slug,
        is_eligible=is_eligible,
        failed_requirements=failed_requirements,
        triggered_exclusions=triggered_exclusions,
    )


def filter_to_eligible_types(
    text: str,
    available_types: list[tuple[str, Optional[dict]]],
    signal_definitions: Optional[list[Signal]] = None,
) -> tuple[list[str], dict[str, EligibilityResult], ComputedSignals]:
    """
    Filter available types to only those the document is eligible for.

    Args:
        text: Document text
        available_types: List of (slug, policy_json) tuples
        signal_definitions: Optional list of signal definitions

    Returns:
        Tuple of (eligible_slugs, eligibility_results_by_slug, computed_signals)
    """
    # Compute signals once for all types
    defs = signal_definitions or get_builtin_signals()
    signals = compute_all_signals(text, defs)

    eligible_slugs = []
    results = {}

    for slug, policy_json in available_types:
        policy = parse_policy(policy_json)
        result = evaluate_eligibility(slug, signals, policy)
        results[slug] = result

        if result.is_eligible:
            eligible_slugs.append(slug)
            logger.debug(f"Type '{slug}' is eligible")
        else:
            logger.debug(
                f"Type '{slug}' not eligible: req={result.failed_requirements}, excl={result.triggered_exclusions}"
            )

    return eligible_slugs, results, signals


# =============================================================================
# Acceptance Criteria
# =============================================================================

def should_accept_trained_model(
    prediction_confidence: float,
    second_best_confidence: float,
    policy: ClassificationPolicy,
) -> bool:
    """Check if a trained model prediction should be accepted based on policy."""
    acceptance = policy.acceptance.trained_model

    if not acceptance.enabled:
        return False

    if prediction_confidence < acceptance.min_confidence:
        return False

    margin = prediction_confidence - second_best_confidence
    if margin < acceptance.min_margin:
        return False

    return True


def should_use_deterministic(policy: ClassificationPolicy) -> bool:
    """Check if deterministic classification should be used."""
    return policy.acceptance.deterministic.enabled


def should_use_llm(policy: ClassificationPolicy) -> bool:
    """Check if LLM classification should be used."""
    return policy.acceptance.llm.enabled


def validate_llm_evidence(
    evidence: str,
    text: str,
    policy: ClassificationPolicy,
) -> bool:
    """Validate LLM evidence if required by policy."""
    if not policy.acceptance.llm.require_evidence:
        return True

    if not evidence:
        return False

    evidence_norm = evidence.lower().strip()
    text_norm = text.lower()

    # Direct match
    if evidence_norm in text_norm:
        return True

    # Compact match (remove non-alphanumeric)
    evidence_compact = re.sub(r"[^a-z0-9]", "", evidence_norm)
    text_compact = re.sub(r"[^a-z0-9]", "", text_norm)

    if len(evidence_compact) >= 8 and evidence_compact in text_compact:
        return True

    # Word overlap (50%+)
    evidence_words = set(re.findall(r"\b[a-z]{3,}\b", evidence_norm))
    text_words = set(re.findall(r"\b[a-z]{3,}\b", text_norm))

    if evidence_words:
        overlap = len(evidence_words & text_words) / len(evidence_words)
        if overlap >= 0.5:
            return True

    return False


def create_unknown_decision(
    eligibility_results: dict[str, EligibilityResult],
    reason: str,
    computed_signals: Optional[dict[str, Union[bool, int]]] = None,
) -> ClassificationDecision:
    """Create a decision for unknown document type."""
    return ClassificationDecision(
        slug="unknown",  # Hardcoded fallback label
        confidence=0.0,
        method="none",
        rationale=reason,
        eligibility_results=eligibility_results,
        computed_signals=computed_signals,
    )


# =============================================================================
# Classification Methods
# =============================================================================

def classify_with_trained_model(
    text: str,
    eligible_slugs: list[str],
    type_policies: dict[str, ClassificationPolicy],
    model_name: Optional[str] = None,
) -> Optional[ClassificationScore]:
    """Attempt classification using trained model (NB + BERT)."""
    if not eligible_slugs:
        return None

    # Try Naive Bayes
    nb_pred = None
    try:
        from app.services.doc_type_classifier import classifier_service

        pred = classifier_service().predict(
            text,
            allowed_labels=eligible_slugs,
            model_name=model_name,
        )
        if pred:
            nb_pred = pred
    except Exception as e:
        logger.debug(f"NB classifier not available: {e}")

    # Try BERT
    bert_pred = None
    try:
        from app.services.bert_classifier import bert_classifier_service

        pred = bert_classifier_service().predict(
            text,
            model_name=model_name,
            allowed_labels=eligible_slugs,
        )
        if pred:
            bert_pred = pred
    except Exception as e:
        logger.debug(f"BERT classifier not available: {e}")

    # Select best prediction
    best_pred = None
    best_method = "naive_bayes"
    second_best_conf = 0.0

    if nb_pred and bert_pred:
        if bert_pred.confidence > nb_pred.confidence + 0.1:
            best_pred = bert_pred
            best_method = "bert"
            second_best_conf = nb_pred.confidence
        else:
            best_pred = nb_pred
            second_best_conf = bert_pred.confidence
    elif bert_pred:
        best_pred = bert_pred
        best_method = "bert"
        if hasattr(bert_pred, "all_scores") and bert_pred.all_scores:
            scores = sorted(bert_pred.all_scores.values(), reverse=True)
            if len(scores) > 1:
                second_best_conf = scores[1]
    elif nb_pred:
        best_pred = nb_pred

    if not best_pred:
        return None

    # Check acceptance criteria
    policy = type_policies.get(best_pred.label, DEFAULT_POLICY)

    if not should_accept_trained_model(best_pred.confidence, second_best_conf, policy):
        logger.debug(
            f"Trained model rejected: conf={best_pred.confidence:.2f}, "
            f"margin={best_pred.confidence - second_best_conf:.2f}"
        )
        return None

    return ClassificationScore(
        slug=best_pred.label,
        confidence=best_pred.confidence,
        method=f"trained_model:{best_method}",
        details={
            "nb_score": nb_pred.confidence if nb_pred else None,
            "bert_score": bert_pred.confidence if bert_pred else None,
            "margin": best_pred.confidence - second_best_conf,
        },
    )


def classify_deterministic(
    text: str,
    eligible_slugs: list[str],
    type_hints: dict[str, Optional[str]],
    type_policies: dict[str, ClassificationPolicy],
) -> Optional[ClassificationScore]:
    """Attempt deterministic classification using keywords/regex."""
    if not eligible_slugs:
        return None

    text_lower = text.lower()
    scores: dict[str, int] = {}

    for slug in eligible_slugs:
        policy = type_policies.get(slug, DEFAULT_POLICY)
        if not should_use_deterministic(policy):
            continue

        hints = type_hints.get(slug)
        if not hints:
            continue

        score = 0
        disqualified = False

        for hint_line in hints.strip().split("\n"):
            hint_line = hint_line.strip()
            if not hint_line:
                continue

            if hint_line.startswith("kw:"):
                keyword = hint_line[3:].strip().lower()
                if keyword in text_lower:
                    score += 1
            elif hint_line.startswith("re:"):
                try:
                    pattern = hint_line[3:].strip()
                    if re.search(pattern, text, re.IGNORECASE):
                        score += 3
                except re.error:
                    pass
            elif hint_line.startswith("not:"):
                neg_word = hint_line[4:].strip().lower()
                if neg_word in text_lower:
                    disqualified = True
                    break
            else:
                if hint_line.lower() in text_lower:
                    score += 1

        if not disqualified and score > 0:
            scores[slug] = score

    if not scores:
        return None

    max_score = max(scores.values())
    best_slugs = [s for s, sc in scores.items() if sc == max_score]

    if len(best_slugs) == 1 and max_score >= 1:
        return ClassificationScore(
            slug=best_slugs[0],
            confidence=0.95,
            method="deterministic",
            details={"score": max_score, "all_scores": scores},
        )

    return None
