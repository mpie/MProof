"""
Classification Policy API.

Single canonical policy format. No versioning.
Import/export for signals and policies.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, ValidationError
from sqlalchemy import text

from app.models.classification_policy import (
    ClassificationPolicy,
    GlobalClassificationConfig,
    ConfigExportBundle,
    ConfigImportResult,
    SignalExport,
    PolicyExportEntry,
    SignalRequirement,
)
from app.services.policy_loader import (
    load_global_config,
    save_global_config,
    validate_policy_json,
    parse_policy,
)
from app.services.signal_engine import Signal, compute_all_signals
from app.api.signals import load_all_signals_from_db

logger = logging.getLogger(__name__)


def _parse_policy_json_field(policy_json) -> Optional[dict]:
    """Parse policy JSON field from database (may be string or dict)."""
    if policy_json is None:
        return None
    if isinstance(policy_json, str):
        try:
            return json.loads(policy_json)
        except json.JSONDecodeError:
            return None
    return policy_json
router = APIRouter()


# =============================================================================
# Response Schemas
# =============================================================================

class PolicyValidationResponse(BaseModel):
    valid: bool
    error: Optional[str] = None


class GlobalConfigResponse(BaseModel):
    config: GlobalClassificationConfig


class TypePolicyResponse(BaseModel):
    slug: str
    policy: Optional[ClassificationPolicy] = None
    has_policy: bool


class SignalValue(BaseModel):
    key: str
    label: str
    value: object
    signal_type: str


class RequirementResult(BaseModel):
    signal: str
    op: str
    required_value: object
    actual_value: object
    passed: bool


class EligibilityPreviewResponse(BaseModel):
    is_eligible: bool
    computed_signals: list[SignalValue]
    requirement_results: list[RequirementResult]
    exclusion_results: list[RequirementResult]
    failed_requirements: list[str]
    triggered_exclusions: list[str]


class EligibilityPreviewRequest(BaseModel):
    text: str
    policy: Optional[ClassificationPolicy] = None


# =============================================================================
# Global Config Endpoints
# =============================================================================

@router.get("/classification/config/global", response_model=GlobalConfigResponse)
async def get_global_config():
    """Get global classification configuration."""
    config = load_global_config()
    return GlobalConfigResponse(config=config)


@router.put("/classification/config/global", response_model=GlobalConfigResponse)
async def update_global_config(config: GlobalClassificationConfig = Body(...)):
    """Update global classification configuration."""
    save_global_config(config)
    return GlobalConfigResponse(config=config)


# =============================================================================
# Type Policy Endpoints
# =============================================================================

@router.get("/document-types/{slug}/policy", response_model=TypePolicyResponse)
async def get_document_type_policy(slug: str):
    """Get classification policy for a document type."""
    from app.main import async_session_maker

    async with async_session_maker() as session:
        result = await session.execute(
            text("SELECT slug, classification_policy_json FROM document_types WHERE slug = :slug"),
            {"slug": slug},
        )
        row = result.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Document type not found")

        policy_json = _parse_policy_json_field(row.classification_policy_json)
        if policy_json:
            policy = parse_policy(policy_json)
            return TypePolicyResponse(slug=slug, policy=policy, has_policy=True)

        return TypePolicyResponse(slug=slug, policy=None, has_policy=False)


@router.put("/document-types/{slug}/policy", response_model=TypePolicyResponse)
async def update_document_type_policy(slug: str, policy: ClassificationPolicy = Body(...)):
    """Update classification policy for a document type."""
    from datetime import datetime
    from app.main import async_session_maker

    async with async_session_maker() as session:
        result = await session.execute(
            text("SELECT id FROM document_types WHERE slug = :slug"),
            {"slug": slug},
        )
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="Document type not found")

        await session.execute(
            text("""
                UPDATE document_types
                SET classification_policy_json = :policy, updated_at = :updated_at
                WHERE slug = :slug
            """),
            {
                "slug": slug,
                "policy": json.dumps(policy.model_dump()),
                "updated_at": datetime.now(),
            },
        )
        await session.commit()

        return TypePolicyResponse(slug=slug, policy=policy, has_policy=True)


@router.delete("/document-types/{slug}/policy")
async def delete_document_type_policy(slug: str):
    """Delete classification policy (reset to default)."""
    from datetime import datetime
    from app.main import async_session_maker

    async with async_session_maker() as session:
        result = await session.execute(
            text("SELECT id FROM document_types WHERE slug = :slug"),
            {"slug": slug},
        )
        if not result.fetchone():
            raise HTTPException(status_code=404, detail="Document type not found")

        await session.execute(
            text("""
                UPDATE document_types
                SET classification_policy_json = NULL, updated_at = :updated_at
                WHERE slug = :slug
            """),
            {"slug": slug, "updated_at": datetime.now()},
        )
        await session.commit()

        return {"ok": True, "message": "Policy deleted, default will be used"}


@router.post("/document-types/{slug}/policy/validate", response_model=PolicyValidationResponse)
async def validate_type_policy(slug: str, policy_json: dict = Body(...)):
    """Validate classification policy JSON."""
    is_valid, error = validate_policy_json(policy_json)
    return PolicyValidationResponse(valid=is_valid, error=error)


# =============================================================================
# Eligibility Preview
# =============================================================================

@router.post("/document-types/{slug}/policy/preview", response_model=EligibilityPreviewResponse)
async def preview_eligibility(slug: str, request: EligibilityPreviewRequest):
    """
    Preview eligibility for a document.

    Test how a document would be evaluated against the policy.
    """
    from app.main import async_session_maker

    # Load signals
    signals = await load_all_signals_from_db()

    # Get or use provided policy
    if request.policy:
        policy = request.policy
    else:
        async with async_session_maker() as session:
            result = await session.execute(
                text("SELECT classification_policy_json FROM document_types WHERE slug = :slug"),
                {"slug": slug},
            )
            row = result.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Document type not found")
            policy = parse_policy(_parse_policy_json_field(row.classification_policy_json))

    # Compute signals
    computed = compute_all_signals(request.text, signals)

    # Build signal values list
    signal_values = [
        SignalValue(
            key=s.key,
            label=s.label,
            value=computed.get(s.key),
            signal_type=s.signal_type,
        )
        for s in signals
    ]

    # Evaluate requirements
    requirement_results: list[RequirementResult] = []
    failed_requirements: list[str] = []

    for req in policy.requirements:
        actual = computed.get(req.signal, 0)
        passed = req.evaluate(actual)
        requirement_results.append(RequirementResult(
            signal=req.signal,
            op=req.op,
            required_value=req.value,
            actual_value=actual,
            passed=passed,
        ))
        if not passed:
            failed_requirements.append(f"{req.signal} {req.op} {req.value}")

    # Evaluate exclusions
    exclusion_results: list[RequirementResult] = []
    triggered_exclusions: list[str] = []

    for excl in policy.exclusions:
        actual = computed.get(excl.signal, 0)
        passed = excl.evaluate(actual)
        exclusion_results.append(RequirementResult(
            signal=excl.signal,
            op=excl.op,
            required_value=excl.value,
            actual_value=actual,
            passed=passed,
        ))
        if passed:  # Exclusion triggers when condition is MET
            triggered_exclusions.append(f"{excl.signal} {excl.op} {excl.value}")

    is_eligible = len(failed_requirements) == 0 and len(triggered_exclusions) == 0

    return EligibilityPreviewResponse(
        is_eligible=is_eligible,
        computed_signals=signal_values,
        requirement_results=requirement_results,
        exclusion_results=exclusion_results,
        failed_requirements=failed_requirements,
        triggered_exclusions=triggered_exclusions,
    )


# =============================================================================
# Import/Export (Canonical Only)
# =============================================================================

@router.get("/classification/config/export", response_model=ConfigExportBundle)
async def export_config():
    """
    Export configuration: user signals + policies.

    Only user-defined signals are exported.
    Built-in signals are not exported (already in system).
    """
    from app.main import async_session_maker

    async with async_session_maker() as session:
        # Export user-defined signals only
        result = await session.execute(
            text("""
                SELECT key, label, description, signal_type, compute_kind, config_json
                FROM classification_signals
                WHERE source = 'user'
                ORDER BY key
            """)
        )
        signal_rows = result.fetchall()

        signals = [
            SignalExport(
                key=row.key,
                label=row.label,
                description=row.description,
                signal_type=row.signal_type,
                compute_kind=row.compute_kind,
                config_json=row.config_json,
            )
            for row in signal_rows
        ]

        # Export policies
        result = await session.execute(
            text("SELECT slug, classification_policy_json FROM document_types ORDER BY slug")
        )
        policy_rows = result.fetchall()

        policies = []
        for row in policy_rows:
            policy = None
            policy_json = _parse_policy_json_field(row.classification_policy_json)
            if policy_json:
                try:
                    policy = ClassificationPolicy.model_validate(policy_json)
                except ValidationError:
                    logger.warning(f"Invalid policy for '{row.slug}', skipping")

            policies.append(PolicyExportEntry(slug=row.slug, policy=policy))

        return ConfigExportBundle(signals=signals, policies=policies)


@router.post("/classification/config/import", response_model=ConfigImportResult)
async def import_config(bundle: ConfigExportBundle = Body(...)):
    """
    Import configuration: user signals + policies.

    Upsert based on key/slug. Validates strictly.
    """
    from datetime import datetime
    from app.main import async_session_maker

    errors = []
    imported_signals = []
    imported_policies = []

    # Validate all policies first
    for entry in bundle.policies:
        if entry.policy:
            try:
                ClassificationPolicy.model_validate(entry.policy.model_dump())
            except ValidationError as e:
                errors.append(f"Invalid policy for '{entry.slug}': {e}")

    if errors:
        return ConfigImportResult(
            success=False,
            imported_signals=[],
            imported_policies=[],
            errors=errors,
        )

    async with async_session_maker() as session:
        # Import user signals (upsert)
        for sig in bundle.signals:
            try:
                result = await session.execute(
                    text("SELECT id, source FROM classification_signals WHERE key = :key"),
                    {"key": sig.key}
                )
                row = result.fetchone()

                if row:
                    if row.source == "builtin":
                        errors.append(f"Cannot overwrite built-in signal '{sig.key}'")
                        continue

                    # Update existing user signal
                    await session.execute(
                        text("""
                            UPDATE classification_signals
                            SET label = :label, description = :description,
                                signal_type = :signal_type, compute_kind = :compute_kind,
                                config_json = :config_json, updated_at = CURRENT_TIMESTAMP
                            WHERE key = :key
                        """),
                        {
                            "key": sig.key,
                            "label": sig.label,
                            "description": sig.description,
                            "signal_type": sig.signal_type,
                            "compute_kind": sig.compute_kind,
                            "config_json": json.dumps(sig.config_json) if sig.config_json else None,
                        }
                    )
                else:
                    # Insert new user signal
                    await session.execute(
                        text("""
                            INSERT INTO classification_signals
                            (key, label, description, signal_type, source, compute_kind, config_json)
                            VALUES (:key, :label, :description, :signal_type, 'user', :compute_kind, :config_json)
                        """),
                        {
                            "key": sig.key,
                            "label": sig.label,
                            "description": sig.description,
                            "signal_type": sig.signal_type,
                            "compute_kind": sig.compute_kind,
                            "config_json": json.dumps(sig.config_json) if sig.config_json else None,
                        }
                    )

                imported_signals.append(sig.key)

            except Exception as e:
                errors.append(f"Fout bij importeren signaal '{sig.key}': {e}")

        # Import policies (upsert)
        for entry in bundle.policies:
            try:
                result = await session.execute(
                    text("SELECT id FROM document_types WHERE slug = :slug"),
                    {"slug": entry.slug},
                )
                if not result.fetchone():
                    logger.info(f"Skipping policy for non-existent type '{entry.slug}'")
                    continue

                policy_value = None
                if entry.policy:
                    policy_value = json.dumps(entry.policy.model_dump())

                await session.execute(
                    text("""
                        UPDATE document_types
                        SET classification_policy_json = :policy, updated_at = :updated_at
                        WHERE slug = :slug
                    """),
                    {
                        "slug": entry.slug,
                        "policy": policy_value,
                        "updated_at": datetime.now(),
                    },
                )
                imported_policies.append(entry.slug)

            except Exception as e:
                errors.append(f"Error importing policy '{entry.slug}': {e}")

        await session.commit()

    return ConfigImportResult(
        success=len(errors) == 0,
        imported_signals=imported_signals,
        imported_policies=imported_policies,
        errors=errors,
    )


# =============================================================================
# Schema
# =============================================================================

@router.get("/classification/policy/schema")
async def get_policy_schema():
    """Get JSON schema for classification policy."""
    return {
        "policy_schema": ClassificationPolicy.model_json_schema(),
        "global_config_schema": GlobalClassificationConfig.model_json_schema(),
        "signal_requirement_schema": SignalRequirement.model_json_schema(),
    }
