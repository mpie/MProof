"""
Policy Loader - Load and manage classification policies.

Single canonical format. No versioning.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from app.models.classification_policy import (
    ClassificationPolicy,
    GlobalClassificationConfig,
    DEFAULT_POLICY,
    DEFAULT_GLOBAL_CONFIG,
)

logger = logging.getLogger(__name__)

# Cached global config
_global_config: Optional[GlobalClassificationConfig] = None
_global_config_path: Optional[Path] = None


def _get_config_dir() -> Path:
    """Get the config directory path."""
    return Path(__file__).resolve().parents[2] / "config"


def get_global_config_path() -> Path:
    """Get the path to the global classification config file."""
    return _get_config_dir() / "classification.global.json"


def load_global_config(force_reload: bool = False) -> GlobalClassificationConfig:
    """
    Load the global classification config from file.

    Returns default config if file doesn't exist or is invalid.
    """
    global _global_config, _global_config_path

    config_path = get_global_config_path()

    if not force_reload and _global_config is not None and _global_config_path == config_path:
        return _global_config

    if not config_path.exists():
        logger.info(f"Global config file not found at {config_path}, using defaults")
        _global_config = DEFAULT_GLOBAL_CONFIG
        _global_config_path = config_path
        return _global_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        _global_config = GlobalClassificationConfig.model_validate(data)
        _global_config_path = config_path
        logger.info(f"Loaded global classification config from {config_path}")
        return _global_config

    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Invalid global config file: {e}")
        _global_config = DEFAULT_GLOBAL_CONFIG
        _global_config_path = config_path
        return _global_config


def save_global_config(config: GlobalClassificationConfig) -> None:
    """Save the global classification config to file."""
    global _global_config, _global_config_path

    config_path = get_global_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2)

    _global_config = config
    _global_config_path = config_path
    logger.info(f"Saved global classification config to {config_path}")


def parse_policy(policy_json: Optional[dict]) -> ClassificationPolicy:
    """
    Parse a document type's classification policy JSON.

    Returns default policy if input is None or invalid.
    """
    if policy_json is None:
        return DEFAULT_POLICY

    try:
        return ClassificationPolicy.model_validate(policy_json)
    except ValidationError as e:
        logger.warning(f"Invalid policy JSON, using defaults: {e}")
        return DEFAULT_POLICY


def validate_policy_json(policy_json: dict) -> tuple[bool, Optional[str]]:
    """
    Validate a classification policy JSON.

    Returns (is_valid, error_message).
    """
    try:
        ClassificationPolicy.model_validate(policy_json)
        return True, None
    except ValidationError as e:
        return False, str(e)


def validate_global_config_json(config_json: dict) -> tuple[bool, Optional[str]]:
    """
    Validate a global classification config JSON.

    Returns (is_valid, error_message).
    """
    try:
        GlobalClassificationConfig.model_validate(config_json)
        return True, None
    except ValidationError as e:
        return False, str(e)
