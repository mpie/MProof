"""
Signal Computation Engine.

Computes classification signals from document text.
Only GENERIC built-in signals. No domain-specific hardcoding.
User-defined signals (keyword_set, regex_set) for domain-specific logic.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Generic Regex Patterns (no domain-specific terms)
# =============================================================================

# IBAN: international standard format
IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ \-]?[A-Z0-9]){11,30}\b")

# Date: DD-MM-YYYY format (common in Dutch/European documents)
DATE_PATTERN = re.compile(r"\b\d{2}-\d{2}-\d{4}\b")

# Amount: Euro amounts like €1.234,56 or 1.234,56
AMOUNT_PATTERN = re.compile(r"(?:€\s*)?\d{1,3}(?:\.\d{3})*(?:,\d{2})\b")


# =============================================================================
# Signal Definition
# =============================================================================

@dataclass
class Signal:
    """Definition of a classification signal."""

    key: str
    label: str
    description: str
    signal_type: str  # "boolean" or "count"
    source: str  # "builtin" or "user"
    compute_kind: str  # "builtin", "keyword_set", "regex_set"
    config_json: Optional[dict] = None


@dataclass
class ComputedSignals:
    """All computed signals for a document."""

    values: dict[str, Union[bool, int]]
    text_length: int
    line_count: int

    def get(self, key: str, default: Union[bool, int] = 0) -> Union[bool, int]:
        """Get a signal value by key."""
        return self.values.get(key, default)


# =============================================================================
# Built-in Signal Computation (GENERIC ONLY)
# =============================================================================

def compute_builtin_signal(key: str, text: str, lines: list[str]) -> Union[bool, int]:
    """
    Compute a built-in signal.

    Only generic signals. No domain-specific logic.
    """
    if key == "iban_present":
        return bool(IBAN_PATTERN.search(text))

    elif key == "date_count":
        return len(DATE_PATTERN.findall(text))

    elif key == "amount_count":
        return len(AMOUNT_PATTERN.findall(text))

    elif key == "date_amount_row_count":
        # Count lines containing both date AND amount
        count = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            has_date = bool(DATE_PATTERN.search(line))
            has_amount = bool(AMOUNT_PATTERN.search(line))
            if has_date and has_amount:
                count += 1
        return count

    elif key == "line_count":
        return sum(1 for line in lines if line.strip())

    elif key == "token_count":
        return len(text.split())

    else:
        logger.warning(f"Unknown builtin signal: {key}")
        return 0


# =============================================================================
# User-defined Signal Computation
# =============================================================================

def compute_keyword_set_signal(text: str, config: dict) -> bool:
    """
    Compute a keyword_set signal.

    Returns True if condition is met based on match_mode:
    - "any": at least one keyword matches
    - "all": all keywords match
    """
    keywords = config.get("keywords", [])
    if not keywords:
        return False

    match_mode = config.get("match_mode", "any")
    text_lower = text.lower()

    if match_mode == "all":
        return all(kw.lower() in text_lower for kw in keywords)
    else:  # "any" is default
        return any(kw.lower() in text_lower for kw in keywords)


def compute_regex_set_signal(text: str, config: dict) -> bool:
    """
    Compute a regex_set signal.

    Returns True if condition is met based on match_mode:
    - "any": at least one pattern matches
    - "all": all patterns match
    """
    patterns = config.get("patterns", [])
    if not patterns:
        return False

    match_mode = config.get("match_mode", "any")

    def matches(pattern: str) -> bool:
        try:
            # Use compile cache
            if pattern not in _regex_cache:
                _regex_cache[pattern] = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            compiled = _regex_cache[pattern]
            return bool(compiled.search(text))
        except re.error as e:
            logger.debug(f"Invalid regex pattern: {pattern} - {e}")
            return False

    if match_mode == "all":
        return all(matches(p) for p in patterns)
    else:  # "any" is default
        return any(matches(p) for p in patterns)


# =============================================================================
# Main Computation
# =============================================================================

def compute_signal(signal: Signal, text: str, lines: list[str]) -> Union[bool, int]:
    """Compute a single signal value."""
    if signal.compute_kind == "builtin":
        return compute_builtin_signal(signal.key, text, lines)

    elif signal.compute_kind == "keyword_set":
        config = signal.config_json or {}
        return compute_keyword_set_signal(text, config)

    elif signal.compute_kind == "regex_set":
        config = signal.config_json or {}
        return compute_regex_set_signal(text, config)

    else:
        logger.warning(f"Unknown compute_kind: {signal.compute_kind}")
        return False if signal.signal_type == "boolean" else 0


# Regex compile cache
_regex_cache: dict[str, re.Pattern] = {}

def compute_all_signals(text: str, signals: list[Signal]) -> ComputedSignals:
    """
    Compute all signal values for a document.

    Args:
        text: Document text
        signals: List of signal definitions to compute

    Returns:
        ComputedSignals with all values
    """
    if not text:
        return ComputedSignals(
            values={s.key: False if s.signal_type == "boolean" else 0 for s in signals},
            text_length=0,
            line_count=0,
        )

    # Truncate text for regex matching to prevent CPU issues
    text = text[:200_000]

    # Normalize whitespace but keep line breaks
    normalized = re.sub(r"[^\S\n]+", " ", text)
    lines = normalized.strip().split("\n")

    values: dict[str, Union[bool, int]] = {}

    for signal in signals:
        try:
            values[signal.key] = compute_signal(signal, normalized, lines)
        except Exception as e:
            logger.error(f"Error computing signal '{signal.key}': {e}")
            values[signal.key] = False if signal.signal_type == "boolean" else 0

    return ComputedSignals(
        values=values,
        text_length=len(text),
        line_count=len(lines),
    )


# =============================================================================
# Default Built-in Signals (GENERIC ONLY - no domain terms)
# =============================================================================

BUILTIN_SIGNALS = [
    Signal(
        key="iban_present",
        label="IBAN aanwezig",
        description="Document bevat minimaal één IBAN nummer",
        signal_type="boolean",
        source="builtin",
        compute_kind="builtin",
    ),
    Signal(
        key="date_count",
        label="Aantal datums",
        description="Aantal datums (DD-MM-YYYY formaat) in document",
        signal_type="count",
        source="builtin",
        compute_kind="builtin",
    ),
    Signal(
        key="amount_count",
        label="Aantal bedragen",
        description="Aantal geldbedragen (€X.XXX,XX formaat) in document",
        signal_type="count",
        source="builtin",
        compute_kind="builtin",
    ),
    Signal(
        key="date_amount_row_count",
        label="Transactieregels",
        description="Aantal regels met zowel datum als bedrag",
        signal_type="count",
        source="builtin",
        compute_kind="builtin",
    ),
    Signal(
        key="line_count",
        label="Aantal regels",
        description="Aantal niet-lege regels in document",
        signal_type="count",
        source="builtin",
        compute_kind="builtin",
    ),
    Signal(
        key="token_count",
        label="Aantal woorden",
        description="Aantal woorden (whitespace-gescheiden tokens)",
        signal_type="count",
        source="builtin",
        compute_kind="builtin",
    ),
]


def get_builtin_signals() -> list[Signal]:
    """Get the built-in signal definitions."""
    return BUILTIN_SIGNALS.copy()
