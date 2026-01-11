"""
Generic Feature Extractor for document classification.

This module extracts features from document text for use in eligibility
evaluation. All regex patterns are generic and not specific to any
document type or bank.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Generic patterns (not bank/document-type specific)
# Date pattern: DD-MM-YYYY format (common in Dutch documents)
DATE_PATTERN = re.compile(r"\b\d{2}-\d{2}-\d{4}\b")

# Amount pattern: Euro amounts like €1.234,56 or 1.234,56
# Matches: €1.234,56, 1.234,56, €123,45, 123,45
AMOUNT_PATTERN = re.compile(r"(?:€\s*)?\d{1,3}(?:\.\d{3})*(?:,\d{2})\b")

# IBAN pattern for normalization
IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ \-]?[A-Z0-9]){11,30}\b")


@dataclass(frozen=True)
class DocumentFeatures:
    """Extracted features from a document."""

    txn_rows: int  # Lines containing both date and amount
    date_count: int  # Total count of dates
    amount_count: int  # Total count of monetary amounts
    text_length: int  # Length of original text
    line_count: int  # Number of lines


def extract_features(text: str) -> DocumentFeatures:
    """
    Extract generic features from document text.

    This function is robust to OCR errors and normalizes whitespace.

    Args:
        text: Raw document text (may contain OCR artifacts)

    Returns:
        DocumentFeatures with extracted counts
    """
    if not text:
        return DocumentFeatures(
            txn_rows=0,
            date_count=0,
            amount_count=0,
            text_length=0,
            line_count=0,
        )

    # Normalize whitespace but keep line breaks for txn_rows
    # Collapse multiple spaces but preserve newlines
    normalized = re.sub(r"[^\S\n]+", " ", text)
    lines = normalized.strip().split("\n")

    # Count dates and amounts in full text
    all_dates = DATE_PATTERN.findall(normalized)
    all_amounts = AMOUNT_PATTERN.findall(normalized)

    # Count transaction rows (lines with both date AND amount)
    txn_rows = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        has_date = bool(DATE_PATTERN.search(line))
        has_amount = bool(AMOUNT_PATTERN.search(line))
        if has_date and has_amount:
            txn_rows += 1

    return DocumentFeatures(
        txn_rows=txn_rows,
        date_count=len(all_dates),
        amount_count=len(all_amounts),
        text_length=len(text),
        line_count=len(lines),
    )


def preprocess_text_for_classification(
    text: str,
    normalize_pii: bool = True,
) -> str:
    """
    Preprocess text for classification and training vectorization.

    IMPORTANT: This normalization is ONLY for classification and training.
    Metadata extraction uses raw text.

    Args:
        text: Raw document text
        normalize_pii: Whether to normalize PII (IBAN, dates, amounts)

    Returns:
        Preprocessed text with optional PII normalization
    """
    if not text:
        return ""

    result = text

    if normalize_pii:
        # Normalize IBANs -> __IBAN__
        result = IBAN_PATTERN.sub("__IBAN__", result)

        # Normalize dates -> __DATE__
        result = DATE_PATTERN.sub("__DATE__", result)

        # Normalize amounts -> __AMOUNT__
        result = AMOUNT_PATTERN.sub("__AMOUNT__", result)

    # Collapse multiple whitespace but keep structure
    result = re.sub(r"[^\S\n]+", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def check_keyword_present(
    text: str,
    keyword: str,
    case_sensitive: bool = False,
) -> bool:
    """Check if a keyword is present in text."""
    if case_sensitive:
        return keyword in text
    return keyword.lower() in text.lower()


def check_regex_match(text: str, pattern: str) -> bool:
    """Check if a regex pattern matches anywhere in text."""
    try:
        return bool(re.search(pattern, text, re.IGNORECASE | re.MULTILINE))
    except re.error:
        # Invalid regex, treat as no match
        return False


def check_any_keyword_present(
    text: str,
    keywords: list[str],
    case_sensitive: bool = False,
) -> bool:
    """Check if any of the keywords are present in text."""
    if not keywords:
        return True  # No requirement = passes
    return any(check_keyword_present(text, kw, case_sensitive) for kw in keywords)


def check_any_regex_match(text: str, patterns: list[str]) -> bool:
    """Check if any of the regex patterns match text."""
    if not patterns:
        return True  # No requirement = passes
    return any(check_regex_match(text, p) for p in patterns)
