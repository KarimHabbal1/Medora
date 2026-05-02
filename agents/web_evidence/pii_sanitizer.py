"""Deterministic PII removal for privacy-safe medical web evidence queries."""

from __future__ import annotations

import re
from typing import Any

from agents.web_evidence.schemas import PrivacyReport, WebEvidenceRequest


EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}\b")
DOB_RE = re.compile(r"\b(?:dob|date of birth)\s*[:\-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.I)
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
ID_RE = re.compile(r"\b(?:mrn|hospital id|patient id|national id|ssn)\s*[:#-]?\s*[A-Z0-9-]{4,}\b", re.I)
ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,5}\s+"
    r"(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|building|bldg|apartment|apt)\b",
    re.I,
)
NAME_RE = re.compile(
    r"\b(?:my name is|patient name is|name:)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b",
    re.I,
)

ALLOWED_CONTEXT_KEYS = {
    "age",
    "age_range",
    "sex",
    "symptoms",
    "red_flags",
    "comorbidities",
    "pregnancy_status",
    "immunocompromised",
    "medications",
}

FREE_TEXT_KEYS = {
    "name",
    "email",
    "phone",
    "address",
    "dob",
    "date_of_birth",
    "hospital_id",
    "patient_id",
    "national_id",
    "location",
    "notes",
    "free_text",
}


def _age_range(age: Any) -> str | None:
    try:
        value = int(age)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    if value < 1:
        return "infant"
    if value < 13:
        return "child"
    if value < 18:
        return "adolescent"
    if value < 40:
        return "adult"
    if value < 65:
        return "middle-aged adult"
    return "older adult"


def _clean_list(values: Any, removed: list[str]) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        text, markers = sanitize_question(item)
        removed.extend(markers)
        text = text.strip()
        if text:
            cleaned.append(text)
    return cleaned


def sanitize_question(text: str) -> tuple[str, list[str]]:
    """Remove obvious identifiers from free text using deterministic regex rules."""
    removed: list[str] = []
    sanitized = text or ""

    patterns = [
        ("email", EMAIL_RE),
        ("phone_number", PHONE_RE),
        ("date_of_birth", DOB_RE),
        ("exact_date", DATE_RE),
        ("hospital_or_national_id", ID_RE),
        ("address", ADDRESS_RE),
        ("name", NAME_RE),
    ]
    for label, pattern in patterns:
        if pattern.search(sanitized):
            removed.append(label)
            sanitized = pattern.sub(" ", sanitized)

    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized, removed


def sanitize_patient_context(context: dict[str, Any]) -> tuple[dict[str, Any], PrivacyReport]:
    """Return de-identified patient context plus a privacy report."""
    removed: list[str] = []
    kept: list[str] = []
    sanitized: dict[str, Any] = {}

    for key, value in (context or {}).items():
        normalized_key = key.lower()
        if normalized_key in FREE_TEXT_KEYS or normalized_key not in ALLOWED_CONTEXT_KEYS:
            if value not in (None, "", [], {}):
                removed.append(normalized_key)
            continue

        if normalized_key == "age":
            age_range = _age_range(value)
            if age_range:
                sanitized["age_range"] = age_range
                kept.append("age_range")
            removed.append("exact_age")
            continue

        if normalized_key in {"symptoms", "red_flags", "comorbidities", "medications"}:
            cleaned = _clean_list(value, removed)
            if cleaned:
                sanitized[normalized_key] = cleaned
                kept.append(normalized_key)
            continue

        if normalized_key in {"sex", "pregnancy_status"} and isinstance(value, str):
            cleaned, markers = sanitize_question(value)
            removed.extend(markers)
            if cleaned:
                sanitized[normalized_key] = cleaned.lower()
                kept.append(normalized_key)
            continue

        if normalized_key == "immunocompromised" and isinstance(value, bool):
            sanitized[normalized_key] = value
            kept.append(normalized_key)

    report = PrivacyReport(
        removed_identifiers=sorted(set(removed)),
        kept_medical_context=sorted(set(kept)),
        deidentified=True,
    )
    return sanitized, report


def build_privacy_safe_terms(request: WebEvidenceRequest) -> tuple[list[str], PrivacyReport]:
    """Build search terms from the question and allowed de-identified context."""
    question, removed_from_question = sanitize_question(request.clinical_question)
    context, report = sanitize_patient_context(request.context_dict())
    removed = set(report.removed_identifiers) | set(removed_from_question)

    terms = [question]
    for key in ("age_range", "sex", "symptoms", "red_flags", "comorbidities", "pregnancy_status", "medications"):
        value = context.get(key)
        if isinstance(value, list):
            terms.extend(value)
        elif value not in (None, "", False):
            terms.append(str(value))
    if context.get("immunocompromised"):
        terms.append("immunocompromised")

    privacy_report = PrivacyReport(
        removed_identifiers=sorted(removed),
        kept_medical_context=report.kept_medical_context,
        deidentified=True,
    )
    return [term for term in terms if term], privacy_report
