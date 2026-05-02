"""Privacy-safe query construction for the web evidence agent."""

from __future__ import annotations

import re

from agents.web_evidence.config import TRUSTED_DOMAIN_HINTS
from agents.web_evidence.pii_sanitizer import build_privacy_safe_terms
from agents.web_evidence.schemas import PrivacyReport, WebEvidenceRequest


GUIDELINE_TRIGGERS = {
    "guideline",
    "guidelines",
    "recommendation",
    "recommendations",
    "latest",
    "updated",
    "treatment",
    "management",
    "screening",
    "vaccine",
    "vaccination",
    "warning",
}


def _normalize_query(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_search_query(request: WebEvidenceRequest) -> tuple[str, PrivacyReport]:
    """Build a web-search query that excludes raw patient identifiers."""
    terms, privacy_report = build_privacy_safe_terms(request)
    raw = " ".join(terms)
    lowered = raw.lower()

    additives = []
    if any(trigger in lowered for trigger in GUIDELINE_TRIGGERS):
        additives.extend(["guideline", "recommendation"])
    if "latest" in lowered or "updated" in lowered:
        additives.append("updated")
    additives.extend(TRUSTED_DOMAIN_HINTS)

    query = _normalize_query(" ".join([raw, *additives]))
    return query, privacy_report
