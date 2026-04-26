"""Deterministic local mock LLM for offline tests and validation."""

from __future__ import annotations

import re
from typing import Any

from agents.web_evidence.llm.base import BaseLocalLLM


class MockLocalLLM(BaseLocalLLM):
    """Fake model that returns valid deterministic JSON without network calls."""

    provider = "mock"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or "mock-local-llm"

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_hint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return task-shaped JSON inferred from the prompt text."""
        system = system_prompt.lower()
        task = f"{system_prompt}\n{user_prompt}".lower()
        if "synthesis assistant" in system:
            return {
                "summary": (
                    "For physician review, supplied source text supports a preliminary evidence summary. "
                    "This is not a final diagnosis and requires clinician judgment."
                )
            }
        if "extraction assistant" in system or ("claim" in task and "source text" in task):
            return self._claims(user_prompt)
        if "conflict checker" in system:
            return {"decision": "accept_with_caution", "conflicts": [], "reason": "Mock LLM found no explicit conflict."}
        if "safety reviewer" in system:
            return {"decision": "accept", "reason": "Mock LLM found no direct patient instructions.", "warnings": []}
        if "final reviewer" in system or "final review" in system:
            return {"decision": "accept_with_caution", "reason": "Mock LLM accepts for physician review with deterministic caution."}
        return {"decision": "accept_with_caution", "reason": "Mock LLM fallback response."}

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Return cautious deterministic text."""
        return (
            "For physician review, the supplied sources suggest preliminary evidence relevant to the clinical question. "
            "This summary uses only provided source material, is not a final diagnosis, and requires clinician judgment."
        )

    def _claims(self, user_prompt: str) -> dict[str, Any]:
        url_match = re.search(r"Source URL:\s*(.+)", user_prompt)
        source_url = url_match.group(1).strip() if url_match else ""
        text_match = re.search(r"Source text:\s*(.*?)(?:\n\nReturn JSON:|\Z)", user_prompt, re.DOTALL)
        text = text_match.group(1).strip() if text_match else user_prompt
        sentences = re.split(r"(?<=[.!?])\s+", text)
        claims = []
        for sentence in sentences:
            cleaned = re.sub(r"\s+", " ", sentence).strip()
            if len(cleaned) < 35:
                continue
            if not any(word in cleaned.lower() for word in ("guideline", "recommend", "should", "warning", "screening", "treatment", "management")):
                continue
            claims.append(
                {
                    "claim": cleaned[:500],
                    "source_url": source_url,
                    "supporting_quote": cleaned[:240],
                    "confidence": "moderate",
                }
            )
            if len(claims) >= 3:
                break
        return {"claims": claims}
