"""Model-agnostic interface for local open-source LLM backends."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any


class BaseLocalLLM(ABC):
    """Base class for hospital-controlled local LLM providers."""

    provider: str = "base"

    @abstractmethod
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_hint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a JSON object. Implementations must not call hosted LLM APIs."""

    @abstractmethod
    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Generate plain text using a local model."""


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from a model response."""
    stripped = (text or "").strip()
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence:
        value = json.loads(fence.group(1))
        if isinstance(value, dict):
            return value

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(stripped[start : end + 1])
        if isinstance(value, dict):
            return value

    raise ValueError("No JSON object found in local LLM response.")


def strict_json_retry_prompt(user_prompt: str, schema_hint: dict[str, Any] | None = None) -> str:
    """Add stricter JSON-only instructions for a retry."""
    schema_text = json.dumps(schema_hint or {}, indent=2)
    return (
        f"{user_prompt}\n\n"
        "Retry requirement: return one valid JSON object only. "
        "Do not include markdown, commentary, or prose outside JSON.\n"
        f"Schema hint:\n{schema_text}"
    )


class NullLocalLLM(BaseLocalLLM):
    """Provider used when local LLM support is intentionally disabled."""

    provider = "none"

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_hint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a structured disabled-provider error."""
        return {"error": "local_llm_disabled", "provider": self.provider}

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Return a structured disabled-provider message."""
        return "ERROR: local_llm_disabled"


def get_local_llm(provider: str = "none", model: str | None = None) -> BaseLocalLLM:
    """Return the requested local LLM provider."""
    normalized = (provider or "none").lower()
    if normalized == "mock":
        from agents.web_evidence.llm.mock_llm import MockLocalLLM

        return MockLocalLLM(model=model)
    if normalized == "ollama":
        from agents.web_evidence.llm.ollama_client import OllamaLocalLLM

        return OllamaLocalLLM(model=model)
    if normalized == "hf":
        from agents.web_evidence.llm.hf_client import HuggingFaceLocalLLM

        return HuggingFaceLocalLLM(model=model)
    return NullLocalLLM()
