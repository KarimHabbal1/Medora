"""Local Ollama backend for the web evidence agent."""

from __future__ import annotations

import json
import os
from typing import Any

from agents.web_evidence.llm.base import BaseLocalLLM, parse_json_object


class OllamaLocalLLM(BaseLocalLLM):
    """HTTP client for a locally running Ollama server."""

    provider = "ollama"

    def __init__(self, model: str | None = None, base_url: str | None = None, timeout_seconds: int = 45) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.model = model or os.getenv("MEDORA_LOCAL_LLM_MODEL") or "llama3.1:8b"
        self.timeout_seconds = timeout_seconds

    def _generate(self, system_prompt: str, user_prompt: str, json_mode: bool) -> str:
        try:
            import requests

            payload: dict[str, Any] = {
                "model": self.model,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "options": {"temperature": 0},
            }
            if json_mode:
                payload["format"] = "json"
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
                headers={"User-Agent": "MedoraWebEvidenceAgent/1.0"},
            )
            response.raise_for_status()
            data = response.json()
            return str(data.get("response", ""))
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: ollama_unavailable: {exc}"

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_hint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate JSON through local Ollama and parse the response."""
        prompt = user_prompt
        if schema_hint:
            prompt = f"{user_prompt}\n\nReturn JSON matching this schema hint:\n{json.dumps(schema_hint, indent=2)}"
        text = self._generate(system_prompt, prompt, json_mode=True)
        if text.startswith("ERROR:"):
            return {"error": text, "provider": self.provider, "model": self.model}
        try:
            return parse_json_object(text)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"json_parse_error: {exc}", "raw_text": text[:500], "provider": self.provider, "model": self.model}

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text through local Ollama."""
        return self._generate(system_prompt, user_prompt, json_mode=False)
