"""Optional local Hugging Face Transformers backend."""

from __future__ import annotations

import json
import os
from typing import Any

from agents.web_evidence.llm.base import BaseLocalLLM, parse_json_object


class HuggingFaceLocalLLM(BaseLocalLLM):
    """Local Transformers text-generation backend.

    The model is loaded lazily and with local_files_only=True so a hospital
    deployment can control model availability explicitly.
    """

    provider = "hf"

    def __init__(self, model: str | None = None, device: str | None = None, max_new_tokens: int = 512) -> None:
        self.model = model or os.getenv("MEDORA_HF_MODEL") or ""
        self.device = device or os.getenv("MEDORA_HF_DEVICE") or "cpu"
        self.max_new_tokens = max_new_tokens
        self._pipeline = None
        self._load_error: str | None = None

    def _ensure_pipeline(self):
        if self._pipeline is not None or self._load_error:
            return self._pipeline
        if not self.model:
            self._load_error = "MEDORA_HF_MODEL is not configured."
            return None
        try:
            from transformers import pipeline

            device_arg = -1 if self.device == "cpu" else self.device
            self._pipeline = pipeline(
                "text-generation",
                model=self.model,
                device=device_arg,
                local_files_only=True,
            )
            return self._pipeline
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"hf_backend_unavailable: {exc}"
            return None

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        pipe = self._ensure_pipeline()
        if pipe is None:
            return f"ERROR: {self._load_error}"
        prompt = f"{system_prompt}\n\n{user_prompt}"
        try:
            outputs = pipe(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_full_text=False,
            )
            return str(outputs[0].get("generated_text", ""))
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: hf_generation_failed: {exc}"

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_hint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate and parse a JSON object using a local HF model."""
        prompt = user_prompt
        if schema_hint:
            prompt = f"{user_prompt}\n\nReturn JSON matching this schema hint:\n{json.dumps(schema_hint, indent=2)}"
        text = self._generate(system_prompt, prompt)
        if text.startswith("ERROR:"):
            return {"error": text, "provider": self.provider, "model": self.model}
        try:
            return parse_json_object(text)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"json_parse_error: {exc}", "raw_text": text[:500], "provider": self.provider, "model": self.model}

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text using a local HF model."""
        return self._generate(system_prompt, user_prompt)
