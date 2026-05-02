"""Local open-source LLM backends for the web evidence agent."""

from agents.web_evidence.llm.base import BaseLocalLLM, get_local_llm
from agents.web_evidence.llm.hf_client import HuggingFaceLocalLLM
from agents.web_evidence.llm.mock_llm import MockLocalLLM
from agents.web_evidence.llm.ollama_client import OllamaLocalLLM

__all__ = [
    "BaseLocalLLM",
    "HuggingFaceLocalLLM",
    "MockLocalLLM",
    "OllamaLocalLLM",
    "get_local_llm",
]
