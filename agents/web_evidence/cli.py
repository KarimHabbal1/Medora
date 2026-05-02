"""Command-line interface for the Phase 6 web evidence agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.web_evidence.agent import run_web_evidence_agent
from agents.web_evidence.experiments.approaches import APPROACHES
from agents.web_evidence.schemas import WebEvidenceRequest
from agents.web_evidence.search_client import StaticSearchClient


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the Medora web evidence agent.")
    parser.add_argument("--question", required=True, help="Clinical question to investigate.")
    parser.add_argument("--approach", default="deterministic_only", choices=sorted(APPROACHES), help="Reasoning approach to run.")
    parser.add_argument("--provider", default=None, choices=["ollama", "hf", "mock", "none"], help="Local LLM provider.")
    parser.add_argument("--model", default=None, help="Local model name, for example llama3.1:8b.")
    parser.add_argument("--reason-for-web", default=None, help="Reason web evidence is needed.")
    parser.add_argument("--max-sources", type=int, default=5, help="Maximum ranked sources to include.")
    parser.add_argument("--mock", action="store_true", help="Use deterministic offline search and fetch mode.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional path to write JSON output.")
    return parser.parse_args()


def main() -> None:
    """Run the CLI and print pretty JSON."""
    args = parse_args()
    request = WebEvidenceRequest(
        clinical_question=args.question,
        patient_context=None,
        reason_for_web=args.reason_for_web,
    )
    client = StaticSearchClient() if args.mock else None
    provider = args.provider
    if args.mock and provider is None:
        provider = "mock"
    result = run_web_evidence_agent(
        request,
        max_sources=args.max_sources,
        search_client=client,
        use_mock_fetch=args.mock,
        approach=args.approach,
        llm_provider=provider,
        llm_model=args.model,
    )
    payload = result.to_dict()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
