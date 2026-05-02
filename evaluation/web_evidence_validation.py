"""Phase 6 web evidence approach comparison runner."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.web_evidence.experiments.approaches import APPROACHES
from agents.web_evidence.experiments.runner import run_approach_comparison
from config import RESULTS_DIR


def parse_args() -> argparse.Namespace:
    """Parse validation script arguments."""
    parser = argparse.ArgumentParser(description="Compare Phase 6 web evidence approaches.")
    parser.add_argument("--mock", action="store_true", help="Force deterministic mock search/fetch mode.")
    parser.add_argument("--provider", default="mock", choices=["ollama", "hf", "mock", "none"], help="Local LLM provider for LLM approaches.")
    parser.add_argument("--model", default=None, help="Local model name, for example llama3.1:8b.")
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=[
            "deterministic_only",
            "llm_synthesis_only",
            "llm_claims_and_synthesis",
            "full_llm_council",
            "hybrid_recommended",
        ],
        choices=sorted(APPROACHES),
        help="Approaches to compare.",
    )
    parser.add_argument("--max-questions", type=int, default=None, help="Limit number of validation questions.")
    parser.add_argument("--max-sources", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    """Run validation and save results."""
    args = parse_args()
    use_mock = args.mock or not os.getenv("SEARXNG_BASE_URL")
    provider = "mock" if use_mock and args.provider == "none" else args.provider
    payload = run_approach_comparison(
        approach_names=args.approaches,
        provider=provider,
        model=args.model,
        max_questions=args.max_questions,
        max_sources=args.max_sources,
        mock_mode=use_mock,
        output_dir=RESULTS_DIR,
    )
    print("Comparison table:")
    for row in payload["comparison_table"]:
        print(
            f"- {row['approach']}: overall={row['overall_score']} "
            f"source={row['source_quality_score']} evidence={row['evidence_completeness_score']} "
            f"safety={row['safety_score']} privacy={row['privacy_score']}"
        )
    print(f"Recommended approach: {payload['recommended_approach']}")
    print(f"Saved validation results to {payload['json_output_path']}")


if __name__ == "__main__":
    main()
