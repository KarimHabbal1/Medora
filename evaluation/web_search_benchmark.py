"""
Web Search Benchmark — MedCaseReasoning Test Set C
Tests the web search agent (SearXNG → fetch → LLM) as a standalone diagnostic engine.

Question: When the textbook doesn't have the answer, can web search fill the gap?

Usage:
    python evaluation/web_search_benchmark.py
    python evaluation/web_search_benchmark.py --num-cases 10 --models gpt-5.4-mini
    python evaluation/web_search_benchmark.py --searxng-url http://localhost:8080 --provider openai
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: 'openai' package is not installed.\nInstall it with: pip install openai")
    sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package is not installed.\nInstall it with: pip install datasets")
    sys.exit(1)

from agents.web_search import search_medical_evidence, make_llm  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "benchmark"
JUDGE_MODEL_DEFAULT = "gpt-5.4-mini"
SLEEP_BETWEEN_CASES = 2  # seconds — avoid overwhelming SearXNG


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
_judge_client: OpenAI | None = None


def _get_judge_client() -> OpenAI:
    global _judge_client
    if _judge_client is None:
        _judge_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _judge_client


JUDGE_SYSTEM_PROMPT = """\
You are a medical diagnosis evaluator. Given a predicted diagnosis and a ground truth diagnosis,
determine if they match clinically.

Return a JSON object with one field:
{
    "match": "exact_match" | "semantic_match" | "partial_match" | "mismatch"
}

Definitions:
- exact_match: identical or trivially equivalent (e.g., spacing, punctuation differences)
- semantic_match: clinically equivalent — same condition, possibly different wording
- partial_match: overlapping but incomplete — predicted is a superset, subset, or related diagnosis
- mismatch: clearly different diagnoses

Return ONLY valid JSON."""


def _judge_match(predicted: str, ground_truth: str, judge_model: str) -> str:
    """
    Three-tier matching:
    1. String similarity > 0.8 → exact_match
    2. GPT judge → semantic_match / partial_match / mismatch
    3. Fallback to string similarity on judge failure
    """
    from difflib import SequenceMatcher

    pred_norm = predicted.strip().lower()
    gt_norm = ground_truth.strip().lower()

    # Tier 1: string similarity
    ratio = SequenceMatcher(None, pred_norm, gt_norm).ratio()
    if ratio >= 0.8:
        return "exact_match"

    # Tier 2: GPT judge
    try:
        client = _get_judge_client()
        resp = client.chat.completions.create(
            model=judge_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Predicted diagnosis: {predicted}\n"
                        f"Ground truth diagnosis: {ground_truth}"
                    ),
                },
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content or "{}"
        result = json.loads(raw)
        match = result.get("match", "mismatch")
        if match in ("exact_match", "semantic_match", "partial_match", "mismatch"):
            return match
        return "mismatch"
    except Exception:  # noqa: BLE001
        # Tier 3: fallback string similarity
        if ratio >= 0.5:
            return "partial_match"
        return "mismatch"


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------
def load_test_cases(num_cases: int) -> list[dict]:
    """Load MedCaseReasoning Test Set C from HuggingFace."""
    print(f"Loading MedCaseReasoning (test split C) — {num_cases} cases ...")
    ds = load_dataset("zou-lab/MedCaseReasoning", split="test")
    cases = []
    for i, row in enumerate(ds):
        if i >= num_cases:
            break
        cases.append(
            {
                "case_idx": i + 1,
                "case_prompt": str(row.get("case_prompt") or row.get("question") or ""),
                "final_diagnosis": str(
                    row.get("final_diagnosis") or row.get("answer") or ""
                ),
            }
        )
    print(f"  Loaded {len(cases)} cases.\n")
    return cases


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------
def run_case(
    case: dict,
    llm,
    searxng_url: str,
    max_sources: int,
    judge_model: str,
) -> dict:
    """Run web search on one case and evaluate against ground truth."""
    case_prompt = case["case_prompt"]
    ground_truth = case["final_diagnosis"]

    t0 = time.monotonic()
    predicted_diagnosis = ""
    num_sources = 0
    error = None

    try:
        result = search_medical_evidence(
            symptoms=case_prompt,
            llm=llm,
            searxng_url=searxng_url,
            max_sources=max_sources,
        )
        predicted_diagnosis = result.get("primary_diagnosis", "").strip()
        num_sources = len(result.get("sources", []))
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    latency = round(time.monotonic() - t0, 3)

    # Match evaluation
    if not predicted_diagnosis or error:
        match_type = "mismatch"
        is_correct = False
    else:
        match_type = _judge_match(predicted_diagnosis, ground_truth, judge_model)
        is_correct = match_type in ("exact_match", "semantic_match")

    return {
        "case_idx": case["case_idx"],
        "case_prompt_preview": case_prompt[:200],
        "ground_truth": ground_truth,
        "predicted_diagnosis": predicted_diagnosis,
        "match_type": match_type,
        "is_correct": is_correct,
        "num_sources": num_sources,
        "latency_seconds": latency,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------
def compute_summary(results: list[dict], model: str, num_cases: int) -> dict:
    completed = [r for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]

    match_breakdown: dict[str, int] = {}
    for r in results:
        mt = r["match_type"]
        match_breakdown[mt] = match_breakdown.get(mt, 0) + 1

    correct = sum(1 for r in results if r["is_correct"])
    accuracy = round(correct / len(results), 4) if results else 0.0

    latencies = sorted(r["latency_seconds"] for r in completed)
    mean_lat = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    median_lat = round(latencies[len(latencies) // 2], 3) if latencies else 0.0
    p95_lat = (
        round(latencies[int(len(latencies) * 0.95)], 3) if len(latencies) >= 20 else
        round(latencies[-1], 3) if latencies else 0.0
    )

    sources_counts = [r["num_sources"] for r in completed]
    avg_sources = round(sum(sources_counts) / len(sources_counts), 2) if sources_counts else 0.0

    return {
        "model": model,
        "num_cases": num_cases,
        "num_completed": len(completed),
        "num_errors": len(errors),
        "accuracy": accuracy,
        "match_breakdown": match_breakdown,
        "mean_latency_s": mean_lat,
        "median_latency_s": median_lat,
        "p95_latency_s": p95_lat,
        "avg_sources_per_case": avg_sources,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def print_results_table(summary: dict, model: str) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Web Search Benchmark — {model}")
    print(sep)
    print(f"  Cases run        : {summary['num_cases']}")
    print(f"  Completed        : {summary['num_completed']}")
    print(f"  Errors           : {summary['num_errors']}")
    print(f"  Accuracy         : {summary['accuracy'] * 100:.1f}%")
    print()
    print("  Match breakdown:")
    for mt, count in sorted(summary["match_breakdown"].items()):
        print(f"    {mt:<20}: {count}")
    print()
    print(f"  Mean latency     : {summary['mean_latency_s']}s")
    print(f"  Median latency   : {summary['median_latency_s']}s")
    print(f"  P95 latency      : {summary['p95_latency_s']}s")
    print(f"  Avg sources/case : {summary['avg_sources_per_case']}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_results(
    all_results: dict[str, list[dict]],
    summaries: dict[str, dict],
    num_cases: int,
    models: list[str],
    searxng_url: str,
    judge_model: str,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"web_search_benchmark_{ts}.json"

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark": "web_search",
        "dataset": "zou-lab/MedCaseReasoning",
        "split": "test",
        "num_cases": num_cases,
        "models": models,
        "searxng_url": searxng_url,
        "judge_model": judge_model,
        "summaries": summaries,
        "per_case_results": all_results,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"  Results saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the web search agent on MedCaseReasoning Test Set C."
    )
    parser.add_argument(
        "--models", nargs="+", default=["gpt-5.4-mini"], help="Models to test"
    )
    parser.add_argument("--num-cases", type=int, default=50, help="Number of test cases")
    parser.add_argument(
        "--searxng-url",
        default=None,
        help="SearXNG base URL (overrides SEARXNG_BASE_URL env var)",
    )
    parser.add_argument(
        "--judge-model", default=JUDGE_MODEL_DEFAULT, help="OpenAI model used as judge"
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider for the web search agent: ollama | openai",
    )
    parser.add_argument(
        "--ollama-url", default=None, help="Ollama base URL (e.g. http://localhost:11434)"
    )
    parser.add_argument(
        "--max-sources", type=int, default=5, help="Max sources to fetch per case"
    )
    args = parser.parse_args()

    searxng_url = (args.searxng_url or os.getenv("SEARXNG_BASE_URL") or "").rstrip("/")
    if not searxng_url:
        print(
            "ERROR: No SearXNG URL provided.\n"
            "Set SEARXNG_BASE_URL in .env or pass --searxng-url."
        )
        sys.exit(1)

    print("\nMedora — Web Search Benchmark")
    print(f"  Dataset       : zou-lab/MedCaseReasoning (test)")
    print(f"  Num cases     : {args.num_cases}")
    print(f"  Models        : {args.models}")
    print(f"  SearXNG URL   : {searxng_url}")
    print(f"  Judge model   : {args.judge_model}")
    print(f"  Provider      : {args.provider or 'auto (env)'}")
    print(f"  Max sources   : {args.max_sources}\n")

    # Load dataset once
    cases = load_test_cases(args.num_cases)

    all_results: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}

    for model in args.models:
        print(f"--- Model: {model} ---\n")

        # Build LLM once per model
        llm = make_llm(model=model, provider=args.provider, ollama_url=args.ollama_url)

        model_results = []
        for case in cases:
            idx = case["case_idx"]
            preview = case["case_prompt"][:80].replace("\n", " ")
            print(f"  [{idx:03d}/{len(cases)}] {preview}...")

            result = run_case(
                case=case,
                llm=llm,
                searxng_url=searxng_url,
                max_sources=args.max_sources,
                judge_model=args.judge_model,
            )

            status = "CORRECT" if result["is_correct"] else result["match_type"].upper()
            print(
                f"           GT: {result['ground_truth'][:60]}"
            )
            print(
                f"        Pred: {result['predicted_diagnosis'][:60]}"
            )
            print(
                f"        [{status}] sources={result['num_sources']} "
                f"latency={result['latency_seconds']}s"
                + (f" ERROR: {result['error']}" if result["error"] else "")
            )
            print()

            model_results.append(result)

            if idx < len(cases):
                time.sleep(SLEEP_BETWEEN_CASES)

        all_results[model] = model_results
        summary = compute_summary(model_results, model, args.num_cases)
        summaries[model] = summary
        print_results_table(summary, model)

    save_results(
        all_results=all_results,
        summaries=summaries,
        num_cases=args.num_cases,
        models=args.models,
        searxng_url=searxng_url,
        judge_model=args.judge_model,
    )

    print("Web search benchmark complete.\n")


if __name__ == "__main__":
    main()
