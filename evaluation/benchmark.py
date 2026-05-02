"""
MedCaseReasoning Benchmark — Phase 8
=====================================
Evaluates multiple LLMs on the MedCaseReasoning dataset (897 cases) by running
each case through Medora's RAG pipeline (bi-encoder → BGE reranker → LLM) and
measuring correctness, latency, token usage, JSON adherence, and retrieval quality.

The RAG infrastructure (bi-encoder, reranker, ChromaDB) is loaded once and shared
across all model runs to ensure fair comparison.

Usage:
    python evaluation/benchmark.py --profile api
    python evaluation/benchmark.py --profile ollama --ollama-url http://my-ec2:11434
    python evaluation/benchmark.py --models gpt-5.4-mini
    python evaluation/benchmark.py --num-cases 100 --output-dir data/results/benchmark
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Environment ───────────────────────────────────────────────────────────────
from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
from config import (  # noqa: E402
    CHROMA_DIR,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    RERANK_TOP_K_RETRIEVE,
    RERANK_TOP_K_RETURN,
)

# ── RAG pipeline ──────────────────────────────────────────────────────────────
from rag.reranker import (  # noqa: E402
    retrieve_and_rerank,
    open_collection,
    load_bi_encoder,
    load_cross_encoder,
    detect_device,
)

# ── Prompt from triage agent ──────────────────────────────────────────────────
from agents.triage_agent import _DIAGNOSIS_SYSTEM, _chunks_to_context  # noqa: E402

# ── LangChain ─────────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

# ── Benchmark config ──────────────────────────────────────────────────────────
from evaluation.benchmark_config import (  # noqa: E402
    ALL_MODELS,
    JUDGE_MODEL,
    PROFILES,
    get_models_by_names,
    get_models_by_profile,
)

# Alias for backward compatibility within this module
BENCHMARK_MODELS = ALL_MODELS

# Timeout (seconds) per case — sourced from model config when available
TIMEOUT_OPENAI = 120
TIMEOUT_OLLAMA = 300


# ─────────────────────────────────────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm(model_config: dict, ollama_url: str):
    """Instantiate the correct LangChain chat model for a given model config."""
    provider = model_config["provider"]
    model_id = model_config["model_id"]

    if provider == "openai":
        return ChatOpenAI(model=model_id, temperature=0)

    if provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama  # type: ignore
        except ImportError:
            try:
                from langchain_ollama import ChatOllama  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "Install langchain-community or langchain-ollama to use Ollama models. "
                    "Run: pip install langchain-community"
                ) from exc

        return ChatOllama(
            model=model_id,
            base_url=ollama_url,
            temperature=0,
        )

    raise ValueError(f"Unknown provider: {provider!r}")


def _check_ollama_available(model_config: dict, ollama_url: str) -> bool:
    """
    Attempt a lightweight ping to the Ollama server to verify it is reachable
    and the requested model is present.
    Returns True if available, False with a printed warning otherwise.
    """
    import urllib.error
    import urllib.request

    name = model_config["name"]
    model_id = model_config["model_id"]

    # Check server reachability
    try:
        with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError) as exc:
        print(f"  [WARN] Ollama server at {ollama_url} not reachable: {exc}")
        print(f"  [SKIP] Skipping model '{name}'.")
        return False

    # Check model presence
    available_models = [m.get("name", "") for m in data.get("models", [])]
    if not any(model_id in m for m in available_models):
        print(f"  [WARN] Model '{model_id}' not found on Ollama server.")
        print(f"         Available: {available_models}")
        print(f"  [SKIP] Skipping model '{name}'.")
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_dataset(split: str = "test", num_cases: int | None = None) -> list[dict]:
    """
    Load the MedCaseReasoning dataset from HuggingFace.

    Each case has:
        case_prompt     — natural language patient presentation
        final_diagnosis — ground truth diagnosis string
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Install the HuggingFace datasets library: pip install datasets"
        ) from exc

    print(f"Loading MedCaseReasoning dataset (split={split!r})...")
    ds = load_dataset("zou-lab/MedCaseReasoning", split=split)
    cases = list(ds)

    if num_cases is not None:
        cases = cases[:num_cases]

    print(f"  Loaded {len(cases):,} cases.")
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# Diagnosis extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

_PRIMARY_DX_PATTERN = re.compile(
    r"##\s*Primary\s+Diagnosis\s*\n+(.+?)(?:\n\n|\n##|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_primary_diagnosis(report_text: str) -> str | None:
    """
    Parse the '## Primary Diagnosis' section from the model's report.
    Returns the first non-empty line of that section, stripped of markdown.
    Returns None if the section is not found.
    """
    m = _PRIMARY_DX_PATTERN.search(report_text)
    if not m:
        return None

    section_text = m.group(1).strip()
    # Take only the first meaningful line
    for line in section_text.splitlines():
        line = line.strip().lstrip("-*•").strip()
        if line:
            # Strip confidence annotations like "(confidence: high)"
            line = re.sub(r"\s*[\(\[].*?confidence.*?[\)\]]", "", line, flags=re.IGNORECASE).strip()
            # Strip trailing colon
            line = line.rstrip(":").strip()
            return line if line else None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Diagnosis matching
# ─────────────────────────────────────────────────────────────────────────────

def _string_similarity(a: str, b: str) -> float:
    """Normalized string similarity in [0, 1]."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _judge_diagnosis_match(
    system_diagnosis: str,
    ground_truth: str,
    judge_llm,
) -> str:
    """
    Use the judge LLM (gpt-5.4-mini) to determine whether the system's diagnosis
    matches the ground truth.

    Returns one of:
        "exact_match"    — string similarity > 0.8
        "semantic_match" — LLM says they refer to the same condition
        "partial_match"  — LLM says they are related but not the same
        "mismatch"       — LLM says they are different conditions
    """
    # Fast path: high string similarity → exact match
    if _string_similarity(system_diagnosis, ground_truth) >= 0.8:
        return "exact_match"

    judge_prompt = f"""\
Does the system diagnosis match the ground truth diagnosis?
They may use different medical terminology for the same condition.

Ground truth:    {ground_truth}
System diagnosis: {system_diagnosis}

Respond with ONLY one of these four words (no explanation):
  semantic_match  — same condition, different wording
  partial_match   — related / overlapping, but not the same
  mismatch        — clearly different conditions"""

    try:
        response = judge_llm.invoke([
            SystemMessage(content=(
                "You are a medical terminology expert. Given two diagnosis strings, "
                "decide whether they refer to the same clinical condition."
            )),
            HumanMessage(content=judge_prompt),
        ])
        verdict = response.content.strip().lower().split()[0]
        if verdict in ("semantic_match", "partial_match", "mismatch"):
            return verdict
        # Fallback if the model returns something unexpected
        return "mismatch"
    except Exception:
        # If judge fails, fall back to string similarity alone
        sim = _string_similarity(system_diagnosis, ground_truth)
        if sim >= 0.6:
            return "partial_match"
        return "mismatch"


# ─────────────────────────────────────────────────────────────────────────────
# Token counting
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token (GPT tokenizer approximation)."""
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────────────────────
# BenchmarkRunner
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    Orchestrates multi-LLM benchmarking on the MedCaseReasoning dataset.

    The RAG infrastructure (bi-encoder, reranker, ChromaDB) is loaded once in
    __init__ and reused across all model runs.
    """

    def __init__(
        self,
        models: list[dict],
        dataset_split: str = "test",
        num_cases: int | None = None,
        ollama_url: str = "http://localhost:11434",
        judge_model: str = JUDGE_MODEL,
        retrieve_k: int = RERANK_TOP_K_RETRIEVE,
        return_k: int = RERANK_TOP_K_RETURN,
        no_rag: bool = False,
    ):
        self.models = models
        self.ollama_url = ollama_url
        self.retrieve_k = retrieve_k
        self.return_k = return_k
        self.no_rag = no_rag

        # ── Load dataset ──────────────────────────────────────────────────────
        self.cases = _load_dataset(dataset_split, num_cases)

        # ── Load RAG pipeline (shared) — skip if --no-rag ────────────────────
        if not no_rag:
            print("\nLoading RAG pipeline (shared across all models)...")
            self._device = detect_device()
            self._collection = open_collection(CHROMA_DIR)
            self._bi_encoder = load_bi_encoder(EMBEDDING_MODEL, self._device)
            self._reranker, self._reranker_device = load_cross_encoder(RERANKER_MODEL)
            print(
                f"  RAG pipeline ready "
                f"(bi-encoder on {self._device}, reranker on {self._reranker_device})"
            )
        else:
            print("\n  [NO-RAG MODE] Skipping RAG pipeline — raw model output only.")

        # ── Load judge model ──────────────────────────────────────────────────
        print(f"\nLoading judge model ({judge_model})...")
        self._judge_llm = ChatOpenAI(model=judge_model, temperature=0)
        print("  Judge model ready.")

    # ── Per-case runner ───────────────────────────────────────────────────────

    def run_single_case(
        self,
        case: dict,
        llm: Any,
        model_config: dict,
        case_idx: int,
        total_cases: int,
    ) -> dict:
        """
        Run one MedCaseReasoning case through the pipeline.

        Pipeline:
            1. Use case_prompt as the RAG query
            2. Retrieve + rerank chunks from ChromaDB
            3. LLM generates diagnosis from case_prompt + retrieved context
            4. Extract primary diagnosis from report
            5. Judge match against ground_truth

        Returns a per-case result dict.
        """
        model_name = model_config["name"]
        provider = model_config["provider"]
        case_prompt: str = case.get("case_prompt", "") or case.get("question", "") or ""
        ground_truth: str = case.get("final_diagnosis", "") or case.get("answer", "")

        timeout = TIMEOUT_OPENAI if provider == "openai" else TIMEOUT_OLLAMA

        result: dict = {
            "case_idx": case_idx,
            "model": model_name,
            "case_prompt_preview": case_prompt[:200],
            "ground_truth": ground_truth,
            "system_diagnosis": None,
            "match_type": "mismatch",
            "is_correct": False,
            "retrieval_relevant": False,
            "json_adherence": False,
            "latency_seconds": None,
            "tokens_used": 0,
            "error": None,
        }

        print(
            f"  [{case_idx:>4}/{total_cases}] {model_name} — "
            f"{case_prompt[:80].strip()!r}..."
        )

        t_start = time.perf_counter()

        try:
            if self.no_rag:
                # ── NO-RAG MODE: give case directly to LLM ──────────────────
                user_content = (
                    f"Patient presentation:\n{case_prompt}\n\n"
                    f"Based on your medical knowledge, produce a structured diagnosis report."
                )
            else:
                # ── Step 1: RAG retrieval + reranking ─────────────────────────
                chunks = retrieve_and_rerank(
                    case_prompt,
                    self._collection,
                    self._bi_encoder,
                    self._reranker,
                    self.retrieve_k,
                    self.return_k,
                )

                # ── Step 2: Check retrieval relevance ─────────────────────────
                gt_words = set(ground_truth.lower().split())
                for chunk in chunks:
                    chunk_text = chunk.get("text", "").lower()
                    if gt_words:
                        overlap = sum(1 for w in gt_words if w in chunk_text)
                        if overlap / len(gt_words) >= 0.4:
                            result["retrieval_relevant"] = True
                            break

                # ── Step 3: Build prompt ──────────────────────────────────────
                context = _chunks_to_context(chunks)
                user_content = (
                    f"Patient presentation:\n{case_prompt}\n\n"
                    f"Retrieved medical textbook passages:\n\n{context}"
                )

            prompt_tokens = _estimate_tokens(_DIAGNOSIS_SYSTEM + user_content)

            # ── Step 4: LLM generation with timeout ───────────────────────────
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError(f"LLM call exceeded {timeout}s timeout")

            # signal.alarm only works on Unix; skip on Windows
            has_alarm = hasattr(signal, "SIGALRM")
            if has_alarm:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout)

            try:
                response = llm.invoke([
                    SystemMessage(content=_DIAGNOSIS_SYSTEM),
                    HumanMessage(content=user_content),
                ])
                report_text: str = response.content.strip()
            finally:
                if has_alarm:
                    signal.alarm(0)  # cancel alarm

            t_end = time.perf_counter()
            result["latency_seconds"] = round(t_end - t_start, 3)

            response_tokens = _estimate_tokens(report_text)
            result["tokens_used"] = prompt_tokens + response_tokens

            # ── Step 5: Extract primary diagnosis ─────────────────────────────
            system_dx = _extract_primary_diagnosis(report_text)
            result["json_adherence"] = system_dx is not None  # proxy: found the section

            if system_dx is None:
                # Could not parse a primary diagnosis from the report
                result["error"] = "Could not extract Primary Diagnosis section from report"
                print(f"    [WARN] No Primary Diagnosis section found.")
                return result

            result["system_diagnosis"] = system_dx

            # ── Step 6: Judge match ───────────────────────────────────────────
            match_type = _judge_diagnosis_match(
                system_dx, ground_truth, self._judge_llm
            )
            result["match_type"] = match_type
            result["is_correct"] = match_type in ("exact_match", "semantic_match")

            correctness_flag = "CORRECT" if result["is_correct"] else "WRONG"
            print(
                f"    [{correctness_flag}] match={match_type}  "
                f"gt={ground_truth!r}  sys={system_dx!r}  "
                f"lat={result['latency_seconds']:.1f}s"
            )

        except TimeoutError as exc:
            t_end = time.perf_counter()
            result["latency_seconds"] = round(t_end - t_start, 3)
            result["error"] = str(exc)
            print(f"    [TIMEOUT] {exc}")

        except Exception as exc:
            t_end = time.perf_counter()
            result["latency_seconds"] = round(t_end - t_start, 3)
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    [ERROR] {exc}")
            traceback.print_exc()

        return result

    # ── Per-model runner ──────────────────────────────────────────────────────

    def run_model(self, model_config: dict) -> list[dict]:
        """Run all cases through one model and return per-case results."""
        model_name = model_config["name"]
        provider = model_config["provider"]

        print(f"\n{'='*70}")
        print(f"  Benchmarking model: {model_name}  ({provider})")
        print(f"{'='*70}")

        # ── Availability check for Ollama ─────────────────────────────────────
        if provider == "ollama":
            if not _check_ollama_available(model_config, self.ollama_url):
                return []

        # ── Build LLM ─────────────────────────────────────────────────────────
        try:
            llm = _build_llm(model_config, self.ollama_url)
        except Exception as exc:
            print(f"  [ERROR] Could not build LLM for '{model_name}': {exc}")
            return []

        # ── Run cases ─────────────────────────────────────────────────────────
        results: list[dict] = []
        total = len(self.cases)

        for i, case in enumerate(self.cases, start=1):
            case_result = self.run_single_case(case, llm, model_config, i, total)
            results.append(case_result)

        return results

    # ── All models ────────────────────────────────────────────────────────────

    def run_all(self) -> dict[str, list[dict]]:
        """Run all configured models sequentially. Returns {model_name: [case_results]}."""
        all_results: dict[str, list[dict]] = {}

        for model_config in self.models:
            name = model_config["name"]
            case_results = self.run_model(model_config)
            all_results[name] = case_results

        return all_results

    # ── Metric computation ────────────────────────────────────────────────────

    def compute_metrics(self, results: list[dict]) -> dict:
        """Compute aggregate metrics from a list of per-case result dicts."""
        if not results:
            return {"num_cases": 0, "note": "no results"}

        num_cases = len(results)
        completed = [r for r in results if r.get("error") is None or r.get("system_diagnosis")]
        latencies = [r["latency_seconds"] for r in results if r["latency_seconds"] is not None]

        # Accuracy
        correct = sum(1 for r in results if r.get("is_correct", False))
        accuracy = correct / num_cases if num_cases else 0.0

        # Match breakdown
        match_counts: dict[str, int] = {}
        for r in results:
            mt = r.get("match_type", "mismatch")
            match_counts[mt] = match_counts.get(mt, 0) + 1

        # Latency
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
        sorted_lat = sorted(latencies)
        median_lat = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0
        p95_idx = int(len(sorted_lat) * 0.95)
        p95_lat = sorted_lat[min(p95_idx, len(sorted_lat) - 1)] if sorted_lat else 0.0

        # JSON adherence (found Primary Diagnosis section)
        json_ok = sum(1 for r in results if r.get("json_adherence", False))
        json_error_rate = 1.0 - (json_ok / num_cases) if num_cases else 1.0

        # Retrieval hit rate
        retrieval_hits = sum(1 for r in results if r.get("retrieval_relevant", False))
        retrieval_hit_rate = retrieval_hits / num_cases if num_cases else 0.0

        # Tokens
        total_tokens = sum(r.get("tokens_used", 0) for r in results)

        # Errors
        error_count = sum(1 for r in results if r.get("error") and not r.get("system_diagnosis"))

        return {
            "num_cases": num_cases,
            "num_completed": len(completed),
            "num_errors": error_count,
            "accuracy": round(accuracy, 4),
            "match_breakdown": match_counts,
            "mean_latency_s": round(mean_lat, 3),
            "median_latency_s": round(median_lat, 3),
            "p95_latency_s": round(p95_lat, 3),
            "json_error_rate": round(json_error_rate, 4),
            "retrieval_hit_rate": round(retrieval_hit_rate, 4),
            "total_tokens": total_tokens,
        }

    # ── Save results ──────────────────────────────────────────────────────────

    def save_results(
        self,
        all_results: dict[str, list[dict]],
        output_dir: Path | None = None,
    ) -> tuple[Path, Path]:
        """
        Save per-case and summary results to JSON files.

        Returns (full_results_path, summary_path).
        """
        if output_dir is None:
            output_dir = PROJECT_ROOT / "data" / "results" / "benchmark"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # ── Full per-case results ─────────────────────────────────────────────
        full_path = output_dir / f"benchmark_results_{ts}.json"
        full_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_cases": len(self.cases),
            "models": [m["name"] for m in self.models],
            "rag_config": {
                "bi_encoder": EMBEDDING_MODEL,
                "reranker": RERANKER_MODEL,
                "retrieve_k": self.retrieve_k,
                "return_k": self.return_k,
            },
            "results": all_results,
        }
        with open(full_path, "w", encoding="utf-8") as fh:
            json.dump(full_payload, fh, indent=2, ensure_ascii=False)
        print(f"\n  Full results saved to: {full_path}")

        # ── Summary (aggregate metrics per model) ─────────────────────────────
        summary_path = output_dir / f"benchmark_summary_{ts}.json"
        summary: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_cases": len(self.cases),
            "models": {},
        }
        for model_name, case_results in all_results.items():
            summary["models"][model_name] = self.compute_metrics(case_results)

        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"  Summary saved to:      {summary_path}")

        return full_path, summary_path

    # ── Comparison table ──────────────────────────────────────────────────────

    def print_comparison_table(self, all_results: dict[str, list[dict]]) -> None:
        """Print a side-by-side model comparison table to stdout."""

        metrics_per_model: dict[str, dict] = {
            name: self.compute_metrics(results)
            for name, results in all_results.items()
        }

        model_names = list(metrics_per_model.keys())
        col_w = 18

        separator = "-" * (28 + col_w * len(model_names))
        header_sep = "=" * (28 + col_w * len(model_names))

        print(f"\n{header_sep}")
        print("  Medora — LLM Benchmark Comparison (MedCaseReasoning)")
        print(f"  Cases per model: {len(self.cases):,}")
        print(header_sep)

        # Header
        row = f"  {'Metric':<26}"
        for name in model_names:
            row += f"  {name:<{col_w - 2}}"
        print(row)
        print(separator)

        def _fmt_pct(v: float) -> str:
            return f"{v:.1%}"

        def _fmt_s(v: float) -> str:
            return f"{v:.2f}s"

        def _fmt_int(v: int) -> str:
            return f"{v:,}"

        metrics_to_display: list[tuple[str, str, callable]] = [
            ("Accuracy",           "accuracy",           _fmt_pct),
            ("JSON Error Rate",    "json_error_rate",    _fmt_pct),
            ("Retrieval Hit Rate", "retrieval_hit_rate", _fmt_pct),
            ("Mean Latency",       "mean_latency_s",     _fmt_s),
            ("Median Latency",     "median_latency_s",   _fmt_s),
            ("P95 Latency",        "p95_latency_s",      _fmt_s),
            ("Total Tokens",       "total_tokens",       _fmt_int),
            ("Errors",             "num_errors",         _fmt_int),
            ("Cases Run",          "num_cases",          _fmt_int),
        ]

        for label, key, fmt in metrics_to_display:
            row = f"  {label:<26}"
            for name in model_names:
                val = metrics_per_model[name].get(key, 0)
                row += f"  {fmt(val):<{col_w - 2}}"
            print(row)

        print(separator)

        # Match breakdown sub-table
        print("  Match breakdown:")
        for match_type in ("exact_match", "semantic_match", "partial_match", "mismatch"):
            row = f"    {match_type:<24}"
            for name in model_names:
                count = metrics_per_model[name].get("match_breakdown", {}).get(match_type, 0)
                total = metrics_per_model[name].get("num_cases", 1) or 1
                pct = count / total
                row += f"  {count:>4} ({pct:.0%}){'':<{col_w - 12}}"
            print(row)

        print(header_sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Medora Phase 8 — Multi-LLM Benchmark on MedCaseReasoning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        default=None,
        choices=list(PROFILES.keys()),
        metavar="PROFILE",
        help=(
            "Execution profile: api (OpenAI models, run locally), "
            "ollama (local models, run on EC2), quick, api-ceiling, full. "
            f"Available: {list(PROFILES.keys())}. "
            "Overridden by --models if both are specified."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL_NAME",
        help=(
            "Specific model names to benchmark (must match 'name' in benchmark_config.py). "
            "Overrides --profile. "
            f"Available: {[m['name'] for m in ALL_MODELS]}."
        ),
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=50,
        metavar="N",
        help="Number of test cases to run per model (default: 50, max: 897).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        metavar="URL",
        help="Base URL of the Ollama server (e.g. http://my-ec2:11434).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/results/benchmark",
        metavar="PATH",
        help="Directory to write benchmark output files.",
    )
    parser.add_argument(
        "--judge-model",
        default=JUDGE_MODEL,
        metavar="MODEL",
        help="OpenAI model to use as the diagnosis match judge.",
    )
    parser.add_argument(
        "--dataset-split",
        default="test",
        metavar="SPLIT",
        help="HuggingFace dataset split to use (default: test).",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=RERANK_TOP_K_RETRIEVE,
        metavar="N",
        help=f"Candidates to fetch from bi-encoder (default: {RERANK_TOP_K_RETRIEVE}).",
    )
    parser.add_argument(
        "--return-k",
        type=int,
        default=RERANK_TOP_K_RETURN,
        metavar="N",
        help=f"Passages to keep after reranking (default: {RERANK_TOP_K_RETURN}).",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Skip RAG retrieval — test raw model diagnostic ability only.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Select models ─────────────────────────────────────────────────────────
    if args.models:
        # --models takes priority over --profile
        selected_models = get_models_by_names(args.models)
        if not selected_models:
            print(
                f"[ERROR] No valid models found in: {args.models}\n"
                f"  Available: {[m['name'] for m in ALL_MODELS]}",
                file=sys.stderr,
            )
            sys.exit(1)
    elif args.profile:
        selected_models = get_models_by_profile(args.profile)
        print(f"  Profile '{args.profile}': {PROFILES[args.profile]['description']}")
    else:
        # Default: all models
        selected_models = ALL_MODELS

    num_cases = min(args.num_cases, 897)
    output_dir = PROJECT_ROOT / args.output_dir

    print("\n" + "=" * 70)
    print("  Medora — LLM Benchmarking Framework")
    print("  Dataset: MedCaseReasoning (zou-lab/MedCaseReasoning)")
    print(f"  Cases:   {num_cases}")
    print(f"  Models:  {[m['name'] for m in selected_models]}")
    print(f"  Judge:   {args.judge_model}")
    print(f"  Output:  {output_dir}")
    print("=" * 70)

    # ── Build runner ──────────────────────────────────────────────────────────
    runner = BenchmarkRunner(
        models=selected_models,
        dataset_split=args.dataset_split,
        num_cases=num_cases,
        ollama_url=args.ollama_url,
        judge_model=args.judge_model,
        retrieve_k=args.retrieve_k,
        return_k=args.return_k,
        no_rag=args.no_rag,
    )

    # ── Run benchmark ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    all_results = runner.run_all()
    elapsed = time.perf_counter() - t0

    print(f"\nBenchmark complete in {elapsed / 60:.1f} minutes.")

    # ── Print comparison table ────────────────────────────────────────────────
    runner.print_comparison_table(all_results)

    # ── Save results ──────────────────────────────────────────────────────────
    runner.save_results(all_results, output_dir=output_dir)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
