"""
Phase 3 — LLM-as-Judge Evaluation
Sends the top-1 retrieved chunk from each of 3 pipeline configurations to
GPT-4o and asks it to rate relevance on a 1-5 scale.

Configurations:
  1. Bi-encoder only (k=3)
  2. Bi-encoder + BAAI/bge-reranker-v2-m3  (retrieve 10 → return 3)
  3. Bi-encoder + ncbi/MedCPT-Cross-Encoder (retrieve 10 → return 3)

Usage:
    python evaluation/llm_judge.py
    python evaluation/llm_judge.py --skip-medcpt
    python evaluation/llm_judge.py --queries 5
    python evaluation/llm_judge.py --skip-medcpt --queries 5
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Project root on sys.path so rag/ and config.py are importable ─────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Load .env before importing anything that reads env vars ──────────────────
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# ── OpenAI availability check ─────────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    print(
        "ERROR: 'openai' package is not installed.\n"
        "Install it with:  pip install openai"
    )
    sys.exit(1)

# ── Project imports ───────────────────────────────────────────────────────────
from config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    RERANKER_COMPARISON_MODEL,
    RERANK_TOP_K_RETRIEVE,
    RERANK_TOP_K_RETURN,
    RESULTS_DIR,
)

from rag.reranker import (
    MANUAL_QUERIES,
    detect_device,
    open_collection,
    load_bi_encoder,
    load_cross_encoder,
    retrieve_only,
    retrieve_and_rerank,
)

# ── Constants ─────────────────────────────────────────────────────────────────
JUDGE_MODEL = "gpt-4o"
RATE_LIMIT_SLEEP = 0.5  # seconds between API calls

SYSTEM_PROMPT = """\
You are a medical relevance assessor. Given a clinical query and a retrieved \
text passage from a medical textbook, rate how relevant the passage is to \
answering the query.

Rate on a 1-5 scale:
5 = Perfectly relevant — directly answers the query
4 = Highly relevant — contains key information for the query
3 = Moderately relevant — related but doesn't directly answer
2 = Slightly relevant — tangentially related
1 = Not relevant — wrong topic entirely

Respond with ONLY a JSON object: {"score": <int>, "reason": "<one sentence>"}\
"""

# ── OpenAI client ─────────────────────────────────────────────────────────────
client = OpenAI()  # reads OPENAI_API_KEY from env


# ── JSON parsing (robust — handles markdown code fences) ─────────────────────

def parse_json_response(text: str) -> dict:
    """
    Parse a JSON object from GPT-4o's response.
    Tries three strategies in order:
      1. Direct json.loads on the raw text.
      2. Extract from ```json ... ``` or ``` ... ``` code fences.
      3. Regex search for the first {...} block in the text.
    Raises ValueError if all strategies fail.
    """
    # Strategy 1: direct parse
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract from code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: find the first {...} block
    brace_match = re.search(r"\{[^{}]*\}", stripped, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {text!r}")


# ── GPT-4o scoring ────────────────────────────────────────────────────────────

def score_chunk(query: str, chunk_text: str) -> dict:
    """
    Send a (query, chunk_text) pair to GPT-4o and return the parsed
    {"score": int, "reason": str} dict.

    On any error (API failure, parse failure) returns {"score": None, "reason": <error>}.
    """
    user_message = (
        f"Clinical query: {query}\n\n"
        f"Retrieved passage:\n{chunk_text}"
    )

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )
        raw_text = response.choices[0].message.content or ""
        return parse_json_response(raw_text)
    except Exception as exc:
        return {"score": None, "reason": f"ERROR: {exc}"}


# ── Retrieve helpers (return text for rank-1 result) ─────────────────────────

def get_top1_text_biencoder(query: str, collection, bi_encoder, k: int = 3) -> tuple[str, dict]:
    """
    Bi-encoder only retrieval. Returns (chunk_text, result_dict) for rank-1.
    retrieve_only does not include documents by default, so we re-query with
    documents included.
    """
    import chromadb  # already loaded by open_collection

    vec = bi_encoder.encode(
        query, convert_to_numpy=True, normalize_embeddings=True
    ).tolist()

    raw = collection.query(
        query_embeddings=[vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunk_id = raw["ids"][0][0]
    text     = raw["documents"][0][0]
    meta     = raw["metadatas"][0][0]
    dist     = raw["distances"][0][0]

    result = {
        "chunk_id": chunk_id,
        "text":     text,
        "chapter":  meta.get("chapter", ""),
        "section":  meta.get("section", ""),
        "distance": round(float(dist), 6),
    }
    return text, result


def get_top1_text_reranked(
    query: str,
    collection,
    bi_encoder,
    reranker,
    retrieve_k: int,
    return_k: int,
) -> tuple[str, dict]:
    """
    Bi-encoder + cross-encoder retrieval. Returns (chunk_text, result_dict) for rank-1.
    """
    results = retrieve_and_rerank(
        query, collection, bi_encoder, reranker, retrieve_k, return_k
    )
    top1 = results[0]
    return top1["text"], top1


# ── Per-query evaluation ──────────────────────────────────────────────────────

def evaluate_query(
    entry: dict,
    collection,
    bi_encoder,
    bge_reranker,
    medcpt_reranker,            # may be None if --skip-medcpt
    retrieve_k: int,
    return_k: int,
) -> dict:
    """
    Run all 3 (or 2) configurations for a single query, score each with GPT-4o,
    and return a structured result dict.
    """
    query = entry["query"]

    # --- Config 1: bi-encoder only ---
    text_bi, result_bi = get_top1_text_biencoder(
        query, collection, bi_encoder, k=return_k
    )
    time.sleep(RATE_LIMIT_SLEEP)
    score_bi = score_chunk(query, text_bi)
    time.sleep(RATE_LIMIT_SLEEP)

    # --- Config 2: bi-encoder + BGE reranker ---
    text_bge, result_bge = get_top1_text_reranked(
        query, collection, bi_encoder, bge_reranker, retrieve_k, return_k
    )
    time.sleep(RATE_LIMIT_SLEEP)
    score_bge = score_chunk(query, text_bge)
    time.sleep(RATE_LIMIT_SLEEP)

    # --- Config 3: bi-encoder + MedCPT (optional) ---
    if medcpt_reranker is not None:
        text_medcpt, result_medcpt = get_top1_text_reranked(
            query, collection, bi_encoder, medcpt_reranker, retrieve_k, return_k
        )
        time.sleep(RATE_LIMIT_SLEEP)
        score_medcpt = score_chunk(query, text_medcpt)
        time.sleep(RATE_LIMIT_SLEEP)
    else:
        text_medcpt  = None
        result_medcpt = None
        score_medcpt  = None

    return {
        "query":            query,
        "category":         entry.get("category", ""),
        "expected_chapter": entry.get("expected_chapter", ""),
        "expected_section": entry.get("expected_section"),
        "configs": {
            "biencoder_only": {
                "chunk_text":    text_bi,
                "chapter":       result_bi["chapter"],
                "section":       result_bi["section"],
                "distance":      result_bi["distance"],
                "judge_score":   score_bi.get("score"),
                "judge_reason":  score_bi.get("reason"),
            },
            "bge_reranker": {
                "chunk_text":    text_bge,
                "chapter":       result_bge["chapter"],
                "section":       result_bge["section"],
                "rerank_score":  result_bge.get("rerank_score"),
                "original_rank": result_bge.get("original_rank"),
                "judge_score":   score_bge.get("score"),
                "judge_reason":  score_bge.get("reason"),
            },
            "medcpt_reranker": {
                "chunk_text":    text_medcpt,
                "chapter":       result_medcpt["chapter"]       if result_medcpt else None,
                "section":       result_medcpt["section"]       if result_medcpt else None,
                "rerank_score":  result_medcpt.get("rerank_score") if result_medcpt else None,
                "original_rank": result_medcpt.get("original_rank") if result_medcpt else None,
                "judge_score":   score_medcpt.get("score")  if score_medcpt else None,
                "judge_reason":  score_medcpt.get("reason") if score_medcpt else None,
            } if medcpt_reranker is not None else None,
        },
    }


# ── Summary printing ──────────────────────────────────────────────────────────

def safe_avg(scores: list) -> float | None:
    """Return mean of non-None numeric scores, or None if empty."""
    valid = [s for s in scores if isinstance(s, (int, float))]
    return round(sum(valid) / len(valid), 3) if valid else None


def print_summary(results: list[dict], skip_medcpt: bool) -> None:
    """Print a summary table of average scores and per-query scores."""
    configs_keys = ["biencoder_only", "bge_reranker"]
    config_labels = {
        "biencoder_only": "Bi-encoder only",
        "bge_reranker":   "BGE reranker",
        "medcpt_reranker": "MedCPT reranker",
    }
    if not skip_medcpt:
        configs_keys.append("medcpt_reranker")

    # Average scores
    avg_scores = {}
    for key in configs_keys:
        scores = []
        for r in results:
            cfg = r["configs"].get(key)
            if cfg:
                scores.append(cfg.get("judge_score"))
        avg_scores[key] = safe_avg(scores)

    col_w = 20
    sep   = "-" * (30 + col_w * len(configs_keys))

    print("\n" + "=" * (30 + col_w * len(configs_keys)))
    print("  LLM-as-Judge Evaluation — Summary")
    print("=" * (30 + col_w * len(configs_keys)))

    # Header
    header = f"  {'Configuration':<28}"
    for key in configs_keys:
        header += f"  {config_labels[key]:<{col_w - 2}}"
    print(header)
    print(sep)

    # Average row
    avg_row = f"  {'Average GPT-4o Score (1-5)':<28}"
    for key in configs_keys:
        val = avg_scores[key]
        avg_row += f"  {(str(val) if val is not None else 'N/A'):<{col_w - 2}}"
    print(avg_row)
    print(sep)

    # Per-query scores
    print(f"\n  {'#':<4}  {'Query':<55}", end="")
    for key in configs_keys:
        print(f"  {config_labels[key][:col_w - 2]:<{col_w - 2}}", end="")
    print()
    print(sep)

    for i, r in enumerate(results, start=1):
        q_short = r["query"][:52] + "..." if len(r["query"]) > 55 else r["query"]
        print(f"  {i:<4}  {q_short:<55}", end="")
        for key in configs_keys:
            cfg = r["configs"].get(key)
            score = cfg.get("judge_score") if cfg else None
            score_str = str(score) if score is not None else "N/A"
            print(f"  {score_str:<{col_w - 2}}", end="")
        print()

    print("=" * (30 + col_w * len(configs_keys)) + "\n")


# ── Output persistence ────────────────────────────────────────────────────────

def save_results(
    results: list[dict],
    skip_medcpt: bool,
    retrieve_k: int,
    return_k: int,
) -> Path:
    """Save all per-query judge results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "llm_judge_results.json"

    # Compute per-config averages for the summary block
    configs_keys = ["biencoder_only", "bge_reranker"]
    if not skip_medcpt:
        configs_keys.append("medcpt_reranker")

    averages = {}
    for key in configs_keys:
        scores = [
            r["configs"][key].get("judge_score")
            for r in results
            if r["configs"].get(key)
        ]
        averages[key] = safe_avg(scores)

    payload = {
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "phase":             "3",
        "judge_model":       JUDGE_MODEL,
        "bi_encoder_model":  EMBEDDING_MODEL,
        "bge_reranker_model": RERANKER_MODEL,
        "medcpt_model":      RERANKER_COMPARISON_MODEL if not skip_medcpt else None,
        "retrieve_k":        retrieve_k,
        "return_k":          return_k,
        "num_queries":       len(results),
        "skip_medcpt":       skip_medcpt,
        "average_scores":    averages,
        "per_query":         results,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to {out_path}")
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3 — LLM-as-Judge evaluation. "
            "Rates the top-1 retrieved chunk from each pipeline config using GPT-4o."
        )
    )
    parser.add_argument(
        "--skip-medcpt",
        action="store_true",
        help="Skip Config 3 (bi-encoder + MedCPT). Useful on low-memory machines.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=None,
        metavar="N",
        help="Only run the first N queries (useful for quick testing).",
    )
    args = parser.parse_args()

    skip_medcpt = args.skip_medcpt
    queries     = MANUAL_QUERIES[: args.queries] if args.queries else MANUAL_QUERIES

    retrieve_k = RERANK_TOP_K_RETRIEVE
    return_k   = RERANK_TOP_K_RETURN

    print(f"\nPhase 3 — LLM-as-Judge Evaluation")
    print(f"  judge_model={JUDGE_MODEL}")
    print(f"  bi_encoder={EMBEDDING_MODEL}")
    print(f"  bge_reranker={RERANKER_MODEL}")
    if not skip_medcpt:
        print(f"  medcpt_reranker={RERANKER_COMPARISON_MODEL}")
    print(f"  retrieve_k={retrieve_k}  return_k={return_k}")
    print(f"  num_queries={len(queries)}")
    print(f"  skip_medcpt={skip_medcpt}\n")

    # ── Infrastructure ────────────────────────────────────────────────────────
    device     = detect_device()
    collection = open_collection(CHROMA_DIR)
    bi_encoder = load_bi_encoder(EMBEDDING_MODEL, device)

    bge_reranker, bge_device = load_cross_encoder(RERANKER_MODEL)
    print(f"  (BGE cross-encoder running on {bge_device})")

    medcpt_reranker = None
    if not skip_medcpt:
        medcpt_reranker, medcpt_device = load_cross_encoder(RERANKER_COMPARISON_MODEL)
        print(f"  (MedCPT cross-encoder running on {medcpt_device})")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\nRunning LLM-as-Judge scoring ({len(queries)} queries) ...\n")
    all_results = []

    for i, entry in enumerate(queries, start=1):
        q_short = entry["query"][:60]
        print(f"  [{i:02d}/{len(queries)}] {q_short}")

        result = evaluate_query(
            entry,
            collection,
            bi_encoder,
            bge_reranker,
            medcpt_reranker,
            retrieve_k,
            return_k,
        )

        # Quick inline score display
        s_bi  = result["configs"]["biencoder_only"]["judge_score"]
        s_bge = result["configs"]["bge_reranker"]["judge_score"]
        s_med = (
            result["configs"]["medcpt_reranker"]["judge_score"]
            if not skip_medcpt and result["configs"]["medcpt_reranker"]
            else "—"
        )
        print(
            f"         scores → bi-encoder={s_bi}  BGE={s_bge}  MedCPT={s_med}"
        )

        all_results.append(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    print_summary(all_results, skip_medcpt)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results(all_results, skip_medcpt, retrieve_k, return_k)

    print("LLM-as-Judge evaluation complete.\n")


if __name__ == "__main__":
    main()
