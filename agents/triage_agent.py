"""
Phase 4.2 — Triage Agent
========================
A single-query clinical analysis engine that takes a clinical question (either
direct from a doctor or derived from the Intake Agent's summary), retrieves
relevant textbook passages using the RAG pipeline (Phases 2+3), and generates
evidence-based clinical analysis.

Two input modes:
  1. Direct query  — doctor asks a clinical question directly.
  2. From Intake   — receives a structured IntakeSession.get_summary() dict,
                     auto-generates 2-4 clinical questions, and produces a
                     comprehensive synthesis report.

The Triage Agent does NOT run for EMERGENCY cases — the Intake Agent already
escalated. It handles routine and urgent cases only.

Usage:
    # Interactive direct query mode
    python agents/triage_agent.py

    # Single query, non-interactive
    python agents/triage_agent.py --query "What are the causes of hemoptysis in a young smoker?"

    # From a saved intake summary
    python agents/triage_agent.py --from-intake data/results/last_intake.json

    # Choose model
    python agents/triage_agent.py --model gpt-4o-mini
"""

import argparse
import json
import sys
from pathlib import Path

# ── Project root on sys.path so config.py is importable ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Environment ───────────────────────────────────────────────────────────────
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

# ── RAG pipeline imports ──────────────────────────────────────────────────────
from rag.reranker import (  # noqa: E402
    retrieve_and_rerank,
    open_collection,
    load_bi_encoder,
    load_cross_encoder,
    detect_device,
)

# ── Config imports ────────────────────────────────────────────────────────────
from config import (  # noqa: E402
    CHROMA_DIR,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    RERANK_TOP_K_RETRIEVE,
    RERANK_TOP_K_RETURN,
)

# ── LangChain ─────────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Prompt constants
# ─────────────────────────────────────────────────────────────────────────────

_CLINICAL_ANALYSIS_SYSTEM = """\
You are a clinical decision support system. Given a clinical query and relevant passages
from a medical textbook (CURRENT Medical Diagnosis and Treatment), provide a structured
clinical analysis.

Your analysis must be:
- Grounded ONLY in the provided textbook passages — do not add information from your
  general training data
- Structured with clear sections
- Include specific references to the source passages (cite by chapter and section)
- Clinically precise and actionable

Structure your response as:

## Clinical Analysis
[Direct answer to the query with evidence from the passages]

## Key Findings from Evidence
[Bullet points of the most relevant facts from the retrieved passages]

## Differential Considerations
[If applicable: conditions to consider based on the evidence]

## Recommended Next Steps
[Clinical actions suggested by the textbook evidence]

## Sources
[List each retrieved passage with chapter, section, and a brief description of what it contributed]

If the retrieved passages do not contain sufficient information to answer the query,
state this clearly rather than guessing.\
"""

_QUERY_GENERATION_SYSTEM = """\
You are a clinical triage planning assistant. Given a patient intake summary, generate
2-4 specific clinical questions that a doctor would need answered to manage this patient.

Focus on:
- Differential diagnosis for the presenting symptoms
- Management considerations based on the patient's specific answers
- Any red flags that need further investigation

Return a JSON array of question strings only. No explanation, no markdown fences.\
"""

_SYNTHESIS_SYSTEM = """\
You are a clinical decision support system. Given multiple clinical analyses from
a patient intake, synthesize a comprehensive clinical report.

Rules:
- Base all recommendations ONLY on the retrieved textbook passages provided.
- Cite textbook chapters and sections explicitly.
- Be clinically precise and actionable.

Structure your response as:

## Patient Overview
[Summary of presenting complaints and key findings from intake]

## Differential Diagnosis
[Ranked list of likely diagnoses with supporting evidence]

## Recommended Management Plan
[Evidence-based management recommendations]

## Investigations Required
[Specific tests and their clinical justification]

## Red Flags and Safety Netting
[What to watch for, when to escalate]

## Sources
[All textbook passages used, organised by topic]\
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build context string from retrieved chunks
# ─────────────────────────────────────────────────────────────────────────────

def _chunks_to_context(chunks: list[dict]) -> str:
    """
    Format a list of retrieved/reranked chunk dicts into a numbered context
    block suitable for LLM injection.

    Each chunk dict is expected to have: text, chapter, section, rerank_score.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        chapter = chunk.get("chapter", "Unknown Chapter")
        section = chunk.get("section", "Unknown Section")
        text    = chunk.get("text", "").strip()
        score   = chunk.get("rerank_score")
        score_str = f"  [rerank score: {score:.4f}]" if score is not None else ""
        parts.append(
            f"--- Passage {i} ---\n"
            f"Chapter : {chapter}\n"
            f"Section : {section}{score_str}\n\n"
            f"{text}"
        )
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Core function: triage_query
# ─────────────────────────────────────────────────────────────────────────────

def triage_query(
    query: str,
    collection,
    bi_encoder,
    reranker,
    llm: ChatOpenAI,
    retrieve_k: int = RERANK_TOP_K_RETRIEVE,
    return_k: int = RERANK_TOP_K_RETURN,
) -> dict:
    """
    Single-query triage: retrieve relevant textbook chunks and generate a
    structured clinical analysis grounded in those chunks.

    Args:
        query:       the clinical question to answer.
        collection:  open ChromaDB collection object.
        bi_encoder:  loaded SentenceTransformer bi-encoder.
        reranker:    loaded CrossEncoder reranker.
        llm:         ChatOpenAI instance for generation.
        retrieve_k:  number of candidates to fetch from the bi-encoder.
        return_k:    number of passages to keep after reranking.

    Returns:
        {
            "query": str,
            "retrieved_chunks": list[dict],   # top return_k chunks with metadata
            "analysis": str,                  # LLM-generated clinical analysis
        }
    """
    # Step 1: Retrieve and rerank
    chunks = retrieve_and_rerank(
        query, collection, bi_encoder, reranker, retrieve_k, return_k
    )

    # Step 2: Build context
    context = _chunks_to_context(chunks)

    # Step 3: Generate clinical analysis
    user_content = (
        f"Clinical Query:\n{query}\n\n"
        f"Retrieved Textbook Passages:\n\n{context}"
    )
    response = llm.invoke([
        SystemMessage(content=_CLINICAL_ANALYSIS_SYSTEM),
        HumanMessage(content=user_content),
    ])
    analysis = response.content.strip()

    return {
        "query":            query,
        "retrieved_chunks": chunks,
        "analysis":         analysis,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core function: triage_from_intake
# ─────────────────────────────────────────────────────────────────────────────

def triage_from_intake(
    intake_summary: dict,
    collection,
    bi_encoder,
    reranker,
    llm: ChatOpenAI,
    retrieve_k: int = RERANK_TOP_K_RETRIEVE,
    return_k: int = RERANK_TOP_K_RETURN,
) -> dict:
    """
    Generate a comprehensive clinical analysis from an IntakeSession.get_summary()
    dict.

    Emergency cases are caught early and deferred — the Triage Agent only handles
    routine and urgent presentations.

    Args:
        intake_summary: dict returned by IntakeSession.get_summary(), containing
                        symptoms, urgency, answers, triggered_red_flags,
                        specialty_routing, initial_workup, etc.
        collection:     open ChromaDB collection object.
        bi_encoder:     loaded SentenceTransformer bi-encoder.
        reranker:       loaded CrossEncoder reranker.
        llm:            ChatOpenAI instance.
        retrieve_k:     bi-encoder candidate count.
        return_k:       passages to keep after reranking.

    Returns:
        {
            "intake_symptoms":       list[str],
            "urgency":               str,
            "generated_queries":     list[str],
            "per_query_results":     list[dict],
            "comprehensive_analysis": str,
        }
    """
    urgency   = intake_summary.get("urgency", "routine").lower()
    symptoms  = intake_summary.get("symptoms", [])
    answers   = intake_summary.get("answers", {})
    red_flags = intake_summary.get("triggered_red_flags", [])

    # Step 1: Emergency guard
    if urgency == "emergency":
        return {
            "intake_symptoms":        symptoms,
            "urgency":                urgency,
            "generated_queries":      [],
            "per_query_results":      [],
            "comprehensive_analysis": (
                "Emergency case — Triage Agent defers to emergency services. "
                "This presentation has already been escalated by the Intake Agent. "
                "No further triage analysis is required."
            ),
        }

    # Step 2: Generate clinical questions from the intake summary
    symptoms_str  = ", ".join(symptoms) if symptoms else "unspecified"
    answers_block = "\n".join(
        f"  Q: {q}\n  A: {a}" for q, a in answers.items()
    ) if answers else "  (no answers recorded)"
    flags_block = (
        "\n".join(f"  - {rf.get('flag', '')} [{rf.get('urgency', '')}]" for rf in red_flags)
        if red_flags else "  (none)"
    )

    query_gen_user = (
        f"Patient intake summary:\n\n"
        f"Symptoms: {symptoms_str}\n"
        f"Urgency:  {urgency}\n\n"
        f"Patient answers:\n{answers_block}\n\n"
        f"Triggered red flags:\n{flags_block}\n\n"
        f"Generate 2-4 specific clinical questions that a doctor would need "
        f"answered to manage this patient."
    )

    gen_response = llm.invoke([
        SystemMessage(content=_QUERY_GENERATION_SYSTEM),
        HumanMessage(content=query_gen_user),
    ])

    raw_queries = gen_response.content.strip()
    # Strip markdown fences if present
    if raw_queries.startswith("```"):
        parts = raw_queries.split("```")
        inner = parts[1] if len(parts) > 1 else raw_queries
        if inner.startswith("json"):
            inner = inner[4:]
        raw_queries = inner.strip()

    try:
        generated_queries: list[str] = json.loads(raw_queries)
        if not isinstance(generated_queries, list):
            generated_queries = [raw_queries]
    except (json.JSONDecodeError, ValueError):
        # Fall back to treating the whole response as a single question
        generated_queries = [raw_queries]

    # Step 3: Run triage_query for each generated question
    per_query_results: list[dict] = []
    for q in generated_queries:
        result = triage_query(
            q, collection, bi_encoder, reranker, llm, retrieve_k, return_k
        )
        per_query_results.append(result)

    # Step 4: Synthesize all analyses into a comprehensive report
    # Collect all retrieved chunks across queries (deduplicated by chunk_id)
    seen_chunk_ids: set[str] = set()
    all_chunks: list[dict] = []
    for r in per_query_results:
        for chunk in r["retrieved_chunks"]:
            cid = chunk.get("chunk_id", "")
            if cid not in seen_chunk_ids:
                all_chunks.append(chunk)
                seen_chunk_ids.add(cid)

    combined_context = _chunks_to_context(all_chunks)

    individual_analyses = "\n\n".join(
        f"--- Analysis for: {r['query']} ---\n{r['analysis']}"
        for r in per_query_results
    )

    synthesis_user = (
        f"Patient intake summary:\n\n"
        f"Symptoms : {symptoms_str}\n"
        f"Urgency  : {urgency}\n\n"
        f"Patient answers:\n{answers_block}\n\n"
        f"Triggered red flags:\n{flags_block}\n\n"
        f"=== Individual Clinical Analyses ===\n\n"
        f"{individual_analyses}\n\n"
        f"=== All Retrieved Textbook Passages ===\n\n"
        f"{combined_context}\n\n"
        f"Using ALL of the above, produce the comprehensive clinical report."
    )

    synthesis_response = llm.invoke([
        SystemMessage(content=_SYNTHESIS_SYSTEM),
        HumanMessage(content=synthesis_user),
    ])
    comprehensive_analysis = synthesis_response.content.strip()

    return {
        "intake_symptoms":        symptoms,
        "urgency":                urgency,
        "generated_queries":      generated_queries,
        "per_query_results":      per_query_results,
        "comprehensive_analysis": comprehensive_analysis,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TriageSession class
# ─────────────────────────────────────────────────────────────────────────────

class TriageSession:
    """
    Manages triage queries with loaded models, so the expensive model loading
    happens once per session rather than once per query.

    Usage:
        session = TriageSession()

        # Direct query mode
        result = session.query("What are the causes of hemoptysis in a young smoker?")
        print(result["analysis"])

        # From Intake Agent mode
        result = session.from_intake(intake_summary_dict)
        print(result["comprehensive_analysis"])
    """

    def __init__(self, llm_model: str = "gpt-4o"):
        print("\nInitialising Triage Agent models...")
        self._llm        = ChatOpenAI(model=llm_model, temperature=0)
        self._device     = detect_device()
        self._collection = open_collection(CHROMA_DIR)
        self._bi_encoder = load_bi_encoder(EMBEDDING_MODEL, self._device)
        self._reranker, self._reranker_device = load_cross_encoder(RERANKER_MODEL)
        print(
            f"Triage Agent ready  "
            f"(bi-encoder on {self._device}, "
            f"reranker on {self._reranker_device})\n"
        )

    def query(
        self,
        clinical_query: str,
        retrieve_k: int = RERANK_TOP_K_RETRIEVE,
        return_k: int = RERANK_TOP_K_RETURN,
    ) -> dict:
        """
        Direct query mode — answer a single clinical question.

        Returns:
            {"query", "retrieved_chunks", "analysis"}
        """
        return triage_query(
            clinical_query,
            self._collection,
            self._bi_encoder,
            self._reranker,
            self._llm,
            retrieve_k=retrieve_k,
            return_k=return_k,
        )

    def from_intake(
        self,
        intake_summary: dict,
        retrieve_k: int = RERANK_TOP_K_RETRIEVE,
        return_k: int = RERANK_TOP_K_RETURN,
    ) -> dict:
        """
        From Intake Agent mode — generate a comprehensive analysis from a
        structured intake summary dict.

        Returns:
            {"intake_symptoms", "urgency", "generated_queries",
             "per_query_results", "comprehensive_analysis"}
        """
        return triage_from_intake(
            intake_summary,
            self._collection,
            self._bi_encoder,
            self._reranker,
            self._llm,
            retrieve_k=retrieve_k,
            return_k=return_k,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_retrieved_chunks(chunks: list[dict]) -> None:
    """Print the retrieved/reranked chunks with their provenance."""
    print("\n" + "─" * 60)
    print(f"  Retrieved passages ({len(chunks)}):")
    print("─" * 60)
    for i, chunk in enumerate(chunks, start=1):
        chapter = chunk.get("chapter", "Unknown Chapter")
        section = chunk.get("section", "Unknown Section")
        score   = chunk.get("rerank_score")
        orig    = chunk.get("original_rank")
        score_str = f"  score={score:.4f}" if score is not None else ""
        orig_str  = f"  orig_rank={orig}" if orig is not None else ""
        print(f"  [{i}] {chapter} / {section}{score_str}{orig_str}")
    print("─" * 60 + "\n")


def _print_query_result(result: dict) -> None:
    """Print a single triage_query result."""
    print("\n" + "=" * 60)
    print(f"  Query: {result['query']}")
    print("=" * 60)
    _print_retrieved_chunks(result["retrieved_chunks"])
    print(result["analysis"])
    print()


def _print_intake_result(result: dict) -> None:
    """Print the full triage_from_intake result."""
    print("\n" + "=" * 60)
    print("  Triage Agent — Intake Analysis")
    print("=" * 60)
    symptoms = result.get("intake_symptoms", [])
    urgency  = result.get("urgency", "unknown")
    queries  = result.get("generated_queries", [])
    print(f"  Symptoms : {', '.join(symptoms) if symptoms else 'N/A'}")
    print(f"  Urgency  : {urgency.upper()}")

    if result.get("per_query_results"):
        print(f"\n  Generated {len(queries)} clinical question(s):")
        for i, q in enumerate(queries, start=1):
            print(f"    {i}. {q}")

        print("\n" + "─" * 60)
        print("  Per-question retrieved passages:")
        print("─" * 60)
        for r in result["per_query_results"]:
            print(f"\n  Q: {r['query']}")
            _print_retrieved_chunks(r["retrieved_chunks"])

    print("\n" + "=" * 60)
    print("  COMPREHENSIVE CLINICAL REPORT")
    print("=" * 60)
    print(result["comprehensive_analysis"])
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Medora Phase 4.2 — Clinical Triage Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        metavar="MODEL",
        help="OpenAI model name to use (e.g. gpt-4o, gpt-4o-mini).",
    )
    parser.add_argument(
        "--from-intake",
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON file containing an IntakeSession.get_summary() dict. "
            "When provided, runs in intake-analysis mode and exits."
        ),
    )
    parser.add_argument(
        "--query",
        default=None,
        metavar="QUESTION",
        help=(
            "Single clinical query — prints the result and exits (non-interactive)."
        ),
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Mode 1: from-intake file ──────────────────────────────────────────────
    if args.from_intake:
        intake_path = Path(args.from_intake)
        if not intake_path.exists():
            print(f"Error: intake file not found: {intake_path}", file=sys.stderr)
            sys.exit(1)

        with open(intake_path, encoding="utf-8") as fh:
            intake_summary = json.load(fh)

        session = TriageSession(llm_model=args.model)
        result  = session.from_intake(
            intake_summary,
            retrieve_k=args.retrieve_k,
            return_k=args.return_k,
        )
        _print_intake_result(result)
        return

    # ── Mode 2: single --query, non-interactive ───────────────────────────────
    if args.query:
        session = TriageSession(llm_model=args.model)
        result  = session.query(
            args.query,
            retrieve_k=args.retrieve_k,
            return_k=args.return_k,
        )
        _print_query_result(result)
        return

    # ── Mode 3: interactive direct query ─────────────────────────────────────
    print("\nMedora Triage Agent")
    print("=" * 60)
    print("Type a clinical question to retrieve evidence-based analysis.")
    print("Type 'quit' to exit.")
    print("=" * 60 + "\n")

    session = TriageSession(llm_model=args.model)

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession interrupted.\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye.\n")
            break

        result = session.query(
            user_input,
            retrieve_k=args.retrieve_k,
            return_k=args.return_k,
        )
        _print_query_result(result)


if __name__ == "__main__":
    main()
