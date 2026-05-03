"""
Pipeline Benchmark — Phase 9
=============================
Two-part evaluation suite that tests the full Medora RAG pipeline on test cases
derived from the textbook itself (ground truth retrieval validation) and on
filtered MedQA USMLE questions (external clinical validity).

Part 1 — generate_textbook_cases
    Samples chunks from ChromaDB and uses GPT-5.4-mini to generate realistic patient
    presentations for conditions described in the textbook. Because we know
    exactly which chunks contain the answer, this gives precise retrieval metrics.

Part 2 — filter_medqa_to_textbook
    Loads the MedQA-USMLE dataset from HuggingFace and filters to questions
    about conditions covered in our 42 textbook chapters. Only "most likely
    diagnosis" questions are kept.

Part 3 — PipelineBenchmarkRunner
    Runs the full RAG pipeline (bi-encoder → reranker → LLM) on either test set
    and captures both retrieval-level and generation-level metrics so we can
    distinguish retrieval failures from reasoning failures.

Usage:
    python evaluation/pipeline_benchmark.py --generate --num-cases 50
    python evaluation/pipeline_benchmark.py --filter-medqa --num-cases 50
    python evaluation/pipeline_benchmark.py --run --profile api --test-set textbook
    python evaluation/pipeline_benchmark.py --run --profile ollama --test-set both
    python evaluation/pipeline_benchmark.py --run --models gpt-5.4-mini --test-set medqa
    python evaluation/pipeline_benchmark.py --generate --run --test-set textbook
"""

from __future__ import annotations

import argparse
import json
import random
import re
import signal
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

# ── Prompt helpers from triage agent ─────────────────────────────────────────
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

TIMEOUT_OPENAI = 120
TIMEOUT_OLLAMA = 300


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

EVAL_DATA_DIR = PROJECT_ROOT / "data" / "evaluation"
TEXTBOOK_CASES_PATH = EVAL_DATA_DIR / "textbook_test_cases.json"
MEDQA_CASES_PATH = EVAL_DATA_DIR / "medqa_filtered_cases.json"

# ── Chapters to skip when generating test cases (intake/meta chapters) ────────
_SKIP_CHAPTERS = {"Common Symptoms"}

# ── Keywords that indicate a diagnostic chunk (worth generating a case from) ──
_DIAGNOSTIC_MARKERS = [
    "essentials of diagnosis",
    "general considerations",
    "clinical findings",
    "symptoms and signs",
    "treatment",
    "differential diagnosis",
    "prognosis",
]

# ── MedQA filter phrases (keep only "most likely diagnosis" questions) ─────────
_DIAGNOSIS_QUESTION_PHRASES = [
    "most likely diagnosis",
    "most likely cause",
    "most likely responsible",
    "most likely the cause",
]


# ─────────────────────────────────────────────────────────────────────────────
# LLM helpers (mirrors benchmark.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm(model_config: dict, ollama_url: str):
    """Instantiate the correct LangChain chat model."""
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
                    "Install langchain-community or langchain-ollama: "
                    "pip install langchain-community"
                ) from exc
        return ChatOllama(model=model_id, base_url=ollama_url, temperature=0)

    raise ValueError(f"Unknown provider: {provider!r}")


def _check_ollama_available(model_config: dict, ollama_url: str) -> bool:
    """Ping Ollama to verify the model is loaded."""
    import urllib.error
    import urllib.request

    name = model_config["name"]
    model_id = model_config["model_id"]

    try:
        with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError) as exc:
        print(f"  [WARN] Ollama server at {ollama_url} not reachable: {exc}")
        print(f"  [SKIP] Skipping model '{name}'.")
        return False

    available_models = [m.get("name", "") for m in data.get("models", [])]
    if not any(model_id in m for m in available_models):
        print(f"  [WARN] Model '{model_id}' not found on Ollama server.")
        print(f"         Available: {available_models}")
        print(f"  [SKIP] Skipping model '{name}'.")
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def _string_similarity(a: str, b: str) -> float:
    """Normalised Levenshtein-ratio similarity in [0, 1]."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


_PRIMARY_DX_PATTERN = re.compile(
    r"##\s*Primary\s+Diagnosis\s*\n+(.+?)(?:\n\n|\n##|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_primary_diagnosis(report_text: str) -> str | None:
    """Parse '## Primary Diagnosis' section from the model's report."""
    m = _PRIMARY_DX_PATTERN.search(report_text)
    if not m:
        return None
    section_text = m.group(1).strip()
    for line in section_text.splitlines():
        line = line.strip().lstrip("-*•").strip()
        if line:
            line = re.sub(
                r"\s*[\(\[].*?confidence.*?[\)\]]", "", line, flags=re.IGNORECASE
            ).strip()
            line = line.rstrip(":").strip()
            return line if line else None
    return None


def _judge_diagnosis_match(
    system_diagnosis: str,
    ground_truth: str,
    judge_llm,
) -> str:
    """
    Use the judge LLM to classify the match between model output and ground truth.

    Returns one of:
        "exact_match"    — string similarity > 0.8
        "semantic_match" — LLM says same condition
        "partial_match"  — LLM says related but not identical
        "mismatch"       — LLM says different conditions
    """
    if _string_similarity(system_diagnosis, ground_truth) >= 0.8:
        return "exact_match"

    judge_prompt = (
        f"Does the system diagnosis match the ground truth diagnosis?\n"
        f"They may use different medical terminology for the same condition.\n\n"
        f"Ground truth:     {ground_truth}\n"
        f"System diagnosis: {system_diagnosis}\n\n"
        f"Respond with ONLY one of these four words (no explanation):\n"
        f"  semantic_match  — same condition, different wording\n"
        f"  partial_match   — related / overlapping, but not the same\n"
        f"  mismatch        — clearly different conditions"
    )

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
        return "mismatch"
    except Exception:
        sim = _string_similarity(system_diagnosis, ground_truth)
        if sim >= 0.6:
            return "partial_match"
        return "mismatch"


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — Test case generator from the textbook
# ─────────────────────────────────────────────────────────────────────────────

_CASE_GENERATION_SYSTEM = """\
You are a clinical case writer for medical education.
Given a medical textbook passage about a specific condition, write a realistic
patient presentation AND identify the correct diagnosis.

Return ONLY a JSON object — no markdown fences, no explanation.
Schema:
{
    "patient_presentation": "I'm a 45-year-old woman...",
    "ground_truth_diagnosis": "Condition Name",
    "chapter": "chapter name from the passage",
    "section": "section name",
    "difficulty": "easy" | "medium" | "hard"
}"""

_CASE_GENERATION_TEMPLATE = """\
Here is a medical textbook passage (chapter: {chapter}, section: {section}):

---
{chunk_text}
---

Rules for the patient presentation:
- Write in first person from the patient's perspective
- Include key symptoms and history that would lead to the correct diagnosis
- Use everyday language, NOT medical terminology
- Include some but NOT all diagnostic features (make it realistic, not a textbook dump)
- Include 2-3 relevant history details (age, gender, risk factors)
- Keep it to 3-5 sentences

Rules for difficulty:
- easy:   presentation strongly suggests the diagnosis (classic features)
- medium: presentation includes the diagnosis but requires clinical reasoning
- hard:   presentation is subtle and requires differential diagnosis reasoning

Return the JSON object now."""


def _is_diagnostic_chunk(chunk_text: str) -> bool:
    """
    Return True if the chunk is likely about a specific diagnosable condition
    (has clinical findings, treatment info, or essentials of diagnosis).
    """
    lower = chunk_text.lower()
    return any(marker in lower for marker in _DIAGNOSTIC_MARKERS)


def generate_textbook_cases(
    collection,
    llm,
    num_cases: int = 50,
    chapters: list[str] | None = None,
) -> list[dict]:
    """
    Sample chunks from ChromaDB and use GPT-4o to generate realistic patient
    presentations for conditions described in the textbook.

    Args:
        collection:  open ChromaDB collection (tmt_chunks)
        llm:         ChatOpenAI instance for case generation (gpt-5.4-mini recommended)
        num_cases:   target number of cases to generate
        chapters:    if provided, restrict sampling to these chapters only

    Returns:
        List of case dicts, each with:
            patient_presentation, ground_truth_diagnosis, chapter, section,
            difficulty, source_chunk_id, source_chunk_preview
    """
    EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {num_cases} textbook test cases from ChromaDB...")

    # ── Step 1: Get all chunk metadata to understand chapter distribution ──────
    print("  Fetching chunk metadata from ChromaDB...")
    all_meta = collection.get(include=["metadatas", "documents"])
    chunk_ids = all_meta["ids"]
    metadatas = all_meta["metadatas"]
    documents = all_meta["documents"]

    # Build chapter → [(chunk_id, doc, meta), ...] index
    chapter_index: dict[str, list[tuple[str, str, dict]]] = {}
    for cid, doc, meta in zip(chunk_ids, documents, metadatas):
        chapter = meta.get("chapter", "Unknown")
        if chapter in _SKIP_CHAPTERS:
            continue
        if chapters and chapter not in chapters:
            continue
        if not _is_diagnostic_chunk(doc):
            continue
        chapter_index.setdefault(chapter, []).append((cid, doc, meta))

    available_chapters = sorted(chapter_index.keys())
    print(f"  {len(available_chapters)} chapters with diagnostic chunks available.")

    if not available_chapters:
        print("  [ERROR] No qualifying chunks found. Check ChromaDB content.")
        return []

    # ── Step 2: Spread cases evenly across chapters ────────────────────────────
    # Aim for ~1-2 cases per chapter; cycle through chapters until num_cases met
    cases_per_chapter, remainder = divmod(num_cases, len(available_chapters))
    if cases_per_chapter == 0:
        cases_per_chapter = 1

    sampling_plan: list[tuple[str, str, str, dict]] = []  # (cid, doc, chapter, meta)
    for i, chapter in enumerate(available_chapters):
        target = cases_per_chapter + (1 if i < remainder else 0)
        pool = chapter_index[chapter]
        random.shuffle(pool)
        for cid, doc, meta in pool[:target]:
            sampling_plan.append((cid, doc, chapter, meta))

    # Trim or top-up to exact num_cases
    random.shuffle(sampling_plan)
    sampling_plan = sampling_plan[:num_cases]

    print(f"  Sampled {len(sampling_plan)} chunks across {len(available_chapters)} chapters.")

    # ── Step 3: Generate a case for each sampled chunk ─────────────────────────
    cases: list[dict] = []
    skipped = 0

    for idx, (chunk_id, doc_text, chapter, meta) in enumerate(sampling_plan, start=1):
        section = meta.get("section", "")
        print(f"  [{idx:>4}/{len(sampling_plan)}] chapter={chapter!r}  section={section[:40]!r}")

        user_prompt = _CASE_GENERATION_TEMPLATE.format(
            chapter=chapter,
            section=section,
            chunk_text=doc_text[:1500],  # trim very long chunks
        )

        try:
            response = llm.invoke([
                SystemMessage(content=_CASE_GENERATION_SYSTEM),
                HumanMessage(content=user_prompt),
            ])
            raw = response.content.strip()

            # Strip any accidental markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)

            parsed = json.loads(raw)

            # Validate required fields
            required = ("patient_presentation", "ground_truth_diagnosis", "chapter", "difficulty")
            if not all(k in parsed for k in required):
                print(f"    [WARN] Missing fields in LLM response — skipping chunk.")
                skipped += 1
                continue

            # Augment with source metadata
            parsed["source"] = "textbook_generated"
            parsed["source_chunk_id"] = chunk_id
            parsed["source_chunk_preview"] = doc_text[:300]
            parsed["source_chapter"] = chapter
            parsed["source_section"] = section

            cases.append(parsed)
            print(
                f"    [OK] diagnosis={parsed['ground_truth_diagnosis']!r}  "
                f"difficulty={parsed['difficulty']!r}"
            )

        except json.JSONDecodeError as exc:
            print(f"    [WARN] JSON parse error: {exc} — skipping chunk.")
            skipped += 1
        except Exception as exc:
            print(f"    [ERROR] {type(exc).__name__}: {exc} — skipping chunk.")
            skipped += 1

    # ── Step 4: Save to disk ──────────────────────────────────────────────────
    with open(TEXTBOOK_CASES_PATH, "w", encoding="utf-8") as fh:
        json.dump(cases, fh, indent=2, ensure_ascii=False)

    print(
        f"\n  Generated {len(cases)} cases ({skipped} skipped)."
        f"\n  Saved to: {TEXTBOOK_CASES_PATH}"
    )
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — Filter MedQA USMLE to textbook-covered conditions
# ─────────────────────────────────────────────────────────────────────────────

_MEDQA_FILTER_SYSTEM = """\
You are a medical curriculum expert. Decide whether a diagnosis would reasonably
be discussed in any of the given medical textbook chapters.

A condition is "covered" if:
  - It falls within the scope of ANY listed chapter (e.g., "Pneumonia" is covered by "Pulmonary Disorders")
  - It is a common internal medicine condition that would appear in a general medical reference
  - The chapter name broadly covers the organ system or disease category

Be INCLUSIVE — most common medical conditions are covered by at least one chapter.
Only say "no" for highly specialized conditions outside internal medicine (e.g., ophthalmologic surgery, pediatric-only conditions).

Respond with ONLY one word: yes  or  no"""

_MEDQA_FILTER_TEMPLATE = """\
Textbook chapters:
{chapters_list}

Question answer (diagnosis to check): {diagnosis}

Is this diagnosis covered in the textbook chapters above? Answer yes or no."""


def _get_textbook_chapters(collection) -> list[str]:
    """Return sorted list of unique chapter names from the ChromaDB collection."""
    all_meta = collection.get(include=["metadatas"])
    chapters: set[str] = set()
    for meta in all_meta["metadatas"]:
        ch = meta.get("chapter", "").strip()
        if ch:
            chapters.add(ch)
    return sorted(chapters)


def filter_medqa_to_textbook(
    collection,
    num_cases: int = 50,
    judge_llm: Any | None = None,
) -> list[dict]:
    """
    Load MedQA USMLE from HuggingFace, filter to 'most likely diagnosis'
    questions whose correct answer is covered in our textbook chapters.

    Args:
        collection:  open ChromaDB collection (used to enumerate chapters)
        num_cases:   maximum number of cases to keep
        judge_llm:   ChatOpenAI instance (gpt-5.4-mini) for coverage check.
                     If None, a gpt-5.4-mini instance is created automatically.

    Returns:
        List of case dicts, each with:
            case_prompt, ground_truth_diagnosis, options, source
    """
    EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if judge_llm is None:
        judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)

    # ── Load chapters from ChromaDB ────────────────────────────────────────────
    print("\nFetching textbook chapters from ChromaDB...")
    textbook_chapters = _get_textbook_chapters(collection)
    print(f"  Found {len(textbook_chapters)} chapters.")
    chapters_list_str = "\n".join(f"  - {ch}" for ch in textbook_chapters)

    # ── Load MedQA dataset ────────────────────────────────────────────────────
    print("\nLoading MedQA-USMLE dataset from HuggingFace...")
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Install the HuggingFace datasets library: pip install datasets"
        ) from exc

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    all_cases = list(ds)
    print(f"  Loaded {len(all_cases):,} MedQA cases.")

    # ── Step 1: Filter to "most likely diagnosis" questions ───────────────────
    diagnosis_questions = [
        c for c in all_cases
        if any(
            phrase in (c.get("question") or c.get("sent1") or "").lower()
            for phrase in _DIAGNOSIS_QUESTION_PHRASES
        )
    ]
    print(f"  {len(diagnosis_questions):,} questions ask for 'most likely diagnosis'.")

    if not diagnosis_questions:
        print("  [ERROR] No diagnosis questions found in MedQA dataset.")
        return []

    # Shuffle to get variety
    random.shuffle(diagnosis_questions)

    # ── Step 2: Filter to textbook-covered conditions ─────────────────────────
    print(f"\n  Filtering to textbook-covered conditions (target: {num_cases} cases)...")
    kept: list[dict] = []
    checked = 0
    skipped_coverage = 0

    for raw in diagnosis_questions:
        if len(kept) >= num_cases:
            break

        checked += 1

        # Parse MedQA fields
        question_text: str = raw.get("question", "")
        correct_answer: str = raw.get("answer", "")
        options_raw = raw.get("options") or {}
        options_list = [f"{k}: {v}" for k, v in sorted(options_raw.items())] if isinstance(options_raw, dict) else []

        if not correct_answer.strip() or not question_text.strip():
            continue

        # First check: is this answer actually a DIAGNOSIS (not a treatment/mechanism/test)?
        try:
            dx_check = judge_llm.invoke([
                SystemMessage(content=(
                    "You classify medical answers. Is this answer a DIAGNOSIS "
                    "(a disease, condition, or syndrome)? Or is it something else "
                    "(a treatment, drug, mechanism, lab test, inheritance pattern, procedure)?\n\n"
                    "Respond with ONLY one word: diagnosis  or  other"
                )),
                HumanMessage(content=f"Answer: {correct_answer.strip()}"),
            ])
            if dx_check.content.strip().lower().split()[0] != "diagnosis":
                continue
        except Exception:
            continue

        # Then check: is this diagnosis covered in our textbook?
        filter_prompt = _MEDQA_FILTER_TEMPLATE.format(
            chapters_list=chapters_list_str,
            diagnosis=correct_answer.strip(),
        )

        try:
            resp = judge_llm.invoke([
                SystemMessage(content=_MEDQA_FILTER_SYSTEM),
                HumanMessage(content=filter_prompt),
            ])
            answer = resp.content.strip().lower().split()[0]
            is_covered = answer == "yes"
        except Exception as exc:
            print(f"    [WARN] Coverage check error: {exc}")
            is_covered = False

        if not is_covered:
            skipped_coverage += 1
            continue

        kept.append({
            "case_prompt": question_text,
            "ground_truth_diagnosis": correct_answer.strip(),
            "options": options_list,
            "source": "MedQA-USMLE",
        })

        print(
            f"  [KEPT {len(kept):>4}/{num_cases}] "
            f"dx={correct_answer.strip()!r} (checked {checked})"
        )

    # ── Save to disk ──────────────────────────────────────────────────────────
    with open(MEDQA_CASES_PATH, "w", encoding="utf-8") as fh:
        json.dump(kept, fh, indent=2, ensure_ascii=False)

    print(
        f"\n  MedQA filter complete: {len(kept)} kept, "
        f"{skipped_coverage} skipped (not covered), "
        f"{checked} total checked."
        f"\n  Saved to: {MEDQA_CASES_PATH}"
    )
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 — Pipeline Benchmark Runner
# ─────────────────────────────────────────────────────────────────────────────

class PipelineBenchmarkRunner:
    """
    Runs the full Medora RAG pipeline on a test set and measures both retrieval-
    level and generation-level metrics.

    Extra metrics beyond benchmark.py:
        retrieval_precision   — % of retrieved chunks from the correct chapter
        retrieval_recall      — did ANY retrieved chunk contain the GT diagnosis
        pipeline_accuracy     — % of cases where full pipeline gets it right
        retrieval_only_fail   — retrieval failed, LLM still correct (lucky)
        generation_only_fail  — retrieval OK, LLM still wrong (reasoning gap)
    """

    def __init__(
        self,
        test_cases: list[dict],
        models: list[dict],
        ollama_url: str = "http://localhost:11434",
        judge_model: str = JUDGE_MODEL,
        retrieve_k: int = RERANK_TOP_K_RETRIEVE,
        return_k: int = RERANK_TOP_K_RETURN,
        no_rag: bool = False,
    ):
        self.test_cases = test_cases
        self.models = models
        self.ollama_url = ollama_url
        self.retrieve_k = retrieve_k
        self.return_k = return_k
        self.no_rag = no_rag

        # ── Load RAG pipeline (shared across all model runs) — skip if no_rag ─
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

    # ── Retrieval quality helpers ─────────────────────────────────────────────

    @staticmethod
    def _compute_retrieval_recall(
        chunks: list[dict],
        ground_truth: str,
    ) -> bool:
        """
        Return True if ANY retrieved chunk contains meaningful information about
        the ground truth diagnosis (40% keyword overlap threshold).
        """
        gt_words = [w for w in ground_truth.lower().split() if len(w) > 3]
        if not gt_words:
            return False
        for chunk in chunks:
            chunk_text = chunk.get("text", "").lower()
            overlap = sum(1 for w in gt_words if w in chunk_text)
            if overlap / len(gt_words) >= 0.4:
                return True
        return False

    @staticmethod
    def _compute_retrieval_precision(
        chunks: list[dict],
        source_chapter: str | None,
    ) -> float:
        """
        Return the fraction of retrieved chunks that come from the same chapter
        as the ground truth source. Returns 0.0 if source_chapter is unknown.
        """
        if not source_chapter or not chunks:
            return 0.0
        hits = sum(
            1 for c in chunks
            if c.get("chapter", "").lower() == source_chapter.lower()
        )
        return hits / len(chunks)

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
        Run one test case through the full pipeline.

        Pipeline:
            1. case_prompt → bi-encoder retrieval
            2. BGE cross-encoder reranking
            3. Compute retrieval_recall and retrieval_precision
            4. LLM generates diagnosis from case_prompt + retrieved context
            5. Extract primary diagnosis from report
            6. Judge match against ground_truth_diagnosis
        """
        model_name = model_config["name"]
        provider = model_config["provider"]
        timeout = TIMEOUT_OPENAI if provider == "openai" else TIMEOUT_OLLAMA

        # ── Normalise case fields ─────────────────────────────────────────────
        case_prompt: str = (
            case.get("patient_presentation")
            or case.get("case_prompt")
            or case.get("question")
            or ""
        )
        ground_truth: str = (
            case.get("ground_truth_diagnosis")
            or case.get("final_diagnosis")
            or case.get("answer")
            or ""
        )
        source_chapter: str | None = (
            case.get("source_chapter") or case.get("chapter")
        )

        result: dict = {
            "case_idx": case_idx,
            "model": model_name,
            "case_prompt_preview": case_prompt[:200],
            "ground_truth": ground_truth,
            "source": case.get("source", "unknown"),
            "source_chapter": source_chapter,
            "difficulty": case.get("difficulty"),
            # ── Retrieval metrics ─────────────────────────────────────────────
            "retrieval_recall": False,
            "retrieval_precision": 0.0,
            # ── Generation metrics ────────────────────────────────────────────
            "system_diagnosis": None,
            "match_type": "mismatch",
            "is_correct": False,
            "json_adherence": False,
            # ── Failure attribution ───────────────────────────────────────────
            "retrieval_only_fail": False,   # retrieval failed but LLM correct
            "generation_only_fail": False,  # retrieval OK but LLM wrong
            # ── Timing / cost ────────────────────────────────────────────────
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
                prompt_tokens = _estimate_tokens(_DIAGNOSIS_SYSTEM + user_content)
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

                # ── Step 2: Retrieval quality metrics ─────────────────────────
                retrieval_recall = self._compute_retrieval_recall(chunks, ground_truth)
                retrieval_precision = self._compute_retrieval_precision(chunks, source_chapter)
                result["retrieval_recall"] = retrieval_recall
                result["retrieval_precision"] = round(retrieval_precision, 4)

                # ── Step 3: Build LLM prompt ──────────────────────────────────
                context = _chunks_to_context(chunks)
                user_content = (
                    f"Patient presentation:\n{case_prompt}\n\n"
                    f"Retrieved medical textbook passages:\n\n{context}"
                )
                prompt_tokens = _estimate_tokens(_DIAGNOSIS_SYSTEM + user_content)

            # ── Step 4: LLM generation with timeout ───────────────────────────
            def _timeout_handler(signum, frame):
                raise TimeoutError(f"LLM call exceeded {timeout}s timeout")

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
                    signal.alarm(0)

            t_end = time.perf_counter()
            result["latency_seconds"] = round(t_end - t_start, 3)
            result["tokens_used"] = prompt_tokens + _estimate_tokens(report_text)

            # ── Step 5: Extract primary diagnosis ─────────────────────────────
            system_dx = _extract_primary_diagnosis(report_text)
            result["json_adherence"] = system_dx is not None

            if system_dx is None:
                result["error"] = "Could not extract Primary Diagnosis section from report"
                print(f"    [WARN] No Primary Diagnosis section found.")
                # Attribution: if retrieval was fine but generation failed
                if retrieval_recall:
                    result["generation_only_fail"] = True
                return result

            result["system_diagnosis"] = system_dx

            # ── Step 6: Judge correctness ─────────────────────────────────────
            match_type = _judge_diagnosis_match(system_dx, ground_truth, self._judge_llm)
            result["match_type"] = match_type
            is_correct = match_type in ("exact_match", "semantic_match")
            result["is_correct"] = is_correct

            # ── Step 7: Failure attribution ───────────────────────────────────
            if not retrieval_recall and is_correct:
                # Retrieval missed but LLM still got it (parametric knowledge)
                result["retrieval_only_fail"] = True
            elif retrieval_recall and not is_correct:
                # Retrieval found the right content but LLM still got it wrong
                result["generation_only_fail"] = True

            flag = "CORRECT" if is_correct else "WRONG"
            print(
                f"    [{flag}] match={match_type}  "
                f"ret_recall={retrieval_recall}  ret_prec={retrieval_precision:.0%}  "
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
        """Run all test cases through one model. Returns per-case results."""
        model_name = model_config["name"]
        provider = model_config["provider"]

        print(f"\n{'='*70}")
        print(f"  Benchmarking model: {model_name}  ({provider})")
        print(f"{'='*70}")

        if provider == "ollama":
            if not _check_ollama_available(model_config, self.ollama_url):
                return []

        try:
            llm = _build_llm(model_config, self.ollama_url)
        except Exception as exc:
            print(f"  [ERROR] Could not build LLM for '{model_name}': {exc}")
            return []

        results: list[dict] = []
        total = len(self.test_cases)

        for i, case in enumerate(self.test_cases, start=1):
            case_result = self.run_single_case(case, llm, model_config, i, total)
            results.append(case_result)

        return results

    # ── All models ────────────────────────────────────────────────────────────

    def run_all(self) -> dict[str, list[dict]]:
        """Run all configured models sequentially."""
        all_results: dict[str, list[dict]] = {}
        for model_config in self.models:
            name = model_config["name"]
            case_results = self.run_model(model_config)
            all_results[name] = case_results
        return all_results

    # ── Metric computation ────────────────────────────────────────────────────

    def compute_metrics(self, results: list[dict]) -> dict:
        """
        Compute aggregate metrics from a list of per-case result dicts.

        Returns the standard benchmark.py metrics PLUS the pipeline-specific
        retrieval and failure-attribution metrics.
        """
        if not results:
            return {"num_cases": 0, "note": "no results"}

        num_cases = len(results)
        completed = [r for r in results if r.get("system_diagnosis") is not None]
        latencies = [r["latency_seconds"] for r in results if r["latency_seconds"] is not None]

        # ── Core accuracy ──────────────────────────────────────────────────────
        correct = sum(1 for r in results if r.get("is_correct", False))
        pipeline_accuracy = correct / num_cases if num_cases else 0.0

        # ── Match breakdown ────────────────────────────────────────────────────
        match_counts: dict[str, int] = {}
        for r in results:
            mt = r.get("match_type", "mismatch")
            match_counts[mt] = match_counts.get(mt, 0) + 1

        # ── Retrieval metrics (KEY — tells us if RAG is working) ───────────────
        recall_hits = sum(1 for r in results if r.get("retrieval_recall", False))
        retrieval_recall = recall_hits / num_cases if num_cases else 0.0

        prec_values = [r.get("retrieval_precision", 0.0) for r in results]
        retrieval_precision = sum(prec_values) / len(prec_values) if prec_values else 0.0

        # ── Failure attribution ────────────────────────────────────────────────
        retrieval_only_fail = sum(
            1 for r in results if r.get("retrieval_only_fail", False)
        )
        generation_only_fail = sum(
            1 for r in results if r.get("generation_only_fail", False)
        )
        retrieval_only_fail_rate = retrieval_only_fail / num_cases if num_cases else 0.0
        generation_only_fail_rate = generation_only_fail / num_cases if num_cases else 0.0

        # ── Latency ───────────────────────────────────────────────────────────
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
        sorted_lat = sorted(latencies)
        median_lat = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0
        p95_lat = sorted_lat[min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)] if sorted_lat else 0.0

        # ── JSON adherence ─────────────────────────────────────────────────────
        json_ok = sum(1 for r in results if r.get("json_adherence", False))
        json_error_rate = 1.0 - (json_ok / num_cases) if num_cases else 1.0

        # ── Tokens / errors ───────────────────────────────────────────────────
        total_tokens = sum(r.get("tokens_used", 0) for r in results)
        error_count = sum(1 for r in results if r.get("error") and not r.get("system_diagnosis"))

        # ── Difficulty breakdown (textbook cases only) ─────────────────────────
        difficulty_breakdown: dict[str, dict] = {}
        for r in results:
            diff = r.get("difficulty") or "unknown"
            if diff not in difficulty_breakdown:
                difficulty_breakdown[diff] = {"total": 0, "correct": 0}
            difficulty_breakdown[diff]["total"] += 1
            if r.get("is_correct"):
                difficulty_breakdown[diff]["correct"] += 1
        for diff, counts in difficulty_breakdown.items():
            t = counts["total"] or 1
            counts["accuracy"] = round(counts["correct"] / t, 4)

        return {
            "num_cases": num_cases,
            "num_completed": len(completed),
            "num_errors": error_count,
            # ── Core ──────────────────────────────────────────────────────────
            "pipeline_accuracy": round(pipeline_accuracy, 4),
            "match_breakdown": match_counts,
            # ── Retrieval (the KEY metrics) ────────────────────────────────────
            "retrieval_recall": round(retrieval_recall, 4),
            "retrieval_precision": round(retrieval_precision, 4),
            # ── Failure attribution ────────────────────────────────────────────
            "retrieval_only_fail_rate": round(retrieval_only_fail_rate, 4),
            "generation_only_fail_rate": round(generation_only_fail_rate, 4),
            # ── Latency ───────────────────────────────────────────────────────
            "mean_latency_s": round(mean_lat, 3),
            "median_latency_s": round(median_lat, 3),
            "p95_latency_s": round(p95_lat, 3),
            # ── Misc ──────────────────────────────────────────────────────────
            "json_error_rate": round(json_error_rate, 4),
            "total_tokens": total_tokens,
            "difficulty_breakdown": difficulty_breakdown,
        }

    # ── Save results ──────────────────────────────────────────────────────────

    def save_results(
        self,
        all_results: dict[str, list[dict]],
        test_set_label: str = "textbook",
    ) -> tuple[Path, Path]:
        """Save per-case and summary results. Returns (full_path, summary_path)."""
        EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # ── Full per-case results ─────────────────────────────────────────────
        full_path = EVAL_DATA_DIR / f"pipeline_benchmark_results_{ts}.json"
        full_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_set": test_set_label,
            "num_cases": len(self.test_cases),
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
        summary_path = EVAL_DATA_DIR / f"pipeline_benchmark_summary_{ts}.json"
        summary: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_set": test_set_label,
            "num_cases": len(self.test_cases),
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
        col_w = 20

        separator = "-" * (32 + col_w * len(model_names))
        header_sep = "=" * (32 + col_w * len(model_names))

        print(f"\n{header_sep}")
        print("  Medora — Pipeline Benchmark")
        print(f"  Cases: {len(self.test_cases):,}   "
              f"RAG: retrieve_k={self.retrieve_k}, return_k={self.return_k}")
        print(header_sep)

        row = f"  {'Metric':<30}"
        for name in model_names:
            row += f"  {name:<{col_w - 2}}"
        print(row)
        print(separator)

        def _pct(v: float) -> str:
            return f"{v:.1%}"

        def _sec(v: float) -> str:
            return f"{v:.2f}s"

        def _int(v: int) -> str:
            return f"{v:,}"

        metrics_to_display: list[tuple[str, str, Any]] = [
            # Core
            ("Pipeline Accuracy",         "pipeline_accuracy",           _pct),
            ("JSON Error Rate",           "json_error_rate",             _pct),
            # Retrieval (the critical metrics)
            ("Retrieval Recall",          "retrieval_recall",            _pct),
            ("Retrieval Precision",       "retrieval_precision",         _pct),
            # Failure attribution
            ("Retrieval-Only Fail Rate",  "retrieval_only_fail_rate",    _pct),
            ("Generation-Only Fail Rate", "generation_only_fail_rate",   _pct),
            # Latency
            ("Mean Latency",              "mean_latency_s",              _sec),
            ("Median Latency",            "median_latency_s",            _sec),
            ("P95 Latency",               "p95_latency_s",               _sec),
            # Misc
            ("Total Tokens",              "total_tokens",                _int),
            ("Errors",                    "num_errors",                  _int),
            ("Cases Run",                 "num_cases",                   _int),
        ]

        for label, key, fmt in metrics_to_display:
            row = f"  {label:<30}"
            for name in model_names:
                val = metrics_per_model[name].get(key, 0)
                row += f"  {fmt(val):<{col_w - 2}}"
            print(row)

        print(separator)

        # Match breakdown sub-table
        print("  Match breakdown:")
        for match_type in ("exact_match", "semantic_match", "partial_match", "mismatch"):
            row = f"    {match_type:<28}"
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
        description="Medora Phase 9 — Pipeline Benchmark (textbook + MedQA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate test cases from the textbook (ChromaDB → GPT-4o → JSON)",
    )
    parser.add_argument(
        "--filter-medqa",
        action="store_true",
        help="Filter MedQA-USMLE to textbook-covered conditions",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the pipeline benchmark on the selected test set",
    )
    parser.add_argument(
        "--test-set",
        choices=["textbook", "medqa", "both"],
        default="textbook",
        help="Which test set to run the benchmark on",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=50,
        metavar="N",
        help="Number of cases to generate or filter",
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
        "--ollama-url",
        default="http://localhost:11434",
        metavar="URL",
        help="Base URL of the Ollama server",
    )
    parser.add_argument(
        "--judge-model",
        default=JUDGE_MODEL,
        metavar="MODEL",
        help="OpenAI model to use as the diagnosis match judge",
    )
    parser.add_argument(
        "--generate-model",
        default=JUDGE_MODEL,
        metavar="MODEL",
        help="OpenAI model to use for test case generation",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=RERANK_TOP_K_RETRIEVE,
        metavar="N",
        help=f"Candidates to fetch from bi-encoder (default: {RERANK_TOP_K_RETRIEVE})",
    )
    parser.add_argument(
        "--return-k",
        type=int,
        default=RERANK_TOP_K_RETURN,
        metavar="N",
        help=f"Passages to keep after reranking (default: {RERANK_TOP_K_RETURN})",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Skip RAG retrieval — test raw model diagnostic ability only.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not (args.generate or args.filter_medqa or args.run):
        print(
            "[ERROR] Specify at least one action: --generate, --filter-medqa, or --run",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Ensure output directory exists ────────────────────────────────────────
    EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

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

    print("\n" + "=" * 70)
    print("  Medora — Pipeline Benchmark")
    print(f"  generate={args.generate}  filter-medqa={args.filter_medqa}  run={args.run}")
    print(f"  test-set={args.test_set}  num-cases={args.num_cases}")
    print(f"  generate-model={args.generate_model}  judge-model={args.judge_model}")
    if args.run:
        print(f"  models={[m['name'] for m in selected_models]}")
    print("=" * 70)

    # ── We may need the collection for generation and/or filtering ────────────
    _collection_cache: dict[str, Any] = {}

    def _get_collection():
        if "coll" not in _collection_cache:
            _collection_cache["coll"] = open_collection(CHROMA_DIR)
        return _collection_cache["coll"]

    # ─────────────────────────────────────────────────────────────────────────
    # Action: --generate
    # ─────────────────────────────────────────────────────────────────────────
    if args.generate:
        print("\n[STEP] Generating textbook test cases...")
        gen_llm = ChatOpenAI(model=args.generate_model, temperature=0.3)
        generate_textbook_cases(
            collection=_get_collection(),
            llm=gen_llm,
            num_cases=args.num_cases,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Action: --filter-medqa
    # ─────────────────────────────────────────────────────────────────────────
    if args.filter_medqa:
        print("\n[STEP] Filtering MedQA to textbook conditions...")
        filter_medqa_to_textbook(
            collection=_get_collection(),
            num_cases=args.num_cases,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Action: --run
    # ─────────────────────────────────────────────────────────────────────────
    if args.run:
        # Determine which test sets to load
        load_textbook = args.test_set in ("textbook", "both")
        load_medqa = args.test_set in ("medqa", "both")

        test_cases: list[dict] = []

        if load_textbook:
            if not TEXTBOOK_CASES_PATH.exists():
                print(
                    f"[ERROR] Textbook cases file not found: {TEXTBOOK_CASES_PATH}\n"
                    f"  Run with --generate first.",
                    file=sys.stderr,
                )
                sys.exit(1)
            with open(TEXTBOOK_CASES_PATH, encoding="utf-8") as fh:
                tb_cases = json.load(fh)
            print(f"\nLoaded {len(tb_cases):,} textbook test cases.")
            test_cases.extend(tb_cases)

        if load_medqa:
            if not MEDQA_CASES_PATH.exists():
                print(
                    f"[ERROR] MedQA cases file not found: {MEDQA_CASES_PATH}\n"
                    f"  Run with --filter-medqa first.",
                    file=sys.stderr,
                )
                sys.exit(1)
            with open(MEDQA_CASES_PATH, encoding="utf-8") as fh:
                mq_cases = json.load(fh)
            print(f"Loaded {len(mq_cases):,} MedQA filtered test cases.")
            test_cases.extend(mq_cases)

        if not test_cases:
            print("[ERROR] No test cases loaded.", file=sys.stderr)
            sys.exit(1)

        print(f"\nTotal test cases for benchmark: {len(test_cases):,}")

        # ── Build and run the benchmark ────────────────────────────────────────
        runner = PipelineBenchmarkRunner(
            test_cases=test_cases,
            models=selected_models,
            ollama_url=args.ollama_url,
            judge_model=args.judge_model,
            retrieve_k=args.retrieve_k,
            return_k=args.return_k,
            no_rag=args.no_rag,
        )

        t0 = time.perf_counter()
        all_results = runner.run_all()
        elapsed = time.perf_counter() - t0

        print(f"\nBenchmark complete in {elapsed / 60:.1f} minutes.")

        runner.print_comparison_table(all_results)
        runner.save_results(all_results, test_set_label=args.test_set)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
