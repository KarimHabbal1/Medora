"""
Phase 5 — Triage Agent
======================
A LangGraph-powered clinical DIAGNOSTIC engine that produces actual diagnoses
(not just "analysis") grounded in the medical textbook via RAG.

Two modes:

  Mode A — Common symptoms (from Intake Agent)
    Receives the structured intake summary (symptoms, Q&A answers, red flags,
    urgency). Single-pass RAG → diagnosis.

  Mode B — Uncommon symptoms (raw patient complaint)
    Receives only the raw complaint. Multi-pass RAG (max 3 passes):
      Pass 1: broad retrieval → generate 4-6 clinician questions
      Pass 2: patient answers → targeted retrieval → preliminary diagnosis
              evaluate whether critical finding warrants Pass 3
      Pass 3 (if needed): targeted search on critical finding → refine diagnosis
              HARD STOP after Pass 3.

Usage:
    python agents/triage_agent.py --from-intake data/results/last_intake.json
    python agents/triage_agent.py --query "I have a rash on my legs"
    python agents/triage_agent.py
    python agents/triage_agent.py --model gpt-4o-mini --retrieve-k 15 --return-k 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Annotated, TypedDict

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

# ── LangGraph / LangChain imports ─────────────────────────────────────────────
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# State definition
# ─────────────────────────────────────────────────────────────────────────────

class TriageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    # Input context
    mode: str                       # "common" or "uncommon"
    intake_summary: dict | None     # raw from Intake Agent (Mode A) — kept for reference
    clinical_picture: dict | None   # parsed clean version of intake_summary
    raw_complaint: str              # raw patient text (Mode B)

    # RAG results
    retrieved_chunks: list[dict]    # all unique chunks retrieved across passes

    # Multi-pass tracking (Mode B)
    current_pass: int               # 1, 2, or 3
    generated_questions: list[str]  # questions generated from Pass 1
    patient_answers: dict           # question → answer (collected during Pass 2)
    current_question_idx: int
    needs_refinement: bool          # whether Pass 3 is needed
    refinement_reason: str          # why Pass 3 was triggered
    refinement_search_query: str    # targeted query for Pass 3

    # Mode A follow-up tracking (when intake answers are insufficient)
    info_sufficient: bool           # True if intake answers are enough for diagnosis
    followup_questions: list[str]   # additional questions for Mode A
    followup_answers: dict          # follow-up question → answer
    followup_question_idx: int      # index into followup_questions
    followup_phase: bool            # True when in Mode A follow-up questioning

    # Output
    diagnosis: dict                 # the final diagnosis report
    diagnosis_complete: bool


# ─────────────────────────────────────────────────────────────────────────────
# Prompt constants
# ─────────────────────────────────────────────────────────────────────────────

_QUESTION_GENERATION_SYSTEM = """\
You are a clinical diagnostician. Based on the following medical textbook passages about
the patient's complaint, generate 4-6 specific questions that would help narrow the
differential diagnosis.

Rules:
- Ask questions that will differentiate between the most likely conditions
- Use patient-friendly language
- Focus on: onset/duration, character, aggravating/relieving factors, associated symptoms,
  relevant history
- Return a JSON array of question strings

Return ONLY the JSON array. No explanation, no markdown fences.\
"""

_DIAGNOSIS_SYSTEM = """\
You are a clinical diagnostic system. Based on the patient's presentation and the
following medical textbook passages, produce a structured diagnosis report.

CRITICAL: Base your diagnosis ONLY on the evidence from the provided textbook passages.
Do not add diagnoses or clinical reasoning from your general training data. If the
evidence is insufficient for a confident diagnosis, state this explicitly.

Structure your report as:

## Primary Diagnosis
[Most likely diagnosis with confidence level: high/moderate/low]

## Differential Diagnoses
[Ranked list, each with:
- Condition name
- Key supporting evidence from the patient's presentation
- Key evidence from the textbook passages
- Why it's more or less likely than the primary]

## Clinical Reasoning
[Step-by-step reasoning connecting the patient's symptoms/answers to the diagnosis,
citing specific textbook passages]

## Recommended Investigations
[Specific tests to confirm/rule out diagnoses, with clinical justification from the textbook]

## Management Considerations
[Initial management steps suggested by the textbook evidence]

## Red Flags & Safety Netting
[What to watch for that would change the diagnosis]

## Sources
[Each textbook passage used, with chapter, section, and what it contributed]\
"""

_REFINEMENT_EVALUATION_SYSTEM = """\
You are a clinical reasoning evaluator. After reviewing a patient's answers to clinical
questions, determine whether any answer revealed a CRITICAL FINDING that significantly
changes or narrows the differential diagnosis — warranting one additional targeted
evidence search.

A critical finding is one that:
- Introduces a new, specific diagnosis not previously considered
- Strongly rules in or rules out a major condition
- Reveals an important historical fact (trauma, prior procedure, family history) that
  redirects the clinical picture

Respond with ONLY a valid JSON object:
{
  "needs_refinement": true or false,
  "reason": "one sentence explaining what the critical finding is, or null if no refinement needed",
  "search_query": "a specific targeted search query for the critical finding, or null if not needed"
}

No explanation, no markdown.\
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_fence(text: str) -> str:
    """Remove markdown code fences from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        inner = parts[1] if len(parts) > 1 else text
        if inner.startswith("json"):
            inner = inner[4:]
        return inner.strip()
    return text


def _chunks_to_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for LLM injection."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        chapter = chunk.get("chapter", "Unknown Chapter")
        section = chunk.get("section", "Unknown Section")
        text = chunk.get("text", "").strip()
        score = chunk.get("rerank_score")
        score_str = f"  [rerank score: {score:.4f}]" if score is not None else ""
        parts.append(
            f"--- Passage {i} ---\n"
            f"Chapter : {chapter}\n"
            f"Section : {section}{score_str}\n\n"
            f"{text}"
        )
    return "\n\n".join(parts)


def _format_qa(qa: dict) -> str:
    """Format a question→answer dict as a readable block."""
    if not qa:
        return "  (none)"
    lines = []
    for q, a in qa.items():
        lines.append(f"  Q: {q}\n  A: {a}")
    return "\n".join(lines)


def _deduplicate_chunks(
    existing: list[dict],
    new_chunks: list[dict],
    seen_ids: set[str],
) -> tuple[list[dict], set[str]]:
    """Add new_chunks to existing, deduplicating by chunk_id."""
    result = list(existing)
    for chunk in new_chunks:
        cid = chunk.get("chunk_id", "")
        if cid not in seen_ids:
            result.append(chunk)
            seen_ids.add(cid)
    return result, seen_ids


def parse_intake_to_clinical_picture(intake_summary: dict, llm: ChatOpenAI) -> dict:
    """
    Distill a raw intake summary into a clean clinical picture.

    Strips out conversation artifacts (question text, clinician note, specialty routing,
    workup lists) and produces a structured clinical picture with key-value pairs
    that the Triage Agent can reason over.

    Returns:
        {
            "symptoms": ["Chest Pain", "Hemoptysis"],
            "urgency": "urgent",
            "clinical_findings": {
                "pain_quality": "extremely tight",
                "pain_radiation": "arms",
                ...
            },
            "red_flags": [{"flag": "...", "urgency": "..."}]
        }
    """
    system_prompt = """You are a clinical data parser. Given a set of intake question-answer pairs
from a patient interview, extract and organize the clinical information into a clean
structured format.

Rules:
- Strip out the question text — only keep the clinical information from the answers
- Use short, descriptive keys (e.g., "pain_quality", "onset", "smoking_history")
- Combine related information (if the patient gave follow-up details, merge them)
- Remove redundant information (if the same fact appears in multiple answers, keep it once)
- Keep the clinical meaning intact — do not interpret or diagnose, just organize

Return a JSON object with descriptive keys and concise values.
For example:
{
    "onset": "1 week ago, sudden worsening 2 days ago",
    "pain_quality": "extremely tight",
    "pain_radiation": "bilateral arms",
    "triggers": "exercise worsens",
    "associated_symptoms": "shortness of breath, hemoptysis",
    "cardiovascular_history": "previous stroke (5 years ago)",
    "smoking_history": "10 years, quit 2 years ago"
}

Return ONLY valid JSON."""

    answers = intake_summary.get("answers", {})

    if not answers:
        return {
            "symptoms": intake_summary.get("symptoms", []),
            "urgency": intake_summary.get("urgency", "routine"),
            "clinical_findings": {},
            "red_flags": intake_summary.get("triggered_red_flags", []),
        }

    answers_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in answers.items())

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Intake Q&A pairs:\n\n{answers_text}"),
    ])

    raw = _strip_fence(response.content)
    try:
        clinical_findings = json.loads(raw)
        if not isinstance(clinical_findings, dict):
            clinical_findings = {}
    except (json.JSONDecodeError, ValueError):
        clinical_findings = {}

    # Only keep red flag name and urgency — strip implications (the Triage Agent
    # will form its own clinical reasoning)
    clean_red_flags = [
        {"flag": rf.get("flag", ""), "urgency": rf.get("urgency", "routine")}
        for rf in intake_summary.get("triggered_red_flags", [])
        if rf.get("flag")
    ]

    return {
        "symptoms": intake_summary.get("symptoms", []),
        "urgency": intake_summary.get("urgency", "routine"),
        "clinical_findings": clinical_findings,
        "red_flags": clean_red_flags,
    }


def _clinical_picture_to_query(clinical_picture: dict) -> str:
    """Build a RAG query from a parsed clinical picture."""
    symptoms = clinical_picture.get("symptoms", [])
    urgency = clinical_picture.get("urgency", "routine")
    findings = clinical_picture.get("clinical_findings", {})
    red_flags = clinical_picture.get("red_flags", [])

    symptom_str = ", ".join(symptoms) if symptoms else "unspecified complaint"
    parts = [f"Patient presenting with: {symptom_str}. Urgency: {urgency}."]

    if findings:
        parts.append("Clinical findings:")
        for key, value in findings.items():
            label = key.replace("_", " ")
            parts.append(f"  - {label}: {value}")

    if red_flags:
        flag_names = [rf.get("flag", "") for rf in red_flags if rf.get("flag")]
        if flag_names:
            parts.append(f"Red flags present: {', '.join(flag_names)}.")

    return " ".join(parts)


def _clinical_picture_to_context(clinical_picture: dict) -> str:
    """Format a clinical picture for injection into LLM prompts."""
    symptoms = clinical_picture.get("symptoms", [])
    urgency = clinical_picture.get("urgency", "routine")
    findings = clinical_picture.get("clinical_findings", {})
    red_flags = clinical_picture.get("red_flags", [])

    lines = [
        f"Presenting symptoms: {', '.join(symptoms) if symptoms else 'N/A'}",
        f"Urgency level: {urgency.upper()}",
        "",
        "Clinical findings:",
    ]
    if findings:
        for key, value in findings.items():
            label = key.replace("_", " ").title()
            lines.append(f"  {label}: {value}")
    else:
        lines.append("  (none recorded)")

    if red_flags:
        lines.append("")
        lines.append("Red flags:")
        for rf in red_flags:
            lines.append(f"  - {rf.get('flag', '')} [{rf.get('urgency', '')}]")

    return "\n".join(lines)


def _build_pass2_query(raw_complaint: str, patient_answers: dict) -> str:
    """Build a refined query from the complaint + all patient answers."""
    parts = [f"Patient complaint: {raw_complaint}."]
    if patient_answers:
        parts.append("Patient history from follow-up questions:")
        for q, a in patient_answers.items():
            parts.append(f"  - {q}: {a}")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Node builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_analyze_input_node(llm: ChatOpenAI):
    """
    Determines mode from state. For common mode, parses the raw intake
    summary into a clean clinical picture (stripping question text,
    clinician notes, and other artifacts).
    """
    def analyze_input(state: TriageState) -> dict:
        intake_summary = state.get("intake_summary")
        raw_complaint = state.get("raw_complaint", "")

        clinical_picture = None
        if intake_summary and intake_summary.get("answers"):
            mode = "common"
            print("[Triage] Parsing intake summary into clean clinical picture...")
            clinical_picture = parse_intake_to_clinical_picture(intake_summary, llm)
            findings = clinical_picture.get("clinical_findings", {})
            print(f"[Triage] Extracted {len(findings)} clinical findings:")
            for key, value in findings.items():
                print(f"  {key}: {value}")
        else:
            mode = "uncommon"

        return {
            "mode": mode,
            "clinical_picture": clinical_picture,
            "current_pass": 1,
            "retrieved_chunks": [],
            "generated_questions": [],
            "patient_answers": {},
            "current_question_idx": 0,
            "needs_refinement": False,
            "refinement_reason": "",
            "refinement_search_query": "",
            "diagnosis": {},
            "diagnosis_complete": False,
        }

    return analyze_input


def _build_check_sufficiency_node(llm: ChatOpenAI):
    """
    Mode A only: Criteria-based sufficiency check.

    Step 1: Extract diagnostic criteria from the retrieved textbook passages
            (what information does the textbook say is needed to diagnose?)
    Step 2: Compare against the intake answers (what do we actually know?)
    Step 3: Identify specific gaps (which criteria are not covered?)
    Step 4: Generate follow-up questions targeting those gaps

    This is grounded in the textbook, not LLM judgment.
    """
    extract_criteria_prompt = """You are a clinical knowledge extraction system.
From the following medical textbook passages, extract the KEY DIAGNOSTIC CRITERIA —
the specific clinical findings and history points that the textbook says are needed
to differentiate between the likely conditions.

Focus on information from sections like "Essentials of Diagnosis", "Clinical Findings",
"Symptoms and Signs", and "General Considerations".

IMPORTANT: Categorize each criterion by whether a PATIENT can provide it:

- "patient_reportable": true — things a patient can tell you about:
  onset pattern, duration, symptom character, triggers, associated symptoms,
  medication use, past medical history, family history, lifestyle factors,
  travel history, exposure history, allergies

- "patient_reportable": false — things that require a clinician:
  physical exam findings (heart murmurs, lung crackles, skin examination details),
  lab results (blood tests, D-dimer, troponin, cultures),
  imaging (X-ray, CT, MRI, ultrasound),
  procedures (ECG interpretation, biopsy results)

Return a JSON array of criteria objects:
[
    {
        "criterion": "specific finding or information needed",
        "why": "which condition(s) it helps identify or rule out",
        "category": "history" | "symptom" | "examination" | "investigation",
        "patient_reportable": true | false
    }
]

Only include criteria that are DIAGNOSTICALLY IMPORTANT — things that would change
which condition is most likely. Skip generic criteria like "take vital signs."
Aim for 5-10 key criteria.

Return ONLY valid JSON."""

    gap_analysis_prompt = """You are a clinical gap analysis system.
Given:
1. A list of diagnostic criteria needed (from the medical textbook), each marked
   as patient_reportable (true/false)
2. What we actually know from the patient's intake answers

Perform a structured comparison:
- ONLY analyze criteria where patient_reportable is TRUE
  (ignore examination and investigation criteria — the patient cannot provide those)
- For each patient-reportable criterion, determine if the intake answers COVER it
- Identify CRITICAL GAPS — patient-reportable criteria not covered that would
  significantly change the differential diagnosis

Return a JSON object:
{
    "covered": [{"criterion": "...", "covered_by": "which answer covers this"}],
    "patient_gaps": [{"criterion": "...", "why_critical": "how knowing this would change the diagnosis"}],
    "clinician_gaps": [{"criterion": "...", "note": "requires exam or test — include in recommendations"}],
    "sufficient": true/false,
    "followup_questions": ["patient-friendly question targeting gap 1", "..."]
}

Rules for followup_questions:
- ONLY target patient_gaps (things the patient CAN answer)
- NEVER ask about examination findings, lab results, or imaging
- Maximum 3 questions
- Use patient-friendly language
- Each question should target a specific missing criterion

clinician_gaps should list the examination/investigation criteria that are missing —
these will be included in the diagnosis report as recommended next steps for the doctor.

If there are no critical patient_gaps, set "sufficient": true and "followup_questions": [].

Return ONLY valid JSON."""

    def check_sufficiency(state: TriageState) -> dict:
        mode = state.get("mode", "uncommon")
        if mode != "common":
            return {"info_sufficient": True}

        clinical_picture = state.get("clinical_picture") or {}
        retrieved_chunks = state.get("retrieved_chunks", [])
        context = _chunks_to_context(retrieved_chunks)
        clinical_context = _clinical_picture_to_context(clinical_picture)

        # Step 1: Extract diagnostic criteria from textbook passages
        print("[Triage] Extracting diagnostic criteria from textbook passages...")
        criteria_response = llm.invoke([
            SystemMessage(content=extract_criteria_prompt),
            HumanMessage(content=f"Medical textbook passages:\n\n{context}"),
        ])

        criteria_raw = _strip_fence(criteria_response.content)
        try:
            criteria_list = json.loads(criteria_raw)
            if not isinstance(criteria_list, list):
                criteria_list = []
        except (json.JSONDecodeError, ValueError):
            criteria_list = []

        if not criteria_list:
            print("[Triage] Could not extract criteria — proceeding with diagnosis.")
            return {"info_sufficient": True}

        print(f"[Triage] Extracted {len(criteria_list)} diagnostic criteria.")

        # Step 2 + 3 + 4: Gap analysis — compare criteria against intake answers
        print("[Triage] Running gap analysis against intake answers...")
        criteria_json = json.dumps(criteria_list, indent=2)

        gap_response = llm.invoke([
            SystemMessage(content=gap_analysis_prompt),
            HumanMessage(
                content=(
                    f"Diagnostic criteria needed:\n{criteria_json}\n\n"
                    f"Patient's intake information:\n{clinical_context}"
                )
            ),
        ])

        gap_raw = _strip_fence(gap_response.content)
        try:
            gap_result = json.loads(gap_raw)
            is_sufficient = gap_result.get("sufficient", True)
            covered = gap_result.get("covered", [])
            patient_gaps = gap_result.get("patient_gaps", [])
            clinician_gaps = gap_result.get("clinician_gaps", [])
            followups = gap_result.get("followup_questions", []) if not is_sufficient else []
        except (json.JSONDecodeError, ValueError):
            is_sufficient = True
            covered = []
            patient_gaps = []
            clinician_gaps = []
            followups = []

        # Log the analysis
        print(f"[Triage] Criteria covered: {len(covered)}")
        print(f"[Triage] Patient-reportable gaps: {len(patient_gaps)}")
        print(f"[Triage] Clinician-only gaps: {len(clinician_gaps)} (→ recommended investigations)")
        if patient_gaps:
            for g in patient_gaps:
                print(f"  PATIENT GAP: {g.get('criterion', '?')} — {g.get('why_critical', '?')}")
        if clinician_gaps:
            for g in clinician_gaps:
                print(f"  CLINICIAN GAP: {g.get('criterion', '?')} — {g.get('note', '?')}")

        if is_sufficient or not followups:
            print("[Triage] Information sufficient for diagnosis.")
            return {"info_sufficient": True}

        print(f"[Triage] {len(gaps)} critical gap(s) found — generating {len(followups)} follow-up question(s).")
        return {
            "info_sufficient": False,
            "followup_questions": followups[:3],  # hard cap at 3
            "followup_answers": {},
            "followup_question_idx": 0,
            "followup_phase": True,
        }

    return check_sufficiency


def _build_ask_followup_mode_a_node():
    """Mode A follow-up: asks the next follow-up question."""
    def ask_followup_mode_a(state: TriageState) -> dict:
        questions = state.get("followup_questions", [])
        idx = state.get("followup_question_idx", 0)

        if idx >= len(questions):
            return {"followup_phase": False}

        question = questions[idx]
        print(f"\n[Triage] Follow-up question {idx + 1}/{len(questions)}: {question}")
        return {"messages": [AIMessage(content=question)]}

    return ask_followup_mode_a


def _build_process_followup_answer_node():
    """Mode A follow-up: records the patient's answer to a follow-up question."""
    def process_followup_answer(state: TriageState) -> dict:
        human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not human_msgs:
            return {}

        answer = human_msgs[-1].content
        questions = state.get("followup_questions", [])
        idx = state.get("followup_question_idx", 0)
        followup_answers = dict(state.get("followup_answers", {}))

        if idx < len(questions):
            followup_answers[questions[idx]] = answer
            print(f"[Triage] Recorded follow-up answer {idx + 1}/{len(questions)}.")

        next_idx = idx + 1
        all_done = next_idx >= len(questions)

        return {
            "followup_answers": followup_answers,
            "followup_question_idx": next_idx,
            "followup_phase": not all_done,
        }

    return process_followup_answer


def _build_retrieve_evidence_node(collection, bi_encoder, reranker, retrieve_k: int, return_k: int):
    """
    Runs RAG: bi-encoder retrieval → cross-encoder reranking.
    Builds different queries depending on mode and current pass.
    Deduplicates chunks across passes.
    """
    def retrieve_evidence(state: TriageState) -> dict:
        mode = state.get("mode", "uncommon")
        current_pass = state.get("current_pass", 1)
        intake_summary = state.get("intake_summary")
        raw_complaint = state.get("raw_complaint", "")
        patient_answers = state.get("patient_answers", {})
        refinement_search_query = state.get("refinement_search_query", "")
        existing_chunks = state.get("retrieved_chunks", [])

        # Build the query based on mode and pass
        clinical_picture = state.get("clinical_picture")
        if mode == "common":
            query = _clinical_picture_to_query(clinical_picture or {})
        elif current_pass == 1:
            query = raw_complaint
        elif current_pass == 2:
            query = _build_pass2_query(raw_complaint, patient_answers)
        else:
            # Pass 3: targeted search on the critical finding
            query = refinement_search_query if refinement_search_query else _build_pass2_query(raw_complaint, patient_answers)

        print(f"\n[Triage] Retrieving evidence (mode={mode}, pass={current_pass})...")
        print(f"[Triage] Query: {query[:120]}{'...' if len(query) > 120 else ''}")

        new_chunks = retrieve_and_rerank(
            query, collection, bi_encoder, reranker, retrieve_k, return_k
        )

        # Deduplicate across passes
        seen_ids: set[str] = {c.get("chunk_id", "") for c in existing_chunks}
        all_chunks, _ = _deduplicate_chunks(existing_chunks, new_chunks, seen_ids)

        print(f"[Triage] Retrieved {len(new_chunks)} chunks, total unique: {len(all_chunks)}")

        return {"retrieved_chunks": all_chunks}

    return retrieve_evidence


def _build_generate_questions_node(llm: ChatOpenAI):
    """
    Mode B, Pass 1 only.
    Uses retrieved chunks to generate 4-6 clinical questions.
    """
    def generate_questions(state: TriageState) -> dict:
        retrieved_chunks = state.get("retrieved_chunks", [])
        raw_complaint = state.get("raw_complaint", "")

        context = _chunks_to_context(retrieved_chunks)

        response = llm.invoke([
            SystemMessage(content=_QUESTION_GENERATION_SYSTEM),
            HumanMessage(
                content=(
                    f"Patient complaint: {raw_complaint}\n\n"
                    f"Medical textbook passages:\n\n{context}"
                )
            ),
        ])

        raw = _strip_fence(response.content)
        try:
            questions = json.loads(raw)
            if not isinstance(questions, list):
                questions = [raw]
        except (json.JSONDecodeError, ValueError):
            questions = [raw]

        print(f"[Triage] Generated {len(questions)} clinical questions.")

        return {
            "generated_questions": questions,
            "current_question_idx": 0,
        }

    return generate_questions


def _build_ask_question_node():
    """
    Mode B only.
    Emits the next question from generated_questions as an AI message.
    Graph pauses here for user input.
    """
    def ask_question(state: TriageState) -> dict:
        questions = state.get("generated_questions", [])
        idx = state.get("current_question_idx", 0)

        if idx >= len(questions):
            # All questions answered — mark complete for routing
            return {"current_question_idx": idx}

        question_text = questions[idx]
        msg = AIMessage(content=question_text)
        print(f"\n[Triage] Question {idx + 1}/{len(questions)}: {question_text}")

        return {"messages": [msg]}

    return ask_question


def _build_process_answer_node():
    """
    Mode B only.
    Records the patient's answer and increments the question index.
    When all questions are answered, signals that retrieval (Pass 2) should run.
    """
    def process_answer(state: TriageState) -> dict:
        human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not human_msgs:
            return {}

        patient_answer = human_msgs[-1].content
        questions = state.get("generated_questions", [])
        idx = state.get("current_question_idx", 0)

        if idx >= len(questions):
            return {}

        current_question = questions[idx]
        patient_answers = dict(state.get("patient_answers", {}))
        patient_answers[current_question] = patient_answer

        next_idx = idx + 1
        all_done = next_idx >= len(questions)

        print(f"[Triage] Recorded answer {idx + 1}/{len(questions)}.")

        return {
            "patient_answers": patient_answers,
            "current_question_idx": next_idx,
        }

    return process_answer


def _build_evaluate_refinement_node(llm: ChatOpenAI):
    """
    Mode B only, runs after Pass 2 diagnosis.
    Evaluates whether any patient answer revealed a critical finding.
    """
    def evaluate_refinement(state: TriageState) -> dict:
        raw_complaint = state.get("raw_complaint", "")
        patient_answers = state.get("patient_answers", {})
        diagnosis = state.get("diagnosis", {})
        diagnosis_text = diagnosis.get("report", "")

        qa_block = _format_qa(patient_answers)

        response = llm.invoke([
            SystemMessage(content=_REFINEMENT_EVALUATION_SYSTEM),
            HumanMessage(
                content=(
                    f"Patient complaint: {raw_complaint}\n\n"
                    f"Patient answers to clinical questions:\n{qa_block}\n\n"
                    f"Preliminary diagnosis:\n{diagnosis_text[:2000]}"
                )
            ),
        ])

        raw = _strip_fence(response.content)
        try:
            parsed = json.loads(raw)
            needs_refinement = bool(parsed.get("needs_refinement", False))
            reason = parsed.get("reason") or ""
            search_query = parsed.get("search_query") or ""
        except (json.JSONDecodeError, ValueError):
            needs_refinement = False
            reason = ""
            search_query = ""

        if needs_refinement:
            print(f"[Triage] Pass 3 triggered: {reason}")
        else:
            print("[Triage] No refinement needed. Diagnosis complete.")

        return {
            "needs_refinement": needs_refinement,
            "refinement_reason": reason,
            "refinement_search_query": search_query,
        }

    return evaluate_refinement


def _build_generate_diagnosis_node(llm: ChatOpenAI):
    """
    The core node. Produces the actual diagnosis from all retrieved chunks
    plus all available clinical context.
    """
    def generate_diagnosis(state: TriageState) -> dict:
        mode = state.get("mode", "uncommon")
        retrieved_chunks = state.get("retrieved_chunks", [])
        intake_summary = state.get("intake_summary")
        raw_complaint = state.get("raw_complaint", "")
        patient_answers = state.get("patient_answers", {})
        current_pass = state.get("current_pass", 1)
        needs_refinement = state.get("needs_refinement", False)

        context = _chunks_to_context(retrieved_chunks)

        if mode == "common":
            clinical_picture = state.get("clinical_picture") or {}
            clinical_context = _clinical_picture_to_context(clinical_picture)
            followup_answers = state.get("followup_answers", {})
            followup_block = ""
            if followup_answers:
                followup_block = "\n\nAdditional follow-up findings:\n" + "\n".join(
                    f"  {q}: {a}" for q, a in followup_answers.items()
                )
            user_content = (
                f"Patient clinical context:\n{clinical_context}{followup_block}\n\n"
                f"Retrieved medical textbook passages:\n\n{context}"
            )
        else:
            qa_block = _format_qa(patient_answers)
            user_content = (
                f"Patient complaint: {raw_complaint}\n\n"
                f"Clinical history from follow-up questions:\n{qa_block}\n\n"
                f"Retrieved medical textbook passages:\n\n{context}"
            )

        print(f"\n[Triage] Generating diagnosis (mode={mode}, pass={current_pass})...")

        response = llm.invoke([
            SystemMessage(content=_DIAGNOSIS_SYSTEM),
            HumanMessage(content=user_content),
        ])
        report_text = response.content.strip()

        # Mark complete if: Mode A, or Pass 3 done, or pass >= 3 hard cap
        is_final = (
            mode == "common"
            or current_pass >= 3
            or (current_pass == 2 and not needs_refinement)
        )

        diagnosis = {
            "report": report_text,
            "mode": mode,
            "pass": current_pass,
            "num_chunks_used": len(retrieved_chunks),
        }

        return {
            "diagnosis": diagnosis,
            "diagnosis_complete": is_final,
        }

    return generate_diagnosis


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_graph(
    llm: ChatOpenAI,
    collection,
    bi_encoder,
    reranker,
    retrieve_k: int,
    return_k: int,
) -> object:
    """Build and compile the LangGraph triage workflow."""

    analyze_input_fn = _build_analyze_input_node(llm)
    retrieve_evidence_fn = _build_retrieve_evidence_node(collection, bi_encoder, reranker, retrieve_k, return_k)
    check_sufficiency_fn = _build_check_sufficiency_node(llm)
    ask_followup_mode_a_fn = _build_ask_followup_mode_a_node()
    process_followup_answer_fn = _build_process_followup_answer_node()
    generate_questions_fn = _build_generate_questions_node(llm)
    ask_question_fn = _build_ask_question_node()
    process_answer_fn = _build_process_answer_node()
    evaluate_refinement_fn = _build_evaluate_refinement_node(llm)
    generate_diagnosis_fn = _build_generate_diagnosis_node(llm)

    # ── Routing functions ──────────────────────────────────────────────────────

    def route_after_analyze(state: TriageState) -> str:
        return "retrieve_evidence"

    def route_after_retrieve(state: TriageState) -> str:
        mode = state.get("mode", "uncommon")
        current_pass = state.get("current_pass", 1)

        if mode == "common":
            return "check_sufficiency"

        # Mode B
        if current_pass == 1:
            return "generate_questions"
        # Pass 2 or 3 — generate diagnosis
        return "generate_diagnosis"

    def route_after_sufficiency(state: TriageState) -> str:
        if state.get("info_sufficient", True):
            return "generate_diagnosis"
        return "ask_followup_mode_a"

    def route_after_ask_followup_mode_a(state: TriageState) -> str:
        return END  # pause for user input

    def route_after_process_followup(state: TriageState) -> str:
        if state.get("followup_phase", False):
            return "ask_followup_mode_a"
        # All follow-ups answered — do a second retrieval with enriched context, then diagnose
        return "retrieve_evidence_enriched"

    def route_after_questions(state: TriageState) -> str:
        return "ask_question"

    def route_after_ask_question(state: TriageState) -> str:
        # Graph pauses here — this edge leads to END so user can respond
        return END

    def route_after_process_answer(state: TriageState) -> str:
        questions = state.get("generated_questions", [])
        idx = state.get("current_question_idx", 0)

        if idx < len(questions):
            # More questions remain
            return "ask_question"
        # All questions answered → retrieve for Pass 2
        return "retrieve_evidence_pass2"

    def route_after_diagnosis(state: TriageState) -> str:
        mode = state.get("mode", "uncommon")
        current_pass = state.get("current_pass", 1)

        if mode == "common":
            return END

        if current_pass >= 3:
            # Hard cap — always done after Pass 3
            return END

        if current_pass == 2:
            return "evaluate_refinement"

        return END

    def route_after_evaluate_refinement(state: TriageState) -> str:
        if state.get("needs_refinement", False):
            return "retrieve_evidence_pass3"
        return END

    # ── Graph assembly ─────────────────────────────────────────────────────────
    graph = StateGraph(TriageState)

    graph.add_node("analyze_input", analyze_input_fn)
    graph.add_node("retrieve_evidence", retrieve_evidence_fn)

    # We need two additional retrieve_evidence nodes for pass 2 and pass 3
    # since LangGraph requires unique node names. We wrap the same function
    # but update current_pass in state before calling it.

    def retrieve_evidence_pass2(state: TriageState) -> dict:
        updated = dict(state)
        updated["current_pass"] = 2
        result = retrieve_evidence_fn(updated)
        return {**result, "current_pass": 2}

    def retrieve_evidence_pass3(state: TriageState) -> dict:
        updated = dict(state)
        updated["current_pass"] = 3
        result = retrieve_evidence_fn(updated)
        return {**result, "current_pass": 3}

    def generate_diagnosis_pass2(state: TriageState) -> dict:
        updated = dict(state)
        updated["current_pass"] = 2
        result = generate_diagnosis_fn(updated)
        return result

    def generate_diagnosis_pass3(state: TriageState) -> dict:
        updated = dict(state)
        updated["current_pass"] = 3
        updated["needs_refinement"] = False  # force final on pass 3
        result = generate_diagnosis_fn(updated)
        return {**result, "diagnosis_complete": True}

    # Mode A sufficiency check + follow-up nodes
    graph.add_node("check_sufficiency", check_sufficiency_fn)
    graph.add_node("ask_followup_mode_a", ask_followup_mode_a_fn)
    graph.add_node("process_followup_answer", process_followup_answer_fn)

    def retrieve_evidence_enriched(state: TriageState) -> dict:
        """Re-retrieve with enriched context (clinical picture + follow-up answers)."""
        clinical_picture = state.get("clinical_picture") or {}
        followup_answers = state.get("followup_answers", {})
        base_query = _clinical_picture_to_query(clinical_picture)
        followup_context = " ".join(f"{q}: {a}" for q, a in followup_answers.items())
        enriched_query = f"{base_query} Additional context: {followup_context}"

        print(f"\n[Triage] Re-retrieving with enriched context (follow-up answers included)...")
        print(f"[Triage] Query: {enriched_query[:120]}{'...' if len(enriched_query) > 120 else ''}")

        existing_chunks = state.get("retrieved_chunks", [])
        new_chunks = retrieve_and_rerank(
            enriched_query, collection, bi_encoder, reranker, retrieve_k, return_k
        )
        seen_ids: set[str] = {c.get("chunk_id", "") for c in existing_chunks}
        all_chunks, _ = _deduplicate_chunks(existing_chunks, new_chunks, seen_ids)
        print(f"[Triage] Retrieved {len(new_chunks)} chunks, total unique: {len(all_chunks)}")
        return {"retrieved_chunks": all_chunks}

    graph.add_node("retrieve_evidence_enriched", retrieve_evidence_enriched)

    # Mode B nodes
    graph.add_node("generate_questions", generate_questions_fn)
    graph.add_node("ask_question", ask_question_fn)
    graph.add_node("process_answer", process_answer_fn)
    graph.add_node("retrieve_evidence_pass2", retrieve_evidence_pass2)
    graph.add_node("retrieve_evidence_pass3", retrieve_evidence_pass3)
    graph.add_node("generate_diagnosis", generate_diagnosis_fn)      # Mode A / pass-1 fallback
    graph.add_node("generate_diagnosis_pass2", generate_diagnosis_pass2)
    graph.add_node("generate_diagnosis_pass3", generate_diagnosis_pass3)
    graph.add_node("evaluate_refinement", evaluate_refinement_fn)

    # ── Edges ──────────────────────────────────────────────────────────────────

    graph.add_edge(START, "analyze_input")

    graph.add_conditional_edges(
        "analyze_input",
        route_after_analyze,
        {"retrieve_evidence": "retrieve_evidence"},
    )

    graph.add_conditional_edges(
        "retrieve_evidence",
        route_after_retrieve,
        {
            "check_sufficiency": "check_sufficiency",
            "generate_diagnosis": "generate_diagnosis",
            "generate_questions": "generate_questions",
        },
    )

    # Mode A sufficiency check path
    graph.add_conditional_edges(
        "check_sufficiency",
        route_after_sufficiency,
        {
            "generate_diagnosis": "generate_diagnosis",
            "ask_followup_mode_a": "ask_followup_mode_a",
        },
    )

    # Mode A follow-up path
    graph.add_edge("ask_followup_mode_a", END)  # pause for user input

    graph.add_conditional_edges(
        "process_followup_answer",
        route_after_process_followup,
        {
            "ask_followup_mode_a": "ask_followup_mode_a",
            "retrieve_evidence_enriched": "retrieve_evidence_enriched",
        },
    )

    graph.add_edge("retrieve_evidence_enriched", "generate_diagnosis")

    # Mode A direct diagnosis path (when sufficient)
    graph.add_edge("generate_diagnosis", END)

    # Mode B Pass 1 path
    graph.add_edge("generate_questions", "ask_question")

    # ask_question → END (pause for user input; session resumes at process_answer)
    graph.add_edge("ask_question", END)

    graph.add_conditional_edges(
        "process_answer",
        route_after_process_answer,
        {
            "ask_question": "ask_question",
            "retrieve_evidence_pass2": "retrieve_evidence_pass2",
        },
    )

    graph.add_edge("retrieve_evidence_pass2", "generate_diagnosis_pass2")

    graph.add_conditional_edges(
        "generate_diagnosis_pass2",
        route_after_diagnosis,
        {
            END: END,
            "evaluate_refinement": "evaluate_refinement",
        },
    )

    graph.add_conditional_edges(
        "evaluate_refinement",
        route_after_evaluate_refinement,
        {
            "retrieve_evidence_pass3": "retrieve_evidence_pass3",
            END: END,
        },
    )

    graph.add_edge("retrieve_evidence_pass3", "generate_diagnosis_pass3")
    graph.add_edge("generate_diagnosis_pass3", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# TriageSession — manages a single triage conversation
# ─────────────────────────────────────────────────────────────────────────────

class TriageSession:
    """
    Manages a single clinical triage session.

    Mode A (common symptoms — from Intake Agent):
        session = TriageSession()
        diagnosis = session.diagnose_from_intake(intake_summary_dict)
        print(diagnosis["report"])

    Mode B (uncommon symptoms — conversational):
        session = TriageSession()
        first_question = session.start_uncommon("I have a rash on my legs")
        print(first_question)
        response = session.respond("It started 3 days ago and is itchy")
        ...
        if session.is_complete():
            print(session.get_diagnosis()["report"])
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o",
        retrieve_k: int = RERANK_TOP_K_RETRIEVE,
        return_k: int = RERANK_TOP_K_RETURN,
    ):
        print("\nInitialising Triage Agent models...")
        self._llm = ChatOpenAI(model=llm_model, temperature=0)
        self._device = detect_device()
        self._collection = open_collection(CHROMA_DIR)
        self._bi_encoder = load_bi_encoder(EMBEDDING_MODEL, self._device)
        self._reranker, self._reranker_device = load_cross_encoder(RERANKER_MODEL)
        self._retrieve_k = retrieve_k
        self._return_k = return_k

        self._graph = _build_graph(
            self._llm,
            self._collection,
            self._bi_encoder,
            self._reranker,
            self._retrieve_k,
            self._return_k,
        )

        self._state: TriageState = self._empty_state()
        self._phase: str = "idle"  # "idle" | "questioning" | "done"

        print(
            f"Triage Agent ready  "
            f"(bi-encoder on {self._device}, "
            f"reranker on {self._reranker_device})\n"
        )

    def _empty_state(self) -> TriageState:
        return {
            "messages": [],
            "mode": "uncommon",
            "intake_summary": None,
            "clinical_picture": None,
            "raw_complaint": "",
            "retrieved_chunks": [],
            "current_pass": 1,
            "generated_questions": [],
            "patient_answers": {},
            "current_question_idx": 0,
            "needs_refinement": False,
            "refinement_reason": "",
            "refinement_search_query": "",
            "info_sufficient": True,
            "followup_questions": [],
            "followup_answers": {},
            "followup_question_idx": 0,
            "followup_phase": False,
            "diagnosis": {},
            "diagnosis_complete": False,
        }

    def _last_ai_text(self) -> str:
        for msg in reversed(self._state["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content
        return ""

    def diagnose_from_intake(self, intake_summary: dict) -> dict | str:
        """
        Mode A: common symptoms. Runs RAG → sufficiency check → diagnosis.

        If intake answers are sufficient: produces diagnosis directly (non-interactive).
        If insufficient: returns the first follow-up question as a string.
        Caller should then use respond_followup() for each answer.

        Returns:
            dict: the diagnosis if completed in one pass
            str: the first follow-up question if more info needed
        """
        self._state = self._empty_state()
        self._state["intake_summary"] = intake_summary
        self._state["mode"] = "common"
        self._phase = "running"

        result = self._graph.invoke(
            {**self._state},
            {"recursion_limit": 20},
        )
        self._state.update(result)

        # Check if the graph paused for follow-up questions
        if not self._state.get("info_sufficient", True) and self._state.get("followup_questions"):
            self._phase = "followup"
            return self._last_ai_text()

        self._phase = "done"
        return self._state.get("diagnosis", {})

    def respond_followup(self, patient_answer: str) -> dict | str:
        """
        Mode A follow-up: process patient's answer to a follow-up question.

        Returns:
            str: next follow-up question if more remain
            dict: the diagnosis when all follow-ups answered and diagnosis generated
        """
        if self._phase == "done":
            return self._state.get("diagnosis", {})

        self._state["messages"].append(HumanMessage(content=patient_answer))

        # Process the follow-up answer
        process_fn = _build_process_followup_answer_node()
        updates = process_fn(self._state)
        self._state.update(updates)

        if self._state.get("followup_phase", False):
            # More follow-up questions remain
            ask_fn = _build_ask_followup_mode_a_node()
            ask_updates = ask_fn(self._state)
            self._state.update(ask_updates)
            return self._last_ai_text()

        # All follow-ups answered — re-retrieve with enriched context and diagnose
        print("\n[Triage] All follow-up questions answered. Re-retrieving and diagnosing...")

        # Enriched retrieval
        clinical_picture = self._state.get("clinical_picture") or {}
        followup_answers = self._state.get("followup_answers", {})
        base_query = _clinical_picture_to_query(clinical_picture)
        followup_context = " ".join(f"{q}: {a}" for q, a in followup_answers.items())
        enriched_query = f"{base_query} Additional context: {followup_context}"

        new_chunks = retrieve_and_rerank(
            enriched_query, self._collection, self._bi_encoder,
            self._reranker, self._retrieve_k, self._return_k
        )
        existing_chunks = self._state.get("retrieved_chunks", [])
        seen_ids = {c.get("chunk_id", "") for c in existing_chunks}
        all_chunks, _ = _deduplicate_chunks(existing_chunks, new_chunks, seen_ids)
        self._state["retrieved_chunks"] = all_chunks

        # Generate diagnosis with full context
        diagnosis_fn = _build_generate_diagnosis_node(self._llm)
        diag_updates = diagnosis_fn(self._state)
        self._state.update(diag_updates)
        self._state["diagnosis_complete"] = True
        self._phase = "done"

        return self._state.get("diagnosis", {})

    def start_uncommon(self, patient_complaint: str) -> str:
        """
        Mode B: start uncommon symptom triage.
        Returns the first clinical question.
        Caller must then use respond() for each answer.
        """
        self._state = self._empty_state()
        self._state["raw_complaint"] = patient_complaint
        self._state["mode"] = "uncommon"
        self._state["messages"].append(HumanMessage(content=patient_complaint))
        self._phase = "questioning"

        result = self._graph.invoke(
            {**self._state},
            {"recursion_limit": 20},
        )
        self._state.update(result)

        if self._state.get("diagnosis_complete"):
            self._phase = "done"
            return self._state.get("diagnosis", {}).get("report", "")

        return self._last_ai_text()

    def respond(self, patient_answer: str) -> str:
        """
        Mode B: process a patient's answer.
        Returns the next question, or the diagnosis report when complete.
        """
        if self._phase == "done":
            return "The triage session is complete. Please see the diagnosis above."

        self._state["messages"].append(HumanMessage(content=patient_answer))

        # Process the answer and decide next step
        process_answer_fn = _build_process_answer_node()
        updates = process_answer_fn(self._state)
        self._state.update(updates)

        questions = self._state.get("generated_questions", [])
        idx = self._state.get("current_question_idx", 0)

        if idx < len(questions):
            # More questions remain — ask the next one
            ask_fn = _build_ask_question_node()
            ask_updates = ask_fn(self._state)
            self._state.update(ask_updates)
            return self._last_ai_text()

        # All questions answered — run the remainder of the graph
        # (retrieve_evidence_pass2 → generate_diagnosis_pass2 → evaluate_refinement
        #  → optionally retrieve_evidence_pass3 → generate_diagnosis_pass3)
        print("[Triage] All questions answered. Running Pass 2 retrieval and diagnosis...")

        # We reinvoke the graph from a clean-slate style, but we need to resume
        # from the retrieve_evidence_pass2 node. LangGraph doesn't support
        # mid-graph resume without checkpointing, so we drive the steps manually.

        retrieve_fn = _build_retrieve_evidence_node(
            self._collection, self._bi_encoder, self._reranker,
            self._retrieve_k, self._return_k,
        )
        evaluate_fn = _build_evaluate_refinement_node(self._llm)
        diagnosis_fn_p2 = _build_generate_diagnosis_node(self._llm)
        diagnosis_fn_p3 = _build_generate_diagnosis_node(self._llm)

        # Pass 2 retrieval
        self._state["current_pass"] = 2
        retrieve_updates = retrieve_fn(self._state)
        self._state.update(retrieve_updates)

        # Pass 2 diagnosis
        diag_updates = diagnosis_fn_p2(self._state)
        self._state.update(diag_updates)

        # Hard cap check
        if self._state.get("current_pass", 2) >= 3:
            self._state["diagnosis_complete"] = True
            self._phase = "done"
            return self._state["diagnosis"]["report"]

        # Evaluate refinement
        eval_updates = evaluate_fn(self._state)
        self._state.update(eval_updates)

        if self._state.get("needs_refinement", False):
            print("[Triage] Running Pass 3 (targeted refinement)...")
            # Pass 3 retrieval
            self._state["current_pass"] = 3
            retrieve3_updates = retrieve_fn(self._state)
            self._state.update(retrieve3_updates)

            # Force final on pass 3
            self._state["needs_refinement"] = False
            diag3_updates = diagnosis_fn_p3(self._state)
            self._state.update(diag3_updates)
            self._state["diagnosis_complete"] = True

        self._phase = "done"
        return self._state["diagnosis"]["report"]

    def is_complete(self) -> bool:
        """True when the diagnosis has been generated."""
        return self._phase == "done"

    def get_diagnosis(self) -> dict:
        """Return the full diagnosis report dict."""
        return self._state.get("diagnosis", {})


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_diagnosis(diagnosis: dict) -> None:
    """Print the diagnosis report to stdout."""
    print("\n" + "=" * 70)
    print("  TRIAGE AGENT — DIAGNOSIS REPORT")
    print("=" * 70)
    mode = diagnosis.get("mode", "unknown")
    pass_num = diagnosis.get("pass", 0)
    num_chunks = diagnosis.get("num_chunks_used", 0)
    print(f"  Mode: {mode.upper()}  |  Pass: {pass_num}  |  Chunks used: {num_chunks}")
    if diagnosis.get("deferred"):
        print("\n  [EMERGENCY — DEFERRED TO EMERGENCY SERVICES]")
    print("=" * 70)
    print()
    print(diagnosis.get("report", "(no report generated)"))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Medora Phase 5 — Clinical Triage Agent (diagnostic engine)",
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
            "Runs Mode A (common symptoms) and exits."
        ),
    )
    parser.add_argument(
        "--query",
        default=None,
        metavar="COMPLAINT",
        help=(
            "Raw patient complaint for Mode B (uncommon symptoms). "
            "Runs interactively — the agent will ask follow-up questions."
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


def _run_mode_b_interactive(session: TriageSession, initial_complaint: str) -> None:
    """Run the Mode B interactive question-answer loop."""
    print(f"\n[Mode B] Starting uncommon symptom triage for: {initial_complaint!r}")
    first_question = session.start_uncommon(initial_complaint)

    if session.is_complete():
        # Diagnosis generated without questions (unlikely but safe)
        _print_diagnosis(session.get_diagnosis())
        return

    print(f"\nAgent: {first_question}\n")

    while not session.is_complete():
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession interrupted.\n")
            return

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye.\n")
            return

        if not user_input:
            continue

        response = session.respond(user_input)
        if session.is_complete():
            _print_diagnosis(session.get_diagnosis())
        else:
            print(f"\nAgent: {response}\n")


def main() -> None:
    args = _parse_args()

    session = TriageSession(
        llm_model=args.model,
        retrieve_k=args.retrieve_k,
        return_k=args.return_k,
    )

    # ── Mode 1: --from-intake ─────────────────────────────────────────────────
    if args.from_intake:
        intake_path = Path(args.from_intake)
        if not intake_path.exists():
            print(f"Error: intake file not found: {intake_path}", file=sys.stderr)
            sys.exit(1)

        with open(intake_path, encoding="utf-8") as fh:
            intake_summary = json.load(fh)

        print(f"\n[Mode A] Loading intake summary from: {intake_path}")
        diagnosis = session.diagnose_from_intake(intake_summary)
        _print_diagnosis(diagnosis)
        return

    # ── Mode 2: --query (Mode B, interactive) ────────────────────────────────
    if args.query:
        _run_mode_b_interactive(session, args.query)
        return

    # ── Mode 3: interactive — prompt for input, auto-detect mode ─────────────
    print("\nMedora Triage Agent")
    print("=" * 70)
    print("Enter your symptom or complaint to begin.")
    print("If this is from an intake summary, load it with --from-intake.")
    print("Type 'quit' to exit.")
    print("=" * 70 + "\n")

    try:
        complaint = input("Patient complaint: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.\n")
        return

    if not complaint or complaint.lower() in ("quit", "exit", "q"):
        print("\nGoodbye.\n")
        return

    _run_mode_b_interactive(session, complaint)


if __name__ == "__main__":
    main()
