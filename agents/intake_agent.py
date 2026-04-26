"""
Phase 4.1 — Intake Agent (v2)
===========================
A LangGraph-powered clinical intake agent with four major improvements:

  Fix 1: Context-aware questions — passes ALL previous Q&A pairs when rephrasing
          the next question, enabling natural references to prior answers.
  Fix 2: Follow-up on vague answers — adds a clarity-check LLM call after each
          answer; asks ONE targeted follow-up if the answer is genuinely uninformative.
  Fix 3: Better red flag detection — clinically permissive prompt that maps
          everyday patient language to clinical concepts; checks ALL accumulated
          answers (not just the current one).
  Fix 4: Multi-symptom support — detects multiple symptoms, merges their question
          lists into one deduplicated flow, and pools red flags / urgency rules.
  Fix 5: Clinical depth probing — detects answers that are clear but clinically
          incomplete (e.g., "yes" to smoking) and asks for specific details.
  Fix 6: Pre-fill from initial message — extracts information already stated in
          the patient's opening message and skips/acknowledges those questions.

Usage (CLI):
    python agents/intake_agent.py
    python agents/intake_agent.py --model gpt-4o-mini
    python agents/intake_agent.py --symptom "Chest Pain"
    python agents/intake_agent.py --symptom "Chest Pain,Hemoptysis"
"""

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

# ── Project root on sys.path so config.py is importable ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import SYMPTOMS_DIR  # noqa: E402

# ── Environment ───────────────────────────────────────────────────────────────
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

# ── LangGraph / LangChain imports ─────────────────────────────────────────────
from typing import Annotated, TypedDict  # noqa: E402

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402

# ── Load all 11 symptom objects at module level ───────────────────────────────

_SYMPTOM_FILE = SYMPTOMS_DIR / "tmt_symptoms_gpt4o.json"

with open(_SYMPTOM_FILE, encoding="utf-8") as _fh:
    _ALL_SYMPTOMS: list[dict] = json.load(_fh)

# Map canonical name → full object (case-insensitive lookup by key)
_SYMPTOM_MAP: dict[str, dict] = {s["symptom"].lower(): s for s in _ALL_SYMPTOMS}
_SYMPTOM_NAMES: list[str] = [s["symptom"] for s in _ALL_SYMPTOMS]

# Urgency priority order (higher index = more severe)
_URGENCY_RANK = {"routine": 0, "urgent": 1, "emergency": 2}


# ─────────────────────────────────────────────────────────────────────────────
# State definition
# ─────────────────────────────────────────────────────────────────────────────

class IntakeState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    # Fix 4: multi-symptom fields (replace singular symptom_name / symptom_data)
    symptom_names: list[str]          # all detected canonical symptom names
    symptom_data_list: list[dict]     # all matching full symptom objects
    merged_questions: list[str]       # deduplicated, ordered question list
    all_red_flags: list[dict]         # pooled red flags from all symptoms
    all_urgency_rules: list[dict]     # pooled urgency rules from all symptoms

    current_question_idx: int         # index into merged_questions
    answers: dict                     # question text → patient answer text

    # Fix 2: follow-up tracking
    pending_followup: bool            # True when last answer was vague
    followup_question_idx: int | None # question idx we are following up on

    prefilled_answers: dict              # Fix 6: questions answered by the initial message
    triggered_red_flags: list[dict]   # red flag dicts that were matched
    urgency: str                      # "routine" | "urgent" | "emergency"
    escalated: bool                   # True if emergency escalation triggered
    intake_complete: bool             # True after all questions answered
    summary: str                      # final clinician handover note
    clarification_attempts: int       # number of failed symptom detection attempts
    uncommon_symptom: bool            # True if symptom not in the 11 common symptoms
    raw_complaint: str                # original patient complaint (for triage handoff)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_fence(text: str) -> str:
    """Remove markdown code fences from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        # parts[1] is the fenced block (may start with "json\n")
        inner = parts[1] if len(parts) > 1 else text
        if inner.startswith("json"):
            inner = inner[4:]
        return inner.strip()
    return text


def _format_previous_qa(answers: dict) -> str:
    """Format Q&A history as a readable block for prompt injection."""
    if not answers:
        return "  (none yet)"
    lines = []
    for q, a in answers.items():
        lines.append(f"  Q: {q}\n  A: {a}")
    return "\n".join(lines)


def _make_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=0)


# ─────────────────────────────────────────────────────────────────────────────
# Node builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_detect_symptom_node(llm: ChatOpenAI):
    """
    Detects one or more symptoms from the patient's message.
    Returns a JSON array of matched symptom names (Fix 4).
    """
    symptom_list_str = "\n".join(f"  - {name}" for name in _SYMPTOM_NAMES)

    system_prompt = f"""You are a clinical triage assistant. Your task is to identify which
medical symptom(s) the patient is describing from the following list of 11 symptoms:

{symptom_list_str}

Rules:
- Read the patient's message carefully.
- Return a JSON array of matched symptom names using the EXACT capitalisation from the list above.
- You may return more than one symptom if the patient clearly describes multiple.
- If the patient's message does not clearly match any symptom, return: ["NONE"]

Return a JSON array of matched symptom names. Examples:
- Patient says "I have chest pain": ["Chest Pain"]
- Patient says "chest pain and coughing blood": ["Chest Pain", "Hemoptysis"]
- Patient says "I feel tired and have a headache": ["Fatigue", "Acute Headache"]
If no symptoms match, return: ["NONE"]
Return ONLY the JSON array.
"""

    def detect_symptom(state: IntakeState) -> dict:
        human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not human_msgs:
            return {}

        patient_text = human_msgs[-1].content

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Patient says: {patient_text}"),
        ])

        raw = _strip_fence(response.content)
        try:
            detected_list = json.loads(raw)
            if not isinstance(detected_list, list):
                detected_list = ["NONE"]
        except (json.JSONDecodeError, ValueError):
            detected_list = ["NONE"]

        # Filter to only valid symptom names
        valid_detected = [
            name for name in detected_list
            if name.lower() in _SYMPTOM_MAP
        ]

        if not valid_detected:
            # Could not detect — ask for clarification
            attempts = state.get("clarification_attempts", 0) + 1
            if attempts >= 3:
                # Uncommon symptom — route to Triage Agent Mode B
                human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
                raw_complaint = human_msgs[0].content if human_msgs else patient_text
                handoff_msg = AIMessage(
                    content=(
                        "Your symptoms don't match the common conditions I'm equipped to assess "
                        "through structured questions. I'm transferring you to our diagnostic "
                        "system, which will analyze your case using our medical knowledge base "
                        "and ask you targeted questions."
                    )
                )
                return {
                    "messages": [handoff_msg],
                    "clarification_attempts": attempts,
                    "uncommon_symptom": True,
                    "raw_complaint": raw_complaint,
                    "intake_complete": True,
                }

            clarify_msg = AIMessage(
                content=(
                    "I want to make sure I understand your main concern. "
                    "Could you describe your primary symptom more specifically? "
                    "For instance, are you experiencing chest pain, difficulty breathing, "
                    "a cough, palpitations, or something else?"
                )
            )
            return {
                "messages": [clarify_msg],
                "clarification_attempts": attempts,
            }

        # Load all matching symptom objects
        symptom_objs = [_SYMPTOM_MAP[name.lower()] for name in valid_detected]

        # Build a friendly intro listing all detected symptoms
        if len(valid_detected) == 1:
            symptom_phrase = valid_detected[0].lower()
        else:
            symptom_phrase = (
                ", ".join(s.lower() for s in valid_detected[:-1])
                + f" and {valid_detected[-1].lower()}"
            )

        intro = AIMessage(
            content=(
                f"Thank you for sharing that. I understand you're experiencing {symptom_phrase}. "
                f"I'd like to ask you a few questions to better understand your situation "
                f"and help your care team. Please answer as best you can."
            )
        )

        # Pool all_red_flags and all_urgency_rules across detected symptoms
        pooled_red_flags: list[dict] = []
        seen_flags: set[str] = set()
        pooled_urgency_rules: list[dict] = []
        seen_rules: set[str] = set()
        for obj in symptom_objs:
            for rf in obj.get("red_flags", []):
                key = rf.get("flag", "")
                if key not in seen_flags:
                    pooled_red_flags.append(rf)
                    seen_flags.add(key)
            for ur in obj.get("urgency_rules", []):
                key = ur.get("criteria", "")
                if key not in seen_rules:
                    pooled_urgency_rules.append(ur)
                    seen_rules.add(key)

        return {
            "messages": [intro],
            "symptom_names": valid_detected,
            "symptom_data_list": symptom_objs,
            "all_red_flags": pooled_red_flags,
            "all_urgency_rules": pooled_urgency_rules,
            # merged_questions will be filled by merge_questions node
            "merged_questions": [],
            "current_question_idx": 0,
            "answers": {},
            "triggered_red_flags": [],
            "urgency": "routine",
            "escalated": False,
            "intake_complete": False,
            "summary": "",
            "clarification_attempts": 0,
            "pending_followup": False,
            "followup_question_idx": None,
        }

    return detect_symptom


def _build_merge_questions_node(llm: ChatOpenAI):
    """
    Fix 4: Merges essential_questions from all detected symptom objects into
    a single deduplicated, logically ordered list (5-8 questions).
    """
    system_prompt = """You are a clinical intake planner. Given essential questions from multiple symptom assessments,
create a unified, non-redundant list of intake questions that covers ALL symptoms efficiently.

Rules:
- Remove duplicate or near-duplicate questions (e.g., "duration of cough" and "duration of symptoms" → keep one)
- Order questions logically: general → specific, onset/duration → character → associated symptoms → history
- Keep all questions that are unique to a specific symptom
- Return a JSON array of question strings
- Aim for 5-8 questions total, even if combining from multiple symptoms
Return ONLY the JSON array. No explanation, no markdown."""

    def merge_questions(state: IntakeState) -> dict:
        symptom_data_list = state.get("symptom_data_list", [])
        symptom_names = state.get("symptom_names", [])

        # Collect all essential questions with their source symptom
        all_questions: list[str] = []
        for obj in symptom_data_list:
            all_questions.extend(obj.get("essential_questions", []))

        # If only one symptom, no merging needed
        if len(symptom_data_list) <= 1:
            return {"merged_questions": all_questions}

        # Multiple symptoms — ask LLM to deduplicate and order
        questions_json = json.dumps(all_questions, indent=2)
        symptom_phrase = ", ".join(symptom_names)

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Symptoms being assessed: {symptom_phrase}\n\n"
                    f"Essential questions collected from all symptom assessments:\n"
                    f"{questions_json}"
                )
            ),
        ])

        raw = _strip_fence(response.content)
        try:
            merged = json.loads(raw)
            if not isinstance(merged, list):
                merged = all_questions
        except (json.JSONDecodeError, ValueError):
            merged = all_questions

        return {"merged_questions": merged}

    return merge_questions


def _build_prefill_from_initial_message_node(llm: ChatOpenAI):
    """
    Fix 6: Reviews the patient's initial message against the merged question list
    and pre-fills answers for anything already stated. Pre-filled questions are
    skipped during the intake or acknowledged briefly.
    """
    system_prompt = """You are a clinical intake assistant reviewing a patient's initial message.
Given a list of intake questions and the patient's opening statement, determine which
questions (if any) have ALREADY been answered by what the patient said.

Rules:
- Only mark a question as answered if the patient's message clearly and specifically
  provides the information the question asks for.
- Extract the relevant answer text from the patient's message.
- Do NOT infer answers that are not explicitly stated.
- Return a JSON object where keys are the EXACT question strings and values are the
  extracted answers. Only include questions that were answered.
- If no questions were answered, return an empty object: {}
Return ONLY valid JSON. No explanation, no markdown."""

    def prefill_from_initial_message(state: IntakeState) -> dict:
        merged_questions = state.get("merged_questions", [])
        human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not human_msgs or not merged_questions:
            return {"prefilled_answers": {}}

        initial_message = human_msgs[0].content
        questions_json = json.dumps(merged_questions, indent=2)

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Patient's initial message: \"{initial_message}\"\n\n"
                    f"Intake questions:\n{questions_json}"
                )
            ),
        ])

        raw = _strip_fence(response.content)
        try:
            prefilled = json.loads(raw)
            if not isinstance(prefilled, dict):
                prefilled = {}
        except (json.JSONDecodeError, ValueError):
            prefilled = {}

        # Merge prefilled answers into the answers dict
        updated_answers = dict(state.get("answers", {}))
        updated_answers.update(prefilled)

        return {
            "prefilled_answers": prefilled,
            "answers": updated_answers,
        }

    return prefill_from_initial_message


def _build_ask_question_node(llm: ChatOpenAI):
    """
    Fix 1: Asks the next essential question conversationally, passing ALL previous
    Q&A pairs so the LLM can reference what the patient already said.
    """
    system_prompt = """You are a compassionate clinical intake assistant talking to a patient.
Your task is to ask the next clinical question in a conversational way.

Rules:
- Rephrase the clinical question into warm, clear, plain-English language.
- Reference relevant details from the patient's previous answers naturally.
  For example, if they already mentioned arm pain, you might say "You mentioned the pain spreads to your arms — are you also noticing any shortness of breath or nausea?"
- Keep the clinical intent exactly the same — do not change what is being asked.
- Keep the question concise (1-3 sentences maximum).
- Do not add disclaimers, preambles, or extra commentary.
- Output only the rephrased question."""

    def ask_question(state: IntakeState) -> dict:
        merged_questions = state.get("merged_questions", [])
        idx = state["current_question_idx"]
        prefilled = state.get("prefilled_answers", {})

        # Fix 6: Skip questions that were pre-filled from the initial message
        while idx < len(merged_questions) and merged_questions[idx] in prefilled:
            idx += 1

        if idx >= len(merged_questions):
            return {"intake_complete": True, "current_question_idx": idx}

        clinical_question = merged_questions[idx]
        symptom_names = state.get("symptom_names", [])
        symptom_phrase = ", ".join(symptom_names) if symptom_names else "unknown"
        answers = state.get("answers", {})
        previous_qa = _format_previous_qa(answers)

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Symptom being assessed: {symptom_phrase}\n"
                    f"Previous answers so far:\n{previous_qa}\n\n"
                    f"Next clinical question to ask: {clinical_question}"
                )
            ),
        ])

        question_msg = AIMessage(content=response.content.strip())
        return {"messages": [question_msg], "current_question_idx": idx}

    return ask_question


def _build_ask_followup_node(llm: ChatOpenAI):
    """
    Fix 2: Asks a targeted follow-up when the patient's previous answer was vague.
    Rephrases the same question more specifically, referencing what the patient said.
    """
    system_prompt = """You are a compassionate clinical intake assistant.
The patient gave an answer that needs more detail — either too vague or clinically incomplete.
Your task is to ask a targeted follow-up to get the specific information needed.

Rules:
- Acknowledge what the patient said so they feel heard.
- Ask for the specific missing information. Examples:
  - If they said "yes" to smoking: ask how long they smoked, how much, and when they quit
  - If they said "yes" to heart disease: ask what specific conditions they have
  - If they said "sometimes": ask how often specifically
  - If they said "arms": ask which arm, and whether it's constant or intermittent
- Keep it brief (1-2 sentences).
- Do not add disclaimers or preambles.
- Output only the follow-up question."""

    def ask_followup(state: IntakeState) -> dict:
        merged_questions = state.get("merged_questions", [])
        idx = state["current_question_idx"]

        if idx >= len(merged_questions):
            return {"intake_complete": True, "pending_followup": False}

        clinical_question = merged_questions[idx]
        answers = state.get("answers", {})
        # The vague answer is the most recent patient message
        human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        vague_answer = human_msgs[-1].content if human_msgs else ""

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Original clinical question: {clinical_question}\n"
                    f"Patient's vague answer: {vague_answer}"
                )
            ),
        ])

        followup_msg = AIMessage(content=response.content.strip())
        return {
            "messages": [followup_msg],
            "pending_followup": False,
            "followup_question_idx": idx,  # mark that we've already followed up on this idx
        }

    return ask_followup


def _build_process_answer_node(llm: ChatOpenAI):
    """
    Fix 2 + Fix 3:
    - Records the patient's answer.
    - Fix 3: Checks ALL accumulated answers against the pooled red flags using a
      clinically permissive prompt that maps everyday language to medical concepts.
    - Fix 2: After recording, checks if the answer is vague; if so and we haven't
      already followed up on this question, sets pending_followup=True.
    - Updates urgency. Increments question index.
    """
    # Fix 3: Improved red flag detection prompt
    red_flag_system = """You are a clinical red flag detection system.
Given a patient's answers to intake questions and a list of clinical red flags,
determine whether any red flags are CLINICALLY CONSISTENT with what the patient described.

Important: Patients use everyday language, not medical terms. You must interpret their
description clinically. For example:
- "The pain has been going on for hours" is consistent with "Prolonged chest pain episodes"
- "I feel a tearing sensation in my back" is consistent with "Chest pain with differential blood pressures" (aortic dissection)
- "My heart has been pounding non-stop" is consistent with "Sustained palpitations"

Return a JSON array of triggered red flag objects. Each triggered object should be copied
exactly from the red_flags list provided. If no red flags are indicated, return an empty array [].

Be clinically thoughtful — flag when the patient's description reasonably matches the
clinical concept, even if they don't use the exact medical terminology.
But do NOT over-flag — the patient saying "I have some chest pain" alone does not indicate
"Prolonged chest pain episodes" unless they describe duration or severity.
Return ONLY valid JSON. No explanation, no markdown."""

    # Fix 2 + Fix 5: Combined clarity and clinical depth check
    clarity_system = """You assess whether a patient's answer to a clinical question is adequate for clinical decision-making.
Given the question and answer, respond with ONLY a JSON object:
{"adequate": true/false, "issue": "vague" or "incomplete" or null, "reason": "one sentence explaining what is missing"}

Check for TWO types of problems:

1. VAGUE answers — genuinely uninformative responses like "sometimes", "not sure", "maybe", "I don't know"
   → {"adequate": false, "issue": "vague", "reason": "..."}

2. CLINICALLY INCOMPLETE answers — the answer is clear but missing critical details a clinician would need:
   - "yes" to smoking history → needs duration, quantity, when quit
   - "yes" to heart disease → needs what specific conditions
   - "yes" to medications → needs which medications
   - A single word like "arms" → needs more specificity (both arms? left arm?)

   → {"adequate": false, "issue": "incomplete", "reason": "..."}

An answer IS adequate if it provides enough detail for clinical assessment, even if not exhaustive.
Be practical — "sharp pain" is adequate for a pain quality question, but "yes" is NOT adequate
for "do you have risk factors for heart disease?"."""

    def process_answer(state: IntakeState) -> dict:
        human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not human_msgs:
            return {}

        patient_answer = human_msgs[-1].content
        merged_questions = state.get("merged_questions", [])
        idx = state["current_question_idx"]

        if idx >= len(merged_questions):
            return {"intake_complete": True}

        current_question = merged_questions[idx]
        all_red_flags = state.get("all_red_flags", [])
        answers = state.get("answers", {})

        # Fix 2: if we're in followup mode, accept the answer without clarity check
        already_followed_up = (
            state.get("followup_question_idx") == idx
        )

        # Record answer
        updated_answers = dict(answers)
        updated_answers[current_question] = patient_answer

        # Fix 3: Check ALL accumulated answers (not just the current one)
        new_triggered: list[dict] = []
        if all_red_flags:
            all_answers_text = "\n".join(
                f"  Q: {q}\n  A: {a}" for q, a in updated_answers.items()
            )
            symptom_phrase = ", ".join(state.get("symptom_names", []))

            rf_response = llm.invoke([
                SystemMessage(content=red_flag_system),
                HumanMessage(
                    content=(
                        f"Symptoms being assessed: {symptom_phrase}\n\n"
                        f"All patient answers so far:\n{all_answers_text}\n\n"
                        f"Red flags to check:\n{json.dumps(all_red_flags, indent=2)}"
                    )
                ),
            ])

            raw = _strip_fence(rf_response.content)
            try:
                new_triggered = json.loads(raw)
                if not isinstance(new_triggered, list):
                    new_triggered = []
            except (json.JSONDecodeError, ValueError):
                new_triggered = []

        # Merge with existing triggered flags (deduplicate by flag text)
        existing_flags = state.get("triggered_red_flags", [])
        existing_flag_texts = {f["flag"] for f in existing_flags}
        merged_flags = list(existing_flags)
        for flag in new_triggered:
            if flag.get("flag") not in existing_flag_texts:
                merged_flags.append(flag)
                existing_flag_texts.add(flag.get("flag"))

        # Determine new urgency (only escalate, never de-escalate)
        current_urgency = state.get("urgency", "routine")
        new_urgency = current_urgency
        for flag in new_triggered:
            flag_urgency = flag.get("urgency", "routine")
            if _URGENCY_RANK.get(flag_urgency, 0) > _URGENCY_RANK.get(new_urgency, 0):
                new_urgency = flag_urgency

        # Fix 2 + Fix 5: Adequacy check (vague OR clinically incomplete)
        pending_followup = False
        if not already_followed_up:
            clarity_response = llm.invoke([
                SystemMessage(content=clarity_system),
                HumanMessage(
                    content=(
                        f"Question: {current_question}\n"
                        f"Patient answer: {patient_answer}"
                    )
                ),
            ])
            raw_clarity = _strip_fence(clarity_response.content)
            try:
                clarity_parsed = json.loads(raw_clarity)
                is_adequate = clarity_parsed.get("adequate", True)
            except (json.JSONDecodeError, ValueError):
                is_adequate = True

            if not is_adequate:
                pending_followup = True

        # Determine next state
        next_idx = idx + 1 if not pending_followup else idx
        all_done = (next_idx >= len(merged_questions)) and not pending_followup

        return {
            "answers": updated_answers,
            "triggered_red_flags": merged_flags,
            "urgency": new_urgency,
            "current_question_idx": next_idx,
            "intake_complete": all_done,
            "pending_followup": pending_followup,
        }

    return process_answer


def _build_escalate_node(llm: ChatOpenAI):
    """Generates an emergency escalation message."""
    def escalate(state: IntakeState) -> dict:
        escalation_msg = AIMessage(
            content=(
                "IMPORTANT: Based on what you've described, your symptoms require immediate "
                "medical attention. Please call emergency services (911) or go to your nearest "
                "emergency department right away.\n\n"
                "Do not wait. Your safety is the priority. A clinical summary is being prepared "
                "for your care team."
            )
        )
        return {
            "messages": [escalation_msg],
            "escalated": True,
        }

    return escalate


def _build_assess_urgency_node(llm: ChatOpenAI):
    """
    Final urgency classification against ALL pooled urgency_rules.
    Fix 4: uses all_urgency_rules (pooled from all symptoms).
    """
    system_prompt = """You are a clinical urgency classifier.
Given:
- A patient's symptoms
- Their answers to intake questions
- The urgency rules for those symptoms

Evaluate which urgency level best fits the overall clinical picture.
Respond with a JSON object with exactly two keys:
  "urgency": one of "routine", "urgent", or "emergency"
  "rationale": a brief one-sentence clinical explanation

Return ONLY valid JSON. No explanation, no markdown."""

    def assess_urgency(state: IntakeState) -> dict:
        all_urgency_rules = state.get("all_urgency_rules", [])
        answers = state.get("answers", {})
        current_urgency = state.get("urgency", "routine")
        symptom_phrase = ", ".join(state.get("symptom_names", []))

        answers_text = "\n".join(
            f"  Q: {q}\n  A: {a}" for q, a in answers.items()
        )

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Symptoms: {symptom_phrase}\n\n"
                    f"Patient's answers:\n{answers_text}\n\n"
                    f"Urgency rules:\n{json.dumps(all_urgency_rules, indent=2)}\n\n"
                    f"Current urgency from red flags: {current_urgency}"
                )
            ),
        ])

        raw = _strip_fence(response.content)
        try:
            parsed = json.loads(raw)
            llm_urgency = parsed.get("urgency", current_urgency)
        except (json.JSONDecodeError, ValueError):
            llm_urgency = current_urgency

        # Final urgency: take the higher of flag-based and rule-based
        final_urgency = current_urgency
        if _URGENCY_RANK.get(llm_urgency, 0) > _URGENCY_RANK.get(final_urgency, 0):
            final_urgency = llm_urgency

        return {"urgency": final_urgency}

    return assess_urgency


def _build_generate_summary_node(llm: ChatOpenAI):
    """
    Fix 4: Produces the structured clinician handover note covering ALL detected
    symptoms, with deduplicated specialty routing and workup lists.
    """
    system_prompt = """You are a clinical documentation assistant.
Generate a concise, structured clinician handover note based on a patient intake session.

The note must include these sections in order, using plain markdown headers (##):
1. Presenting Complaint
2. History of Presenting Complaint (answers to each essential question)
3. Red Flags Identified (if any)
4. Urgency Classification & Recommended Action
5. Specialty Routing Recommendation
6. Suggested Initial Workup
7. Key Examination Findings to Elicit
8. Admission Criteria
9. Referral Criteria

Be precise and clinical. Use bullet points within sections. Omit empty sections gracefully.
"""

    def generate_summary(state: IntakeState) -> dict:
        answers = state.get("answers", {})
        triggered_flags = state.get("triggered_red_flags", [])
        urgency = state.get("urgency", "routine")
        escalated = state.get("escalated", False)
        symptom_names = state.get("symptom_names", [])
        symptom_data_list = state.get("symptom_data_list", [])
        all_urgency_rules = state.get("all_urgency_rules", [])

        # Determine urgency action from rules or defaults
        urgency_action = "Routine clinical assessment."
        if urgency == "emergency":
            urgency_action = "EMERGENCY — activate emergency response immediately."
        elif urgency == "urgent":
            for rule in all_urgency_rules:
                if rule.get("urgency") == "urgent":
                    urgency_action = rule.get("action", "Urgent evaluation required.")
                    break

        # Deduplicate multi-symptom lists
        def _dedup(items: list) -> list:
            seen = set()
            result = []
            for item in items:
                key = str(item)
                if key not in seen:
                    result.append(item)
                    seen.add(key)
            return result

        specialty_routing = _dedup(
            [r for obj in symptom_data_list for r in obj.get("specialty_routing", [])]
        )
        initial_workup = _dedup(
            [w for obj in symptom_data_list for w in obj.get("initial_workup", [])]
        )
        key_exam_findings = _dedup(
            [e for obj in symptom_data_list for e in obj.get("key_exam_findings", [])]
        )
        when_to_admit = _dedup(
            [a for obj in symptom_data_list for a in obj.get("when_to_admit", [])]
        )
        when_to_refer = _dedup(
            [r for obj in symptom_data_list for r in obj.get("when_to_refer", [])]
        )

        context = {
            "symptoms": symptom_names,
            "answers": [{"question": q, "answer": a} for q, a in answers.items()],
            "triggered_red_flags": triggered_flags,
            "urgency": urgency,
            "urgency_action": urgency_action,
            "escalated": escalated,
            "specialty_routing": specialty_routing,
            "initial_workup": initial_workup,
            "key_exam_findings": key_exam_findings,
            "when_to_admit": when_to_admit,
            "when_to_refer": when_to_refer,
        }

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Generate a clinician handover note for the following intake session:\n\n"
                    f"{json.dumps(context, indent=2)}"
                )
            ),
        ])

        summary_text = response.content.strip()

        summary_msg = AIMessage(
            content=(
                f"\n{'='*60}\n"
                f"CLINICIAN HANDOVER SUMMARY\n"
                f"{'='*60}\n"
                f"{summary_text}\n"
                f"{'='*60}"
            )
        )

        return {
            "messages": [summary_msg],
            "summary": summary_text,
            "intake_complete": True,
        }

    return generate_summary


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_graph(llm: ChatOpenAI) -> StateGraph:
    """Build and compile the LangGraph intake workflow."""

    detect_symptom_fn = _build_detect_symptom_node(llm)
    merge_questions_fn = _build_merge_questions_node(llm)
    prefill_fn = _build_prefill_from_initial_message_node(llm)
    ask_question_fn = _build_ask_question_node(llm)
    ask_followup_fn = _build_ask_followup_node(llm)
    process_answer_fn = _build_process_answer_node(llm)
    escalate_fn = _build_escalate_node(llm)
    assess_urgency_fn = _build_assess_urgency_node(llm)
    generate_summary_fn = _build_generate_summary_node(llm)

    # ── Routing functions ──────────────────────────────────────────────────────

    def route_after_detect(state: IntakeState) -> str:
        if state.get("symptom_names"):
            return "merge_questions"
        return "detect_symptom"

    def route_after_merge(state: IntakeState) -> str:
        return "ask_question"

    def route_after_process_answer(state: IntakeState) -> str:
        if state.get("urgency") == "emergency":
            return "escalate"
        if state.get("pending_followup"):
            return "ask_followup"
        if state.get("intake_complete"):
            return "assess_urgency"
        return "ask_question"

    # ── Graph assembly ─────────────────────────────────────────────────────────
    graph = StateGraph(IntakeState)

    graph.add_node("detect_symptom", detect_symptom_fn)
    graph.add_node("merge_questions", merge_questions_fn)
    graph.add_node("prefill", prefill_fn)
    graph.add_node("ask_question", ask_question_fn)
    graph.add_node("ask_followup", ask_followup_fn)
    graph.add_node("process_answer", process_answer_fn)
    graph.add_node("escalate", escalate_fn)
    graph.add_node("assess_urgency", assess_urgency_fn)
    graph.add_node("generate_summary", generate_summary_fn)

    graph.add_edge(START, "detect_symptom")

    graph.add_conditional_edges(
        "detect_symptom",
        route_after_detect,
        {"merge_questions": "merge_questions", "detect_symptom": "detect_symptom"},
    )

    # merge_questions → prefill → ask_question
    graph.add_edge("merge_questions", "prefill")
    graph.add_edge("prefill", "ask_question")

    # ask_question → END (pause; session manager resumes at process_answer)
    graph.add_edge("ask_question", END)

    # ask_followup → END (pause; session manager resumes at process_answer)
    graph.add_edge("ask_followup", END)

    graph.add_conditional_edges(
        "process_answer",
        route_after_process_answer,
        {
            "escalate": "escalate",
            "ask_followup": "ask_followup",
            "ask_question": "ask_question",
            "assess_urgency": "assess_urgency",
        },
    )

    graph.add_edge("escalate", "generate_summary")
    graph.add_edge("assess_urgency", "generate_summary")
    graph.add_edge("generate_summary", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# IntakeSession — manages a single patient conversation
# ─────────────────────────────────────────────────────────────────────────────

class IntakeSession:
    """
    Manages a single clinical intake conversation.

    Usage:
        session = IntakeSession()
        print(session.start("I have chest pain and I'm coughing blood"))
        print(session.respond("It started about 3 hours ago"))
        ...
        summary = session.get_summary()
    """

    def __init__(self, llm_model: str = "gpt-4o", skip_to_symptom: str | None = None):
        self._llm = _make_llm(llm_model)
        self._graph = _build_graph(self._llm)
        self._state: IntakeState = {
            "messages": [],
            "symptom_names": [],
            "symptom_data_list": [],
            "merged_questions": [],
            "all_red_flags": [],
            "all_urgency_rules": [],
            "current_question_idx": 0,
            "answers": {},
            "pending_followup": False,
            "followup_question_idx": None,
            "prefilled_answers": {},
            "triggered_red_flags": [],
            "urgency": "routine",
            "escalated": False,
            "intake_complete": False,
            "summary": "",
            "clarification_attempts": 0,
            "uncommon_symptom": False,
            "raw_complaint": "",
        }
        self._phase: str = "detect"  # "detect" | "questioning" | "done" | "uncommon"

        # Pre-specified symptom(s) for testing — support comma-separated list
        if skip_to_symptom:
            names_raw = [n.strip() for n in skip_to_symptom.split(",")]
            valid_names = [n for n in names_raw if n.lower() in _SYMPTOM_MAP]
            if valid_names:
                symptom_objs = [_SYMPTOM_MAP[n.lower()] for n in valid_names]

                pooled_red_flags: list[dict] = []
                seen_flags: set[str] = set()
                pooled_urgency_rules: list[dict] = []
                seen_rules: set[str] = set()
                for obj in symptom_objs:
                    for rf in obj.get("red_flags", []):
                        key = rf.get("flag", "")
                        if key not in seen_flags:
                            pooled_red_flags.append(rf)
                            seen_flags.add(key)
                    for ur in obj.get("urgency_rules", []):
                        key = ur.get("criteria", "")
                        if key not in seen_rules:
                            pooled_urgency_rules.append(ur)
                            seen_rules.add(key)

                # Merge questions eagerly (using simple concatenation for single
                # symptom, or flattening all for multi — the merge node will run
                # during start() via graph invocation when questioning phase begins)
                all_qs: list[str] = []
                for obj in symptom_objs:
                    all_qs.extend(obj.get("essential_questions", []))

                self._state.update({
                    "symptom_names": valid_names,
                    "symptom_data_list": symptom_objs,
                    "all_red_flags": pooled_red_flags,
                    "all_urgency_rules": pooled_urgency_rules,
                    "merged_questions": all_qs,  # will be refined by merge node
                    "current_question_idx": 0,
                    "urgency": "routine",
                })
                self._phase = "questioning"

    def _last_ai_text(self) -> str:
        """Extract the last AI message text from state."""
        for msg in reversed(self._state["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content
        return ""

    def start(self, patient_message: str) -> str:
        """
        Begin the intake session with the patient's first message.
        Returns the agent's response.
        """
        if self._phase == "questioning":
            # Symptom pre-specified — run merge → prefill → ask first question
            self._state["messages"].append(HumanMessage(content=patient_message))

            # Run merge_questions to deduplicate if multiple symptoms
            merge_fn = _build_merge_questions_node(self._llm)
            merge_updates = merge_fn(self._state)
            self._state.update(merge_updates)

            # Fix 6: Pre-fill answers from initial message
            prefill_fn = _build_prefill_from_initial_message_node(self._llm)
            prefill_updates = prefill_fn(self._state)
            self._state.update(prefill_updates)

            # Ask first non-prefilled question
            ask_fn = _build_ask_question_node(self._llm)
            ask_updates = ask_fn(self._state)
            self._state.update(ask_updates)
            return self._last_ai_text()

        # Normal flow: detect symptom → merge questions → ask first question
        self._state["messages"].append(HumanMessage(content=patient_message))
        result = self._graph.invoke(
            {**self._state, "messages": self._state["messages"]},
            {"recursion_limit": 15},
        )
        self._state.update(result)

        if self._state.get("uncommon_symptom"):
            self._phase = "uncommon"
        elif self._state.get("symptom_names"):
            self._phase = "questioning"
        # else remains "detect" for clarification loop

        return self._last_ai_text()

    def respond(self, patient_message: str) -> str:
        """
        Process a patient's response to a question.
        Returns the next question, follow-up, urgency message, or final summary.
        """
        if self._phase == "done":
            return "The intake session is complete. Please see the summary above."

        self._state["messages"].append(HumanMessage(content=patient_message))

        if self._phase == "detect":
            result = self._graph.invoke(
                {**self._state, "messages": self._state["messages"]},
                {"recursion_limit": 15},
            )
            self._state.update(result)
            if self._state.get("uncommon_symptom"):
                self._phase = "uncommon"
            elif self._state.get("symptom_names"):
                self._phase = "questioning"
            return self._last_ai_text()

        if self._phase == "uncommon":
            return "The intake session has been routed to the diagnostic system. Please see above."

        # Phase: questioning
        process_answer_fn = _build_process_answer_node(self._llm)
        ask_question_fn = _build_ask_question_node(self._llm)
        ask_followup_fn = _build_ask_followup_node(self._llm)
        escalate_fn = _build_escalate_node(self._llm)
        assess_urgency_fn = _build_assess_urgency_node(self._llm)
        generate_summary_fn = _build_generate_summary_node(self._llm)

        # Step 1: process the answer (includes red flag check + clarity check)
        updates = process_answer_fn(self._state)
        self._state.update(updates)

        urgency = self._state.get("urgency", "routine")
        intake_complete = self._state.get("intake_complete", False)
        pending_followup = self._state.get("pending_followup", False)

        # Step 2: route based on outcome
        if urgency == "emergency":
            esc_updates = escalate_fn(self._state)
            self._state.update(esc_updates)
            sum_updates = generate_summary_fn(self._state)
            self._state.update(sum_updates)
            self._phase = "done"
            return self._last_ai_text()

        if pending_followup:
            followup_updates = ask_followup_fn(self._state)
            self._state.update(followup_updates)
            return self._last_ai_text()

        if intake_complete:
            urg_updates = assess_urgency_fn(self._state)
            self._state.update(urg_updates)
            sum_updates = generate_summary_fn(self._state)
            self._state.update(sum_updates)
            self._phase = "done"
            return self._last_ai_text()

        # More questions remain
        ask_updates = ask_question_fn(self._state)
        self._state.update(ask_updates)
        return self._last_ai_text()

    def is_complete(self) -> bool:
        """True when the intake has concluded (summary available or routed to triage)."""
        return self._phase in ("done", "uncommon")

    def is_uncommon(self) -> bool:
        """True when the symptom was not recognized and should be routed to Triage Agent Mode B."""
        return self._phase == "uncommon"

    def get_raw_complaint(self) -> str:
        """Return the raw patient complaint for Triage Agent Mode B handoff."""
        return self._state.get("raw_complaint", "")

    def get_summary(self) -> dict:
        """
        Return the structured summary dict when intake is complete.
        Returns an empty dict if the intake is still in progress.
        """
        if not self.is_complete():
            return {}

        symptom_data_list = self._state.get("symptom_data_list", [])

        def _dedup(items: list) -> list:
            seen = set()
            result = []
            for item in items:
                key = str(item)
                if key not in seen:
                    result.append(item)
                    seen.add(key)
            return result

        return {
            "symptoms": self._state.get("symptom_names", []),
            "urgency": self._state.get("urgency", "routine"),
            "escalated": self._state.get("escalated", False),
            "answers": self._state.get("answers", {}),
            "triggered_red_flags": self._state.get("triggered_red_flags", []),
            "specialty_routing": _dedup(
                [r for obj in symptom_data_list for r in obj.get("specialty_routing", [])]
            ),
            "initial_workup": _dedup(
                [w for obj in symptom_data_list for w in obj.get("initial_workup", [])]
            ),
            "key_exam_findings": _dedup(
                [e for obj in symptom_data_list for e in obj.get("key_exam_findings", [])]
            ),
            "when_to_admit": _dedup(
                [a for obj in symptom_data_list for a in obj.get("when_to_admit", [])]
            ),
            "when_to_refer": _dedup(
                [r for obj in symptom_data_list for r in obj.get("when_to_refer", [])]
            ),
            "clinician_note": self._state.get("summary", ""),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Medora Phase 5 — Clinical Intake Agent (interactive CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        metavar="MODEL",
        help="OpenAI model name to use (e.g. gpt-4o, gpt-4o-mini).",
    )
    parser.add_argument(
        "--symptom",
        default=None,
        metavar="NAME",
        help=(
            "Skip symptom detection and start directly with named symptom(s). "
            "Supports comma-separated values for multiple symptoms. "
            f"Valid values: {', '.join(_SYMPTOM_NAMES)}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print("\nMedora Intake Agent")
    print("=" * 50)
    print("Type your symptoms to begin. Type 'quit' to exit.")
    if args.symptom:
        print(f"[Testing mode: symptom pre-set to '{args.symptom}']")
    print("=" * 50 + "\n")

    session = IntakeSession(llm_model=args.model, skip_to_symptom=args.symptom)

    # ── First turn ────────────────────────────────────────────────────────────
    first_input = input("You: ").strip()
    if first_input.lower() in ("quit", "exit", "q"):
        print("\nGoodbye.\n")
        return
    response = session.start(first_input)
    print(f"\nAgent: {response}\n")

    # ── Subsequent turns ──────────────────────────────────────────────────────
    while not session.is_complete():
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession interrupted.\n")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye.\n")
            break

        if not user_input:
            continue

        response = session.respond(user_input)
        print(f"\nAgent: {response}\n")

    # ── Post-session: route to Triage Agent ─────────────────────────────────
    if not session.is_complete():
        return

    if session.is_uncommon():
        # Uncommon symptom → Triage Agent Mode B (interactive)
        print("\n" + "=" * 50)
        print("  Routing to Triage Agent (uncommon symptom)...")
        print("=" * 50 + "\n")

        from agents.triage_agent import TriageSession as TriageSessionCls
        triage = TriageSessionCls(llm_model=args.model)
        raw_complaint = session.get_raw_complaint()
        triage_response = triage.start_uncommon(raw_complaint)
        print(f"Agent: {triage_response}\n")

        while not triage.is_complete():
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nSession interrupted.\n")
                break
            if user_input.lower() in ("quit", "exit", "q"):
                print("\nGoodbye.\n")
                break
            if not user_input:
                continue
            triage_response = triage.respond(user_input)
            print(f"\nAgent: {triage_response}\n")

    else:
        # Common symptom → show intake summary, then Triage Agent Mode A
        summary = session.get_summary()
        print("\n" + "=" * 50)
        print("Intake complete.")
        symptoms = summary.get("symptoms", [])
        print(f"  Symptoms  : {', '.join(symptoms) if symptoms else 'N/A'}")
        print(f"  Urgency   : {summary.get('urgency', 'N/A').upper()}")
        print(f"  Escalated : {summary.get('escalated', False)}")
        red_flags = summary.get("triggered_red_flags", [])
        if red_flags:
            print(f"  Red Flags : {len(red_flags)} triggered")
            for rf in red_flags:
                print(f"    - {rf.get('flag', '')} [{rf.get('urgency', '')}]")
        print("=" * 50)

        if summary.get("escalated"):
            print("\n  EMERGENCY ESCALATION — Patient directed to call 911.")
            print("  Triage Agent not invoked (escalated mid-conversation).\n")
        else:
            print("\n  Routing to Triage Agent for diagnosis...\n")
            from agents.triage_agent import TriageSession as TriageSessionCls
            triage = TriageSessionCls(llm_model=args.model)
            result = triage.diagnose_from_intake(summary)

            if isinstance(result, str):
                # Follow-up questions needed
                print(f"\nAgent: {result}\n")
                while not triage.is_complete():
                    try:
                        user_input = input("You: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\n\nSession interrupted.\n")
                        break
                    if user_input.lower() in ("quit", "exit", "q"):
                        print("\nGoodbye.\n")
                        break
                    if not user_input:
                        continue
                    followup_result = triage.respond_followup(user_input)
                    if isinstance(followup_result, str):
                        print(f"\nAgent: {followup_result}\n")
                    else:
                        _print_diagnosis(followup_result)
            elif isinstance(result, dict):
                if result.get("deferred"):
                    print(f"\n  {result.get('report', 'Emergency case deferred.')}\n")
                else:
                    _print_diagnosis(result)
            else:
                print("\n  Triage Agent returned no result.\n")


if __name__ == "__main__":
    main()
