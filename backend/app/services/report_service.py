"""
Clinical report generation service — builds a ClinicalReportResponse
from the real TriageSession diagnosis output.

Replaces the previous mock implementation.
Agent logic is NOT modified; this module only parses agent output.
"""

import logging
import os
import re
import sys
from datetime import datetime, timezone
from uuid import uuid4, UUID
from typing import Dict, Any, Optional
from ..schemas.triage import ClinicalReportResponse
from ..schemas.enums import UrgencyLevel

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

logger = logging.getLogger(__name__)


def _extract_section(report_text: str, section_name: str) -> Optional[str]:
    """Extract a named section from the agent's markdown-formatted report."""
    pattern = rf"(?:#+\s*)?{re.escape(section_name)}\s*\n(.*?)(?=\n#+\s|\Z)"
    match = re.search(pattern, report_text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def _parse_urgency(urgency_str: str) -> UrgencyLevel:
    """Map agent urgency string to UrgencyLevel enum."""
    mapping = {
        "routine": UrgencyLevel.routine,
        "low": UrgencyLevel.routine,
        "medium": UrgencyLevel.urgent,
        "urgent": UrgencyLevel.urgent,
        "high": UrgencyLevel.emergency,
        "emergency": UrgencyLevel.emergency,
        "critical": UrgencyLevel.emergency,
    }
    return mapping.get(urgency_str.lower().strip(), UrgencyLevel.unknown)


_STRUCTURE_SYMPTOMS_PROMPT = """\
You are a clinical data structurer. Given a patient conversation history from a medical \
triage system, extract and structure the symptoms into a concise clinical summary \
optimized for a diagnostic web search.

Rules:
- Extract all symptoms, their characteristics, and relevant history
- Include: onset, duration, character, location, severity, aggravating/relieving factors, associated symptoms
- Use clinical terminology where appropriate but keep it searchable
- Do NOT include any diagnosis or clinical reasoning — only patient-reported information
- Be concise — one paragraph, no bullet points

Return ONLY the structured symptom summary as plain text. No JSON, no markdown.\
"""


def _structure_symptoms_via_llm(intake_summary: Dict[str, Any], llm) -> str:
    """Use LLM to structure patient conversation data into a clean symptom summary."""
    from langchain_core.messages import SystemMessage, HumanMessage

    parts = []

    # Intake data (Mode A — common symptoms)
    symptoms = intake_summary.get("symptoms", [])
    if symptoms:
        parts.append(f"Presenting symptoms: {', '.join(symptoms)}")

    answers = intake_summary.get("answers", {})
    if answers:
        parts.append("Patient interview responses:")
        for q, a in answers.items():
            parts.append(f"  Q: {q}")
            parts.append(f"  A: {a}")

    red_flags = intake_summary.get("red_flags", [])
    if red_flags:
        flags = [f.get("flag", str(f)) if isinstance(f, dict) else str(f) for f in red_flags]
        parts.append(f"Red flags identified: {', '.join(flags)}")

    urgency = intake_summary.get("urgency", "")
    if urgency and urgency != "unknown":
        parts.append(f"Urgency: {urgency}")

    # Raw complaint (Mode B — uncommon symptoms)
    raw_complaint = intake_summary.get("raw_complaint", "")
    if raw_complaint:
        parts.append(f"Patient complaint: {raw_complaint}")

    # Triage Q&A (Mode B — collected during Pass 1/2/3)
    triage_answers = intake_summary.get("triage_answers", {})
    if triage_answers:
        parts.append("Triage follow-up responses:")
        for q, a in triage_answers.items():
            parts.append(f"  Q: {q}")
            parts.append(f"  A: {a}")

    if not parts:
        return ""

    conversation_text = "\n".join(parts)

    response = llm.invoke([
        SystemMessage(content=_STRUCTURE_SYMPTOMS_PROMPT),
        HumanMessage(content=conversation_text),
    ])

    return response.content.strip()


def _run_web_search(intake_summary: Dict[str, Any], diagnosis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Structure symptoms via LLM and run the web search agent."""
    try:
        from agents.web_search import search_medical_evidence, make_llm as ws_make_llm
        from config import make_llm as config_make_llm

        searxng_url = os.getenv("SEARXNG_BASE_URL", "").rstrip("/")
        print(f"[WebSearch] SEARXNG_BASE_URL = {searxng_url!r}", flush=True)
        if not searxng_url:
            print("[WebSearch] Skipped — no SEARXNG_BASE_URL configured", flush=True)
            return None

        # Use LLM to structure the symptoms from conversation data
        structuring_llm = config_make_llm()
        structured_symptoms = _structure_symptoms_via_llm(intake_summary, structuring_llm)
        print(f"[WebSearch] Structured symptoms: {structured_symptoms[:150]}", flush=True)

        if not structured_symptoms.strip():
            print("[WebSearch] Skipped — no symptoms to structure", flush=True)
            return None

        provider = os.getenv("MEDORA_LLM_PROVIDER", "openai")
        model = "gpt-4o-mini" if provider == "openai" else "llama3.1:8b"
        ws_llm = ws_make_llm(model=model, provider=provider)

        result = search_medical_evidence(
            symptoms=structured_symptoms,
            llm=ws_llm,
            searxng_url=searxng_url,
            max_sources=5,
        )

        return {
            "primary_diagnosis": result.get("primary_diagnosis", ""),
            "confidence": result.get("confidence", ""),
            "evidence_summary": result.get("evidence_summary", ""),
            "key_findings": result.get("key_findings", []),
            "differential_diagnoses": result.get("differential_diagnoses", []),
            "sources": [
                {"title": s.get("title", ""), "url": s.get("url", ""), "domain": s.get("domain", "")}
                for s in result.get("sources", [])
            ],
        }
    except Exception as exc:
        print(f"[WebSearch] FAILED: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        return None


def generate_clinical_report(
    session_id: str,
    patient_id: str,
    doctor_id: str,
    intake_summary: Optional[Dict[str, Any]] = None,
    diagnosis: Optional[Dict[str, Any]] = None,
) -> ClinicalReportResponse:
    """
    Build a ClinicalReportResponse from agent outputs.

    If intake_summary and diagnosis are provided, parses the real agent data.
    Otherwise falls back to a placeholder report (for backward compatibility).
    """
    if diagnosis and intake_summary:
        return _build_from_agent_output(session_id, patient_id, doctor_id, intake_summary, diagnosis)
    else:
        return _build_placeholder(session_id, patient_id, doctor_id)


def _build_from_agent_output(
    session_id: str,
    patient_id: str,
    doctor_id: str,
    intake_summary: Dict[str, Any],
    diagnosis: Dict[str, Any],
) -> ClinicalReportResponse:
    """Parse real agent output into the ClinicalReport schema."""
    report_text = diagnosis.get("report", "")
    mode = diagnosis.get("mode", "unknown")
    pass_count = diagnosis.get("pass", 1)
    num_chunks = diagnosis.get("num_chunks_used", 0)

    # Extract urgency from intake summary
    urgency_str = intake_summary.get("urgency", "unknown")
    urgency = _parse_urgency(urgency_str)

    # Extract symptoms
    symptoms = intake_summary.get("symptoms", [])
    if isinstance(symptoms, list):
        presenting_complaints = {"symptoms": symptoms}
    else:
        presenting_complaints = {"symptoms": [str(symptoms)]}

    # Parse sections from the diagnosis report text
    history = _extract_section(report_text, "Clinical Reasoning")
    # Strip Clinical Reasoning from the summary since it's shown separately
    summary_for_display = re.sub(
        r'(?:#+\s*)?Clinical Reasoning\s*\n.*?(?=\n#+\s|\Z)',
        '', report_text, flags=re.IGNORECASE | re.DOTALL
    ).strip()
    suspected = _extract_section(report_text, "Primary Diagnosis")
    red_flags_text = _extract_section(report_text, "Red Flags")
    recommended = _extract_section(report_text, "Management Considerations") or _extract_section(report_text, "Recommended")
    workup = _extract_section(report_text, "Recommended Investigations") or _extract_section(report_text, "Investigations")
    routing = _extract_section(report_text, "Specialist Referral") or _extract_section(report_text, "Routing")

    # Build red flags from intake summary if available
    red_flags = intake_summary.get("red_flags", [])
    triggered_red_flags = {"flags": red_flags} if red_flags else {}

    # Check for escalation
    escalated = intake_summary.get("escalated", False)
    escalation_msg = intake_summary.get("escalation_message") if escalated else None

    # Run web search for external evidence
    print(f"[WebSearch] Starting web search. intake keys: {list(intake_summary.keys())}", flush=True)
    web_results = _run_web_search(intake_summary, diagnosis)
    print(f"[WebSearch] Result: {'found' if web_results else 'None'}", flush=True)

    return ClinicalReportResponse(
        id=uuid4(),
        session_id=UUID(session_id),
        patient_id=UUID(patient_id),
        doctor_id=UUID(doctor_id),
        presenting_complaints=presenting_complaints,
        history_of_presenting_complaint={"reasoning": history} if history else {},
        summary_text=summary_for_display[:2000] if summary_for_display else "Assessment completed.",
        suspected_conditions={"primary": suspected} if suspected else {},
        triggered_red_flags=triggered_red_flags,
        urgency_level=urgency,
        recommended_action=recommended or "Follow up with your healthcare provider.",
        specialty_routing={"routing": routing} if routing else {},
        suggested_workup={"workup": workup} if workup else {},
        key_exam_findings={},
        admission_criteria={},
        referral_criteria={},
        external_escalation_completed=escalated,
        escalation_message=escalation_msg,
        visible_to_patient=False,
        model_version=f"{mode}_pass{pass_count}",
        diagnosis_mode=mode,
        diagnosis_pass_count=pass_count,
        chunks_used_count=num_chunks,
        web_search_results=web_results,
        generated_at=datetime.now(timezone.utc),
    )


def _build_placeholder(
    session_id: str,
    patient_id: str,
    doctor_id: str,
) -> ClinicalReportResponse:
    """Backward-compatible placeholder when agent output is not available."""
    return ClinicalReportResponse(
        id=uuid4(),
        session_id=UUID(session_id),
        patient_id=UUID(patient_id),
        doctor_id=UUID(doctor_id),
        presenting_complaints={"complaint": "Pending agent integration"},
        history_of_presenting_complaint={},
        summary_text="Assessment completed. Agent-generated report pending.",
        suspected_conditions={},
        triggered_red_flags={},
        urgency_level=UrgencyLevel.unknown,
        recommended_action="Follow up with your healthcare provider.",
        specialty_routing={},
        suggested_workup={},
        key_exam_findings={},
        admission_criteria={},
        referral_criteria={},
        external_escalation_completed=False,
        escalation_message=None,
        visible_to_patient=False,
        model_version="1.0",
        diagnosis_mode=None,
        diagnosis_pass_count=None,
        chunks_used_count=None,
        web_search_results=None,
        generated_at=datetime.now(timezone.utc),
    )