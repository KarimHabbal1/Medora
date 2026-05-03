"""
Clinical report generation service — builds a ClinicalReportResponse
from the real TriageSession diagnosis output.

Replaces the previous mock implementation.
Agent logic is NOT modified; this module only parses agent output.
"""

import re
from datetime import datetime, timezone
from uuid import uuid4, UUID
from typing import Dict, Any, Optional
from ..schemas.triage import ClinicalReportResponse
from ..schemas.enums import UrgencyLevel


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

    return ClinicalReportResponse(
        id=uuid4(),
        session_id=UUID(session_id),
        patient_id=UUID(patient_id),
        doctor_id=UUID(doctor_id),
        presenting_complaints=presenting_complaints,
        history_of_presenting_complaint={"reasoning": history} if history else {},
        summary_text=report_text[:2000] if report_text else "Assessment completed.",
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
        generated_at=datetime.now(timezone.utc),
    )