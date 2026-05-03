"""
Triage session router — patient-facing endpoints for the agentic workflow.

Key safety guarantees:
  - Patient endpoints use PatientTriageSessionResponse (no urgency/diagnosis leak)
  - Diagnosis data is stored in ClinicalReport (doctor-only, visible_to_patient=False)
  - Emergency escalation messages ARE forwarded to patients (safety-critical)
  - SessionMessages with message_type='diagnosis' are excluded from patient message lists
"""

import asyncio
import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from sqlalchemy.orm import Session
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
from ..database import get_db
from ..auth.dependencies import get_current_patient_user
from ..models import User, PatientProfile, DoctorProfile, TriageSession, SessionMessage, ClinicalReport
from ..models.enums import AgentPhase
from ..schemas.triage import (
    TriageSessionCreate, PatientTriageSessionResponse,
    MessageCreate, MessageResponse, SessionPhaseResponse,
)
from ..config import settings
from ..services.session_manager import AgentSessionManager
from ..services.intake_agent_service import get_process_result
from ..services.report_service import generate_clinical_report

router = APIRouter(prefix="/triage", tags=["triage"])


def _get_manager() -> AgentSessionManager:
    return AgentSessionManager.get_instance()


@router.post("/sessions", response_model=PatientTriageSessionResponse)
def create_session(
    session_data: TriageSessionCreate,
    current_user: User = Depends(get_current_patient_user),
    db: Session = Depends(get_db),
):
    patient_profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="Patient profile not found")

    # Prefer the assigned doctor from the patient profile.
    doctor = None
    if patient_profile.assigned_doctor_id:
        doctor = db.query(DoctorProfile).filter(DoctorProfile.id == patient_profile.assigned_doctor_id).first()
    if not doctor:
        # Fallback to a demo doctor if no assigned doctor exists.
        doctor = db.query(DoctorProfile).first()
    if not doctor:
        raise HTTPException(status_code=400, detail="No doctors available")

    session = TriageSession(
        patient_id=patient_profile.id,
        doctor_id=doctor.id,
        hospital_id=patient_profile.hospital_id,
        chief_complaint=session_data.chief_complaint,
        status="active",
        urgency_level="unknown",
        escalation_type="none",
        chat_retention_policy="keep_full_history",
        agent_phase=AgentPhase.intake,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    # Create the in-memory agent session with patient context
    manager = _get_manager()
    patient_name = current_user.full_name
    patient_context = manager.get_patient_context(patient_name)
    manager.create_session(
        session_id=str(session.id),
        patient_context=patient_context,
        provider=settings.llm_provider,
        ollama_url=settings.ollama_url,
    )

    return session


@router.get("/sessions", response_model=List[PatientTriageSessionResponse])
def get_sessions(current_user: User = Depends(get_current_patient_user), db: Session = Depends(get_db)):
    patient_profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="Patient profile not found")

    sessions = (
        db.query(TriageSession)
        .filter(TriageSession.patient_id == patient_profile.id)
        .order_by(TriageSession.started_at.desc())
        .all()
    )
    return sessions


@router.get("/sessions/{session_id}", response_model=PatientTriageSessionResponse)
def get_session(session_id: str, current_user: User = Depends(get_current_patient_user), db: Session = Depends(get_db)):
    patient_profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="Patient profile not found")

    session = db.query(TriageSession).filter(
        TriageSession.id == session_id,
        TriageSession.patient_id == patient_profile.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/sessions/{session_id}/phase", response_model=SessionPhaseResponse)
def get_session_phase(session_id: str, current_user: User = Depends(get_current_patient_user), db: Session = Depends(get_db)):
    """Return the current agent phase for the session."""
    patient_profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="Patient profile not found")

    session = db.query(TriageSession).filter(
        TriageSession.id == session_id,
        TriageSession.patient_id == patient_profile.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionPhaseResponse(
        phase=session.agent_phase,
        is_escalated=(session.agent_phase == AgentPhase.escalated),
    )


@router.post("/sessions/{session_id}/message", response_model=MessageResponse)
async def send_message(
    session_id: str,
    message_data: MessageCreate,
    current_user: User = Depends(get_current_patient_user),
    db: Session = Depends(get_db),
):
    patient_profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="Patient profile not found")

    session = db.query(TriageSession).filter(
        TriageSession.id == session_id,
        TriageSession.patient_id == patient_profile.id,
        TriageSession.status == "active"
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Active session not found")

    # Save patient message
    patient_message = SessionMessage(
        session_id=session.id,
        sender="patient",
        content=message_data.content,
        message_type="text"
    )
    db.add(patient_message)
    db.commit()

    # Process through the agent pipeline (session_manager handles phase routing)
    result = await get_process_result(str(session.id), message_data.content)

    # Save agent response as a SessionMessage
    agent_message = SessionMessage(
        session_id=session.id,
        sender=result["sender"],
        content=result["response_text"],
        message_type=result["message_type"],
        is_persisted_after_summary=True,
        is_visible_to_doctor=True,
        is_deleted=False,
    )
    db.add(agent_message)

    # Update session phase in the database
    phase_mapping = {
        "intake": AgentPhase.intake,
        "triage_mode_a": AgentPhase.triage_mode_a,
        "triage_mode_b": AgentPhase.triage_mode_b,
        "escalated": AgentPhase.escalated,
        "completed": AgentPhase.completed,
    }
    new_phase = phase_mapping.get(result["phase"], AgentPhase.intake)
    session.agent_phase = new_phase

    # Store intake summary (doctor-only data) when intake completes
    if result.get("intake_summary"):
        session.intake_summary_json = result["intake_summary"]
        # Update detected symptoms and urgency on the session
        summary = result["intake_summary"]
        session.detected_symptoms = {"symptoms": summary.get("symptoms", [])}
        urgency = summary.get("urgency", "unknown")
        session.urgency_level = urgency

    # Handle escalation
    if result["phase"] == "escalated":
        session.escalation_type = "emergency_call"

    # Handle triage completion — generate the clinical report
    if result.get("triage_complete") and result.get("diagnosis"):
        _finalize_session(
            session, patient_profile, current_user, result, db
        )

    db.commit()
    db.refresh(agent_message)

    return agent_message


@router.get("/sessions/{session_id}/messages", response_model=List[MessageResponse])
def get_messages(session_id: str, current_user: User = Depends(get_current_patient_user), db: Session = Depends(get_db)):
    patient_profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="Patient profile not found")

    session = db.query(TriageSession).filter(
        TriageSession.id == session_id,
        TriageSession.patient_id == patient_profile.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Exclude diagnosis-type messages from patient view
    messages = db.query(SessionMessage).filter(
        SessionMessage.session_id == session.id,
        SessionMessage.message_type != "diagnosis",
    ).all()
    return messages


@router.post("/sessions/{session_id}/end")
def end_session(session_id: str, current_user: User = Depends(get_current_patient_user), db: Session = Depends(get_db)):
    patient_profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="Patient profile not found")

    session = db.query(TriageSession).filter(
        TriageSession.id == session_id,
        TriageSession.patient_id == patient_profile.id,
        TriageSession.status == "active"
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Active session not found")

    session.status = "completed"
    session.ended_at = datetime.now(timezone.utc)
    session.agent_phase = AgentPhase.completed

    # Get agent state for report generation
    manager = _get_manager()
    agent_state = manager.get_session(str(session.id))

    intake_summary = None
    diagnosis = None
    if agent_state:
        intake_summary = agent_state.intake_summary
        if agent_state.triage and agent_state.triage.is_complete():
            diagnosis = agent_state.triage.get_diagnosis()

    # Generate report (uses real agent data if available, placeholder otherwise)
    report = generate_clinical_report(
        str(session.id),
        str(patient_profile.id),
        str(session.doctor_id),
        intake_summary=intake_summary,
        diagnosis=diagnosis,
    )
    db_report = ClinicalReport(
        session_id=session.id,
        patient_id=patient_profile.id,
        doctor_id=session.doctor_id,
        presenting_complaints=report.presenting_complaints,
        history_of_presenting_complaint=report.history_of_presenting_complaint,
        summary_text=report.summary_text,
        suspected_conditions=report.suspected_conditions,
        triggered_red_flags=report.triggered_red_flags,
        urgency_level=report.urgency_level,
        recommended_action=report.recommended_action,
        specialty_routing=report.specialty_routing,
        suggested_workup=report.suggested_workup,
        key_exam_findings=report.key_exam_findings,
        admission_criteria=report.admission_criteria,
        referral_criteria=report.referral_criteria,
        external_escalation_completed=report.external_escalation_completed,
        escalation_message=report.escalation_message,
        visible_to_patient=False,
        model_version=report.model_version,
        diagnosis_mode=report.diagnosis_mode,
        diagnosis_pass_count=report.diagnosis_pass_count,
        chunks_used_count=report.chunks_used_count,
        web_search_results=report.web_search_results,
        generated_at=report.generated_at,
    )
    db.add(db_report)

    # Clean up the in-memory session
    manager.remove_session(str(session.id))

    db.commit()

    # Run memory/feedback updates in background — don't block the response
    if intake_summary:
        patient_name = current_user.full_name

        async def _background_updates():
            try:
                await asyncio.to_thread(
                    manager.update_patient_memory, patient_name, intake_summary, diagnosis
                )
            except Exception:
                logger.exception("Background: failed to update PatientMemory for %s", patient_name)
            try:
                if diagnosis:
                    await asyncio.to_thread(
                        manager.save_feedback_case, patient_name, intake_summary, diagnosis
                    )
            except Exception:
                logger.exception("Background: failed to save feedback case for %s", patient_name)

        asyncio.create_task(_background_updates())

    return {"message": "Session ended", "report_id": str(db_report.id)}


@router.get("/sessions/{session_id}/report")
def get_report(session_id: str, current_user: User = Depends(get_current_patient_user), db: Session = Depends(get_db)):
    """Clinical reports are doctor-only — patients cannot access them."""
    raise HTTPException(status_code=403, detail="Clinical reports are doctor-only")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _finalize_session(
    session: TriageSession,
    patient_profile: PatientProfile,
    current_user: User,
    result: dict,
    db: Session,
) -> None:
    """Generate clinical report and update memory when triage auto-completes."""
    session.status = "completed"
    session.ended_at = datetime.now(timezone.utc)
    session.agent_phase = AgentPhase.completed

    intake_summary = result.get("intake_summary")
    diagnosis = result.get("diagnosis")

    report = generate_clinical_report(
        str(session.id),
        str(patient_profile.id),
        str(session.doctor_id),
        intake_summary=intake_summary,
        diagnosis=diagnosis,
    )
    db_report = ClinicalReport(
        session_id=session.id,
        patient_id=patient_profile.id,
        doctor_id=session.doctor_id,
        presenting_complaints=report.presenting_complaints,
        history_of_presenting_complaint=report.history_of_presenting_complaint,
        summary_text=report.summary_text,
        suspected_conditions=report.suspected_conditions,
        triggered_red_flags=report.triggered_red_flags,
        urgency_level=report.urgency_level,
        recommended_action=report.recommended_action,
        specialty_routing=report.specialty_routing,
        suggested_workup=report.suggested_workup,
        key_exam_findings=report.key_exam_findings,
        admission_criteria=report.admission_criteria,
        referral_criteria=report.referral_criteria,
        external_escalation_completed=report.external_escalation_completed,
        escalation_message=report.escalation_message,
        visible_to_patient=False,
        model_version=report.model_version,
        diagnosis_mode=report.diagnosis_mode,
        diagnosis_pass_count=report.diagnosis_pass_count,
        chunks_used_count=report.chunks_used_count,
        generated_at=report.generated_at,
    )
    db.add(db_report)

    manager = _get_manager()
    manager.remove_session(str(session.id))

    # Run memory/feedback updates in background — don't block the response
    if intake_summary:
        patient_name = current_user.full_name

        async def _background_updates():
            try:
                await asyncio.to_thread(
                    manager.update_patient_memory, patient_name, intake_summary, diagnosis
                )
            except Exception:
                logger.exception("Background: failed to update PatientMemory for %s", patient_name)
            try:
                if diagnosis:
                    await asyncio.to_thread(
                        manager.save_feedback_case, patient_name, intake_summary, diagnosis
                    )
            except Exception:
                logger.exception("Background: failed to save feedback case for %s", patient_name)

        asyncio.create_task(_background_updates())