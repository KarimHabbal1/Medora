from fastapi import APIRouter, Depends, HTTPException
from typing import List
from sqlalchemy.orm import Session
from datetime import datetime
from ..database import get_db
from ..auth.dependencies import get_current_patient_user
from ..models import User, PatientProfile, DoctorProfile, TriageSession, SessionMessage, ClinicalReport
from ..schemas.triage import TriageSessionCreate, TriageSessionResponse, MessageCreate, MessageResponse, ClinicalReportResponse
from ..services.intake_agent_service import process_patient_message
from ..services.report_service import generate_clinical_report

router = APIRouter(prefix="/triage", tags=["triage"])


@router.post("/sessions", response_model=TriageSessionResponse)
def create_session(session_data: TriageSessionCreate, current_user: User = Depends(get_current_patient_user), db: Session = Depends(get_db)):
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
        chat_retention_policy="keep_full_history"
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/sessions", response_model=List[TriageSessionResponse])
def get_sessions(current_user: User = Depends(get_current_patient_user), db: Session = Depends(get_db)):
    patient_profile = db.query(PatientProfile).filter(PatientProfile.user_id == current_user.id).first()
    if not patient_profile:
        raise HTTPException(status_code=404, detail="Patient profile not found")
    
    sessions = db.query(TriageSession).filter(TriageSession.patient_id == patient_profile.id).all()
    return sessions


@router.get("/sessions/{session_id}", response_model=TriageSessionResponse)
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


@router.post("/sessions/{session_id}/message", response_model=MessageResponse)
def send_message(session_id: str, message_data: MessageCreate, current_user: User = Depends(get_current_patient_user), db: Session = Depends(get_db)):
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
    
    # Get agent response
    agent_response = process_patient_message(str(session.id), message_data.content)
    agent_message = SessionMessage(
        session_id=session.id,
        sender=agent_response.sender,
        content=agent_response.content,
        message_type=agent_response.message_type,
        is_persisted_after_summary=agent_response.is_persisted_after_summary,
        is_visible_to_doctor=agent_response.is_visible_to_doctor,
        is_deleted=agent_response.is_deleted
    )
    db.add(agent_message)
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
    
    messages = db.query(SessionMessage).filter(SessionMessage.session_id == session.id).all()
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
    session.ended_at = datetime.utcnow()
    db.commit()
    
    # Generate report
    report = generate_clinical_report(str(session.id), str(patient_profile.id), str(session.doctor_id))
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
        visible_to_patient=report.visible_to_patient,
        model_version=report.model_version,
        generated_at=report.generated_at
    )
    db.add(db_report)
    db.commit()
    
    return {"message": "Session ended", "report_id": str(db_report.id)}


@router.get("/sessions/{session_id}/report", response_model=ClinicalReportResponse)
def get_report(session_id: str, current_user: User = Depends(get_current_patient_user), db: Session = Depends(get_db)):
    raise HTTPException(status_code=403, detail="Clinical reports are doctor-only")