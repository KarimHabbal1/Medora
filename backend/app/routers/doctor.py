from fastapi import APIRouter, Depends, HTTPException
from typing import List
from sqlalchemy import exists
from sqlalchemy.orm import Session
from ..database import get_db
from ..auth.dependencies import get_current_doctor_user
from ..models import User, DoctorProfile, PatientProfile, TriageSession, ClinicalReport, DoctorFeedback
from ..schemas.doctor import PatientSummary, ReportSummary, FeedbackCreate
from ..services.session_manager import AgentSessionManager

router = APIRouter(prefix="/doctor", tags=["doctor"])


@router.get("/dashboard")
def get_dashboard(current_user: User = Depends(get_current_doctor_user), db: Session = Depends(get_db)):
    doctor_profile = db.query(DoctorProfile).filter(DoctorProfile.user_id == current_user.id).first()
    if not doctor_profile:
        raise HTTPException(status_code=404, detail="Doctor profile not found")
    
    total_patients = db.query(PatientProfile).filter(PatientProfile.assigned_doctor_id == doctor_profile.id).count()
    pending_reports = (
        db.query(ClinicalReport)
        .join(TriageSession, TriageSession.id == ClinicalReport.session_id)
        .filter(
            TriageSession.doctor_id == doctor_profile.id,
            ~exists().where(DoctorFeedback.report_id == ClinicalReport.id),
        )
        .count()
    )
    
    return {"total_patients": total_patients, "pending_reports": pending_reports}


@router.get("/patients", response_model=List[PatientSummary])
def get_patients(current_user: User = Depends(get_current_doctor_user), db: Session = Depends(get_db)):
    doctor_profile = db.query(DoctorProfile).filter(DoctorProfile.user_id == current_user.id).first()
    if not doctor_profile:
        raise HTTPException(status_code=404, detail="Doctor profile not found")
    
    patients = db.query(PatientProfile).filter(PatientProfile.assigned_doctor_id == doctor_profile.id).all()
    result = []
    for p in patients:
        last_session = db.query(TriageSession).filter(TriageSession.patient_id == p.id).order_by(TriageSession.started_at.desc()).first()
        result.append(PatientSummary(
            id=p.id,
            user_id=p.user_id,
            full_name=p.user.full_name,
            last_triage=last_session.ended_at.isoformat() if last_session and last_session.ended_at else None
        ))
    return result


@router.get("/patients/{patient_id}")
def get_patient(patient_id: str, current_user: User = Depends(get_current_doctor_user), db: Session = Depends(get_db)):
    doctor_profile = db.query(DoctorProfile).filter(DoctorProfile.user_id == current_user.id).first()
    if not doctor_profile:
        raise HTTPException(status_code=404, detail="Doctor profile not found")
    
    patient = db.query(PatientProfile).filter(
        PatientProfile.id == patient_id,
        PatientProfile.assigned_doctor_id == doctor_profile.id
    ).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return {
        "id": patient.id,
        "user_id": patient.user_id,
        "full_name": patient.user.full_name,
        "date_of_birth": patient.date_of_birth,
        "sex": patient.sex,
        "height_cm": patient.height_cm,
        "weight_kg": patient.weight_kg
    }


@router.get("/patients/{patient_id}/reports", response_model=List[ReportSummary])
def get_patient_reports(patient_id: str, current_user: User = Depends(get_current_doctor_user), db: Session = Depends(get_db)):
    doctor_profile = db.query(DoctorProfile).filter(DoctorProfile.user_id == current_user.id).first()
    if not doctor_profile:
        raise HTTPException(status_code=404, detail="Doctor profile not found")
    
    reports = db.query(ClinicalReport).filter(
        ClinicalReport.patient_id == patient_id,
        ClinicalReport.doctor_id == doctor_profile.id
    ).all()
    
    result = []
    for r in reports:
        result.append(ReportSummary(
            id=r.id,
            session_id=r.session_id,
            patient_id=r.patient_id,
            patient_name=r.patient.user.full_name,
            generated_at=r.generated_at,
            urgency_level=r.urgency_level
        ))
    return result


@router.get("/reports", response_model=List[ReportSummary])
def get_reports(current_user: User = Depends(get_current_doctor_user), db: Session = Depends(get_db)):
    doctor_profile = db.query(DoctorProfile).filter(DoctorProfile.user_id == current_user.id).first()
    if not doctor_profile:
        raise HTTPException(status_code=404, detail="Doctor profile not found")
    
    reports = db.query(ClinicalReport).filter(ClinicalReport.doctor_id == doctor_profile.id).all()
    
    result = []
    for r in reports:
        result.append(ReportSummary(
            id=r.id,
            session_id=r.session_id,
            patient_id=r.patient_id,
            patient_name=r.patient.user.full_name,
            generated_at=r.generated_at,
            urgency_level=r.urgency_level
        ))
    return result


@router.get("/reports/{report_id}")
def get_report(report_id: str, current_user: User = Depends(get_current_doctor_user), db: Session = Depends(get_db)):
    doctor_profile = db.query(DoctorProfile).filter(DoctorProfile.user_id == current_user.id).first()
    if not doctor_profile:
        raise HTTPException(status_code=404, detail="Doctor profile not found")
    
    report = db.query(ClinicalReport).filter(
        ClinicalReport.id == report_id,
        ClinicalReport.doctor_id == doctor_profile.id
    ).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    return report


@router.get("/reports/{report_id}/full-chain")
def get_report_full_chain(report_id: str, current_user: User = Depends(get_current_doctor_user), db: Session = Depends(get_db)):
    """Return the full diagnostic chain: intake summary + clinical picture + report."""
    doctor_profile = db.query(DoctorProfile).filter(DoctorProfile.user_id == current_user.id).first()
    if not doctor_profile:
        raise HTTPException(status_code=404, detail="Doctor profile not found")
    
    report = db.query(ClinicalReport).filter(
        ClinicalReport.id == report_id,
        ClinicalReport.doctor_id == doctor_profile.id
    ).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    session = db.query(TriageSession).filter(TriageSession.id == report.session_id).first()
    
    return {
        "report": report,
        "intake_summary": session.intake_summary_json if session else None,
        "clinical_picture": session.clinical_picture_json if session else None,
        "agent_phase": session.agent_phase.value if session and session.agent_phase else None,
    }


@router.post("/reports/{report_id}/feedback")
def submit_feedback(report_id: str, feedback: FeedbackCreate, current_user: User = Depends(get_current_doctor_user), db: Session = Depends(get_db)):
    doctor_profile = db.query(DoctorProfile).filter(DoctorProfile.user_id == current_user.id).first()
    if not doctor_profile:
        raise HTTPException(status_code=404, detail="Doctor profile not found")
    
    report = db.query(ClinicalReport).filter(
        ClinicalReport.id == report_id,
        ClinicalReport.doctor_id == doctor_profile.id
    ).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Check if feedback already exists
    existing = db.query(DoctorFeedback).filter(
        DoctorFeedback.report_id == report.id,
        DoctorFeedback.doctor_id == doctor_profile.id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Feedback already submitted")
    
    db_feedback = DoctorFeedback(
        report_id=report.id,
        doctor_id=doctor_profile.id,
        rating=feedback.rating,
        correction_text=feedback.correction_text,
        feedback_category=feedback.feedback_category
    )
    db.add(db_feedback)
    db.commit()
    
    # Also forward to FeedbackStore for training data pipeline
    try:
        manager = AgentSessionManager.get_instance()
        decision = "confirmed" if feedback.rating == "thumbs_up" else "rejected"
        # Look up the session's intake summary to find the case in the feedback store
        session = db.query(TriageSession).filter(TriageSession.id == report.session_id).first()
        if session and session.intake_summary_json:
            patient_name = report.patient.user.full_name
            # Try to find and update the case in FeedbackStore
            pending = manager.feedback_store.get_pending_cases()
            for case in pending:
                if case.get("patient_name") == patient_name:
                    manager.feedback_store.submit_feedback(
                        case_id=case["case_id"],
                        doctor_decision=decision,
                        doctor_diagnosis=feedback.correction_text or "",
                        doctor_notes=str(feedback.feedback_category) if feedback.feedback_category else "",
                    )
                    break
    except Exception:
        pass  # FeedbackStore sync is best-effort

    return {"message": "Feedback submitted"}


@router.get("/feedback/statistics")
def get_feedback_statistics(current_user: User = Depends(get_current_doctor_user)):
    """Return aggregate feedback statistics from the FeedbackStore."""
    try:
        manager = AgentSessionManager.get_instance()
        stats = manager.feedback_store.get_statistics()
        return stats
    except Exception:
        return {"total_cases": 0, "confirmed": 0, "rejected": 0, "pending": 0}


@router.get("/feedback/pending")
def get_pending_feedback_cases(current_user: User = Depends(get_current_doctor_user)):
    """Return all pending feedback cases awaiting doctor review."""
    try:
        manager = AgentSessionManager.get_instance()
        cases = manager.feedback_store.get_pending_cases()
        return cases
    except Exception:
        return []