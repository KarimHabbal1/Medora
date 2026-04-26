from fastapi import APIRouter, Depends
from typing import List
from ..auth.dependencies import get_current_doctor_user
from ..schemas.doctor import PatientSummary, ReportSummary, FeedbackCreate
from ..utils.mocks import mock_patients, mock_reports_summary

router = APIRouter(prefix="/doctor", tags=["doctor"])


@router.get("/dashboard")
def get_dashboard(current_user = Depends(get_current_doctor_user)):
    # Mock dashboard data
    return {"total_patients": 10, "pending_reports": 5}


@router.get("/patients", response_model=List[PatientSummary])
def get_patients(current_user = Depends(get_current_doctor_user)):
    # Mock data
    return mock_patients


@router.get("/patients/{patient_id}")
def get_patient(patient_id: int, current_user = Depends(get_current_doctor_user)):
    # Mock patient details
    return {"id": patient_id, "name": "John Doe"}


@router.get("/patients/{patient_id}/reports", response_model=List[ReportSummary])
def get_patient_reports(patient_id: int, current_user = Depends(get_current_doctor_user)):
    # Mock reports for patient
    return mock_reports_summary


@router.get("/reports", response_model=List[ReportSummary])
def get_reports(current_user = Depends(get_current_doctor_user)):
    # Mock all reports
    return mock_reports_summary


@router.get("/reports/{report_id}")
def get_report(report_id: str, current_user = Depends(get_current_doctor_user)):
    # Mock report details
    return {"id": report_id, "summary": "Mock report"}


@router.post("/reports/{report_id}/feedback")
def submit_feedback(report_id: str, feedback: FeedbackCreate, current_user = Depends(get_current_doctor_user)):
    # Placeholder: save feedback
    return {"message": "Feedback submitted"}