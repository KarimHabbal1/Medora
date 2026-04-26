from typing import List
from ..schemas.triage import TriageSession, Message, Report
from ..schemas.doctor import PatientSummary, ReportSummary
from ..schemas.patient import MedicalHistory, Consent


# Mock data for triage sessions
mock_sessions = [
    TriageSession(id="session-1", patient_id=1, status="active", created_at="2023-10-01T10:00:00Z"),
    TriageSession(id="session-2", patient_id=1, status="completed", created_at="2023-09-15T14:30:00Z"),
]

mock_messages = {
    "session-1": [
        Message(id="msg-1", session_id="session-1", content="Hello, I'm feeling unwell.", role="user", timestamp="2023-10-01T10:01:00Z"),
        Message(id="msg-2", session_id="session-1", content="Can you describe your symptoms?", role="assistant", timestamp="2023-10-01T10:02:00Z"),
    ]
}

mock_reports = {
    "session-1": Report(id="report-1", session_id="session-1", summary="Patient reports headache and nausea.", recommendations=["Rest", "Hydrate"], urgency_level="low"),
}

# Mock data for doctor
mock_patients = [
    PatientSummary(id=1, full_name="John Doe", last_triage="2023-10-01T10:00:00Z"),
]

mock_reports_summary = [
    ReportSummary(id="report-1", patient_id=1, patient_name="John Doe", created_at="2023-10-01T11:00:00Z", status="pending"),
]

# Mock data for patient
mock_medical_history = MedicalHistory(
    conditions=["Hypertension"],
    medications=["Lisinopril"],
    allergies=["Penicillin"]
)

mock_consents = [
    Consent(type="data_sharing", granted=True, granted_at="2023-01-01T00:00:00Z"),
]