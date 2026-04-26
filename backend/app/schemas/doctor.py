from pydantic import BaseModel
from typing import List, Optional


class PatientSummary(BaseModel):
    id: int
    full_name: str
    last_triage: Optional[str] = None


class ReportSummary(BaseModel):
    id: str
    patient_id: int
    patient_name: str
    created_at: str
    status: str


class FeedbackCreate(BaseModel):
    comments: str
    approved: bool