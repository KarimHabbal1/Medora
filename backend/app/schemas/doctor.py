from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from .enums import DoctorFeedbackRating, FeedbackCategory


class PatientSummary(BaseModel):
    id: UUID
    user_id: UUID
    full_name: str
    last_triage: Optional[str] = None


class ReportSummary(BaseModel):
    id: UUID
    session_id: UUID
    patient_id: UUID
    patient_name: str
    generated_at: datetime
    urgency_level: str


class FeedbackCreate(BaseModel):
    rating: DoctorFeedbackRating
    correction_text: Optional[str] = None
    feedback_category: Optional[FeedbackCategory] = None