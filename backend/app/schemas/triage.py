from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from .enums import TriageSessionStatus, UrgencyLevel, EscalationType, ChatRetentionPolicy, MessageSender, MessageType, AgentPhase


class TriageSessionCreate(BaseModel):
    chief_complaint: Optional[str] = None


class TriageSessionResponse(BaseModel):
    """Full session response — used by doctor-facing endpoints."""
    id: UUID
    patient_id: UUID
    doctor_id: UUID
    hospital_id: UUID
    status: TriageSessionStatus
    chief_complaint: Optional[str]
    detected_symptoms: Optional[Dict[str, Any]]
    urgency_level: UrgencyLevel
    escalation_type: EscalationType
    chat_retention_policy: ChatRetentionPolicy
    agent_phase: Optional[AgentPhase]
    started_at: datetime
    ended_at: Optional[datetime]

    class Config:
        from_attributes = True


class PatientTriageSessionResponse(BaseModel):
    """Patient-safe session response — excludes urgency, symptoms, and diagnosis data."""
    id: UUID
    status: TriageSessionStatus
    chief_complaint: Optional[str]
    agent_phase: Optional[AgentPhase]
    started_at: datetime
    ended_at: Optional[datetime]

    class Config:
        from_attributes = True


class SessionPhaseResponse(BaseModel):
    """Current agent phase for the session."""
    phase: Optional[AgentPhase]
    is_escalated: bool = False

    class Config:
        from_attributes = True


class MessageCreate(BaseModel):
    content: str


class MessageResponse(BaseModel):
    id: UUID
    session_id: UUID
    sender: MessageSender
    content: str
    message_type: MessageType
    is_persisted_after_summary: bool
    is_visible_to_doctor: bool
    is_deleted: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ClinicalReportResponse(BaseModel):
    id: UUID
    session_id: UUID
    patient_id: UUID
    doctor_id: UUID
    presenting_complaints: Optional[Any]
    history_of_presenting_complaint: Optional[Any]
    summary_text: str
    suspected_conditions: Optional[Any]
    triggered_red_flags: Optional[Any]
    urgency_level: UrgencyLevel
    recommended_action: str
    specialty_routing: Optional[Any]
    suggested_workup: Optional[Any]
    key_exam_findings: Optional[Any]
    admission_criteria: Optional[Any]
    referral_criteria: Optional[Any]
    external_escalation_completed: bool
    escalation_message: Optional[str]
    visible_to_patient: bool
    model_version: Optional[str]
    diagnosis_mode: Optional[str]
    diagnosis_pass_count: Optional[int]
    chunks_used_count: Optional[int]
    web_search_results: Optional[Any]
    generated_at: datetime

    class Config:
        from_attributes = True