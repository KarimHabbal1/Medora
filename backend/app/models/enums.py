from enum import Enum


class UserRole(str, Enum):
    patient = "patient"
    doctor = "doctor"
    admin = "admin"


class RegistrationMethod(str, Enum):
    admin_created = "admin_created"
    self_signup = "self_signup"


class TriageSessionStatus(str, Enum):
    active = "active"
    completed = "completed"
    cancelled = "cancelled"


class UrgencyLevel(str, Enum):
    routine = "routine"
    urgent = "urgent"
    emergency = "emergency"
    unknown = "unknown"


class EscalationType(str, Enum):
    none = "none"
    emergency_call = "emergency_call"
    complex_diagnosis_agent = "complex_diagnosis_agent"


class ChatRetentionPolicy(str, Enum):
    summary_only = "summary_only"
    keep_full_history = "keep_full_history"


class MessageSender(str, Enum):
    patient = "patient"
    intake_agent = "intake_agent"
    rag_agent = "rag_agent"
    triage_agent = "triage_agent"
    system = "system"


class MessageType(str, Enum):
    text = "text"
    question = "question"
    answer = "answer"
    warning = "warning"
    summary = "summary"
    stream_delta = "stream_delta"
    diagnosis = "diagnosis"
    escalation = "escalation"


class AgentPhase(str, Enum):
    intake = "intake"
    triage_mode_a = "triage_mode_a"
    triage_mode_b = "triage_mode_b"
    escalated = "escalated"
    completed = "completed"


class ConsentType(str, Enum):
    medical_disclaimer = "medical_disclaimer"
    data_storage = "data_storage"
    ai_assistance = "ai_assistance"
    chat_history_storage = "chat_history_storage"


class DoctorFeedbackRating(str, Enum):
    thumbs_up = "thumbs_up"
    thumbs_down = "thumbs_down"


class FeedbackCategory(str, Enum):
    wrong_urgency = "wrong_urgency"
    wrong_diagnosis = "wrong_diagnosis"
    missing_info = "missing_info"
    unsafe_response = "unsafe_response"
    irrelevant_sources = "irrelevant_sources"
    other = "other"