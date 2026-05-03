from sqlalchemy import Column, String, UUID, TIMESTAMP, Boolean, Text, Integer, Float, Date, Numeric, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid

from ..database import Base
from .enums import (
    UserRole, RegistrationMethod, TriageSessionStatus, UrgencyLevel, EscalationType,
    ChatRetentionPolicy, MessageSender, MessageType, AgentPhase, ConsentType, DoctorFeedbackRating, FeedbackCategory
)


class Hospital(Base):
    __tablename__ = "hospitals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    address = Column(Text, nullable=True)
    contact_email = Column(String, nullable=True)
    local_server_identifier = Column(String, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    users = relationship("User", back_populates="hospital")
    doctor_profiles = relationship("DoctorProfile", back_populates="hospital")
    patient_profiles = relationship("PatientProfile", back_populates="hospital")
    triage_sessions = relationship("TriageSession", back_populates="hospital")
    audit_logs = relationship("AuditLog", back_populates="hospital")


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hospital_id = Column(UUID(as_uuid=True), ForeignKey("hospitals.id"), nullable=False)
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    phone = Column(String, nullable=True)
    password_hash = Column(Text, nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False)
    is_active = Column(Boolean, default=True)
    registration_method = Column(SQLEnum(RegistrationMethod), nullable=False)
    created_by_admin_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    last_login_at = Column(TIMESTAMP, nullable=True)

    # Relationships
    hospital = relationship("Hospital", back_populates="users")
    doctor_profile = relationship("DoctorProfile", back_populates="user", uselist=False)
    patient_profile = relationship("PatientProfile", back_populates="user", uselist=False)
    created_users = relationship("User", backref="created_by_admin", remote_side=[id])
    audit_logs = relationship("AuditLog", back_populates="user")


class DoctorProfile(Base):
    __tablename__ = "doctor_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), unique=True, nullable=False)
    hospital_id = Column(UUID(as_uuid=True), ForeignKey("hospitals.id"), nullable=False)
    specialty = Column(String, nullable=False)
    license_number = Column(String, nullable=True)
    department = Column(String, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="doctor_profile")
    hospital = relationship("Hospital", back_populates="doctor_profiles")
    patient_profiles = relationship("PatientProfile", back_populates="assigned_doctor")
    triage_sessions = relationship("TriageSession", back_populates="doctor")
    clinical_reports = relationship("ClinicalReport", back_populates="doctor")
    doctor_feedbacks = relationship("DoctorFeedback", back_populates="doctor")


class PatientProfile(Base):
    __tablename__ = "patient_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), unique=True, nullable=False)
    hospital_id = Column(UUID(as_uuid=True), ForeignKey("hospitals.id"), nullable=False)
    assigned_doctor_id = Column(UUID(as_uuid=True), ForeignKey("doctor_profiles.id"), nullable=True)
    date_of_birth = Column(Date, nullable=True)
    sex = Column(String, nullable=True)
    height_cm = Column(Numeric, nullable=True)
    weight_kg = Column(Numeric, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="patient_profile")
    hospital = relationship("Hospital", back_populates="patient_profiles")
    assigned_doctor = relationship("DoctorProfile", back_populates="patient_profiles")
    patient_medical_history = relationship("PatientMedicalHistory", back_populates="patient", uselist=False)
    patient_consents = relationship("PatientConsent", back_populates="patient")
    triage_sessions = relationship("TriageSession", back_populates="patient")
    clinical_reports = relationship("ClinicalReport", back_populates="patient")


class PatientMedicalHistory(Base):
    __tablename__ = "patient_medical_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patient_profiles.id"), unique=True, nullable=False)
    chronic_conditions = Column(JSONB, nullable=True)
    medications = Column(JSONB, nullable=True)
    allergies = Column(JSONB, nullable=True)
    surgeries = Column(JSONB, nullable=True)
    family_history = Column(JSONB, nullable=True)
    smoking_status = Column(String, nullable=True)
    pregnancy_status = Column(String, nullable=True)
    additional_notes = Column(Text, nullable=True)
    skipped = Column(Boolean, default=False)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    # Relationships
    patient = relationship("PatientProfile", back_populates="patient_medical_history")


class PatientConsent(Base):
    __tablename__ = "patient_consents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patient_profiles.id"), nullable=False)
    consent_type = Column(SQLEnum(ConsentType), nullable=False)
    accepted = Column(Boolean, nullable=False)
    accepted_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    patient = relationship("PatientProfile", back_populates="patient_consents")


class TriageSession(Base):
    __tablename__ = "triage_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patient_profiles.id"), nullable=False)
    doctor_id = Column(UUID(as_uuid=True), ForeignKey("doctor_profiles.id"), nullable=False)
    hospital_id = Column(UUID(as_uuid=True), ForeignKey("hospitals.id"), nullable=False)
    status = Column(SQLEnum(TriageSessionStatus), nullable=False)
    chief_complaint = Column(Text, nullable=True)
    detected_symptoms = Column(JSONB, nullable=True)
    urgency_level = Column(SQLEnum(UrgencyLevel), nullable=False)
    escalation_type = Column(SQLEnum(EscalationType), nullable=False)
    chat_retention_policy = Column(SQLEnum(ChatRetentionPolicy), nullable=False)
    agent_phase = Column(SQLEnum(AgentPhase), nullable=True, default=AgentPhase.intake)
    intake_summary_json = Column(JSONB, nullable=True)
    clinical_picture_json = Column(JSONB, nullable=True)
    started_at = Column(TIMESTAMP, server_default=func.now())
    ended_at = Column(TIMESTAMP, nullable=True)

    # Relationships
    patient = relationship("PatientProfile", back_populates="triage_sessions")
    doctor = relationship("DoctorProfile", back_populates="triage_sessions")
    hospital = relationship("Hospital", back_populates="triage_sessions")
    session_messages = relationship("SessionMessage", back_populates="session")
    clinical_report = relationship("ClinicalReport", back_populates="session", uselist=False)
    rag_queries = relationship("RagQuery", back_populates="session")
    performance_logs = relationship("PerformanceLog", back_populates="session")


class SessionMessage(Base):
    __tablename__ = "session_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("triage_sessions.id"), nullable=False)
    sender = Column(SQLEnum(MessageSender), nullable=False)
    content = Column(Text, nullable=False)
    message_type = Column(SQLEnum(MessageType), nullable=False)
    is_persisted_after_summary = Column(Boolean, default=True)
    is_visible_to_doctor = Column(Boolean, default=True)
    is_deleted = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    session = relationship("TriageSession", back_populates="session_messages")


class ClinicalReport(Base):
    __tablename__ = "clinical_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("triage_sessions.id"), unique=True, nullable=False)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patient_profiles.id"), nullable=False)
    doctor_id = Column(UUID(as_uuid=True), ForeignKey("doctor_profiles.id"), nullable=False)
    presenting_complaints = Column(JSONB, nullable=True)
    history_of_presenting_complaint = Column(JSONB, nullable=True)
    summary_text = Column(Text, nullable=False)
    suspected_conditions = Column(JSONB, nullable=True)
    triggered_red_flags = Column(JSONB, nullable=True)
    urgency_level = Column(SQLEnum(UrgencyLevel), nullable=False)
    recommended_action = Column(Text, nullable=False)
    specialty_routing = Column(JSONB, nullable=True)
    suggested_workup = Column(JSONB, nullable=True)
    key_exam_findings = Column(JSONB, nullable=True)
    admission_criteria = Column(JSONB, nullable=True)
    referral_criteria = Column(JSONB, nullable=True)
    external_escalation_completed = Column(Boolean, default=False)
    escalation_message = Column(Text, nullable=True)
    visible_to_patient = Column(Boolean, default=False)
    model_version = Column(String, nullable=True)
    diagnosis_mode = Column(String, nullable=True)
    diagnosis_pass_count = Column(Integer, nullable=True)
    chunks_used_count = Column(Integer, nullable=True)
    web_search_results = Column(JSONB, nullable=True)
    generated_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    session = relationship("TriageSession", back_populates="clinical_report")
    patient = relationship("PatientProfile", back_populates="clinical_reports")
    doctor = relationship("DoctorProfile", back_populates="clinical_reports")
    doctor_feedbacks = relationship("DoctorFeedback", back_populates="report")


class DoctorFeedback(Base):
    __tablename__ = "doctor_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id = Column(UUID(as_uuid=True), ForeignKey("clinical_reports.id"), nullable=False)
    doctor_id = Column(UUID(as_uuid=True), ForeignKey("doctor_profiles.id"), nullable=False)
    rating = Column(SQLEnum(DoctorFeedbackRating), nullable=False)
    correction_text = Column(Text, nullable=True)
    feedback_category = Column(SQLEnum(FeedbackCategory), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    report = relationship("ClinicalReport", back_populates="doctor_feedbacks")
    doctor = relationship("DoctorProfile", back_populates="doctor_feedbacks")


class RagQuery(Base):
    __tablename__ = "rag_queries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("triage_sessions.id"), nullable=False)
    query_text = Column(Text, nullable=False)
    retrieve_k = Column(Integer, default=10)
    final_k = Column(Integer, default=3)
    embedding_model = Column(String, nullable=True)
    reranker_model = Column(String, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    session = relationship("TriageSession", back_populates="rag_queries")
    rag_retrieved_chunks = relationship("RagRetrievedChunk", back_populates="rag_query")
    performance_logs = relationship("PerformanceLog", back_populates="rag_query")


class RagRetrievedChunk(Base):
    __tablename__ = "rag_retrieved_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rag_query_id = Column(UUID(as_uuid=True), ForeignKey("rag_queries.id"), nullable=False)
    chunk_id = Column(String, nullable=False)
    original_rank = Column(Integer, nullable=False)
    final_rank = Column(Integer, nullable=True)
    vector_distance = Column(Float, nullable=True)
    rerank_score = Column(Float, nullable=True)
    chapter = Column(String, nullable=True)
    section = Column(String, nullable=True)
    subsection = Column(String, nullable=True)
    used_in_final_answer = Column(Boolean, default=False)

    # Relationships
    rag_query = relationship("RagQuery", back_populates="rag_retrieved_chunks")


class Symptom(Base):
    __tablename__ = "symptoms"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    body_systems = Column(JSONB, nullable=True)
    epidemiology = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    symptom_questions = relationship("SymptomQuestion", back_populates="symptom")
    symptom_red_flags = relationship("SymptomRedFlag", back_populates="symptom")
    symptom_urgency_rules = relationship("SymptomUrgencyRule", back_populates="symptom")
    symptom_workup_items = relationship("SymptomWorkupItem", back_populates="symptom")


class SymptomQuestion(Base):
    __tablename__ = "symptom_questions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symptom_id = Column(UUID(as_uuid=True), ForeignKey("symptoms.id"), nullable=False)
    question_text = Column(Text, nullable=False)
    order_index = Column(Integer, nullable=False)

    # Relationships
    symptom = relationship("Symptom", back_populates="symptom_questions")


class SymptomRedFlag(Base):
    __tablename__ = "symptom_red_flags"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symptom_id = Column(UUID(as_uuid=True), ForeignKey("symptoms.id"), nullable=False)
    flag = Column(Text, nullable=False)
    implication = Column(Text, nullable=False)
    urgency = Column(SQLEnum(UrgencyLevel), nullable=False)

    # Relationships
    symptom = relationship("Symptom", back_populates="symptom_red_flags")


class SymptomUrgencyRule(Base):
    __tablename__ = "symptom_urgency_rules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symptom_id = Column(UUID(as_uuid=True), ForeignKey("symptoms.id"), nullable=False)
    criteria = Column(Text, nullable=False)
    urgency = Column(SQLEnum(UrgencyLevel), nullable=False)
    action = Column(Text, nullable=False)

    # Relationships
    symptom = relationship("Symptom", back_populates="symptom_urgency_rules")


class SymptomWorkupItem(Base):
    __tablename__ = "symptom_workup_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symptom_id = Column(UUID(as_uuid=True), ForeignKey("symptoms.id"), nullable=False)
    item_text = Column(Text, nullable=False)

    # Relationships
    symptom = relationship("Symptom", back_populates="symptom_workup_items")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    hospital_id = Column(UUID(as_uuid=True), ForeignKey("hospitals.id"), nullable=False)
    action = Column(String, nullable=False)
    entity_type = Column(String, nullable=False)
    entity_id = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="audit_logs")
    hospital = relationship("Hospital", back_populates="audit_logs")


class PerformanceLog(Base):
    __tablename__ = "performance_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("triage_sessions.id"), nullable=True)
    rag_query_id = Column(UUID(as_uuid=True), ForeignKey("rag_queries.id"), nullable=True)
    retrieval_time_ms = Column(Integer, nullable=True)
    rerank_time_ms = Column(Integer, nullable=True)
    llm_time_ms = Column(Integer, nullable=True)
    total_time_ms = Column(Integer, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    session = relationship("TriageSession", back_populates="performance_logs")
    rag_query = relationship("RagQuery", back_populates="performance_logs")