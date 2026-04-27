"""Initial full schema

Revision ID: 001_initial
Revises: 
Create Date: 2024-04-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enums first
    userrole = postgresql.ENUM('patient', 'doctor', 'admin', name='userrole')
    userrole.create(op.get_bind())

    registrationmethod = postgresql.ENUM('admin_created', 'self_signup', name='registrationmethod')
    registrationmethod.create(op.get_bind())

    triagesessionstatus = postgresql.ENUM('active', 'completed', 'cancelled', name='triagesessionstatus')
    triagesessionstatus.create(op.get_bind())

    urgencylevel = postgresql.ENUM('routine', 'urgent', 'emergency', 'unknown', name='urgencylevel')
    urgencylevel.create(op.get_bind())

    escalationtype = postgresql.ENUM('none', 'emergency_call', 'complex_diagnosis_agent', name='escalationtype')
    escalationtype.create(op.get_bind())

    chatretentionpolicy = postgresql.ENUM('summary_only', 'keep_full_history', name='chatretentionpolicy')
    chatretentionpolicy.create(op.get_bind())

    messagesender = postgresql.ENUM('patient', 'intake_agent', 'rag_agent', 'system', name='messagesender')
    messagesender.create(op.get_bind())

    messagetype = postgresql.ENUM('text', 'question', 'answer', 'warning', 'summary', 'stream_delta', name='messagetype')
    messagetype.create(op.get_bind())

    consenttype = postgresql.ENUM('medical_disclaimer', 'data_storage', 'ai_assistance', 'chat_history_storage', name='consenttype')
    consenttype.create(op.get_bind())

    doctorfeedbackrating = postgresql.ENUM('thumbs_up', 'thumbs_down', name='doctorfeedbackrating')
    doctorfeedbackrating.create(op.get_bind())

    feedbackcategory = postgresql.ENUM('wrong_urgency', 'wrong_diagnosis', 'missing_info', 'unsafe_response', 'irrelevant_sources', 'other', name='feedbackcategory')
    feedbackcategory.create(op.get_bind())

    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

    # Create hospitals table
    op.create_table('hospitals',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('address', sa.Text(), nullable=True),
        sa.Column('contact_email', sa.String(), nullable=True),
        sa.Column('local_server_identifier', sa.String(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False)
    )

    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('hospital_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('hospitals.id'), nullable=False),
        sa.Column('full_name', sa.String(), nullable=False),
        sa.Column('email', sa.String(), unique=True, nullable=False),
        sa.Column('phone', sa.String(), nullable=True),
        sa.Column('password_hash', sa.Text(), nullable=False),
        sa.Column('role', userrole, nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('registration_method', registrationmethod, nullable=False),
        sa.Column('created_by_admin_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_login_at', sa.TIMESTAMP(), nullable=True)
    )

    # Create doctor_profiles table
    op.create_table('doctor_profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), unique=True, nullable=False),
        sa.Column('hospital_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('hospitals.id'), nullable=False),
        sa.Column('specialty', sa.String(), nullable=False),
        sa.Column('license_number', sa.String(), nullable=True),
        sa.Column('department', sa.String(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False)
    )

    # Create patient_profiles table
    op.create_table('patient_profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), unique=True, nullable=False),
        sa.Column('hospital_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('hospitals.id'), nullable=False),
        sa.Column('assigned_doctor_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('doctor_profiles.id'), nullable=True),
        sa.Column('date_of_birth', sa.Date(), nullable=True),
        sa.Column('sex', sa.String(), nullable=True),
        sa.Column('height_cm', sa.Numeric(), nullable=True),
        sa.Column('weight_kg', sa.Numeric(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False)
    )

    # Create patient_medical_history table
    op.create_table('patient_medical_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('patient_profiles.id'), unique=True, nullable=False),
        sa.Column('chronic_conditions', postgresql.JSONB(), nullable=True),
        sa.Column('medications', postgresql.JSONB(), nullable=True),
        sa.Column('allergies', postgresql.JSONB(), nullable=True),
        sa.Column('surgeries', postgresql.JSONB(), nullable=True),
        sa.Column('family_history', postgresql.JSONB(), nullable=True),
        sa.Column('smoking_status', sa.String(), nullable=True),
        sa.Column('pregnancy_status', sa.String(), nullable=True),
        sa.Column('additional_notes', sa.Text(), nullable=True),
        sa.Column('skipped', sa.Boolean(), default=False, nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False)
    )

    # Create patient_consents table
    op.create_table('patient_consents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('patient_profiles.id'), nullable=False),
        sa.Column('consent_type', consenttype, nullable=False),
        sa.Column('accepted', sa.Boolean(), nullable=False),
        sa.Column('accepted_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False)
    )

    # Create triage_sessions table
    op.create_table('triage_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('patient_profiles.id'), nullable=False),
        sa.Column('doctor_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('doctor_profiles.id'), nullable=False),
        sa.Column('hospital_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('hospitals.id'), nullable=False),
        sa.Column('status', triagesessionstatus, nullable=False),
        sa.Column('chief_complaint', sa.Text(), nullable=True),
        sa.Column('detected_symptoms', postgresql.JSONB(), nullable=True),
        sa.Column('urgency_level', urgencylevel, nullable=False),
        sa.Column('escalation_type', escalationtype, nullable=False),
        sa.Column('chat_retention_policy', chatretentionpolicy, nullable=False),
        sa.Column('started_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False),
        sa.Column('ended_at', sa.TIMESTAMP(), nullable=True)
    )

    # Create session_messages table
    op.create_table('session_messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('triage_sessions.id'), nullable=False),
        sa.Column('sender', messagesender, nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('message_type', messagetype, nullable=False),
        sa.Column('is_persisted_after_summary', sa.Boolean(), default=True, nullable=False),
        sa.Column('is_visible_to_doctor', sa.Boolean(), default=True, nullable=False),
        sa.Column('is_deleted', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False)
    )

    # Create clinical_reports table
    op.create_table('clinical_reports',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('triage_sessions.id'), unique=True, nullable=False),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('patient_profiles.id'), nullable=False),
        sa.Column('doctor_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('doctor_profiles.id'), nullable=False),
        sa.Column('presenting_complaints', postgresql.JSONB(), nullable=True),
        sa.Column('history_of_presenting_complaint', postgresql.JSONB(), nullable=True),
        sa.Column('summary_text', sa.Text(), nullable=False),
        sa.Column('suspected_conditions', postgresql.JSONB(), nullable=True),
        sa.Column('triggered_red_flags', postgresql.JSONB(), nullable=True),
        sa.Column('urgency_level', urgencylevel, nullable=False),
        sa.Column('recommended_action', sa.Text(), nullable=False),
        sa.Column('specialty_routing', postgresql.JSONB(), nullable=True),
        sa.Column('suggested_workup', postgresql.JSONB(), nullable=True),
        sa.Column('key_exam_findings', postgresql.JSONB(), nullable=True),
        sa.Column('admission_criteria', postgresql.JSONB(), nullable=True),
        sa.Column('referral_criteria', postgresql.JSONB(), nullable=True),
        sa.Column('external_escalation_completed', sa.Boolean(), default=False, nullable=False),
        sa.Column('escalation_message', sa.Text(), nullable=True),
        sa.Column('visible_to_patient', sa.Boolean(), default=False, nullable=False),
        sa.Column('model_version', sa.String(), nullable=True),
        sa.Column('generated_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False)
    )

    # Create doctor_feedback table
    op.create_table('doctor_feedback',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('report_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('clinical_reports.id'), nullable=False),
        sa.Column('doctor_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('doctor_profiles.id'), nullable=False),
        sa.Column('rating', doctorfeedbackrating, nullable=False),
        sa.Column('correction_text', sa.Text(), nullable=True),
        sa.Column('feedback_category', feedbackcategory, nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False)
    )

    # Create rag_queries table
    op.create_table('rag_queries',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('triage_sessions.id'), nullable=False),
        sa.Column('query_text', sa.Text(), nullable=False),
        sa.Column('retrieve_k', sa.Integer(), default=10, nullable=False),
        sa.Column('final_k', sa.Integer(), default=3, nullable=False),
        sa.Column('embedding_model', sa.String(), nullable=True),
        sa.Column('reranker_model', sa.String(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False)
    )

    # Create rag_retrieved_chunks table
    op.create_table('rag_retrieved_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('rag_query_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('rag_queries.id'), nullable=False),
        sa.Column('chunk_id', sa.String(), nullable=False),
        sa.Column('original_rank', sa.Integer(), nullable=False),
        sa.Column('final_rank', sa.Integer(), nullable=True),
        sa.Column('vector_distance', sa.Float(), nullable=True),
        sa.Column('rerank_score', sa.Float(), nullable=True),
        sa.Column('chapter', sa.String(), nullable=True),
        sa.Column('section', sa.String(), nullable=True),
        sa.Column('subsection', sa.String(), nullable=True),
        sa.Column('used_in_final_answer', sa.Boolean(), default=False, nullable=False)
    )

    # Create symptoms table
    op.create_table('symptoms',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(), unique=True, nullable=False),
        sa.Column('body_systems', postgresql.JSONB(), nullable=True),
        sa.Column('epidemiology', sa.Text(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False)
    )

    # Create symptom_questions table
    op.create_table('symptom_questions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('symptom_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('symptoms.id'), nullable=False),
        sa.Column('question_text', sa.Text(), nullable=False),
        sa.Column('order_index', sa.Integer(), nullable=False)
    )

    # Create symptom_red_flags table
    op.create_table('symptom_red_flags',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('symptom_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('symptoms.id'), nullable=False),
        sa.Column('flag', sa.Text(), nullable=False),
        sa.Column('implication', sa.Text(), nullable=False),
        sa.Column('urgency', urgencylevel, nullable=False)
    )

    # Create symptom_urgency_rules table
    op.create_table('symptom_urgency_rules',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('symptom_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('symptoms.id'), nullable=False),
        sa.Column('criteria', sa.Text(), nullable=False),
        sa.Column('urgency', urgencylevel, nullable=False),
        sa.Column('action', sa.Text(), nullable=False)
    )

    # Create symptom_workup_items table
    op.create_table('symptom_workup_items',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('symptom_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('symptoms.id'), nullable=False),
        sa.Column('item_text', sa.Text(), nullable=False)
    )

    # Create audit_logs table
    op.create_table('audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('hospital_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('hospitals.id'), nullable=False),
        sa.Column('action', sa.String(), nullable=False),
        sa.Column('entity_type', sa.String(), nullable=False),
        sa.Column('entity_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False)
    )

    # Create performance_logs table
    op.create_table('performance_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('triage_sessions.id'), nullable=True),
        sa.Column('rag_query_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('rag_queries.id'), nullable=True),
        sa.Column('retrieval_time_ms', sa.Integer(), nullable=True),
        sa.Column('rerank_time_ms', sa.Integer(), nullable=True),
        sa.Column('llm_time_ms', sa.Integer(), nullable=True),
        sa.Column('total_time_ms', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False)
    )


def downgrade() -> None:
    op.drop_table('performance_logs')
    op.drop_table('audit_logs')
    op.drop_table('symptom_workup_items')
    op.drop_table('symptom_urgency_rules')
    op.drop_table('symptom_red_flags')
    op.drop_table('symptom_questions')
    op.drop_table('symptoms')
    op.drop_table('rag_retrieved_chunks')
    op.drop_table('rag_queries')
    op.drop_table('doctor_feedback')
    op.drop_table('clinical_reports')
    op.drop_table('session_messages')
    op.drop_table('triage_sessions')
    op.drop_table('patient_consents')
    op.drop_table('patient_medical_history')
    op.drop_table('patient_profiles')
    op.drop_table('doctor_profiles')
    op.drop_table('users')
    op.drop_table('hospitals')

    # Drop enums
    op.execute('DROP TYPE IF EXISTS urgencylevel')
    op.execute('DROP TYPE IF EXISTS userrole')
    op.execute('DROP TYPE IF EXISTS registrationmethod')
    op.execute('DROP TYPE IF EXISTS triagesessionstatus')
    op.execute('DROP TYPE IF EXISTS escalationtype')
    op.execute('DROP TYPE IF EXISTS chatretentionpolicy')
    op.execute('DROP TYPE IF EXISTS messagesender')
    op.execute('DROP TYPE IF EXISTS messagetype')
    op.execute('DROP TYPE IF EXISTS consenttype')
    op.execute('DROP TYPE IF EXISTS doctorfeedbackrating')
    op.execute('DROP TYPE IF EXISTS feedbackcategory')