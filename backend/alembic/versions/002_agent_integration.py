"""Agent integration - add phase tracking and diagnosis metadata

Revision ID: 002_agent_integration
Revises: 001_initial
Create Date: 2026-05-03
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '002_agent_integration'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create the agentphase enum type
    agent_phase_enum = sa.Enum(
        'intake', 'triage_mode_a', 'triage_mode_b', 'escalated', 'completed',
        name='agentphase'
    )
    agent_phase_enum.create(op.get_bind(), checkfirst=True)

    # Add new enum values to existing enums
    op.execute("ALTER TYPE messagesender ADD VALUE IF NOT EXISTS 'triage_agent'")
    op.execute("ALTER TYPE messagetype ADD VALUE IF NOT EXISTS 'diagnosis'")
    op.execute("ALTER TYPE messagetype ADD VALUE IF NOT EXISTS 'escalation'")

    # TriageSession: add agent_phase, intake_summary_json, clinical_picture_json
    op.add_column('triage_sessions', sa.Column('agent_phase', agent_phase_enum, nullable=True))
    op.add_column('triage_sessions', sa.Column('intake_summary_json', JSONB, nullable=True))
    op.add_column('triage_sessions', sa.Column('clinical_picture_json', JSONB, nullable=True))

    # ClinicalReport: add diagnosis_mode, diagnosis_pass_count, chunks_used_count
    op.add_column('clinical_reports', sa.Column('diagnosis_mode', sa.String(), nullable=True))
    op.add_column('clinical_reports', sa.Column('diagnosis_pass_count', sa.Integer(), nullable=True))
    op.add_column('clinical_reports', sa.Column('chunks_used_count', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('clinical_reports', 'chunks_used_count')
    op.drop_column('clinical_reports', 'diagnosis_pass_count')
    op.drop_column('clinical_reports', 'diagnosis_mode')
    op.drop_column('triage_sessions', 'clinical_picture_json')
    op.drop_column('triage_sessions', 'intake_summary_json')
    op.drop_column('triage_sessions', 'agent_phase')

    sa.Enum(name='agentphase').drop(op.get_bind(), checkfirst=True)
