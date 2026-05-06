"""Add web_search_results column to clinical_reports

Revision ID: 003_web_search
Revises: 002_agent_integration
Create Date: 2026-05-04
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = '003_web_search'
down_revision = '002_agent_integration'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('clinical_reports', sa.Column('web_search_results', JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column('clinical_reports', 'web_search_results')
