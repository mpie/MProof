"""Add external_reference to documents

Revision ID: 008
Revises: 007
Create Date: 2026-05-27
"""
from alembic import op
import sqlalchemy as sa


revision = '008'
down_revision = '007'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('documents', sa.Column('external_reference', sa.String(255), nullable=True))
    op.create_index('ix_documents_external_reference', 'documents', ['external_reference'], unique=False)


def downgrade():
    op.drop_index('ix_documents_external_reference', table_name='documents')
    op.drop_column('documents', 'external_reference')
