"""Add feedback_status and corrected_doc_type to documents

Revision ID: 012
Revises: 011
Create Date: 2026-05-27
"""
from alembic import op
import sqlalchemy as sa

revision = '012'
down_revision = '011'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('documents', sa.Column(
        'feedback_status',
        sa.Enum('confirmed', 'rejected', name='feedback_status_enum'),
        nullable=True
    ))
    op.add_column('documents', sa.Column(
        'corrected_doc_type',
        sa.String(100),
        nullable=True
    ))


def downgrade():
    op.drop_column('documents', 'corrected_doc_type')
    op.drop_column('documents', 'feedback_status')
