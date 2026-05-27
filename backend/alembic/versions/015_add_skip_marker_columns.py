"""Add skip_marker_used and skip_marker_position columns to documents

Revision ID: 015
Revises: 014
Create Date: 2026-05-27
"""
from alembic import op
import sqlalchemy as sa

revision = '015'
down_revision = '014'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()

    # Add skip_marker_used if missing
    existing = conn.execute(
        sa.text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'documents' AND COLUMN_NAME = 'skip_marker_used'")
    ).fetchone()
    if not existing:
        op.add_column('documents', sa.Column('skip_marker_used', sa.String(500), nullable=True))

    # Add skip_marker_position if missing
    existing = conn.execute(
        sa.text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'documents' AND COLUMN_NAME = 'skip_marker_position'")
    ).fetchone()
    if not existing:
        op.add_column('documents', sa.Column('skip_marker_position', sa.Integer(), nullable=True))


def downgrade():
    op.drop_column('documents', 'skip_marker_position')
    op.drop_column('documents', 'skip_marker_used')
