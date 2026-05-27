"""Add callback_url to documents

Revision ID: 009
Revises: 008
Create Date: 2026-05-27
"""
from alembic import op
import sqlalchemy as sa

revision = '009'
down_revision = '008'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('documents', sa.Column('callback_url', sa.String(2048), nullable=True))


def downgrade():
    op.drop_column('documents', 'callback_url')
