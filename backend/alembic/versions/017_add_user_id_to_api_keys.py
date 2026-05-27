"""Add user_id FK to api_keys

Revision ID: 017
Revises: 016
Create Date: 2026-05-27
"""
from alembic import op
import sqlalchemy as sa

revision = '017'
down_revision = '016'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'api_keys',
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True),
    )
    op.create_index('ix_api_keys_user_id', 'api_keys', ['user_id'])


def downgrade():
    op.drop_index('ix_api_keys_user_id', table_name='api_keys')
    op.drop_column('api_keys', 'user_id')
