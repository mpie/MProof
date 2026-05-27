"""Add api_keys table

Revision ID: 013
Revises: 012
Create Date: 2026-05-27
"""
from alembic import op
import sqlalchemy as sa

revision = '013'
down_revision = '012'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'api_keys',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('client_id', sa.String(64), nullable=False, unique=True),
        sa.Column('client_secret_hash', sa.String(128), nullable=False),
        sa.Column('scopes', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_api_keys_client_id', 'api_keys', ['client_id'], unique=True)


def downgrade():
    op.drop_index('ix_api_keys_client_id', table_name='api_keys')
    op.drop_table('api_keys')
