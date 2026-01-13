"""Add app_settings table for persistent configuration

Revision ID: 004
Revises: 003
Create Date: 2026-01-11
"""
from alembic import op
import sqlalchemy as sa


revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Check if table already exists (idempotent migration)
    try:
        conn = op.get_bind()
        inspector = sa.inspect(conn)
        tables = inspector.get_table_names()
    except Exception:
        # If inspection fails, assume table doesn't exist (safer to try creating)
        tables = []
    
    if 'app_settings' not in tables:
        op.create_table(
            'app_settings',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('key', sa.String(100), nullable=False),
            sa.Column('value', sa.Text(), nullable=False),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('updated_at', sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_app_settings_key'), 'app_settings', ['key'], unique=True)
        
        # Insert default LLM provider setting
        op.execute(
            "INSERT INTO app_settings (key, value, description, created_at, updated_at) "
            "VALUES ('llm_provider', 'ollama', 'Active LLM provider (ollama or vllm)', datetime('now'), datetime('now'))"
        )


def downgrade() -> None:
    op.drop_index(op.f('ix_app_settings_key'), table_name='app_settings')
    op.drop_table('app_settings')
