"""Add classification system: signals table and policy column.

Revision ID: 003
Revises: 002
Create Date: 2026-01-10

This migration creates the canonical classification system:
- classification_signals table for generic built-in and user-defined signals
- classification_policy_json column on document_types

NO VERSIONING. Single canonical schema.
"""
from alembic import op
import sqlalchemy as sa


revision = '003'
down_revision = '002_skip_markers'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add classification_policy_json column to document_types (if it doesn't exist)
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('document_types')]
    if 'classification_policy_json' not in columns:
        op.add_column(
            'document_types',
            sa.Column('classification_policy_json', sa.JSON(), nullable=True)
        )

    # Create classification_signals table (if it doesn't exist)
    tables = inspector.get_table_names()
    if 'classification_signals' not in tables:
        op.create_table(
            'classification_signals',
            sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column('key', sa.String(100), nullable=False, unique=True, index=True),
            sa.Column('label', sa.String(255), nullable=False),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('signal_type', sa.String(20), nullable=False),  # 'boolean' or 'count'
            sa.Column('source', sa.String(20), nullable=False),  # 'builtin' or 'user'
            sa.Column('compute_kind', sa.String(20), nullable=False),  # 'builtin', 'keyword_set', 'regex_set'
            sa.Column('config_json', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
        )

        # Seed ONLY generic built-in signals (no domain-specific term lists)
        op.execute("""
            INSERT INTO classification_signals (key, label, description, signal_type, source, compute_kind, config_json) VALUES
            ('iban_present', 'IBAN aanwezig', 'Document bevat minimaal één IBAN nummer', 'boolean', 'builtin', 'builtin', NULL),
            ('date_count', 'Aantal datums', 'Aantal datums (DD-MM-YYYY formaat) in document', 'count', 'builtin', 'builtin', NULL),
            ('amount_count', 'Aantal bedragen', 'Aantal geldbedragen (€X.XXX,XX formaat) in document', 'count', 'builtin', 'builtin', NULL),
            ('date_amount_row_count', 'Transactieregels', 'Aantal regels met zowel datum als bedrag', 'count', 'builtin', 'builtin', NULL),
            ('line_count', 'Aantal regels', 'Aantal niet-lege regels in document', 'count', 'builtin', 'builtin', NULL),
            ('token_count', 'Aantal woorden', 'Aantal woorden (whitespace-gescheiden tokens)', 'count', 'builtin', 'builtin', NULL)
        """)


def downgrade() -> None:
    op.drop_table('classification_signals')
    op.drop_column('document_types', 'classification_policy_json')
