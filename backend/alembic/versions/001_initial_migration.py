"""Initial migration

Revision ID: 001_initial
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = inspector.get_table_names()
    
    # Create subjects table (only if it doesn't exist)
    if 'subjects' not in existing_tables:
        op.create_table(
            'subjects',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('name', sa.String(length=255), nullable=False),
            sa.Column('name_normalized', sa.String(length=255), nullable=False),
            sa.Column('context', sa.Enum('person', 'company', 'dossier', 'other', name='contextenum'), nullable=False),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('(datetime(\'now\'))'), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(datetime(\'now\'))'), nullable=False),
            sa.PrimaryKeyConstraint('id'),
            sa.Index('ix_subjects_name', 'name'),
            sa.Index('ix_subjects_name_normalized', 'name_normalized'),
            sa.Index('ix_subjects_context', 'context')
        )

    # Create document_types table (only if it doesn't exist)
    if 'document_types' not in existing_tables:
        op.create_table(
            'document_types',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('name', sa.String(length=255), nullable=False),
            sa.Column('slug', sa.String(length=100), nullable=False),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('classification_hints', sa.Text(), nullable=True),
            sa.Column('extraction_prompt_preamble', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('(datetime(\'now\'))'), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(datetime(\'now\'))'), nullable=False),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('slug'),
            sa.Index('ix_document_types_slug', 'slug')
        )

    # Create document_type_fields table (only if it doesn't exist)
    if 'document_type_fields' not in existing_tables:
        op.create_table(
            'document_type_fields',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('document_type_id', sa.Integer(), nullable=False),
            sa.Column('key', sa.String(length=100), nullable=False),
            sa.Column('label', sa.String(length=255), nullable=False),
            sa.Column('field_type', sa.Enum('string', 'date', 'number', 'money', 'currency', 'iban', 'enum', name='fieldtypeenum'), nullable=False),
            sa.Column('required', sa.Boolean(), nullable=False),
            sa.Column('enum_values', sa.JSON(), nullable=True),
            sa.Column('regex', sa.String(length=500), nullable=True),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('examples', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('(datetime(\'now\'))'), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(datetime(\'now\'))'), nullable=False),
            sa.ForeignKeyConstraint(['document_type_id'], ['document_types.id'], ),
            sa.PrimaryKeyConstraint('id')
        )

    # Create documents table (only if it doesn't exist)
    if 'documents' not in existing_tables:
        op.create_table(
            'documents',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('subject_id', sa.Integer(), nullable=False),
            sa.Column('original_filename', sa.String(length=500), nullable=False),
            sa.Column('mime_type', sa.String(length=100), nullable=False),
            sa.Column('size_bytes', sa.Integer(), nullable=False),
            sa.Column('sha256', sa.String(length=64), nullable=False),
            sa.Column('status', sa.Enum('queued', 'processing', 'done', 'error', name='documentstatusenum'), nullable=False),
            sa.Column('progress', sa.Integer(), nullable=False),
            sa.Column('stage', sa.String(length=100), nullable=True),
            sa.Column('error_message', sa.Text(), nullable=True),
            sa.Column('doc_type_slug', sa.String(length=100), nullable=True),
            sa.Column('doc_type_confidence', sa.Float(), nullable=True),
            sa.Column('doc_type_rationale', sa.Text(), nullable=True),
            sa.Column('metadata_json', sa.JSON(), nullable=True),
            sa.Column('metadata_validation_json', sa.JSON(), nullable=True),
            sa.Column('metadata_evidence_json', sa.JSON(), nullable=True),
            sa.Column('risk_score', sa.Integer(), nullable=True),
            sa.Column('risk_signals_json', sa.JSON(), nullable=True),
            sa.Column('ocr_used', sa.Boolean(), nullable=False),
            sa.Column('ocr_quality', sa.String(length=20), nullable=True),
            sa.Column('created_at', sa.DateTime(), server_default=sa.text('(datetime(\'now\'))'), nullable=False),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(datetime(\'now\'))'), nullable=False),
            sa.ForeignKeyConstraint(['subject_id'], ['subjects.id'], ),
            sa.PrimaryKeyConstraint('id')
        )


def downgrade() -> None:
    op.drop_table('documents')
    op.drop_table('document_type_fields')
    op.drop_table('document_types')
    op.drop_table('subjects')