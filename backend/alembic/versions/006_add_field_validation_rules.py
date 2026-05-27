"""Add validation_rules column to document_type_fields

Revision ID: 006
Revises: 005
Create Date: 2026-05-27
"""
from alembic import op
import sqlalchemy as sa


revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [c["name"] for c in inspector.get_columns("document_type_fields")]
    if "validation_rules" not in columns:
        op.add_column(
            "document_type_fields",
            sa.Column("validation_rules", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [c["name"] for c in inspector.get_columns("document_type_fields")]
    if "validation_rules" in columns:
        op.drop_column("document_type_fields", "validation_rules")
