"""Seed taxatierapport document type and fields

Revision ID: 014
Revises: 013
Create Date: 2026-05-27
"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime, timezone

revision = '014'
down_revision = '013'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    now = datetime.now(timezone.utc)

    # Insert document type if not exists
    existing = conn.execute(
        sa.text("SELECT id FROM document_types WHERE slug = 'taxatierapport'")
    ).fetchone()

    if existing:
        return

    conn.execute(
        sa.text("""
            INSERT INTO document_types (name, slug, description, created_at, updated_at)
            VALUES (:name, :slug, :description, :created_at, :updated_at)
        """),
        {
            "name": "Taxatierapport",
            "slug": "taxatierapport",
            "description": "Officieel taxatierapport van een onroerend goed object",
            "created_at": now,
            "updated_at": now,
        }
    )

    doc_type_id = conn.execute(
        sa.text("SELECT id FROM document_types WHERE slug = 'taxatierapport'")
    ).fetchone()[0]

    fields = [
        ("bouwjaar",            "Bouwjaar",           "number", False),
        ("energielabel",        "Energielabel",        "string", False),
        ("exploitatielasten",   "Exploitatielasten",   "money",  False),
        ("locatie_beoordeling", "Locatie beoordeling", "string", False),
        ("oppervlakte_m2",      "Oppervlakte (m2)",    "number", False),
        ("vastgoedtype",        "vastgoedtype",         "string", False),
    ]

    for key, label, field_type, required in fields:
        existing_field = conn.execute(
            sa.text("SELECT id FROM document_type_fields WHERE document_type_id = :dtid AND `key` = :key"),
            {"dtid": doc_type_id, "key": key}
        ).fetchone()
        if not existing_field:
            conn.execute(
                sa.text("""
                    INSERT INTO document_type_fields
                    (document_type_id, `key`, label, field_type, required, created_at, updated_at)
                    VALUES (:dtid, :key, :label, :field_type, :required, :created_at, :updated_at)
                """),
                {
                    "dtid": doc_type_id,
                    "key": key,
                    "label": label,
                    "field_type": field_type,
                    "required": required,
                    "created_at": now,
                    "updated_at": now,
                }
            )


def downgrade():
    conn = op.get_bind()
    row = conn.execute(
        sa.text("SELECT id FROM document_types WHERE slug = 'taxatierapport'")
    ).fetchone()
    if row:
        conn.execute(
            sa.text("DELETE FROM document_type_fields WHERE document_type_id = :id"),
            {"id": row[0]}
        )
        conn.execute(
            sa.text("DELETE FROM document_types WHERE id = :id"),
            {"id": row[0]}
        )
