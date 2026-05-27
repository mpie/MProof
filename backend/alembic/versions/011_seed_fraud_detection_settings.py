"""Seed fraud detection settings into app_settings

Revision ID: 011
Revises: 010
Create Date: 2026-05-27
"""
from alembic import op
import sqlalchemy as sa

revision = '011'
down_revision = '010'
branch_labels = None
depends_on = None

_DEFAULTS = [
    ('ela_enabled',          'false', 'Error Level Analysis aan/uit (JPEG manipulatie detectie)'),
    ('exif_enabled',         'false', 'EXIF analyse aan/uit (editing software detectie)'),
    ('ela_min_size',         '150',   'Minimale afbeeldingsgrootte (px) voor ELA analyse'),
    ('ela_quality',          '95',    'JPEG kwaliteit voor ELA re-compressie (85-99)'),
    ('ela_allow_non_jpeg',   'false', 'ELA ook uitvoeren op niet-JPEG afbeeldingen'),
    ('ela_scale_for_heatmap','77',    'Schaalfactor voor ELA heatmap visualisatie'),
]


def upgrade():
    for key, value, description in _DEFAULTS:
        op.execute(sa.text(
            "INSERT INTO app_settings (`key`, value, description, created_at, updated_at) "
            "VALUES (:k, :v, :d, NOW(), NOW()) "
            "ON DUPLICATE KEY UPDATE updated_at = updated_at"
        ).bindparams(k=key, v=value, d=description))


def downgrade():
    for key, _, _ in _DEFAULTS:
        op.execute(sa.text(
            "DELETE FROM app_settings WHERE `key` = :k"
        ).bindparams(k=key))
