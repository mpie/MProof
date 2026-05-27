"""Move LLM/vLLM settings from .env to app_settings table

Revision ID: 010
Revises: 009
Create Date: 2026-05-27
"""
from alembic import op
import sqlalchemy as sa

revision = '010'
down_revision = '009'
branch_labels = None
depends_on = None

_DEFAULTS = [
    ('ollama_base_url',    'http://localhost:11434',                          'Ollama base URL'),
    ('ollama_model',       'mistral:7b',                                      'Ollama model name'),
    ('ollama_timeout',     '60.0',                                            'Ollama request timeout (seconds)'),
    ('ollama_max_retries', '2',                                               'Ollama max retry attempts'),
    ('ollama_max_tokens',  '8192',                                            'Ollama max tokens for response'),
    ('vllm_base_url',      'http://localhost:8000',                           'vLLM base URL'),
    ('vllm_model',         'mistral:latest',                                  'vLLM model name'),
    ('vllm_timeout',       '60.0',                                            'vLLM request timeout (seconds)'),
    ('vllm_max_retries',   '2',                                               'vLLM max retry attempts'),
    ('vllm_max_tokens',    '2048',                                            'vLLM max tokens for response'),
]


def upgrade():
    for key, value, description in _DEFAULTS:
        op.execute(sa.text(
            "INSERT INTO app_settings (`key`, value, description, created_at, updated_at) "
            "VALUES (:k, :v, :d, NOW(), NOW()) "
            "ON DUPLICATE KEY UPDATE updated_at = updated_at"
        ).bindparams(k=key, v=value, d=description))


def downgrade():
    keys = [k for k, _, _ in _DEFAULTS]
    for key in keys:
        op.execute(sa.text(
            "DELETE FROM app_settings WHERE `key` = :k"
        ).bindparams(k=key))
