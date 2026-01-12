"""Add skip_markers table

Revision ID: 002_skip_markers
Revises: 001_initial
Create Date: 2026-01-10

"""
from alembic import op
import sqlalchemy as sa

revision = '002_skip_markers'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create skip_markers table
    op.create_table(
        'skip_markers',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('pattern', sa.String(length=500), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_regex', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(datetime(\'now\'))'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(datetime(\'now\'))'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Insert default skip markers
    op.execute("""
        INSERT INTO skip_markers (pattern, description, is_regex, is_active)
        VALUES
            ('Algemene Voorwaarden', 'Skip content after general terms section', 0, 1),
            ('Terms and Conditions', 'Skip content after T&C section (English)', 0, 1),
            ('Terms & Conditions', 'Skip content after T&C section (English alt)', 0, 1),
            ('Voorwaarden en Condities', 'Skip content after conditions section (Dutch)', 0, 1),
            ('Dit is een automatisch gegenereerd', 'Skip auto-generated footer content', 0, 1),
            ('This is an automatically generated', 'Skip auto-generated footer content (English)', 0, 1),
            ('Pagina [0-9]+ van [0-9]+', 'Skip page number patterns (regex)', 1, 0)
    """)


def downgrade() -> None:
    op.drop_table('skip_markers')
