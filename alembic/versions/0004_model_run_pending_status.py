from __future__ import annotations

from alembic import op

revision = "0004_model_run_pending_status"
down_revision = "0003_agent_audit_and_lineage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TYPE run_status ADD VALUE IF NOT EXISTS 'pending'")


def downgrade() -> None:
    # PostgreSQL enums do not support dropping a single value safely in place.
    # This migration is intentionally irreversible.
    return
