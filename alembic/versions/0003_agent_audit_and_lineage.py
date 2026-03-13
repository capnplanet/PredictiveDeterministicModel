from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0003_agent_audit_and_lineage"
down_revision = "0002_agent_runtime"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("model_runs", sa.Column("created_by_agent_run_id", sa.String(), nullable=True))
    op.create_foreign_key(
        "fk_model_runs_created_by_agent_run_id",
        "model_runs",
        "agent_runs",
        ["created_by_agent_run_id"],
        ["agent_run_id"],
    )

    op.create_table(
        "agent_audit_events",
        sa.Column("event_id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("agent_run_id", sa.String(), sa.ForeignKey("agent_runs.agent_run_id"), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("actor", sa.String(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("agent_audit_events")
    op.drop_constraint("fk_model_runs_created_by_agent_run_id", "model_runs", type_="foreignkey")
    op.drop_column("model_runs", "created_by_agent_run_id")
