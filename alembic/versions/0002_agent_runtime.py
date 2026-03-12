from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002_agent_runtime"
down_revision = "0001_init"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("agent_run_id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "planning",
                "awaiting_approval",
                "executing",
                "paused",
                "completed",
                "failed",
                "aborted",
                name="agent_run_status",
            ),
            nullable=False,
        ),
        sa.Column("plan", sa.JSON(), nullable=False),
        sa.Column("current_step_index", sa.Integer(), nullable=False),
        sa.Column("max_steps", sa.Integer(), nullable=False),
        sa.Column("step_retries", sa.Integer(), nullable=False),
        sa.Column("require_approval", sa.Boolean(), nullable=False),
        sa.Column("run_context", sa.JSON(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
    )

    op.create_table(
        "agent_steps",
        sa.Column("step_id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("agent_run_id", sa.String(), sa.ForeignKey("agent_runs.agent_run_id"), nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=False),
        sa.Column("tool_name", sa.String(), nullable=False),
        sa.Column("arguments", sa.JSON(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "running",
                "success",
                "failed",
                "skipped",
                name="agent_step_status",
            ),
            nullable=False,
        ),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("output", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("agent_run_id", "step_index", name="uq_agent_step_run_index"),
    )


def downgrade() -> None:
    op.drop_table("agent_steps")
    op.drop_table("agent_runs")

    agent_step_status = sa.Enum(
        "pending",
        "running",
        "success",
        "failed",
        "skipped",
        name="agent_step_status",
    )
    agent_run_status = sa.Enum(
        "pending",
        "planning",
        "awaiting_approval",
        "executing",
        "paused",
        "completed",
        "failed",
        "aborted",
        name="agent_run_status",
    )

    agent_step_status.drop(op.get_bind(), checkfirst=True)
    agent_run_status.drop(op.get_bind(), checkfirst=True)
