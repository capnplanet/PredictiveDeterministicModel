from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0005_training_tasks"
down_revision = "0004_model_run_pending_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "training_tasks",
        sa.Column("task_id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column(
            "status",
            sa.Enum("pending", "running", "success", "failed", name="training_task_status"),
            nullable=False,
        ),
        sa.Column("queue_name", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=True, unique=True),
        sa.Column("request_payload", sa.JSON(), nullable=False),
        sa.Column("celery_task_id", sa.String(), nullable=True),
        sa.Column("run_id", sa.String(), sa.ForeignKey("model_runs.run_id"), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.create_index("ix_training_tasks_status_created", "training_tasks", ["status", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_training_tasks_status_created", table_name="training_tasks")
    op.drop_table("training_tasks")

    training_task_status = sa.Enum("pending", "running", "success", "failed", name="training_task_status")
    training_task_status.drop(op.get_bind(), checkfirst=True)
