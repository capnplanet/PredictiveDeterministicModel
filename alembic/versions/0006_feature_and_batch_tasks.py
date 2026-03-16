from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0006_feature_and_batch_tasks"
down_revision = "0005_training_tasks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    task_status = postgresql.ENUM(
        "pending",
        "running",
        "success",
        "failed",
        name="training_task_status",
        create_type=False,
    )

    op.create_table(
        "feature_extraction_tasks",
        sa.Column("task_id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("status", task_status, nullable=False),
        sa.Column("queue_name", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=True, unique=True),
        sa.Column("request_payload", sa.JSON(), nullable=False),
        sa.Column("celery_task_id", sa.String(), nullable=True),
        sa.Column("updated_artifacts", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_feature_extraction_tasks_status_created",
        "feature_extraction_tasks",
        ["status", "created_at"],
    )

    op.create_table(
        "batch_inference_tasks",
        sa.Column("task_id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("status", task_status, nullable=False),
        sa.Column("queue_name", sa.String(), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=True, unique=True),
        sa.Column("request_payload", sa.JSON(), nullable=False),
        sa.Column("celery_task_id", sa.String(), nullable=True),
        sa.Column("run_id", sa.String(), sa.ForeignKey("model_runs.run_id"), nullable=True),
        sa.Column("result_payload", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_batch_inference_tasks_status_created",
        "batch_inference_tasks",
        ["status", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_batch_inference_tasks_status_created", table_name="batch_inference_tasks")
    op.drop_table("batch_inference_tasks")

    op.drop_index("ix_feature_extraction_tasks_status_created", table_name="feature_extraction_tasks")
    op.drop_table("feature_extraction_tasks")
