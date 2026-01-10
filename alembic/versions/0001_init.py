from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "entities",
        sa.Column("entity_id", sa.String(), primary_key=True),
        sa.Column("attributes", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "events",
        sa.Column("event_id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("entity_id", sa.String(), sa.ForeignKey("entities.entity_id"), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("event_value", sa.Float(), nullable=False),
        sa.Column("event_metadata", sa.JSON(), nullable=False),
    )

    op.create_table(
        "interactions",
        sa.Column("interaction_id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column(
            "src_entity_id",
            sa.String(),
            sa.ForeignKey("entities.entity_id"),
            nullable=False,
        ),
        sa.Column(
            "dst_entity_id",
            sa.String(),
            sa.ForeignKey("entities.entity_id"),
            nullable=False,
        ),
        sa.Column("interaction_type", sa.String(), nullable=False),
        sa.Column("interaction_value", sa.Float(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
    )

    op.create_table(
        "artifacts",
        sa.Column("artifact_id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("entity_id", sa.String(), sa.ForeignKey("entities.entity_id"), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.Column(
            "artifact_type",
            sa.Enum("image", "video", "audio", name="artifact_type"),
            nullable=False,
        ),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(), nullable=False, unique=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column(
            "feature_status",
            sa.Enum("pending", "done", "failed", name="feature_status"),
            nullable=False,
        ),
        sa.Column("feature_dim", sa.Integer(), nullable=True),
        sa.Column("feature_version_hash", sa.String(), nullable=True),
    )

    op.create_table(
        "model_runs",
        sa.Column("run_id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("model_sha256", sa.String(), nullable=False),
        sa.Column("data_manifest", sa.JSON(), nullable=False),
        sa.Column("status", sa.Enum("success", "failed", name="run_status"), nullable=False),
        sa.Column("logs_path", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("model_runs")
    op.drop_table("artifacts")
    op.drop_table("interactions")
    op.drop_table("events")
    op.drop_table("entities")

    run_status = sa.Enum("success", "failed", name="run_status")
    feature_status = sa.Enum("pending", "done", "failed", name="feature_status")
    artifact_type = sa.Enum("image", "video", "audio", name="artifact_type")

    run_status.drop(op.get_bind(), checkfirst=True)
    feature_status.drop(op.get_bind(), checkfirst=True)
    artifact_type.drop(op.get_bind(), checkfirst=True)
