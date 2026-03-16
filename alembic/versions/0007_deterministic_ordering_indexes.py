from __future__ import annotations

from alembic import op

revision = "0007_det_order_indexes"
down_revision = "0006_feature_and_batch_tasks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Align indexes with deterministic ordering and high-volume query paths.
    op.create_index(
        "ix_entities_created_entity_id",
        "entities",
        ["created_at", "entity_id"],
    )
    op.create_index(
        "ix_events_entity_timestamp_event_id",
        "events",
        ["entity_id", "timestamp", "event_id"],
    )
    op.create_index(
        "ix_interactions_src_dst_timestamp_interaction_id",
        "interactions",
        ["src_entity_id", "dst_entity_id", "timestamp", "interaction_id"],
    )
    op.create_index(
        "ix_artifacts_entity_artifact_id",
        "artifacts",
        ["entity_id", "artifact_id"],
    )
    op.create_index(
        "ix_artifacts_feature_status_artifact_id",
        "artifacts",
        ["feature_status", "artifact_id"],
    )
    op.create_index(
        "ix_model_runs_created_run_id",
        "model_runs",
        ["created_at", "run_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_model_runs_created_run_id", table_name="model_runs")
    op.drop_index("ix_artifacts_feature_status_artifact_id", table_name="artifacts")
    op.drop_index("ix_artifacts_entity_artifact_id", table_name="artifacts")
    op.drop_index("ix_interactions_src_dst_timestamp_interaction_id", table_name="interactions")
    op.drop_index("ix_events_entity_timestamp_event_id", table_name="events")
    op.drop_index("ix_entities_created_entity_id", table_name="entities")
