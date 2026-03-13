from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import (
    event as sqlalchemy_event,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Entity(Base):
    __tablename__ = "entities"

    entity_id = Column(String, primary_key=True)
    attributes = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    events = relationship("Event", back_populates="entity")
    artifacts = relationship("Artifact", back_populates="entity")


class Event(Base):
    __tablename__ = "events"

    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, nullable=False)
    entity_id = Column(String, ForeignKey("entities.entity_id"), nullable=False)
    event_type = Column(String, nullable=False)
    event_value = Column(Float, nullable=False)
    event_metadata = Column(JSON, nullable=False, default=dict)

    entity = relationship("Entity", back_populates="events")


class Interaction(Base):
    __tablename__ = "interactions"

    interaction_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, nullable=False)
    src_entity_id = Column(String, ForeignKey("entities.entity_id"), nullable=False)
    dst_entity_id = Column(String, ForeignKey("entities.entity_id"), nullable=False)
    interaction_type = Column(String, nullable=False)
    interaction_value = Column(Float, nullable=False)
    meta = Column("metadata", JSON, nullable=False, default=dict)

    src_entity = relationship("Entity", foreign_keys=[src_entity_id])
    dst_entity = relationship("Entity", foreign_keys=[dst_entity_id])


ArtifactType = Enum("image", "video", "audio", name="artifact_type")
FeatureStatus = Enum("pending", "done", "failed", name="feature_status")
RunStatus = Enum("success", "failed", name="run_status")
AgentRunStatus = Enum(
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
AgentStepStatus = Enum(
    "pending",
    "running",
    "success",
    "failed",
    "skipped",
    name="agent_step_status",
)


class Artifact(Base):
    __tablename__ = "artifacts"

    artifact_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id = Column(String, ForeignKey("entities.entity_id"), nullable=True)
    timestamp = Column(DateTime, nullable=True)
    artifact_type = Column(ArtifactType, nullable=False)
    file_path = Column(Text, nullable=False)
    sha256 = Column(String, nullable=False, unique=True)
    meta = Column("metadata", JSON, nullable=False, default=dict)
    feature_status = Column(FeatureStatus, nullable=False, default="pending")
    feature_dim = Column(Integer, nullable=True)
    feature_version_hash = Column(String, nullable=True)

    entity = relationship("Entity", back_populates="artifacts")


class ModelRun(Base):
    __tablename__ = "model_runs"

    run_id = Column(String, primary_key=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    config = Column(JSON, nullable=False)
    metrics = Column(JSON, nullable=False)
    model_sha256 = Column(String, nullable=False)
    data_manifest = Column(JSON, nullable=False)
    status = Column(RunStatus, nullable=False)
    logs_path = Column(Text, nullable=False)
    created_by_agent_run_id = Column(String, ForeignKey("agent_runs.agent_run_id"), nullable=True)

    __table_args__ = (
        UniqueConstraint("model_sha256", "run_id", name="uq_modelrun_modelsha_runid"),
    )


class AgentRun(Base):
    __tablename__ = "agent_runs"

    agent_run_id = Column(String, primary_key=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    goal = Column(Text, nullable=False)
    status = Column(AgentRunStatus, nullable=False, default="pending")
    plan = Column(JSON, nullable=False, default=list)
    current_step_index = Column(Integer, nullable=False, default=0)
    max_steps = Column(Integer, nullable=False, default=8)
    step_retries = Column(Integer, nullable=False, default=1)
    require_approval = Column(Boolean, nullable=False, default=True)
    run_context = Column(JSON, nullable=False, default=dict)
    metrics = Column(JSON, nullable=False, default=dict)
    last_error = Column(Text, nullable=True)

    steps = relationship("AgentStep", back_populates="agent_run", cascade="all, delete-orphan")


class AgentStep(Base):
    __tablename__ = "agent_steps"

    step_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_run_id = Column(String, ForeignKey("agent_runs.agent_run_id"), nullable=False)
    step_index = Column(Integer, nullable=False)
    tool_name = Column(String, nullable=False)
    arguments = Column(JSON, nullable=False, default=dict)
    status = Column(AgentStepStatus, nullable=False, default="pending")
    retry_count = Column(Integer, nullable=False, default=0)
    output = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    agent_run = relationship("AgentRun", back_populates="steps")

    __table_args__ = (
        UniqueConstraint("agent_run_id", "step_index", name="uq_agent_step_run_index"),
    )


class AgentAuditEvent(Base):
    __tablename__ = "agent_audit_events"

    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_run_id = Column(String, ForeignKey("agent_runs.agent_run_id"), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    event_type = Column(String, nullable=False)
    actor = Column(String, nullable=False, default="system")
    details = Column(JSON, nullable=False, default=dict)

    agent_run = relationship("AgentRun")


@sqlalchemy_event.listens_for(AgentAuditEvent, "before_update")
def _prevent_agent_audit_event_update(*_args: object, **_kwargs: object) -> None:
    raise ValueError("Agent audit events are immutable and cannot be updated")


@sqlalchemy_event.listens_for(AgentAuditEvent, "before_delete")
def _prevent_agent_audit_event_delete(*_args: object, **_kwargs: object) -> None:
    raise ValueError("Agent audit events are immutable and cannot be deleted")
