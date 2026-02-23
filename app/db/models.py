from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
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

    __table_args__ = (
        UniqueConstraint("model_sha256", "run_id", name="uq_modelrun_modelsha_runid"),
    )
