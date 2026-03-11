from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class IngestionReportModel(BaseModel):
    total_rows: int
    success_rows: int
    failed_rows: int
    errors: List[str]


class ArtifactIngestionReportModel(BaseModel):
    total_rows: int
    success_rows: int
    failed_rows: int
    errors: List[str]


class TrainConfigModel(BaseModel):
    epochs: int = 3
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 1234
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    split_strategy: str = "random"
    corpus_name: str = "default"
    threshold_policy_version: str = "v1"
    enforce_thresholds: bool = False


class TrainRequest(BaseModel):
    config: Optional[TrainConfigModel] = None


class TrainResponse(BaseModel):
    run_id: str
    metrics: Dict[str, float]


class RunInfo(BaseModel):
    run_id: str
    created_at: datetime
    config: Dict[str, object]
    metrics: Dict[str, float]
    model_sha256: str
    data_manifest: Dict[str, object]
    status: str
    logs_path: str


class PredictRequest(BaseModel):
    entity_ids: List[str] = Field(..., description="Entity IDs to predict for")
    run_id: Optional[str] = Field(None, description="Specific run to use; latest if omitted")
    explanations: bool = Field(True, description="Whether to compute explanations")


class AttentionExplanation(BaseModel):
    token_weights: List[float]


class ArtifactAttribution(BaseModel):
    sha256: str
    contribution: float


class EntityExplanation(BaseModel):
    fused_attribution: List[float]
    attention: AttentionExplanation
    artifact_attributions: List[ArtifactAttribution]


class EntityPrediction(BaseModel):
    entity_id: str
    regression: float
    probability: float
    ranking_score: float
    embedding: List[float]
    narrative: Optional[str] = None
    explanation: Optional[EntityExplanation] = None


class PredictResponse(BaseModel):
    run_id: str
    predictions: List[EntityPrediction]
