from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

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
    narrative_mode: Literal["template", "llm", "both"] = Field(
        "template",
        description=(
            "Narrative output mode. template keeps deterministic text; "
            "llm requests long-form narrative; both includes both."
        ),
    )


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
    narrative_template: Optional[str] = None
    narrative_long: Optional[str] = None
    narrative_source: Optional[str] = None
    explanation: Optional[EntityExplanation] = None


class PredictResponse(BaseModel):
    run_id: str
    predictions: List[EntityPrediction]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language query")
    run_id: Optional[str] = Field(None, description="Specific run to use; latest if omitted")
    limit: int = Field(5, ge=1, le=25)


class QueryResult(BaseModel):
    entity_id: str
    regression: float
    probability: float
    ranking_score: float
    narrative: Optional[str] = None


class QueryResponse(BaseModel):
    run_id: str
    query: str
    interpreted_as: str
    llm_used: bool
    results: List[QueryResult]


class DemoPreloadRequest(BaseModel):
    profile: Literal["small", "medium"] = "small"
    reset_existing: bool = True
    extract_features: bool = True
    train_model: bool = True


class DemoPreloadTrainingSummary(BaseModel):
    run_id: str
    metrics: Dict[str, float]


class DemoPreloadResponse(BaseModel):
    profile: str
    output_dir: str
    entities: IngestionReportModel
    events: IngestionReportModel
    interactions: IngestionReportModel
    artifacts_manifest: ArtifactIngestionReportModel
    single_artifact: Dict[str, str]
    features: Dict[str, int]
    training: Optional[DemoPreloadTrainingSummary] = None
    sample_entity_ids: List[str] = []


class AgentGoalRequest(BaseModel):
    goal: str = Field(..., min_length=5, max_length=500)
    context: Dict[str, Any] = Field(default_factory=dict)


class AgentStepPlan(BaseModel):
    step_index: int
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "running", "success", "failed", "skipped"] = "pending"
    retry_count: int = 0
    output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class AgentRunResponse(BaseModel):
    agent_run_id: str
    created_at: datetime
    updated_at: datetime
    goal: str
    status: Literal[
        "pending",
        "planning",
        "awaiting_approval",
        "executing",
        "paused",
        "completed",
        "failed",
        "aborted",
    ]
    current_step_index: int
    max_steps: int
    step_retries: int
    require_approval: bool
    context: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    plan: List[AgentStepPlan] = Field(default_factory=list)
    last_error: Optional[str] = None


class AgentPlanResponse(BaseModel):
    agent_run_id: str
    status: str
    goal: str
    plan: List[AgentStepPlan]


class AgentPlanApprovalRequest(BaseModel):
    approved: bool
    reviewer_notes: Optional[str] = Field(default=None, max_length=500)


class AgentStepExecuteRequest(BaseModel):
    force_continue: bool = False


class AgentStepResult(BaseModel):
    agent_run_id: str
    step_id: str
    step_index: int
    tool_name: str
    status: Literal["pending", "running", "success", "failed", "skipped"]
    retry_count: int
    output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class AgentControlRequest(BaseModel):
    action: Literal["pause", "resume", "abort"]
    reason: Optional[str] = Field(default=None, max_length=500)


class AgentStatusResponse(BaseModel):
    agent_run_id: str
    status: str
    current_step_index: int
    total_steps: int
    completed_steps: int
    failed_steps: int
    goal: str
    updated_at: datetime
    last_error: Optional[str] = None


class AgentAuditEventResponse(BaseModel):
    event_id: str
    agent_run_id: str
    created_at: datetime
    event_type: str
    actor: str
    details: Dict[str, Any] = Field(default_factory=dict)
