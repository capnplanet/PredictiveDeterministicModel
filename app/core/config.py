from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Deterministic Multimodal Analytics"
    database_url: str = "postgresql+psycopg2://analytics:analytics@localhost:5432/analytics"
    artifacts_root: str = "artifacts"
    data_root: str = "data"
    llm_enabled: bool = Field(default=False, validation_alias=AliasChoices("llm_enabled", "LLM_ENABLED"))
    llm_provider: str = "huggingface"
    llm_endpoint_url: str = Field(
        default="",
        validation_alias=AliasChoices("llm_endpoint_url", "LLM_ENDPOINT_URL", "HF_ENDPOINT_URL", "hf_URL_Endpoint"),
    )
    llm_api_token: str = Field(
        default="",
        validation_alias=AliasChoices("llm_api_token", "LLM_API_TOKEN", "HF_TOKEN"),
    )
    llm_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    llm_timeout_seconds: float = 5.0
    llm_max_tokens: int = 500
    llm_temperature: float = 0.2
    agent_enabled: bool = Field(default=False, validation_alias=AliasChoices("agent_enabled", "AGENT_ENABLED"))
    agent_max_plan_steps: int = 8
    agent_max_step_retries: int = 1
    agent_step_timeout_seconds: float = 30.0
    agent_require_approval: bool = True
    agent_enforce_determinism: bool = Field(
        default=True,
        validation_alias=AliasChoices("agent_enforce_determinism", "AGENT_ENFORCE_DETERMINISM"),
    )
    celery_enabled: bool = Field(default=False, validation_alias=AliasChoices("celery_enabled", "CELERY_ENABLED"))
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias=AliasChoices("celery_broker_url", "CELERY_BROKER_URL"),
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/1",
        validation_alias=AliasChoices("celery_result_backend", "CELERY_RESULT_BACKEND"),
    )
    celery_task_always_eager: bool = Field(
        default=False,
        validation_alias=AliasChoices("celery_task_always_eager", "CELERY_TASK_ALWAYS_EAGER"),
    )
    queue_training_name: str = Field(
        default="training",
        validation_alias=AliasChoices("queue_training_name", "QUEUE_TRAINING_NAME"),
    )
    queue_feature_extraction_name: str = Field(
        default="extraction",
        validation_alias=AliasChoices("queue_feature_extraction_name", "QUEUE_FEATURE_EXTRACTION_NAME"),
    )
    queue_batch_inference_name: str = Field(
        default="batch_inference",
        validation_alias=AliasChoices("queue_batch_inference_name", "QUEUE_BATCH_INFERENCE_NAME"),
    )
    queue_training_max_concurrency: int = Field(
        default=2,
        validation_alias=AliasChoices("queue_training_max_concurrency", "QUEUE_TRAINING_MAX_CONCURRENCY"),
    )
    queue_feature_extraction_max_concurrency: int = Field(
        default=2,
        validation_alias=AliasChoices(
            "queue_feature_extraction_max_concurrency",
            "QUEUE_FEATURE_EXTRACTION_MAX_CONCURRENCY",
        ),
    )
    queue_batch_inference_max_concurrency: int = Field(
        default=2,
        validation_alias=AliasChoices(
            "queue_batch_inference_max_concurrency",
            "QUEUE_BATCH_INFERENCE_MAX_CONCURRENCY",
        ),
    )
    async_fallback_sync_execution: bool = Field(
        default=True,
        validation_alias=AliasChoices("async_fallback_sync_execution", "ASYNC_FALLBACK_SYNC_EXECUTION"),
    )

    class Config:
        env_prefix = ""
        env_file = ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
