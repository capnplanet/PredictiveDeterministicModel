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

    class Config:
        env_prefix = ""
        env_file = ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
