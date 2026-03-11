from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Deterministic Multimodal Analytics"
    database_url: str = "postgresql+psycopg2://analytics:analytics@localhost:5432/analytics"
    artifacts_root: str = "artifacts"
    data_root: str = "data"
    llm_enabled: bool = False
    llm_provider: str = "huggingface"
    llm_endpoint_url: str = ""
    llm_api_token: str = ""
    llm_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    llm_timeout_seconds: float = 5.0
    llm_max_tokens: int = 500
    llm_temperature: float = 0.2

    class Config:
        env_prefix = ""
        env_file = ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
