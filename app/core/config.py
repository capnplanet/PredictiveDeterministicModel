from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Deterministic Multimodal Analytics"
    database_url: str = "postgresql+psycopg2://analytics:analytics@localhost:5432/analytics"
    artifacts_root: str = "artifacts"
    data_root: str = "data"

    class Config:
        env_prefix = ""
        env_file = ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
