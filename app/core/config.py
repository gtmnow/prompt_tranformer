from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./prompt_transformer.db"
    app_env: str = "development"
    log_level: str = "INFO"
    port: int = 8000
    enable_request_logging: bool = False
    railway_auto_migrate: bool = True
    railway_seed_on_start: bool = False
    host: str = "0.0.0.0"
    require_service_auth: bool = False
    prompt_transformer_api_key: str = ""
    allowed_client_ids_raw: str = Field(default="hermanprompt", alias="ALLOWED_CLIENT_IDS")
    structure_evaluator_enabled: bool = False
    structure_evaluator_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("STRUCTURE_EVALUATOR_API_KEY", "OPENAI_API_KEY"),
    )
    structure_evaluator_base_url: str = "https://api.openai.com/v1"
    structure_evaluator_model: str = "gpt-4.1-mini"
    structure_evaluator_timeout_seconds: float = 15.0
    db_pool_size: int = 20
    db_max_overflow: int = 40
    db_pool_timeout_seconds: float = 15.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def allowed_client_ids(self) -> set[str]:
        return {client_id.strip() for client_id in self.allowed_client_ids_raw.split(",") if client_id.strip()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
