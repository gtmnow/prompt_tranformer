from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str | None = Field(default=None, alias="DATABASE_URL")
    app_env: str = "development"
    log_level: str = "INFO"
    port: int = 8000
    enable_request_logging: bool = False
    enable_transform_timing_logs: bool = True
    enable_profile_cache: bool = False
    profile_cache_ttl_seconds: float = 300.0
    enable_async_score_persistence: bool = False
    score_persistence_debounce_seconds: float = 0.5
    score_persistence_workers: int = 4
    railway_auto_migrate: bool = True
    railway_seed_on_start: bool = False
    herman_db_canonical_mode: bool = False
    herman_db_version_table: str = "alembic_version"
    herman_db_allowed_revisions_raw: str = Field(
        default="20260504_0006,20260504_0007,20260504_0008,20260504_0009,20260505_0010,20260505_0011,20260505_0012,20260505_0013,20260505_0014,20260505_0015,20260506_0016",
        validation_alias=AliasChoices("HERMAN_DB_ALLOWED_REVISIONS", "HERMAN_DB_ALLOWED_REVISIONS_RAW"),
    )
    host: str = "0.0.0.0"
    require_service_auth: bool = False
    prompt_transformer_api_key: str = ""
    allowed_client_ids_raw: str = Field(default="hermanprompt", alias="ALLOWED_CLIENT_IDS")
    shared_secret_vault_master_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "HERMAN_SHARED_SECRET_VAULT_MASTER_KEY",
            "HERMAN_RUNTIME_SECRET_VAULT_MASTER_KEY",
            "HERMAN_ADMIN_SECRET_VAULT_MASTER_KEY",
        ),
    )
    shared_secret_vault_local_key_path: str = "./data/.secret_vault.key"
    structure_evaluator_enabled: bool = False
    structure_evaluator_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("STRUCTURE_EVALUATOR_API_KEY", "OPENAI_API_KEY"),
    )
    openai_api_key: str = Field(default="", validation_alias=AliasChoices("OPENAI_API_KEY"))
    xai_api_key: str = Field(default="", validation_alias=AliasChoices("XAI_API_KEY"))
    azure_openai_api_key: str = Field(default="", validation_alias=AliasChoices("AZURE_OPENAI_API_KEY"))
    anthropic_api_key: str = Field(default="", validation_alias=AliasChoices("ANTHROPIC_API_KEY"))
    structure_evaluator_base_url: str = "https://api.openai.com/v1"
    structure_evaluator_model: str = "gpt-4.1-mini"
    structure_evaluator_timeout_seconds: float = 15.0
    db_pool_size: int = 20
    db_max_overflow: int = 40
    db_pool_timeout_seconds: float = 15.0
    enable_reference_retrieval: bool = True
    reference_retrieval_min_query_terms: int = 4
    reference_retrieval_min_score: float = 0.08
    reference_context_max_sources: int = 3
    reference_context_max_words: int = 220
    reference_context_max_words_per_source: int = 80
    final_response_max_output_tokens: int = 1000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def allowed_client_ids(self) -> set[str]:
        return {client_id.strip() for client_id in self.allowed_client_ids_raw.split(",") if client_id.strip()}

    @property
    def herman_db_allowed_revisions(self) -> set[str]:
        return {
            revision.strip()
            for revision in self.herman_db_allowed_revisions_raw.split(",")
            if revision.strip()
        }

    @property
    def effective_herman_db_canonical_mode(self) -> bool:
        if self.herman_db_canonical_mode:
            return True
        return bool(self.database_url) and not self.database_url.startswith("sqlite") and self.app_env.lower() != "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
