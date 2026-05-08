from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
import logging
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.logging import configure_application_logging
from app.core.rules import get_rule_registry
from app.services.llm_provider_profiles import LlmProviderProfileService
from app.services.llm_model_catalog import LlmModelCatalogService
from app.db.session import engine
from app.schema_contract import SchemaContractError, validate_schema_contract


logger = logging.getLogger("prompt_transformer.main")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.effective_herman_db_canonical_mode:
        try:
            validate_schema_contract(
                engine=engine,
                version_table=settings.herman_db_version_table,
                allowed_revisions=settings.herman_db_allowed_revisions,
            )
        except SchemaContractError:
            logger.warning(
                "Schema contract validation failed during startup, continuing with shared DB ownership model.",
                exc_info=True,
            )
    profile_service = LlmProviderProfileService()
    LlmModelCatalogService().discover_and_register()

    for provider in profile_service.list_supported_providers():
        logger.info(
            "LLM model whitelist loaded for provider '%s': %s",
            provider,
            ", ".join(profile_service.list_supported_models(provider)),
        )
        try:
            resolved_default = profile_service.resolve_default_model(provider)
            logger.info(
                "LLM model default for provider '%s' is '%s'.",
                provider,
                resolved_default,
            )
        except Exception as exc:  # pragma: no cover - defensive startup path
            logger.warning(
                "Unable to resolve default model for provider '%s' during startup: %s",
                provider,
                exc,
            )
    get_rule_registry()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_application_logging(settings.log_level)
    app = FastAPI(
        title="Prompt Transformer",
        version="0.1.0",
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url="/redoc" if settings.app_env != "production" else None,
        lifespan=lifespan,
    )
    app.include_router(api_router)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"detail": jsonable_encoder(exc.errors())},
        )

    return app


app = create_app()
