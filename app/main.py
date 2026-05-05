from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.logging import configure_application_logging
from app.core.rules import get_rule_registry
from app.db.session import engine
from app.schema_contract import validate_schema_contract


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if settings.effective_herman_db_canonical_mode:
        validate_schema_contract(
            engine=engine,
            version_table=settings.herman_db_version_table,
            allowed_revisions=settings.herman_db_allowed_revisions,
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
