from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.core.config import get_settings
from app.db.session import get_db
from app.main import create_app
from app.models.profile import FinalProfile
from app.models.prompt_score import ConversationPromptScore
from app.models.request_log import PromptTransformRequest


SQLALCHEMY_DATABASE_URL = "sqlite://"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    settings = get_settings()
    original_require_service_auth = settings.require_service_auth
    original_prompt_transformer_api_key = settings.prompt_transformer_api_key
    original_allowed_client_ids_raw = settings.allowed_client_ids_raw

    settings.require_service_auth = True
    settings.prompt_transformer_api_key = "test-transformer-key"
    settings.allowed_client_ids_raw = "hermanprompt,synthreo"

    def override_get_db() -> Generator[Session, None, None]:
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    settings.require_service_auth = original_require_service_auth
    settings.prompt_transformer_api_key = original_prompt_transformer_api_key
    settings.allowed_client_ids_raw = original_allowed_client_ids_raw
