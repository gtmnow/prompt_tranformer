import httpx

from unittest.mock import patch

from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import SQLAlchemyError

from app.db.seed import PROFILE_ROWS
from app.models.profile import FinalProfile
from app.models.prompt_score import ConversationPromptScore
from app.models.request_log import PromptTransformRequest
from app.core.config import get_settings
from app.db.session import get_db
from app.models.rag import RagCollection, RagDocument, RagQuotaPolicy, RagRetrievalEvent
from app.services.guide_me_generation import GuideMeGenerationResult
from app.services.rag_retrieval_service import RagRetrievalResult
from app.services.rag_vectorizer import vectorize_text
from app.services.llm_types import NormalizedTokenUsage
from app.services.runtime_llm import RuntimeLlmConfig


AUTH_HEADERS = {
    "Authorization": "Bearer test-transformer-key",
    "X-Client-Id": "hermanprompt",
}


def _runtime_config(
    *,
    scoring_enabled: bool = True,
    provider: str = "openai",
    model: str = "gpt-4.1",
    endpoint_url: str | None = "https://api.openai.com/v1",
) -> RuntimeLlmConfig:
    return RuntimeLlmConfig(
        tenant_id="tenant_1",
        user_id_hash="user_1",
        provider=provider,
        model=model,
        endpoint_url=endpoint_url,
        api_key="test-key",
        transformation_enabled=True,
        scoring_enabled=scoring_enabled,
        credential_status="valid",
        source_kind="customer_managed",
    )


def _seed_final_profiles(client) -> None:
    db = next(client.app.dependency_overrides[get_db]())
    try:
        for row in PROFILE_ROWS:
            db.add(FinalProfile(**row))
        db.commit()
    finally:
        db.close()


def _update_profile(client, user_id: str, **overrides) -> None:
    db = next(client.app.dependency_overrides[get_db]())
    try:
        profile = db.get(FinalProfile, user_id)
        if profile is None:
            raise AssertionError(f"Missing profile for {user_id}")
        for key, value in overrides.items():
            setattr(profile, key, value)
        db.commit()
    finally:
        db.close()


def _sqlite_columns(client, table_name: str) -> dict[str, dict[str, object]]:
    db = next(client.app.dependency_overrides[get_db]())
    try:
        rows = db.execute(text(f"pragma table_info('{table_name}')")).mappings().all()
        return {str(row["name"]): dict(row) for row in rows}
    finally:
        db.close()


def test_metadata_created_rag_schema_matches_canonical_column_contract(client) -> None:
    quota_columns = _sqlite_columns(client, RagQuotaPolicy.__tablename__)
    collection_columns = _sqlite_columns(client, RagCollection.__tablename__)
    document_columns = _sqlite_columns(client, RagDocument.__tablename__)
    retrieval_columns = _sqlite_columns(client, RagRetrievalEvent.__tablename__)

    assert quota_columns["scope_target"]["type"].upper() == "VARCHAR(30)"
    assert collection_columns["tenant_id"]["notnull"] == 0
    assert document_columns["tenant_id"]["notnull"] == 0
    assert document_columns["status"]["type"].upper() == "VARCHAR(30)"
    assert retrieval_columns["tenant_id"]["notnull"] == 0
    assert retrieval_columns["score"]["type"].upper() in {"FLOAT", "REAL"}


def _seed_canonical_membership_schema(client) -> None:
    db = next(client.app.dependency_overrides[get_db]())
    try:
        db.execute(text("drop table if exists user_membership_profiles"))
        db.execute(text("drop table if exists user_tenant_membership"))
        db.execute(text("drop table if exists tenants"))
        db.execute(
            text(
                """
                create table tenants (
                    id varchar(36) primary key,
                    service_tier_definition_id varchar(36)
                )
                """
            )
        )
        db.execute(
            text(
                """
                create table user_tenant_membership (
                    id varchar(36) primary key,
                    user_id_hash varchar(255) not null,
                    tenant_id varchar(36) not null,
                    status varchar(32),
                    is_primary boolean
                )
                """
            )
        )
        db.execute(
            text(
                """
                create table user_membership_profiles (
                    id varchar(36) primary key,
                    tenant_membership_id varchar(36) not null,
                    title varchar(255),
                    initial_user_type integer
                )
                """
            )
        )
        db.execute(
            text(
                """
                insert into tenants (id, service_tier_definition_id)
                values ('tenant_1', 'tier_1')
                """
            )
        )
        db.execute(
            text(
                """
                insert into user_tenant_membership (id, user_id_hash, tenant_id, status, is_primary)
                values ('membership_1', 'user_1', 'tenant_1', 'active', true)
                """
            )
        )
        db.execute(
            text(
                """
                insert into user_membership_profiles (id, tenant_membership_id, title, initial_user_type)
                values ('profile_1', 'membership_1', 'Coach', 3)
                """
            )
        )
        db.execute(
            text(
                """
                insert into rag_quota_policies (
                    id,
                    policy_key,
                    scope_target,
                    service_tier_definition_id,
                    tenant_id,
                    user_type,
                    org_max_file_bytes,
                    user_max_file_bytes,
                    org_max_document_count,
                    user_max_document_count,
                    org_max_total_bytes,
                    user_max_total_bytes,
                    org_max_extracted_text_bytes,
                    user_max_extracted_text_bytes,
                    org_max_chunks_per_document,
                    user_max_chunks_per_document,
                    org_max_retrieved_chunks,
                    user_max_retrieved_chunks,
                    max_retrieved_chunks_total,
                    is_active
                ) values (
                    'policy_service_tier_default',
                    'service-tier-default',
                    'service_tier',
                    'tier_1',
                    null,
                    null,
                    1000,
                    2000,
                    10,
                    20,
                    10000,
                    20000,
                    5000,
                    6000,
                    7,
                    8,
                    4,
                    5,
                    9,
                    true
                )
                """
            )
        )
        db.execute(
            text(
                """
                insert into rag_quota_policies (
                    id,
                    policy_key,
                    scope_target,
                    service_tier_definition_id,
                    tenant_id,
                    user_type,
                    org_max_file_bytes,
                    user_max_file_bytes,
                    org_max_document_count,
                    user_max_document_count,
                    org_max_total_bytes,
                    user_max_total_bytes,
                    org_max_extracted_text_bytes,
                    user_max_extracted_text_bytes,
                    org_max_chunks_per_document,
                    user_max_chunks_per_document,
                    org_max_retrieved_chunks,
                    user_max_retrieved_chunks,
                    max_retrieved_chunks_total,
                    is_active
                ) values (
                    'policy_service_tier_user_3',
                    'service-tier-user-3',
                    'service_tier',
                    'tier_1',
                    null,
                    3,
                    1000,
                    2000,
                    10,
                    20,
                    10000,
                    20000,
                    5000,
                    6000,
                    7,
                    8,
                    4,
                    5,
                    9,
                    true
                )
                """
            )
        )
        db.commit()
    finally:
        db.close()


def _recreate_canonical_rag_schema(client) -> None:
    db = next(client.app.dependency_overrides[get_db]())
    try:
        for table_name in (
            "rag_retrieval_events",
            "rag_chunks",
            "rag_document_blobs",
            "rag_documents",
            "rag_collections",
            "rag_quota_policies",
        ):
            db.execute(text(f"drop table if exists {table_name}"))

        db.execute(
            text(
                """
                create table rag_quota_policies (
                    id varchar(36) primary key,
                    policy_key varchar(100) not null unique,
                    scope_target varchar(30) not null,
                    service_tier_definition_id varchar(36),
                    tenant_id varchar(36),
                    user_type integer,
                    org_max_file_bytes bigint not null,
                    user_max_file_bytes bigint not null,
                    org_max_document_count integer not null,
                    user_max_document_count integer not null,
                    org_max_total_bytes bigint not null,
                    user_max_total_bytes bigint not null,
                    org_max_extracted_text_bytes bigint not null,
                    user_max_extracted_text_bytes bigint not null,
                    org_max_chunks_per_document integer not null,
                    user_max_chunks_per_document integer not null,
                    org_max_retrieved_chunks integer not null,
                    user_max_retrieved_chunks integer not null,
                    max_retrieved_chunks_total integer not null,
                    is_active boolean not null default true,
                    created_at datetime not null default current_timestamp,
                    updated_at datetime not null default current_timestamp
                )
                """
            )
        )
        db.execute(
            text(
                """
                create table rag_collections (
                    id varchar(36) primary key,
                    scope_type varchar(20) not null,
                    tenant_id varchar(36),
                    user_id_hash varchar(255),
                    name varchar(200) not null,
                    is_active boolean not null default true,
                    retrieval_enabled boolean not null default true,
                    max_results integer,
                    created_at datetime not null default current_timestamp,
                    updated_at datetime not null default current_timestamp
                )
                """
            )
        )
        db.execute(
            text(
                """
                create table rag_documents (
                    id varchar(36) primary key,
                    collection_id varchar(36) not null,
                    scope_type varchar(20) not null,
                    tenant_id varchar(36),
                    user_id_hash varchar(255),
                    filename varchar(255) not null,
                    media_type varchar(120),
                    size_bytes bigint not null,
                    status varchar(30) not null default 'pending',
                    status_message text,
                    sha256 varchar(64) not null,
                    source_kind varchar(30) not null default 'database_blob',
                    uploaded_by_admin_user_id varchar(36),
                    uploaded_by_user_id_hash varchar(255),
                    uploaded_at datetime not null default current_timestamp,
                    processed_at datetime,
                    disabled_at datetime
                )
                """
            )
        )
        db.execute(
            text(
                """
                create table rag_document_blobs (
                    document_id varchar(36) primary key,
                    content_bytes blob not null
                )
                """
            )
        )
        db.execute(
            text(
                """
                create table rag_chunks (
                    id varchar(36) primary key,
                    document_id varchar(36) not null,
                    chunk_index integer not null,
                    chunk_text text not null,
                    token_count integer not null,
                    embedding_vector json not null,
                    embedding_model varchar(100) not null,
                    created_at datetime not null default current_timestamp,
                    unique(document_id, chunk_index)
                )
                """
            )
        )
        db.execute(
            text(
                """
                create table rag_retrieval_events (
                    id varchar(36) primary key,
                    conversation_id varchar(255) not null,
                    user_id_hash varchar(255) not null,
                    tenant_id varchar(36),
                    scope_type varchar(20) not null,
                    document_id varchar(36) not null,
                    chunk_id varchar(36) not null,
                    rank integer not null,
                    score real not null,
                    created_at datetime not null default current_timestamp
                )
                """
            )
        )
        db.commit()
    finally:
        db.close()


def test_transform_uses_db_profile(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_123",
            "conversation_id": "conv_123",
            "user_id_hash": "user_1",
            "raw_prompt": "Explain this concept simply",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == "conv_123"
    assert body["result_type"] == "transformed"
    assert body["task_type"] == "explanation"
    assert body["metadata"]["persona_source"] == "db_profile"
    assert "Start with the direct answer" in body["transformed_prompt"]


def test_transform_uses_summary_override(client) -> None:
    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_456",
            "conversation_id": "conv_456",
            "user_id_hash": "user_missing",
            "raw_prompt": "Summarize this article",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "summary_type": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == "conv_456"
    assert body["result_type"] == "transformed"
    assert body["task_type"] == "summarization"
    assert body["metadata"]["persona_source"] == "summary_override"


def test_transform_falls_back_to_generic_default(client) -> None:
    with patch(
        "app.services.transformer_engine.RuntimeLlmResolver.resolve",
        return_value=_runtime_config(model="unknown-model"),
    ):
        response = client.post(
            "/api/transform_prompt",
            headers=AUTH_HEADERS,
            json={
                "session_id": "sess_789",
                "conversation_id": "conv_789",
                "user_id_hash": "user_missing",
                "raw_prompt": "What should I do next?",
                "target_llm": {"provider": "openai", "model": "unknown-model"},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == "conv_789"
    assert body["result_type"] == "transformed"
    assert body["task_type"] == "unknown"
    assert body["metadata"]["persona_source"] == "generic_default"
    assert body["metadata"]["used_fallback_model"] is True
    assert body["metadata"]["resolved_model"] == "unknown-model"


def test_execute_chat_keeps_runtime_xai_model_and_reports_retrieval_metadata(client) -> None:
    _seed_final_profiles(client)

    with (
        patch(
            "app.services.transformer_engine.RuntimeLlmResolver.resolve",
            return_value=_runtime_config(
                provider="xai",
                model="grok-3-mini",
                endpoint_url="https://api.x.ai/v1",
            ),
        ),
        patch(
            "app.services.transformer_engine.RagRetrievalService.retrieve",
            return_value=RagRetrievalResult(
                assembled_references=[
                    {
                        "filename": "resume-notes.txt",
                        "chunk_text": "Candidate prefers concise scorecards and recruiter-facing summaries.",
                    }
                ],
                tenant_chunk_count=0,
                user_chunk_count=1,
                document_count=1,
            ),
        ),
        patch("app.services.final_response_service.httpx.Client") as httpx_client,
    ):
        mock_client = httpx_client.return_value.__enter__.return_value
        mock_client.post.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Here is the xAI response."}],
                }
            ]
        }

        response = client.post(
            "/api/chat/execute",
            headers=AUTH_HEADERS,
            json={
                "session_id": "sess_xai_chat",
                "conversation_id": "conv_xai_chat",
                "user_id_hash": "user_1",
                "raw_prompt": "Draft a concise recruiter summary.",
                "target_llm": {"provider": "xai", "model": "grok-3-mini"},
                "conversation_history": [],
                "attachments": [],
                "transform_enabled": True,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_text"] == "Here is the xAI response."
    assert body["metadata"]["resolved_provider"] == "xai"
    assert body["metadata"]["resolved_model"] == "grok-3-mini"
    assert body["metadata"]["used_fallback_model"] is True
    assert body["metadata"]["retrieval_used"] is True
    assert body["metadata"]["retrieval_scope_counts"] == {"tenant": 0, "user": 1}
    assert body["metadata"]["retrieval_document_count"] == 1

    request_json = mock_client.post.call_args.kwargs["json"]
    assert request_json["model"] == "grok-3-mini"
    assert "input" in request_json
    assert "messages" not in request_json
    assert request_json["tools"] == [{"type": "web_search"}]


def test_execute_chat_returns_504_when_final_response_provider_times_out(client) -> None:
    _seed_final_profiles(client)

    with (
        patch(
            "app.services.transformer_engine.RuntimeLlmResolver.resolve",
            return_value=_runtime_config(),
        ),
        patch(
            "app.services.transformer_engine.RagRetrievalService.retrieve",
            return_value=RagRetrievalResult(
                assembled_references=[],
                tenant_chunk_count=0,
                user_chunk_count=0,
                document_count=0,
            ),
        ),
        patch("app.services.final_response_service.httpx.Client") as httpx_client,
    ):
        mock_client = httpx_client.return_value.__enter__.return_value
        mock_client.post.side_effect = httpx.ReadTimeout(
            "The read operation timed out",
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        )

        response = client.post(
            "/api/chat/execute",
            headers=AUTH_HEADERS,
            json={
                "session_id": "sess_timeout",
                "conversation_id": "conv_timeout",
                "user_id_hash": "user_1",
                "raw_prompt": "Draft a concise recruiter summary.",
                "target_llm": {"provider": "openai", "model": "gpt-4.1"},
                "conversation_history": [],
                "attachments": [],
                "transform_enabled": True,
            },
        )

    assert response.status_code == 504
    assert response.json()["detail"] == "LLM provider timed out after 45 seconds."


def test_user_scope_retrieval_skips_unrelated_chunks_when_query_has_no_overlap(client) -> None:
    from app.models.rag import RagChunk, RagCollection, RagDocument
    from app.services.rag_retrieval_service import RagRetrievalService

    _seed_canonical_membership_schema(client)

    db = next(client.app.dependency_overrides[get_db]())
    try:
        collection = RagCollection(
            id="collection_user_1",
            scope_type="user",
            tenant_id="tenant_1",
            user_id_hash="user_1",
            name="Personal Context",
            is_active=True,
            retrieval_enabled=True,
            max_results=2,
        )
        document = RagDocument(
            id="document_user_1",
            collection_id=collection.id,
            scope_type="user",
            tenant_id="tenant_1",
            user_id_hash="user_1",
            filename="personal-notes.txt",
            media_type="text/plain",
            size_bytes=128,
            status="ready",
            status_message=None,
            sha256="abc123",
            source_kind="database_blob",
            uploaded_by_admin_user_id=None,
            uploaded_by_user_id_hash="user_1",
        )
        chunk = RagChunk(
            id="chunk_user_1",
            document_id=document.id,
            chunk_index=0,
            chunk_text="I prefer concise recruiter summaries and bullet-style responses.",
            token_count=9,
            embedding_vector=vectorize_text("I prefer concise recruiter summaries and bullet-style responses."),
        )
        db.add(collection)
        db.add(document)
        db.add(chunk)
        db.commit()

        result = RagRetrievalService(db).retrieve(
            tenant_id="tenant_1",
            user_id_hash="user_1",
            conversation_id="conv_no_overlap",
            raw_prompt="What time is the meeting tomorrow?",
            conversation_history=[],
        )

        assert result.user_chunk_count == 0
        assert result.document_count == 0
        assert result.assembled_references == []
    finally:
        db.close()


def test_user_scope_retrieval_falls_back_for_personal_context_queries(client) -> None:
    from app.models.rag import RagChunk, RagCollection, RagDocument
    from app.services.rag_retrieval_service import RagRetrievalService

    _seed_canonical_membership_schema(client)

    db = next(client.app.dependency_overrides[get_db]())
    try:
        collection = RagCollection(
            id="collection_user_resume_fallback",
            scope_type="user",
            tenant_id="tenant_1",
            user_id_hash="user_1",
            name="Personal Context",
            is_active=True,
            retrieval_enabled=True,
            max_results=2,
        )
        document = RagDocument(
            id="document_user_resume_fallback",
            collection_id=collection.id,
            scope_type="user",
            tenant_id="tenant_1",
            user_id_hash="user_1",
            filename="resume.txt",
            media_type="text/plain",
            size_bytes=256,
            status="ready",
            status_message=None,
            sha256="fallbackhash",
            source_kind="database_blob",
            uploaded_by_admin_user_id=None,
            uploaded_by_user_id_hash="user_1",
        )
        chunk = RagChunk(
            id="chunk_user_resume_fallback",
            document_id=document.id,
            chunk_index=0,
            chunk_text="I have led a marketing ops team that shipped two B2B launch campaigns.",
            token_count=14,
            embedding_vector=vectorize_text("I have led a marketing ops team that shipped two B2B launch campaigns."),
        )
        db.add(collection)
        db.add(document)
        db.add(chunk)
        db.commit()

        result = RagRetrievalService(db).retrieve(
            tenant_id="tenant_1",
            user_id_hash="user_1",
            conversation_id="conv_personal_fallback",
            raw_prompt="Can you review my resume and tell me what to improve?",
            conversation_history=[],
        )

        assert result.user_chunk_count == 1
        assert result.document_count == 1
        assert result.assembled_references
        assert result.assembled_references[0]["filename"] == "resume.txt"
    finally:
        db.close()


def test_user_scope_retrieval_marks_personal_context_disabled_when_set_false(client) -> None:
    from app.models.rag import RagChunk, RagCollection, RagDocument
    from app.services.rag_retrieval_service import RagRetrievalService

    _seed_canonical_membership_schema(client)

    db = next(client.app.dependency_overrides[get_db]())
    try:
        collection = RagCollection(
            id="collection_user_resume_disabled",
            scope_type="user",
            tenant_id="tenant_1",
            user_id_hash="user_1",
            name="Personal Context",
            is_active=True,
            retrieval_enabled=False,
            max_results=2,
        )
        document = RagDocument(
            id="document_user_resume_disabled",
            collection_id=collection.id,
            scope_type="user",
            tenant_id="tenant_1",
            user_id_hash="user_1",
            filename="resume.txt",
            media_type="text/plain",
            size_bytes=256,
            status="ready",
            status_message=None,
            sha256="disabledhash",
            source_kind="database_blob",
            uploaded_by_admin_user_id=None,
            uploaded_by_user_id_hash="user_1",
        )
        chunk = RagChunk(
            id="chunk_user_resume_disabled",
            document_id=document.id,
            chunk_index=0,
            chunk_text="I led GTM launches with measurable growth.",
            token_count=9,
            embedding_vector=vectorize_text("I led GTM launches with measurable growth."),
        )
        db.add(collection)
        db.add(document)
        db.add(chunk)
        db.commit()

        result = RagRetrievalService(db).retrieve(
            tenant_id="tenant_1",
            user_id_hash="user_1",
            conversation_id="conv_personal_disabled",
            raw_prompt="Can you review my resume and tell me what to improve?",
            conversation_history=[],
        )

        assert result.user_chunk_count == 0
        assert result.document_count == 0
        assert result.assembled_references == []
        assert result.skipped_reason == "user_context_disabled"
    finally:
        db.close()


def test_upload_user_document_works_against_canonical_herman_db_rag_schema(client) -> None:
    _recreate_canonical_rag_schema(client)
    _seed_canonical_membership_schema(client)

    response = client.post(
        "/api/rag/user-documents",
        headers=AUTH_HEADERS,
        data={"tenant_id": "tenant_1", "user_id_hash": "user_1"},
        files={"file": ("personal-notes.txt", b"I prefer concise recruiter summaries.", "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["document"]["status"] == "ready"
    assert body["usage"]["document_count"] == 1


def test_retrieval_events_write_against_canonical_herman_db_rag_schema(client) -> None:
    from app.models.rag import RagChunk, RagCollection, RagDocument
    from app.services.rag_retrieval_service import RagRetrievalService

    _recreate_canonical_rag_schema(client)
    _seed_canonical_membership_schema(client)

    db = next(client.app.dependency_overrides[get_db]())
    try:
        collection = RagCollection(
            id="collection_user_canonical",
            scope_type="user",
            tenant_id="tenant_1",
            user_id_hash="user_1",
            name="Personal Context",
            is_active=True,
            retrieval_enabled=True,
            max_results=2,
        )
        document = RagDocument(
            id="document_user_canonical",
            collection_id=collection.id,
            scope_type="user",
            tenant_id="tenant_1",
            user_id_hash="user_1",
            filename="personal-notes.txt",
            media_type="text/plain",
            size_bytes=64,
            status="ready",
            status_message=None,
            sha256="abc123",
            source_kind="database_blob",
            uploaded_by_admin_user_id=None,
            uploaded_by_user_id_hash="user_1",
        )
        chunk = RagChunk(
            id="chunk_user_canonical",
            document_id=document.id,
            chunk_index=0,
            chunk_text="I prefer concise recruiter summaries and bullet-style responses.",
            token_count=9,
            embedding_vector=vectorize_text("I prefer concise recruiter summaries and bullet-style responses."),
        )
        db.add(collection)
        db.add(document)
        db.add(chunk)
        db.commit()

        result = RagRetrievalService(db).retrieve(
            tenant_id="tenant_1",
            user_id_hash="user_1",
            conversation_id="conv_canonical_retrieval",
            raw_prompt="Tell me about recruiter summaries.",
            conversation_history=[],
        )
        db.commit()

        stored_score = db.execute(text("select score from rag_retrieval_events limit 1")).scalar_one()
        assert result.user_chunk_count == 1
        assert isinstance(stored_score, (int, float))
    finally:
        db.close()


def test_execute_chat_continues_when_retrieval_event_logging_fails(client) -> None:
    from app.models.rag import RagChunk, RagCollection, RagDocument

    _seed_final_profiles(client)
    _recreate_canonical_rag_schema(client)
    _seed_canonical_membership_schema(client)

    db = next(client.app.dependency_overrides[get_db]())
    try:
        collection = RagCollection(
            id="collection_user_resume",
            scope_type="user",
            tenant_id="tenant_1",
            user_id_hash="user_1",
            name="Personal Context",
            is_active=True,
            retrieval_enabled=True,
            max_results=2,
        )
        document = RagDocument(
            id="document_user_resume",
            collection_id=collection.id,
            scope_type="user",
            tenant_id="tenant_1",
            user_id_hash="user_1",
            filename="resume.txt",
            media_type="text/plain",
            size_bytes=256,
            status="ready",
            status_message=None,
            sha256="resumehash",
            source_kind="database_blob",
            uploaded_by_admin_user_id=None,
            uploaded_by_user_id_hash="user_1",
        )
        chunk = RagChunk(
            id="chunk_user_resume",
            document_id=document.id,
            chunk_index=0,
            chunk_text="Experienced product marketer and growth operator with B2B SaaS hiring and GTM leadership background.",
            token_count=14,
            embedding_vector=vectorize_text(
                "Experienced product marketer and growth operator with B2B SaaS hiring and GTM leadership background."
            ),
        )
        db.add(collection)
        db.add(document)
        db.add(chunk)
        db.commit()
    finally:
        db.close()

    with (
        patch(
            "app.services.transformer_engine.RuntimeLlmResolver.resolve",
            return_value=_runtime_config(
                provider="xai",
                model="grok-3-mini",
                endpoint_url="https://api.x.ai/v1",
            ),
        ),
        patch(
            "app.services.rag_retrieval_service.RagRetrievalService._persist_retrieval_events",
            side_effect=SQLAlchemyError("rag analytics write failed"),
        ),
        patch("app.services.final_response_service.httpx.Client") as httpx_client,
    ):
        mock_client = httpx_client.return_value.__enter__.return_value
        mock_client.post.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Your background best fits GTM, lifecycle, and product marketing roles.",
                        }
                    ],
                }
            ]
        }

        response = client.post(
            "/api/chat/execute",
            headers=AUTH_HEADERS,
            json={
                "session_id": "sess_resume_market_fit",
                "conversation_id": "conv_resume_market_fit",
                "user_id_hash": "user_1",
                "raw_prompt": "based on my resume what kind of jobs are my best fit in today's marketplace?",
                "target_llm": {"provider": "xai", "model": "grok-3-mini"},
                "conversation_history": [],
                "attachments": [],
                "transform_enabled": True,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_text"] == "Your background best fits GTM, lifecycle, and product marketing roles."
    assert body["metadata"]["retrieval_used"] is True
    assert body["metadata"]["retrieval_scope_counts"]["user"] == 1
    assert body["metadata"]["retrieval_document_count"] == 1


def test_execute_chat_reports_user_context_disabled_reason_when_personal_toggle_is_off(client) -> None:
    from app.models.rag import RagChunk, RagCollection, RagDocument

    _seed_final_profiles(client)
    _recreate_canonical_rag_schema(client)
    _seed_canonical_membership_schema(client)

    db = next(client.app.dependency_overrides[get_db]())
    try:
        collection = RagCollection(
            id="collection_user_resume_disabled_chat",
            scope_type="user",
            tenant_id="tenant_1",
            user_id_hash="user_1",
            name="Personal Context",
            is_active=True,
            retrieval_enabled=False,
            max_results=2,
        )
        document = RagDocument(
            id="document_user_resume_disabled_chat",
            collection_id=collection.id,
            scope_type="user",
            tenant_id="tenant_1",
            user_id_hash="user_1",
            filename="resume.txt",
            media_type="text/plain",
            size_bytes=256,
            status="ready",
            status_message=None,
            sha256="disabledhash",
            source_kind="database_blob",
            uploaded_by_admin_user_id=None,
            uploaded_by_user_id_hash="user_1",
        )
        chunk = RagChunk(
            id="chunk_user_resume_disabled_chat",
            document_id=document.id,
            chunk_index=0,
            chunk_text="I led GTM launches with measurable growth.",
            token_count=9,
            embedding_vector=vectorize_text("I led GTM launches with measurable growth."),
        )
        db.add(collection)
        db.add(document)
        db.add(chunk)
        db.commit()
    finally:
        db.close()

    with (
        patch(
            "app.services.transformer_engine.RuntimeLlmResolver.resolve",
            return_value=_runtime_config(
                provider="xai",
                model="grok-3-mini",
                endpoint_url="https://api.x.ai/v1",
            ),
        ),
        patch("app.services.final_response_service.httpx.Client") as httpx_client,
    ):
        mock_client = httpx_client.return_value.__enter__.return_value
        mock_client.post.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Here is a general answer."}],
                }
            ]
        }

        response = client.post(
            "/api/chat/execute",
            headers=AUTH_HEADERS,
            json={
                "session_id": "sess_resume_disabled",
                "conversation_id": "conv_resume_disabled",
                "user_id_hash": "user_1",
                "raw_prompt": "Can you review my resume and tell me what to improve?",
                "target_llm": {"provider": "xai", "model": "grok-3-mini"},
                "conversation_history": [],
                "attachments": [],
                "transform_enabled": True,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_text"] == "Here is a general answer."
    assert body["metadata"]["retrieval_used"] is False
    assert body["metadata"]["retrieval_scope_counts"] == {"tenant": 0, "user": 0}
    assert body["metadata"]["retrieval_skipped_reason"] == "user_context_disabled"


def test_execute_chat_skips_retrieval_for_short_queries_and_reports_reason(client) -> None:
    _seed_final_profiles(client)

    with (
        patch(
            "app.services.transformer_engine.RuntimeLlmResolver.resolve",
            return_value=_runtime_config(),
        ),
        patch("app.services.transformer_engine.RagRetrievalService.retrieve") as retrieve_mock,
        patch("app.services.final_response_service.httpx.Client") as httpx_client,
    ):
        mock_client = httpx_client.return_value.__enter__.return_value
        mock_client.post.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {
            "output_text": "Sure."
        }

        response = client.post(
            "/api/chat/execute",
            headers=AUTH_HEADERS,
            json={
                "session_id": "sess_short_query",
                "conversation_id": "conv_short_query",
                "user_id_hash": "user_1",
                "raw_prompt": "Hi there",
                "target_llm": {"provider": "openai", "model": "gpt-4.1"},
                "conversation_history": [],
                "attachments": [],
                "transform_enabled": True,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_text"] == "Sure."
    assert body["metadata"]["retrieval_used"] is False
    assert body["metadata"]["retrieval_skipped_reason"] == "query_too_short"
    retrieve_mock.assert_not_called()


def test_user_rag_limits_follow_canonical_membership_schema(client) -> None:
    _seed_canonical_membership_schema(client)

    response = client.get(
        "/api/rag/limits/user/me",
        headers=AUTH_HEADERS,
        params={"tenant_id": "tenant_1", "user_id_hash": "user_1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["limits"]["policy_key"] == "service-tier-user-3"
    assert body["limits"]["max_document_count"] == 20
    assert body["usage"]["document_count"] == 0


def test_list_user_documents_returns_controlled_client_error_when_rag_policies_are_missing(client) -> None:
    response = client.get(
        "/api/rag/user-documents/me",
        headers=AUTH_HEADERS,
        params={"tenant_id": "tenant_1", "user_id_hash": "user_1"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]


def test_list_user_documents_returns_503_for_database_errors(client) -> None:
    with patch(
        "app.api.routes.RagLimitResolver.resolve_user_limits",
        side_effect=OperationalError("select 1", {}, Exception("db down")),
    ):
        response = client.get(
            "/api/rag/user-documents/me",
            headers=AUTH_HEADERS,
            params={"tenant_id": "tenant_1", "user_id_hash": "user_1"},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Database unavailable"


def test_upload_user_document_returns_400_for_domain_validation_errors(client) -> None:
    with patch(
        "app.api.routes.RagIngestionService.upload_document",
        side_effect=ValueError("Unsupported document type"),
    ):
        response = client.post(
            "/api/rag/user-documents",
            headers=AUTH_HEADERS,
            data={"tenant_id": "tenant_1", "user_id_hash": "user_1"},
            files={"file": ("notes.exe", b"not-a-doc", "application/octet-stream")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported document type"


def test_invalid_summary_type_returns_400(client) -> None:
    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_999",
            "conversation_id": "conv_999",
            "user_id_hash": "user_1",
            "raw_prompt": "Explain this concept simply",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "summary_type": 10,
        },
    )

    assert response.status_code == 400


def test_transform_requires_service_credentials(client) -> None:
    response = client.post(
        "/api/transform_prompt",
        json={
            "session_id": "sess_missing_auth",
            "conversation_id": "conv_missing_auth",
            "user_id_hash": "user_1",
            "raw_prompt": "Explain this concept simply",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid client identity."


def test_transform_rejects_invalid_service_credentials(client) -> None:
    response = client.post(
        "/api/transform_prompt",
        headers={
            "Authorization": "Bearer wrong-key",
            "X-Client-Id": "hermanprompt",
        },
        json={
            "session_id": "sess_wrong_auth",
            "conversation_id": "conv_wrong_auth",
            "user_id_hash": "user_1",
            "raw_prompt": "Explain this concept simply",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid service credentials."


def test_transform_rejects_mismatched_conversation_ids(client) -> None:
    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_conv_mismatch",
            "conversation_id": "conv_top_level",
            "user_id_hash": "user_1",
            "raw_prompt": "Explain this concept simply",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "conversation": {
                    "conversation_id": "conv_nested",
                "requirements": {
                    "task": {"value": "Explain this concept simply", "status": "present"}
                },
                "enforcement": {
                    "level": "moderate",
                    "status": "not_evaluated",
                    "missing_fields": [],
                    "last_evaluated_at": None,
                },
            },
        },
    )

    assert response.status_code == 400


def test_transform_returns_coaching_when_full_enforcement_missing_fields(client) -> None:
    _seed_final_profiles(client)
    _update_profile(client, "user_1", prompt_enforcement_level="full")

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_coaching",
            "conversation_id": "conv_coaching",
            "user_id_hash": "user_1",
            "raw_prompt": "Explain how this works",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "coaching"
    assert body["transformed_prompt"] is None
    assert body["coaching_tip"].startswith("Coaching:")
    assert body["conversation"]["enforcement"]["status"] == "needs_coaching"
    assert "who" in body["conversation"]["enforcement"]["missing_fields"]


def test_transform_allows_demo_enforcement_override(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_override",
            "conversation_id": "conv_override",
            "user_id_hash": "user_1",
            "raw_prompt": "Explain how this works",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "full",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "coaching"
    assert body["conversation"]["enforcement"]["level"] == "full"
    assert "policy:enforcement:override" in body["metadata"]["rules_applied"]


def test_transform_derives_context_and_output_from_compact_prompt(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_compact_prompt",
            "conversation_id": "conv_compact_prompt",
            "user_id_hash": "user_1",
            "raw_prompt": "you are telling jokes at a kids birthday party, and just give me the joke in the chat.",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "full",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "coaching"
    assert body["conversation"]["requirements"]["who"]["status"] == "present"
    assert body["conversation"]["requirements"]["context"]["status"] == "present"
    assert body["conversation"]["requirements"]["output"]["status"] == "present"
    assert body["conversation"]["enforcement"]["missing_fields"] == ["labeled_structure"]


def test_transform_requires_more_than_generic_joke_prompt_under_full_enforcement(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_joke_prompt",
            "conversation_id": "conv_joke_prompt",
            "user_id_hash": "user_1",
            "raw_prompt": "tell me a joke",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "full",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "coaching"
    assert body["transformed_prompt"] is None
    assert set(body["conversation"]["enforcement"]["missing_fields"]) == {"who", "context", "output"}


def test_transform_accepts_unlabeled_natural_language_prompt_under_moderate_enforcement(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_moderate_natural_language",
            "conversation_id": "conv_moderate_natural_language",
            "user_id_hash": "user_1",
            "raw_prompt": (
                "You are a senior Python software engineer. "
                "Explain how to design a REST API rate-limiting system for a SaaS application. "
                "I am preparing for a backend system design interview and need a clear answer. "
                "Provide the answer in the chat as a concise structured response with an overview, "
                "core components, request flow, tradeoffs, and one example."
            ),
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "moderate",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "transformed"
    assert body["scoring"]["structural_score"] == 100
    assert body["conversation"]["requirements"]["who"]["status"] == "present"
    assert body["conversation"]["requirements"]["task"]["status"] == "present"
    assert body["conversation"]["requirements"]["context"]["status"] == "present"
    assert body["conversation"]["requirements"]["output"]["status"] == "present"
    for field_name in ("who", "task", "context", "output"):
        requirement = body["conversation"]["requirements"][field_name]
        assert set(requirement.keys()) >= {
            "value",
            "status",
            "heuristic_score",
            "llm_score",
            "max_score",
            "reason",
            "improvement_hint",
        }
        assert requirement["heuristic_score"] is not None
        assert requirement["max_score"] == 25
        assert requirement["reason"]


def test_transform_allows_derived_fields_under_moderate_enforcement_with_coaching(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_moderate_derived_ok",
            "conversation_id": "conv_moderate_derived_ok",
            "user_id_hash": "user_1",
            "raw_prompt": "You are a comedian at a kids birthday party tell me a joke",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "moderate",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "transformed"
    assert body["coaching_tip"] is not None
    assert body["conversation"]["enforcement"]["status"] == "passes"
    assert body["conversation"]["requirements"]["who"]["status"] in {"present", "derived"}
    assert body["conversation"]["requirements"]["task"]["status"] in {"present", "derived"}
    assert body["conversation"]["requirements"]["context"]["status"] in {"present", "derived"}
    assert body["conversation"]["requirements"]["output"]["status"] == "derived"


def test_transform_requires_labels_under_full_enforcement_even_when_elements_are_present(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_full_requires_labels",
            "conversation_id": "conv_full_requires_labels",
            "user_id_hash": "user_1",
            "raw_prompt": (
                "You are a senior Python software engineer. "
                "Explain how to design a REST API rate-limiting system for a SaaS application. "
                "I am preparing for a backend system design interview and need a clear answer. "
                "Provide the answer in the chat as a concise structured response with an overview, "
                "core components, request flow, tradeoffs, and one example."
            ),
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "full",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "coaching"
    assert body["coaching_tip"] == "Coaching: for full guidance, format the prompt with Who, Task, Context, and Output labels."
    assert body["conversation"]["enforcement"]["missing_fields"] == ["labeled_structure"]
    assert body["scoring"]["structural_score"] == 100


def test_transform_accepts_labeled_prompt_under_full_enforcement(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_full_labeled",
            "conversation_id": "conv_full_labeled",
            "user_id_hash": "user_1",
            "raw_prompt": (
                "Who: Senior Python software engineer\n"
                "Task: Explain how to design a REST API rate-limiting system for a SaaS application.\n"
                "Context: I am preparing for a backend system design interview and need a clear answer.\n"
                "Output: Provide the answer in the chat as a concise structured response with an overview, core components, request flow, tradeoffs, and one example."
            ),
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "full",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "transformed"
    assert body["scoring"]["structural_score"] == 100


def test_transform_returns_light_coaching_under_low_enforcement_without_blocking(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_low_guidance",
            "conversation_id": "conv_low_guidance",
            "user_id_hash": "user_1",
            "raw_prompt": "tell me a joke",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "low",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "transformed"
    assert body["coaching_tip"] == "Coaching: include the role you want the AI to play, the setting or intended use, and how you want the answer delivered next time for a stronger prompt."
    assert body["scoring"]["structural_score"] == 30


def test_transform_returns_warning_findings_when_compliance_check_enabled(client) -> None:
    _seed_final_profiles(client)
    _update_profile(
        client,
        "user_1",
        prompt_enforcement_level="none",
        compliance_check_enabled=True,
    )

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_compliance_warning",
            "conversation_id": "conv_compliance_warning",
            "user_id_hash": "user_1",
            "raw_prompt": "Please provide financial advice for my retirement plan.",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "transformed"
    assert body["findings"][0]["type"] == "compliance"
    assert body["findings"][0]["severity"] == "medium"


def test_transform_blocks_when_pii_check_detects_high_risk_content(client) -> None:
    _seed_final_profiles(client)
    _update_profile(
        client,
        "user_1",
        prompt_enforcement_level="none",
        pii_check_enabled=True,
    )

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_pii_blocked",
            "conversation_id": "conv_pii_blocked",
            "user_id_hash": "user_1",
            "raw_prompt": "Write an email for alice@example.com and bob@example.com about our offer.",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "blocked"
    assert body["transformed_prompt"] is None
    assert body["blocking_message"] is not None
    assert body["conversation"]["enforcement"]["status"] == "blocked"
    assert body["findings"][0]["type"] == "pii"
    assert body["findings"][0]["severity"] == "high"


def test_transform_logs_new_result_fields_when_request_logging_enabled(client) -> None:
    _seed_final_profiles(client)
    _update_profile(client, "user_1", prompt_enforcement_level="full")
    settings = get_settings()
    original_enable_request_logging = settings.enable_request_logging
    settings.enable_request_logging = True

    try:
        response = client.post(
            "/api/transform_prompt",
            headers=AUTH_HEADERS,
            json={
                "session_id": "sess_logging",
                "conversation_id": "conv_logging",
                "user_id_hash": "user_1",
                "raw_prompt": "Explain how this works",
                "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            },
        )
        assert response.status_code == 200

        db = next(client.app.dependency_overrides[get_db]())
        try:
            log_row = db.query(PromptTransformRequest).filter_by(session_id="sess_logging").one()
            assert log_row.conversation_id == "conv_logging"
            assert log_row.result_type == "coaching"
            assert log_row.enforcement_level == "full"
            assert log_row.conversation_json["enforcement"]["status"] == "needs_coaching"
        finally:
            db.close()
    finally:
        settings.enable_request_logging = original_enable_request_logging


def test_transform_returns_request_log_id_and_persists_request_row(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_request_identity",
            "conversation_id": "conv_request_identity",
            "user_id_hash": "user_1",
            "raw_prompt": "Explain this concept simply",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["request_log_id"] is not None

    db = next(client.app.dependency_overrides[get_db]())
    try:
        log_row = db.get(PromptTransformRequest, body["metadata"]["request_log_id"])
        assert log_row is not None
        assert log_row.session_id == "sess_request_identity"
        assert log_row.metadata_json["request_log_id"] == body["metadata"]["request_log_id"]
    finally:
        db.close()


def test_final_response_usage_endpoint_updates_request_token_usage(client) -> None:
    _seed_final_profiles(client)

    transform_response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_final_usage",
            "conversation_id": "conv_final_usage",
            "user_id_hash": "user_1",
            "raw_prompt": "Explain this concept simply",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert transform_response.status_code == 200
    request_log_id = transform_response.json()["metadata"]["request_log_id"]

    usage_response = client.post(
        "/api/request_usage/final_response",
        headers=AUTH_HEADERS,
        json={
            "request_log_id": request_log_id,
            "provider": "openai",
            "model": "gpt-4.1",
            "usage": {
                "input_tokens": 820,
                "output_tokens": 260,
                "total_tokens": 1080,
                "raw_usage": {
                    "prompt_tokens": 820,
                    "completion_tokens": 260,
                    "total_tokens": 1080,
                },
            },
        },
    )

    assert usage_response.status_code == 200
    assert usage_response.json() == {
        "request_log_id": request_log_id,
        "status": "updated",
    }

    db = next(client.app.dependency_overrides[get_db]())
    try:
        log_row = db.get(PromptTransformRequest, request_log_id)
        assert log_row is not None
        assert log_row.token_usage_json["final_response"]["input_tokens"] == 820
        assert log_row.token_usage_json["final_response"]["output_tokens"] == 260
        assert log_row.token_usage_json["final_response"]["total_tokens"] == 1080
        assert log_row.token_usage_json["final_response"]["calls"] == 1
        assert log_row.token_usage_json["providers"] == [
            {
                "category": "final_response",
                "purpose": "final_response",
                "provider": "openai",
                "model": "gpt-4.1",
                "input_tokens": 820,
                "output_tokens": 260,
                "total_tokens": 1080,
                "reasoning_tokens": None,
                "cache_read_tokens": None,
                "cache_write_tokens": None,
                "raw_usage": {
                    "prompt_tokens": 820,
                    "completion_tokens": 260,
                    "total_tokens": 1080,
                },
            }
        ]
    finally:
        db.close()


def test_final_response_usage_endpoint_is_idempotent_for_same_request(client) -> None:
    _seed_final_profiles(client)

    transform_response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_final_usage_idempotent",
            "conversation_id": "conv_final_usage_idempotent",
            "user_id_hash": "user_1",
            "raw_prompt": "Explain this concept simply",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert transform_response.status_code == 200
    request_log_id = transform_response.json()["metadata"]["request_log_id"]

    usage_payload = {
        "request_log_id": request_log_id,
        "provider": "openai",
        "model": "gpt-4.1",
        "usage": {
            "input_tokens": 820,
            "output_tokens": 260,
            "total_tokens": 1080,
        },
    }

    first_response = client.post(
        "/api/request_usage/final_response",
        headers=AUTH_HEADERS,
        json=usage_payload,
    )
    second_response = client.post(
        "/api/request_usage/final_response",
        headers=AUTH_HEADERS,
        json=usage_payload,
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    db = next(client.app.dependency_overrides[get_db]())
    try:
        log_row = db.get(PromptTransformRequest, request_log_id)
        assert log_row is not None
        assert log_row.token_usage_json["final_response"]["input_tokens"] == 820
        assert log_row.token_usage_json["final_response"]["output_tokens"] == 260
        assert log_row.token_usage_json["final_response"]["total_tokens"] == 1080
        assert log_row.token_usage_json["final_response"]["calls"] == 1
    finally:
        db.close()


def test_guide_me_helper_persists_admin_token_usage(client) -> None:
    with (
        patch("app.services.transformer_engine.RuntimeLlmResolver.resolve", return_value=_runtime_config()),
        patch(
            "app.services.transformer_engine.GuideMeGenerationService.generate",
            return_value=GuideMeGenerationResult(
                payload={"task": "Reduce unqualified applicants by 30%."},
                usage=NormalizedTokenUsage(
                    input_tokens=120,
                    output_tokens=45,
                    total_tokens=165,
                ),
            ),
        ),
    ):
        response = client.post(
            "/api/guide_me/generate",
            headers=AUTH_HEADERS,
            json={
                "session_id": "sess_guide_usage",
                "conversation_id": "conv_guide_usage",
                "user_id_hash": "user_1",
                "target_llm": {"provider": "openai", "model": "gpt-4.1"},
                "helper_kind": "answer_extraction",
                "prompt": "Return strict JSON with optional keys who, task, context, output.",
            },
        )

    assert response.status_code == 200
    assert response.json()["payload"]["task"] == "Reduce unqualified applicants by 30%."

    db = next(client.app.dependency_overrides[get_db]())
    try:
        log_row = (
            db.query(PromptTransformRequest)
            .filter_by(session_id="sess_guide_usage", conversation_id="conv_guide_usage")
            .one()
        )
        assert log_row.task_type == "guide_me"
        assert log_row.result_type == "transformed"
        assert log_row.metadata_json["request_kind"] == "guide_me"
        assert log_row.metadata_json["helper_kind"] == "answer_extraction"
        assert log_row.metadata_json["resolved_provider"] == "openai"
        assert log_row.metadata_json["resolved_model"] == "gpt-4.1"
        assert log_row.token_usage_json["admin"]["input_tokens"] == 120
        assert log_row.token_usage_json["admin"]["output_tokens"] == 45
        assert log_row.token_usage_json["admin"]["total_tokens"] == 165
        assert log_row.token_usage_json["admin"]["calls"] == 1
        assert log_row.token_usage_json["admin"]["by_purpose"]["guide_me"]["total_tokens"] == 165
        assert log_row.token_usage_json["providers"] == [
            {
                "category": "admin",
                "purpose": "guide_me",
                "provider": "openai",
                "model": "gpt-4.1",
                "input_tokens": 120,
                "output_tokens": 45,
                "total_tokens": 165,
                "reasoning_tokens": None,
                "cache_read_tokens": None,
                "cache_write_tokens": None,
                "raw_usage": None,
            }
        ]
    finally:
        db.close()


def test_request_logger_append_usage_preserves_existing_categories(client) -> None:
    db = next(client.app.dependency_overrides[get_db]())
    try:
        request_row = PromptTransformRequest(
            session_id="sess_usage_merge",
            conversation_id="conv_usage_merge",
            user_id_hash="user_1",
            raw_prompt="Prompt",
            transformed_prompt=None,
            task_type="guide_me",
            result_type="transformed",
            coaching_tip=None,
            blocking_message=None,
            target_provider="openai",
            target_model="gpt-4.1",
            persona_source="bypassed",
            used_fallback_model=False,
            enforcement_level="none",
            compliance_check_enabled=False,
            pii_check_enabled=False,
            conversation_json={},
            findings_json=[],
            metadata_json={"request_kind": "guide_me"},
            token_usage_json={
                "final_response": {
                    "input_tokens": 820,
                    "output_tokens": 260,
                    "total_tokens": 1080,
                    "calls": 1,
                    "by_purpose": {
                        "final_response": {
                            "input_tokens": 820,
                            "output_tokens": 260,
                            "total_tokens": 1080,
                            "calls": 1,
                        }
                    },
                },
                "providers": [
                    {
                        "category": "final_response",
                        "purpose": "final_response",
                        "provider": "openai",
                        "model": "gpt-4.1",
                        "input_tokens": 820,
                        "output_tokens": 260,
                        "total_tokens": 1080,
                        "reasoning_tokens": None,
                        "cache_read_tokens": None,
                        "cache_write_tokens": None,
                        "raw_usage": None,
                    }
                ],
            },
        )
        db.add(request_row)
        db.commit()
        db.refresh(request_row)

        from app.services.request_logger import RequestLogger
        from app.services.token_usage import build_usage_entry

        RequestLogger(db).append_usage(
            request_row.id,
            build_usage_entry(
                category="admin",
                purpose="guide_me",
                provider="openai",
                model="gpt-4.1",
                usage=NormalizedTokenUsage(
                    input_tokens=120,
                    output_tokens=45,
                    total_tokens=165,
                ),
            ),
        )

        db.refresh(request_row)
        assert request_row.token_usage_json["final_response"]["total_tokens"] == 1080
        assert request_row.token_usage_json["admin"]["total_tokens"] == 165
        assert request_row.token_usage_json["admin"]["by_purpose"]["guide_me"]["total_tokens"] == 165
        assert len(request_row.token_usage_json["providers"]) == 2
    finally:
        db.close()


def test_guide_me_helper_creates_request_row_without_admin_usage_when_provider_usage_missing(client) -> None:
    with (
        patch("app.services.transformer_engine.RuntimeLlmResolver.resolve", return_value=_runtime_config()),
        patch(
            "app.services.transformer_engine.GuideMeGenerationService.generate",
            return_value=GuideMeGenerationResult(
                payload={"task": "Reduce unqualified applicants by 30%."},
                usage=None,
            ),
        ),
    ):
        response = client.post(
            "/api/guide_me/generate",
            headers=AUTH_HEADERS,
            json={
                "session_id": "sess_guide_no_usage",
                "conversation_id": "conv_guide_no_usage",
                "user_id_hash": "user_1",
                "target_llm": {"provider": "openai", "model": "gpt-4.1"},
                "helper_kind": "answer_extraction",
                "prompt": "Return strict JSON with optional keys who, task, context, output.",
            },
        )

    assert response.status_code == 200

    db = next(client.app.dependency_overrides[get_db]())
    try:
        log_row = (
            db.query(PromptTransformRequest)
            .filter_by(session_id="sess_guide_no_usage", conversation_id="conv_guide_no_usage")
            .one()
        )
        assert log_row.task_type == "guide_me"
        assert log_row.result_type == "transformed"
        assert log_row.metadata_json["request_kind"] == "guide_me"
        assert log_row.token_usage_json is None
    finally:
        db.close()


def test_transform_creates_conversation_score_rollup(client) -> None:
    _seed_final_profiles(client)
    _update_profile(client, "user_1", prompt_enforcement_level="full")

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_score_rollup",
            "conversation_id": "conv_score_rollup",
            "user_id_hash": "user_1",
            "raw_prompt": "Explain how this works",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert response.status_code == 200

    db = next(client.app.dependency_overrides[get_db]())
    try:
        score_row = db.query(ConversationPromptScore).filter_by(conversation_id="conv_score_rollup").one()
        assert score_row.user_id_hash == "user_1"
        assert score_row.initial_score == 30
        assert score_row.final_score == 30
        assert score_row.best_score == 30
        assert score_row.improvement_score == 0
        assert score_row.reached_policy_complete is False
        assert score_row.coaching_turn_count == 1
        assert score_row.transformed_turn_count == 0
    finally:
        db.close()


def test_transform_updates_conversation_score_when_prompt_improves(client) -> None:
    _seed_final_profiles(client)

    first_response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_score_improve_1",
            "conversation_id": "conv_score_improve",
            "user_id_hash": "user_1",
            "raw_prompt": "tell me a joke",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "full",
        },
    )
    assert first_response.status_code == 200
    assert first_response.json()["result_type"] == "coaching"

    second_response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_score_improve_2",
            "conversation_id": "conv_score_improve",
            "user_id_hash": "user_1",
            "raw_prompt": "you are telling jokes at a kids birthday party, and just give me the joke in the chat.",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "full",
        },
    )
    assert second_response.status_code == 200
    assert second_response.json()["result_type"] == "coaching"

    db = next(client.app.dependency_overrides[get_db]())
    try:
        score_row = db.query(ConversationPromptScore).filter_by(conversation_id="conv_score_improve").one()
        assert score_row.initial_score == 30
        assert score_row.final_score == 100
        assert score_row.best_score == 100
        assert score_row.improvement_score == 70
        assert score_row.best_improvement_score == 70
        assert score_row.reached_policy_complete is False
        assert score_row.coaching_turn_count == 2
        assert score_row.transformed_turn_count == 0
    finally:
        db.close()


def test_conversation_memory_does_not_inflate_current_turn_score(client) -> None:
    _seed_final_profiles(client)

    first_response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_memory_score_1",
            "conversation_id": "conv_memory_score",
            "user_id_hash": "user_1",
            "raw_prompt": "tell me a joke",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "moderate",
        },
    )
    assert first_response.status_code == 200
    first_body = first_response.json()

    second_response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_memory_score_2",
            "conversation_id": "conv_memory_score",
            "user_id_hash": "user_1",
            "raw_prompt": "you are a comedian tell me a joke the I can use at a kids birthday party",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "conversation": first_body["conversation"],
            "enforcement_level": "moderate",
        },
    )

    assert second_response.status_code == 200
    second_body = second_response.json()
    assert second_body["result_type"] == "transformed"
    assert second_body["scoring"]["initial_score"] == 30
    assert second_body["scoring"]["final_score"] == 80
    assert second_body["scoring"]["structural_score"] == 80

    db = next(client.app.dependency_overrides[get_db]())
    try:
        score_row = db.query(ConversationPromptScore).filter_by(conversation_id="conv_memory_score").one()
        assert score_row.initial_score == 30
        assert score_row.final_score == 80
        assert score_row.best_score == 80
    finally:
        db.close()


def test_get_conversation_score_returns_rollup_from_database(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_score_fetch",
            "conversation_id": "conv_score_fetch",
            "user_id_hash": "user_1",
            "raw_prompt": "tell me a joke",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "low",
        },
    )
    assert response.status_code == 200

    score_response = client.get(
        "/api/conversation_scores/conv_score_fetch",
        headers=AUTH_HEADERS,
        params={"user_id_hash": "user_1"},
    )

    assert score_response.status_code == 200
    body = score_response.json()
    assert body["conversation_id"] == "conv_score_fetch"
    assert body["user_id_hash"] == "user_1"
    assert body["scoring_version"] == "v4"
    assert body["initial_score"] == 30
    assert body["final_score"] == 30
    assert body["initial_llm_score"] in {None, 30}
    assert body["final_llm_score"] in {None, 30}
    assert body["structural_score"] == 30
    assert body["best_score"] == 30
    assert body["improvement_score"] == 0
    assert body["conversation"]["conversation_id"] == "conv_score_fetch"
    assert body["conversation"]["requirements"]["task"]["heuristic_score"] == 25
    assert body["conversation"]["requirements"]["task"]["max_score"] == 25
    assert body["conversation"]["requirements"]["task"]["reason"]
    assert body["conversation"]["requirements"]["who"]["improvement_hint"] is not None


def test_score_rollup_persists_hybrid_score_details(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_hybrid_score_details",
            "conversation_id": "conv_hybrid_score_details",
            "user_id_hash": "user_1",
            "raw_prompt": (
                "Who: Senior Python software engineer\n"
                "Task: Explain how to design a REST API rate-limiting system for a SaaS application.\n"
                "Context: I am preparing for a backend system design interview.\n"
                "Output: Provide the answer in the chat with an overview, tradeoffs, and one example."
            ),
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "moderate",
        },
    )

    assert response.status_code == 200

    db = next(client.app.dependency_overrides[get_db]())
    try:
        score_row = db.query(ConversationPromptScore).filter_by(
            conversation_id="conv_hybrid_score_details"
        ).one()
        assert score_row.scoring_version == "v4"
        assert score_row.score_details_json["heuristic_score"] >= 0
        assert score_row.score_details_json["final_score"] == score_row.final_score
        assert score_row.score_details_json["scoring_method"] in {"heuristic_only_v1", "hybrid_llm_v2"}
        assert "heuristic_field_statuses" in score_row.score_details_json
        assert "llm_field_statuses" in score_row.score_details_json
        assert set(score_row.score_details_json["requirements"]["task"].keys()) >= {
            "value",
            "status",
            "heuristic_score",
            "llm_score",
            "max_score",
            "reason",
            "improvement_hint",
        }
        assert score_row.final_llm_score == score_row.score_details_json["llm_score"]
    finally:
        db.close()


def test_numeric_llm_dimension_scores_are_used_for_llm_total(client, monkeypatch) -> None:
    _seed_final_profiles(client)

    from app.services.structure_evaluator import StructureEvaluationService

    monkeypatch.setattr(StructureEvaluationService, "is_enabled", lambda self: True)
    monkeypatch.setattr(
        StructureEvaluationService,
        "evaluate",
        lambda self, **kwargs: {
            "who": {"value": "comedian", "status": "present", "score": 21},
            "task": {"value": "tell me a joke", "status": "present", "score": 24},
            "context": {"value": "for a kids birthday party", "status": "derived", "score": 16},
            "output": {"value": "in the chat", "status": "derived", "score": 11},
            "coaching_tip": "Coaching: add a little more output detail next time.",
        },
    )

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_numeric_llm",
            "conversation_id": "conv_numeric_llm",
            "user_id_hash": "user_1",
            "raw_prompt": "you are a comedian tell me a joke for a kids birthday party",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "moderate",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scoring"]["final_llm_score"] == 72

    db = next(client.app.dependency_overrides[get_db]())
    try:
        score_row = db.query(ConversationPromptScore).filter_by(conversation_id="conv_numeric_llm").one()
        assert score_row.final_llm_score == 72
        assert score_row.score_details_json["llm_field_points"] == {
            "who": 21,
            "task": 24,
            "context": 16,
            "output": 11,
        }
        assert score_row.score_details_json["llm_score"] == 72
    finally:
        db.close()


def test_llm_score_uses_current_prompt_not_prior_conversation_memory(client, monkeypatch) -> None:
    _seed_final_profiles(client)

    from app.services.structure_evaluator import StructureEvaluationService

    calls: list[dict] = []

    def fake_evaluate(self, **kwargs):
        calls.append(kwargs)
        prompt = kwargs["raw_prompt"]
        if prompt == "tell me a joke":
            return {
                "who": {"value": None, "status": "missing", "score": 0},
                "task": {"value": "tell me a joke", "status": "present", "score": 25},
                "context": {"value": None, "status": "missing", "score": 0},
                "output": {"value": "joke in chat", "status": "derived", "score": 5},
                "coaching_tip": "Coaching: add audience or style.",
            }
        return {
            "who": {"value": "comedian", "status": "present", "score": 24},
            "task": {"value": "tell me a joke", "status": "present", "score": 25},
            "context": {"value": None, "status": "missing", "score": 0},
            "output": {"value": "joke in chat", "status": "derived", "score": 8},
            "coaching_tip": "Coaching: add context.",
        }

    monkeypatch.setattr(StructureEvaluationService, "is_enabled", lambda self: True)
    monkeypatch.setattr(StructureEvaluationService, "evaluate", fake_evaluate)

    first_response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_llm_current_1",
            "conversation_id": "conv_llm_current",
            "user_id_hash": "user_1",
            "raw_prompt": "tell me a joke",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "moderate",
        },
    )
    assert first_response.status_code == 200
    first_body = first_response.json()

    second_response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_llm_current_2",
            "conversation_id": "conv_llm_current",
            "user_id_hash": "user_1",
            "raw_prompt": "you are a comedian tell me a joke",
            "conversation": first_body["conversation"],
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "moderate",
        },
    )
    assert second_response.status_code == 200
    second_body = second_response.json()
    assert second_body["scoring"]["final_llm_score"] == 57
    assert "existing_requirements" not in calls[0]
    assert "existing_requirements" not in calls[1]


def test_scoring_example_task_and_context_only_scores_55(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_score_example_75",
            "conversation_id": "conv_score_example_75",
            "user_id_hash": "user_1",
            "raw_prompt": (
                "Explain how to design a REST API rate-limiting system for a SaaS application.\n\n"
                "I am preparing for a backend system design interview and need an answer that helps me "
                "understand the architecture, tradeoffs, and implementation choices clearly."
            ),
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "moderate",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scoring"]["structural_score"] == 55


def test_scoring_example_with_role_scores_80(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_score_example_100",
            "conversation_id": "conv_score_example_100",
            "user_id_hash": "user_1",
            "raw_prompt": (
                "You are a Senior Python software engineer\n\n"
                "Explain how to design a REST API rate-limiting system for a SaaS application.\n\n"
                "I am preparing for a backend system design interview and need an answer that helps me "
                "understand the architecture, tradeoffs, and implementation choices clearly."
            ),
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "moderate",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scoring"]["structural_score"] == 80


def test_scoring_example_task_only_scores_30(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_score_example_30",
            "conversation_id": "conv_score_example_30",
            "user_id_hash": "user_1",
            "raw_prompt": "Explain how to design a REST API rate-limiting system for a SaaS application.",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "low",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scoring"]["structural_score"] == 30
