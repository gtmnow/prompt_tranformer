from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.exc import OperationalError, TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Session

from app.api.deps import require_service_auth
from app.db.session import get_db
from app.schemas.transform import (
    ExecuteChatRequest,
    ExecuteChatResponse,
    FinalResponseUsageRequest,
    FinalResponseUsageResponse,
    ConversationScoreResponse,
    GuideMeHelperRequest,
    GuideMeHelperResponse,
    RagCollectionSummary,
    RagCollectionUpdateRequest,
    RagDocumentDeleteResponse,
    RagDocumentListResponse,
    RagDocumentMutationResponse,
    RagDocumentSummary,
    EffectiveRagQuota,
    RagLimitsResponse,
    RagUsageSummary,
    ResolvedProfileResponse,
    TransformPromptRequest,
    TransformPromptResponse,
)
from app.services.profile_resolver import ProfileResolver
from app.services.conversation_scores import ConversationScoreService
from app.services.llm_types import NormalizedTokenUsage
from app.services.rag_ingestion_service import RagIngestionService
from app.services.rag_limit_resolver import RagLimitResolver
from app.services.request_logger import RequestLogger
from app.services.transformer_engine import TransformerEngine
from app.services.token_usage import build_usage_entry

router = APIRouter(prefix="/api", tags=["prompt-transformer"])


def _to_document_summary(document) -> RagDocumentSummary:
    return RagDocumentSummary(
        id=document.id,
        collection_id=document.collection_id,
        scope_type=document.scope_type,
        tenant_id=document.tenant_id,
        user_id_hash=document.user_id_hash,
        filename=document.filename,
        media_type=document.media_type,
        size_bytes=document.size_bytes,
        status=document.status,
        status_message=document.status_message,
        uploaded_by_admin_user_id=document.uploaded_by_admin_user_id,
        uploaded_by_user_id_hash=document.uploaded_by_user_id_hash,
        uploaded_at=document.uploaded_at.isoformat(),
        processed_at=document.processed_at.isoformat() if document.processed_at else None,
        disabled_at=document.disabled_at.isoformat() if document.disabled_at else None,
    )


def _to_collection_summary(collection) -> RagCollectionSummary:
    return RagCollectionSummary(
        id=collection.id,
        scope_type=collection.scope_type,
        tenant_id=collection.tenant_id,
        user_id_hash=collection.user_id_hash,
        name=collection.name,
        is_active=collection.is_active,
        retrieval_enabled=collection.retrieval_enabled,
        max_results=collection.max_results,
    )


def _to_usage_summary(usage) -> RagUsageSummary:
    return RagUsageSummary(
        document_count=usage.document_count,
        total_bytes=usage.total_bytes,
        ready_documents=usage.ready_documents,
        processing_documents=usage.processing_documents,
        failed_documents=usage.failed_documents,
        disabled_documents=usage.disabled_documents,
    )


def _to_limits_summary(limits) -> EffectiveRagQuota:
    return EffectiveRagQuota(
        scope=limits.scope,
        policy_source=limits.policy_source,
        policy_key=limits.policy_key,
        max_file_bytes=limits.max_file_bytes,
        max_document_count=limits.max_document_count,
        max_total_bytes=limits.max_total_bytes,
        max_extracted_text_bytes=limits.max_extracted_text_bytes,
        max_chunks_per_document=limits.max_chunks_per_document,
        max_retrieved_chunks=limits.max_retrieved_chunks,
        max_retrieved_chunks_total=limits.max_retrieved_chunks_total,
    )


@router.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/transform_prompt", response_model=TransformPromptResponse)
def transform_prompt(
    payload: TransformPromptRequest,
    _: str = Depends(require_service_auth),
    db: Session = Depends(get_db),
) -> TransformPromptResponse:
    try:
        engine = TransformerEngine(db_session=db)
        return engine.transform(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except OperationalError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from exc
    except SQLAlchemyTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection pool exhausted",
        ) from exc


@router.post("/chat/execute", response_model=ExecuteChatResponse)
def execute_chat(
    payload: ExecuteChatRequest,
    _: str = Depends(require_service_auth),
    db: Session = Depends(get_db),
) -> ExecuteChatResponse:
    try:
        engine = TransformerEngine(db_session=db)
        return engine.execute_chat(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except OperationalError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from exc
    except SQLAlchemyTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection pool exhausted",
        ) from exc


@router.post("/guide_me/generate", response_model=GuideMeHelperResponse)
def generate_guide_me_helper(
    payload: GuideMeHelperRequest,
    _: str = Depends(require_service_auth),
    db: Session = Depends(get_db),
) -> GuideMeHelperResponse:
    try:
        engine = TransformerEngine(db_session=db)
        return engine.generate_guide_me_helper(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except OperationalError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from exc
    except SQLAlchemyTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection pool exhausted",
        ) from exc


@router.get("/conversation_scores/{conversation_id}", response_model=ConversationScoreResponse)
def get_conversation_score(
    conversation_id: str,
    user_id_hash: str,
    _: str = Depends(require_service_auth),
    db: Session = Depends(get_db),
) -> ConversationScoreResponse:
    try:
        service = ConversationScoreService(db_session=db)
        return service.get_conversation_score(conversation_id=conversation_id, user_id_hash=user_id_hash)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except OperationalError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from exc
    except SQLAlchemyTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection pool exhausted",
        ) from exc


@router.get("/profiles/resolve", response_model=ResolvedProfileResponse)
def resolve_profile(
    user_id_hash: str,
    summary_type: int | None = None,
    _: str = Depends(require_service_auth),
    db: Session = Depends(get_db),
) -> ResolvedProfileResponse:
    try:
        resolver = ProfileResolver(db_session=db)
        persona = resolver.resolve(user_id_hash=user_id_hash, summary_type=summary_type)
        return ResolvedProfileResponse(
            user_id_hash=user_id_hash,
            summary_type=summary_type,
            profile_version=persona.profile_version,
            persona_source=persona.source,
            prompt_enforcement_level=persona.prompt_enforcement_level,
            compliance_check_enabled=persona.compliance_check_enabled,
            pii_check_enabled=persona.pii_check_enabled,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except OperationalError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from exc
    except SQLAlchemyTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection pool exhausted",
        ) from exc


@router.post("/request_usage/final_response", response_model=FinalResponseUsageResponse)
def record_final_response_usage(
    payload: FinalResponseUsageRequest,
    _: str = Depends(require_service_auth),
    db: Session = Depends(get_db),
) -> FinalResponseUsageResponse:
    try:
        request_logger = RequestLogger(db_session=db)
        usage_entry = build_usage_entry(
            category="final_response",
            purpose="final_response",
            provider=payload.provider,
            model=payload.model,
            usage=NormalizedTokenUsage(**payload.usage.model_dump()),
        )
        request_logger.set_final_response_usage(payload.request_log_id, usage_entry)
        return FinalResponseUsageResponse(
            request_log_id=payload.request_log_id,
            status="updated",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except OperationalError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from exc
    except SQLAlchemyTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection pool exhausted",
        ) from exc


@router.get("/rag/limits/tenant/{tenant_id}", response_model=RagLimitsResponse)
def get_tenant_rag_limits(
    tenant_id: str,
    _: str = Depends(require_service_auth),
    db: Session = Depends(get_db),
) -> RagLimitsResponse:
    resolver = RagLimitResolver(db)
    return RagLimitsResponse(
        limits=_to_limits_summary(resolver.resolve_tenant_limits(tenant_id)),
        usage=_to_usage_summary(resolver.summarize_usage(tenant_id=tenant_id, scope="tenant")),
    )


@router.get("/rag/limits/user/me", response_model=RagLimitsResponse)
def get_user_rag_limits(
    user_id_hash: str,
    tenant_id: str,
    _: str = Depends(require_service_auth),
    db: Session = Depends(get_db),
) -> RagLimitsResponse:
    resolver = RagLimitResolver(db)
    return RagLimitsResponse(
        limits=_to_limits_summary(resolver.resolve_user_limits(tenant_id, user_id_hash)),
        usage=_to_usage_summary(resolver.summarize_usage(tenant_id=tenant_id, scope="user", user_id_hash=user_id_hash)),
    )


@router.get("/rag/tenant-documents/{tenant_id}", response_model=RagDocumentListResponse)
def list_tenant_documents(
    tenant_id: str,
    _: str = Depends(require_service_auth),
    db: Session = Depends(get_db),
) -> RagDocumentListResponse:
    ingestion = RagIngestionService(db)
    resolver = RagLimitResolver(db)
    collection, documents = ingestion.list_documents(scope="tenant", tenant_id=tenant_id)
    return RagDocumentListResponse(
        collection=_to_collection_summary(collection),
        limits=_to_limits_summary(resolver.resolve_tenant_limits(tenant_id)),
        usage=_to_usage_summary(resolver.summarize_usage(tenant_id=tenant_id, scope="tenant")),
        documents=[_to_document_summary(item) for item in documents],
    )


@router.get("/rag/user-documents/me", response_model=RagDocumentListResponse)
def list_user_documents(
    user_id_hash: str,
    tenant_id: str,
    _: str = Depends(require_service_auth),
    db: Session = Depends(get_db),
) -> RagDocumentListResponse:
    ingestion = RagIngestionService(db)
    resolver = RagLimitResolver(db)
    collection, documents = ingestion.list_documents(scope="user", tenant_id=tenant_id, user_id_hash=user_id_hash)
    return RagDocumentListResponse(
        collection=_to_collection_summary(collection),
        limits=_to_limits_summary(resolver.resolve_user_limits(tenant_id, user_id_hash)),
        usage=_to_usage_summary(resolver.summarize_usage(tenant_id=tenant_id, scope="user", user_id_hash=user_id_hash)),
        documents=[_to_document_summary(item) for item in documents],
    )


@router.post("/rag/tenant-documents", response_model=RagDocumentMutationResponse)
async def upload_tenant_document(
    tenant_id: str = Form(...),
    uploaded_by_admin_user_id: str | None = Form(default=None),
    file: UploadFile = File(...),
    _: str = Depends(require_service_auth),
    db: Session = Depends(get_db),
) -> RagDocumentMutationResponse:
    ingestion = RagIngestionService(db)
    resolver = RagLimitResolver(db)
    document = ingestion.upload_document(
        scope="tenant",
        tenant_id=tenant_id,
        user_id_hash=None,
        filename=file.filename or "document",
        media_type=file.content_type,
        content_bytes=await file.read(),
        uploaded_by_admin_user_id=uploaded_by_admin_user_id,
    )
    collection, _ = ingestion.list_documents(scope="tenant", tenant_id=tenant_id)
    return RagDocumentMutationResponse(
        collection=_to_collection_summary(collection),
        limits=_to_limits_summary(resolver.resolve_tenant_limits(tenant_id)),
        usage=_to_usage_summary(resolver.summarize_usage(tenant_id=tenant_id, scope="tenant")),
        document=_to_document_summary(document),
    )


@router.post("/rag/user-documents", response_model=RagDocumentMutationResponse)
async def upload_user_document(
    tenant_id: str = Form(...),
    user_id_hash: str = Form(...),
    file: UploadFile = File(...),
    _: str = Depends(require_service_auth),
    db: Session = Depends(get_db),
) -> RagDocumentMutationResponse:
    ingestion = RagIngestionService(db)
    resolver = RagLimitResolver(db)
    document = ingestion.upload_document(
        scope="user",
        tenant_id=tenant_id,
        user_id_hash=user_id_hash,
        filename=file.filename or "document",
        media_type=file.content_type,
        content_bytes=await file.read(),
        uploaded_by_user_id_hash=user_id_hash,
    )
    collection, _ = ingestion.list_documents(scope="user", tenant_id=tenant_id, user_id_hash=user_id_hash)
    return RagDocumentMutationResponse(
        collection=_to_collection_summary(collection),
        limits=_to_limits_summary(resolver.resolve_user_limits(tenant_id, user_id_hash)),
        usage=_to_usage_summary(resolver.summarize_usage(tenant_id=tenant_id, scope="user", user_id_hash=user_id_hash)),
        document=_to_document_summary(document),
    )


@router.delete("/rag/tenant-documents/{document_id}", response_model=RagDocumentDeleteResponse)
def delete_tenant_document(
    document_id: str,
    tenant_id: str,
    _: str = Depends(require_service_auth),
    db: Session = Depends(get_db),
) -> RagDocumentDeleteResponse:
    ingestion = RagIngestionService(db)
    resolver = RagLimitResolver(db)
    collection, _ = ingestion.list_documents(scope="tenant", tenant_id=tenant_id)
    ingestion.delete_document(document_id, tenant_id=tenant_id)
    return RagDocumentDeleteResponse(
        deleted_document_id=document_id,
        collection=_to_collection_summary(collection),
        limits=_to_limits_summary(resolver.resolve_tenant_limits(tenant_id)),
        usage=_to_usage_summary(resolver.summarize_usage(tenant_id=tenant_id, scope="tenant")),
    )


@router.delete("/rag/user-documents/{document_id}", response_model=RagDocumentDeleteResponse)
def delete_user_document(
    document_id: str,
    tenant_id: str,
    user_id_hash: str,
    _: str = Depends(require_service_auth),
    db: Session = Depends(get_db),
) -> RagDocumentDeleteResponse:
    ingestion = RagIngestionService(db)
    resolver = RagLimitResolver(db)
    collection, _ = ingestion.list_documents(scope="user", tenant_id=tenant_id, user_id_hash=user_id_hash)
    ingestion.delete_document(document_id, tenant_id=tenant_id, user_id_hash=user_id_hash)
    return RagDocumentDeleteResponse(
        deleted_document_id=document_id,
        collection=_to_collection_summary(collection),
        limits=_to_limits_summary(resolver.resolve_user_limits(tenant_id, user_id_hash)),
        usage=_to_usage_summary(resolver.summarize_usage(tenant_id=tenant_id, scope="user", user_id_hash=user_id_hash)),
    )


@router.post("/rag/documents/{document_id}/reprocess", response_model=RagDocumentMutationResponse)
def reprocess_document(
    document_id: str,
    _: str = Depends(require_service_auth),
    db: Session = Depends(get_db),
) -> RagDocumentMutationResponse:
    ingestion = RagIngestionService(db)
    resolver = RagLimitResolver(db)
    document = ingestion.reprocess_document(document_id)
    collection, _ = ingestion.list_documents(
        scope=document.scope_type,
        tenant_id=document.tenant_id,
        user_id_hash=document.user_id_hash,
    )
    limits = (
        resolver.resolve_tenant_limits(document.tenant_id)
        if document.scope_type == "tenant"
        else resolver.resolve_user_limits(document.tenant_id, document.user_id_hash or "")
    )
    usage = resolver.summarize_usage(
        tenant_id=document.tenant_id,
        scope=document.scope_type,
        user_id_hash=document.user_id_hash,
    )
    return RagDocumentMutationResponse(
        collection=_to_collection_summary(collection),
        limits=_to_limits_summary(limits),
        usage=_to_usage_summary(usage),
        document=_to_document_summary(document),
    )


@router.patch("/rag/collections/{collection_id}", response_model=RagDocumentListResponse)
def update_collection(
    collection_id: str,
    payload: RagCollectionUpdateRequest,
    _: str = Depends(require_service_auth),
    db: Session = Depends(get_db),
) -> RagDocumentListResponse:
    ingestion = RagIngestionService(db)
    resolver = RagLimitResolver(db)
    collection = ingestion.update_collection(
        collection_id=collection_id,
        retrieval_enabled=payload.retrieval_enabled,
        is_active=payload.is_active,
        max_results=payload.max_results,
    )
    collection, documents = ingestion.list_documents(
        scope=collection.scope_type,
        tenant_id=collection.tenant_id,
        user_id_hash=collection.user_id_hash,
    )
    limits = (
        resolver.resolve_tenant_limits(collection.tenant_id)
        if collection.scope_type == "tenant"
        else resolver.resolve_user_limits(collection.tenant_id, collection.user_id_hash or "")
    )
    usage = resolver.summarize_usage(
        tenant_id=collection.tenant_id,
        scope=collection.scope_type,
        user_id_hash=collection.user_id_hash,
    )
    return RagDocumentListResponse(
        collection=_to_collection_summary(collection),
        limits=_to_limits_summary(limits),
        usage=_to_usage_summary(usage),
        documents=[_to_document_summary(item) for item in documents],
    )
