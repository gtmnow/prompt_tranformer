from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class RagQuotaPolicy(Base):
    __tablename__ = "rag_quota_policies"
    __table_args__ = (
        CheckConstraint(
            "scope_target in ('global_default', 'service_tier', 'tenant_override')",
            name="ck_rag_quota_policies_scope_target",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    policy_key: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    scope_target: Mapped[str] = mapped_column(String(32), nullable=False)
    service_tier_definition_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    tenant_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    user_type: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    org_max_file_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    user_max_file_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    org_max_document_count: Mapped[int] = mapped_column(Integer, nullable=False)
    user_max_document_count: Mapped[int] = mapped_column(Integer, nullable=False)
    org_max_total_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    user_max_total_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    org_max_extracted_text_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    user_max_extracted_text_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    org_max_chunks_per_document: Mapped[int] = mapped_column(Integer, nullable=False)
    user_max_chunks_per_document: Mapped[int] = mapped_column(Integer, nullable=False)
    org_max_retrieved_chunks: Mapped[int] = mapped_column(Integer, nullable=False)
    user_max_retrieved_chunks: Mapped[int] = mapped_column(Integer, nullable=False)
    max_retrieved_chunks_total: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class RagCollection(Base):
    __tablename__ = "rag_collections"
    __table_args__ = (
        CheckConstraint("scope_type in ('tenant', 'user')", name="ck_rag_collections_scope_type"),
        CheckConstraint(
            "(scope_type = 'tenant' and tenant_id is not null and user_id_hash is null)"
            " or (scope_type = 'user' and tenant_id is not null and user_id_hash is not null)",
            name="ck_rag_collections_scope_identity",
        ),
        UniqueConstraint("scope_type", "tenant_id", "user_id_hash", name="uq_rag_collections_scope_identity"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scope_type: Mapped[str] = mapped_column(String(20), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    user_id_hash: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    retrieval_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    max_results: Mapped[int | None] = mapped_column(Integer, nullable=True)
    settings_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class RagDocument(Base):
    __tablename__ = "rag_documents"
    __table_args__ = (
        CheckConstraint("scope_type in ('tenant', 'user')", name="ck_rag_documents_scope_type"),
        CheckConstraint(
            "status in ('pending', 'processing', 'ready', 'failed', 'disabled')",
            name="ck_rag_documents_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    collection_id: Mapped[str] = mapped_column(ForeignKey("rag_collections.id", ondelete="CASCADE"), nullable=False, index=True)
    scope_type: Mapped[str] = mapped_column(String(20), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    user_id_hash: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    media_type: Mapped[str] = mapped_column(String(120), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    status_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_kind: Mapped[str] = mapped_column(String(50), nullable=False, default="database_blob")
    uploaded_by_admin_user_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    uploaded_by_user_id_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    disabled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class RagDocumentBlob(Base):
    __tablename__ = "rag_document_blobs"

    document_id: Mapped[str] = mapped_column(
        ForeignKey("rag_documents.id", ondelete="CASCADE"),
        primary_key=True,
    )
    content_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class RagChunk(Base):
    __tablename__ = "rag_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_rag_chunks_document_index"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("rag_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_vector: Mapped[list[float]] = mapped_column(JSON, nullable=False, default=list)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False, default="deterministic-hash-v1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class RagRetrievalEvent(Base):
    __tablename__ = "rag_retrieval_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_id_hash: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    scope_type: Mapped[str] = mapped_column(String(20), nullable=False)
    document_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    chunk_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
