from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
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

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    policy_key: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    scope_target: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    service_tier_definition_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    tenant_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    user_type: Mapped[int | None] = mapped_column(Integer, nullable=True)
    org_max_file_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    user_max_file_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    org_max_document_count: Mapped[int] = mapped_column(Integer, nullable=False)
    user_max_document_count: Mapped[int] = mapped_column(Integer, nullable=False)
    org_max_total_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    user_max_total_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    org_max_extracted_text_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    user_max_extracted_text_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
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

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scope_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    tenant_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    user_id_hash: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    retrieval_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    max_results: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class RagDocument(Base):
    __tablename__ = "rag_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    collection_id: Mapped[str] = mapped_column(ForeignKey("rag_collections.id"), nullable=False, index=True)
    scope_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    tenant_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    user_id_hash: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    media_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending", index=True)
    status_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(30), nullable=False, default="database_blob")
    uploaded_by_admin_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    uploaded_by_user_id_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    disabled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RagDocumentBlob(Base):
    __tablename__ = "rag_document_blobs"

    document_id: Mapped[str] = mapped_column(
        ForeignKey("rag_documents.id"),
        primary_key=True,
    )
    content_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


class RagChunk(Base):
    __tablename__ = "rag_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_rag_chunks_document_index"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("rag_documents.id"), nullable=False, index=True)
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
    tenant_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    scope_type: Mapped[str] = mapped_column(String(20), nullable=False)
    document_id: Mapped[str] = mapped_column(String(36), nullable=False)
    chunk_id: Mapped[str] = mapped_column(String(36), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
