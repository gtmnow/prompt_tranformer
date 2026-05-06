from __future__ import annotations

import hashlib
import logging
import math
import re
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.models.rag import RagChunk, RagCollection, RagDocument, RagRetrievalEvent
from app.services.rag_limit_resolver import RagLimitResolver

VECTOR_SIZE = 32
logger = logging.getLogger("prompt_transformer.rag_retrieval")


@dataclass(frozen=True)
class RagRetrievalResult:
    assembled_references: list[dict[str, str]]
    tenant_chunk_count: int
    user_chunk_count: int
    document_count: int


class RagRetrievalService:
    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session
        self.limit_resolver = RagLimitResolver(db_session)

    def retrieve(
        self,
        *,
        tenant_id: str,
        user_id_hash: str,
        conversation_id: str,
        raw_prompt: str,
        conversation_history: list,
    ) -> RagRetrievalResult:
        query_text = raw_prompt.strip()
        if conversation_history:
            query_text = f"{query_text}\n\nRecent conversation:\n" + "\n".join(
                f"{turn.transformed_text}\n{turn.assistant_text}" for turn in conversation_history[-2:]
            )
        query_vector = self._vectorize(query_text)
        tenant_limits = self.limit_resolver.resolve_tenant_limits(tenant_id)
        user_limits = self.limit_resolver.resolve_user_limits(tenant_id, user_id_hash)

        tenant_results = self._retrieve_scope(
            scope="tenant",
            tenant_id=tenant_id,
            user_id_hash=None,
            query_vector=query_vector,
            top_k=tenant_limits.max_retrieved_chunks,
        )
        user_results = self._retrieve_scope(
            scope="user",
            tenant_id=tenant_id,
            user_id_hash=user_id_hash,
            query_vector=query_vector,
            top_k=user_limits.max_retrieved_chunks,
        )

        combined = sorted(tenant_results + user_results, key=lambda item: item["score"], reverse=True)
        combined = combined[: min(tenant_limits.max_retrieved_chunks_total, user_limits.max_retrieved_chunks_total)]
        if not combined:
            return RagRetrievalResult([], 0, 0, 0)

        tenant_count = sum(1 for item in combined if item["scope_type"] == "tenant")
        user_count = sum(1 for item in combined if item["scope_type"] == "user")
        document_ids = {item["document_id"] for item in combined}

        try:
            self._persist_retrieval_events(
                conversation_id=conversation_id,
                user_id_hash=user_id_hash,
                tenant_id=tenant_id,
                combined=combined,
            )
        except SQLAlchemyError:
            logger.exception(
                "rag_retrieval_event_persist_failed conversation_id=%s tenant_id=%s user_id_hash=%s",
                conversation_id,
                tenant_id,
                user_id_hash,
            )
        return RagRetrievalResult(
            assembled_references=[
                {"filename": item["filename"], "chunk_text": item["chunk_text"]}
                for item in combined
            ],
            tenant_chunk_count=tenant_count,
            user_chunk_count=user_count,
            document_count=len(document_ids),
        )

    def _persist_retrieval_events(
        self,
        *,
        conversation_id: str,
        user_id_hash: str,
        tenant_id: str,
        combined: list[dict[str, str | float]],
    ) -> None:
        # Retrieval analytics should never be able to take down end-user chat.
        with self.db_session.begin_nested():
            for rank, item in enumerate(combined, start=1):
                self.db_session.add(
                    RagRetrievalEvent(
                        id=str(uuid.uuid4()),
                        conversation_id=conversation_id,
                        user_id_hash=user_id_hash,
                        tenant_id=tenant_id,
                        scope_type=str(item["scope_type"]),
                        document_id=str(item["document_id"]),
                        chunk_id=str(item["chunk_id"]),
                        rank=rank,
                        score=float(item["score"]),
                    )
                )
            self.db_session.flush()

    def _retrieve_scope(
        self,
        *,
        scope: str,
        tenant_id: str,
        user_id_hash: str | None,
        query_vector: list[float],
        top_k: int,
    ) -> list[dict[str, str | float]]:
        if top_k <= 0:
            return []
        collection = self.db_session.scalars(
            select(RagCollection).where(
                RagCollection.scope_type == scope,
                RagCollection.tenant_id == tenant_id,
                RagCollection.user_id_hash == (user_id_hash if scope == "user" else None),
                RagCollection.is_active.is_(True),
                RagCollection.retrieval_enabled.is_(True),
            ).limit(1)
        ).first()
        if collection is None:
            return []
        rows = self.db_session.execute(
            select(RagChunk, RagDocument)
            .join(RagDocument, RagDocument.id == RagChunk.document_id)
            .where(
                RagDocument.collection_id == collection.id,
                RagDocument.status == "ready",
            )
        ).all()
        scored: list[dict[str, str | float | int]] = []
        fallback_candidates: list[dict[str, str | float | int]] = []
        for chunk, document in rows:
            item = {
                "scope_type": document.scope_type,
                "document_id": document.id,
                "chunk_id": chunk.id,
                "filename": document.filename,
                "chunk_text": chunk.chunk_text,
                "score": 0.0,
                "sort_epoch": document.uploaded_at.timestamp() if document.uploaded_at else 0.0,
                "chunk_index": chunk.chunk_index,
            }
            score = self._cosine(query_vector, chunk.embedding_vector or [])
            if score > 0:
                item["score"] = score
                scored.append(item)
                continue
            if scope == "user":
                fallback_candidates.append(item)
        limit = collection.max_results if collection.max_results is not None else top_k
        effective_limit = min(top_k, limit)
        ranked = sorted(scored, key=lambda item: float(item["score"]), reverse=True)[:effective_limit]

        if scope == "user" and len(ranked) < effective_limit and fallback_candidates:
            seen_chunk_ids = {str(item["chunk_id"]) for item in ranked}
            fallback_ranked = sorted(
                fallback_candidates,
                key=lambda item: (float(item["sort_epoch"]), -int(item["chunk_index"])),
                reverse=True,
            )
            for item in fallback_ranked:
                if str(item["chunk_id"]) in seen_chunk_ids:
                    continue
                ranked.append(item)
                seen_chunk_ids.add(str(item["chunk_id"]))
                if len(ranked) >= effective_limit:
                    break

        return [
            {
                "scope_type": str(item["scope_type"]),
                "document_id": str(item["document_id"]),
                "chunk_id": str(item["chunk_id"]),
                "filename": str(item["filename"]),
                "chunk_text": str(item["chunk_text"]),
                "score": float(item["score"]),
            }
            for item in ranked
        ]

    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * VECTOR_SIZE
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = digest[0] % VECTOR_SIZE
            vector[bucket] += 1.0
        magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / magnitude for value in vector]

    def _cosine(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        return float(sum(a * b for a, b in zip(left, right)))
