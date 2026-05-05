from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models.rag import RagCollection, RagDocument, RagQuotaPolicy


@dataclass(frozen=True)
class ResolvedRagQuota:
    scope: str
    policy_source: str
    policy_key: str
    max_file_bytes: int
    max_document_count: int
    max_total_bytes: int
    max_extracted_text_bytes: int
    max_chunks_per_document: int
    max_retrieved_chunks: int
    max_retrieved_chunks_total: int


@dataclass(frozen=True)
class RagUsage:
    document_count: int
    total_bytes: int
    ready_documents: int
    processing_documents: int
    failed_documents: int
    disabled_documents: int


class RagLimitResolver:
    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session

    def resolve_tenant_limits(self, tenant_id: str) -> ResolvedRagQuota:
        tenant_policy = self._find_policy(tenant_id=tenant_id, scope="tenant", user_type=None)
        return self._build_resolved_quota(tenant_policy, scope="tenant")

    def resolve_user_limits(self, tenant_id: str, user_id_hash: str) -> ResolvedRagQuota:
        user_type = self._resolve_user_type(user_id_hash)
        policy = self._find_policy(tenant_id=tenant_id, scope="user", user_type=user_type)
        return self._build_resolved_quota(policy, scope="user")

    def summarize_usage(self, *, tenant_id: str, scope: str, user_id_hash: str | None = None) -> RagUsage:
        collection = self._find_collection(tenant_id=tenant_id, scope=scope, user_id_hash=user_id_hash)
        if collection is None:
            return RagUsage(0, 0, 0, 0, 0, 0)

        documents = self.db_session.scalars(
            select(RagDocument).where(RagDocument.collection_id == collection.id)
        ).all()
        return RagUsage(
            document_count=len(documents),
            total_bytes=sum(max(0, item.size_bytes) for item in documents),
            ready_documents=sum(1 for item in documents if item.status == "ready"),
            processing_documents=sum(1 for item in documents if item.status in {"pending", "processing"}),
            failed_documents=sum(1 for item in documents if item.status == "failed"),
            disabled_documents=sum(1 for item in documents if item.status == "disabled"),
        )

    def _find_policy(self, *, tenant_id: str, scope: str, user_type: str | None) -> RagQuotaPolicy:
        tenant_row = self.db_session.execute(
            text("select service_tier_definition_id from tenants where id = :tenant_id limit 1"),
            {"tenant_id": tenant_id},
        ).mappings().first()
        service_tier_definition_id = str((tenant_row or {}).get("service_tier_definition_id") or "").strip() or None

        candidates = self.db_session.scalars(
            select(RagQuotaPolicy).where(RagQuotaPolicy.is_active.is_(True))
        ).all()

        if scope == "user" and user_type:
            for candidate in candidates:
                if (
                    candidate.scope_target == "tenant_override"
                    and candidate.tenant_id == tenant_id
                    and candidate.user_type == user_type
                ):
                    return candidate
            for candidate in candidates:
                if (
                    candidate.scope_target == "service_tier"
                    and candidate.service_tier_definition_id == service_tier_definition_id
                    and candidate.user_type == user_type
                ):
                    return candidate

        for candidate in candidates:
            if (
                candidate.scope_target == "tenant_override"
                and candidate.tenant_id == tenant_id
                and candidate.user_type is None
            ):
                return candidate

        for candidate in candidates:
            if (
                candidate.scope_target == "service_tier"
                and candidate.service_tier_definition_id == service_tier_definition_id
                and candidate.user_type is None
            ):
                return candidate

        for candidate in candidates:
            if candidate.scope_target == "global_default":
                return candidate
        raise ValueError("No active RAG quota policy found")

    def _build_resolved_quota(self, policy: RagQuotaPolicy, *, scope: str) -> ResolvedRagQuota:
        if scope == "tenant":
            return ResolvedRagQuota(
                scope="tenant",
                policy_source=self._policy_source_label(policy),
                policy_key=policy.policy_key,
                max_file_bytes=policy.org_max_file_bytes,
                max_document_count=policy.org_max_document_count,
                max_total_bytes=policy.org_max_total_bytes,
                max_extracted_text_bytes=policy.org_max_extracted_text_bytes,
                max_chunks_per_document=policy.org_max_chunks_per_document,
                max_retrieved_chunks=policy.org_max_retrieved_chunks,
                max_retrieved_chunks_total=policy.max_retrieved_chunks_total,
            )
        return ResolvedRagQuota(
            scope="user",
            policy_source=self._policy_source_label(policy),
            policy_key=policy.policy_key,
            max_file_bytes=policy.user_max_file_bytes,
            max_document_count=policy.user_max_document_count,
            max_total_bytes=policy.user_max_total_bytes,
            max_extracted_text_bytes=policy.user_max_extracted_text_bytes,
            max_chunks_per_document=policy.user_max_chunks_per_document,
            max_retrieved_chunks=policy.user_max_retrieved_chunks,
            max_retrieved_chunks_total=policy.max_retrieved_chunks_total,
        )

    @staticmethod
    def _policy_source_label(policy: RagQuotaPolicy) -> str:
        return {
            "global_default": "Default",
            "service_tier": "Service Tier",
            "tenant_override": "Organization Override",
        }.get(policy.scope_target, "Default")

    def _resolve_user_type(self, user_id_hash: str) -> str | None:
        row = self.db_session.execute(
            text(
                """
                select initial_user_type
                from user_membership_profiles
                where user_id_hash = :user_id_hash
                order by id desc
                limit 1
                """
            ),
            {"user_id_hash": user_id_hash},
        ).mappings().first()
        value = str((row or {}).get("initial_user_type") or "").strip()
        return value or None

    def _find_collection(self, *, tenant_id: str, scope: str, user_id_hash: str | None) -> RagCollection | None:
        query = select(RagCollection).where(
            RagCollection.scope_type == scope,
            RagCollection.tenant_id == tenant_id,
        )
        if scope == "user":
            query = query.where(RagCollection.user_id_hash == user_id_hash)
        else:
            query = query.where(RagCollection.user_id_hash.is_(None))
        return self.db_session.scalars(query.limit(1)).first()
