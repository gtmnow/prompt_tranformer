from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import zlib
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.rag import RagChunk, RagCollection, RagDocument, RagDocumentBlob
from app.services.rag_limit_resolver import RagLimitResolver

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None


SUPPORTED_MEDIA_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
    "text/csv",
}
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
VECTOR_SIZE = 96
CHUNK_WORD_TARGET = 140
CHUNK_WORD_OVERLAP = 30


class RagIngestionService:
    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session
        self.limit_resolver = RagLimitResolver(db_session)

    def upload_document(
        self,
        *,
        scope: str,
        tenant_id: str,
        user_id_hash: str | None,
        filename: str,
        media_type: str | None,
        content_bytes: bytes,
        uploaded_by_admin_user_id: str | None = None,
        uploaded_by_user_id_hash: str | None = None,
    ) -> RagDocument:
        normalized_media_type = self._normalize_media_type(filename=filename, media_type=media_type)
        self._validate_supported_type(filename=filename, media_type=normalized_media_type)
        limits = (
            self.limit_resolver.resolve_tenant_limits(tenant_id)
            if scope == "tenant"
            else self.limit_resolver.resolve_user_limits(tenant_id, user_id_hash or "")
        )
        usage = self.limit_resolver.summarize_usage(tenant_id=tenant_id, scope=scope, user_id_hash=user_id_hash)
        self._enforce_admission_limits(
            limits=limits,
            usage=usage,
            incoming_size=len(content_bytes),
        )

        collection = self._get_or_create_collection(scope=scope, tenant_id=tenant_id, user_id_hash=user_id_hash)
        now = datetime.now(timezone.utc)
        document = RagDocument(
            id=str(uuid.uuid4()),
            collection_id=collection.id,
            scope_type=scope,
            tenant_id=tenant_id,
            user_id_hash=user_id_hash,
            filename=filename,
            media_type=normalized_media_type,
            size_bytes=len(content_bytes),
            status="processing",
            status_message=None,
            sha256=hashlib.sha256(content_bytes).hexdigest(),
            source_kind="database_blob",
            uploaded_by_admin_user_id=uploaded_by_admin_user_id,
            uploaded_by_user_id_hash=uploaded_by_user_id_hash,
            uploaded_at=now,
            created_at=now,
            updated_at=now,
        )
        self.db_session.add(document)
        self.db_session.add(RagDocumentBlob(document_id=document.id, content_bytes=content_bytes))
        self.db_session.flush()

        try:
            self._process_document(document=document, content_bytes=content_bytes, limits=limits)
        except ValueError as exc:
            document.status = "failed"
            document.status_message = str(exc)
            document.processed_at = datetime.now(timezone.utc)
        self.db_session.commit()
        self.db_session.refresh(document)
        return document

    def reprocess_document(self, document_id: str) -> RagDocument:
        document = self.db_session.get(RagDocument, document_id)
        if document is None:
            raise ValueError("Document not found")
        blob = self.db_session.get(RagDocumentBlob, document_id)
        if blob is None:
            raise ValueError("Document blob not found")
        limits = (
            self.limit_resolver.resolve_tenant_limits(document.tenant_id)
            if document.scope_type == "tenant"
            else self.limit_resolver.resolve_user_limits(document.tenant_id, document.user_id_hash or "")
        )
        document.status = "processing"
        document.status_message = None
        self.db_session.execute(delete(RagChunk).where(RagChunk.document_id == document.id))
        try:
            self._process_document(document=document, content_bytes=blob.content_bytes, limits=limits)
        except ValueError as exc:
            document.status = "failed"
            document.status_message = str(exc)
            document.processed_at = datetime.now(timezone.utc)
        self.db_session.commit()
        self.db_session.refresh(document)
        return document

    def delete_document(self, document_id: str, *, tenant_id: str, user_id_hash: str | None = None) -> None:
        document = self.db_session.get(RagDocument, document_id)
        if document is None:
            raise ValueError("Document not found")
        if document.tenant_id != tenant_id:
            raise ValueError("Document does not belong to the requested tenant")
        if document.scope_type == "user" and document.user_id_hash != user_id_hash:
            raise ValueError("Document does not belong to the requested user")
        self.db_session.delete(document)
        self.db_session.commit()

    def update_collection(
        self,
        *,
        collection_id: str,
        retrieval_enabled: bool | None,
        is_active: bool | None,
        max_results: int | None,
    ) -> RagCollection:
        collection = self.db_session.get(RagCollection, collection_id)
        if collection is None:
            raise ValueError("Collection not found")
        if retrieval_enabled is not None:
            collection.retrieval_enabled = retrieval_enabled
        if is_active is not None:
            collection.is_active = is_active
        if max_results is not None:
            collection.max_results = max_results
        self.db_session.commit()
        self.db_session.refresh(collection)
        return collection

    def list_documents(self, *, scope: str, tenant_id: str, user_id_hash: str | None = None) -> tuple[RagCollection, list[RagDocument]]:
        collection = self._get_or_create_collection(scope=scope, tenant_id=tenant_id, user_id_hash=user_id_hash)
        documents = self.db_session.scalars(
            select(RagDocument)
            .where(RagDocument.collection_id == collection.id)
            .order_by(RagDocument.uploaded_at.desc())
        ).all()
        return collection, documents

    def _get_or_create_collection(self, *, scope: str, tenant_id: str, user_id_hash: str | None) -> RagCollection:
        query = select(RagCollection).where(
            RagCollection.scope_type == scope,
            RagCollection.tenant_id == tenant_id,
            RagCollection.user_id_hash == (user_id_hash if scope == "user" else None),
        )
        collection = self.db_session.scalars(query.limit(1)).first()
        if collection is not None:
            return collection
        collection = RagCollection(
            id=str(uuid.uuid4()),
            scope_type=scope,
            tenant_id=tenant_id,
            user_id_hash=user_id_hash if scope == "user" else None,
            name="Personal Context" if scope == "user" else "Organization Knowledge",
            is_active=True,
            retrieval_enabled=True,
            max_results=None,
        )
        self.db_session.add(collection)
        self.db_session.flush()
        return collection

    def _enforce_admission_limits(self, *, limits: Any, usage: Any, incoming_size: int) -> None:
        if incoming_size > limits.max_file_bytes:
            raise ValueError("File exceeds the maximum size for this knowledge library")
        if usage.document_count >= limits.max_document_count:
            raise ValueError("Document limit reached for this knowledge library")
        if usage.total_bytes + incoming_size > limits.max_total_bytes:
            raise ValueError("Storage limit reached for this knowledge library")

    def _process_document(self, *, document: RagDocument, content_bytes: bytes, limits: Any) -> None:
        text = self._extract_text(document.filename, document.media_type, content_bytes)
        normalized = self._normalize_text(text)
        if not normalized:
            raise ValueError("No readable text could be extracted from this document")
        extracted_bytes = len(normalized.encode("utf-8"))
        if extracted_bytes > limits.max_extracted_text_bytes:
            raise ValueError("Document exceeds the maximum extracted text size")
        chunks = self._chunk_text(normalized)
        if len(chunks) > limits.max_chunks_per_document:
            raise ValueError("Document exceeds the maximum chunk count")
        document.extracted_text = normalized
        document.metadata_json = {
            "extracted_text_bytes": extracted_bytes,
            "chunk_count": len(chunks),
        }
        for index, chunk in enumerate(chunks):
            self.db_session.add(
                RagChunk(
                    id=str(uuid.uuid4()),
                    document_id=document.id,
                    chunk_index=index,
                    chunk_text=chunk,
                    token_count=len(chunk.split()),
                    embedding_vector=self._vectorize(chunk),
                )
            )
        document.status = "ready"
        document.status_message = None
        document.processed_at = datetime.now(timezone.utc)

    def _normalize_media_type(self, *, filename: str, media_type: str | None) -> str:
        if media_type and media_type.strip():
            return media_type.strip().lower()
        suffix = Path(filename).suffix.lower()
        return {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".csv": "text/csv",
        }.get(suffix, "application/octet-stream")

    def _validate_supported_type(self, *, filename: str, media_type: str) -> None:
        if media_type in SUPPORTED_MEDIA_TYPES:
            return
        if Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS:
            return
        raise ValueError("Unsupported document type")

    def _extract_text(self, filename: str, media_type: str, content_bytes: bytes) -> str:
        suffix = Path(filename).suffix.lower()
        if media_type in {"text/plain", "text/markdown"} or suffix in {".txt", ".md"}:
            return content_bytes.decode("utf-8", errors="ignore")
        if media_type == "text/csv" or suffix == ".csv":
            return self._extract_csv(content_bytes)
        if media_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or suffix == ".docx":
            return self._extract_docx(content_bytes)
        if media_type == "application/pdf" or suffix == ".pdf":
            return self._extract_pdf(content_bytes)
        raise ValueError("Unsupported document type")

    def _extract_csv(self, content_bytes: bytes) -> str:
        decoded = content_bytes.decode("utf-8", errors="ignore")
        rows = []
        reader = csv.reader(io.StringIO(decoded))
        for row in reader:
            rows.append(" | ".join(cell.strip() for cell in row if cell.strip()))
        return "\n".join(row for row in rows if row.strip())

    def _extract_docx(self, content_bytes: bytes) -> str:
        with zipfile.ZipFile(io.BytesIO(content_bytes)) as archive:
            xml_bytes = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml_bytes)
        texts = [node.text.strip() for node in root.iter() if node.text and node.text.strip()]
        return "\n".join(texts)

    def _extract_pdf(self, content_bytes: bytes) -> str:
        pypdf_text = self._extract_pdf_with_pypdf(content_bytes)
        if pypdf_text:
            return pypdf_text

        operator_text = self._extract_pdf_text_operators(content_bytes)
        if operator_text:
            return operator_text

        decoded = content_bytes.decode("latin-1", errors="ignore")
        candidates = re.findall(r"[A-Za-z0-9][A-Za-z0-9 ,.;:()'\"/%_-]{3,}", decoded)
        return "\n".join(candidates[:5000])

    def _normalize_text(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        normalized = re.sub(r"[ \t]+", " ", normalized)
        return normalized.strip()

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        step = max(1, CHUNK_WORD_TARGET - CHUNK_WORD_OVERLAP)
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + CHUNK_WORD_TARGET]).strip()
            if chunk:
                chunks.append(chunk)
            if start + CHUNK_WORD_TARGET >= len(words):
                break
        return chunks

    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * VECTOR_SIZE
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vector
        frequencies: dict[str, int] = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        for token, frequency in frequencies.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:2], "big") % VECTOR_SIZE
            weight = 1.0 + math.log1p(frequency)
            vector[bucket] += weight
        for left, right in zip(tokens, tokens[1:]):
            digest = hashlib.sha256(f"{left}_{right}".encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:2], "big") % VECTOR_SIZE
            vector[bucket] += 0.7
        for token in tokens:
            if len(token) < 5:
                continue
            for index in range(0, len(token) - 2):
                trigram = token[index : index + 3]
                digest = hashlib.sha256(f"tri:{trigram}".encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:2], "big") % VECTOR_SIZE
                vector[bucket] += 0.15
        magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [round(value / magnitude, 6) for value in vector]

    def _extract_pdf_with_pypdf(self, content_bytes: bytes) -> str:
        if PdfReader is None:
            return ""
        try:
            reader = PdfReader(io.BytesIO(content_bytes))
            pages = []
            for page in reader.pages:
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
        except Exception:
            return ""

    def _extract_pdf_text_operators(self, content_bytes: bytes) -> str:
        decoded = content_bytes.decode("latin-1", errors="ignore")
        stream_matches = re.findall(r"stream\r?\n(.*?)\r?\nendstream", decoded, flags=re.S)
        blocks: list[str] = []
        for stream in stream_matches:
            raw_bytes = stream.encode("latin-1", errors="ignore")
            candidates = [raw_bytes]
            try:
                candidates.append(zlib.decompress(raw_bytes))
            except Exception:
                pass
            for candidate in candidates:
                candidate_text = self._extract_text_from_pdf_stream(candidate)
                if candidate_text:
                    blocks.append(candidate_text)
        return "\n\n".join(blocks)

    def _extract_text_from_pdf_stream(self, stream_bytes: bytes) -> str:
        text = stream_bytes.decode("latin-1", errors="ignore")
        blocks = re.findall(r"BT(.*?)ET", text, flags=re.S)
        parts: list[str] = []
        for block in blocks:
            literal_strings = [self._decode_pdf_literal(value) for value in re.findall(r"\((.*?)\)\s*Tj", block, flags=re.S)]
            array_strings: list[str] = []
            for array_block in re.findall(r"\[(.*?)\]\s*TJ", block, flags=re.S):
                array_strings.extend(self._decode_pdf_literal(value) for value in re.findall(r"\((.*?)\)", array_block, flags=re.S))
            merged = " ".join(segment.strip() for segment in literal_strings + array_strings if segment.strip())
            if merged:
                parts.append(merged)
        return "\n".join(parts)

    def _decode_pdf_literal(self, value: str) -> str:
        value = value.replace("\\(", "(").replace("\\)", ")").replace("\\n", "\n").replace("\\r", " ")
        value = value.replace("\\t", " ").replace("\\\\", "\\")
        value = re.sub(r"\\([0-7]{1,3})", lambda match: chr(int(match.group(1), 8)), value)
        return value
