from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class PromptTransformRequest(Base):
    __tablename__ = "prompt_transform_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    conversation_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True, default="")
    user_id_hash: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    raw_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    transformed_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    task_type: Mapped[str] = mapped_column(String(100), nullable=False)
    result_type: Mapped[str] = mapped_column(String(50), nullable=False, default="transformed")
    coaching_tip: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    blocking_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    target_provider: Mapped[str] = mapped_column(String(100), nullable=False)
    target_model: Mapped[str] = mapped_column(String(100), nullable=False)
    persona_source: Mapped[str] = mapped_column(String(100), nullable=False)
    used_fallback_model: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    enforcement_level: Mapped[str] = mapped_column(String(20), nullable=False, default="none")
    compliance_check_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    pii_check_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    conversation_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    findings_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
