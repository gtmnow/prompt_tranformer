from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Integer, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ConversationPromptScore(Base):
    __tablename__ = "conversation_prompt_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    user_id_hash: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    task_type: Mapped[str] = mapped_column(String(100), nullable=False, default="unknown")
    conversation_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    conversation_ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    conversation_deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    enforcement_level: Mapped[str] = mapped_column(String(20), nullable=False, default="none")
    initial_score: Mapped[int] = mapped_column(Integer, nullable=False)
    best_score: Mapped[int] = mapped_column(Integer, nullable=False)
    final_score: Mapped[int] = mapped_column(Integer, nullable=False)
    initial_llm_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    best_llm_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    final_llm_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    improvement_score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    best_improvement_score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    passed_without_coaching: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reached_policy_complete: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    coaching_turn_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    blocked_turn_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    transformed_turn_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    who_status: Mapped[str] = mapped_column(String(20), nullable=False, default="missing")
    task_status: Mapped[str] = mapped_column(String(20), nullable=False, default="missing")
    context_status: Mapped[str] = mapped_column(String(20), nullable=False, default="missing")
    output_status: Mapped[str] = mapped_column(String(20), nullable=False, default="missing")
    score_details_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    scoring_version: Mapped[str] = mapped_column(String(50), nullable=False, default="v1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
