from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.prompt_score import ConversationPromptScore
from app.schemas.transform import ConversationScoreResponse


class ConversationScoreService:
    def __init__(self, db_session: Session):
        self.db_session = db_session

    def get_conversation_score(self, *, conversation_id: str, user_id: str) -> ConversationScoreResponse:
        score_row = (
            self.db_session.query(ConversationPromptScore)
            .filter_by(conversation_id=conversation_id, user_id_hash=user_id)
            .one_or_none()
        )
        if score_row is None:
            raise ValueError("Conversation score not found.")

        return ConversationScoreResponse(
            conversation_id=score_row.conversation_id,
            user_id=score_row.user_id_hash,
            scoring_version=score_row.scoring_version,
            initial_score=score_row.initial_score,
            best_score=score_row.best_score,
            final_score=score_row.final_score,
            initial_llm_score=score_row.initial_llm_score,
            best_llm_score=score_row.best_llm_score,
            final_llm_score=score_row.final_llm_score,
            structural_score=score_row.final_score,
            improvement_score=score_row.improvement_score,
            best_improvement_score=score_row.best_improvement_score,
            last_scored_at=score_row.last_scored_at.isoformat(),
        )
