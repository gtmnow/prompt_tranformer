from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.prompt_score import ConversationPromptScore
from app.schemas.transform import ConversationEnforcement, ConversationRequirement, ConversationScoreResponse, ConversationState


class ConversationScoreService:
    def __init__(self, db_session: Session):
        self.db_session = db_session

    def get_conversation_score(self, *, conversation_id: str, user_id_hash: str) -> ConversationScoreResponse:
        score_row = (
            self.db_session.query(ConversationPromptScore)
            .filter_by(conversation_id=conversation_id, user_id_hash=user_id_hash)
            .one_or_none()
        )
        if score_row is None:
            raise ValueError("Conversation score not found.")

        score_details = score_row.score_details_json or {}
        raw_requirements = score_details.get("raw_requirements", score_details.get("requirements", {}))
        conversation = None
        if isinstance(raw_requirements, dict):
            requirements = {}
            for field_name in ("who", "task", "context", "output"):
                raw_requirement = raw_requirements.get(field_name)
                if isinstance(raw_requirement, dict):
                    requirements[field_name] = ConversationRequirement.model_validate(raw_requirement)
                else:
                    requirements[field_name] = ConversationRequirement(
                        value=None,
                        status=getattr(score_row, f"{field_name}_status"),
                    )
            conversation = ConversationState(
                conversation_id=score_row.conversation_id,
                requirements=requirements,
                enforcement=ConversationEnforcement(
                    level=str(score_details.get("enforcement_level", score_row.enforcement_level)),
                    status=str(score_details.get("enforcement_status", "not_evaluated")),
                    missing_fields=[
                        str(field_name)
                        for field_name in score_details.get("missing_fields", [])
                        if isinstance(field_name, str)
                    ],
                    last_evaluated_at=str(score_details.get("calculated_at", score_row.last_scored_at.isoformat())),
                ),
            )

        return ConversationScoreResponse(
            conversation_id=score_row.conversation_id,
            user_id_hash=score_row.user_id_hash,
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
            conversation=conversation,
        )
