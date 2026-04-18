from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.api.deps import require_service_auth
from app.db.session import get_db
from app.schemas.transform import (
    ConversationScoreResponse,
    TransformPromptRequest,
    TransformPromptResponse,
)
from app.services.conversation_scores import ConversationScoreService
from app.services.transformer_engine import TransformerEngine

router = APIRouter(prefix="/api", tags=["prompt-transformer"])


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


@router.get("/conversation_scores/{conversation_id}", response_model=ConversationScoreResponse)
def get_conversation_score(
    conversation_id: str,
    user_id: str,
    _: str = Depends(require_service_auth),
    db: Session = Depends(get_db),
) -> ConversationScoreResponse:
    try:
        service = ConversationScoreService(db_session=db)
        return service.get_conversation_score(conversation_id=conversation_id, user_id=user_id)
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
