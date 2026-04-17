from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.api.deps import require_service_auth
from app.db.session import get_db
from app.schemas.transform import TransformPromptRequest, TransformPromptResponse
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
