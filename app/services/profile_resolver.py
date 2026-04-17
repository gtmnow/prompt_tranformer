from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from app.core.rules import get_rule_registry
from app.models.profile import FinalProfile


PROFILE_FIELDS = (
    "structure",
    "answer_first",
    "tone_directness",
    "detail_level",
    "ambiguity_reduction",
    "exploration_level",
    "context_loading",
)


@dataclass(frozen=True)
class ResolvedPersona:
    values: dict[str, float]
    source: str
    profile_version: str
    prompt_enforcement_level: str
    compliance_check_enabled: bool
    pii_check_enabled: bool


class ProfileResolver:
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.rule_registry = get_rule_registry()

    def resolve(self, user_id: str, summary_type: Optional[int]) -> ResolvedPersona:
        if summary_type is not None:
            return self._from_summary_override(summary_type)

        db_profile = self.db_session.get(FinalProfile, user_id)
        if db_profile is not None:
            return ResolvedPersona(
                values={field: float(getattr(db_profile, field)) for field in PROFILE_FIELDS},
                source="db_profile",
                profile_version=db_profile.profile_version,
                prompt_enforcement_level=db_profile.prompt_enforcement_level,
                compliance_check_enabled=bool(db_profile.compliance_check_enabled),
                pii_check_enabled=bool(db_profile.pii_check_enabled),
            )

        return self._generic_default()

    def _from_summary_override(self, summary_type: int) -> ResolvedPersona:
        personas = self.rule_registry.summary_personas.get("summary_types", {})
        persona = personas.get(str(summary_type))
        if persona is None:
            raise ValueError("Invalid summary_type")
        values = {field: float(persona[field]) for field in PROFILE_FIELDS}
        return ResolvedPersona(
            values=values,
            source="summary_override",
            profile_version=f"summary_type_{summary_type}",
            prompt_enforcement_level="none",
            compliance_check_enabled=False,
            pii_check_enabled=False,
        )

    def _generic_default(self) -> ResolvedPersona:
        defaults = self.rule_registry.summary_personas["generic_default"]
        values = {field: float(defaults[field]) for field in PROFILE_FIELDS}
        return ResolvedPersona(
            values=values,
            source="generic_default",
            profile_version="generic_default",
            prompt_enforcement_level="none",
            compliance_check_enabled=False,
            pii_check_enabled=False,
        )
