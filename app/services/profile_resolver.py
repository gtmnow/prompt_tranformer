from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
import time
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import get_settings
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
    _cache: dict[str, tuple[float, ResolvedPersona]] = {}
    _cache_lock = Lock()

    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.settings = get_settings()
        self.rule_registry = get_rule_registry()

    def resolve(self, user_id_hash: str, summary_type: Optional[int]) -> ResolvedPersona:
        if summary_type is not None:
            return self._from_summary_override(summary_type)

        cached_persona = self._get_cached_persona(user_id_hash)
        if cached_persona is not None:
            return cached_persona

        db_profile = self.db_session.get(FinalProfile, user_id_hash)
        if db_profile is not None:
            resolved_persona = ResolvedPersona(
                values={field: float(getattr(db_profile, field)) for field in PROFILE_FIELDS},
                source="db_profile",
                profile_version=db_profile.profile_version,
                prompt_enforcement_level=self._normalize_prompt_enforcement_level(db_profile.prompt_enforcement_level),
                compliance_check_enabled=bool(db_profile.compliance_check_enabled),
                pii_check_enabled=bool(db_profile.pii_check_enabled),
            )
            self._set_cached_persona(user_id_hash, resolved_persona)
            return resolved_persona

        resolved_persona = self._generic_default()
        self._set_cached_persona(user_id_hash, resolved_persona)
        return resolved_persona

    def _get_cached_persona(self, user_id_hash: str) -> ResolvedPersona | None:
        if not self.settings.enable_profile_cache:
            return None

        now = time.monotonic()
        with self._cache_lock:
            cached_entry = self._cache.get(user_id_hash)
            if cached_entry is None:
                return None
            expires_at, resolved_persona = cached_entry
            if expires_at <= now:
                self._cache.pop(user_id_hash, None)
                return None
            return resolved_persona

    def _set_cached_persona(self, user_id_hash: str, resolved_persona: ResolvedPersona) -> None:
        if not self.settings.enable_profile_cache:
            return

        expires_at = time.monotonic() + self.settings.profile_cache_ttl_seconds
        with self._cache_lock:
            self._cache[user_id_hash] = (expires_at, resolved_persona)

    @classmethod
    def invalidate_cache(cls, user_id_hash: str) -> None:
        with cls._cache_lock:
            cls._cache.pop(user_id_hash, None)

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

    @staticmethod
    def _normalize_prompt_enforcement_level(value: str | None) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"none", "low", "moderate", "full"}:
            return normalized
        return "none"
