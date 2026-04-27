from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models.profile import (
    BehavioralAdjProfile,
    BrainChemistryProfile,
    EnvironmentDetailsProfile,
    FinalProfile,
    TypeDetailProfile,
)
from app.services.profile_resolver import PROFILE_FIELDS, ProfileResolver


CONTROL_FIELDS = (
    "prompt_enforcement_level",
    "compliance_check_enabled",
    "pii_check_enabled",
)

LAYER_MODELS = (
    ("type_detail", TypeDetailProfile),
    ("brain_chemistry", BrainChemistryProfile),
    ("environment_details", EnvironmentDetailsProfile),
    ("behaviorial_adj", BehavioralAdjProfile),
)


@dataclass(frozen=True)
class FinalProfileBuildResult:
    user_id_hash: str
    profile_version: str
    applied_layers: tuple[str, ...]


class ProfileBuilder:
    def __init__(self, db_session: Session):
        self.db_session = db_session

    def recompute_final_profile(self, user_id_hash: str) -> FinalProfileBuildResult:
        base_profile = self.db_session.get(TypeDetailProfile, user_id_hash)
        if base_profile is None:
            raise ValueError(f"Foundational type_detail profile is required for user_id_hash={user_id_hash}")

        payload = self._profile_to_payload(base_profile)
        applied_layers = ["type_detail"]

        for layer_name, model in LAYER_MODELS[1:]:
            layer_profile = self.db_session.get(model, user_id_hash)
            if layer_profile is None:
                continue
            payload.update(self._profile_to_payload(layer_profile))
            applied_layers.append(layer_name)

        payload["profile_version"] = self._build_profile_version(base_profile.profile_version, applied_layers)

        final_profile = self.db_session.get(FinalProfile, user_id_hash)
        if final_profile is None:
            final_profile = FinalProfile(user_id_hash=user_id_hash, **payload)
            self.db_session.add(final_profile)
        else:
            for field_name, value in payload.items():
                setattr(final_profile, field_name, value)

        self.db_session.flush()
        ProfileResolver.invalidate_cache(user_id_hash)
        return FinalProfileBuildResult(
            user_id_hash=user_id_hash,
            profile_version=str(payload["profile_version"]),
            applied_layers=tuple(applied_layers),
        )

    @staticmethod
    def _profile_to_payload(profile: TypeDetailProfile) -> dict[str, float | str | bool]:
        payload = {field: float(getattr(profile, field)) for field in PROFILE_FIELDS}
        payload.update({field: getattr(profile, field) for field in CONTROL_FIELDS})
        payload["profile_version"] = profile.profile_version
        return payload

    @staticmethod
    def _build_profile_version(base_profile_version: str, applied_layers: list[str]) -> str:
        if len(applied_layers) == 1:
            return base_profile_version
        return f"{base_profile_version}+layers"
