from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.rules import get_rule_registry


@dataclass(frozen=True)
class ResolvedLlmProviderProfile:
    provider: str
    requested_model: str
    resolved_model: str
    api_family: str
    endpoint_path: str
    auth_scheme: str
    auth_header_name: str | None
    version_header_name: str | None
    version_header_value: str | None
    json_mode: str
    token_parameter: str
    supports_system_prompt: bool
    request_timeout_seconds: float
    raw: dict[str, Any]


class LlmProviderProfileService:
    def __init__(self) -> None:
        self.rules = get_rule_registry().llm_provider_profiles

    def resolve(self, provider: str, model: str) -> ResolvedLlmProviderProfile:
        default_profile = self.rules.get("default", {})
        providers = self.rules.get("providers", {})
        provider_profile = providers.get(provider, {})
        if not provider_profile:
            raise ValueError(f"Unsupported provider profile: {provider}")

        models = provider_profile.get("models", {})
        resolved_model = model if model in models else provider_profile.get("default_model", model)
        model_profile = models.get(resolved_model, {})

        merged = {**default_profile, **provider_profile, **model_profile}
        merged.pop("models", None)
        merged.pop("default_model", None)

        return ResolvedLlmProviderProfile(
            provider=provider,
            requested_model=model,
            resolved_model=resolved_model,
            api_family=str(merged.get("api_family") or default_profile.get("api_family") or "responses"),
            endpoint_path=str(merged.get("endpoint_path") or default_profile.get("endpoint_path") or "/responses"),
            auth_scheme=str(merged.get("auth_scheme") or default_profile.get("auth_scheme") or "bearer"),
            auth_header_name=self._normalize_optional_string(merged.get("auth_header_name")),
            version_header_name=self._normalize_optional_string(merged.get("version_header_name")),
            version_header_value=self._normalize_optional_string(merged.get("version_header_value")),
            json_mode=str(merged.get("json_mode") or default_profile.get("json_mode") or "prompt_only"),
            token_parameter=str(merged.get("token_parameter") or default_profile.get("token_parameter") or "max_output_tokens"),
            supports_system_prompt=bool(merged.get("supports_system_prompt", True)),
            request_timeout_seconds=float(merged.get("request_timeout_seconds") or default_profile.get("request_timeout_seconds") or 15.0),
            raw=merged,
        )

    def _normalize_optional_string(self, value: object) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None
