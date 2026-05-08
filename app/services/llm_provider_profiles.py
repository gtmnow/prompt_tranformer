from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.core.rules import get_rule_registry


logger = logging.getLogger("prompt_transformer.llm_provider_profiles")


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
    supports_image_generation: bool = False
    supports_temperature: bool | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    model_is_malformed: bool = False
    used_fallback_model: bool = False
    supports_web_search: bool = False


class LlmProviderProfileService:
    _discovered_models: dict[str, set[str]] = {}

    def __init__(self) -> None:
        self.rules = get_rule_registry().llm_provider_profiles

    @classmethod
    def register_discovered_models(
        cls,
        provider: str,
        discovered_models: set[str],
    ) -> None:
        normalized_provider = provider.strip().casefold()
        cls._discovered_models[normalized_provider] = {
            str(model).strip() for model in discovered_models if str(model).strip()
        }

    def resolve(self, provider: str, model: str) -> ResolvedLlmProviderProfile:
        normalized_provider = self._normalize_provider(provider)
        default_profile = self.rules.get("default", {})
        providers = self.rules.get("providers", {})
        provider_profile = providers.get(normalized_provider)
        if not provider_profile:
            raise ValueError(f"Unsupported provider profile: {provider}")

        provider_profile = dict(provider_profile)
        models = provider_profile.get("models", {})
        if not isinstance(models, dict):
            models = {}
        discovered_models = self._discovered_models.get(normalized_provider, set())
        models = {**models, **{model: {} for model in discovered_models if model not in models}}

        resolved_model, model_is_malformed, used_fallback_model = self._resolve_model(
            provider=normalized_provider,
            configured_models=models,
            configured_default_model=provider_profile.get("default_model"),
            requested_model=model,
        )
        model_profile = models.get(resolved_model, {})

        merged = {**default_profile, **provider_profile, **model_profile}
        merged.pop("models", None)
        merged.pop("default_model", None)

        return ResolvedLlmProviderProfile(
            provider=normalized_provider,
            requested_model=model,
            resolved_model=resolved_model,
            model_is_malformed=model_is_malformed,
            used_fallback_model=used_fallback_model,
            api_family=str(merged.get("api_family") or default_profile.get("api_family") or "responses"),
            endpoint_path=str(merged.get("endpoint_path") or default_profile.get("endpoint_path") or "/responses"),
            auth_scheme=str(merged.get("auth_scheme") or default_profile.get("auth_scheme") or "bearer"),
            auth_header_name=self._normalize_optional_string(merged.get("auth_header_name")),
            version_header_name=self._normalize_optional_string(merged.get("version_header_name")),
            version_header_value=self._normalize_optional_string(merged.get("version_header_value")),
            json_mode=str(merged.get("json_mode") or default_profile.get("json_mode") or "prompt_only"),
            token_parameter=str(merged.get("token_parameter") or default_profile.get("token_parameter") or "max_output_tokens"),
            supports_system_prompt=bool(merged.get("supports_system_prompt", True)),
            supports_image_generation=bool(merged.get("supports_image_generation", False)),
            supports_temperature=None
            if merged.get("supports_temperature") is None
            else bool(merged.get("supports_temperature")),
            request_timeout_seconds=float(merged.get("request_timeout_seconds") or default_profile.get("request_timeout_seconds") or 15.0),
            supports_web_search=bool(merged.get("supports_web_search", False)),
            raw=merged,
        )

    def list_supported_providers(self) -> list[str]:
        return sorted(self._get_provider_configs().keys())

    def list_supported_models(self, provider: str) -> list[str]:
        normalized_provider = self._normalize_provider(provider)
        providers = self._get_provider_configs()
        provider_profile = providers.get(normalized_provider, {})
        if not provider_profile:
            return []

        models = provider_profile.get("models", {})
        if not isinstance(models, dict):
            models = {}
        discovered_models = self._discovered_models.get(normalized_provider, set())
        models = {**models, **{model: {} for model in discovered_models if model not in models}}
        if not isinstance(models, dict):
            return []
        return sorted(models.keys())

    def resolve_default_model(self, provider: str) -> str:
        normalized_provider = self._normalize_provider(provider)
        providers = self._get_provider_configs()
        provider_profile = providers.get(normalized_provider, {})
        if not provider_profile:
            raise ValueError(f"Unsupported provider profile: {provider}")

        default_model = provider_profile.get("default_model")
        models = provider_profile.get("models", {})
        if not isinstance(models, dict):
            models = {}
        discovered_models = self._discovered_models.get(normalized_provider, set())
        models = {**models, **{model: {} for model in discovered_models if model not in models}}

        if not isinstance(default_model, str) or not default_model.strip():
            fallback_model = self._lowest_version_model(models.keys())
            logger.warning(
                "No default model configured for provider '%s'. Using fallback model '%s'.",
                normalized_provider,
                fallback_model,
            )
            return fallback_model

        resolved_default, _, used_fallback_model = self._resolve_model(
            provider=normalized_provider,
            configured_models=models,
            configured_default_model=default_model,
            requested_model=default_model,
        )
        if used_fallback_model:
            logger.warning(
                "Configured default model '%s' is not available for provider '%s'. "
                "Using fallback model '%s' instead.",
                default_model,
                normalized_provider,
                resolved_default,
            )

        return resolved_default

    def _resolve_model(
        self,
        *,
        provider: str,
        configured_models: dict[str, Any],
        configured_default_model: object,
        requested_model: str,
    ) -> tuple[str, bool, bool]:
        requested = (requested_model or "").strip()
        if not requested:
            raise ValueError("Requested model must be a non-empty string.")

        normalized_requested = self._normalize_model_string(requested)
        normalized_model_lookup = {
            self._normalize_model_string(model_name): model_name
            for model_name in configured_models.keys()
            if isinstance(model_name, str)
        }

        if requested in configured_models:
            return requested, False, False

        if normalized_requested in normalized_model_lookup:
            normalized_match = normalized_model_lookup[normalized_requested]
            if normalized_match != requested:
                logger.warning(
                    "Corrected malformed model for provider '%s': requested '%s' -> resolved '%s'.",
                    provider,
                    requested,
                    normalized_match,
                )
            return normalized_match, True, False

        close_matches = difflib.get_close_matches(
            normalized_requested,
            normalized_model_lookup.keys(),
            n=1,
            cutoff=0.84,
        )
        if close_matches:
            match = normalized_model_lookup[close_matches[0]]
            logger.warning(
                "Corrected malformed model for provider '%s': requested '%s' -> resolved '%s'.",
                provider,
                requested,
                match,
            )
            return match, True, False

        fallback_model = self._resolve_default_model(
            provider=provider,
            configured_models=configured_models,
            configured_default_model=configured_default_model,
        )
        logger.warning(
            "Unsupported model '%s' for provider '%s'. Falling back to '%s'.",
            requested,
            provider,
            fallback_model,
        )
        return fallback_model, True, True

    def _resolve_default_model(
        self,
        *,
        provider: str,
        configured_models: dict[str, Any],
        configured_default_model: object,
    ) -> str:
        if not configured_models:
            raise ValueError(f"No models configured for provider '{provider}'.")

        normalized_default = self._normalize_optional_string(configured_default_model)
        if normalized_default and normalized_default in configured_models:
            return normalized_default

        if normalized_default and normalized_default not in configured_models:
            logger.warning(
                "Configured default model '%s' for provider '%s' is not currently supported.",
                normalized_default,
                provider,
            )

        fallback_model = self._lowest_version_model(configured_models.keys())
        logger.warning(
            "default model for %s not available",
            provider,
        )
        return fallback_model

    def _lowest_version_model(self, models: set | list[str] | dict[str, Any]) -> str:
        if isinstance(models, dict):
            model_names = list(models.keys())
        else:
            model_names = list(models)

        if not model_names:
            raise ValueError("No model candidates available.")

        return sorted(model_names, key=lambda item: (self._model_version_key(item), item))[0]

    def _model_version_key(self, model: str) -> tuple[int, ...]:
        return tuple(
            int(part)
            for part in re.findall(r"\d+", str(model))
            if part.isdigit()
        )

    def _get_provider_configs(self) -> dict[str, dict[str, Any]]:
        providers = self.rules.get("providers", {})
        normalized = {}
        for provider_name, provider_profile in providers.items():
            if isinstance(provider_name, str) and isinstance(provider_profile, dict):
                normalized[provider_name.casefold()] = provider_profile
        return normalized

    def _normalize_provider(self, provider: str) -> str:
        return (provider or "").strip().casefold()

    def _normalize_model_string(self, model: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", str(model).strip().lower())
        normalized = re.sub(r"-+", "-", normalized)
        return normalized.strip("-")

    def _normalize_optional_string(self, value: object) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None
