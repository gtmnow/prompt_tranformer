from __future__ import annotations

import logging

from app.core.config import get_settings
from app.services.llm_adapters.registry import LlmAdapterRegistry
from app.services.llm_provider_profiles import LlmProviderProfileService


logger = logging.getLogger("prompt_transformer.llm_model_catalog")


class LlmModelCatalogService:
    def __init__(self) -> None:
        self.adapter_registry = LlmAdapterRegistry()

    def discover_openai_models(self, provider: str, base_url: str, api_key: str | None) -> set[str]:
        provider_normalized = provider.casefold().strip()
        if not api_key:
            return set()

        try:
            adapter = self.adapter_registry.resolve(provider_normalized)
        except ValueError:
            return set()

        discover_method = getattr(adapter, "discover_models", None)
        if discover_method is None:
            return set()

        try:
            return discover_method(base_url=base_url, api_key=api_key)
        except Exception:
            logger.warning(
                "Model discovery failed for provider %s during bootstrap.",
                provider_normalized,
                exc_info=True,
            )
            return set()

    def discover_and_register(self) -> None:
        settings = get_settings()
        profile_service = LlmProviderProfileService()
        provider_key_lookup = {
            "openai": settings.structure_evaluator_api_key,
            "xai": settings.xai_api_key,
            "azure_openai": settings.azure_openai_api_key,
            "anthropic": settings.anthropic_api_key,
        }
        provider_base_urls = {
            "openai": "https://api.openai.com/v1",
            "xai": "https://api.x.ai/v1",
            "azure_openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1",
        }

        providers = profile_service.list_supported_providers()
        for provider in providers:
            discovered = self.discover_openai_models(
                provider=provider,
                base_url=provider_base_urls.get(provider, provider_base_urls["openai"]),
                api_key=provider_key_lookup.get(provider, ""),
            )
            if not discovered:
                continue
            configured = set(profile_service.list_supported_models(provider))
            additional = sorted(discovered - configured)
            if additional:
                logger.info(
                    "Discovered %d new models for provider '%s': %s",
                    len(additional),
                    provider,
                    ", ".join(additional),
                )
            LlmProviderProfileService.register_discovered_models(provider, discovered)
