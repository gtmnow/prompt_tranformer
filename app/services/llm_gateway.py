from __future__ import annotations

from app.services.llm_adapters.registry import LlmAdapterRegistry
from app.services.llm_provider_profiles import LlmProviderProfileService
from app.services.llm_types import TransformerLlmError, TransformerLlmRequest, TransformerLlmResponse


class LlmGatewayService:
    def __init__(self) -> None:
        self.profiles = LlmProviderProfileService()
        self.registry = LlmAdapterRegistry()

    def invoke(
        self,
        request: TransformerLlmRequest,
    ) -> tuple[TransformerLlmResponse | None, TransformerLlmError | None]:
        profile = self.profiles.resolve(request.provider, request.model)
        adapter = self.registry.resolve(request.provider)
        effective_request = request.model_copy(
            update={
                "model": profile.resolved_model,
                "timeout_seconds": profile.request_timeout_seconds,
            }
        )
        return adapter.invoke(effective_request, profile)
