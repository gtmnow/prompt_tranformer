from __future__ import annotations

from abc import ABC, abstractmethod

from app.services.llm_provider_profiles import ResolvedLlmProviderProfile
from app.services.llm_types import TransformerLlmError, TransformerLlmRequest, TransformerLlmResponse


class BaseLlmAdapter(ABC):
    provider_name: str

    @abstractmethod
    def invoke(
        self,
        request: TransformerLlmRequest,
        profile: ResolvedLlmProviderProfile,
    ) -> tuple[TransformerLlmResponse | None, TransformerLlmError | None]:
        raise NotImplementedError
