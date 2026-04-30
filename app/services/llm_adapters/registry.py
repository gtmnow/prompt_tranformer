from __future__ import annotations

from app.services.llm_adapters.anthropic import AnthropicAdapter
from app.services.llm_adapters.azure_openai import AzureOpenAIAdapter
from app.services.llm_adapters.base import BaseLlmAdapter
from app.services.llm_adapters.openai import OpenAIAdapter
from app.services.llm_adapters.xai import XAIAdapter


class LlmAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, BaseLlmAdapter] = {
            "openai": OpenAIAdapter(),
            "xai": XAIAdapter(),
            "azure_openai": AzureOpenAIAdapter(),
            "anthropic": AnthropicAdapter(),
        }

    def resolve(self, provider: str) -> BaseLlmAdapter:
        adapter = self._adapters.get(provider)
        if adapter is None:
            raise ValueError(f"No LLM adapter registered for provider '{provider}'")
        return adapter
