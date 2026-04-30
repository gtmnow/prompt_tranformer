from __future__ import annotations

from app.services.llm_adapters.openai import OpenAIAdapter


class AzureOpenAIAdapter(OpenAIAdapter):
    provider_name = "azure_openai"
