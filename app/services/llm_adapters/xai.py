from __future__ import annotations

from app.services.llm_adapters.openai import OpenAIAdapter


class XAIAdapter(OpenAIAdapter):
    provider_name = "xai"
