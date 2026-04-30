from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


LlmProvider = Literal["openai", "xai", "azure_openai", "anthropic"]
ExpectedOutput = Literal["text", "json"]


class TransformerLlmRequest(BaseModel):
    provider: LlmProvider
    model: str = Field(min_length=1, max_length=200)
    base_url: str = Field(min_length=1, max_length=500)
    api_key: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    max_output_tokens: int = Field(default=300, ge=1, le=8000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    expected_output: ExpectedOutput = "text"
    timeout_seconds: float = Field(default=15.0, gt=0.0, le=120.0)


class TransformerLlmResponse(BaseModel):
    provider: LlmProvider
    model: str
    output_text: str
    status_code: int | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    raw_payload: dict[str, Any] | list[Any] | None = None


class TransformerLlmError(BaseModel):
    provider: LlmProvider
    model: str
    code: str
    message: str
    status_code: int | None = None
    raw_payload: dict[str, Any] | list[Any] | None = None
