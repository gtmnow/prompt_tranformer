from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


LlmProvider = Literal["openai", "xai", "azure_openai", "anthropic"]
ExpectedOutput = Literal["text", "json"]


class TransformerLlmRequest(BaseModel):
    provider: LlmProvider
    model: str = Field(min_length=1, max_length=200)
    base_url: str = Field(min_length=1, max_length=500)
    api_key: str = Field(min_length=1)
    system_prompt: str = ""
    user_prompt: str = ""
    conversation_messages: list[dict[str, Any]] = Field(default_factory=list)
    input_items: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: dict[str, Any] | str | None = None
    text_format: dict[str, Any] | None = None
    max_output_tokens: int = Field(default=300, ge=1, le=8000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    expected_output: ExpectedOutput = "text"
    timeout_seconds: float = Field(default=15.0, gt=0.0, le=120.0)

    @model_validator(mode="after")
    def validate_prompt_sources(self) -> "TransformerLlmRequest":
        has_prompt_text = bool(self.system_prompt.strip() or self.user_prompt.strip())
        has_structured_input = bool(self.conversation_messages or self.input_items)
        if not has_prompt_text and not has_structured_input:
            raise ValueError("TransformerLlmRequest requires prompt text or structured input items.")
        return self


class NormalizedTokenUsage(BaseModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)
    raw_usage: dict[str, Any] | None = None

    @model_validator(mode="after")
    def populate_total_tokens(self) -> "NormalizedTokenUsage":
        if self.total_tokens is None and self.input_tokens is not None and self.output_tokens is not None:
            self.total_tokens = self.input_tokens + self.output_tokens
        return self


class TransformerLlmResponse(BaseModel):
    provider: LlmProvider
    model: str
    output_text: str
    status_code: int | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    normalized_usage: NormalizedTokenUsage | None = None
    raw_payload: dict[str, Any] | list[Any] | None = None


class TransformerLlmError(BaseModel):
    provider: LlmProvider
    model: str
    code: str
    message: str
    status_code: int | None = None
    raw_payload: dict[str, Any] | list[Any] | None = None
