from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


LlmProvider = Literal["openai", "xai", "azure_openai", "anthropic"]
ExpectedOutput = Literal["text", "json"]
LlmRequestPurpose = Literal["structure_evaluator", "guide_me", "final_response"]
LlmMessageRole = Literal["system", "user", "assistant"]
LlmContentPartType = Literal["text", "image_file"]
LlmToolType = Literal["code_interpreter", "image_generation"]


class TransformerLlmContentPart(BaseModel):
    type: LlmContentPartType
    text: str | None = None
    file_id: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "TransformerLlmContentPart":
        if self.type == "text" and not (self.text or "").strip():
            raise ValueError("Text content parts require text.")
        if self.type == "image_file" and not (self.file_id or "").strip():
            raise ValueError("Image file content parts require file_id.")
        return self


class TransformerLlmMessage(BaseModel):
    role: LlmMessageRole
    content: list[TransformerLlmContentPart] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_content(self) -> "TransformerLlmMessage":
        if not self.content:
            raise ValueError("LLM messages require at least one content part.")
        return self


class TransformerLlmToolRequest(BaseModel):
    type: LlmToolType
    file_ids: list[str] = Field(default_factory=list)
    quality: str | None = None


class TransformerLlmRequest(BaseModel):
    purpose: LlmRequestPurpose = "structure_evaluator"
    provider: LlmProvider
    model: str = Field(min_length=1, max_length=200)
    base_url: str = Field(min_length=1, max_length=500)
    api_key: str = Field(min_length=1)
    system_prompt: str = ""
    user_prompt: str = ""
    messages: list[TransformerLlmMessage] = Field(default_factory=list)
    tools: list[TransformerLlmToolRequest] = Field(default_factory=list)
    max_output_tokens: int = Field(default=300, ge=1, le=8000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    expected_output: ExpectedOutput = "text"
    timeout_seconds: float = Field(default=15.0, gt=0.0, le=120.0)

    @model_validator(mode="after")
    def populate_messages(self) -> "TransformerLlmRequest":
        if self.messages:
            return self

        messages: list[TransformerLlmMessage] = []
        if self.system_prompt.strip():
            messages.append(
                TransformerLlmMessage(
                    role="system",
                    content=[TransformerLlmContentPart(type="text", text=self.system_prompt)],
                )
            )
        if self.user_prompt.strip():
            messages.append(
                TransformerLlmMessage(
                    role="user",
                    content=[TransformerLlmContentPart(type="text", text=self.user_prompt)],
                )
            )
        if not messages:
            raise ValueError("TransformerLlmRequest requires messages or prompt text.")
        self.messages = messages
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
    generated_images: list[dict[str, Any]] = Field(default_factory=list)
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
