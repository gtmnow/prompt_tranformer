from __future__ import annotations

from dataclasses import dataclass

from app.schemas.transform import AttachmentReference, ConversationHistoryTurn, GeneratedImagePayload
from app.services.llm_gateway import LlmGatewayService
from app.services.llm_request_factory import LlmRequestFactory
from app.services.llm_types import NormalizedTokenUsage
from app.services.runtime_llm import RuntimeLlmConfig


@dataclass(frozen=True)
class FinalResponseResult:
    text: str
    generated_images: list[GeneratedImagePayload]
    usage: NormalizedTokenUsage | None


class FinalResponseService:
    def __init__(self) -> None:
        self.gateway = LlmGatewayService()
        self.request_factory = LlmRequestFactory()

    def generate(
        self,
        *,
        runtime_config: RuntimeLlmConfig,
        resolved_model: str,
        transformed_prompt: str,
        conversation_history: list[ConversationHistoryTurn],
        attachments: list[AttachmentReference],
    ) -> FinalResponseResult:
        request = self.request_factory.build_final_response_request(
            runtime_config=runtime_config,
            resolved_model=resolved_model,
            transformed_prompt=transformed_prompt,
            conversation_history=conversation_history,
            attachments=attachments,
        )
        response, error = self.gateway.invoke(request)
        if error is not None:
            raise ValueError(f"LLM provider request failed: {error.message}")
        if response is None:
            raise ValueError("LLM provider returned no response.")

        generated_images = [GeneratedImagePayload(**image) for image in response.generated_images]
        text = response.output_text
        if not text and generated_images:
            text = "Generated image attached."
        if not text and not generated_images:
            raise ValueError("LLM provider returned an empty response.")

        return FinalResponseResult(
            text=text,
            generated_images=generated_images,
            usage=response.normalized_usage,
        )
