from __future__ import annotations

import json

from app.schemas.transform import AttachmentReference, ConversationHistoryTurn
from app.services.llm_types import (
    TransformerLlmContentPart,
    TransformerLlmMessage,
    TransformerLlmRequest,
    TransformerLlmToolRequest,
)
from app.services.runtime_llm import RuntimeLlmConfig


DOCUMENT_KINDS = {"document"}
IMAGE_KINDS = {"image"}
IMAGE_GENERATION_KEYWORDS = {
    "generate image",
    "create image",
    "make image",
    "draw",
    "redraw",
    "illustrate",
    "render",
    "turn this into",
    "convert this into",
    "cartoon style",
    "anime style",
    "edit this image",
    "restyle",
}


class LlmRequestFactory:
    def build_structure_evaluator_request(
        self,
        *,
        runtime_config: RuntimeLlmConfig,
        system_prompt: str,
        raw_prompt: str,
        enforcement_level: str,
        timeout_seconds: float,
        max_output_tokens: int = 300,
    ) -> TransformerLlmRequest:
        return TransformerLlmRequest(
            purpose="structure_evaluator",
            provider=runtime_config.provider,  # type: ignore[arg-type]
            model=runtime_config.model,
            base_url=resolve_base_url(runtime_config.endpoint_url, runtime_config.provider),
            api_key=runtime_config.api_key,
            messages=[
                TransformerLlmMessage(
                    role="system",
                    content=[TransformerLlmContentPart(type="text", text=system_prompt)],
                ),
                TransformerLlmMessage(
                    role="user",
                    content=[
                        TransformerLlmContentPart(
                            type="text",
                            text=json.dumps({"prompt": raw_prompt, "enforcement_level": enforcement_level}),
                        )
                    ],
                ),
            ],
            system_prompt=system_prompt,
            user_prompt=json.dumps({"prompt": raw_prompt, "enforcement_level": enforcement_level}),
            max_output_tokens=max_output_tokens,
            temperature=0.0,
            expected_output="json",
            timeout_seconds=timeout_seconds,
        )

    def build_guide_me_request(
        self,
        *,
        runtime_config: RuntimeLlmConfig,
        prompt: str,
        max_output_tokens: int,
    ) -> TransformerLlmRequest:
        system_prompt = (
            "Return only valid JSON that strictly follows the user's instructions. "
            "Do not add markdown fences, commentary, or extra keys."
        )
        return TransformerLlmRequest(
            purpose="guide_me",
            provider=runtime_config.provider,  # type: ignore[arg-type]
            model=runtime_config.model,
            base_url=resolve_base_url(runtime_config.endpoint_url, runtime_config.provider),
            api_key=runtime_config.api_key,
            system_prompt=system_prompt,
            user_prompt=prompt,
            messages=[
                TransformerLlmMessage(
                    role="system",
                    content=[TransformerLlmContentPart(type="text", text=system_prompt)],
                ),
                TransformerLlmMessage(
                    role="user",
                    content=[TransformerLlmContentPart(type="text", text=prompt)],
                ),
            ],
            max_output_tokens=max_output_tokens,
            temperature=0.0,
            expected_output="json",
        )

    def build_final_response_request(
        self,
        *,
        runtime_config: RuntimeLlmConfig,
        resolved_model: str,
        transformed_prompt: str,
        conversation_history: list[ConversationHistoryTurn],
        attachments: list[AttachmentReference],
        max_output_tokens: int = 800,
    ) -> TransformerLlmRequest:
        messages: list[TransformerLlmMessage] = []
        for turn in conversation_history:
            messages.append(
                TransformerLlmMessage(
                    role="user",
                    content=[TransformerLlmContentPart(type="text", text=turn.transformed_text)],
                )
            )
            messages.append(
                TransformerLlmMessage(
                    role="assistant",
                    content=[TransformerLlmContentPart(type="text", text=turn.assistant_text)],
                )
            )

        latest_content = [TransformerLlmContentPart(type="text", text=transformed_prompt)]
        image_file_ids = [attachment.provider_file_id for attachment in attachments if attachment.kind in IMAGE_KINDS and attachment.provider_file_id]
        for file_id in image_file_ids:
            latest_content.append(TransformerLlmContentPart(type="image_file", file_id=file_id))
        document_file_ids = [
            attachment.provider_file_id
            for attachment in attachments
            if attachment.kind in DOCUMENT_KINDS and attachment.provider_file_id
        ]
        for file_id in document_file_ids:
            latest_content.append(TransformerLlmContentPart(type="document_file", file_id=file_id))
        messages.append(TransformerLlmMessage(role="user", content=latest_content))

        should_generate_image = wants_image_generation(transformed_prompt)
        tools: list[TransformerLlmToolRequest] = []
        if document_file_ids and runtime_config.provider in {"openai", "azure_openai"}:
            tools.append(TransformerLlmToolRequest(type="code_interpreter", file_ids=document_file_ids))
        if should_generate_image and runtime_config.provider in {"openai", "azure_openai", "xai"}:
            tools.append(TransformerLlmToolRequest(type="image_generation", quality="high"))

        return TransformerLlmRequest(
            purpose="final_response",
            provider=runtime_config.provider,  # type: ignore[arg-type]
            model=resolved_model,
            base_url=resolve_base_url(runtime_config.endpoint_url, runtime_config.provider),
            api_key=runtime_config.api_key,
            user_prompt=transformed_prompt,
            messages=messages,
            tools=tools,
            max_output_tokens=max_output_tokens,
            temperature=0.2,
            expected_output="text",
        )


def wants_image_generation(prompt: str) -> bool:
    normalized = prompt.casefold()
    return any(keyword in normalized for keyword in IMAGE_GENERATION_KEYWORDS)


def resolve_base_url(endpoint_url: str | None, provider: str) -> str:
    normalized = (endpoint_url or "").strip()
    if normalized:
        return normalized
    defaults = {
        "openai": "https://api.openai.com/v1",
        "xai": "https://api.x.ai/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "azure_openai": "https://api.openai.com/v1",
    }
    return defaults.get(provider.strip().casefold(), "https://api.openai.com/v1")
