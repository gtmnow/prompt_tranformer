from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.schemas.transform import AttachmentReference, ConversationHistoryTurn, GeneratedImagePayload
from app.services.llm_provider_profiles import LlmProviderProfileService, ResolvedLlmProviderProfile
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
OPENAI_IMAGE_GENERATION_MODELS = {
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gpt-image-1",
}


@dataclass(frozen=True)
class FinalResponseResult:
    text: str
    generated_images: list[GeneratedImagePayload]
    usage: dict[str, Any] | None


@dataclass(frozen=True)
class FinalResponseProviderError(Exception):
    message: str
    status_code: int = 502

    def __str__(self) -> str:
        return self.message


class FinalResponseService:
    def __init__(self) -> None:
        self.provider_profiles = LlmProviderProfileService()

    def generate(
        self,
        *,
        runtime_config: RuntimeLlmConfig,
        resolved_model: str,
        transformed_prompt: str,
        conversation_history: list[ConversationHistoryTurn],
        attachments: list[AttachmentReference],
        reference_context: str | None = None,
    ) -> FinalResponseResult:
        final_prompt = transformed_prompt if not reference_context else f"{reference_context}\n\n{transformed_prompt}"
        provider = runtime_config.provider.strip().casefold()
        if provider in {"openai", "xai"}:
            return self._generate_openai_like_response(
                runtime_config=runtime_config,
                resolved_model=resolved_model,
                transformed_prompt=final_prompt,
                conversation_history=conversation_history,
                attachments=attachments,
            )
        raise ValueError(f"Final response generation is not implemented for provider: {runtime_config.provider}")

    def _generate_openai_like_response(
        self,
        *,
        runtime_config: RuntimeLlmConfig,
        resolved_model: str,
        transformed_prompt: str,
        conversation_history: list[ConversationHistoryTurn],
        attachments: list[AttachmentReference],
    ) -> FinalResponseResult:
        document_attachments = [attachment for attachment in attachments if attachment.kind in DOCUMENT_KINDS]
        image_attachments = [attachment for attachment in attachments if attachment.kind in IMAGE_KINDS]
        wants_image_generation = _wants_image_generation(transformed_prompt)

        if wants_image_generation and not _supports_openai_image_generation(resolved_model):
            raise ValueError("Image generation is not supported with the resolved model.")

        profile = self.provider_profiles.resolve(runtime_config.provider, resolved_model)
        url = f"{_resolve_base_url(runtime_config.endpoint_url, runtime_config.provider).rstrip('/')}/{profile.endpoint_path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {runtime_config.api_key}", "Content-Type": "application/json"}
        payload = _build_openai_like_payload(
            profile=profile,
            model=resolved_model,
            conversation_history=conversation_history,
            transformed_prompt=transformed_prompt,
            image_attachments=image_attachments,
            document_attachments=document_attachments,
            wants_image_generation=wants_image_generation,
            max_output_tokens=800,
        )

        try:
            with httpx.Client(timeout=profile.request_timeout_seconds) as client:
                response = client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise FinalResponseProviderError(
                f"LLM provider timed out after {profile.request_timeout_seconds:g} seconds.",
                status_code=504,
            ) from exc
        except httpx.HTTPError as exc:
            raise FinalResponseProviderError(
                f"LLM provider request failed before a response was received: {exc}",
                status_code=502,
            ) from exc

        if response.status_code >= 400:
            detail = _extract_error_detail(response)
            raise FinalResponseProviderError(
                f"LLM provider request failed: {detail}",
                status_code=_map_provider_status_code(response.status_code),
            )

        data = response.json()
        text = _extract_output_text(data, profile=profile)
        generated_images = _extract_generated_images(data, profile=profile)
        if not text and generated_images:
            text = "Generated image attached."
        if not text and not generated_images:
            raise ValueError("LLM provider returned an empty response.")

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        return FinalResponseResult(text=text, generated_images=generated_images, usage=usage)


def _build_openai_like_payload(
    *,
    profile: ResolvedLlmProviderProfile,
    model: str,
    conversation_history: list[ConversationHistoryTurn],
    transformed_prompt: str,
    image_attachments: list[AttachmentReference],
    document_attachments: list[AttachmentReference],
    wants_image_generation: bool,
    max_output_tokens: int,
) -> dict[str, Any]:
    if profile.api_family == "chat_completions":
        payload: dict[str, Any] = {
            "model": model,
            "messages": _build_messages(
                conversation_history=conversation_history,
                transformed_prompt=transformed_prompt,
            ),
            profile.token_parameter: max_output_tokens,
        }
        if _supports_temperature_parameter(model):
            payload["temperature"] = 0.2
        return payload

    payload = {
        "model": model,
        "input": _build_input_items(
            conversation_history=conversation_history,
            transformed_prompt=transformed_prompt,
            image_attachments=image_attachments,
        ),
        profile.token_parameter: max_output_tokens,
        "store": False,
    }

    if _supports_temperature_parameter(model):
        payload["temperature"] = 0.2

    if not wants_image_generation:
        payload["text"] = {"format": {"type": "text"}}

    tools = _build_tools(document_attachments=document_attachments, wants_image_generation=wants_image_generation)
    if tools:
        payload["tools"] = tools
        if document_attachments and not wants_image_generation:
            payload["tool_choice"] = {"type": "code_interpreter"}

    return payload


def _build_messages(
    *,
    conversation_history: list[ConversationHistoryTurn],
    transformed_prompt: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in conversation_history:
        messages.append({"role": "user", "content": turn.transformed_text})
        messages.append({"role": "assistant", "content": turn.assistant_text})
    messages.append({"role": "user", "content": transformed_prompt})
    return messages


def _build_tools(
    *,
    document_attachments: list[AttachmentReference],
    wants_image_generation: bool,
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    if document_attachments:
        tools.append(
            {
                "type": "code_interpreter",
                "container": {
                    "type": "auto",
                    "file_ids": [
                        attachment.provider_file_id
                        for attachment in document_attachments
                        if attachment.provider_file_id
                    ],
                },
            }
        )

    if wants_image_generation:
        tools.append({"type": "image_generation", "quality": "high"})

    return tools


def _build_input_items(
    *,
    conversation_history: list[ConversationHistoryTurn],
    transformed_prompt: str,
    image_attachments: list[AttachmentReference],
) -> list[dict[str, Any]]:
    input_items: list[dict[str, Any]] = []
    for turn in conversation_history:
        input_items.append({"role": "user", "content": [{"type": "input_text", "text": turn.transformed_text}]})
        input_items.append({"role": "assistant", "content": [{"type": "output_text", "text": turn.assistant_text}]})

    latest_content: list[dict[str, Any]] = [{"type": "input_text", "text": transformed_prompt}]
    for attachment in image_attachments:
        if attachment.provider_file_id:
            latest_content.append({"type": "input_image", "file_id": attachment.provider_file_id})
    input_items.append({"role": "user", "content": latest_content})
    return input_items


def _extract_output_text(payload: dict[str, Any], *, profile: ResolvedLlmProviderProfile) -> str:
    if profile.api_family == "chat_completions":
        choices = payload.get("choices")
        if isinstance(choices, list):
            for item in choices:
                if not isinstance(item, dict):
                    continue
                message = item.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        return ""

    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = payload.get("output", [])
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if item.get("type") != "message":
                continue
            for content_item in item.get("content", []):
                if content_item.get("type") == "output_text":
                    text_value = content_item.get("text", "")
                    if isinstance(text_value, str) and text_value.strip():
                        chunks.append(text_value.strip())
        if chunks:
            return "\n".join(chunks).strip()
    return ""


def _extract_generated_images(payload: dict[str, Any], *, profile: ResolvedLlmProviderProfile) -> list[GeneratedImagePayload]:
    if profile.api_family == "chat_completions":
        return []

    output = payload.get("output", [])
    if not isinstance(output, list):
        return []

    images: list[GeneratedImagePayload] = []
    for item in output:
        if item.get("type") != "image_generation_call":
            continue
        result = item.get("result")
        if isinstance(result, str) and result:
            images.append(GeneratedImagePayload(base64_data=result))
    return images


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"status {response.status_code}"

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
    return response.text or f"status {response.status_code}"


def _map_provider_status_code(status_code: int) -> int:
    if status_code in {408, 504}:
        return 504
    if status_code == 429:
        return 503
    return 502


def _wants_image_generation(prompt: str) -> bool:
    normalized = prompt.casefold()
    return any(keyword in normalized for keyword in IMAGE_GENERATION_KEYWORDS)


def _supports_openai_image_generation(model: str) -> bool:
    return model.strip().casefold() in OPENAI_IMAGE_GENERATION_MODELS


def _supports_temperature_parameter(model: str) -> bool:
    return not model.strip().casefold().startswith("gpt-5")


def _resolve_base_url(endpoint_url: str | None, provider: str) -> str:
    normalized = (endpoint_url or "").strip()
    if normalized:
        return normalized
    defaults = {
        "openai": "https://api.openai.com/v1",
        "xai": "https://api.x.ai/v1",
    }
    return defaults.get(provider.strip().casefold(), "https://api.openai.com/v1")
