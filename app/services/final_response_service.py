from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.schemas.transform import AttachmentReference, ConversationHistoryTurn, GeneratedImagePayload
from app.services.llm_gateway import LlmGatewayService
from app.services.llm_provider_profiles import LlmProviderProfileService, ResolvedLlmProviderProfile
from app.services.llm_types import TransformerLlmError, TransformerLlmRequest, TransformerLlmResponse
from app.services.runtime_llm import RuntimeLlmConfig


logger = logging.getLogger("prompt_transformer.final_response_service")

DOCUMENT_KINDS = {"document"}
IMAGE_KINDS = {"image"}
IMAGE_GENERATION_KEYWORDS = {
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
WEB_EXPLICIT_LOOKUP_SIGNALS = {
    "browse the web",
    "browse the internet",
    "search the web",
    "search the internet",
    "search internet",
    "web search",
    "look this up",
    "look it up",
    "online",
}
WEB_FRESHNESS_SIGNALS = {
    "latest",
    "today",
    "current",
    "currently",
    "recent",
    "recent news",
    "top news",
    "headline",
    "headlines",
    "marketplace",
    "public guidance",
}
WEB_LOOKUP_VERBS = {"search", "browse", "look up", "find"}
WEB_PUBLIC_TARGETS = {"internet", "web", "online", "news", "public", "market", "marketplace"}
OPENAI_IMAGE_GENERATION_MODELS = {
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.5",
    "gpt-image-1",
}
OPENAI_WEB_SEARCH_MIN_OUTPUT_TOKENS = 4000
_MAX_OUTPUT_TOKENS_HARD_LIMIT = 8000
_OUTPUT_BUDGET_DIRECTIVE_RE = re.compile(
    r"\n\nOutput budget:\s*do not exceed\s*\d+\s+output tokens\.",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FinalResponseIntent:
    use_web_search: bool = False
    use_image_generation: bool = False


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
        self.gateway = LlmGatewayService()

    def generate(
        self,
        *,
        runtime_config: RuntimeLlmConfig,
        transformed_prompt: str,
        conversation_history: list[ConversationHistoryTurn],
        attachments: list[AttachmentReference],
        intent: FinalResponseIntent | None = None,
        reference_context: str | None = None,
    ) -> FinalResponseResult:
        final_prompt = transformed_prompt if not reference_context else f"{reference_context}\n\n{transformed_prompt}"
        profile = self.provider_profiles.resolve(runtime_config.provider, runtime_config.model)

        effective_intent = intent or resolve_final_response_intent(
            raw_prompt=transformed_prompt,
            transformed_prompt=transformed_prompt,
            supports_web_search=profile.supports_web_search,
        )
        if effective_intent.use_web_search and not _supports_web_search(profile):
            logger.warning(
                "Model %s for provider %s does not support web retrieval",
                profile.resolved_model,
                profile.provider,
            )
            raise FinalResponseProviderError(
                f"Model '{profile.resolved_model}' does not support live web retrievals. "
                "Use a model that supports live web retrieval or disable request_live_web_search.",
                status_code=400,
            )

        resolved_max_output_tokens = _resolve_max_output_tokens(
            profile=profile,
            document_attachments=[attachment for attachment in attachments if attachment.kind in DOCUMENT_KINDS],
            use_image_generation=effective_intent.use_image_generation,
            use_web_search=effective_intent.use_web_search,
            requested_max_output_tokens=runtime_config_output_tokens(runtime_config),
        )
        prompt_without_budget = final_prompt
        attempt_count = 0
        image_generation_attempted = False
        response = None
        payload: dict[str, Any] = {}
        resolved_max_output_tokens_local = resolved_max_output_tokens
        current_intent = effective_intent
        while True:
            prompt_with_budget = _append_max_output_budget(
                prompt=prompt_without_budget,
                max_output_tokens=resolved_max_output_tokens_local,
            )
            request = self._build_request(
                runtime_config=runtime_config,
                transformed_prompt=prompt_with_budget,
                conversation_history=conversation_history,
                attachments=attachments,
                intent=current_intent,
                profile=profile,
                resolved_max_output_tokens=resolved_max_output_tokens_local,
            )
            response, error = self.gateway.invoke(request)
            if error is not None:
                if (
                    current_intent.use_image_generation
                    and not image_generation_attempted
                    and _should_retry_image_request_as_text(error, profile)
                ):
                    image_generation_attempted = True
                    current_intent = FinalResponseIntent(
                        use_web_search=current_intent.use_web_search,
                        use_image_generation=False,
                    )
                    logger.warning(
                        "Image generation request timed out for model %s; retrying as text-only request.",
                        profile.resolved_model,
                    )
                    continue
                raise FinalResponseProviderError(
                    _build_gateway_error_message(error, profile=profile),
                    status_code=_map_gateway_error_status_code(error),
                )
            if response is None:
                raise FinalResponseProviderError("LLM provider returned no response.", status_code=502)

            payload = response.raw_payload if isinstance(response.raw_payload, dict) else {}
            incomplete_reason = _extract_incomplete_reason(payload)
            if incomplete_reason is None:
                break

            if (
                _is_max_output_tokens_incomplete_reason(incomplete_reason)
                and attempt_count < 1
                and _supports_token_retry(resolved_max_output_tokens_local, _MAX_OUTPUT_TOKENS_HARD_LIMIT)
            ):
                attempt_count += 1
                next_max_output_tokens = _bump_max_output_tokens(
                    requested_max_output_tokens=resolved_max_output_tokens_local,
                    hard_limit=_MAX_OUTPUT_TOKENS_HARD_LIMIT,
                )
                if next_max_output_tokens > resolved_max_output_tokens_local:
                    logger.warning(
                        "LLM response incomplete at token limit; retrying with higher max_output_tokens",
                        extra={
                            "provider": profile.provider,
                            "model": profile.resolved_model,
                            "attempt": attempt_count,
                            "previous_max_output_tokens": resolved_max_output_tokens_local,
                            "next_max_output_tokens": next_max_output_tokens,
                        },
                    )
                    resolved_max_output_tokens_local = next_max_output_tokens
                    continue

            incomplete_error = _error_for_incomplete_response_reason(incomplete_reason)
            raise ValueError(incomplete_error)

        text = response.output_text.strip()
        generated_images = _extract_generated_images(payload)
        if not text and generated_images:
            text = "Generated image attached."
        if not text and not generated_images:
            raise ValueError("LLM provider returned an empty response.")

        return FinalResponseResult(text=text, generated_images=generated_images, usage=response.usage)

    def _build_request(
        self,
        *,
        runtime_config: RuntimeLlmConfig,
        transformed_prompt: str,
        conversation_history: list[ConversationHistoryTurn],
        attachments: list[AttachmentReference],
        intent: FinalResponseIntent,
        profile: ResolvedLlmProviderProfile,
        resolved_max_output_tokens: int | None = None,
    ) -> TransformerLlmRequest:
        document_attachments = [attachment for attachment in attachments if attachment.kind in DOCUMENT_KINDS]
        image_attachments = [attachment for attachment in attachments if attachment.kind in IMAGE_KINDS]
        resolved_model = profile.resolved_model

        if intent.use_image_generation and not _supports_image_generation(profile):
            raise ValueError("Image generation is not supported with the resolved model.")

        return TransformerLlmRequest(
            provider=runtime_config.provider,  # type: ignore[arg-type]
            model=resolved_model,
            base_url=_resolve_base_url(runtime_config.endpoint_url, runtime_config.provider),
            api_key=runtime_config.api_key,
            system_prompt="",
            user_prompt=transformed_prompt,
            conversation_messages=_build_messages(
                conversation_history=conversation_history,
                transformed_prompt=transformed_prompt,
            )
            if profile.api_family == "chat_completions"
            else [],
            input_items=_build_input_items(
                conversation_history=conversation_history,
                transformed_prompt=transformed_prompt,
                image_attachments=image_attachments,
            )
            if profile.api_family != "chat_completions"
            else [],
            tools=_build_tools(
                profile=profile,
                document_attachments=document_attachments,
                use_image_generation=intent.use_image_generation,
                use_web_search=intent.use_web_search,
            ),
            text_format=None if intent.use_image_generation else {"format": {"type": "text"}},
            max_output_tokens=resolved_max_output_tokens
            if resolved_max_output_tokens is not None
            else _resolve_max_output_tokens(
                profile=profile,
                document_attachments=document_attachments,
                use_image_generation=intent.use_image_generation,
                use_web_search=intent.use_web_search,
                requested_max_output_tokens=runtime_config_output_tokens(runtime_config),
            ),
            temperature=0.2 if _supports_temperature_parameter(profile) else 0.0,
            expected_output="text",
        )


def runtime_config_output_tokens(runtime_config: RuntimeLlmConfig) -> int:
    return get_settings().final_response_max_output_tokens


def resolve_final_response_intent(
    *,
    raw_prompt: str,
    transformed_prompt: str | None = None,
    supports_web_search: bool = True,
    request_live_web_search: bool | None = None,
) -> FinalResponseIntent:
    transformed = transformed_prompt or ""
    combined = "\n".join(part for part in [raw_prompt, transformed] if part.strip())
    if request_live_web_search is not None:
        return FinalResponseIntent(
            use_web_search=bool(request_live_web_search),
            use_image_generation=_wants_image_generation(combined),
        )
    return FinalResponseIntent(
        use_web_search=(
            supports_web_search
            and _should_use_web_search(raw_prompt=raw_prompt, transformed_prompt=combined)
        ),
        use_image_generation=_wants_image_generation(combined),
    )


def _should_use_web_search(*, raw_prompt: str, transformed_prompt: str) -> bool:
    raw_normalized = raw_prompt.casefold()
    transformed_normalized = transformed_prompt.casefold()
    if any(signal in raw_normalized for signal in WEB_EXPLICIT_LOOKUP_SIGNALS):
        return True
    if any(signal in raw_normalized for signal in WEB_FRESHNESS_SIGNALS):
        return True
    if _has_lookup_and_public_target(raw_normalized):
        return True
    return any(signal in transformed_normalized for signal in WEB_EXPLICIT_LOOKUP_SIGNALS)


def _has_lookup_and_public_target(text: str) -> bool:
    has_lookup_verb = any(verb in text for verb in WEB_LOOKUP_VERBS)
    has_public_target = any(target in text for target in WEB_PUBLIC_TARGETS)
    return has_lookup_verb and has_public_target


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
    profile: ResolvedLlmProviderProfile,
    document_attachments: list[AttachmentReference],
    use_image_generation: bool,
    use_web_search: bool,
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    if _supports_web_search(profile) and use_web_search:
        tools.append({"type": "web_search"})

    if document_attachments and _supports_code_interpreter(profile.provider):
        file_ids = [
            attachment.provider_file_id
            for attachment in document_attachments
            if attachment.provider_file_id
        ]
        if file_ids:
            tools.append(
                {
                    "type": "code_interpreter",
                    "container": {
                        "type": "auto",
                        "file_ids": file_ids,
                    },
                }
            )

    if use_image_generation and _supports_image_generation_tool(profile=profile):
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


def _extract_generated_images(payload: dict[str, Any]) -> list[GeneratedImagePayload]:
    output = payload.get("output", [])
    if not isinstance(output, list):
        return []

    images: list[GeneratedImagePayload] = []
    for item in output:
        if item.get("type") != "image_generation_call":
            continue
        result = item.get("result")
        if isinstance(result, str) and result:
            images.append(GeneratedImagePayload(media_type="image/png", base64_data=result))
            continue
        if not isinstance(result, dict):
            continue
        base64_data, media_type = _extract_image_result(result)
        if not base64_data:
            continue
        images.append(
            GeneratedImagePayload(
                media_type=media_type if media_type else "image/png",
                base64_data=base64_data,
            )
        )
    return images


def _extract_image_result(result: dict[str, Any]) -> tuple[str | None, str | None]:
    for key in ("data", "b64_json", "base64", "base64_data", "result"):
        value = result.get(key)
        if isinstance(value, str):
            normalized = _strip_data_uri(value)
            if normalized:
                return normalized, _extract_media_type(result)

    if (
        isinstance(result.get("image"), dict)
        and isinstance(result["image"], dict)
    ):
        nested_image = result["image"]
        for key in ("data", "b64_json", "base64", "base64_data"):
            value = nested_image.get(key)
            if isinstance(value, str):
                normalized = _strip_data_uri(value)
                if normalized:
                    media_type = _extract_media_type(nested_image) or _extract_media_type(result)
                    return normalized, media_type
    return None, _extract_media_type(result)


def _extract_media_type(result: dict[str, Any]) -> str | None:
    media_type = result.get("media_type") or result.get("mime_type")
    if isinstance(media_type, str):
        normalized = media_type.strip().lower()
        if normalized:
            return normalized
    return None


def _strip_data_uri(value: str) -> str | None:
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.startswith("data:") and ";base64," in normalized:
        return normalized.split(";base64,", 1)[1].strip()
    return normalized


def _extract_incomplete_response_error(payload: dict[str, Any]) -> str | None:
    reason = _extract_incomplete_reason(payload)
    if reason is None:
        return None
    return _error_for_incomplete_response_reason(reason)


def _extract_incomplete_reason(payload: dict[str, Any]) -> str | None:
    status = payload.get("status")
    if status != "incomplete":
        return None

    incomplete_details = payload.get("incomplete_details")
    if isinstance(incomplete_details, dict):
        reason = incomplete_details.get("reason")
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
    return "incomplete"


def _error_for_incomplete_response_reason(reason: str) -> str:
    normalized_reason_lower = reason.strip().casefold().replace("-", "_")
    if normalized_reason_lower in {"max_tokens", "max_output_tokens", "max_output_token"}:
        return "LLM provider response was incomplete because it hit the max_output_tokens limit."
    if reason.strip():
        return f"LLM provider response was incomplete: {reason}."
    return "LLM provider response was incomplete."


def _is_max_output_tokens_incomplete_reason(reason: str) -> bool:
    normalized_reason_lower = reason.strip().casefold().replace("-", "_")
    return normalized_reason_lower in {"max_tokens", "max_output_tokens", "max_output_token"}


def _supports_token_retry(
    current_max_output_tokens: int,
    hard_limit: int,
) -> bool:
    return current_max_output_tokens < hard_limit


def _bump_max_output_tokens(*, requested_max_output_tokens: int, hard_limit: int) -> int:
    if requested_max_output_tokens >= hard_limit:
        return requested_max_output_tokens

    next_tokens = requested_max_output_tokens * 2
    if next_tokens <= requested_max_output_tokens:
        next_tokens = requested_max_output_tokens + 1000
    return min(next_tokens, hard_limit)


def _append_max_output_budget(prompt: str, max_output_tokens: int) -> str:
    cleaned_prompt = prompt.strip()
    if max_output_tokens <= 0:
        return cleaned_prompt
    if max_output_tokens > _MAX_OUTPUT_TOKENS_HARD_LIMIT:
        max_output_tokens = _MAX_OUTPUT_TOKENS_HARD_LIMIT

    directive = (
        f"\n\nOutput budget: do not exceed {max_output_tokens} output tokens."
    )
    if _OUTPUT_BUDGET_DIRECTIVE_RE.search(cleaned_prompt):
        return _OUTPUT_BUDGET_DIRECTIVE_RE.sub(directive, cleaned_prompt)
    return cleaned_prompt + directive


def _map_gateway_error_status_code(error: TransformerLlmError) -> int:
    if error.status_code is not None:
        return error.status_code

    code = error.code.strip().casefold()
    if "timeout" in code:
        return 504
    return 502


def _build_gateway_error_message(
    error: TransformerLlmError,
    *,
    profile: ResolvedLlmProviderProfile,
) -> str:
    if error.status_code is not None:
        return error.message

    code = error.code.strip().casefold()
    if "timeout" in code:
        return f"LLM provider timed out after {profile.request_timeout_seconds:g} seconds."
    return f"LLM provider request failed before a response was received: {error.message}"


def _wants_image_generation(prompt: str) -> bool:
    normalized = prompt.casefold()
    if _contains_generate_image_intent(normalized):
        return True
    return any(keyword in normalized for keyword in IMAGE_GENERATION_KEYWORDS)


def _contains_generate_image_intent(normalized_prompt: str) -> bool:
    return any(
        pattern in normalized_prompt
        for pattern in (
            "generate image",
            "generate an image",
            "generate a image",
            "create image",
            "create an image",
            "create a image",
            "make image",
            "make an image",
            "make a image",
            "draw me",
            "draw a",
            "convert this into",
            "turn this into",
        )
    )


def _supports_image_generation(profile: ResolvedLlmProviderProfile) -> bool:
    if profile.supports_image_generation:
        return True
    return profile.resolved_model.strip().casefold() in OPENAI_IMAGE_GENERATION_MODELS


def _supports_web_search(profile: ResolvedLlmProviderProfile) -> bool:
    return profile.api_family == "responses" and bool(profile.supports_web_search)


def _resolve_max_output_tokens(
    *,
    profile: ResolvedLlmProviderProfile,
    document_attachments: list[AttachmentReference],
    use_image_generation: bool,
    use_web_search: bool,
    requested_max_output_tokens: int,
) -> int:
    tools = _build_tools(
        profile=profile,
        document_attachments=document_attachments,
        use_image_generation=use_image_generation,
        use_web_search=use_web_search,
    )
    if profile.provider == "openai" and any(tool.get("type") == "web_search" for tool in tools):
        return max(requested_max_output_tokens, OPENAI_WEB_SEARCH_MIN_OUTPUT_TOKENS)
    return requested_max_output_tokens


def _supports_code_interpreter(provider: str) -> bool:
    return provider.strip().casefold() in {"openai", "azure_openai"}


def _supports_image_generation_tool(profile: ResolvedLlmProviderProfile) -> bool:
    return profile.provider.strip().casefold() in {"openai", "azure_openai"}


def _supports_temperature_parameter(profile: ResolvedLlmProviderProfile) -> bool:
    if profile.supports_temperature is not None:
        return profile.supports_temperature
    return not profile.resolved_model.strip().casefold().startswith("gpt-5")


def _should_retry_image_request_as_text(
    error: TransformerLlmError,
    profile: ResolvedLlmProviderProfile,
) -> bool:
    if profile.provider.strip().casefold() not in {"openai", "azure_openai"}:
        return False
    if error.status_code is None:
        return "timeout" in error.code.strip().casefold()
    return error.status_code >= 500 or error.status_code == 408


def _resolve_base_url(endpoint_url: str | None, provider: str) -> str:
    normalized = (endpoint_url or "").strip()
    if normalized:
        return normalized
    defaults = {
        "openai": "https://api.openai.com/v1",
        "xai": "https://api.x.ai/v1",
        "azure_openai": "https://api.openai.com/v1",
    }
    return defaults.get(provider.strip().casefold(), "https://api.openai.com/v1")
