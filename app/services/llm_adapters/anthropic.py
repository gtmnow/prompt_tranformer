from __future__ import annotations

from typing import Any

import httpx

from app.services.llm_adapters.base import BaseLlmAdapter
from app.services.llm_provider_profiles import ResolvedLlmProviderProfile
from app.services.llm_types import TransformerLlmError, TransformerLlmRequest, TransformerLlmResponse
from app.services.token_usage import normalize_usage


class AnthropicAdapter(BaseLlmAdapter):
    provider_name = "anthropic"

    def invoke(
        self,
        request: TransformerLlmRequest,
        profile: ResolvedLlmProviderProfile,
    ) -> tuple[TransformerLlmResponse | None, TransformerLlmError | None]:
        url = f"{request.base_url.rstrip('/')}/{profile.endpoint_path.lstrip('/')}"
        headers = {
            "Content-Type": "application/json",
            (profile.auth_header_name or "x-api-key"): request.api_key,
            (profile.version_header_name or "anthropic-version"): profile.version_header_value or "2023-06-01",
        }
        if self._requires_files_beta(request):
            headers["anthropic-beta"] = "files-api-2025-04-14"
        payload = {
            "model": request.model,
            profile.token_parameter: request.max_output_tokens,
            "temperature": request.temperature,
            "messages": self._build_messages(request),
        }
        if profile.supports_system_prompt:
            system_prompt = self._extract_system_prompt(request)
            if system_prompt:
                payload["system"] = system_prompt

        try:
            with httpx.Client(timeout=request.timeout_seconds) as client:
                response = client.post(url, headers=headers, json=payload)
            response_payload = response.json()
            response.raise_for_status()
            output_text = self._extract_output_text(response_payload)
            usage = self._extract_usage(response_payload)
            return (
                TransformerLlmResponse(
                    provider=request.provider,
                    model=request.model,
                    output_text=output_text,
                    generated_images=[],
                    status_code=response.status_code,
                    finish_reason=self._extract_finish_reason(response_payload),
                    usage=usage,
                    normalized_usage=normalize_usage(request.provider, usage),
                    raw_payload=response_payload if isinstance(response_payload, dict) else None,
                ),
                None,
            )
        except httpx.HTTPStatusError as exc:
            payload = self._safe_json(exc.response)
            return None, TransformerLlmError(
                provider=request.provider,
                model=request.model,
                code=self._extract_error_code(payload, fallback=f"HTTP_{exc.response.status_code}"),
                message=self._extract_error_message(payload, exc.response.text),
                status_code=exc.response.status_code,
                raw_payload=payload if isinstance(payload, (dict, list)) else None,
            )
        except httpx.HTTPError as exc:
            return None, TransformerLlmError(
                provider=request.provider,
                model=request.model,
                code=exc.__class__.__name__.upper(),
                message=str(exc),
            )
        except ValueError as exc:
            return None, TransformerLlmError(
                provider=request.provider,
                model=request.model,
                code="INVALID_RESPONSE",
                message=str(exc),
            )

    def _extract_output_text(self, payload: dict[str, Any]) -> str:
        content = payload.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        raise ValueError("Anthropic response returned no text content")

    def _extract_finish_reason(self, payload: dict[str, Any]) -> str | None:
        value = payload.get("stop_reason")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    def _extract_usage(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        usage = payload.get("usage")
        if isinstance(usage, dict):
            return usage
        return None

    def _extract_system_prompt(self, request: TransformerLlmRequest) -> str:
        for message in request.messages:
            if message.role != "system":
                continue
            parts = [part.text.strip() for part in message.content if part.type == "text" and isinstance(part.text, str) and part.text.strip()]
            if parts:
                return "\n".join(parts)
        return ""

    def _build_messages(self, request: TransformerLlmRequest) -> list[dict[str, Any]]:
        if request.tools:
            raise ValueError("Anthropic adapter does not support tool requests in Prompt Transformer.")

        messages: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "system":
                continue

            content: list[dict[str, Any]] = []
            for part in message.content:
                if part.type == "text":
                    if not isinstance(part.text, str) or not part.text.strip():
                        raise ValueError("Anthropic text content requires text.")
                    content.append({"type": "text", "text": part.text})
                elif part.type == "image_file":
                    if not part.file_id:
                        raise ValueError("Anthropic image content requires file_id.")
                    content.append({"type": "image", "source": {"type": "file", "file_id": part.file_id}})
                elif part.type == "document_file":
                    if not part.file_id:
                        raise ValueError("Anthropic document content requires file_id.")
                    content.append({"type": "document", "source": {"type": "file", "file_id": part.file_id}})
                else:
                    raise ValueError("Anthropic adapter received unsupported content type.")
            if content:
                messages.append({"role": message.role, "content": content})
        return messages

    def _requires_files_beta(self, request: TransformerLlmRequest) -> bool:
        return any(part.type in {"image_file", "document_file"} for message in request.messages for part in message.content)

    def _safe_json(self, response: httpx.Response) -> dict[str, Any] | list[Any] | None:
        try:
            payload = response.json()
            if isinstance(payload, (dict, list)):
                return payload
        except Exception:
            return None
        return None

    def _extract_error_code(self, payload: dict[str, Any] | list[Any] | None, fallback: str) -> str:
        if isinstance(payload, dict):
            error_payload = payload.get("error")
            if isinstance(error_payload, dict):
                error_type = error_payload.get("type")
                if isinstance(error_type, str) and error_type.strip():
                    return error_type.strip()
        return fallback

    def _extract_error_message(self, payload: dict[str, Any] | list[Any] | None, fallback: str) -> str:
        if isinstance(payload, dict):
            error_payload = payload.get("error")
            if isinstance(error_payload, dict):
                message = error_payload.get("message")
                if isinstance(message, str) and message.strip():
                    return message.strip()
        return fallback
