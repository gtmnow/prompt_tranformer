from __future__ import annotations

from typing import Any

import httpx

from app.services.llm_adapters.base import BaseLlmAdapter
from app.services.llm_provider_profiles import ResolvedLlmProviderProfile
from app.services.llm_types import (
    TransformerLlmError,
    TransformerLlmMessage,
    TransformerLlmRequest,
    TransformerLlmResponse,
)
from app.services.token_usage import normalize_usage


class OpenAIAdapter(BaseLlmAdapter):
    provider_name = "openai"

    def invoke(
        self,
        request: TransformerLlmRequest,
        profile: ResolvedLlmProviderProfile,
    ) -> tuple[TransformerLlmResponse | None, TransformerLlmError | None]:
        url = f"{request.base_url.rstrip('/')}/{profile.endpoint_path.lstrip('/')}"
        headers = self._build_headers(request, profile)
        payload = self._build_payload(request, profile)
        try:
            response = self._send_with_temperature_fallback(
                url=url,
                headers=headers,
                payload=payload,
                timeout_seconds=request.timeout_seconds,
            )
            response_payload = response.json()
            response.raise_for_status()
            output_text = self._extract_output_text(profile, response_payload)
            usage = self._extract_usage(response_payload)
            return (
                TransformerLlmResponse(
                    provider=request.provider,
                    model=request.model,
                    output_text=output_text,
                    generated_images=self._extract_generated_images(profile, response_payload),
                    status_code=response.status_code,
                    finish_reason=self._extract_finish_reason(profile, response_payload),
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

    def _send_with_temperature_fallback(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(url, headers=headers, json=payload)
            if not self._should_retry_without_temperature(response, payload):
                return response

            retry_payload = dict(payload)
            retry_payload.pop("temperature", None)
            return client.post(url, headers=headers, json=retry_payload)

    def _should_retry_without_temperature(
        self,
        response: httpx.Response,
        payload: dict[str, Any],
    ) -> bool:
        if response.status_code != 400 or "temperature" not in payload:
            return False

        response_payload = self._safe_json(response)
        if not isinstance(response_payload, dict):
            return False

        error_payload = response_payload.get("error")
        if not isinstance(error_payload, dict):
            return False

        message = error_payload.get("message")
        if not isinstance(message, str):
            return False

        normalized = message.lower()
        return "temperature" in normalized and "unsupported parameter" in normalized

    def _build_headers(self, request: TransformerLlmRequest, profile: ResolvedLlmProviderProfile) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if profile.auth_scheme == "api_key":
            header_name = profile.auth_header_name or "api-key"
            headers[header_name] = request.api_key
        else:
            headers["Authorization"] = f"Bearer {request.api_key}"
        return headers

    def _build_payload(self, request: TransformerLlmRequest, profile: ResolvedLlmProviderProfile) -> dict[str, Any]:
        if profile.api_family == "chat_completions":
            self._validate_chat_completion_request(request)
            payload: dict[str, Any] = {
                "model": request.model,
                "messages": self._build_chat_messages(request, profile),
                "temperature": request.temperature,
                profile.token_parameter: request.max_output_tokens,
            }
            if request.expected_output == "json" and profile.json_mode == "response_format_json_object":
                payload["response_format"] = {"type": "json_object"}
            return payload

        payload = {
            "model": request.model,
            "input": self._build_responses_input(request, profile),
            "temperature": request.temperature,
            profile.token_parameter: request.max_output_tokens,
            "store": False,
        }
        if request.expected_output == "json":
            payload["text"] = {"format": {"type": "json_object"}}
        elif request.purpose != "final_response":
            payload["text"] = {"format": {"type": "text"}}

        tools = self._build_tools(request)
        if tools:
            payload["tools"] = tools
            if any(tool.get("type") == "code_interpreter" for tool in tools) and not any(
                tool.get("type") == "image_generation" for tool in tools
            ):
                payload["tool_choice"] = {"type": "code_interpreter"}
        return payload

    def _build_chat_messages(self, request: TransformerLlmRequest, profile: ResolvedLlmProviderProfile) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for message in request.messages:
            if message.role == "system" and not profile.supports_system_prompt:
                continue
            text = self._flatten_text_content(message)
            if not text:
                continue
            messages.append({"role": message.role, "content": text})
        return messages

    def _build_responses_input(
        self,
        request: TransformerLlmRequest,
        profile: ResolvedLlmProviderProfile,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "system" and not profile.supports_system_prompt:
                continue
            items.append(
                {
                    "role": message.role,
                    "content": self._build_responses_content(message),
                }
            )
        return items

    def _extract_output_text(self, profile: ResolvedLlmProviderProfile, payload: dict[str, Any]) -> str:
        if profile.api_family == "chat_completions":
            choices = payload.get("choices")
            if isinstance(choices, list):
                for item in choices:
                    if not isinstance(item, dict):
                        continue
                    message = item.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str) and content.strip():
                            return content.strip()
            raise ValueError("OpenAI-compatible chat completion returned no text content")

        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output = payload.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        text = part.get("text")
                        if isinstance(text, str) and text.strip():
                            return text.strip()
        raise ValueError("Responses API returned no output text")

    def _extract_finish_reason(self, profile: ResolvedLlmProviderProfile, payload: dict[str, Any]) -> str | None:
        if profile.api_family == "chat_completions":
            choices = payload.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    value = first.get("finish_reason")
                    if isinstance(value, str) and value.strip():
                        return value.strip()
            return None
        value = payload.get("status")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    def _extract_usage(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        usage = payload.get("usage")
        if isinstance(usage, dict):
            return usage
        return None

    def _extract_generated_images(self, profile: ResolvedLlmProviderProfile, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if profile.api_family == "chat_completions":
            return []

        output = payload.get("output", [])
        if not isinstance(output, list):
            return []

        images: list[dict[str, Any]] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "image_generation_call":
                continue
            result = item.get("result")
            if isinstance(result, str) and result:
                images.append({"media_type": "image/png", "base64_data": result})
        return images

    def _build_responses_content(self, message: TransformerLlmMessage) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for part in message.content:
            if part.type == "text" and part.text is not None:
                content_type = "output_text" if message.role == "assistant" else "input_text"
                content.append({"type": content_type, "text": part.text})
            elif part.type == "image_file" and part.file_id is not None:
                content.append({"type": "input_image", "file_id": part.file_id})
        return content

    def _build_tools(self, request: TransformerLlmRequest) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for tool in request.tools:
            if tool.type == "code_interpreter":
                tools.append(
                    {
                        "type": "code_interpreter",
                        "container": {
                            "type": "auto",
                            "file_ids": tool.file_ids,
                        },
                    }
                )
            elif tool.type == "image_generation":
                payload: dict[str, Any] = {"type": "image_generation"}
                if tool.quality:
                    payload["quality"] = tool.quality
                tools.append(payload)
        return tools

    def _validate_chat_completion_request(self, request: TransformerLlmRequest) -> None:
        if request.tools:
            raise ValueError("Chat completions adapter does not support tool requests.")
        for message in request.messages:
            for part in message.content:
                if part.type != "text":
                    raise ValueError("Chat completions adapter only supports text content.")

    def _flatten_text_content(self, message: TransformerLlmMessage) -> str:
        parts = [part.text.strip() for part in message.content if part.type == "text" and isinstance(part.text, str) and part.text.strip()]
        return "\n".join(parts).strip()

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
                code = error_payload.get("code")
                if isinstance(code, str) and code.strip():
                    return code.strip()
        return fallback

    def _extract_error_message(self, payload: dict[str, Any] | list[Any] | None, fallback: str) -> str:
        if isinstance(payload, dict):
            error_payload = payload.get("error")
            if isinstance(error_payload, dict):
                message = error_payload.get("message")
                if isinstance(message, str) and message.strip():
                    return message.strip()
        return fallback
