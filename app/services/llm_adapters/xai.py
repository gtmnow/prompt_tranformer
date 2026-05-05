from __future__ import annotations

from dataclasses import replace
from typing import Any

import httpx

from app.services.llm_adapters.openai import OpenAIAdapter
from app.services.llm_provider_profiles import ResolvedLlmProviderProfile
from app.services.llm_types import TransformerLlmError, TransformerLlmRequest, TransformerLlmResponse
from app.services.token_usage import normalize_usage


class XAIAdapter(OpenAIAdapter):
    provider_name = "xai"

    def invoke(
        self,
        request: TransformerLlmRequest,
        profile: ResolvedLlmProviderProfile,
    ) -> tuple[TransformerLlmResponse | None, TransformerLlmError | None]:
        if request.purpose == "final_response" and self._is_image_generation_request(request):
            return self._invoke_image_generation(request)
        if request.purpose == "final_response" and self._requires_responses_api(request):
            responses_profile = replace(
                profile,
                api_family="responses",
                endpoint_path="/responses",
                token_parameter="max_output_tokens",
            )
            return super().invoke(request, responses_profile)
        return super().invoke(request, profile)

    def _invoke_image_generation(
        self,
        request: TransformerLlmRequest,
    ) -> tuple[TransformerLlmResponse | None, TransformerLlmError | None]:
        url = f"{request.base_url.rstrip('/')}/images/generations"
        headers = {
            "Authorization": f"Bearer {request.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": "grok-imagine-image",
            "prompt": request.user_prompt or self._flatten_prompt_text(request),
            "response_format": "b64_json",
        }

        try:
            with httpx.Client(timeout=request.timeout_seconds) as client:
                response = client.post(url, headers=headers, json=payload)
            response_payload = response.json()
            response.raise_for_status()
            usage = response_payload.get("usage") if isinstance(response_payload, dict) else None
            generated_images = self._extract_xai_generated_images(response_payload)
            if not generated_images:
                raise ValueError("xAI image generation returned no images.")
            return (
                TransformerLlmResponse(
                    provider=request.provider,
                    model="grok-imagine-image",
                    output_text="",
                    generated_images=generated_images,
                    status_code=response.status_code,
                    finish_reason="completed",
                    usage=usage if isinstance(usage, dict) else None,
                    normalized_usage=normalize_usage(request.provider, usage if isinstance(usage, dict) else None),
                    raw_payload=response_payload if isinstance(response_payload, dict) else None,
                ),
                None,
            )
        except httpx.HTTPStatusError as exc:
            payload = self._safe_json(exc.response)
            return None, TransformerLlmError(
                provider=request.provider,
                model="grok-imagine-image",
                code=self._extract_error_code(payload, fallback=f"HTTP_{exc.response.status_code}"),
                message=self._extract_error_message(payload, exc.response.text),
                status_code=exc.response.status_code,
                raw_payload=payload if isinstance(payload, (dict, list)) else None,
            )
        except httpx.HTTPError as exc:
            return None, TransformerLlmError(
                provider=request.provider,
                model="grok-imagine-image",
                code=exc.__class__.__name__.upper(),
                message=str(exc),
            )
        except ValueError as exc:
            return None, TransformerLlmError(
                provider=request.provider,
                model="grok-imagine-image",
                code="INVALID_RESPONSE",
                message=str(exc),
            )

    def _extract_xai_generated_images(self, payload: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        if not isinstance(data, list):
            return []

        images: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            base64_data = item.get("b64_json")
            if isinstance(base64_data, str) and base64_data:
                images.append({"media_type": "image/jpeg", "base64_data": base64_data})
        return images

    def _is_image_generation_request(self, request: TransformerLlmRequest) -> bool:
        return any(tool.type == "image_generation" for tool in request.tools)

    def _requires_responses_api(self, request: TransformerLlmRequest) -> bool:
        return any(part.type in {"image_file", "document_file"} for message in request.messages for part in message.content)

    def _flatten_prompt_text(self, request: TransformerLlmRequest) -> str:
        parts: list[str] = []
        for message in request.messages:
            if message.role != "user":
                continue
            parts.extend(
                part.text.strip()
                for part in message.content
                if part.type == "text" and isinstance(part.text, str) and part.text.strip()
            )
        return "\n".join(parts).strip()
