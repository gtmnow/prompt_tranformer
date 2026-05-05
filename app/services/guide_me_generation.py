from __future__ import annotations

import json
from dataclasses import dataclass

from app.schemas.transform import GuideMeHelperKind
from app.services.llm_gateway import LlmGatewayService
from app.services.llm_types import TransformerLlmRequest
from app.services.llm_types import NormalizedTokenUsage
from app.services.runtime_llm import RuntimeLlmConfig


@dataclass(frozen=True)
class GuideMeGenerationResult:
    payload: dict
    usage: NormalizedTokenUsage | None


class GuideMeGenerationService:
    def __init__(self) -> None:
        self.gateway = LlmGatewayService()

    def generate(
        self,
        *,
        helper_kind: GuideMeHelperKind,
        prompt: str,
        runtime_config: RuntimeLlmConfig,
        max_output_tokens: int,
    ) -> GuideMeGenerationResult:
        request = TransformerLlmRequest(
            provider=runtime_config.provider,  # type: ignore[arg-type]
            model=runtime_config.model,
            base_url=runtime_config.endpoint_url or "https://api.openai.com/v1",
            api_key=runtime_config.api_key,
            system_prompt=self._build_system_prompt(helper_kind),
            user_prompt=prompt,
            max_output_tokens=max_output_tokens,
            temperature=0,
            expected_output="json",
        )
        response, error = self.gateway.invoke(request)
        if error is not None:
            raise ValueError(f"Guide Me helper request failed: {error.message}")
        if response is None:
            raise ValueError("Guide Me helper request returned no response.")
        try:
            payload = json.loads(response.output_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Guide Me helper returned invalid JSON for {helper_kind}.") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Guide Me helper returned a non-object payload for {helper_kind}.")
        return GuideMeGenerationResult(
            payload=payload,
            usage=response.normalized_usage,
        )

    def _build_system_prompt(self, helper_kind: GuideMeHelperKind) -> str:
        return (
            "You are a structured assistant for Herman Guide Me. "
            f"The current helper kind is '{helper_kind}'. "
            "Return only a valid JSON object with no markdown, commentary, or code fences. "
            "Use concise string values unless the user prompt clearly asks for an array or nested object. "
            "Do not include fields that are unsupported by the request."
        )
