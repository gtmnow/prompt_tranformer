from __future__ import annotations

import json
from dataclasses import dataclass

from app.schemas.transform import GuideMeHelperKind
from app.services.llm_gateway import LlmGatewayService
from app.services.llm_request_factory import LlmRequestFactory
from app.services.llm_types import NormalizedTokenUsage
from app.services.runtime_llm import RuntimeLlmConfig


@dataclass(frozen=True)
class GuideMeGenerationResult:
    payload: dict
    usage: NormalizedTokenUsage | None


class GuideMeGenerationService:
    def __init__(self) -> None:
        self.gateway = LlmGatewayService()
        self.request_factory = LlmRequestFactory()

    def generate(
        self,
        *,
        helper_kind: GuideMeHelperKind,
        prompt: str,
        runtime_config: RuntimeLlmConfig,
        max_output_tokens: int,
    ) -> GuideMeGenerationResult:
        request = self.request_factory.build_guide_me_request(
            runtime_config=runtime_config,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
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
