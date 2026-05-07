from __future__ import annotations

import unittest
from unittest.mock import patch

from app.services.guide_me_generation import GuideMeGenerationError, GuideMeGenerationService
from app.services.llm_types import NormalizedTokenUsage, TransformerLlmError, TransformerLlmResponse
from app.services.runtime_llm import RuntimeLlmConfig


class GuideMeGenerationServiceTests(unittest.TestCase):
    def test_generate_returns_payload_and_usage(self) -> None:
        service = GuideMeGenerationService()
        runtime_config = RuntimeLlmConfig(
            tenant_id="tenant_1",
            user_id_hash="user_1",
            provider="openai",
            model="gpt-4.1",
            endpoint_url="https://api.openai.com/v1",
            api_key="test-key",
            transformation_enabled=True,
            scoring_enabled=True,
            credential_status="valid",
            source_kind="customer_managed",
        )

        with patch.object(
            service.gateway,
            "invoke",
            return_value=(
                TransformerLlmResponse(
                    provider="openai",
                    model="gpt-4.1",
                    output_text='{"task":"Reduce unqualified applicants by 30%."}',
                    normalized_usage=NormalizedTokenUsage(
                        input_tokens=90,
                        output_tokens=30,
                        total_tokens=120,
                    ),
                ),
                None,
            ),
        ) as invoke:
            result = service.generate(
                helper_kind="answer_extraction",
                prompt="Return strict JSON with the best task field.",
                runtime_config=runtime_config,
                max_output_tokens=400,
            )

        invoke.assert_called_once()
        self.assertEqual(result.payload["task"], "Reduce unqualified applicants by 30%.")
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.usage.total_tokens, 120)

    def test_generate_raises_provider_error_with_status_code(self) -> None:
        service = GuideMeGenerationService()
        runtime_config = RuntimeLlmConfig(
            tenant_id="tenant_1",
            user_id_hash="user_1",
            provider="openai",
            model="gpt-4.1",
            endpoint_url="https://api.openai.com/v1",
            api_key="test-key",
            transformation_enabled=True,
            scoring_enabled=True,
            credential_status="valid",
            source_kind="customer_managed",
        )

        with patch.object(
            service.gateway,
            "invoke",
            return_value=(
                None,
                TransformerLlmError(
                    provider="openai",
                    model="gpt-4.1",
                    code="RATE_LIMITED",
                    message="Rate limit exceeded",
                    status_code=429,
                ),
            ),
        ) as invoke:
            with self.assertRaises(GuideMeGenerationError) as exc_info:
                service.generate(
                    helper_kind="answer_extraction",
                    prompt="Return strict JSON with the best task field.",
                    runtime_config=runtime_config,
                    max_output_tokens=400,
                )

        invoke.assert_called_once()
        self.assertEqual(exc_info.exception.status_code, 429)

if __name__ == "__main__":
    unittest.main()
