from __future__ import annotations

import unittest
from unittest.mock import patch

from app.schemas.transform import AttachmentReference, ConversationHistoryTurn
from app.services.final_response_service import FinalResponseService
from app.services.llm_types import NormalizedTokenUsage, TransformerLlmResponse
from app.services.runtime_llm import RuntimeLlmConfig


class FinalResponseServiceTests(unittest.TestCase):
    def test_generate_uses_gateway_and_maps_images_and_usage(self) -> None:
        service = FinalResponseService()
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
                    output_text="Here is the answer.",
                    generated_images=[{"media_type": "image/png", "base64_data": "abc123"}],
                    normalized_usage=NormalizedTokenUsage(
                        input_tokens=100,
                        output_tokens=25,
                        total_tokens=125,
                    ),
                ),
                None,
            ),
        ) as invoke:
            result = service.generate(
                runtime_config=runtime_config,
                resolved_model="gpt-4.1",
                transformed_prompt="Task: Explain this clearly.",
                conversation_history=[
                    ConversationHistoryTurn(
                        transformed_text="Task: summarize the earlier thread.",
                        assistant_text="Earlier assistant reply.",
                    )
                ],
                attachments=[
                    AttachmentReference(
                        id="att_1",
                        kind="image",
                        name="diagram.png",
                        provider_file_id="file_123",
                    )
                ],
            )

        invoke.assert_called_once()
        self.assertEqual(result.text, "Here is the answer.")
        self.assertEqual(len(result.generated_images), 1)
        self.assertEqual(result.generated_images[0].base64_data, "abc123")
        self.assertEqual(result.usage.total_tokens, 125)


if __name__ == "__main__":
    unittest.main()
