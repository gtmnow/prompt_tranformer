from __future__ import annotations

import unittest

from app.schemas.transform import ConversationHistoryTurn
from app.services.final_response_service import _build_openai_like_payload, _extract_output_text
from app.services.llm_provider_profiles import ResolvedLlmProviderProfile


def _profile(*, api_family: str, token_parameter: str) -> ResolvedLlmProviderProfile:
    return ResolvedLlmProviderProfile(
        provider="xai" if api_family == "chat_completions" else "openai",
        requested_model="test-model",
        resolved_model="test-model",
        api_family=api_family,
        endpoint_path="/chat/completions" if api_family == "chat_completions" else "/responses",
        auth_scheme="bearer",
        auth_header_name=None,
        version_header_name=None,
        version_header_value=None,
        json_mode="prompt_only",
        token_parameter=token_parameter,
        supports_system_prompt=True,
        request_timeout_seconds=15.0,
        raw={},
    )


class FinalResponseServiceTests(unittest.TestCase):
    def test_build_openai_like_payload_uses_messages_for_chat_completions(self) -> None:
        payload = _build_openai_like_payload(
            profile=_profile(api_family="chat_completions", token_parameter="max_completion_tokens"),
            model="grok-3",
            conversation_history=[
                ConversationHistoryTurn(
                    transformed_text="Task: Summarize the email thread.",
                    assistant_text="Here is the earlier summary.",
                )
            ],
            transformed_prompt="Task: Write a concise reply.",
            image_attachments=[],
            document_attachments=[],
            wants_image_generation=False,
            max_output_tokens=800,
        )

        self.assertIn("messages", payload)
        self.assertNotIn("input", payload)
        self.assertEqual(
            payload["messages"],
            [
                {"role": "user", "content": "Task: Summarize the email thread."},
                {"role": "assistant", "content": "Here is the earlier summary."},
                {"role": "user", "content": "Task: Write a concise reply."},
            ],
        )
        self.assertEqual(payload["max_completion_tokens"], 800)

    def test_extract_output_text_reads_chat_completions_message_content(self) -> None:
        text = _extract_output_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Here is the xAI response.",
                        }
                    }
                ]
            },
            profile=_profile(api_family="chat_completions", token_parameter="max_completion_tokens"),
        )

        self.assertEqual(text, "Here is the xAI response.")


if __name__ == "__main__":
    unittest.main()
