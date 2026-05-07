from __future__ import annotations

import unittest
from unittest.mock import patch

from app.schemas.transform import AttachmentReference, ConversationHistoryTurn
from app.services.final_response_service import (
    OPENAI_WEB_SEARCH_MIN_OUTPUT_TOKENS,
    _build_openai_like_payload,
    _extract_incomplete_response_error,
    _extract_output_text,
)
from app.services.rag_prompt_assembly_service import RagPromptAssemblyService
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

    def test_build_openai_like_payload_uses_configured_output_token_limit(self) -> None:
        with patch("app.services.final_response_service.get_settings") as get_settings_mock:
            get_settings_mock.return_value.final_response_max_output_tokens = 512
            payload = _build_openai_like_payload(
                profile=_profile(api_family="chat_completions", token_parameter="max_completion_tokens"),
                model="grok-3",
                conversation_history=[],
                transformed_prompt="Task: Write a concise reply.",
                image_attachments=[],
                document_attachments=[],
                wants_image_generation=False,
                max_output_tokens=get_settings_mock.return_value.final_response_max_output_tokens,
            )

        self.assertEqual(payload["max_completion_tokens"], 512)

    def test_build_openai_like_payload_includes_web_search_for_responses(self) -> None:
        payload = _build_openai_like_payload(
            profile=_profile(api_family="responses", token_parameter="max_output_tokens"),
            model="gpt-4.1",
            conversation_history=[],
            transformed_prompt="Find the latest product announcement.",
            image_attachments=[],
            document_attachments=[],
            wants_image_generation=False,
            max_output_tokens=800,
        )

        self.assertIn("tools", payload)
        self.assertIn({"type": "web_search"}, payload["tools"])
        self.assertEqual(payload["max_output_tokens"], OPENAI_WEB_SEARCH_MIN_OUTPUT_TOKENS)

    def test_build_openai_like_payload_does_not_include_web_search_without_search_prompt(self) -> None:
        payload = _build_openai_like_payload(
            profile=_profile(api_family="responses", token_parameter="max_output_tokens"),
            model="gpt-4.1",
            conversation_history=[],
            transformed_prompt="Summarize this product strategy memo for exec review.",
            image_attachments=[],
            document_attachments=[],
            wants_image_generation=False,
            max_output_tokens=800,
        )

        self.assertNotIn("tools", payload)
        self.assertEqual(payload["max_output_tokens"], 800)

    def test_build_openai_like_payload_keeps_web_search_available_with_documents(self) -> None:
        payload = _build_openai_like_payload(
            profile=_profile(api_family="responses", token_parameter="max_output_tokens"),
            model="gpt-4.1",
            conversation_history=[],
            transformed_prompt="Compare this document against the latest public guidance.",
            image_attachments=[],
            document_attachments=[
                AttachmentReference(
                    id="doc_1",
                    kind="document",
                    name="brief.pdf",
                    provider_file_id="file_123",
                )
            ],
            wants_image_generation=False,
            max_output_tokens=800,
        )

        self.assertIn("tools", payload)
        self.assertIn({"type": "web_search"}, payload["tools"])
        self.assertEqual(payload["tools"][1]["type"], "code_interpreter")
        self.assertNotIn("tool_choice", payload)
        self.assertEqual(payload["max_output_tokens"], OPENAI_WEB_SEARCH_MIN_OUTPUT_TOKENS)

    def test_build_openai_like_payload_xai_responses_uses_web_search_without_openai_only_tools(self) -> None:
        xai_profile = ResolvedLlmProviderProfile(
            provider="xai",
            requested_model="grok-3-mini",
            resolved_model="grok-3-mini",
            api_family="responses",
            endpoint_path="/responses",
            auth_scheme="bearer",
            auth_header_name=None,
            version_header_name=None,
            version_header_value=None,
            json_mode="prompt_only",
            token_parameter="max_output_tokens",
            supports_system_prompt=True,
            request_timeout_seconds=15.0,
            raw={},
        )

        payload = _build_openai_like_payload(
            profile=xai_profile,
            model="grok-3-mini",
            conversation_history=[],
            transformed_prompt="Find the latest hiring trends.",
            image_attachments=[],
            document_attachments=[
                AttachmentReference(
                    id="doc_1",
                    kind="document",
                    name="brief.pdf",
                    provider_file_id="file_123",
                )
            ],
            wants_image_generation=False,
            max_output_tokens=800,
        )

        self.assertEqual(payload["tools"], [{"type": "web_search"}])
        self.assertIn("input", payload)
        self.assertNotIn("messages", payload)
        self.assertEqual(payload["max_output_tokens"], 800)

    def test_extract_incomplete_response_error_reports_max_tokens_reason(self) -> None:
        error = _extract_incomplete_response_error(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_tokens"},
                "output": [],
            }
        )

        self.assertEqual(
            error,
            "LLM provider response was incomplete because it hit the max_output_tokens limit.",
        )

    def test_rag_prompt_assembly_compresses_and_budgets_reference_context(self) -> None:
        service = RagPromptAssemblyService()

        assembled = service.assemble(
            references=[
                {
                    "filename": "resume.txt",
                    "chunk_text": (
                        "Michael has deep product marketing and GTM leadership experience across B2B SaaS. "
                        "He has led lifecycle, demand generation, and positioning work. "
                        "He also enjoys cooking elaborate weekend breakfasts with friends."
                    ),
                },
                {
                    "filename": "notes.txt",
                    "chunk_text": (
                        "Strong fit areas include product marketing, growth, lifecycle, and GTM strategy roles. "
                        "Keep recommendations concise and recruiter-friendly."
                    ),
                },
            ],
            query_text="What jobs best fit my product marketing and GTM background?",
            max_sources=2,
            max_total_words=25,
            max_words_per_source=15,
        )

        self.assertIsNotNone(assembled)
        assert assembled is not None
        self.assertLessEqual(len(assembled.split()), 45)
        self.assertIn("product marketing", assembled.lower())


if __name__ == "__main__":
    unittest.main()
