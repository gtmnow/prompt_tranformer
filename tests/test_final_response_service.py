from __future__ import annotations

import unittest
from unittest.mock import patch

from app.schemas.transform import AttachmentReference, ConversationHistoryTurn
from app.services.final_response_service import (
    OPENAI_WEB_SEARCH_MIN_OUTPUT_TOKENS,
    FinalResponseIntent,
    FinalResponseProviderError,
    _build_input_items,
    _build_messages,
    _append_max_output_budget,
    _extract_incomplete_response_error,
    _extract_generated_images,
    resolve_final_response_intent,
)
from app.services.llm_provider_profiles import ResolvedLlmProviderProfile
from app.services.llm_provider_profiles import LlmProviderProfileService
from app.services.rag_prompt_assembly_service import RagPromptAssemblyService
from app.core.config import get_settings
from app.services.runtime_llm import RuntimeLlmConfig
from app.services.llm_types import TransformerLlmResponse


def _profile(
    *,
    provider: str = "openai",
    api_family: str,
    token_parameter: str,
    supports_web_search: bool = False,
    raw: dict | None = None,
) -> ResolvedLlmProviderProfile:
    return ResolvedLlmProviderProfile(
        provider=provider,
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
        supports_web_search=supports_web_search,
        raw=raw or {},
    )


def _runtime_config(*, provider: str = "openai", model: str = "gpt-4.1") -> RuntimeLlmConfig:
    return RuntimeLlmConfig(
        tenant_id="tenant_1",
        user_id_hash="user_1",
        provider=provider,
        model=model,
        endpoint_url="https://api.openai.com/v1" if provider != "xai" else "https://api.x.ai/v1",
        api_key="test-key",
        transformation_enabled=True,
        scoring_enabled=True,
        credential_status="valid",
        source_kind="customer_managed",
    )


class FinalResponseServiceTests(unittest.TestCase):
    def test_provider_profiles_preserve_supported_model(self) -> None:
        profile = LlmProviderProfileService().resolve("xai", "grok-4-1")

        self.assertEqual(profile.requested_model, "grok-4-1")
        self.assertEqual(profile.resolved_model, "grok-4-1")
        self.assertEqual(profile.api_family, "responses")

    def test_build_messages_uses_conversation_history_for_chat_completions(self) -> None:
        messages = _build_messages(
            conversation_history=[
                ConversationHistoryTurn(
                    transformed_text="Task: Summarize the email thread.",
                    assistant_text="Here is the earlier summary.",
                )
            ],
            transformed_prompt="Task: Write a concise reply.",
        )

        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "Task: Summarize the email thread."},
                {"role": "assistant", "content": "Here is the earlier summary."},
                {"role": "user", "content": "Task: Write a concise reply."},
            ],
        )

    def test_build_input_items_uses_images_for_responses_api(self) -> None:
        input_items = _build_input_items(
            conversation_history=[],
            transformed_prompt="Task: Review this image.",
            image_attachments=[
                AttachmentReference(
                    id="img_1",
                    kind="image",
                    name="diagram.png",
                    provider_file_id="file_image_123",
                )
            ],
        )

        self.assertEqual(
            input_items,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Task: Review this image."},
                        {"type": "input_image", "file_id": "file_image_123"},
                    ],
                }
            ],
        )

    def test_resolve_final_response_intent_preserves_raw_web_lookup_intent(self) -> None:
        intent = resolve_final_response_intent(
            raw_prompt="What jobs are my best fit in today's marketplace?",
            transformed_prompt="Recommend the best roles for the user based on their background.",
        )

        self.assertEqual(
            intent,
            FinalResponseIntent(use_web_search=True, use_image_generation=False),
        )

    def test_resolve_final_response_intent_does_not_enable_web_search_for_generic_summary(self) -> None:
        intent = resolve_final_response_intent(
            raw_prompt="Summarize this product strategy memo for exec review.",
            transformed_prompt="Summarize the memo clearly for executives.",
        )

        self.assertEqual(
            intent,
            FinalResponseIntent(use_web_search=False, use_image_generation=False),
        )

    def test_resolve_final_response_intent_prefers_explicit_request(self) -> None:
        intent = resolve_final_response_intent(
            raw_prompt="Summarize this product strategy memo for exec review.",
            transformed_prompt="Summarize the memo clearly for executives.",
            request_live_web_search=True,
        )

        self.assertEqual(
            intent,
            FinalResponseIntent(use_web_search=True, use_image_generation=False),
        )

    def test_resolve_final_response_intent_false_overrides_keywords(self) -> None:
        intent = resolve_final_response_intent(
            raw_prompt="What are today's top trends in candidate sourcing?",
            transformed_prompt="What are today's top trends in candidate sourcing?",
            request_live_web_search=False,
        )

        self.assertEqual(
            intent,
            FinalResponseIntent(use_web_search=False, use_image_generation=False),
        )

    def test_generate_includes_max_output_budget_directive(self) -> None:
        from app.services.final_response_service import FinalResponseService

        service = FinalResponseService()
        profile = _profile(
            provider="openai",
            api_family="responses",
            token_parameter="max_output_tokens",
            supports_web_search=True,
        )

        with patch.object(service.provider_profiles, "resolve", return_value=profile), patch.object(
            service.gateway,
            "invoke",
            return_value=(
                TransformerLlmResponse(
                    provider="openai",
                    model="test-model",
                    output_text="Done.",
                    raw_payload={"output": []},
                    usage={},
                ),
                None,
            ),
        ) as invoke_mock:
            service.generate(
                runtime_config=_runtime_config(provider="openai", model="gpt-4.1"),
                transformed_prompt="Summarize the policy briefly.",
                conversation_history=[],
                attachments=[],
                intent=FinalResponseIntent(use_web_search=False, use_image_generation=False),
            )

        actual_request = invoke_mock.call_args.args[0]
        self.assertEqual(
            actual_request.user_prompt,
            _append_max_output_budget("Summarize the policy briefly.", get_settings().final_response_max_output_tokens),
        )

    def test_append_max_output_budget_directive(self) -> None:
        self.assertIn(
            "do not exceed 420 output tokens",
            _append_max_output_budget("Generate a short summary.", 420),
        )

    def test_build_request_uses_profile_resolved_model_and_gateway_fields(self) -> None:
        from app.services.final_response_service import FinalResponseService

        service = FinalResponseService()
        profile = _profile(
            provider="openai",
            api_family="responses",
            token_parameter="max_output_tokens",
            supports_web_search=True,
        )

        with patch.object(service.provider_profiles, "resolve", return_value=profile):
            request = service._build_request(
                runtime_config=_runtime_config(model="unknown-model"),
                transformed_prompt="Find the latest product announcement.",
                conversation_history=[],
                attachments=[],
                intent=FinalResponseIntent(use_web_search=True, use_image_generation=False),
                profile=profile,
            )

        self.assertEqual(request.model, "test-model")
        self.assertEqual(request.tools, [{"type": "web_search"}])
        self.assertEqual(request.text_format, {"format": {"type": "text"}})
        self.assertEqual(request.max_output_tokens, OPENAI_WEB_SEARCH_MIN_OUTPUT_TOKENS)
        self.assertIn("input", {"input": request.input_items})

    def test_build_request_xai_uses_responses_input_without_openai_only_tools(self) -> None:
        from app.services.final_response_service import FinalResponseService

        service = FinalResponseService()
        profile = _profile(
            provider="xai",
            api_family="responses",
            token_parameter="max_output_tokens",
            supports_web_search=False,
        )

        with patch.object(service.provider_profiles, "resolve", return_value=profile):
            request = service._build_request(
                runtime_config=_runtime_config(provider="xai", model="grok-4-1"),
                transformed_prompt="Find the latest hiring trends.",
                conversation_history=[],
                attachments=[
                    AttachmentReference(
                        id="doc_1",
                        kind="document",
                        name="brief.pdf",
                        provider_file_id="file_123",
                    )
                ],
                intent=FinalResponseIntent(use_web_search=True, use_image_generation=False),
                profile=profile,
            )
        self.assertEqual(request.tools, [])
        self.assertTrue(request.input_items)
        self.assertFalse(request.conversation_messages)

    def test_generate_raises_for_unsupported_web_search_model(self) -> None:
        from app.services.final_response_service import FinalResponseService

        service = FinalResponseService()
        unsupported_profile = _profile(
            provider="xai",
            api_family="responses",
            token_parameter="max_output_tokens",
            supports_web_search=False,
        )

        with patch.object(
            service.provider_profiles,
            "resolve",
            return_value=unsupported_profile,
        ):
            with self.assertRaises(FinalResponseProviderError) as exc_info:
                service.generate(
                    runtime_config=_runtime_config(provider="xai", model="grok-4-1"),
                    transformed_prompt="Find the latest hiring trends.",
                    conversation_history=[],
                    attachments=[],
                    intent=FinalResponseIntent(use_web_search=True, use_image_generation=False),
                )
            self.assertIn("does not support live web retrievals", str(exc_info.exception))
            self.assertEqual(exc_info.exception.status_code, 400)

        profile = _profile(
            provider="xai",
            api_family="responses",
            token_parameter="max_output_tokens",
            supports_web_search=True,
        )
        with patch.object(service.provider_profiles, "resolve", return_value=profile):
            request = service._build_request(
                runtime_config=_runtime_config(provider="xai", model="grok-4-1"),
                transformed_prompt="Find the latest hiring trends.",
                conversation_history=[],
                attachments=[
                    AttachmentReference(
                        id="doc_1",
                        kind="document",
                        name="brief.pdf",
                        provider_file_id="file_123",
                    )
                ],
                intent=FinalResponseIntent(use_web_search=False, use_image_generation=False),
                profile=profile,
            )

        self.assertEqual(request.tools, [])
        self.assertTrue(request.input_items)
        self.assertFalse(request.conversation_messages)

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

    def test_extract_incomplete_response_error_reports_max_output_tokens_reason(self) -> None:
        error = _extract_incomplete_response_error(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
            }
        )

        self.assertEqual(
            error,
            "LLM provider response was incomplete because it hit the max_output_tokens limit.",
        )

    def test_extract_generated_images_preserves_media_type(self) -> None:
        images = _extract_generated_images(
            {
                "output": [
                    {
                        "type": "image_generation_call",
                        "result": {
                            "data": "iVBORw0KGgo=",
                            "media_type": "image/jpeg",
                        },
                    }
                ]
            }
        )

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].media_type, "image/jpeg")
        self.assertEqual(images[0].base64_data, "iVBORw0KGgo=")

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
