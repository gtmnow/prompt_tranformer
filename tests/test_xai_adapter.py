from dataclasses import replace

from app.services.llm_adapters.xai import XAIAdapter
from app.services.llm_provider_profiles import ResolvedLlmProviderProfile
from app.services.llm_types import TransformerLlmContentPart, TransformerLlmMessage, TransformerLlmRequest, TransformerLlmToolRequest


def _xai_profile() -> ResolvedLlmProviderProfile:
    return ResolvedLlmProviderProfile(
        provider="xai",
        requested_model="grok-3",
        resolved_model="grok-3",
        api_family="chat_completions",
        endpoint_path="/chat/completions",
        auth_scheme="bearer",
        auth_header_name=None,
        version_header_name=None,
        version_header_value=None,
        json_mode="response_format_json_object",
        token_parameter="max_completion_tokens",
        supports_system_prompt=True,
        request_timeout_seconds=15.0,
        raw={},
    )


def test_xai_adapter_uses_responses_payload_for_uploaded_files() -> None:
    adapter = XAIAdapter()
    request = TransformerLlmRequest(
        purpose="final_response",
        provider="xai",
        model="grok-3",
        base_url="https://api.x.ai/v1",
        api_key="test-key",
        messages=[
            TransformerLlmMessage(
                role="user",
                content=[
                    TransformerLlmContentPart(type="text", text="Summarize this file and image."),
                    TransformerLlmContentPart(type="document_file", file_id="file_doc"),
                    TransformerLlmContentPart(type="image_file", file_id="file_img"),
                ],
            )
        ],
        max_output_tokens=800,
    )

    assert adapter._requires_responses_api(request) is True  # type: ignore[attr-defined]
    responses_profile = replace(
        _xai_profile(),
        api_family="responses",
        endpoint_path="/responses",
        token_parameter="max_output_tokens",
    )
    payload = adapter._build_payload(request, responses_profile)  # type: ignore[attr-defined]
    assert payload["input"][0]["content"][1] == {"type": "input_file", "file_id": "file_doc"}
    assert payload["input"][0]["content"][2] == {"type": "input_file", "file_id": "file_img"}


def test_xai_adapter_detects_image_generation_requests() -> None:
    adapter = XAIAdapter()
    request = TransformerLlmRequest(
        purpose="final_response",
        provider="xai",
        model="grok-3",
        base_url="https://api.x.ai/v1",
        api_key="test-key",
        user_prompt="Create an image of a lighthouse at sunset.",
        tools=[TransformerLlmToolRequest(type="image_generation", quality="high")],
    )

    assert adapter._is_image_generation_request(request) is True  # type: ignore[attr-defined]
