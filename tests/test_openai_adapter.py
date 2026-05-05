import httpx

from app.services.llm_adapters.openai import OpenAIAdapter
from app.services.llm_provider_profiles import ResolvedLlmProviderProfile
from app.services.llm_types import TransformerLlmContentPart, TransformerLlmMessage, TransformerLlmRequest, TransformerLlmToolRequest


def _build_response(status_code: int, payload: dict) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )


def test_openai_adapter_retries_without_temperature_on_unsupported_parameter() -> None:
    adapter = OpenAIAdapter()

    unsupported_temperature = _build_response(
        400,
        {
            "error": {
                "message": "Unsupported parameter: 'temperature' is not supported with this model.",
            }
        },
    )

    assert (
        adapter._should_retry_without_temperature(  # type: ignore[attr-defined]
            unsupported_temperature,
            {"model": "gpt-5.5", "temperature": 0, "input": []},
        )
        is True
    )


def test_openai_adapter_does_not_retry_for_other_bad_requests() -> None:
    adapter = OpenAIAdapter()

    unrelated_bad_request = _build_response(
        400,
        {
            "error": {
                "message": "Unsupported parameter: 'top_p' is not supported with this model.",
            }
        },
    )

    assert (
        adapter._should_retry_without_temperature(  # type: ignore[attr-defined]
            unrelated_bad_request,
            {"model": "gpt-5.5", "temperature": 0, "input": []},
        )
        is False
    )


def test_openai_adapter_builds_final_response_payload_with_tools_and_images() -> None:
    adapter = OpenAIAdapter()
    profile = ResolvedLlmProviderProfile(
        provider="openai",
        requested_model="gpt-4.1",
        resolved_model="gpt-4.1",
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
    request = TransformerLlmRequest(
        purpose="final_response",
        provider="openai",
        model="gpt-4.1",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        messages=[
            TransformerLlmMessage(
                role="user",
                content=[
                    TransformerLlmContentPart(type="text", text="Create image options."),
                    TransformerLlmContentPart(type="image_file", file_id="file_img"),
                    TransformerLlmContentPart(type="document_file", file_id="file_doc_2"),
                ],
            )
        ],
        tools=[
            TransformerLlmToolRequest(type="code_interpreter", file_ids=["file_doc"]),
            TransformerLlmToolRequest(type="image_generation", quality="high"),
        ],
        max_output_tokens=800,
    )

    payload = adapter._build_payload(request, profile)  # type: ignore[attr-defined]

    assert payload["input"][0]["content"][1] == {"type": "input_image", "file_id": "file_img"}
    assert payload["input"][0]["content"][2] == {"type": "input_file", "file_id": "file_doc_2"}
    assert payload["tools"][0]["type"] == "code_interpreter"
    assert payload["tools"][1]["type"] == "image_generation"
