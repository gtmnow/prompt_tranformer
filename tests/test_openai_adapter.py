import httpx

from app.services.llm_adapters.openai import OpenAIAdapter
from app.services.llm_provider_profiles import ResolvedLlmProviderProfile
from app.services.llm_types import TransformerLlmRequest


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


def test_openai_adapter_uses_structured_input_over_prompt_fields() -> None:
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
        provider="openai",
        model="gpt-4.1",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        system_prompt="ignored system",
        user_prompt="ignored prompt",
        input_items=[{"role": "user", "content": [{"type": "input_text", "text": "structured"}]}],
        tools=[{"type": "web_search"}],
        text_format={"format": {"type": "text"}},
    )

    payload = adapter._build_payload(request, profile)  # type: ignore[attr-defined]

    assert payload["input"] == request.input_items
    assert payload["tools"] == [{"type": "web_search"}]
    assert payload["text"] == {"format": {"type": "text"}}


def test_openai_adapter_allows_image_generation_responses_without_text() -> None:
    adapter = OpenAIAdapter()
    profile = ResolvedLlmProviderProfile(
        provider="openai",
        requested_model="gpt-5.5",
        resolved_model="gpt-5.5",
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
        supports_image_generation=True,
        raw={},
    )

    payload = adapter._extract_output_text(  # type: ignore[attr-defined]
        profile=profile,
        payload={
            "output": [
                {
                    "type": "image_generation_call",
                    "result": "iVBORw0KGgo=",
                }
            ]
        },
    )

    assert payload == ""
