import httpx

from app.services.llm_adapters.openai import OpenAIAdapter


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
