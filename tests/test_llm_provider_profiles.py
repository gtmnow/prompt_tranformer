from __future__ import annotations

from app.services.llm_provider_profiles import LlmProviderProfileService


def test_resolve_exact_model_and_normalized_correction() -> None:
    service = LlmProviderProfileService()

    exact = service.resolve("xai", "grok-4-1-fast-reasoning")
    assert exact.requested_model == "grok-4-1-fast-reasoning"
    assert exact.resolved_model == "grok-4-1-fast-reasoning"
    assert exact.model_is_malformed is False
    assert exact.used_fallback_model is False

    corrected = service.resolve("xai", "grok_4_1_fast_reasoning")
    assert corrected.requested_model == "grok_4_1_fast_reasoning"
    assert corrected.resolved_model == "grok-4-1-fast-reasoning"
    assert corrected.model_is_malformed is True
    assert corrected.used_fallback_model is False


def test_resolve_unsupported_model_falls_back_and_marks_fallback() -> None:
    service = LlmProviderProfileService()

    fallback = service.resolve("xai", "grok-4-2-reasoning")

    assert fallback.resolved_model == "grok-4-1-fast-reasoning"
    assert fallback.model_is_malformed is True
    assert fallback.used_fallback_model is True


def test_resolve_default_model_falls_back_when_default_is_not_available() -> None:
    service = LlmProviderProfileService()
    service.rules = {
        "default": {"request_timeout_seconds": 45.0},
        "providers": {
            "xai": {
                "models": {
                    "grok-4-1": {"api_family": "responses"},
                    "grok-4-1-fast": {"api_family": "responses"},
                },
                "default_model": "missing-default",
            }
        },
    }

    default_model = service.resolve_default_model("xai")

    assert default_model == "grok-4-1"
