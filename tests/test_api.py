from app.db.seed import PROFILE_ROWS
from app.models.profile import FinalProfile
from app.models.request_log import PromptTransformRequest
from app.core.config import get_settings
from app.db.session import get_db


AUTH_HEADERS = {
    "Authorization": "Bearer test-transformer-key",
    "X-Client-Id": "hermanprompt",
}


def _seed_final_profiles(client) -> None:
    db = next(client.app.dependency_overrides[get_db]())
    try:
        for row in PROFILE_ROWS:
            db.add(FinalProfile(**row))
        db.commit()
    finally:
        db.close()


def _update_profile(client, user_id: str, **overrides) -> None:
    db = next(client.app.dependency_overrides[get_db]())
    try:
        profile = db.get(FinalProfile, user_id)
        if profile is None:
            raise AssertionError(f"Missing profile for {user_id}")
        for key, value in overrides.items():
            setattr(profile, key, value)
        db.commit()
    finally:
        db.close()


def test_transform_uses_db_profile(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_123",
            "conversation_id": "conv_123",
            "user_id": "user_1",
            "raw_prompt": "Explain this concept simply",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == "conv_123"
    assert body["result_type"] == "transformed"
    assert body["task_type"] == "explanation"
    assert body["metadata"]["persona_source"] == "db_profile"
    assert "Start with the direct answer" in body["transformed_prompt"]


def test_transform_uses_summary_override(client) -> None:
    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_456",
            "conversation_id": "conv_456",
            "user_id": "user_missing",
            "raw_prompt": "Summarize this article",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "summary_type": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == "conv_456"
    assert body["result_type"] == "transformed"
    assert body["task_type"] == "summarization"
    assert body["metadata"]["persona_source"] == "summary_override"


def test_transform_falls_back_to_generic_default(client) -> None:
    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_789",
            "conversation_id": "conv_789",
            "user_id": "user_missing",
            "raw_prompt": "What should I do next?",
            "target_llm": {"provider": "openai", "model": "unknown-model"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == "conv_789"
    assert body["result_type"] == "transformed"
    assert body["task_type"] == "unknown"
    assert body["metadata"]["persona_source"] == "generic_default"
    assert body["metadata"]["used_fallback_model"] is True


def test_invalid_summary_type_returns_400(client) -> None:
    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_999",
            "conversation_id": "conv_999",
            "user_id": "user_1",
            "raw_prompt": "Explain this concept simply",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "summary_type": 10,
        },
    )

    assert response.status_code == 400


def test_transform_requires_service_credentials(client) -> None:
    response = client.post(
        "/api/transform_prompt",
        json={
            "session_id": "sess_missing_auth",
            "conversation_id": "conv_missing_auth",
            "user_id": "user_1",
            "raw_prompt": "Explain this concept simply",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid client identity."


def test_transform_rejects_invalid_service_credentials(client) -> None:
    response = client.post(
        "/api/transform_prompt",
        headers={
            "Authorization": "Bearer wrong-key",
            "X-Client-Id": "hermanprompt",
        },
        json={
            "session_id": "sess_wrong_auth",
            "conversation_id": "conv_wrong_auth",
            "user_id": "user_1",
            "raw_prompt": "Explain this concept simply",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid service credentials."


def test_transform_rejects_mismatched_conversation_ids(client) -> None:
    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_conv_mismatch",
            "conversation_id": "conv_top_level",
            "user_id": "user_1",
            "raw_prompt": "Explain this concept simply",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "conversation": {
                "conversation_id": "conv_nested",
                "requirements": {
                    "task": {"value": "Explain this concept simply", "status": "user_provided"}
                },
                "enforcement": {
                    "level": "moderate",
                    "status": "not_evaluated",
                    "missing_fields": [],
                    "last_evaluated_at": None,
                },
            },
        },
    )

    assert response.status_code == 400


def test_transform_returns_coaching_when_full_enforcement_missing_fields(client) -> None:
    _seed_final_profiles(client)
    _update_profile(client, "user_1", prompt_enforcement_level="full")

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_coaching",
            "conversation_id": "conv_coaching",
            "user_id": "user_1",
            "raw_prompt": "Explain how this works",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "coaching"
    assert body["transformed_prompt"] is None
    assert body["coaching_tip"].startswith("Coaching:")
    assert body["conversation"]["enforcement"]["status"] == "needs_coaching"
    assert "who" in body["conversation"]["enforcement"]["missing_fields"]


def test_transform_allows_demo_enforcement_override(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_override",
            "conversation_id": "conv_override",
            "user_id": "user_1",
            "raw_prompt": "Explain how this works",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "full",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "coaching"
    assert body["conversation"]["enforcement"]["level"] == "full"
    assert "policy:enforcement:override" in body["metadata"]["rules_applied"]


def test_transform_derives_context_and_output_from_compact_prompt(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_compact_prompt",
            "conversation_id": "conv_compact_prompt",
            "user_id": "user_1",
            "raw_prompt": "you are telling jokes at a kids birthday party, and just give me the joke in the chat.",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "full",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "transformed"
    assert body["conversation"]["requirements"]["who"]["status"] != "missing"
    assert body["conversation"]["requirements"]["context"]["status"] != "missing"
    assert body["conversation"]["requirements"]["output"]["status"] != "missing"


def test_transform_requires_more_than_generic_joke_prompt_under_full_enforcement(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_joke_prompt",
            "conversation_id": "conv_joke_prompt",
            "user_id": "user_1",
            "raw_prompt": "tell me a joke",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "full",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "coaching"
    assert body["transformed_prompt"] is None
    assert set(body["conversation"]["enforcement"]["missing_fields"]) == {"who", "context", "output"}


def test_transform_returns_warning_findings_when_compliance_check_enabled(client) -> None:
    _seed_final_profiles(client)
    _update_profile(
        client,
        "user_1",
        prompt_enforcement_level="none",
        compliance_check_enabled=True,
    )

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_compliance_warning",
            "conversation_id": "conv_compliance_warning",
            "user_id": "user_1",
            "raw_prompt": "Please provide financial advice for my retirement plan.",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "transformed"
    assert body["findings"][0]["type"] == "compliance"
    assert body["findings"][0]["severity"] == "medium"


def test_transform_blocks_when_pii_check_detects_high_risk_content(client) -> None:
    _seed_final_profiles(client)
    _update_profile(
        client,
        "user_1",
        prompt_enforcement_level="none",
        pii_check_enabled=True,
    )

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_pii_blocked",
            "conversation_id": "conv_pii_blocked",
            "user_id": "user_1",
            "raw_prompt": "Write an email for alice@example.com and bob@example.com about our offer.",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "blocked"
    assert body["transformed_prompt"] is None
    assert body["blocking_message"] is not None
    assert body["conversation"]["enforcement"]["status"] == "blocked"
    assert body["findings"][0]["type"] == "pii"
    assert body["findings"][0]["severity"] == "high"


def test_transform_logs_new_result_fields_when_request_logging_enabled(client) -> None:
    _seed_final_profiles(client)
    _update_profile(client, "user_1", prompt_enforcement_level="full")
    settings = get_settings()
    original_enable_request_logging = settings.enable_request_logging
    settings.enable_request_logging = True

    try:
        response = client.post(
            "/api/transform_prompt",
            headers=AUTH_HEADERS,
            json={
                "session_id": "sess_logging",
                "conversation_id": "conv_logging",
                "user_id": "user_1",
                "raw_prompt": "Explain how this works",
                "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            },
        )
        assert response.status_code == 200

        db = next(client.app.dependency_overrides[get_db]())
        try:
            log_row = db.query(PromptTransformRequest).filter_by(session_id="sess_logging").one()
            assert log_row.conversation_id == "conv_logging"
            assert log_row.result_type == "coaching"
            assert log_row.enforcement_level == "full"
            assert log_row.conversation_json["enforcement"]["status"] == "needs_coaching"
        finally:
            db.close()
    finally:
        settings.enable_request_logging = original_enable_request_logging
