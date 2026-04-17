from app.db.seed import PROFILE_ROWS
from app.models.profile import FinalProfile
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


def test_transform_uses_db_profile(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_123",
            "user_id": "user_1",
            "raw_prompt": "Explain this concept simply",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_type"] == "explanation"
    assert body["metadata"]["persona_source"] == "db_profile"
    assert "Start with the direct answer" in body["transformed_prompt"]


def test_transform_uses_summary_override(client) -> None:
    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_456",
            "user_id": "user_missing",
            "raw_prompt": "Summarize this article",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "summary_type": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_type"] == "summarization"
    assert body["metadata"]["persona_source"] == "summary_override"


def test_transform_falls_back_to_generic_default(client) -> None:
    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_789",
            "user_id": "user_missing",
            "raw_prompt": "What should I do next?",
            "target_llm": {"provider": "openai", "model": "unknown-model"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_type"] == "unknown"
    assert body["metadata"]["persona_source"] == "generic_default"
    assert body["metadata"]["used_fallback_model"] is True


def test_invalid_summary_type_returns_400(client) -> None:
    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_999",
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
            "user_id": "user_1",
            "raw_prompt": "Explain this concept simply",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid service credentials."
