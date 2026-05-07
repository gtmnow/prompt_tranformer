from unittest.mock import patch

from app.schemas.transform import ExecuteChatResponse, GuideMeHelperResponse, TransformMetadata
from app.services.guide_me_generation import GuideMeGenerationError


AUTH_HEADERS = {
    "Authorization": "Bearer test-transformer-key",
    "X-Client-Id": "hermanprompt",
}


def test_execute_chat_route_returns_transformer_owned_response(client) -> None:
    with patch("app.api.routes.TransformerEngine.execute_chat") as execute_chat:
        execute_chat.return_value = ExecuteChatResponse(
            session_id="sess_1",
            conversation_id="conv_1",
            user_id_hash="user_1",
            result_type="transformed",
            task_type="analysis",
            transformed_prompt="Task: Explain the answer.",
            assistant_text="Here is the answer.",
            assistant_images=[],
            conversation=None,
            findings=[],
            scoring=None,
            metadata=TransformMetadata(
                execution_owner="transformer",
                persona_source="db_profile",
                rules_applied=[],
                profile_version="v1",
                requested_provider="openai",
                requested_model="gpt-5",
                resolved_provider="openai",
                resolved_model="gpt-5",
                used_fallback_model=False,
                used_authoritative_tenant_llm=False,
                transformation_applied=True,
                bypass_reason=None,
                request_log_id=1,
            ),
        )

        response = client.post(
            "/api/chat/execute",
            headers=AUTH_HEADERS,
            json={
                "session_id": "sess_1",
                "conversation_id": "conv_1",
                "user_id_hash": "user_1",
                "raw_prompt": "Explain the answer.",
                "target_llm": {"provider": "openai", "model": "gpt-5"},
                "conversation_history": [],
                "attachments": [],
                "transform_enabled": True,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_text"] == "Here is the answer."
    assert body["metadata"]["execution_owner"] == "transformer"


def test_guide_me_helper_route_returns_structured_payload(client) -> None:
    with patch("app.api.routes.TransformerEngine.generate_guide_me_helper") as generate_helper:
        generate_helper.return_value = GuideMeHelperResponse(
            session_id="sess_guide",
            conversation_id="conv_guide",
            user_id_hash="user_1",
            helper_kind="answer_extraction",
            payload={"task": "Reduce unqualified applicants by 30%."},
        )

        response = client.post(
            "/api/guide_me/generate",
            headers=AUTH_HEADERS,
            json={
                "session_id": "sess_guide",
                "conversation_id": "conv_guide",
                "user_id_hash": "user_1",
                "target_llm": {"provider": "openai", "model": "gpt-5"},
                "helper_kind": "answer_extraction",
                "prompt": "Return strict JSON with optional keys who, task, context, output, instructions.",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["helper_kind"] == "answer_extraction"
    assert body["payload"]["task"] == "Reduce unqualified applicants by 30%."


def test_guide_me_helper_route_preserves_upstream_status_code(client) -> None:
    with patch("app.api.routes.TransformerEngine.generate_guide_me_helper") as generate_helper:
        generate_helper.side_effect = GuideMeGenerationError("Rate limit exceeded", status_code=429)
        response = client.post(
            "/api/guide_me/generate",
            headers=AUTH_HEADERS,
            json={
                "session_id": "sess_guide",
                "conversation_id": "conv_guide",
                "user_id_hash": "user_1",
                "target_llm": {"provider": "openai", "model": "gpt-5"},
                "helper_kind": "answer_extraction",
                "prompt": "Return strict JSON with optional keys who, task, context, output, instructions.",
            },
        )
    assert response.status_code == 429
