from app.db.seed import PROFILE_ROWS
from app.models.profile import FinalProfile
from app.models.prompt_score import ConversationPromptScore
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
            "user_id_hash": "user_1",
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
            "user_id_hash": "user_missing",
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
            "user_id_hash": "user_missing",
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
            "user_id_hash": "user_1",
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
            "user_id_hash": "user_1",
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
            "user_id_hash": "user_1",
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
            "user_id_hash": "user_1",
            "raw_prompt": "Explain this concept simply",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "conversation": {
                    "conversation_id": "conv_nested",
                "requirements": {
                    "task": {"value": "Explain this concept simply", "status": "present"}
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
            "user_id_hash": "user_1",
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
            "user_id_hash": "user_1",
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
            "user_id_hash": "user_1",
            "raw_prompt": "you are telling jokes at a kids birthday party, and just give me the joke in the chat.",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "full",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "coaching"
    assert body["conversation"]["requirements"]["who"]["status"] == "present"
    assert body["conversation"]["requirements"]["context"]["status"] == "present"
    assert body["conversation"]["requirements"]["output"]["status"] == "present"
    assert body["conversation"]["enforcement"]["missing_fields"] == ["labeled_structure"]


def test_transform_requires_more_than_generic_joke_prompt_under_full_enforcement(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_joke_prompt",
            "conversation_id": "conv_joke_prompt",
            "user_id_hash": "user_1",
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


def test_transform_accepts_unlabeled_natural_language_prompt_under_moderate_enforcement(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_moderate_natural_language",
            "conversation_id": "conv_moderate_natural_language",
            "user_id_hash": "user_1",
            "raw_prompt": (
                "You are a senior Python software engineer. "
                "Explain how to design a REST API rate-limiting system for a SaaS application. "
                "I am preparing for a backend system design interview and need a clear answer. "
                "Provide the answer in the chat as a concise structured response with an overview, "
                "core components, request flow, tradeoffs, and one example."
            ),
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "moderate",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "transformed"
    assert body["scoring"]["structural_score"] == 100
    assert body["conversation"]["requirements"]["who"]["status"] == "present"
    assert body["conversation"]["requirements"]["task"]["status"] == "present"
    assert body["conversation"]["requirements"]["context"]["status"] == "present"
    assert body["conversation"]["requirements"]["output"]["status"] == "present"
    for field_name in ("who", "task", "context", "output"):
        requirement = body["conversation"]["requirements"][field_name]
        assert set(requirement.keys()) >= {
            "value",
            "status",
            "heuristic_score",
            "llm_score",
            "max_score",
            "reason",
            "improvement_hint",
        }
        assert requirement["heuristic_score"] is not None
        assert requirement["max_score"] == 25
        assert requirement["reason"]


def test_transform_allows_derived_fields_under_moderate_enforcement_with_coaching(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_moderate_derived_ok",
            "conversation_id": "conv_moderate_derived_ok",
            "user_id_hash": "user_1",
            "raw_prompt": "You are a comedian at a kids birthday party tell me a joke",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "moderate",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "transformed"
    assert body["coaching_tip"] is not None
    assert body["conversation"]["enforcement"]["status"] == "passes"
    assert body["conversation"]["requirements"]["who"]["status"] in {"present", "derived"}
    assert body["conversation"]["requirements"]["task"]["status"] in {"present", "derived"}
    assert body["conversation"]["requirements"]["context"]["status"] in {"present", "derived"}
    assert body["conversation"]["requirements"]["output"]["status"] == "derived"


def test_transform_requires_labels_under_full_enforcement_even_when_elements_are_present(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_full_requires_labels",
            "conversation_id": "conv_full_requires_labels",
            "user_id_hash": "user_1",
            "raw_prompt": (
                "You are a senior Python software engineer. "
                "Explain how to design a REST API rate-limiting system for a SaaS application. "
                "I am preparing for a backend system design interview and need a clear answer. "
                "Provide the answer in the chat as a concise structured response with an overview, "
                "core components, request flow, tradeoffs, and one example."
            ),
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "full",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "coaching"
    assert body["coaching_tip"] == "Coaching: for full guidance, format the prompt with Who, Task, Context, and Output labels."
    assert body["conversation"]["enforcement"]["missing_fields"] == ["labeled_structure"]
    assert body["scoring"]["structural_score"] == 100


def test_transform_accepts_labeled_prompt_under_full_enforcement(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_full_labeled",
            "conversation_id": "conv_full_labeled",
            "user_id_hash": "user_1",
            "raw_prompt": (
                "Who: Senior Python software engineer\n"
                "Task: Explain how to design a REST API rate-limiting system for a SaaS application.\n"
                "Context: I am preparing for a backend system design interview and need a clear answer.\n"
                "Output: Provide the answer in the chat as a concise structured response with an overview, core components, request flow, tradeoffs, and one example."
            ),
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "full",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "transformed"
    assert body["scoring"]["structural_score"] == 100


def test_transform_returns_light_coaching_under_low_enforcement_without_blocking(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_low_guidance",
            "conversation_id": "conv_low_guidance",
            "user_id_hash": "user_1",
            "raw_prompt": "tell me a joke",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "low",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_type"] == "transformed"
    assert body["coaching_tip"] == "Coaching: include the role you want the AI to play, the setting or intended use, and how you want the answer delivered next time for a stronger prompt."
    assert body["scoring"]["structural_score"] == 30


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
            "user_id_hash": "user_1",
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
            "user_id_hash": "user_1",
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
                "user_id_hash": "user_1",
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


def test_transform_creates_conversation_score_rollup(client) -> None:
    _seed_final_profiles(client)
    _update_profile(client, "user_1", prompt_enforcement_level="full")

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_score_rollup",
            "conversation_id": "conv_score_rollup",
            "user_id_hash": "user_1",
            "raw_prompt": "Explain how this works",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
        },
    )

    assert response.status_code == 200

    db = next(client.app.dependency_overrides[get_db]())
    try:
        score_row = db.query(ConversationPromptScore).filter_by(conversation_id="conv_score_rollup").one()
        assert score_row.user_id_hash == "user_1"
        assert score_row.initial_score == 30
        assert score_row.final_score == 30
        assert score_row.best_score == 30
        assert score_row.improvement_score == 0
        assert score_row.reached_policy_complete is False
        assert score_row.coaching_turn_count == 1
        assert score_row.transformed_turn_count == 0
    finally:
        db.close()


def test_transform_updates_conversation_score_when_prompt_improves(client) -> None:
    _seed_final_profiles(client)

    first_response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_score_improve_1",
            "conversation_id": "conv_score_improve",
            "user_id_hash": "user_1",
            "raw_prompt": "tell me a joke",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "full",
        },
    )
    assert first_response.status_code == 200
    assert first_response.json()["result_type"] == "coaching"

    second_response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_score_improve_2",
            "conversation_id": "conv_score_improve",
            "user_id_hash": "user_1",
            "raw_prompt": "you are telling jokes at a kids birthday party, and just give me the joke in the chat.",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "full",
        },
    )
    assert second_response.status_code == 200
    assert second_response.json()["result_type"] == "coaching"

    db = next(client.app.dependency_overrides[get_db]())
    try:
        score_row = db.query(ConversationPromptScore).filter_by(conversation_id="conv_score_improve").one()
        assert score_row.initial_score == 30
        assert score_row.final_score == 100
        assert score_row.best_score == 100
        assert score_row.improvement_score == 70
        assert score_row.best_improvement_score == 70
        assert score_row.reached_policy_complete is False
        assert score_row.coaching_turn_count == 2
        assert score_row.transformed_turn_count == 0
    finally:
        db.close()


def test_conversation_memory_does_not_inflate_current_turn_score(client) -> None:
    _seed_final_profiles(client)

    first_response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_memory_score_1",
            "conversation_id": "conv_memory_score",
            "user_id_hash": "user_1",
            "raw_prompt": "tell me a joke",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "moderate",
        },
    )
    assert first_response.status_code == 200
    first_body = first_response.json()

    second_response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_memory_score_2",
            "conversation_id": "conv_memory_score",
            "user_id_hash": "user_1",
            "raw_prompt": "you are a comedian tell me a joke the I can use at a kids birthday party",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "conversation": first_body["conversation"],
            "enforcement_level": "moderate",
        },
    )

    assert second_response.status_code == 200
    second_body = second_response.json()
    assert second_body["result_type"] == "transformed"
    assert second_body["scoring"]["initial_score"] == 30
    assert second_body["scoring"]["final_score"] == 80
    assert second_body["scoring"]["structural_score"] == 80

    db = next(client.app.dependency_overrides[get_db]())
    try:
        score_row = db.query(ConversationPromptScore).filter_by(conversation_id="conv_memory_score").one()
        assert score_row.initial_score == 30
        assert score_row.final_score == 80
        assert score_row.best_score == 80
    finally:
        db.close()


def test_get_conversation_score_returns_rollup_from_database(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_score_fetch",
            "conversation_id": "conv_score_fetch",
            "user_id_hash": "user_1",
            "raw_prompt": "tell me a joke",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "low",
        },
    )
    assert response.status_code == 200

    score_response = client.get(
        "/api/conversation_scores/conv_score_fetch",
        headers=AUTH_HEADERS,
        params={"user_id_hash": "user_1"},
    )

    assert score_response.status_code == 200
    body = score_response.json()
    assert body["conversation_id"] == "conv_score_fetch"
    assert body["user_id_hash"] == "user_1"
    assert body["scoring_version"] == "v4"
    assert body["initial_score"] == 30
    assert body["final_score"] == 30
    assert body["initial_llm_score"] in {None, 30}
    assert body["final_llm_score"] in {None, 30}
    assert body["structural_score"] == 30
    assert body["best_score"] == 30
    assert body["improvement_score"] == 0
    assert body["conversation"]["conversation_id"] == "conv_score_fetch"
    assert body["conversation"]["requirements"]["task"]["heuristic_score"] == 25
    assert body["conversation"]["requirements"]["task"]["max_score"] == 25
    assert body["conversation"]["requirements"]["task"]["reason"]
    assert body["conversation"]["requirements"]["who"]["improvement_hint"] is not None


def test_score_rollup_persists_hybrid_score_details(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_hybrid_score_details",
            "conversation_id": "conv_hybrid_score_details",
            "user_id_hash": "user_1",
            "raw_prompt": (
                "Who: Senior Python software engineer\n"
                "Task: Explain how to design a REST API rate-limiting system for a SaaS application.\n"
                "Context: I am preparing for a backend system design interview.\n"
                "Output: Provide the answer in the chat with an overview, tradeoffs, and one example."
            ),
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "moderate",
        },
    )

    assert response.status_code == 200

    db = next(client.app.dependency_overrides[get_db]())
    try:
        score_row = db.query(ConversationPromptScore).filter_by(
            conversation_id="conv_hybrid_score_details"
        ).one()
        assert score_row.scoring_version == "v4"
        assert score_row.score_details_json["heuristic_score"] >= 0
        assert score_row.score_details_json["final_score"] == score_row.final_score
        assert score_row.score_details_json["scoring_method"] in {"heuristic_only_v1", "hybrid_llm_v2"}
        assert "heuristic_field_statuses" in score_row.score_details_json
        assert "llm_field_statuses" in score_row.score_details_json
        assert set(score_row.score_details_json["requirements"]["task"].keys()) >= {
            "value",
            "status",
            "heuristic_score",
            "llm_score",
            "max_score",
            "reason",
            "improvement_hint",
        }
        assert score_row.final_llm_score == score_row.score_details_json["llm_score"]
    finally:
        db.close()


def test_numeric_llm_dimension_scores_are_used_for_llm_total(client, monkeypatch) -> None:
    _seed_final_profiles(client)

    from app.services.structure_evaluator import StructureEvaluationService

    monkeypatch.setattr(StructureEvaluationService, "is_enabled", lambda self: True)
    monkeypatch.setattr(
        StructureEvaluationService,
        "evaluate",
        lambda self, **kwargs: {
            "who": {"value": "comedian", "status": "present", "score": 21},
            "task": {"value": "tell me a joke", "status": "present", "score": 24},
            "context": {"value": "for a kids birthday party", "status": "derived", "score": 16},
            "output": {"value": "in the chat", "status": "derived", "score": 11},
            "coaching_tip": "Coaching: add a little more output detail next time.",
        },
    )

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_numeric_llm",
            "conversation_id": "conv_numeric_llm",
            "user_id_hash": "user_1",
            "raw_prompt": "you are a comedian tell me a joke for a kids birthday party",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "moderate",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scoring"]["final_llm_score"] == 72

    db = next(client.app.dependency_overrides[get_db]())
    try:
        score_row = db.query(ConversationPromptScore).filter_by(conversation_id="conv_numeric_llm").one()
        assert score_row.final_llm_score == 72
        assert score_row.score_details_json["llm_field_points"] == {
            "who": 21,
            "task": 24,
            "context": 16,
            "output": 11,
        }
        assert score_row.score_details_json["llm_score"] == 72
    finally:
        db.close()


def test_llm_score_uses_current_prompt_not_prior_conversation_memory(client, monkeypatch) -> None:
    _seed_final_profiles(client)

    from app.services.structure_evaluator import StructureEvaluationService

    calls: list[dict] = []

    def fake_evaluate(self, **kwargs):
        calls.append(kwargs)
        prompt = kwargs["raw_prompt"]
        if prompt == "tell me a joke":
            return {
                "who": {"value": None, "status": "missing", "score": 0},
                "task": {"value": "tell me a joke", "status": "present", "score": 25},
                "context": {"value": None, "status": "missing", "score": 0},
                "output": {"value": "joke in chat", "status": "derived", "score": 5},
                "coaching_tip": "Coaching: add audience or style.",
            }
        return {
            "who": {"value": "comedian", "status": "present", "score": 24},
            "task": {"value": "tell me a joke", "status": "present", "score": 25},
            "context": {"value": None, "status": "missing", "score": 0},
            "output": {"value": "joke in chat", "status": "derived", "score": 8},
            "coaching_tip": "Coaching: add context.",
        }

    monkeypatch.setattr(StructureEvaluationService, "is_enabled", lambda self: True)
    monkeypatch.setattr(StructureEvaluationService, "evaluate", fake_evaluate)

    first_response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_llm_current_1",
            "conversation_id": "conv_llm_current",
            "user_id_hash": "user_1",
            "raw_prompt": "tell me a joke",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "moderate",
        },
    )
    assert first_response.status_code == 200
    first_body = first_response.json()

    second_response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_llm_current_2",
            "conversation_id": "conv_llm_current",
            "user_id_hash": "user_1",
            "raw_prompt": "you are a comedian tell me a joke",
            "conversation": first_body["conversation"],
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "moderate",
        },
    )
    assert second_response.status_code == 200
    second_body = second_response.json()
    assert second_body["scoring"]["final_llm_score"] == 57
    assert "existing_requirements" not in calls[0]
    assert "existing_requirements" not in calls[1]


def test_scoring_example_task_and_context_only_scores_55(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_score_example_75",
            "conversation_id": "conv_score_example_75",
            "user_id_hash": "user_1",
            "raw_prompt": (
                "Explain how to design a REST API rate-limiting system for a SaaS application.\n\n"
                "I am preparing for a backend system design interview and need an answer that helps me "
                "understand the architecture, tradeoffs, and implementation choices clearly."
            ),
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "moderate",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scoring"]["structural_score"] == 55


def test_scoring_example_with_role_scores_80(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_score_example_100",
            "conversation_id": "conv_score_example_100",
            "user_id_hash": "user_1",
            "raw_prompt": (
                "You are a Senior Python software engineer\n\n"
                "Explain how to design a REST API rate-limiting system for a SaaS application.\n\n"
                "I am preparing for a backend system design interview and need an answer that helps me "
                "understand the architecture, tradeoffs, and implementation choices clearly."
            ),
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "moderate",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scoring"]["structural_score"] == 80


def test_scoring_example_task_only_scores_30(client) -> None:
    _seed_final_profiles(client)

    response = client.post(
        "/api/transform_prompt",
        headers=AUTH_HEADERS,
        json={
            "session_id": "sess_score_example_30",
            "conversation_id": "conv_score_example_30",
            "user_id_hash": "user_1",
            "raw_prompt": "Explain how to design a REST API rate-limiting system for a SaaS application.",
            "target_llm": {"provider": "openai", "model": "gpt-4.1"},
            "enforcement_level": "low",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scoring"]["structural_score"] == 30
