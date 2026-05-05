from app.schemas.transform import AttachmentReference, ConversationHistoryTurn
from app.services.llm_request_factory import LlmRequestFactory
from app.services.runtime_llm import RuntimeLlmConfig


def _runtime_config(provider: str = "openai", model: str = "gpt-4.1") -> RuntimeLlmConfig:
    return RuntimeLlmConfig(
        tenant_id="tenant_1",
        user_id_hash="user_1",
        provider=provider,
        model=model,
        endpoint_url="https://api.example.com/v1",
        api_key="test-key",
        transformation_enabled=True,
        scoring_enabled=True,
        credential_status="valid",
        source_kind="customer_managed",
    )


def test_build_final_response_request_includes_history_attachments_and_tools() -> None:
    factory = LlmRequestFactory()

    request = factory.build_final_response_request(
        runtime_config=_runtime_config(),
        resolved_model="gpt-4.1",
        transformed_prompt="Create image options from this reference.",
        conversation_history=[
            ConversationHistoryTurn(
                transformed_text="Task: describe the visual.",
                assistant_text="The visual is minimalist.",
            )
        ],
        attachments=[
            AttachmentReference(id="img_1", kind="image", name="image.png", provider_file_id="file_img"),
            AttachmentReference(id="doc_1", kind="document", name="brief.pdf", provider_file_id="file_doc"),
        ],
    )

    assert request.purpose == "final_response"
    assert request.messages[-1].content[0].text == "Create image options from this reference."
    assert request.messages[-1].content[1].file_id == "file_img"
    assert request.messages[-1].content[1].type == "image_file"
    assert request.messages[-1].content[2].file_id == "file_doc"
    assert request.messages[-1].content[2].type == "document_file"
    assert [tool.type for tool in request.tools] == ["code_interpreter", "image_generation"]


def test_build_final_response_request_skips_document_tool_for_anthropic() -> None:
    factory = LlmRequestFactory()

    request = factory.build_final_response_request(
        runtime_config=_runtime_config(provider="anthropic", model="claude-sonnet-4"),
        resolved_model="claude-sonnet-4",
        transformed_prompt="Summarize this document and inspect the image.",
        conversation_history=[],
        attachments=[
            AttachmentReference(id="img_1", kind="image", name="image.png", provider_file_id="file_img"),
            AttachmentReference(id="doc_1", kind="document", name="brief.pdf", provider_file_id="file_doc"),
        ],
    )

    assert request.tools == []


def test_build_structure_evaluator_request_is_json_focused() -> None:
    factory = LlmRequestFactory()

    request = factory.build_structure_evaluator_request(
        runtime_config=_runtime_config(),
        system_prompt="Return JSON only.",
        raw_prompt="Explain rate limiting.",
        enforcement_level="full",
        timeout_seconds=12.5,
    )

    assert request.purpose == "structure_evaluator"
    assert request.expected_output == "json"
    assert request.timeout_seconds == 12.5
    assert request.messages[0].role == "system"
