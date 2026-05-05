from app.services.llm_adapters.anthropic import AnthropicAdapter
from app.services.llm_types import TransformerLlmContentPart, TransformerLlmMessage, TransformerLlmRequest


def test_anthropic_adapter_builds_document_and_image_content_blocks() -> None:
    adapter = AnthropicAdapter()
    request = TransformerLlmRequest(
        purpose="final_response",
        provider="anthropic",
        model="claude-sonnet-4",
        base_url="https://api.anthropic.com/v1",
        api_key="test-key",
        messages=[
            TransformerLlmMessage(
                role="user",
                content=[
                    TransformerLlmContentPart(type="text", text="Summarize and inspect these uploads."),
                    TransformerLlmContentPart(type="document_file", file_id="file_doc"),
                    TransformerLlmContentPart(type="image_file", file_id="file_img"),
                ],
            )
        ],
    )

    payload_messages = adapter._build_messages(request)  # type: ignore[attr-defined]

    assert payload_messages[0]["content"][1] == {
        "type": "document",
        "source": {"type": "file", "file_id": "file_doc"},
    }
    assert payload_messages[0]["content"][2] == {
        "type": "image",
        "source": {"type": "file", "file_id": "file_img"},
    }

