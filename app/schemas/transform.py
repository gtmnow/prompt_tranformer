from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator, field_validator


class TargetLLM(BaseModel):
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=100)


RequirementStatus = Literal["present", "derived", "missing", "user_provided"]
EnforcementLevel = Literal["none", "low", "moderate", "full"]
EnforcementStatus = Literal["not_evaluated", "passes", "needs_coaching", "blocked"]
ResultType = Literal["transformed", "coaching", "blocked"]
FindingType = Literal["compliance", "pii"]
FindingSeverity = Literal["low", "medium", "high"]
GuideMeHelperKind = Literal[
    "contextual_step_examples",
    "refinement_options",
    "specificity_refinement",
    "answer_extraction",
]
AttachmentKind = Literal["document", "image"]


class ConversationRequirement(BaseModel):
    value: Optional[str] = Field(default=None)
    status: RequirementStatus
    heuristic_score: Optional[int] = Field(default=None, ge=0, le=100)
    llm_score: Optional[int] = Field(default=None, ge=0, le=100)
    max_score: Optional[int] = Field(default=None, ge=0, le=100)
    reason: Optional[str] = Field(default=None)
    improvement_hint: Optional[str] = Field(default=None)

    @field_validator("status")
    @classmethod
    def normalize_status(cls, value: str) -> str:
        if value == "user_provided":
            return "present"
        return value


class ConversationEnforcement(BaseModel):
    level: EnforcementLevel
    status: EnforcementStatus
    missing_fields: list[str] = Field(default_factory=list)
    last_evaluated_at: Optional[str] = Field(default=None)


class ConversationState(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=255)
    requirements: dict[str, ConversationRequirement]
    enforcement: ConversationEnforcement


class Finding(BaseModel):
    type: FindingType
    severity: FindingSeverity
    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1)


class PromptScoringSummary(BaseModel):
    scoring_version: str = Field(min_length=1, max_length=50)
    initial_score: int = Field(ge=0, le=100)
    final_score: int = Field(ge=0, le=100)
    initial_llm_score: Optional[int] = Field(default=None, ge=0, le=100)
    final_llm_score: Optional[int] = Field(default=None, ge=0, le=100)
    structural_score: int = Field(ge=0, le=100)


class ResolvedProfileResponse(BaseModel):
    user_id_hash: str = Field(min_length=1, max_length=255)
    summary_type: Optional[int] = Field(default=None, ge=1, le=9)
    profile_version: str = Field(min_length=1, max_length=255)
    persona_source: str = Field(min_length=1, max_length=100)
    prompt_enforcement_level: str = Field(min_length=1, max_length=20)
    compliance_check_enabled: bool
    pii_check_enabled: bool


class ConversationScoreResponse(BaseModel):
    conversation_id: str
    user_id_hash: str
    scoring_version: str = Field(min_length=1, max_length=50)
    initial_score: int = Field(ge=0, le=100)
    best_score: int = Field(ge=0, le=100)
    final_score: int = Field(ge=0, le=100)
    initial_llm_score: Optional[int] = Field(default=None, ge=0, le=100)
    best_llm_score: Optional[int] = Field(default=None, ge=0, le=100)
    final_llm_score: Optional[int] = Field(default=None, ge=0, le=100)
    structural_score: int = Field(ge=0, le=100)
    improvement_score: int
    best_improvement_score: int
    last_scored_at: str
    conversation: Optional[ConversationState] = None


class TransformPromptRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=255)
    conversation_id: str = Field(min_length=1, max_length=255)
    user_id_hash: str = Field(min_length=1, max_length=255)
    raw_prompt: str = Field(min_length=1)
    target_llm: TargetLLM
    conversation: Optional[ConversationState] = Field(default=None)
    summary_type: Optional[int] = Field(default=None)
    enforcement_level: Optional[EnforcementLevel] = Field(default=None)

    @field_validator("summary_type")
    @classmethod
    def validate_summary_type(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value
        if 1 <= value <= 9:
            return value
        raise ValueError("summary_type must be between 1 and 9")

    @model_validator(mode="after")
    def validate_conversation_id_match(self) -> "TransformPromptRequest":
        if self.conversation is not None and self.conversation.conversation_id != self.conversation_id:
            raise ValueError("conversation.conversation_id must match conversation_id")
        return self


class TransformMetadata(BaseModel):
    execution_owner: Literal["transformer"] = "transformer"
    persona_source: Literal["summary_override", "db_profile", "generic_default", "bypassed"]
    rules_applied: list[str]
    profile_version: str
    requested_provider: str
    requested_model: str
    resolved_provider: str
    resolved_model: str
    used_fallback_model: bool
    used_authoritative_tenant_llm: bool = False
    transformation_applied: bool = True
    bypass_reason: Optional[str] = None
    request_log_id: Optional[int] = Field(default=None, ge=1)
    retrieval_used: bool = False
    retrieval_scope_counts: dict[str, int] = Field(default_factory=dict)
    retrieval_document_count: int = 0
    retrieval_skipped_reason: Optional[str] = None
    reference_context_word_count: int = Field(default=0, ge=0)
    final_prompt_char_count: int = Field(default=0, ge=0)
    final_prompt_word_count: int = Field(default=0, ge=0)
    final_response_latency_ms: Optional[float] = Field(default=None, ge=0)


class TokenUsageWritePayload(BaseModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)
    reasoning_tokens: Optional[int] = Field(default=None, ge=0)
    cache_read_tokens: Optional[int] = Field(default=None, ge=0)
    cache_write_tokens: Optional[int] = Field(default=None, ge=0)
    raw_usage: Optional[dict] = None


class FinalResponseUsageRequest(BaseModel):
    request_log_id: int = Field(ge=1)
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=200)
    usage: TokenUsageWritePayload


class FinalResponseUsageResponse(BaseModel):
    request_log_id: int
    status: Literal["updated"]


class TransformPromptResponse(BaseModel):
    session_id: str
    conversation_id: str
    user_id_hash: str
    result_type: ResultType
    task_type: Optional[str] = None
    transformed_prompt: Optional[str] = None
    coaching_tip: Optional[str] = None
    blocking_message: Optional[str] = None
    conversation: Optional[ConversationState] = None
    findings: list[Finding] = Field(default_factory=list)
    scoring: Optional[PromptScoringSummary] = None
    metadata: TransformMetadata


class ConversationHistoryTurn(BaseModel):
    transformed_text: str = Field(min_length=1)
    assistant_text: str = Field(min_length=1)


class AttachmentReference(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    kind: AttachmentKind
    name: str = Field(min_length=1, max_length=255)
    media_type: Optional[str] = None
    provider_file_id: Optional[str] = None
    size_bytes: Optional[int] = Field(default=None, ge=0)


class GeneratedImagePayload(BaseModel):
    media_type: str = "image/png"
    base64_data: str = Field(min_length=1)


class ExecuteChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=255)
    conversation_id: str = Field(min_length=1, max_length=255)
    user_id_hash: str = Field(min_length=1, max_length=255)
    raw_prompt: str = Field(min_length=1)
    target_llm: TargetLLM
    conversation: Optional[ConversationState] = Field(default=None)
    conversation_history: list[ConversationHistoryTurn] = Field(default_factory=list)
    attachments: list[AttachmentReference] = Field(default_factory=list)
    summary_type: Optional[int] = Field(default=None, ge=1, le=9)
    enforcement_level: Optional[EnforcementLevel] = Field(default=None)
    transform_enabled: bool = True
    request_live_web_search: bool = False

    @model_validator(mode="after")
    def validate_conversation_id_match(self) -> "ExecuteChatRequest":
        if self.conversation is not None and self.conversation.conversation_id != self.conversation_id:
            raise ValueError("conversation.conversation_id must match conversation_id")
        return self


class ExecuteChatResponse(BaseModel):
    session_id: str
    conversation_id: str
    user_id_hash: str
    result_type: ResultType
    task_type: Optional[str] = None
    transformed_prompt: Optional[str] = None
    assistant_text: str
    assistant_images: list[GeneratedImagePayload] = Field(default_factory=list)
    coaching_tip: Optional[str] = None
    blocking_message: Optional[str] = None
    conversation: Optional[ConversationState] = None
    findings: list[Finding] = Field(default_factory=list)
    scoring: Optional[PromptScoringSummary] = None
    metadata: TransformMetadata


class GuideMeHelperRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=255)
    conversation_id: str = Field(min_length=1, max_length=255)
    user_id_hash: str = Field(min_length=1, max_length=255)
    target_llm: TargetLLM
    helper_kind: GuideMeHelperKind
    prompt: str = Field(min_length=1)
    max_output_tokens: int = Field(default=800, ge=1, le=4000)


class GuideMeHelperResponse(BaseModel):
    session_id: str
    conversation_id: str
    user_id_hash: str
    helper_kind: GuideMeHelperKind
    payload: dict


RagScopeType = Literal["tenant", "user"]
RagPolicySource = Literal["Default", "Service Tier", "Organization Override"]
RagDocumentStatus = Literal["pending", "processing", "ready", "failed", "disabled"]


class RagUsageSummary(BaseModel):
    document_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    ready_documents: int = Field(ge=0)
    processing_documents: int = Field(ge=0)
    failed_documents: int = Field(ge=0)
    disabled_documents: int = Field(ge=0)


class EffectiveRagQuota(BaseModel):
    scope: RagScopeType
    policy_source: RagPolicySource
    policy_key: str
    max_file_bytes: int = Field(ge=0)
    max_document_count: int = Field(ge=0)
    max_total_bytes: int = Field(ge=0)
    max_extracted_text_bytes: int = Field(ge=0)
    max_chunks_per_document: int = Field(ge=0)
    max_retrieved_chunks: int = Field(ge=0)
    max_retrieved_chunks_total: int = Field(ge=0)


class RagCollectionSummary(BaseModel):
    id: str
    scope_type: RagScopeType
    tenant_id: str
    user_id_hash: str | None = None
    name: str
    is_active: bool
    retrieval_enabled: bool
    max_results: int | None = None


class RagDocumentSummary(BaseModel):
    id: str
    collection_id: str
    scope_type: RagScopeType
    tenant_id: str
    user_id_hash: str | None = None
    filename: str
    media_type: str
    size_bytes: int = Field(ge=0)
    status: RagDocumentStatus
    status_message: str | None = None
    uploaded_by_admin_user_id: str | None = None
    uploaded_by_user_id_hash: str | None = None
    uploaded_at: str
    processed_at: str | None = None
    disabled_at: str | None = None


class RagLimitsResponse(BaseModel):
    limits: EffectiveRagQuota
    usage: RagUsageSummary


class RagDocumentListResponse(BaseModel):
    collection: RagCollectionSummary
    limits: EffectiveRagQuota
    usage: RagUsageSummary
    documents: list[RagDocumentSummary] = Field(default_factory=list)


class RagDocumentMutationResponse(BaseModel):
    collection: RagCollectionSummary
    limits: EffectiveRagQuota
    usage: RagUsageSummary
    document: RagDocumentSummary


class RagDocumentDeleteResponse(BaseModel):
    deleted_document_id: str
    collection: RagCollectionSummary
    limits: EffectiveRagQuota
    usage: RagUsageSummary


class RagCollectionUpdateRequest(BaseModel):
    retrieval_enabled: bool | None = None
    is_active: bool | None = None
    max_results: int | None = Field(default=None, ge=1, le=20)
