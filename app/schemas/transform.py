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
    persona_source: Literal["summary_override", "db_profile", "generic_default"]
    rules_applied: list[str]
    profile_version: str
    requested_model: str
    resolved_model: str
    used_fallback_model: bool


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
