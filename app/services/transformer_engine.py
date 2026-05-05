from __future__ import annotations

import logging
import time

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import configure_application_logging
from app.schemas.transform import (
    ExecuteChatRequest,
    ExecuteChatResponse,
    GuideMeHelperRequest,
    GuideMeHelperResponse,
    TransformMetadata,
    TransformPromptRequest,
    TransformPromptResponse,
)
from app.services.compliance_checks import ComplianceCheckService
from app.services.final_response_service import FinalResponseService
from app.services.guide_me_generation import GuideMeGenerationService
from app.services.llm_policy import LLMPolicyService
from app.services.pii_checks import PIICheckService
from app.services.profile_resolver import ProfileResolver
from app.services.prompt_requirements import PromptRequirementService
from app.services.prompt_scoring import PromptScoringService
from app.services.rag_prompt_assembly_service import RagPromptAssemblyService
from app.services.rag_retrieval_service import RagRetrievalService
from app.services.request_logger import RequestLogger
from app.services.runtime_llm import RuntimeLlmConfigError, RuntimeLlmResolver
from app.services.task_inference import TaskInferenceService
from app.services.token_usage import build_usage_entry, merge_usage, normalize_usage


logger = logging.getLogger("prompt_transformer.transformer_engine")

TASK_INSTRUCTION_DEFAULTS = {
    "summarization": "Summarize the content according to the guidance below.",
    "explanation": "Explain the topic according to the guidance below.",
    "writing": "Produce polished writing according to the guidance below.",
    "planning": "Create a practical plan according to the guidance below.",
    "analysis": "Analyze the material according to the guidance below.",
    "recommendation": "Recommend the best option according to the guidance below.",
    "decision_support": "Provide decision support according to the guidance below.",
    "unknown": "Respond to the user's request according to the guidance below.",
}


class TransformerEngine:
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.settings = get_settings()
        configure_application_logging(self.settings.log_level)
        self.profile_resolver = ProfileResolver(db_session)
        self.task_inference = TaskInferenceService()
        self.llm_policy = LLMPolicyService()
        self.prompt_requirements = PromptRequirementService()
        self.prompt_scoring = PromptScoringService(db_session)
        self.compliance_checks = ComplianceCheckService()
        self.pii_checks = PIICheckService()
        self.request_logger = RequestLogger(db_session)
        self.runtime_llm = RuntimeLlmResolver(db_session)
        self.final_response_service = FinalResponseService()
        self.guide_me_generation = GuideMeGenerationService()
        self.rag_retrieval = RagRetrievalService(db_session)
        self.rag_prompt_assembly = RagPromptAssemblyService()

    def transform(self, payload: TransformPromptRequest) -> TransformPromptResponse:
        started_at = time.perf_counter()
        timings_ms: dict[str, float] = {}
        task_type = "unknown"
        result_type = "error"
        persona_source = "unknown"
        request_log_id: int | None = None

        try:
            runtime_llm = self.runtime_llm.resolve(payload.user_id_hash)
            step_started_at = time.perf_counter()
            persona = self.profile_resolver.resolve(payload.user_id_hash, payload.summary_type)
            timings_ms["profile_resolve"] = (time.perf_counter() - step_started_at) * 1000
            persona_source = persona.source

            effective_enforcement_level = payload.enforcement_level or persona.prompt_enforcement_level

            step_started_at = time.perf_counter()
            task_type, task_rules = self.task_inference.infer(payload.raw_prompt)
            timings_ms["task_inference"] = (time.perf_counter() - step_started_at) * 1000

            step_started_at = time.perf_counter()
            policy = self.llm_policy.resolve(
                provider=runtime_llm.provider,
                model=runtime_llm.model,
            )
            timings_ms["policy_resolve"] = (time.perf_counter() - step_started_at) * 1000

            step_started_at = time.perf_counter()
            conversation, enforcement_rules, coaching_tip, requirement_trace, evaluator_usage_entry = self.prompt_requirements.evaluate(
                conversation_id=payload.conversation_id,
                raw_prompt=payload.raw_prompt,
                conversation=payload.conversation,
                enforcement_level=effective_enforcement_level,
                runtime_config=runtime_llm if runtime_llm.scoring_enabled else None,
            )
            timings_ms["requirements_eval"] = (time.perf_counter() - step_started_at) * 1000

            step_started_at = time.perf_counter()
            findings = []
            if persona.compliance_check_enabled:
                findings.extend(self.compliance_checks.evaluate(payload.raw_prompt))
                enforcement_rules.append("check:compliance:enabled")
            if persona.pii_check_enabled:
                findings.extend(self.pii_checks.evaluate(payload.raw_prompt))
                enforcement_rules.append("check:pii:enabled")
            if payload.enforcement_level is not None:
                enforcement_rules.append("policy:enforcement:override")
            timings_ms["findings_eval"] = (time.perf_counter() - step_started_at) * 1000

            blocking_findings = [finding for finding in findings if finding.severity == "high"]
            transformed_prompt = None
            result_type = "transformed"
            blocking_message = None

            step_started_at = time.perf_counter()
            if blocking_findings:
                result_type = "blocked"
                conversation.enforcement.status = "blocked"
                blocking_message = blocking_findings[0].message
                persona_rules = []
                model_rules = []
            elif conversation.enforcement.status == "needs_coaching":
                result_type = "coaching"
                persona_rules = []
                model_rules = []
            else:
                transformed_prompt, persona_rules, model_rules = self._build_prompt(
                    raw_prompt=payload.raw_prompt,
                    task_type=task_type,
                    persona=persona.values,
                    model_policy=policy.policy,
                )
            timings_ms["prompt_build"] = (time.perf_counter() - step_started_at) * 1000

            rules_applied = task_rules + enforcement_rules + persona_rules + model_rules

            metadata = TransformMetadata(
                execution_owner="transformer",
                persona_source=persona.source,
                rules_applied=rules_applied,
                profile_version=persona.profile_version,
                requested_provider=payload.target_llm.provider,
                requested_model=payload.target_llm.model,
                resolved_provider=runtime_llm.provider,
                resolved_model=policy.resolved_model,
                used_fallback_model=policy.used_fallback_model,
                used_authoritative_tenant_llm=(
                    payload.target_llm.provider != runtime_llm.provider
                    or payload.target_llm.model != runtime_llm.model
                ),
                transformation_applied=result_type == "transformed",
                bypass_reason=None,
                request_log_id=None,
            )

            step_started_at = time.perf_counter()
            score_result = self.prompt_scoring.calculate(
                conversation=conversation,
                result_type=result_type,
                requirement_trace=requirement_trace,
            )
            score_summary = self.prompt_scoring.upsert_conversation_score(
                conversation=conversation,
                user_id_hash=payload.user_id_hash,
                task_type=task_type,
                result_type=result_type,
                score_result=score_result,
            )
            conversation = self.prompt_scoring.enrich_conversation(
                conversation=conversation,
                score_result=score_result,
            )
            timings_ms["scoring_dispatch"] = (time.perf_counter() - step_started_at) * 1000

            step_started_at = time.perf_counter()
            request_row = self.request_logger.log(
                {
                    "session_id": payload.session_id,
                    "conversation_id": payload.conversation_id,
                    "user_id_hash": payload.user_id_hash,
                    "raw_prompt": payload.raw_prompt,
                    "transformed_prompt": transformed_prompt,
                    "task_type": task_type,
                    "result_type": result_type,
                    "coaching_tip": coaching_tip,
                    "blocking_message": blocking_message,
                    "target_provider": runtime_llm.provider,
                    "target_model": policy.resolved_model,
                    "persona_source": persona.source,
                    "used_fallback_model": policy.used_fallback_model,
                    "enforcement_level": effective_enforcement_level,
                    "compliance_check_enabled": persona.compliance_check_enabled,
                    "pii_check_enabled": persona.pii_check_enabled,
                    "conversation_json": conversation.model_dump(),
                    "findings_json": [finding.model_dump() for finding in findings],
                    "metadata_json": metadata.model_dump(),
                    "token_usage_json": merge_usage(None, evaluator_usage_entry),
                }
            )
            request_log_id = request_row.id
            metadata.request_log_id = request_log_id
            request_row.metadata_json = metadata.model_dump()
            self.db_session.add(request_row)
            self.db_session.commit()
            timings_ms["request_log"] = (time.perf_counter() - step_started_at) * 1000

            return TransformPromptResponse(
                session_id=payload.session_id,
                conversation_id=payload.conversation_id,
                user_id_hash=payload.user_id_hash,
                result_type=result_type,
                task_type=task_type,
                transformed_prompt=transformed_prompt,
                coaching_tip=coaching_tip,
                blocking_message=blocking_message,
                conversation=conversation,
                findings=findings,
                scoring=score_summary,
                metadata=metadata,
            )
        except RuntimeLlmConfigError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            self._emit_timing_log(
                payload=payload,
                task_type=task_type,
                result_type=result_type,
                persona_source=persona_source,
                timings_ms=timings_ms,
                total_ms=(time.perf_counter() - started_at) * 1000,
            )

    def _emit_timing_log(
        self,
        payload: TransformPromptRequest,
        task_type: str,
        result_type: str,
        persona_source: str,
        timings_ms: dict[str, float],
        total_ms: float,
    ) -> None:
        if not self.settings.enable_transform_timing_logs:
            return

        timing_parts = [f"{name}_ms={value:.1f}" for name, value in timings_ms.items()]
        logger.info(
            "transform_timing session_id=%s conversation_id=%s user_id_hash=%s task_type=%s result_type=%s persona_source=%s total_ms=%.1f %s",
            payload.session_id,
            payload.conversation_id,
            payload.user_id_hash,
            task_type,
            result_type,
            persona_source,
            total_ms,
            " ".join(timing_parts),
        )

    def execute_chat(self, payload: ExecuteChatRequest) -> ExecuteChatResponse:
        try:
            runtime_llm = self.runtime_llm.resolve(payload.user_id_hash)
        except RuntimeLlmConfigError as exc:
            raise ValueError(str(exc)) from exc

        if not payload.transform_enabled or not runtime_llm.transformation_enabled:
            bypass_reason = (
                "tenant_transformation_disabled"
                if not runtime_llm.transformation_enabled
                else "prompt_transform_disabled"
            )
            bypass_metadata = TransformMetadata(
                execution_owner="transformer",
                persona_source="bypassed",
                rules_applied=[],
                profile_version="bypassed",
                requested_provider=payload.target_llm.provider,
                requested_model=payload.target_llm.model,
                resolved_provider=runtime_llm.provider,
                resolved_model=runtime_llm.model,
                used_fallback_model=False,
                used_authoritative_tenant_llm=(
                    payload.target_llm.provider != runtime_llm.provider
                    or payload.target_llm.model != runtime_llm.model
                ),
                transformation_applied=False,
                bypass_reason=bypass_reason,
                request_log_id=None,
            )
            assistant_result = self.final_response_service.generate(
                runtime_config=runtime_llm,
                resolved_model=runtime_llm.model,
                transformed_prompt=payload.raw_prompt,
                conversation_history=payload.conversation_history,
                attachments=payload.attachments,
                reference_context=self._build_reference_context_for_prompt(
                    tenant_id=runtime_llm.tenant_id,
                    conversation_id=payload.conversation_id,
                    user_id_hash=payload.user_id_hash,
                    raw_prompt=payload.raw_prompt,
                    conversation_history=payload.conversation_history,
                    metadata=bypass_metadata,
                ),
            )
            return ExecuteChatResponse(
                session_id=payload.session_id,
                conversation_id=payload.conversation_id,
                user_id_hash=payload.user_id_hash,
                result_type="transformed",
                task_type="bypassed",
                transformed_prompt=payload.raw_prompt,
                assistant_text=assistant_result.text,
                assistant_images=assistant_result.generated_images,
                coaching_tip=None,
                blocking_message=None,
                conversation=payload.conversation,
                findings=[],
                scoring=None,
                metadata=bypass_metadata,
            )

        transform_response = self.transform(
            TransformPromptRequest(
                session_id=payload.session_id,
                conversation_id=payload.conversation_id,
                user_id_hash=payload.user_id_hash,
                raw_prompt=payload.raw_prompt,
                target_llm=payload.target_llm,
                conversation=payload.conversation,
                summary_type=payload.summary_type,
                enforcement_level=payload.enforcement_level,
            )
        )

        if transform_response.result_type == "blocked":
            assistant_text = transform_response.blocking_message or "This request cannot be sent to the LLM as written."
            return ExecuteChatResponse(
                session_id=transform_response.session_id,
                conversation_id=transform_response.conversation_id,
                user_id_hash=transform_response.user_id_hash,
                result_type=transform_response.result_type,
                task_type=transform_response.task_type,
                transformed_prompt=transform_response.transformed_prompt,
                assistant_text=assistant_text,
                assistant_images=[],
                coaching_tip=transform_response.coaching_tip,
                blocking_message=transform_response.blocking_message,
                conversation=transform_response.conversation,
                findings=transform_response.findings,
                scoring=transform_response.scoring,
                metadata=transform_response.metadata,
            )

        if transform_response.result_type == "coaching":
            assistant_text = _enhance_coaching_tip(
                transform_response.coaching_tip,
                raw_user_text=payload.raw_prompt,
                transformer_conversation=(
                    transform_response.conversation.model_dump()
                    if transform_response.conversation is not None
                    else None
                ),
            )
            return ExecuteChatResponse(
                session_id=transform_response.session_id,
                conversation_id=transform_response.conversation_id,
                user_id_hash=transform_response.user_id_hash,
                result_type=transform_response.result_type,
                task_type=transform_response.task_type,
                transformed_prompt=transform_response.transformed_prompt,
                assistant_text=assistant_text,
                assistant_images=[],
                coaching_tip=transform_response.coaching_tip,
                blocking_message=transform_response.blocking_message,
                conversation=transform_response.conversation,
                findings=transform_response.findings,
                scoring=transform_response.scoring,
                metadata=transform_response.metadata,
            )

        assistant_result = self.final_response_service.generate(
            runtime_config=runtime_llm,
            resolved_model=transform_response.metadata.resolved_model,
            transformed_prompt=transform_response.transformed_prompt or payload.raw_prompt,
            conversation_history=payload.conversation_history,
            attachments=payload.attachments,
            reference_context=self._build_reference_context(runtime_llm.tenant_id, payload, transform_response),
        )
        if transform_response.metadata.request_log_id is not None:
            self.request_logger.set_final_response_usage(
                transform_response.metadata.request_log_id,
                build_usage_entry(
                    category="final_response",
                    purpose="final_response",
                    provider=runtime_llm.provider,
                    model=transform_response.metadata.resolved_model,
                    usage=normalize_usage(runtime_llm.provider, assistant_result.usage),
                ),
            )
        return ExecuteChatResponse(
            session_id=transform_response.session_id,
            conversation_id=transform_response.conversation_id,
            user_id_hash=transform_response.user_id_hash,
            result_type=transform_response.result_type,
            task_type=transform_response.task_type,
            transformed_prompt=transform_response.transformed_prompt,
            assistant_text=assistant_result.text,
            assistant_images=assistant_result.generated_images,
            coaching_tip=transform_response.coaching_tip,
            blocking_message=transform_response.blocking_message,
            conversation=transform_response.conversation,
            findings=transform_response.findings,
            scoring=transform_response.scoring,
            metadata=transform_response.metadata,
        )

    def _build_reference_context(
        self,
        tenant_id: str,
        payload: ExecuteChatRequest,
        transform_response: TransformPromptResponse,
    ) -> str | None:
        return self._build_reference_context_for_prompt(
            tenant_id=tenant_id,
            conversation_id=payload.conversation_id,
            user_id_hash=payload.user_id_hash,
            raw_prompt=payload.raw_prompt,
            conversation_history=payload.conversation_history,
            metadata=transform_response.metadata,
        )

    def _build_reference_context_for_prompt(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
        user_id_hash: str,
        raw_prompt: str,
        conversation_history: list,
        metadata: TransformMetadata,
    ) -> str | None:
        retrieval = self.rag_retrieval.retrieve(
            tenant_id=tenant_id,
            user_id_hash=user_id_hash,
            conversation_id=conversation_id,
            raw_prompt=raw_prompt,
            conversation_history=conversation_history,
        )
        metadata.retrieval_used = bool(retrieval.assembled_references)
        metadata.retrieval_scope_counts = {
            "tenant": retrieval.tenant_chunk_count,
            "user": retrieval.user_chunk_count,
        }
        metadata.retrieval_document_count = retrieval.document_count
        if not retrieval.assembled_references:
            return None
        return self.rag_prompt_assembly.assemble(
            references=retrieval.assembled_references,
        )

    def generate_guide_me_helper(self, payload: GuideMeHelperRequest) -> GuideMeHelperResponse:
        try:
            runtime_llm = self.runtime_llm.resolve(payload.user_id_hash)
            request_row = self.request_logger.log(
                {
                    "session_id": payload.session_id,
                    "conversation_id": payload.conversation_id,
                    "user_id_hash": payload.user_id_hash,
                    "raw_prompt": payload.prompt,
                    "transformed_prompt": None,
                    "task_type": "guide_me",
                    "result_type": "transformed",
                    "coaching_tip": None,
                    "blocking_message": None,
                    "target_provider": runtime_llm.provider,
                    "target_model": runtime_llm.model,
                    "persona_source": "bypassed",
                    "used_fallback_model": False,
                    "enforcement_level": "none",
                    "compliance_check_enabled": False,
                    "pii_check_enabled": False,
                    "conversation_json": {},
                    "findings_json": [],
                    "metadata_json": {
                        "request_kind": "guide_me",
                        "helper_kind": payload.helper_kind,
                        "resolved_provider": runtime_llm.provider,
                        "resolved_model": runtime_llm.model,
                    },
                    "token_usage_json": None,
                }
            )
            helper_result = self.guide_me_generation.generate(
                helper_kind=payload.helper_kind,
                prompt=payload.prompt,
                runtime_config=runtime_llm,
                max_output_tokens=payload.max_output_tokens,
            )
            self.request_logger.append_usage(
                request_row.id,
                build_usage_entry(
                    category="admin",
                    purpose="guide_me",
                    provider=runtime_llm.provider,
                    model=runtime_llm.model,
                    usage=helper_result.usage,
                ),
            )
            return GuideMeHelperResponse(
                session_id=payload.session_id,
                conversation_id=payload.conversation_id,
                user_id_hash=payload.user_id_hash,
                helper_kind=payload.helper_kind,
                payload=helper_result.payload,
            )
        except RuntimeLlmConfigError as exc:
            raise ValueError(str(exc)) from exc

    def _build_prompt(
        self,
        raw_prompt: str,
        task_type: str,
        persona: dict[str, float],
        model_policy: dict,
    ) -> tuple[str, list[str], list[str]]:
        persona_rules = []
        model_rules = []

        lines = [TASK_INSTRUCTION_DEFAULTS[task_type]]

        answer_first = persona["answer_first"] >= 0.65
        if answer_first:
            lines.append("Start with the direct answer before supporting detail.")
            persona_rules.append("persona:answer_first:enabled")

        structure = persona["structure"]
        if structure >= 0.75:
            lines.append("Use a clearly labeled structure with concise sections or bullets.")
            persona_rules.append("persona:structure:high")
        elif structure <= 0.35:
            lines.append("Keep the structure lightweight and natural instead of rigid.")
            persona_rules.append("persona:structure:low")

        detail_level = persona["detail_level"]
        if detail_level >= 0.8:
            lines.append("Include substantive detail, examples, and explicit reasoning.")
            persona_rules.append("persona:detail:high")
        elif detail_level <= 0.35:
            lines.append("Keep the response brief and focused on the essentials.")
            persona_rules.append("persona:detail:low")

        ambiguity = persona["ambiguity_reduction"]
        if ambiguity >= 0.75:
            lines.append("Reduce ambiguity by stating assumptions, constraints, and next actions explicitly.")
            persona_rules.append("persona:ambiguity:high")

        exploration = persona["exploration_level"]
        if exploration >= 0.75:
            lines.append("Offer multiple angles or options before converging on one path.")
            persona_rules.append("persona:exploration:high")
        elif exploration <= 0.3:
            lines.append("Prefer one strong recommendation instead of many alternatives.")
            persona_rules.append("persona:exploration:low")

        context = persona["context_loading"]
        if context >= 0.75:
            lines.append("Load helpful context proactively when it improves the answer.")
            persona_rules.append("persona:context:high")
        elif context <= 0.3:
            lines.append("Avoid extra background unless it is required to answer well.")
            persona_rules.append("persona:context:low")

        directness = persona["tone_directness"]
        if directness >= 0.7:
            lines.append("Use direct, confident phrasing.")
            persona_rules.append("persona:tone:direct")
        elif directness <= 0.35:
            lines.append("Use a softer, more exploratory tone.")
            persona_rules.append("persona:tone:gentle")

        format_strictness = model_policy.get("format_strictness", "medium")
        if format_strictness == "high":
            lines.append("Follow formatting instructions exactly and keep output clean.")
            model_rules.append("model:format:high")

        if model_policy.get("stepwise") == "helpful" and task_type in {"planning", "analysis", "decision_support"}:
            lines.append("Use stepwise reasoning in the visible response only when it improves clarity.")
            model_rules.append("model:stepwise:helpful")

        verbosity = model_policy.get("verbosity", "medium")
        if verbosity == "low":
            lines.append("Bias toward a compact response.")
            model_rules.append("model:verbosity:low")
        elif verbosity == "high":
            lines.append("Bias toward a more expansive response.")
            model_rules.append("model:verbosity:high")

        lines.append("User request:")
        lines.append(raw_prompt.strip())

        return "\n".join(lines), persona_rules, model_rules
