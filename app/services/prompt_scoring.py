from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.core.rules import get_rule_registry
from app.models.prompt_score import ConversationPromptScore
from app.schemas.transform import ConversationRequirement, ConversationState, PromptScoringSummary
from app.services.prompt_requirements import RequirementEvaluationTrace


@dataclass(frozen=True)
class PromptScoreResult:
    scoring_version: str
    initial_score: int
    final_score: int
    initial_llm_score: Optional[int]
    final_llm_score: Optional[int]
    structural_score: int
    field_statuses: dict[str, str]
    field_points: dict[str, int]
    scored_requirements: dict[str, ConversationRequirement]
    heuristic_score: int
    llm_score: Optional[int]
    llm_dimension_scores: dict[str, int] | None
    scoring_method: str
    score_details: dict[str, object]
    conversation_scored_requirements: dict[str, ConversationRequirement]

    def as_summary(self) -> PromptScoringSummary:
        return PromptScoringSummary(
            scoring_version=self.scoring_version,
            initial_score=self.initial_score,
            final_score=self.final_score,
            initial_llm_score=self.initial_llm_score,
            final_llm_score=self.final_llm_score,
            structural_score=self.structural_score,
        )


class PromptScoringService:
    _executor: ThreadPoolExecutor | None = None
    _executor_lock = Lock()
    _pending_scores: dict[str, dict[str, object]] = {}
    _pending_lock = Lock()
    _scheduled_conversations: set[str] = set()

    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.settings = get_settings()
        scoring_rules = get_rule_registry().prompt_scoring
        self.scoring_version = str(scoring_rules.get("version", "v1"))
        self.field_weights = {
            "who": int(scoring_rules.get("field_weights", {}).get("who", 25)),
            "task": int(scoring_rules.get("field_weights", {}).get("task", 25)),
            "context": int(scoring_rules.get("field_weights", {}).get("context", 25)),
            "output": int(scoring_rules.get("field_weights", {}).get("output", 25)),
        }
        self.status_points = {
            "present": int(scoring_rules.get("status_points", {}).get("present", 25)),
            "user_provided": int(scoring_rules.get("status_points", {}).get("present", 25)),
            "derived": int(scoring_rules.get("status_points", {}).get("derived", 5)),
            "missing": int(scoring_rules.get("status_points", {}).get("missing", 0)),
        }
        blend_weights = scoring_rules.get("blend_weights", {})
        heuristic_weight = float(blend_weights.get("heuristic", 0.5))
        llm_weight = float(blend_weights.get("llm", 0.5))
        total_weight = heuristic_weight + llm_weight
        if total_weight <= 0:
            self.heuristic_weight = 1.0
            self.llm_weight = 0.0
        else:
            self.heuristic_weight = heuristic_weight / total_weight
            self.llm_weight = llm_weight / total_weight

    def calculate(
        self,
        *,
        conversation: ConversationState,
        result_type: str,
        requirement_trace: RequirementEvaluationTrace,
        submitted_conversation: ConversationState | None = None,
        submitted_requirement_trace: RequirementEvaluationTrace | None = None,
    ) -> PromptScoreResult:
        initial_result = self._calculate_single_prompt_score(
            conversation=conversation,
            result_type=result_type,
            requirement_trace=requirement_trace,
        )
        final_result = (
            self._calculate_single_prompt_score(
                conversation=submitted_conversation,
                result_type=result_type,
                requirement_trace=submitted_requirement_trace,
            )
            if result_type == "transformed"
            and submitted_conversation is not None
            and submitted_requirement_trace is not None
            else initial_result
        )
        score_details = dict(final_result.score_details)
        score_details["initial_score"] = initial_result.structural_score
        score_details["initial_llm_score"] = initial_result.llm_score
        score_details["raw_requirements"] = {
            field_name: requirement.model_dump()
            for field_name, requirement in initial_result.scored_requirements.items()
        }
        score_details["submitted_requirements"] = {
            field_name: requirement.model_dump()
            for field_name, requirement in final_result.scored_requirements.items()
        }

        return PromptScoreResult(
            scoring_version=self.scoring_version,
            initial_score=initial_result.structural_score,
            final_score=final_result.structural_score,
            initial_llm_score=initial_result.llm_score,
            final_llm_score=final_result.llm_score,
            structural_score=final_result.structural_score,
            field_statuses=final_result.field_statuses,
            field_points=final_result.field_points,
            scored_requirements=final_result.scored_requirements,
            heuristic_score=final_result.heuristic_score,
            llm_score=final_result.llm_score,
            llm_dimension_scores=final_result.llm_dimension_scores,
            scoring_method=final_result.scoring_method,
            score_details=score_details,
            conversation_scored_requirements=initial_result.scored_requirements,
        )

    def _calculate_single_prompt_score(
        self,
        *,
        conversation: ConversationState,
        result_type: str,
        requirement_trace: RequirementEvaluationTrace,
    ) -> PromptScoreResult:
        heuristic_statuses = {
            field_name: requirement_trace.heuristic[field_name].status
            for field_name in self.field_weights
        }
        llm_statuses = (
            {
                field_name: requirement_trace.evaluator[field_name].status
                for field_name in self.field_weights
            }
            if requirement_trace.evaluator_used
            else None
        )
        field_statuses = {
            field_name: requirement_trace.current[field_name].status
            for field_name in self.field_weights
        }
        heuristic_field_points = {
            field_name: self.status_points.get(status, 0)
            for field_name, status in heuristic_statuses.items()
        }
        llm_field_points = (
            requirement_trace.evaluator_scores
            if requirement_trace.evaluator_scores is not None
            else {
                field_name: self.status_points.get(status, 0)
                for field_name, status in llm_statuses.items()
            }
            if llm_statuses is not None
            else None
        )
        field_points = {
            field_name: self.status_points.get(status, 0)
            for field_name, status in field_statuses.items()
        }
        scored_requirements = self._build_scored_requirements(
            conversation=conversation,
            requirement_trace=requirement_trace,
        )
        heuristic_score = sum(heuristic_field_points.values())
        llm_score = sum(llm_field_points.values()) if llm_field_points is not None else None
        fused_field_score = sum(field_points.values())
        structural_score = (
            round((heuristic_score * self.heuristic_weight) + (llm_score * self.llm_weight))
            if llm_score is not None
            else heuristic_score
        )

        score_details = {
            "field_points": field_points,
            "field_statuses": field_statuses,
            "heuristic_field_points": heuristic_field_points,
            "heuristic_field_statuses": heuristic_statuses,
            "llm_field_points": llm_field_points,
            "llm_field_statuses": llm_statuses,
            "weights": self.field_weights,
            "missing_fields": list(conversation.enforcement.missing_fields),
            "enforcement_level": conversation.enforcement.level,
            "enforcement_status": conversation.enforcement.status,
            "result_type": result_type,
            "calculated_at": datetime.now(timezone.utc).isoformat(),
            "scoring_version": self.scoring_version,
            "requirements": {
                field_name: requirement.model_dump()
                for field_name, requirement in scored_requirements.items()
            },
            "heuristic_score": heuristic_score,
            "llm_score": llm_score,
            "fused_field_score": fused_field_score,
            "final_score": structural_score,
            "blend_weights": {
                "heuristic": self.heuristic_weight,
                "llm": self.llm_weight,
            },
            "scoring_method": "hybrid_llm_v2" if requirement_trace.evaluator_used else "heuristic_only_v1",
        }

        return PromptScoreResult(
            scoring_version=self.scoring_version,
            initial_score=structural_score,
            final_score=structural_score,
            initial_llm_score=llm_score,
            final_llm_score=llm_score,
            structural_score=structural_score,
            field_statuses=field_statuses,
            field_points=field_points,
            scored_requirements=scored_requirements,
            heuristic_score=heuristic_score,
            llm_score=llm_score,
            llm_dimension_scores=llm_field_points,
            scoring_method="hybrid_llm_v2" if requirement_trace.evaluator_used else "heuristic_only_v1",
            score_details=score_details,
            conversation_scored_requirements=scored_requirements,
        )

    def upsert_conversation_score(
        self,
        *,
        conversation: ConversationState,
        user_id_hash: str,
        task_type: str,
        result_type: str,
        score_result: PromptScoreResult,
    ) -> PromptScoringSummary:
        if self.settings.enable_async_score_persistence:
            self._enqueue_score_persistence(
                conversation=conversation,
                user_id_hash=user_id_hash,
                task_type=task_type,
                result_type=result_type,
                score_result=score_result,
            )
            return score_result.as_summary()

        score_row = self._upsert_conversation_score_sync(
            db_session=self.db_session,
            conversation=conversation,
            user_id_hash=user_id_hash,
            task_type=task_type,
            result_type=result_type,
            score_result=score_result,
        )
        return self.attach_rollup_scores(
            score_result=score_result,
            score_row=score_row,
        ).as_summary()

    @classmethod
    def _get_executor(cls, max_workers: int) -> ThreadPoolExecutor:
        with cls._executor_lock:
            if cls._executor is None:
                cls._executor = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="prompt-score-writer",
                )
            return cls._executor

    def _enqueue_score_persistence(
        self,
        *,
        conversation: ConversationState,
        user_id_hash: str,
        task_type: str,
        result_type: str,
        score_result: PromptScoreResult,
    ) -> None:
        conversation_id = conversation.conversation_id
        payload = {
            "conversation_id": conversation.conversation_id,
            "enforcement_level": conversation.enforcement.level,
            "enforcement_status": conversation.enforcement.status,
            "user_id_hash": user_id_hash,
            "task_type": task_type,
            "result_type": result_type,
            "score_result": score_result,
        }

        with self._pending_lock:
            self._pending_scores[conversation_id] = payload
            if conversation_id in self._scheduled_conversations:
                return
            self._scheduled_conversations.add(conversation_id)

        self._get_executor(self.settings.score_persistence_workers).submit(
            self._flush_conversation_score,
            conversation_id,
            self.settings.score_persistence_debounce_seconds,
        )

    @classmethod
    def _flush_conversation_score(cls, conversation_id: str, debounce_seconds: float) -> None:
        while True:
            time.sleep(debounce_seconds)
            with cls._pending_lock:
                payload = cls._pending_scores.pop(conversation_id, None)

            if payload is None:
                with cls._pending_lock:
                    cls._scheduled_conversations.discard(conversation_id)
                return

            db_session = SessionLocal()
            try:
                cls._upsert_conversation_score_sync(
                    db_session=db_session,
                    conversation_id=str(payload["conversation_id"]),
                    enforcement_level=str(payload["enforcement_level"]),
                    enforcement_status=str(payload["enforcement_status"]),
                    user_id_hash=str(payload["user_id_hash"]),
                    task_type=str(payload["task_type"]),
                    result_type=str(payload["result_type"]),
                    score_result=payload["score_result"],
                )
            finally:
                db_session.close()

            with cls._pending_lock:
                if conversation_id not in cls._pending_scores:
                    cls._scheduled_conversations.discard(conversation_id)
                    return

    @staticmethod
    def _upsert_conversation_score_sync(
        *,
        db_session: Session,
        score_result: PromptScoreResult,
        user_id_hash: str,
        task_type: str,
        result_type: str,
        conversation: ConversationState | None = None,
        conversation_id: str | None = None,
        enforcement_level: str | None = None,
        enforcement_status: str | None = None,
    ) -> ConversationPromptScore:
        resolved_conversation_id = conversation.conversation_id if conversation is not None else conversation_id
        resolved_enforcement_level = (
            conversation.enforcement.level if conversation is not None else enforcement_level
        )
        resolved_enforcement_status = (
            conversation.enforcement.status if conversation is not None else enforcement_status
        )

        if resolved_conversation_id is None or resolved_enforcement_level is None or resolved_enforcement_status is None:
            raise ValueError("Conversation score persistence requires conversation identity and enforcement state.")

        now = datetime.now(timezone.utc)
        score_row = (
            db_session.query(ConversationPromptScore)
            .filter_by(conversation_id=resolved_conversation_id)
            .one_or_none()
        )

        if score_row is None:
            score_row = ConversationPromptScore(
                conversation_id=resolved_conversation_id,
                user_id_hash=user_id_hash,
                task_type=task_type,
                conversation_started_at=now,
                last_scored_at=now,
                enforcement_level=resolved_enforcement_level,
                initial_score=score_result.initial_score,
                best_score=score_result.final_score,
                final_score=score_result.final_score,
                initial_llm_score=score_result.initial_llm_score,
                best_llm_score=score_result.final_llm_score,
                final_llm_score=score_result.final_llm_score,
                improvement_score=score_result.final_score - score_result.initial_score,
                best_improvement_score=score_result.final_score - score_result.initial_score,
                passed_without_coaching=result_type == "transformed",
                reached_policy_complete=resolved_enforcement_status == "passes",
                coaching_turn_count=1 if result_type == "coaching" else 0,
                blocked_turn_count=1 if result_type == "blocked" else 0,
                transformed_turn_count=1 if result_type == "transformed" else 0,
                who_status=score_result.field_statuses["who"],
                task_status=score_result.field_statuses["task"],
                context_status=score_result.field_statuses["context"],
                output_status=score_result.field_statuses["output"],
                score_details_json=score_result.score_details,
                scoring_version=score_result.scoring_version,
            )
            db_session.add(score_row)
        else:
            score_row.task_type = task_type if task_type != "unknown" else score_row.task_type
            score_row.last_scored_at = now
            score_row.enforcement_level = resolved_enforcement_level
            score_row.final_score = score_result.final_score
            score_row.best_score = max(score_row.best_score, score_result.final_score)
            score_row.final_llm_score = score_result.final_llm_score
            if score_result.final_llm_score is not None:
                score_row.best_llm_score = (
                    score_result.final_llm_score
                    if score_row.best_llm_score is None
                    else max(score_row.best_llm_score, score_result.final_llm_score)
                )
            score_row.improvement_score = score_row.final_score - score_row.initial_score
            score_row.best_improvement_score = score_row.best_score - score_row.initial_score
            score_row.reached_policy_complete = (
                score_row.reached_policy_complete or resolved_enforcement_status == "passes"
            )
            score_row.who_status = score_result.field_statuses["who"]
            score_row.task_status = score_result.field_statuses["task"]
            score_row.context_status = score_result.field_statuses["context"]
            score_row.output_status = score_result.field_statuses["output"]
            score_row.score_details_json = score_result.score_details
            score_row.scoring_version = score_result.scoring_version

            if result_type == "coaching":
                score_row.coaching_turn_count += 1
            elif result_type == "blocked":
                score_row.blocked_turn_count += 1
            else:
                score_row.transformed_turn_count += 1

            score_row.passed_without_coaching = (
                score_row.passed_without_coaching
                if score_row.transformed_turn_count > 0 and score_row.coaching_turn_count == 0
                else False
            )

        db_session.commit()
        return score_row

    def attach_rollup_scores(
        self,
        *,
        score_result: PromptScoreResult,
        score_row: ConversationPromptScore,
    ) -> PromptScoreResult:
        return PromptScoreResult(
            scoring_version=score_row.scoring_version,
            initial_score=score_row.initial_score,
            final_score=score_row.final_score,
            initial_llm_score=score_row.initial_llm_score,
            final_llm_score=score_row.final_llm_score,
            structural_score=score_result.structural_score,
            field_statuses=score_result.field_statuses,
            field_points=score_result.field_points,
            scored_requirements=score_result.scored_requirements,
            heuristic_score=score_result.heuristic_score,
            llm_score=score_result.llm_score,
            llm_dimension_scores=score_result.llm_dimension_scores,
            scoring_method=score_result.scoring_method,
            score_details=score_result.score_details,
            conversation_scored_requirements=score_result.conversation_scored_requirements,
        )

    def enrich_conversation(
        self,
        *,
        conversation: ConversationState,
        score_result: PromptScoreResult,
    ) -> ConversationState:
        return conversation.model_copy(
            update={
                "requirements": {
                    field_name: requirement.model_copy()
                    for field_name, requirement in score_result.conversation_scored_requirements.items()
                }
            }
        )

    def _build_scored_requirements(
        self,
        *,
        conversation: ConversationState,
        requirement_trace: RequirementEvaluationTrace,
    ) -> dict[str, ConversationRequirement]:
        scored_requirements: dict[str, ConversationRequirement] = {}
        for field_name, max_score in self.field_weights.items():
            requirement = conversation.requirements[field_name]
            heuristic_score = self.status_points.get(requirement.status, 0)
            current_requirement = requirement_trace.current[field_name]
            evaluator_requirement = requirement_trace.evaluator[field_name]
            carried_forward = (
                requirement.status != current_requirement.status
                or requirement.value != current_requirement.value
            ) and requirement_trace.evaluator_used and (
                requirement.status != evaluator_requirement.status
                or requirement.value != evaluator_requirement.value
            )
            llm_score = requirement_trace.evaluator_scores.get(field_name) if requirement_trace.evaluator_scores else None
            if carried_forward:
                llm_score = heuristic_score
            reason, improvement_hint = self._describe_requirement(
                field_name=field_name,
                requirement=requirement,
                max_score=max_score,
                llm_score=llm_score,
                carried_forward=carried_forward,
            )
            scored_requirements[field_name] = ConversationRequirement(
                value=requirement.value,
                status=requirement.status,
                heuristic_score=heuristic_score,
                llm_score=llm_score,
                max_score=max_score,
                reason=reason,
                improvement_hint=improvement_hint,
            )
        return scored_requirements

    def _describe_requirement(
        self,
        *,
        field_name: str,
        requirement: ConversationRequirement,
        max_score: int,
        llm_score: int | None,
        carried_forward: bool,
    ) -> tuple[str, str | None]:
        field_labels = {
            "who": "role",
            "task": "task",
            "context": "context",
            "output": "output format",
        }
        improvement_hints = {
            "who": "State the exact role, perspective, or audience the AI should serve.",
            "task": "State the exact outcome and decision criteria.",
            "context": "Add the situation, constraints, or why this answer will be used.",
            "output": "Add the exact structure, format, and length you want back.",
        }
        label = field_labels[field_name]

        if carried_forward:
            return (
                f"{label.capitalize()} is available from earlier conversation state.",
                None,
            )
        if requirement.status == "missing":
            return (
                f"{label.capitalize()} is missing.",
                improvement_hints[field_name],
            )
        if requirement.status == "derived":
            return (
                f"{label.capitalize()} can be inferred, but it is not stated explicitly.",
                improvement_hints[field_name],
            )
        if llm_score is None:
            return (
                f"{label.capitalize()} is explicitly present.",
                None,
            )
        if llm_score >= max_score:
            return (
                f"{label.capitalize()} is clear and specific.",
                None,
            )
        if llm_score >= max(1, int(round(max_score * 0.7))):
            return (
                f"{label.capitalize()} is present but could be more specific.",
                improvement_hints[field_name],
            )
        return (
            f"{label.capitalize()} is present but still too vague.",
            improvement_hints[field_name],
        )
