from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.schemas.transform import ConversationEnforcement, ConversationRequirement, ConversationState
from app.services.runtime_llm import RuntimeLlmConfig
from app.services.structure_evaluator import StructureEvaluationService


REQUIREMENT_FIELDS = ("who", "task", "context", "output")
STATUS_PRIORITY = {
    "missing": 0,
    "derived": 1,
    "present": 2,
}
OUTPUT_KEYWORDS = (
    "json",
    "bullet",
    "bullets",
    "list",
    "table",
    "email",
    "memo",
    "markdown",
    "image",
    "file",
    "code",
    "script",
    "plan",
)
TASK_STARTERS = (
    "tell",
    "explain",
    "summarize",
    "write",
    "draft",
    "analyze",
    "compare",
    "debug",
    "plan",
    "create",
    "generate",
    "recommend",
    "review",
    "help",
)
WHO_PATTERNS = (
    re.compile(r"\bact as (?:an? )?([^.,\n]+)", re.IGNORECASE),
    re.compile(r"\byou are (?:an? )?([^.,\n]+)", re.IGNORECASE),
    re.compile(r"\bas (?:an? )?([^.,\n]+?)(?:,|\.|\n| to | for )", re.IGNORECASE),
)
CONTEXT_PATTERNS = (
    re.compile(r"\bat ([^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\bso that ([^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\bintended for ([^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\bi am preparing[^.!?\n]+", re.IGNORECASE),
    re.compile(r"\bi need [^.!?\n]+", re.IGNORECASE),
    re.compile(r"\bthis is for [^.!?\n]+", re.IGNORECASE),
)
OUTPUT_PATTERNS = (
    re.compile(r"\bin the chat\b", re.IGNORECASE),
    re.compile(r"\bjust give me [^.!?\n]+", re.IGNORECASE),
    re.compile(r"\breturn [^.!?\n]+", re.IGNORECASE),
)
LABEL_PATTERNS = {
    "who": re.compile(r"(?im)^\s*who\s*:\s*.+$"),
    "task": re.compile(r"(?im)^\s*task\s*:\s*.+$"),
    "context": re.compile(r"(?im)^\s*context\s*:\s*.+$"),
    "output": re.compile(r"(?im)^\s*output\s*:\s*.+$"),
}


@dataclass(frozen=True)
class RequirementEvaluationTrace:
    heuristic: dict[str, ConversationRequirement]
    evaluator: dict[str, ConversationRequirement]
    evaluator_scores: dict[str, int] | None
    current: dict[str, ConversationRequirement]
    fused: dict[str, ConversationRequirement]
    evaluator_used: bool


class PromptRequirementService:
    def __init__(self) -> None:
        self.structure_evaluator = StructureEvaluationService()

    def evaluate(
        self,
        conversation_id: str,
        raw_prompt: str,
        conversation: Optional[ConversationState],
        enforcement_level: str,
        runtime_config: RuntimeLlmConfig | None = None,
    ) -> tuple[ConversationState, list[str], Optional[str], "RequirementEvaluationTrace", Optional[dict[str, object]]]:
        requirements, evaluator_used, evaluator_coaching_tip, evaluation_trace, evaluator_usage_entry = self._merge_requirements(
            raw_prompt=raw_prompt,
            conversation=conversation,
            enforcement_level=enforcement_level,
            runtime_config=runtime_config,
        )
        blocking_fields = self._blocking_fields(
            requirements=requirements,
            enforcement_level=enforcement_level,
            raw_prompt=raw_prompt,
        )
        coaching_fields = self._coaching_fields(
            requirements=requirements,
            enforcement_level=enforcement_level,
            raw_prompt=raw_prompt,
            blocking_fields=blocking_fields,
        )
        status = "passes" if not blocking_fields else "needs_coaching"
        reported_fields = blocking_fields or coaching_fields

        updated_conversation = ConversationState(
            conversation_id=conversation_id,
            requirements=requirements,
            enforcement=ConversationEnforcement(
                level=enforcement_level,
                status=status,
                missing_fields=reported_fields,
                last_evaluated_at=datetime.now(timezone.utc).isoformat(),
            ),
        )

        rules_applied = [f"policy:enforcement:{enforcement_level}"]
        for field_name in REQUIREMENT_FIELDS:
            if requirements[field_name].status == "present":
                rules_applied.append(f"requirement:{field_name}:present")
            elif requirements[field_name].status == "derived":
                rules_applied.append(f"requirement:{field_name}:derived")
        if enforcement_level == "full" and self._is_labeled_prompt(raw_prompt):
            rules_applied.append("policy:enforcement:full:labeled")
        if evaluator_used:
            rules_applied.append("requirement:evaluator:llm")

        coaching_tip = None
        if blocking_fields:
            if "labeled_structure" in blocking_fields:
                coaching_tip = self._build_coaching_tip(blocking_fields, enforcement_level)
            else:
                coaching_tip = evaluator_coaching_tip or self._build_coaching_tip(
                    blocking_fields,
                    enforcement_level,
                )
        elif coaching_fields:
            coaching_tip = self._build_coaching_tip(coaching_fields, enforcement_level)

        return updated_conversation, rules_applied, coaching_tip, evaluation_trace, evaluator_usage_entry

    def evaluate_current_prompt(
        self,
        *,
        conversation_id: str,
        raw_prompt: str,
        runtime_config: RuntimeLlmConfig | None = None,
    ) -> tuple[ConversationState, "RequirementEvaluationTrace", Optional[dict[str, object]]]:
        requirements, _, _, evaluation_trace, evaluator_usage_entry = self._merge_requirements(
            raw_prompt=raw_prompt,
            conversation=None,
            enforcement_level="none",
            runtime_config=runtime_config,
        )
        missing_fields = [
            field_name
            for field_name in REQUIREMENT_FIELDS
            if requirements[field_name].status == "missing"
        ]
        conversation = ConversationState(
            conversation_id=conversation_id,
            requirements=requirements,
            enforcement=ConversationEnforcement(
                level="none",
                status="passes",
                missing_fields=missing_fields,
                last_evaluated_at=datetime.now(timezone.utc).isoformat(),
            ),
        )
        return conversation, evaluation_trace, evaluator_usage_entry

    def _build_coaching_tip(self, missing_fields: list[str], enforcement_level: str) -> str:
        if missing_fields == ["labeled_structure"]:
            return "Coaching: for full guidance, format the prompt with Who, Task, Context, and Output labels."

        labels = []
        for field_name in missing_fields:
            if field_name == "labeled_structure":
                labels.append("Who, Task, Context, and Output labels")
            elif field_name == "who":
                labels.append("the role you want the AI to play")
            elif field_name == "context":
                labels.append("the setting or intended use")
            elif field_name == "output":
                labels.append("how you want the answer delivered")
            else:
                labels.append(field_name)

        if len(labels) == 1:
            detail_text = labels[0]
        elif len(labels) == 2:
            detail_text = f"{labels[0]} and {labels[1]}"
        else:
            detail_text = f"{', '.join(labels[:-1])}, and {labels[-1]}"

        if enforcement_level == "low":
            return f"Coaching: include {detail_text} next time for a stronger prompt."
        return f"Coaching: add {detail_text}."

    def _merge_requirements(
        self,
        raw_prompt: str,
        conversation: Optional[ConversationState],
        enforcement_level: str,
        runtime_config: RuntimeLlmConfig | None = None,
    ) -> tuple[
        dict[str, ConversationRequirement],
        bool,
        Optional[str],
        "RequirementEvaluationTrace",
        Optional[dict[str, object]],
    ]:
        existing = conversation.requirements if conversation is not None else {}
        evaluator_result = self.structure_evaluator.evaluate(
            raw_prompt=raw_prompt,
            enforcement_level=enforcement_level,
            runtime_config=runtime_config,
        )
        if isinstance(evaluator_result, tuple):
            evaluator_payload, evaluator_usage_entry = evaluator_result
        else:
            evaluator_payload = evaluator_result
            evaluator_usage_entry = None
        merged_current: dict[str, ConversationRequirement] = {}
        merged_conversation: dict[str, ConversationRequirement] = {}
        heuristic_requirements: dict[str, ConversationRequirement] = {}
        evaluator_requirements: dict[str, ConversationRequirement] = {}
        evaluator_scores: dict[str, int] = {}

        for field_name in REQUIREMENT_FIELDS:
            existing_requirement = existing.get(field_name)
            evaluator_value = self._read_evaluator_requirement(evaluator_payload, field_name)
            current_requirement = self._infer_requirement(field_name, raw_prompt)
            heuristic_requirements[field_name] = current_requirement
            evaluator_requirements[field_name] = evaluator_value or ConversationRequirement(
                value=None,
                status="missing",
            )
            evaluator_score = self._read_evaluator_score(evaluator_payload, field_name)
            if evaluator_score is not None:
                evaluator_scores[field_name] = evaluator_score
            merged_current[field_name] = self._select_best_requirement(
                current=current_requirement,
                evaluator=evaluator_value,
                existing=None,
            )
            merged_conversation[field_name] = self._select_best_requirement(
                current=current_requirement,
                evaluator=evaluator_value,
                existing=existing_requirement,
            )

        evaluator_used = evaluator_payload is not None
        evaluator_coaching_tip = None
        if evaluator_payload is not None:
            coaching_tip = evaluator_payload.get("coaching_tip")
            if isinstance(coaching_tip, str) and coaching_tip.strip():
                evaluator_coaching_tip = coaching_tip.strip()

        return (
            merged_conversation,
            evaluator_used,
            evaluator_coaching_tip,
            RequirementEvaluationTrace(
                heuristic=heuristic_requirements,
                evaluator=evaluator_requirements,
                evaluator_scores=evaluator_scores or None,
                current=merged_current,
                fused=merged_conversation,
                evaluator_used=evaluator_used,
            ),
            evaluator_usage_entry,
        )

    def _read_evaluator_requirement(
        self,
        evaluator_payload: Optional[dict[str, object]],
        field_name: str,
    ) -> Optional[ConversationRequirement]:
        if evaluator_payload is None:
            return None
        raw_value = evaluator_payload.get(field_name)
        if not isinstance(raw_value, dict):
            return None
        value = raw_value.get("value")
        status = raw_value.get("status")
        if status not in {"present", "derived", "missing", "user_provided"}:
            return None
        if value is None:
            return ConversationRequirement(value=None, status="missing")
        if isinstance(value, str) and value.strip():
            return ConversationRequirement(value=value.strip(), status=str(status))
        return ConversationRequirement(value=None, status="missing")

    def _read_evaluator_score(
        self,
        evaluator_payload: Optional[dict[str, object]],
        field_name: str,
    ) -> Optional[int]:
        if evaluator_payload is None:
            return None
        raw_value = evaluator_payload.get(field_name)
        if not isinstance(raw_value, dict):
            return None
        score = raw_value.get("score")
        if isinstance(score, bool):
            return None
        if isinstance(score, (int, float)):
            normalized = int(round(float(score)))
            return max(0, min(25, normalized))
        return None

    def _select_best_requirement(
        self,
        *,
        current: ConversationRequirement,
        evaluator: Optional[ConversationRequirement],
        existing: Optional[ConversationRequirement],
    ) -> ConversationRequirement:
        candidates = [candidate for candidate in (evaluator, current, existing) if candidate is not None]
        if not candidates:
            return ConversationRequirement(value=None, status="missing")
        candidates.sort(key=lambda candidate: STATUS_PRIORITY.get(candidate.status, 0), reverse=True)
        return candidates[0]

    def _infer_requirement(self, field_name: str, raw_prompt: str) -> ConversationRequirement:
        text = raw_prompt.strip()
        if not text:
            return ConversationRequirement(value=None, status="missing")

        if field_name == "task":
            if text.endswith("?") or len(text.split()) >= 3:
                return ConversationRequirement(value=text, status="present")
            first_word = text.split()[0].lower()
            if first_word in TASK_STARTERS:
                return ConversationRequirement(value=text, status="present")
            return ConversationRequirement(value=None, status="missing")

        if field_name == "who":
            for pattern in WHO_PATTERNS:
                match = pattern.search(text)
                if match:
                    return ConversationRequirement(
                        value=match.group(1).strip(" ,."),
                        status="present",
                    )
            return ConversationRequirement(value=None, status="missing")

        if field_name == "context":
            for pattern in CONTEXT_PATTERNS:
                match = pattern.search(text)
                if match:
                    return ConversationRequirement(
                        value=match.group(0).strip(" ,."),
                        status="present",
                    )
            return ConversationRequirement(value=None, status="missing")

        if field_name == "output":
            for pattern in OUTPUT_PATTERNS:
                match = pattern.search(text)
                if match:
                    return ConversationRequirement(
                        value=match.group(0).strip(" ,."),
                        status="present",
                    )
            lower_text = text.lower()
            for keyword in OUTPUT_KEYWORDS:
                if keyword in lower_text:
                    status = "present" if re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE) else "derived"
                    return ConversationRequirement(value=keyword, status=status)
            if any(
                phrase in lower_text
                for phrase in (
                    "answer in",
                    "respond in",
                    "provide the answer in",
                )
            ):
                return ConversationRequirement(value="answer in chat", status="present")
            if any(re.search(rf"\b{re.escape(starter)}\b", lower_text) for starter in TASK_STARTERS) or lower_text.endswith("?"):
                return ConversationRequirement(value="chat response", status="derived")
            return ConversationRequirement(value=None, status="missing")

        return ConversationRequirement(value=None, status="missing")

    def _fields_below_status(
        self,
        requirements: dict[str, ConversationRequirement],
        minimum_status: str,
    ) -> list[str]:
        minimum_priority = STATUS_PRIORITY[minimum_status]
        return [
            field_name
            for field_name in REQUIREMENT_FIELDS
            if STATUS_PRIORITY.get(requirements[field_name].status, 0) < minimum_priority
        ]

    def _blocking_fields(
        self,
        requirements: dict[str, ConversationRequirement],
        enforcement_level: str,
        raw_prompt: str,
    ) -> list[str]:
        if enforcement_level in {"none", "low"}:
            return []

        if enforcement_level == "moderate":
            return self._fields_below_status(requirements, "derived")

        missing = self._fields_below_status(requirements, "present")

        if enforcement_level == "full" and not missing and not self._is_labeled_prompt(raw_prompt):
            return ["labeled_structure"]
        return missing

    def _coaching_fields(
        self,
        *,
        requirements: dict[str, ConversationRequirement],
        enforcement_level: str,
        raw_prompt: str,
        blocking_fields: list[str],
    ) -> list[str]:
        if blocking_fields:
            return blocking_fields
        if enforcement_level == "none":
            return []
        if enforcement_level == "low":
            return self._fields_below_status(requirements, "present")
        if enforcement_level == "moderate":
            return self._fields_below_status(requirements, "present")
        if enforcement_level == "full" and not self._is_labeled_prompt(raw_prompt):
            return ["labeled_structure"]
        return []

    def _is_labeled_prompt(self, raw_prompt: str) -> bool:
        return all(pattern.search(raw_prompt) for pattern in LABEL_PATTERNS.values())
