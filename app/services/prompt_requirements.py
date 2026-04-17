from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from app.schemas.transform import ConversationEnforcement, ConversationRequirement, ConversationState
from app.services.structure_evaluator import StructureEvaluationService


REQUIREMENT_FIELDS = ("who", "task", "context", "output")
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
    re.compile(r"\bfor ([^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\bat ([^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\bso that ([^.!?\n]+)", re.IGNORECASE),
    re.compile(r"\bintended for ([^.!?\n]+)", re.IGNORECASE),
)
OUTPUT_PATTERNS = (
    re.compile(r"\bin the chat\b", re.IGNORECASE),
    re.compile(r"\bjust give me [^.!?\n]+", re.IGNORECASE),
    re.compile(r"\breturn [^.!?\n]+", re.IGNORECASE),
)


class PromptRequirementService:
    def __init__(self) -> None:
        self.structure_evaluator = StructureEvaluationService()

    def evaluate(
        self,
        conversation_id: str,
        raw_prompt: str,
        conversation: Optional[ConversationState],
        enforcement_level: str,
    ) -> tuple[ConversationState, list[str], Optional[str]]:
        requirements, evaluator_used, evaluator_coaching_tip = self._merge_requirements(
            raw_prompt=raw_prompt,
            conversation=conversation,
            enforcement_level=enforcement_level,
        )
        missing_fields = self._missing_fields(requirements, enforcement_level)
        status = "passes" if not missing_fields else "needs_coaching"

        updated_conversation = ConversationState(
            conversation_id=conversation_id,
            requirements=requirements,
            enforcement=ConversationEnforcement(
                level=enforcement_level,
                status=status,
                missing_fields=missing_fields,
                last_evaluated_at=datetime.now(timezone.utc).isoformat(),
            ),
        )

        rules_applied = [f"policy:enforcement:{enforcement_level}"]
        if requirements["who"].status == "derived":
            rules_applied.append("requirement:who:derived")
        if requirements["task"].status == "derived":
            rules_applied.append("requirement:task:derived")
        if requirements["context"].status == "derived":
            rules_applied.append("requirement:context:derived")
        if requirements["output"].status == "derived":
            rules_applied.append("requirement:output:derived")
        if evaluator_used:
            rules_applied.append("requirement:evaluator:llm")

        coaching_tip = None
        if missing_fields:
            coaching_tip = evaluator_coaching_tip or self._build_coaching_tip(missing_fields)

        return updated_conversation, rules_applied, coaching_tip

    def _build_coaching_tip(self, missing_fields: list[str]) -> str:
        labels = []
        for field_name in missing_fields:
            if field_name == "context_or_output":
                labels.append("either the setting or how you want the answer delivered")
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

        return f"Coaching: add {detail_text}."

    def _merge_requirements(
        self,
        raw_prompt: str,
        conversation: Optional[ConversationState],
        enforcement_level: str,
    ) -> tuple[dict[str, ConversationRequirement], bool, Optional[str]]:
        existing = conversation.requirements if conversation is not None else {}
        evaluator_payload = self.structure_evaluator.evaluate(
            raw_prompt=raw_prompt,
            existing_requirements={
                field_name: requirement.model_dump() for field_name, requirement in existing.items()
            },
            enforcement_level=enforcement_level,
        )
        merged: dict[str, ConversationRequirement] = {}

        for field_name in REQUIREMENT_FIELDS:
            existing_requirement = existing.get(field_name)
            if existing_requirement and existing_requirement.value:
                merged[field_name] = existing_requirement
                continue
            evaluator_value = self._read_evaluator_requirement(evaluator_payload, field_name)
            if evaluator_value is not None:
                merged[field_name] = evaluator_value
                continue
            derived_value = self._derive_requirement(field_name, raw_prompt)
            if derived_value:
                merged[field_name] = ConversationRequirement(value=derived_value, status="derived")
            else:
                merged[field_name] = ConversationRequirement(value=None, status="missing")

        evaluator_used = evaluator_payload is not None
        evaluator_coaching_tip = None
        if evaluator_payload is not None:
            coaching_tip = evaluator_payload.get("coaching_tip")
            if isinstance(coaching_tip, str) and coaching_tip.strip():
                evaluator_coaching_tip = coaching_tip.strip()

        return merged, evaluator_used, evaluator_coaching_tip

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
        if status not in {"derived", "missing"}:
            return None
        if value is None:
            return ConversationRequirement(value=None, status="missing")
        if isinstance(value, str) and value.strip():
            return ConversationRequirement(value=value.strip(), status="derived")
        return ConversationRequirement(value=None, status="missing")

    def _derive_requirement(self, field_name: str, raw_prompt: str) -> Optional[str]:
        text = raw_prompt.strip()
        if not text:
            return None

        if field_name == "task":
            if text.endswith("?") or len(text.split()) >= 3:
                return text
            first_word = text.split()[0].lower()
            if first_word in TASK_STARTERS:
                return text
            return None

        if field_name == "who":
            for pattern in WHO_PATTERNS:
                match = pattern.search(text)
                if match:
                    return match.group(1).strip(" ,.")
            return None

        if field_name == "context":
            for pattern in CONTEXT_PATTERNS:
                match = pattern.search(text)
                if match:
                    return match.group(0).strip(" ,.")
            return None

        if field_name == "output":
            for pattern in OUTPUT_PATTERNS:
                match = pattern.search(text)
                if match:
                    return match.group(0).strip(" ,.")
            lower_text = text.lower()
            for keyword in OUTPUT_KEYWORDS:
                if keyword in lower_text:
                    return keyword
            return None

        return None

    def _missing_fields(
        self,
        requirements: dict[str, ConversationRequirement],
        enforcement_level: str,
    ) -> list[str]:
        if enforcement_level == "none":
            return []

        if enforcement_level == "low":
            return ["task"] if requirements["task"].status == "missing" else []

        if enforcement_level == "moderate":
            missing = []
            if requirements["task"].status == "missing":
                missing.append("task")
            if (
                requirements["context"].status == "missing"
                and requirements["output"].status == "missing"
            ):
                missing.append("context_or_output")
            return missing

        missing = [
            field_name
            for field_name in REQUIREMENT_FIELDS
            if requirements[field_name].status == "missing"
        ]
        return missing
