from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from app.core.config import get_settings
from app.core.logging import configure_application_logging
from app.services.llm_gateway import LlmGatewayService
from app.services.runtime_llm import RuntimeLlmConfig
from app.services.llm_types import TransformerLlmRequest


logger = logging.getLogger("prompt_transformer.structure_evaluator")


class StructureEvaluationService:
    def __init__(self) -> None:
        self.settings = get_settings()
        configure_application_logging(self.settings.log_level)
        self.gateway = LlmGatewayService()

    def is_enabled(self) -> bool:
        return bool(self.settings.structure_evaluator_enabled)

    def evaluate(
        self,
        *,
        raw_prompt: str,
        enforcement_level: str,
        runtime_config: RuntimeLlmConfig | None = None,
    ) -> Optional[dict[str, Any]]:
        enabled = self.is_enabled()
        if not enabled or runtime_config is None:
            logger.info(
                "structure_evaluator_skipped enabled=%s flag=%s runtime_config_present=%s model=%s",
                enabled,
                self.settings.structure_evaluator_enabled,
                runtime_config is not None,
                runtime_config.model if runtime_config is not None else None,
            )
            return None

        logger.info(
            "structure_evaluator_request model=%s base_url=%s prompt_chars=%s enforcement_level=%s",
            runtime_config.model,
            runtime_config.endpoint_url or self.settings.structure_evaluator_base_url,
            len(raw_prompt),
            enforcement_level,
        )

        try:
            gateway_request = TransformerLlmRequest(
                provider=runtime_config.provider,  # type: ignore[arg-type]
                model=runtime_config.model,
                base_url=runtime_config.endpoint_url or self.settings.structure_evaluator_base_url,
                api_key=runtime_config.api_key,
                system_prompt=self._build_system_prompt(),
                user_prompt=json.dumps(
                    {
                        "prompt": raw_prompt,
                        "enforcement_level": enforcement_level,
                    }
                ),
                max_output_tokens=300,
                temperature=0,
                expected_output="json",
                timeout_seconds=self.settings.structure_evaluator_timeout_seconds,
            )
            response_payload, response_error = self.gateway.invoke(gateway_request)
            if response_error is not None:
                logger.warning(
                    "structure_evaluator_llm_error provider=%s model=%s code=%s status_code=%s message=%s",
                    response_error.provider,
                    response_error.model,
                    response_error.code,
                    response_error.status_code,
                    response_error.message,
                )
                return None
            if response_payload is None:
                logger.warning("structure_evaluator_empty_gateway_response")
                return None
            text = response_payload.output_text
            if not text:
                logger.warning(
                    "structure_evaluator_empty_output status_code=%s response_keys=%s",
                    response_payload.status_code,
                    (
                        sorted(response_payload.raw_payload.keys())
                        if isinstance(response_payload.raw_payload, dict)
                        else None
                    ),
                )
                return None
            parsed = self._parse_output_json(text)
            if not isinstance(parsed, dict):
                logger.warning(
                    "structure_evaluator_invalid_payload_type parsed_type=%s",
                    type(parsed).__name__,
                )
                return None
            logger.info(
                "structure_evaluator_success status_code=%s returned_fields=%s",
                response_payload.status_code,
                sorted(parsed.keys()),
            )
            field_diagnostics = {}
            for field_name in ("who", "task", "context", "output"):
                field_payload = parsed.get(field_name)
                if isinstance(field_payload, dict):
                    field_diagnostics[field_name] = {
                        "keys": sorted(field_payload.keys()),
                        "has_score": isinstance(field_payload.get("score"), (int, float))
                        and not isinstance(field_payload.get("score"), bool),
                        "status": field_payload.get("status"),
                    }
                else:
                    field_diagnostics[field_name] = {
                        "keys": None,
                        "has_score": False,
                        "status": None,
                    }
            logger.info(
                "structure_evaluator_field_diagnostics %s",
                field_diagnostics,
            )
            return parsed
        except json.JSONDecodeError as exc:
            logger.warning(
                "structure_evaluator_json_decode_error error=%s text_sample=%s",
                str(exc),
                self._truncate_for_log(text if "text" in locals() else None),
            )
            return None
        except ValueError as exc:
            logger.warning(
                "structure_evaluator_value_error error=%s text_sample=%s response_keys=%s",
                str(exc),
                self._truncate_for_log(text if "text" in locals() else None),
                (
                    sorted(response_payload.raw_payload.keys())
                    if "response_payload" in locals() and response_payload is not None and isinstance(response_payload.raw_payload, dict)
                    else None
                ),
            )
            return None

    def _build_system_prompt(self) -> str:
        return (
            "You extract prompt-structure fields from a user prompt. "
            "Return JSON only with keys who, task, context, output, coaching_tip. "
            "For each field except coaching_tip, return an object with keys value, status, and score. "
            "status must be 'present', 'derived', or 'missing'. "
            "score must be an integer from 0 to 25. "
            "'present' means the prompt clearly contains the element in natural language. "
            "'derived' means the element is inferable, but not clearly stated. "
            "'missing' means the prompt does not provide enough information. "
            "Use score to express nuance within each field: "
            "0 means absent, low single digits means weakly inferable, mid-teens means partially useful, "
            "and 25 means fully and clearly specified. "
            "A field can be status='present' without scoring 25 if it is somewhat vague; reserve 25 for strong clarity. "
            "A field should not score above 10 when it is only loosely implied. "
            "Evaluate only the current prompt text. Do not use prior conversation state, prior requirements, or inferred memory "
            "to increase any field status or score. "
            "Do not require labels like Who:, Task:, Context:, or Output: when deciding whether a field is present. "
            "Do not invent defaults, preferences, audiences, formats, or personas. "
            "If the prompt is generic, leave unsupported fields as missing. "
            "Use enforcement_level only to make coaching_tip appropriate in tone. "
            "Examples: "
            "\"tell me a joke\" => task present around 25; output derived around 5; who/context 0. "
            "\"you are telling jokes at a kids birthday party, and just give me the joke in the chat\" "
            "=> who/task/context/output present, each near 25. "
            "\"Explain rate limiting for a SaaS API. I am studying for a system design interview. "
            "Provide the answer in the chat with components, flow, tradeoffs, and one example.\" "
            "=> task/context/output present with strong scores; who 0. "
            "\"Explain how to design a REST API rate-limiting system for a SaaS application. "
            "I am preparing for a backend system design interview and need an answer that helps me understand the architecture, tradeoffs, and implementation choices clearly.\" "
            "=> task/context present with strong scores; who 0; output low or 0 depending on how explicit the response shape is. "
            "\"Who: You are a Senior Python software engineer. "
            "Task: Explain how to design a REST API rate-limiting system for a SaaS application. "
            "Context: I am preparing for a backend system design interview and need an answer that helps me understand the architecture, tradeoffs, and implementation choices clearly.\" "
            "=> who/task/context present with strong scores; output 0. "
            "\"Explain how to design a REST API rate-limiting system for a SaaS application.\" "
            "=> task present near 25; output derived around 5; who/context 0. "
            "Keep coaching_tip short, supportive, compact, and framed as coaching rather than a command."
        )


    def _parse_output_json(self, text: str) -> dict[str, Any]:
        candidates = [
            text.strip(),
            self._strip_code_fences(text),
            self._extract_json_object(text),
        ]
        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            try:
                parsed = json.loads(normalized)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("Unable to parse evaluator JSON payload.")

    def _strip_code_fences(self, text: str) -> str:
        fenced = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        fenced = re.sub(r"\s*```$", "", fenced)
        return fenced.strip()

    def _extract_json_object(self, text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return ""
        return text[start : end + 1].strip()

    def _truncate_for_log(self, value: str | None, limit: int = 500) -> str | None:
        if value is None:
            return None
        return value[:limit]
