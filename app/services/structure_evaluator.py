from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import httpx

from app.core.config import get_settings
from app.core.logging import configure_application_logging


logger = logging.getLogger("prompt_transformer.structure_evaluator")


class StructureEvaluationService:
    def __init__(self) -> None:
        self.settings = get_settings()
        configure_application_logging(self.settings.log_level)

    def is_enabled(self) -> bool:
        return bool(
            self.settings.structure_evaluator_enabled
            and self.settings.structure_evaluator_api_key
            and self.settings.structure_evaluator_model
        )

    def evaluate(
        self,
        *,
        raw_prompt: str,
        enforcement_level: str,
    ) -> Optional[dict[str, Any]]:
        enabled = self.is_enabled()
        if not enabled:
            logger.info(
                "structure_evaluator_skipped enabled=%s flag=%s api_key_present=%s model=%s",
                enabled,
                self.settings.structure_evaluator_enabled,
                bool(self.settings.structure_evaluator_api_key),
                self.settings.structure_evaluator_model,
            )
            return None

        logger.info(
            "structure_evaluator_request model=%s base_url=%s prompt_chars=%s enforcement_level=%s",
            self.settings.structure_evaluator_model,
            self.settings.structure_evaluator_base_url,
            len(raw_prompt),
            enforcement_level,
        )

        payload = {
            "model": self.settings.structure_evaluator_model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self._build_system_prompt(),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(
                                {
                                    "prompt": raw_prompt,
                                    "enforcement_level": enforcement_level,
                                }
                            ),
                        }
                    ],
                },
            ],
            "temperature": 0,
            "max_output_tokens": 300,
            "store": False,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.structure_evaluator_api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=self.settings.structure_evaluator_timeout_seconds) as client:
                response = client.post(
                    f"{self.settings.structure_evaluator_base_url.rstrip('/')}/responses",
                    headers=headers,
                    json=payload,
                )
            response.raise_for_status()
            response_payload = response.json()
            text = self._extract_output_text(response_payload)
            if not text:
                logger.warning(
                    "structure_evaluator_empty_output status_code=%s response_keys=%s",
                    response.status_code,
                    sorted(response_payload.keys()) if isinstance(response_payload, dict) else None,
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
                response.status_code,
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
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "structure_evaluator_http_status_error status_code=%s response_text=%s",
                exc.response.status_code,
                exc.response.text[:500],
            )
            return None
        except httpx.HTTPError as exc:
            logger.warning(
                "structure_evaluator_http_error error=%s",
                str(exc),
            )
            return None
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
                    sorted(response_payload.keys())
                    if "response_payload" in locals() and isinstance(response_payload, dict)
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

    def _extract_output_text(self, payload: dict[str, Any]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        for key in ("text", "content"):
            value = payload.get(key)
            extracted = self._extract_text_value(value)
            if extracted:
                return extracted

        output = payload.get("output", [])
        if isinstance(output, list):
            for item in output:
                extracted = self._extract_text_value(item)
                if extracted:
                    return extracted
        return ""

    def _extract_text_value(self, value: Any) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for key in ("text", "output_text", "content"):
                extracted = self._extract_text_value(value.get(key))
                if extracted:
                    return extracted
            return ""
        if isinstance(value, list):
            for item in value:
                extracted = self._extract_text_value(item)
                if extracted:
                    return extracted
        return ""

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
