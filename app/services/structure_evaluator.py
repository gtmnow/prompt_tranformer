from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from app.core.config import get_settings


class StructureEvaluationService:
    def __init__(self) -> None:
        self.settings = get_settings()

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
        if not self.is_enabled():
            return None

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
            text = self._extract_output_text(response.json())
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                return None
            return parsed
        except (httpx.HTTPError, ValueError, json.JSONDecodeError):
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

        output = payload.get("output", [])
        if isinstance(output, list):
            for item in output:
                if item.get("type") != "message":
                    continue
                for content_item in item.get("content", []):
                    if content_item.get("type") == "output_text":
                        text_value = content_item.get("text", "")
                        if text_value:
                            return str(text_value).strip()
        return ""
