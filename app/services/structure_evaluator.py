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
        existing_requirements: dict[str, dict[str, Any]],
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
                                    "existing_requirements": existing_requirements,
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
            "For each field except coaching_tip, return an object with keys value and status. "
            "status must be 'derived' or 'missing'. "
            "Only mark a field as derived when the prompt text itself gives clear evidence for it. "
            "Do not invent defaults, preferences, audiences, formats, or personas. "
            "If the prompt is generic, leave missing fields as missing. "
            "For enforcement_level='full', be strict: "
            "who requires an explicit role or persona, "
            "context requires an explicit audience, purpose, setting, or intended use, "
            "and output requires an explicit format, channel, file type, or delivery instruction. "
            "For enforcement_level='moderate', you may accept clearly implied context or output, but not guesses. "
            "For enforcement_level='low', require only the task and leave other fields missing unless clearly stated. "
            "Examples: "
            "\"tell me a joke\" => task derived; who/context/output missing. "
            "\"you are telling jokes at a kids birthday party, and just give me the joke in the chat\" "
            "=> who/task/context/output derived. "
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
