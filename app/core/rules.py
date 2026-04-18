from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


RULES_DIR = Path(__file__).resolve().parent.parent / "rules"


@dataclass(frozen=True)
class RuleRegistry:
    summary_personas: dict[str, Any]
    llm_policies: dict[str, Any]
    task_rules: dict[str, Any]
    prompt_scoring: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Rule file {path} must contain a mapping")
    return data


@lru_cache(maxsize=1)
def get_rule_registry() -> RuleRegistry:
    return RuleRegistry(
        summary_personas=_load_yaml(RULES_DIR / "summary_personas.yaml"),
        llm_policies=_load_yaml(RULES_DIR / "llm_policies.yaml"),
        task_rules=_load_yaml(RULES_DIR / "task_rules.yaml"),
        prompt_scoring=_load_yaml(RULES_DIR / "prompt_scoring.yaml"),
    )
