from __future__ import annotations

from typing import Any, Literal, TypedDict

from app.services.llm_types import NormalizedTokenUsage


UsageCategory = Literal["admin", "final_response"]


class TokenUsageProviderEntry(TypedDict):
    category: str
    purpose: str
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    reasoning_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    raw_usage: dict[str, Any] | None


class TokenUsageAccumulator(TypedDict):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    calls: int


class TokenUsagePayload(TypedDict, total=False):
    admin: dict[str, Any]
    final_response: dict[str, Any]
    providers: list[TokenUsageProviderEntry]


def normalize_usage(provider: str, usage_payload: dict[str, Any] | None) -> NormalizedTokenUsage | None:
    if not isinstance(usage_payload, dict):
        return None

    input_tokens = _read_int(
        usage_payload,
        "input_tokens",
        "prompt_tokens",
        "prompt_token_count",
    )
    output_tokens = _read_int(
        usage_payload,
        "output_tokens",
        "completion_tokens",
        "output_token_count",
    )
    total_tokens = _read_int(usage_payload, "total_tokens")
    reasoning_tokens = _read_int(
        usage_payload,
        "reasoning_tokens",
        "output_tokens_details.reasoning_tokens",
    )
    cache_read_tokens = _read_int(
        usage_payload,
        "cache_read_tokens",
        "input_tokens_details.cached_tokens",
        "cache_creation_input_tokens",
    )
    cache_write_tokens = _read_int(
        usage_payload,
        "cache_write_tokens",
        "cache_creation_output_tokens",
    )

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    if all(
        value is None
        for value in (
            input_tokens,
            output_tokens,
            total_tokens,
            reasoning_tokens,
            cache_read_tokens,
            cache_write_tokens,
        )
    ):
        return None

    return NormalizedTokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        raw_usage=usage_payload,
    )


def build_usage_entry(
    *,
    category: UsageCategory,
    purpose: str,
    provider: str,
    model: str,
    usage: NormalizedTokenUsage | None,
) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "category": category,
        "purpose": purpose,
        "provider": provider,
        "model": model,
        "usage": usage.model_dump(),
    }


def merge_usage(
    existing_payload: dict[str, Any] | None,
    entry: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if entry is None:
        return existing_payload

    payload: TokenUsagePayload = _clone_payload(existing_payload)
    category = str(entry["category"])
    purpose = str(entry["purpose"])
    provider = str(entry["provider"])
    model = str(entry["model"])
    usage = entry["usage"]
    if not isinstance(usage, dict):
        return payload

    category_block = payload.setdefault(category, {})
    _merge_accumulator(category_block, usage)

    by_purpose = category_block.setdefault("by_purpose", {})
    purpose_block = by_purpose.setdefault(purpose, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0})
    _merge_accumulator(purpose_block, usage)

    providers = payload.setdefault("providers", [])
    provider_record: TokenUsageProviderEntry = {
        "category": category,
        "purpose": purpose,
        "provider": provider,
        "model": model,
        "input_tokens": _coerce_non_negative_int(usage.get("input_tokens")),
        "output_tokens": _coerce_non_negative_int(usage.get("output_tokens")),
        "total_tokens": _coerce_non_negative_int(usage.get("total_tokens")),
        "reasoning_tokens": _coerce_non_negative_int(usage.get("reasoning_tokens")),
        "cache_read_tokens": _coerce_non_negative_int(usage.get("cache_read_tokens")),
        "cache_write_tokens": _coerce_non_negative_int(usage.get("cache_write_tokens")),
        "raw_usage": usage.get("raw_usage") if isinstance(usage.get("raw_usage"), dict) else None,
    }
    replaced = False
    for index, existing in enumerate(providers):
        if (
            existing.get("category") == category
            and existing.get("purpose") == purpose
            and existing.get("provider") == provider
            and existing.get("model") == model
        ):
            providers[index] = provider_record
            replaced = True
            break
    if not replaced:
        providers.append(provider_record)

    return payload


def replace_category_usage(
    existing_payload: dict[str, Any] | None,
    category: UsageCategory,
    entry: dict[str, Any] | None,
) -> dict[str, Any] | None:
    payload = _clone_payload(existing_payload)
    payload.pop(category, None)

    providers = [
        item
        for item in payload.get("providers", [])
        if item.get("category") != category
    ]
    payload["providers"] = providers
    return merge_usage(payload, entry)


def _clone_payload(existing_payload: dict[str, Any] | None) -> TokenUsagePayload:
    if not isinstance(existing_payload, dict):
        return {"providers": []}

    payload: TokenUsagePayload = {
        "providers": list(existing_payload.get("providers") or []),
    }
    for category in ("admin", "final_response"):
        existing_category = existing_payload.get(category)
        if isinstance(existing_category, dict):
            payload[category] = {
                "input_tokens": int(existing_category.get("input_tokens") or 0),
                "output_tokens": int(existing_category.get("output_tokens") or 0),
                "total_tokens": int(existing_category.get("total_tokens") or 0),
                "calls": int(existing_category.get("calls") or 0),
                "by_purpose": {
                    key: {
                        "input_tokens": int((value or {}).get("input_tokens") or 0),
                        "output_tokens": int((value or {}).get("output_tokens") or 0),
                        "total_tokens": int((value or {}).get("total_tokens") or 0),
                        "calls": int((value or {}).get("calls") or 0),
                    }
                    for key, value in (existing_category.get("by_purpose") or {}).items()
                    if isinstance(value, dict)
                },
            }
    return payload


def _merge_accumulator(target: dict[str, Any], usage: dict[str, Any]) -> None:
    target["input_tokens"] = int(target.get("input_tokens") or 0) + int(usage.get("input_tokens") or 0)
    target["output_tokens"] = int(target.get("output_tokens") or 0) + int(usage.get("output_tokens") or 0)
    target["total_tokens"] = int(target.get("total_tokens") or 0) + int(usage.get("total_tokens") or 0)
    target["calls"] = int(target.get("calls") or 0) + 1


def _read_int(payload: dict[str, Any], *paths: str) -> int | None:
    for path in paths:
        current: Any = payload
        valid_path = True
        for segment in path.split("."):
            if not isinstance(current, dict):
                valid_path = False
                break
            current = current.get(segment)
        if not valid_path:
            continue
        value = _coerce_non_negative_int(current)
        if value is not None:
            return value
    return None


def _coerce_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        normalized = int(value)
        return normalized if normalized >= 0 else None
    return None
