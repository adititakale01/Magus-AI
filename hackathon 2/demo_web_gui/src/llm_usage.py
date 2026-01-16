from __future__ import annotations

from typing import Any


def usage_from_response(resp: Any, *, model: str | None = None, calls: int = 1) -> dict[str, Any]:
    usage = getattr(resp, "usage", None)
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or 0
        total_tokens = usage.get("total_tokens") or 0
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage is not None else 0
        total_tokens = getattr(usage, "total_tokens", 0) if usage is not None else 0

    if not total_tokens and prompt_tokens and completion_tokens:
        total_tokens = int(prompt_tokens) + int(completion_tokens)

    out: dict[str, Any] = {
        "calls": int(calls),
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }
    if model:
        out["model"] = str(model)
    return out


def sum_usage(usages: list[dict[str, Any] | None], *, model: str | None = None) -> dict[str, Any] | None:
    total_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    for usage in usages:
        if not usage:
            continue
        total_calls += int(usage.get("calls") or 0)
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)

    if total_calls == 0 and prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        return None

    out: dict[str, Any] = {
        "calls": int(total_calls),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
    }
    if model:
        out["model"] = str(model)
    return out

