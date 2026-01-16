from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceStep:
    title: str
    summary: str | None = None
    data: Any | None = None
    used_llm: bool = False
    llm_usage: dict[str, Any] | None = None


@dataclass
class LlmUsageTotals:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: dict[str, Any] | None) -> None:
        if not usage:
            return
        self.calls += int(usage.get("calls") or 0)
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or 0)


@dataclass
class RunTrace:
    steps: list[TraceStep] = field(default_factory=list)
    llm_usage: LlmUsageTotals = field(default_factory=LlmUsageTotals)

    def add(
        self,
        title: str,
        *,
        summary: str | None = None,
        data: Any | None = None,
        used_llm: bool = False,
        llm_usage: dict[str, Any] | None = None,
    ) -> None:
        self.steps.append(
            TraceStep(
                title=title,
                summary=summary,
                data=data,
                used_llm=bool(used_llm or bool(llm_usage)),
                llm_usage=llm_usage,
            )
        )
        if llm_usage:
            self.llm_usage.add(llm_usage)
