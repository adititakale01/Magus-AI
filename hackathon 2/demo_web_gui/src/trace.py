from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceStep:
    title: str
    summary: str | None = None
    data: Any | None = None


@dataclass
class RunTrace:
    steps: list[TraceStep] = field(default_factory=list)

    def add(self, title: str, *, summary: str | None = None, data: Any | None = None) -> None:
        self.steps.append(TraceStep(title=title, summary=summary, data=data))

