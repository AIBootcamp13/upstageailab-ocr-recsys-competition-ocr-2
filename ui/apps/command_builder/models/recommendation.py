from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class UseCaseRecommendation:
    """Structured recommendation tied to an OCR training use-case."""

    id: str
    title: str
    description: str
    architecture: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def iter_assignments(self) -> list[tuple[str, Any]]:
        return list(self.parameters.items())
