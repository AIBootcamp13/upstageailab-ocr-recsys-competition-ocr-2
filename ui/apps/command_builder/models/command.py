from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass(slots=True)
class GeneratedCommand:
    """Tracks the UI-facing state of a generated CLI command."""

    generated: str = ""
    edited: str = ""
    overrides: list[str] = field(default_factory=list)
    constant_overrides: list[str] = field(default_factory=list)
    values: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    validation_error: str | None = None

    def clear(self) -> None:
        self.generated = ""
        self.edited = ""
        self.overrides.clear()
        self.constant_overrides.clear()
        self.values.clear()
        self.errors.clear()
        self.validation_error = None

    @property
    def has_command(self) -> bool:
        return bool(self.generated or self.edited)


@dataclass(slots=True)
class ExecutionResult:
    """Holds the outcome of a command execution."""

    status: Literal["idle", "running", "success", "error"] = "idle"
    started_at: datetime | None = None
    duration: float | None = None
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    summary: str | None = None

    def mark_running(self) -> None:
        self.status = "running"
        self.started_at = datetime.utcnow()
        self.duration = None
        self.return_code = None
        self.stdout = ""
        self.stderr = ""
        self.summary = None

    def mark_finished(self, return_code: int, duration: float, stdout: str, stderr: str) -> None:
        self.return_code = return_code
        self.duration = duration
        self.stdout = stdout
        self.stderr = stderr
        self.status = "success" if return_code == 0 else "error"

    @property
    def succeeded(self) -> bool:
        return self.status == "success"


@dataclass(slots=True)
class CommandPageData:
    """Aggregates command-generation state for a specific page."""

    generated: GeneratedCommand = field(default_factory=GeneratedCommand)
    execution: ExecutionResult = field(default_factory=ExecutionResult)
