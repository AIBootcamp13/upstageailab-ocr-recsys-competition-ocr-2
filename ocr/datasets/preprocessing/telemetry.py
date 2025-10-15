"""
Structured Telemetry and Logging for Preprocessing Pipeline.

This module provides structured logging and telemetry utilities for the
preprocessing viewer pipeline, enabling observability and debugging of
geometry validation and correction decisions.
"""

import json
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TelemetryEvent(BaseModel):
    """Structured telemetry event for preprocessing operations."""

    event_type: str = Field(description="Type of telemetry event")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    stage: str = Field(description="Processing stage where event occurred")
    success: bool = Field(description="Whether the operation succeeded")
    confidence: float | None = Field(default=None, description="Confidence score if applicable")
    issues: list[str] = Field(default_factory=list, description="Issues encountered")
    fallback_action: str | None = Field(default=None, description="Fallback action taken")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Additional metrics")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_log_dict(self) -> dict[str, Any]:
        """Convert to dictionary suitable for structured logging."""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "stage": self.stage,
            "success": self.success,
            "confidence": self.confidence,
            "issues": self.issues,
            "fallback_action": self.fallback_action,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string for logging."""
        return json.dumps(self.to_log_dict(), default=str)


class PreprocessingTelemetry:
    """
    Structured telemetry and logging for preprocessing pipeline operations.

    This class provides methods to log and track preprocessing decisions,
    geometry validation results, and fallback actions with structured data.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self.events: list[TelemetryEvent] = []

    def log_geometry_validation(self, stage: str, validation_result: Any, corners: Any, image_shape: tuple[int, int]) -> None:
        """Log geometry validation results with structured telemetry."""
        event = TelemetryEvent(
            event_type="geometry_validation",
            stage=stage,
            success=validation_result.is_valid,
            confidence=validation_result.confidence,
            issues=validation_result.issues,
            fallback_action=validation_result.fallback_recommendation,
            metrics={
                "corner_count": len(corners) if hasattr(corners, "__len__") else 0,
                "image_width": image_shape[1],
                "image_height": image_shape[0],
                **validation_result.metrics,
            },
            metadata={
                "validation_config": validation_result.metadata.get("config", {}) if validation_result.metadata else {},
            },
        )

        self.events.append(event)
        self._log_event(event)

    def log_processing_decision(
        self,
        stage: str,
        decision: str,
        success: bool,
        confidence: float | None = None,
        issues: list[str] | None = None,
        fallback_action: str | None = None,
        **kwargs,
    ) -> None:
        """Log processing decisions with structured telemetry."""
        event = TelemetryEvent(
            event_type="processing_decision",
            stage=stage,
            success=success,
            confidence=confidence,
            issues=issues or [],
            fallback_action=fallback_action,
            metadata=kwargs,
        )

        self.events.append(event)
        self._log_event(event)

    def log_correction_attempt(
        self,
        stage: str,
        correction_type: str,
        success: bool,
        error_message: str | None = None,
        processing_time_ms: float | None = None,
        **kwargs,
    ) -> None:
        """Log correction attempts with outcomes."""
        event = TelemetryEvent(
            event_type="correction_attempt",
            stage=stage,
            success=success,
            issues=[error_message] if error_message else [],
            metrics={
                "correction_type": correction_type,
                "processing_time_ms": processing_time_ms,
            },
            metadata=kwargs,
        )

        self.events.append(event)
        self._log_event(event)

    def log_fallback_action(self, stage: str, fallback_type: str, reason: str, success: bool = True, **kwargs) -> None:
        """Log fallback actions taken."""
        event = TelemetryEvent(
            event_type="fallback_action", stage=stage, success=success, fallback_action=fallback_type, issues=[reason], metadata=kwargs
        )

        self.events.append(event)
        self._log_event(event)

    def get_events_summary(self) -> dict[str, Any]:
        """Get summary of all logged events."""
        total_events = len(self.events)
        successful_events = sum(1 for e in self.events if e.success)
        failed_events = total_events - successful_events

        event_types: dict[str, int] = {}
        stages: dict[str, int] = {}
        fallback_actions: dict[str, int] = {}

        for event in self.events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            stages[event.stage] = stages.get(event.stage, 0) + 1
            if event.fallback_action:
                fallback_actions[event.fallback_action] = fallback_actions.get(event.fallback_action, 0) + 1

        return {
            "total_events": total_events,
            "successful_events": successful_events,
            "failed_events": failed_events,
            "event_types": event_types,
            "stages": stages,
            "fallback_actions": fallback_actions,
        }

    def _log_event(self, event: TelemetryEvent) -> None:
        """Log event using appropriate logging level."""
        log_message = f"[{event.stage}] {event.event_type}: {'SUCCESS' if event.success else 'FAILED'}"

        if event.confidence is not None:
            log_message += f" (confidence: {event.confidence:.3f})"

        if event.issues:
            log_message += f" - Issues: {', '.join(event.issues)}"

        if event.fallback_action:
            log_message += f" - Fallback: {event.fallback_action}"

        # Use appropriate log level
        if not event.success and event.issues:
            self.logger.warning(log_message)
        elif event.fallback_action:
            self.logger.info(log_message)
        else:
            self.logger.debug(log_message)

        # Always log structured data at debug level
        self.logger.debug(f"Structured telemetry: {event.to_json()}")
