from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ui.utils.config_parser import ConfigParser

from ..models.recommendation import UseCaseRecommendation

SUPPORTED_ASSIGNMENT_KEYS = {
    "architecture",
    "encoder",
    "decoder",
    "head",
    "loss",
    "optimizer",
    "learning_rate",
    "batch_size",
    "max_epochs",
    "preprocessing_profile",
    "precision",
    "accumulate_grad_batches",
    "gradient_clip_val",
}


@dataclass(slots=True)
class UseCaseRecommendationService:
    config_parser: ConfigParser

    def __init__(self, config_parser: ConfigParser | None = None) -> None:
        self.config_parser = config_parser or ConfigParser()

    def for_architecture(self, architecture: str | None) -> list[UseCaseRecommendation]:
        metadata = self.config_parser.get_architecture_metadata()
        if architecture and architecture in metadata:
            payload = metadata[architecture]["ui_metadata"].get("use_cases", [])
            return self._materialise(payload, fallback_architecture=architecture)

        # When no architecture is selected yet, surface the leading recommendation for each architecture.
        recommendations: list[UseCaseRecommendation] = []
        for arch_name, info in metadata.items():
            cases = info.get("ui_metadata", {}).get("use_cases", [])
            if not cases:
                continue
            first = self._materialise(cases[:1], fallback_architecture=arch_name)
            recommendations.extend(first)
        return recommendations

    def _materialise(
        self,
        cases: Iterable[dict[str, Any]],
        fallback_architecture: str | None = None,
    ) -> list[UseCaseRecommendation]:
        materialised: list[UseCaseRecommendation] = []
        for case in cases:
            if not isinstance(case, dict):
                continue
            recommendations = case.get("recommendations") or {}
            if not isinstance(recommendations, dict):
                recommendations = {}
            assignments = {
                key: value for key, value in recommendations.items() if key in SUPPORTED_ASSIGNMENT_KEYS and value not in (None, "")
            }
            architecture = recommendations.get("architecture") or fallback_architecture
            materialised.append(
                UseCaseRecommendation(
                    id=str(case.get("id", fallback_architecture or "use_case")),
                    title=str(case.get("title", "Suggested setup")),
                    description=str(case.get("description", "")),
                    architecture=str(architecture) if architecture else None,
                    parameters=assignments,
                )
            )
        return materialised
