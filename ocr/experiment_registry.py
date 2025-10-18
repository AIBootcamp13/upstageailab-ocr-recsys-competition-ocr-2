"""
Experiment Registry System

Provides structured experiment management with unique IDs and metadata tracking.
Replaces fragile experiment-name-based system with reliable ID-based organization.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass
class ExperimentMetadata:
    """Metadata for a single experiment."""

    id: str
    name: str
    description: str | None = None
    created_at: str | None = None
    config: dict[str, Any] | None = None
    tags: list[str] | None = None
    status: str = "active"  # active, completed, failed, archived

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []


class ExperimentRegistry:
    """Thread-safe registry for managing experiments with unique IDs."""

    def __init__(self, registry_path: str = "outputs/experiments/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._experiments: dict[str, ExperimentMetadata] = {}
        self._load_registry()

    def _load_registry(self):
        """Load experiment registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)
                    self._experiments = {exp_id: ExperimentMetadata(**exp_data) for exp_id, exp_data in data.items()}
            except (json.JSONDecodeError, KeyError):
                # Initialize empty registry if corrupted
                self._experiments = {}

    def _save_registry(self):
        """Save experiment registry to disk."""
        data = {exp_id: asdict(exp) for exp_id, exp in self._experiments.items()}
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def create_experiment(
        self, name: str, description: str | None = None, config: dict[str, Any] | None = None, tags: list[str] | None = None
    ) -> str:
        """Create a new experiment with unique ID."""
        with self._lock:
            # Generate unique ID (simple incrementing counter)
            existing_ids = [int(exp_id.split("_")[1]) for exp_id in self._experiments.keys() if exp_id.startswith("exp_")]
            next_id = max(existing_ids) + 1 if existing_ids else 1
            exp_id = f"exp_{next_id:03d}"

            experiment = ExperimentMetadata(id=exp_id, name=name, description=description, config=config, tags=tags or [])

            self._experiments[exp_id] = experiment
            self._save_registry()
            return exp_id

    def get_experiment(self, exp_id: str) -> ExperimentMetadata | None:
        """Get experiment metadata by ID."""
        return self._experiments.get(exp_id)

    def update_experiment(self, exp_id: str, **updates):
        """Update experiment metadata."""
        with self._lock:
            if exp_id in self._experiments:
                exp = self._experiments[exp_id]
                for key, value in updates.items():
                    if hasattr(exp, key):
                        setattr(exp, key, value)
                self._save_registry()

    def list_experiments(self, status: str | None = None, tags: list[str] | None = None) -> list[ExperimentMetadata]:
        """List experiments with optional filtering."""
        experiments = list(self._experiments.values())

        if status:
            experiments = [exp for exp in experiments if exp.status == status]

        if tags:
            experiments = [exp for exp in experiments if exp.tags and any(tag in exp.tags for tag in tags)]

        return sorted(experiments, key=lambda x: x.created_at or "", reverse=True)

    def get_experiment_path(self, exp_id: str) -> Path:
        """Get the base path for an experiment."""
        return Path(f"outputs/experiments/{exp_id}")

    def get_checkpoint_path(self, exp_id: str) -> Path:
        """Get the checkpoint directory for an experiment."""
        return self.get_experiment_path(exp_id) / "checkpoints"

    def get_log_path(self, exp_id: str) -> Path:
        """Get the log directory for an experiment."""
        return self.get_experiment_path(exp_id) / "logs"


# Global registry instance
_registry = None


def get_registry() -> ExperimentRegistry:
    """Get the global experiment registry instance."""
    global _registry
    if _registry is None:
        _registry = ExperimentRegistry()
    return _registry
