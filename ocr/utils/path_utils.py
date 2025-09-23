#!/usr/bin/env python3
"""
Path Utilities for OCR Project

This module provides centralized path resolution utilities for the OCR project.
It handles common path operations and ensures consistent path resolution across all scripts.
Enhanced with modular path configuration for better reusability.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class OCRPathConfig:
    """Configuration class for OCR project paths."""

    # Base directories
    project_root: Path
    data_dir: Path
    config_dir: Path
    output_dir: Path

    # Data subdirectories
    images_dir: Path
    annotations_dir: Path
    pseudo_labels_dir: Path

    # Output subdirectories
    logs_dir: Path
    checkpoints_dir: Path
    submissions_dir: Path

    # Model and config paths
    models_dir: Optional[Path] = None
    pretrained_models_dir: Optional[Path] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "OCRPathConfig":
        """Create OCRPathConfig from dictionary configuration."""
        return cls(
            project_root=Path(config.get("project_root", ".")),
            data_dir=Path(config.get("data_dir", "data")),
            config_dir=Path(config.get("config_dir", "configs")),
            output_dir=Path(config.get("output_dir", "outputs")),
            images_dir=Path(config.get("images_dir", "data/datasets/images")),
            annotations_dir=Path(config.get("annotations_dir", "data/datasets/jsons")),
            pseudo_labels_dir=Path(
                config.get("pseudo_labels_dir", "data/pseudo_label")
            ),
            logs_dir=Path(config.get("logs_dir", "outputs/logs")),
            checkpoints_dir=Path(config.get("checkpoints_dir", "outputs/checkpoints")),
            submissions_dir=Path(config.get("submissions_dir", "outputs/submissions")),
            models_dir=(
                Path(config.get("models_dir", "models"))
                if config.get("models_dir")
                else None
            ),
            pretrained_models_dir=(
                Path(config.get("pretrained_models_dir", "pretrained"))
                if config.get("pretrained_models_dir")
                else None
            ),
        )

    def resolve_path(self, path: Union[str, Path], base: Optional[Path] = None) -> Path:
        """Resolve a path relative to a base directory or project root."""
        path = Path(path)

        if path.is_absolute():
            return path

        if base is None:
            base = self.project_root

        return base / path

    def ensure_directories(self) -> None:
        """Create all necessary directories."""
        directories = [
            self.data_dir,
            self.config_dir,
            self.output_dir,
            self.images_dir,
            self.annotations_dir,
            self.pseudo_labels_dir,
            self.logs_dir,
            self.checkpoints_dir,
            self.submissions_dir,
        ]

        if self.models_dir:
            directories.append(self.models_dir)
        if self.pretrained_models_dir:
            directories.append(self.pretrained_models_dir)

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


class OCRPathResolver:
    """Central path resolution manager for OCR project."""

    def __init__(self, config: Optional[OCRPathConfig] = None):
        self.config = config or self._create_default_config()

    def _create_default_config(self) -> OCRPathConfig:
        """Create default path configuration for OCR project."""
        # Try to detect project root
        current_path = Path.cwd()

        # Look for common project markers
        project_markers = ["pyproject.toml", "requirements.txt", "setup.py", ".git"]

        project_root = current_path
        for parent in [current_path] + list(current_path.parents):
            if any((parent / marker).exists() for marker in project_markers):
                project_root = parent
                break

        return OCRPathConfig(
            project_root=project_root,
            data_dir=project_root / "data",
            config_dir=project_root / "configs",
            output_dir=project_root / "outputs",
            images_dir=project_root / "data" / "datasets" / "images",
            annotations_dir=project_root / "data" / "datasets" / "jsons",
            pseudo_labels_dir=project_root / "data" / "pseudo_label",
            logs_dir=project_root / "outputs" / "logs",
            checkpoints_dir=project_root / "outputs" / "checkpoints",
            submissions_dir=project_root / "outputs" / "submissions",
        )

    def get_data_path(self, dataset: str, split: str = "train") -> Path:
        """Get path to dataset images."""
        return self.config.images_dir / dataset / split

    def get_annotation_path(self, dataset: str, split: str = "train") -> Path:
        """Get path to dataset annotations."""
        return self.config.annotations_dir / f"{split}.json"

    def get_checkpoint_path(self, experiment_name: str, version: str = "v1.0") -> Path:
        """Get path to experiment checkpoints."""
        return self.config.checkpoints_dir / experiment_name / version

    def get_log_path(self, experiment_name: str, version: str = "v1.0") -> Path:
        """Get path to experiment logs."""
        return self.config.logs_dir / experiment_name / version

    def get_submission_path(self, experiment_name: str) -> Path:
        """Get path to experiment submissions."""
        return self.config.submissions_dir / experiment_name

    def resolve_relative_path(
        self, path: Union[str, Path], base: Optional[str] = None
    ) -> Path:
        """Resolve a path that might be relative to different bases."""
        path = Path(path)

        if path.is_absolute():
            return path

        # Handle common relative path patterns
        if base == "project":
            return self.config.project_root / path
        elif base == "data":
            return self.config.data_dir / path
        elif base == "config":
            return self.config.config_dir / path
        elif base == "output":
            return self.config.output_dir / path
        else:
            # Default to project root
            return self.config.project_root / path

    @classmethod
    def from_environment(cls) -> "OCRPathResolver":
        """Create OCRPathResolver from environment variables."""
        config_dict = {}

        # Check for environment variables
        env_mappings = {
            "OCR_PROJECT_ROOT": "project_root",
            "OCR_DATA_DIR": "data_dir",
            "OCR_CONFIG_DIR": "config_dir",
            "OCR_OUTPUT_DIR": "output_dir",
            "OCR_IMAGES_DIR": "images_dir",
            "OCR_ANNOTATIONS_DIR": "annotations_dir",
            "OCR_LOGS_DIR": "logs_dir",
            "OCR_CHECKPOINTS_DIR": "checkpoints_dir",
            "OCR_SUBMISSIONS_DIR": "submissions_dir",
        }

        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                config_dict[config_key] = os.environ[env_var]

        if config_dict:
            path_config = OCRPathConfig.from_dict(config_dict)
            return cls(path_config)
        else:
            return cls()


class PathUtils:
    """Legacy PathUtils class for backward compatibility."""

    # Environment variable for project root
    PROJECT_ROOT_ENV = "OCR_PROJECT_ROOT"

    @classmethod
    def get_project_root(cls) -> Path:
        """
        Get the project root directory.

        Priority order:
        1. Environment variable OCR_PROJECT_ROOT
        2. Auto-detect by searching for a marker file/folder
        """
        # Check environment variable first
        env_root = os.getenv(cls.PROJECT_ROOT_ENV)
        if env_root:
            return Path(env_root).resolve()

        # Auto-detect by walking up the directory tree
        marker_files = [".git", "pyproject.toml", "src", "ocr"]
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            for marker in marker_files:
                if (parent / marker).exists():
                    return parent
        # Fallback
        return current.parent.parent.parent

    @classmethod
    def get_data_path(cls) -> Path:
        """Get the data directory path."""
        return cls.get_project_root() / "data"

    @classmethod
    def get_config_path(cls) -> Path:
        """Get the configs directory path."""
        return cls.get_project_root() / "configs"

    @classmethod
    def get_outputs_path(cls) -> Path:
        """Get the outputs directory path."""
        return cls.get_project_root() / "outputs"

    @classmethod
    def get_images_path(cls) -> Path:
        """Get the images directory path."""
        return cls.get_data_path() / "datasets" / "images"

    @classmethod
    def get_annotations_path(cls) -> Path:
        """Get the annotations directory path."""
        return cls.get_data_path() / "datasets" / "jsons"

    @classmethod
    def get_pseudo_labels_path(cls) -> Path:
        """Get the pseudo labels directory path."""
        return cls.get_data_path() / "pseudo_label"

    @classmethod
    def get_logs_path(cls) -> Path:
        """Get the logs directory path."""
        return cls.get_outputs_path() / "logs"

    @classmethod
    def get_checkpoints_path(cls) -> Path:
        """Get the checkpoints directory path."""
        return cls.get_outputs_path() / "checkpoints"

    @classmethod
    def get_submissions_path(cls) -> Path:
        """Get the submissions directory path."""
        return cls.get_outputs_path() / "submissions"

    @classmethod
    def add_src_to_sys_path(cls) -> None:
        """Add the src directory to sys.path if not already present."""
        src_path = cls.get_project_root() / "ocr"
        src_str = str(src_path)

        if src_str not in sys.path:
            sys.path.insert(0, src_str)

    @classmethod
    def ensure_project_root_env(cls) -> None:
        """Ensure the PROJECT_ROOT environment variable is set."""
        if not os.getenv(cls.PROJECT_ROOT_ENV):
            project_root = cls.get_project_root()
            os.environ[cls.PROJECT_ROOT_ENV] = str(project_root)

    @classmethod
    def validate_paths(cls) -> dict:
        """
        Validate that all expected paths exist.

        Returns:
            Dict with path validation results
        """
        paths_to_check = {
            "project_root": cls.get_project_root(),
            "data": cls.get_data_path(),
            "configs": cls.get_config_path(),
            "outputs": cls.get_outputs_path(),
            "images": cls.get_images_path(),
            "annotations": cls.get_annotations_path(),
            "logs": cls.get_logs_path(),
            "checkpoints": cls.get_checkpoints_path(),
            "submissions": cls.get_submissions_path(),
        }

        results = {}
        for name, path in paths_to_check.items():
            results[name] = {
                "path": path,
                "exists": path.exists(),
                "is_dir": path.is_dir() if path.exists() else False,
                "is_file": path.is_file() if path.exists() else False,
            }

        return results


# Global path resolver instance
_ocr_path_resolver = OCRPathResolver()


def get_path_resolver() -> OCRPathResolver:
    """Get the global OCR path resolver instance."""
    return _ocr_path_resolver


def setup_project_paths(config: Optional[Dict[str, Any]] = None) -> OCRPathResolver:
    """Setup project paths and return resolver.

    Args:
        config: Optional configuration dictionary with path settings

    Returns:
        Configured OCRPathResolver instance
    """
    global _ocr_path_resolver

    if config:
        path_config = OCRPathConfig.from_dict(config)
        _ocr_path_resolver = OCRPathResolver(path_config)
    else:
        _ocr_path_resolver = OCRPathResolver.from_environment()

    # Ensure all directories exist
    _ocr_path_resolver.config.ensure_directories()

    return _ocr_path_resolver


# Convenience functions for backward compatibility and easy importing
def get_project_root() -> Path:
    """Get the project root directory."""
    return PathUtils.get_project_root()


def get_data_path() -> Path:
    """Get the data directory path."""
    return PathUtils.get_data_path()


def get_config_path() -> Path:
    """Get the configs directory path."""
    return PathUtils.get_config_path()


def get_outputs_path() -> Path:
    """Get the outputs directory path."""
    return PathUtils.get_outputs_path()


def get_images_path() -> Path:
    """Get the images directory path."""
    return PathUtils.get_images_path()


def get_annotations_path() -> Path:
    """Get the annotations directory path."""
    return PathUtils.get_annotations_path()


def get_logs_path() -> Path:
    """Get the logs directory path."""
    return PathUtils.get_logs_path()


def get_checkpoints_path() -> Path:
    """Get the checkpoints directory path."""
    return PathUtils.get_checkpoints_path()


def get_submissions_path() -> Path:
    """Get the submissions directory path."""
    return PathUtils.get_submissions_path()


def add_src_to_sys_path() -> None:
    """Add the src directory to sys.path if not already present."""
    PathUtils.add_src_to_sys_path()


def ensure_project_root_env() -> None:
    """Ensure the PROJECT_ROOT environment variable is set."""
    PathUtils.ensure_project_root_env()


def validate_paths() -> dict:
    """Validate that all expected paths exist."""
    return PathUtils.validate_paths()
