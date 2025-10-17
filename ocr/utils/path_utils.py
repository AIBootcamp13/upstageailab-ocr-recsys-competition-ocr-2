#!/usr/bin/env python3
"""
Path Utilities for OCR Project

This module provides centralized path resolution utilities for the OCR project.
It handles common path operations and ensures consistent path resolution across all scripts.
Enhanced with modular path configuration for better reusability.

NEW: For simple path setup in any script, use:

    from ocr.utils.path_utils import setup_paths
    setup_paths()

This automatically handles sys.path setup and environment variables.
"""

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    models_dir: Path | None = None
    pretrained_models_dir: Path | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "OCRPathConfig":
        """Create OCRPathConfig from dictionary configuration."""
        return cls(
            project_root=Path(config.get("project_root", ".")),
            data_dir=Path(config.get("data_dir", "data")),
            config_dir=Path(config.get("config_dir", "configs")),
            output_dir=Path(config.get("output_dir", "outputs")),
            images_dir=Path(config.get("images_dir", "data/datasets/images")),
            annotations_dir=Path(config.get("annotations_dir", "data/datasets/jsons")),
            pseudo_labels_dir=Path(config.get("pseudo_labels_dir", "data/pseudo_label")),
            logs_dir=Path(config.get("logs_dir", "outputs/logs")),
            checkpoints_dir=Path(config.get("checkpoints_dir", "outputs/checkpoints")),
            submissions_dir=Path(config.get("submissions_dir", "outputs/submissions")),
            models_dir=(Path(config.get("models_dir", "models")) if config.get("models_dir") else None),
            pretrained_models_dir=(
                Path(config.get("pretrained_models_dir", "pretrained")) if config.get("pretrained_models_dir") else None
            ),
        )

    def resolve_path(self, path: str | Path, base: Path | None = None) -> Path:
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

    def __init__(self, config: OCRPathConfig | None = None):
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

    def resolve_relative_path(self, path: str | Path, base: str | None = None) -> Path:
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
    """
    DEPRECATED: This class is for backward compatibility only.
    Please use the `get_path_resolver()` instance for all new code.
    """

    # Environment variable for project root
    PROJECT_ROOT_ENV = "OCR_PROJECT_ROOT"

    @classmethod
    def get_project_root(cls) -> Path:
        """
        DEPRECATED: Use `get_path_resolver().config.project_root` instead.
        Get the project root directory.

        Priority order:
        1. Environment variable OCR_PROJECT_ROOT
        2. Auto-detect by searching for a marker file/folder
        """
        warnings.warn(
            "PathUtils.get_project_root() is deprecated. Use `get_path_resolver().config.project_root` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        """DEPRECATED: Use `get_path_resolver().config.data_dir` instead."""
        warnings.warn(
            "PathUtils.get_data_path() is deprecated. Use `get_path_resolver().config.data_dir` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.get_project_root() / "data"

    @classmethod
    def get_config_path(cls) -> Path:
        """DEPRECATED: Use `get_path_resolver().config.config_dir` instead."""
        warnings.warn(
            "PathUtils.get_config_path() is deprecated. Use `get_path_resolver().config.config_dir` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.get_project_root() / "configs"

    @classmethod
    def get_outputs_path(cls) -> Path:
        """DEPRECATED: Use `get_path_resolver().config.output_dir` instead."""
        warnings.warn(
            "PathUtils.get_outputs_path() is deprecated. Use `get_path_resolver().config.output_dir` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.get_project_root() / "outputs"

    @classmethod
    def get_images_path(cls) -> Path:
        """DEPRECATED: Use `get_path_resolver().config.images_dir` instead."""
        warnings.warn(
            "PathUtils.get_images_path() is deprecated. Use `get_path_resolver().config.images_dir` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.get_data_path() / "datasets" / "images"

    @classmethod
    def get_annotations_path(cls) -> Path:
        """DEPRECATED: Use `get_path_resolver().config.annotations_dir` instead."""
        warnings.warn(
            "PathUtils.get_annotations_path() is deprecated. Use `get_path_resolver().config.annotations_dir` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.get_data_path() / "datasets" / "jsons"

    @classmethod
    def get_pseudo_labels_path(cls) -> Path:
        """DEPRECATED: Use `get_path_resolver().config.pseudo_labels_dir` instead."""
        warnings.warn(
            "PathUtils.get_pseudo_labels_path() is deprecated. Use `get_path_resolver().config.pseudo_labels_dir` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.get_data_path() / "pseudo_label"

    @classmethod
    def get_logs_path(cls) -> Path:
        """DEPRECATED: Use `get_path_resolver().config.logs_dir` instead."""
        warnings.warn(
            "PathUtils.get_logs_path() is deprecated. Use `get_path_resolver().config.logs_dir` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.get_outputs_path() / "logs"

    @classmethod
    def get_checkpoints_path(cls) -> Path:
        """DEPRECATED: Use `get_path_resolver().config.checkpoints_dir` instead."""
        warnings.warn(
            "PathUtils.get_checkpoints_path() is deprecated. Use `get_path_resolver().config.checkpoints_dir` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.get_outputs_path() / "checkpoints"

    @classmethod
    def get_submissions_path(cls) -> Path:
        """DEPRECATED: Use `get_path_resolver().config.submissions_dir` instead."""
        warnings.warn(
            "PathUtils.get_submissions_path() is deprecated. Use `get_path_resolver().config.submissions_dir` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.get_outputs_path() / "submissions"

    @classmethod
    def add_src_to_sys_path(cls) -> None:
        """DEPRECATED: Use `setup_paths()` function instead."""
        warnings.warn(
            "PathUtils.add_src_to_sys_path() is deprecated. Use `setup_paths()` function instead.", DeprecationWarning, stacklevel=2
        )
        src_path = cls.get_project_root() / "ocr"
        src_str = str(src_path)

        if src_str not in sys.path:
            sys.path.insert(0, src_str)

    @classmethod
    def ensure_project_root_env(cls) -> None:
        """DEPRECATED: Use `setup_paths()` function instead."""
        warnings.warn(
            "PathUtils.ensure_project_root_env() is deprecated. Use `setup_paths()` function instead.", DeprecationWarning, stacklevel=2
        )
        if not os.getenv(cls.PROJECT_ROOT_ENV):
            project_root = cls.get_project_root()
            os.environ[cls.PROJECT_ROOT_ENV] = str(project_root)

    @classmethod
    def validate_paths(cls) -> dict:
        """
        DEPRECATED: Use `get_path_resolver().validate_paths()` instead.

        Validate that all expected paths exist.

        Returns:
            Dict with path validation results
        """
        warnings.warn(
            "PathUtils.validate_paths() is deprecated. Use `get_path_resolver().validate_paths()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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

    @classmethod
    def setup_paths(cls) -> Path:
        """
        DEPRECATED: Use `setup_paths()` function instead.

        Setup project paths and sys.path for any script.

        This is the RECOMMENDED way to handle path resolution in the OCR project.
        Call this function at the top of any script that needs to import from
        the OCR project modules.

        Usage:
            from ocr.utils.path_utils import setup_paths
            setup_paths()

            # Now you can import from any OCR module
            from ocr.datasets.preprocessing import DocumentPreprocessor
            from ui.utils.config_parser import ConfigParser

        This replaces the old brittle approach of:
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            sys.path.append(str(project_root))
        """
        warnings.warn("PathUtils.setup_paths() is deprecated. Use `setup_paths()` function instead.", DeprecationWarning, stacklevel=2)
        # Add project root to sys.path if not already there
        project_root = cls.get_project_root()
        project_root_str = str(project_root)

        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)

        # Set environment variable for consistency
        os.environ[cls.PROJECT_ROOT_ENV] = project_root_str

        return project_root


# Global path resolver instance
_ocr_path_resolver = OCRPathResolver()


def get_path_resolver() -> OCRPathResolver:
    """Get the global OCR path resolver instance."""
    return _ocr_path_resolver


def setup_project_paths(config: dict[str, Any] | None = None) -> OCRPathResolver:
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
    """DEPRECATED: Use `get_path_resolver().config.project_root` instead."""
    warnings.warn(
        "get_project_root() is deprecated. Use `get_path_resolver().config.project_root` instead.", DeprecationWarning, stacklevel=2
    )
    return get_path_resolver().config.project_root


def get_data_path() -> Path:
    """DEPRECATED: Use `get_path_resolver().config.data_dir` instead."""
    warnings.warn("get_data_path() is deprecated. Use `get_path_resolver().config.data_dir` instead.", DeprecationWarning, stacklevel=2)
    return get_path_resolver().config.data_dir


def get_config_path() -> Path:
    """DEPRECATED: Use `get_path_resolver().config.config_dir` instead."""
    warnings.warn("get_config_path() is deprecated. Use `get_path_resolver().config.config_dir` instead.", DeprecationWarning, stacklevel=2)
    return get_path_resolver().config.config_dir


def get_outputs_path() -> Path:
    """DEPRECATED: Use `get_path_resolver().config.output_dir` instead."""
    warnings.warn(
        "get_outputs_path() is deprecated. Use `get_path_resolver().config.output_dir` instead.", DeprecationWarning, stacklevel=2
    )
    return get_path_resolver().config.output_dir


def get_images_path() -> Path:
    """DEPRECATED: Use `get_path_resolver().config.images_dir` instead."""
    warnings.warn("get_images_path() is deprecated. Use `get_path_resolver().config.images_dir` instead.", DeprecationWarning, stacklevel=2)
    return get_path_resolver().config.images_dir


def get_annotations_path() -> Path:
    """DEPRECATED: Use `get_path_resolver().config.annotations_dir` instead."""
    warnings.warn(
        "get_annotations_path() is deprecated. Use `get_path_resolver().config.annotations_dir` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_path_resolver().config.annotations_dir


def get_logs_path() -> Path:
    """DEPRECATED: Use `get_path_resolver().config.logs_dir` instead."""
    warnings.warn("get_logs_path() is deprecated. Use `get_path_resolver().config.logs_dir` instead.", DeprecationWarning, stacklevel=2)
    return get_path_resolver().config.logs_dir


def get_checkpoints_path() -> Path:
    """DEPRECATED: Use `get_path_resolver().config.checkpoints_dir` instead."""
    warnings.warn(
        "get_checkpoints_path() is deprecated. Use `get_path_resolver().config.checkpoints_dir` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_path_resolver().config.checkpoints_dir


def get_submissions_path() -> Path:
    """DEPRECATED: Use `get_path_resolver().config.submissions_dir` instead."""
    warnings.warn(
        "get_submissions_path() is deprecated. Use `get_path_resolver().config.submissions_dir` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_path_resolver().config.submissions_dir


def add_src_to_sys_path() -> None:
    """DEPRECATED: Use `setup_project_paths()` function instead."""
    warnings.warn("add_src_to_sys_path() is deprecated. Use `setup_project_paths()` function instead.", DeprecationWarning, stacklevel=2)
    # Add project root to sys.path if not already present
    src_path = get_path_resolver().config.project_root / "ocr"
    src_str = str(src_path)

    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def ensure_project_root_env() -> None:
    """DEPRECATED: Use `setup_project_paths()` function instead."""
    warnings.warn(
        "ensure_project_root_env() is deprecated. Use `setup_project_paths()` function instead.", DeprecationWarning, stacklevel=2
    )
    # Ensure the PROJECT_ROOT environment variable is set
    if not os.getenv(PathUtils.PROJECT_ROOT_ENV):
        project_root = str(get_path_resolver().config.project_root)
        os.environ[PathUtils.PROJECT_ROOT_ENV] = project_root


def validate_paths() -> dict:
    """DEPRECATED: Use `get_path_resolver().validate_paths()` instead."""
    warnings.warn("validate_paths() is deprecated. Use `get_path_resolver().validate_paths()` instead.", DeprecationWarning, stacklevel=2)
    # Validate that all expected paths exist using the new path resolver
    resolver = get_path_resolver()
    paths_to_check = {
        "project_root": resolver.config.project_root,
        "data": resolver.config.data_dir,
        "configs": resolver.config.config_dir,
        "outputs": resolver.config.output_dir,
        "images": resolver.config.images_dir,
        "annotations": resolver.config.annotations_dir,
        "logs": resolver.config.logs_dir,
        "checkpoints": resolver.config.checkpoints_dir,
        "submissions": resolver.config.submissions_dir,
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


def setup_paths():
    """DEPRECATED: Use `setup_project_paths()` instead."""
    warnings.warn("setup_paths() is deprecated. Use `setup_project_paths()` function instead.", DeprecationWarning, stacklevel=2)
    # Add project root to sys.path if not already there
    resolver = get_path_resolver()
    project_root = resolver.config.project_root
    project_root_str = str(project_root)

    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    # Set environment variable for consistency
    os.environ[PathUtils.PROJECT_ROOT_ENV] = project_root_str

    return project_root
