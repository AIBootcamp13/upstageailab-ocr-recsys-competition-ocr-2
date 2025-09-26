"""Core abstract base classes and registry for OCR framework components."""

from .base_classes import (
    BaseDecoder,
    BaseEncoder,
    BaseHead,
    BaseLoss,
    BaseMetric,
)
from .registry import ComponentRegistry, get_registry, registry

__all__ = [
    "BaseEncoder",
    "BaseDecoder",
    "BaseHead",
    "BaseLoss",
    "BaseMetric",
    "ComponentRegistry",
    "get_registry",
    "registry",
]
