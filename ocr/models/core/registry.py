"""Component registry system for plug-and-play OCR architectures."""

from __future__ import annotations

import inspect
from typing import Any

from .base_classes import BaseDecoder, BaseEncoder, BaseHead, BaseLoss, BaseMetric


class ComponentRegistry:
    """Central registry for OCR architecture components.

    Enables plug-and-play architecture experimentation by providing
    a centralized way to register and discover components.
    """

    def __init__(self) -> None:
        self._encoders: dict[str, type[BaseEncoder]] = {}
        self._decoders: dict[str, type[BaseDecoder]] = {}
        self._heads: dict[str, type[BaseHead]] = {}
        self._losses: dict[str, type[BaseLoss]] = {}
        self._metrics: dict[str, type[BaseMetric]] = {}

        # Architecture presets (collections of compatible components)
        self._architectures: dict[str, dict[str, str]] = {}

    def register_encoder(self, name: str, encoder_class: type[BaseEncoder]) -> None:
        """Register an encoder implementation.

        Args:
            name: Unique name for the encoder
            encoder_class: Encoder class that inherits from BaseEncoder
        """
        if not issubclass(encoder_class, BaseEncoder):
            raise TypeError(f"Encoder class must inherit from BaseEncoder, got {encoder_class}")
        self._encoders[name] = encoder_class

    def register_decoder(self, name: str, decoder_class: type[BaseDecoder]) -> None:
        """Register a decoder implementation.

        Args:
            name: Unique name for the decoder
            decoder_class: Decoder class that inherits from BaseDecoder
        """
        if not issubclass(decoder_class, BaseDecoder):
            raise TypeError(f"Decoder class must inherit from BaseDecoder, got {decoder_class}")
        self._decoders[name] = decoder_class

    def register_head(self, name: str, head_class: type[BaseHead]) -> None:
        """Register a head implementation.

        Args:
            name: Unique name for the head
            head_class: Head class that inherits from BaseHead
        """
        if not issubclass(head_class, BaseHead):
            raise TypeError(f"Head class must inherit from BaseHead, got {head_class}")
        self._heads[name] = head_class

    def register_loss(self, name: str, loss_class: type[BaseLoss]) -> None:
        """Register a loss function implementation.

        Args:
            name: Unique name for the loss function
            loss_class: Loss class that inherits from BaseLoss
        """
        if not issubclass(loss_class, BaseLoss):
            raise TypeError(f"Loss class must inherit from BaseLoss, got {loss_class}")
        self._losses[name] = loss_class

    def register_metric(self, name: str, metric_class: type[BaseMetric]) -> None:
        """Register a metric implementation.

        Args:
            name: Unique name for the metric
            metric_class: Metric class that inherits from BaseMetric
        """
        if not issubclass(metric_class, BaseMetric):
            raise TypeError(f"Metric class must inherit from BaseMetric, got {metric_class}")
        self._metrics[name] = metric_class

    def register_architecture(self, name: str, encoder: str, decoder: str, head: str, loss: str, **kwargs) -> None:
        """Register a complete architecture preset.

        Args:
            name: Unique name for the architecture
            encoder: Name of the registered encoder
            decoder: Name of the registered decoder
            head: Name of the registered head
            loss: Name of the registered loss
            **kwargs: Additional architecture metadata
        """
        self._architectures[name] = {"encoder": encoder, "decoder": decoder, "head": head, "loss": loss, **kwargs}

    def get_encoder(self, name: str) -> type[BaseEncoder]:
        """Get a registered encoder class.

        Args:
            name: Name of the registered encoder

        Returns:
            Encoder class

        Raises:
            KeyError: If encoder is not registered
        """
        if name not in self._encoders:
            available = list(self._encoders.keys())
            raise KeyError(f"Encoder '{name}' not registered. Available: {available}")
        return self._encoders[name]

    def get_decoder(self, name: str) -> type[BaseDecoder]:
        """Get a registered decoder class.

        Args:
            name: Name of the registered decoder

        Returns:
            Decoder class

        Raises:
            KeyError: If decoder is not registered
        """
        if name not in self._decoders:
            available = list(self._decoders.keys())
            raise KeyError(f"Decoder '{name}' not registered. Available: {available}")
        return self._decoders[name]

    def get_head(self, name: str) -> type[BaseHead]:
        """Get a registered head class.

        Args:
            name: Name of the registered head

        Returns:
            Head class

        Raises:
            KeyError: If head is not registered
        """
        if name not in self._heads:
            available = list(self._heads.keys())
            raise KeyError(f"Head '{name}' not registered. Available: {available}")
        return self._heads[name]

    def get_loss(self, name: str) -> type[BaseLoss]:
        """Get a registered loss class.

        Args:
            name: Name of the registered loss

        Returns:
            Loss class

        Raises:
            KeyError: If loss is not registered
        """
        if name not in self._losses:
            available = list(self._losses.keys())
            raise KeyError(f"Loss '{name}' not registered. Available: {available}")
        return self._losses[name]

    def get_metric(self, name: str) -> type[BaseMetric]:
        """Get a registered metric class.

        Args:
            name: Name of the registered metric

        Returns:
            Metric class

        Raises:
            KeyError: If metric is not registered
        """
        if name not in self._metrics:
            available = list(self._metrics.keys())
            raise KeyError(f"Metric '{name}' not registered. Available: {available}")
        return self._metrics[name]

    def get_architecture(self, name: str) -> dict[str, str]:
        """Get a registered architecture preset.

        Args:
            name: Name of the registered architecture

        Returns:
            Dictionary containing component names for the architecture

        Raises:
            KeyError: If architecture is not registered
        """
        if name not in self._architectures:
            available = list(self._architectures.keys())
            raise KeyError(f"Architecture '{name}' not registered. Available: {available}")
        return self._architectures[name].copy()

    def list_encoders(self) -> list[str]:
        """List all registered encoder names."""
        return list(self._encoders.keys())

    def list_decoders(self) -> list[str]:
        """List all registered decoder names."""
        return list(self._decoders.keys())

    def list_heads(self) -> list[str]:
        """List all registered head names."""
        return list(self._heads.keys())

    def list_losses(self) -> list[str]:
        """List all registered loss names."""
        return list(self._losses.keys())

    def list_metrics(self) -> list[str]:
        """List all registered metric names."""
        return list(self._metrics.keys())

    def list_architectures(self) -> list[str]:
        """List all registered architecture names."""
        return list(self._architectures.keys())

    def create_architecture_components(self, architecture_name: str, **component_configs) -> dict[str, Any]:
        """Create all components for a registered architecture.

        Args:
            architecture_name: Name of the registered architecture
            **component_configs: Configuration dictionaries for each component
                                Keys should be 'encoder_config', 'decoder_config', etc.

        Returns:
            Dictionary containing instantiated components
        """
        arch_config = self.get_architecture(architecture_name)

        components: dict[str, Any] = {}

        # Create encoder
        encoder_name = component_configs.get("encoder_name") or arch_config["encoder"]
        encoder_class = self.get_encoder(encoder_name)
        encoder_config = self._filter_component_kwargs(encoder_class, component_configs.get("encoder_config", {}))
        components["encoder"] = encoder_class(**encoder_config)

        # Create decoder
        decoder_name = component_configs.get("decoder_name") or arch_config["decoder"]
        decoder_class = self.get_decoder(decoder_name)
        decoder_config = component_configs.get("decoder_config", {})
        decoder_config.setdefault("in_channels", components["encoder"].out_channels)
        decoder_config = self._filter_component_kwargs(decoder_class, decoder_config)
        components["decoder"] = decoder_class(**decoder_config)

        # Create head
        head_name = component_configs.get("head_name") or arch_config["head"]
        head_class = self.get_head(head_name)
        head_config = component_configs.get("head_config", {})
        head_config.setdefault("in_channels", components["decoder"].out_channels)
        head_config = self._filter_component_kwargs(head_class, head_config)
        components["head"] = head_class(**head_config)

        # Create loss
        loss_name = component_configs.get("loss_name") or arch_config["loss"]
        loss_class = self.get_loss(loss_name)
        loss_config = self._filter_component_kwargs(loss_class, component_configs.get("loss_config", {}))
        components["loss"] = loss_class(**loss_config)

        return components

    @staticmethod
    def _filter_component_kwargs(component_cls: type, config: dict[str, Any]) -> dict[str, Any]:
        """Filter configuration dictionary to match component signature.

        Prevents leaking kwargs from one architecture override into another when
        switching presets within the same Hydra composition.
        """

        if not config:
            return {}

        signature = inspect.signature(component_cls.__init__)  # type: ignore[misc]
        parameters = signature.parameters

        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            return config

        allowed_keys = {name for name, param in parameters.items() if name != "self"}
        return {key: value for key, value in config.items() if key in allowed_keys}


# Global registry instance
registry = ComponentRegistry()


def get_registry() -> ComponentRegistry:
    """Get the global component registry instance."""
    return registry
