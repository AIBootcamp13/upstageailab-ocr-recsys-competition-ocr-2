"""UNet decoder for DBNet architecture."""

from itertools import accumulate

import torch
import torch.nn as nn

from ..core import BaseDecoder


class UNetDecoder(BaseDecoder):
    """UNet decoder for DBNet architecture with skip connections."""

    def __init__(
        self,
        in_channels: list[int],
        strides: list[int] | None = None,
        inner_channels: int = 256,
        output_channels: int = 64,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(in_channels=in_channels, **kwargs)

        if strides is None:
            strides = [4, 8, 16, 32]

        assert len(strides) == len(in_channels), "Mismatch in 'strides' and 'in_channels' lengths."

        # Calculate upscale factors for decoder
        upscale_factors = [strides[idx] // strides[idx - 1] for idx in range(1, len(strides))]
        outscale_factors = list(accumulate(upscale_factors, lambda x, y: x * y))

        # Create upsampling layers
        self.upsamples = nn.ModuleList()
        for upscale in upscale_factors:
            self.upsamples.append(nn.Upsample(scale_factor=upscale, mode="nearest"))

        # Create inner projection layers
        self.inners = nn.ModuleList()
        for in_channel in in_channels:
            self.inners.append(nn.Conv2d(in_channel, inner_channels, kernel_size=1, bias=bias))

        # Create outer projection and upsampling layers
        self.outers = nn.ModuleList()
        for outscale in reversed(outscale_factors):
            outer = nn.Sequential(
                nn.Conv2d(inner_channels, output_channels, kernel_size=3, padding=1, bias=bias),
                nn.Upsample(scale_factor=outscale, mode="nearest"),
            )
            self.outers.append(outer)
        self.outers.append(nn.Conv2d(inner_channels, output_channels, kernel_size=3, padding=1, bias=bias))

        # Initialize weights
        self.upsamples.apply(self.weights_init)
        self.inners.apply(self.weights_init)
        self.outers.apply(self.weights_init)

        self._output_channels = output_channels

    def weights_init(self, m: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(1e-4)
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.fill_(1e-4)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Decode features through U-Net architecture."""
        # Project input features to common channel dimension
        in_features = [inner(feat) for feat, inner in zip(features, self.inners, strict=False)]

        # Build upsampled features with skip connections
        up_features = []
        up = in_features[-1]
        for i in range(len(in_features) - 1, 0, -1):
            up = self.upsamples[i - 1](up) + in_features[i - 1]
            up_features.append(up)

        # Apply outer projections and upsampling
        out_features = [self.outers[0](in_features[-1])]
        out_features += [outer(feat) for feat, outer in zip(up_features, self.outers[1:], strict=False)]

        # Return the final decoded feature map
        return out_features[-1]

    @property
    def out_channels(self) -> int:
        """Return the number of output channels from the decoder."""
        return self._output_channels
