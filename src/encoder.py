"""Sensor encoder: PaPaGei wrapper with ResNet-1D fallback."""

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Single residual block for 1-D convolutions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection (identity or projection)
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """18-block ResNet-1D encoder for 1-channel PPG signals.

    Architecture:
        - Input: (batch, 1, 1250)
        - Initial conv: 1 → 64 channels
        - 18 residual blocks, base_filters=64 doubling every 4 blocks
          (blocks 0-3: 64, 4-7: 128, 8-11: 256, 12-17: 512)
        - Global average pooling
        - Output: (batch, 512)
    """

    def __init__(self, input_channels: int = 1, base_filters: int = 64,
                 n_blocks: int = 18, output_dim: int = 512):
        super().__init__()
        self.output_dim = output_dim

        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_filters, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Build residual blocks
        blocks = []
        in_ch = base_filters
        for i in range(n_blocks):
            # Double channels every 4 blocks
            group = i // 4
            out_ch = min(base_filters * (2 ** group), output_dim)
            stride = 2 if (i > 0 and i % 4 == 0) else 1
            blocks.append(ResidualBlock1D(in_ch, out_ch, stride=stride))
            in_ch = out_ch

        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Final projection to output_dim if last block channels differ
        if in_ch != output_dim:
            self.proj = nn.Linear(in_ch, output_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, 1, 1250) PPG segment tensor.

        Returns:
            (batch, 512) embedding.
        """
        out = self.stem(x)
        out = self.blocks(out)
        out = self.gap(out).squeeze(-1)
        out = self.proj(out)
        return out


def _try_load_papagei() -> nn.Module | None:
    """Attempt to load the PaPaGei encoder from HuggingFace."""
    try:
        from papagei import PaPaGei  # type: ignore
        model = PaPaGei.from_pretrained("nokia-bell-labs/papagei")
        return model
    except Exception:
        pass

    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("nokia-bell-labs/papagei",
                                          trust_remote_code=True)
        return model
    except Exception:
        return None


def get_encoder(use_papagei: bool = True) -> nn.Module:
    """Return a PPG encoder.

    Tries PaPaGei first (if *use_papagei* is True), then falls back to a
    custom ResNet-1D.  All weights are frozen.

    Args:
        use_papagei: Whether to attempt loading PaPaGei.

    Returns:
        Frozen nn.Module with .output_dim attribute (512).
    """
    encoder = None
    if use_papagei:
        encoder = _try_load_papagei()
        if encoder is not None:
            if not hasattr(encoder, "output_dim"):
                encoder.output_dim = 512

    if encoder is None:
        encoder = ResNet1D(input_channels=1, base_filters=64, n_blocks=18,
                           output_dim=512)

    # Freeze all parameters
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    return encoder
