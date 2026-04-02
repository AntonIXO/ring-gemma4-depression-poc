"""MacroTrendEncoder: temporal encoder for pre-aggregated health metrics.

Design rationale
----------------
Smart rings (Oura, RingConn, Ultrahuman) expose pre-computed half-day or
daily aggregates, NOT raw PPG waveforms. The input is a tabular time-series
matrix  X ∈ R^{T×F}  where T ≈ 7-28 time steps and F ≈ 7-15 features
(heart_rate_mean, hrv_rmssd, hrv_sdnn, spo2_mean, skin_temp_delta, steps,
respiratory_rate, …).

Architecture choice: **lightweight Transformer encoder**.

Why Transformer over 1D-CNN for this data:
  1. T is small (≤28), so O(T²) self-attention is trivial.
  2. Features are *heterogeneous tabular metrics* with different scales and
     semantics — a learned Linear projection handles this naturally, while
     convolutions assume spatial/temporal locality in the feature dimension.
  3. Self-attention across time steps captures non-local dependencies like
     "HRV dropped 3 days ago AND temperature rose today → onset pattern".
  4. Per-step hidden states give the downstream SensorProjector a meaningful
     *sequence* to pool over (T → n_tokens), preserving temporal resolution.

This aligns with optiHealth 2.0 Ch.15: "Custom Temporal Encoder (a 1D-CNN
or lightweight Transformer) parses this matrix to extract macro-dynamics."

The encoder is frozen after initialisation, consistent with the rest of the
pipeline (frozen encoder → trainable projector → frozen+LoRA LLM).
"""

import math

import torch
import torch.nn as nn


class MacroTrendEncoder(nn.Module):
    """Transformer encoder for aggregated wearable health metrics.

    Input:  (batch, T, n_features)  — tabular time-series
    Output: (batch, T, output_dim)  — per-step embeddings

    The downstream SensorProjector pools T → n_tokens and projects
    output_dim → llm_dim, so we preserve the full temporal sequence here.

    Architecture:
        Linear(n_features → d_model)   feature projection
        + sinusoidal positional encoding
        → N TransformerEncoder layers (d_model, nhead, ff)
        → Linear(d_model → output_dim)  if d_model != output_dim

    Parameters (with defaults):
        n_features=10, d_model=256, nhead=4, n_layers=4, ff_mult=4,
        output_dim=512, max_len=128, dropout=0.1
        → ~2.7 M parameters.
    """

    def __init__(
        self,
        n_features: int = 10,
        d_model: int = 256,
        nhead: int = 4,
        n_layers: int = 4,
        ff_mult: int = 4,
        output_dim: int = 512,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.n_features = n_features
        self.d_model = d_model

        # Project heterogeneous features into d_model space
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )

        # Sinusoidal positional encoding (not learned — keeps param count low)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        # Transformer encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for better gradient flow
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Output projection to match expected encoder_dim
        if d_model != output_dim:
            self.out_proj = nn.Linear(d_model, output_dim)
        else:
            self.out_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a tabular time-series.

        Args:
            x: (batch, T, n_features) — health metric values per time step.
               NaN values should be replaced with 0.0 before calling this
               (see preprocessing.preprocess_tabular).

        Returns:
            (batch, T, output_dim) — per-step embeddings.
        """
        T = x.size(1)

        # Feature projection + positional encoding
        h = self.feature_proj(x)          # (batch, T, d_model)
        h = h + self.pe[:, :T, :]         # add positional encoding

        # Self-attention across time steps
        h = self.transformer(h)           # (batch, T, d_model)

        # Project to output_dim
        return self.out_proj(h)           # (batch, T, output_dim)


def get_encoder(
    n_features: int = 10,
    d_model: int = 256,
    n_layers: int = 4,
    output_dim: int = 512,
    **kwargs,
) -> MacroTrendEncoder:
    """Create and freeze a MacroTrendEncoder.

    All weights are frozen after construction (consistent with the
    PaPaGei/ResNet-1D convention in the original pipeline).

    Args:
        n_features: Number of input health-metric features.
        d_model: Transformer hidden dimension.
        n_layers: Number of transformer layers.
        output_dim: Output embedding dimension.

    Returns:
        Frozen MacroTrendEncoder with .output_dim attribute.
    """
    encoder = MacroTrendEncoder(
        n_features=n_features,
        d_model=d_model,
        n_layers=n_layers,
        output_dim=output_dim,
        **kwargs,
    )

    # Freeze all parameters
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    return encoder
