"""SensorProjector: maps encoder embeddings into LLM token space."""

import torch
import torch.nn as nn


class SensorProjector(nn.Module):
    """Project sensor encoder embeddings into LLM token space.

    Architecture:
        AdaptiveAvgPool1d(n_tokens) → Linear(encoder_dim, llm_dim)
        → GELU → LayerNorm(llm_dim) → Linear(llm_dim, llm_dim)

    The input is (batch, N_segments, encoder_dim) — a variable number of
    segment embeddings.  We treat N_segments as the temporal axis and pool
    it down to *n_tokens* fixed virtual tokens, then project each token
    into the LLM hidden dimension.

    Trainable parameters ≈ 4 M (with default dims).
    """

    def __init__(self, encoder_dim: int = 512, llm_dim: int = 2048,
                 n_tokens: int = 16):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.n_tokens = n_tokens

        # Temporal pooling across segments → n_tokens
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)

        # 2-layer MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim, llm_dim),
            nn.GELU(),
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, sensor_embeddings: torch.Tensor) -> torch.Tensor:
        """Project sensor embeddings to virtual LLM tokens.

        Args:
            sensor_embeddings: (batch, N_segments, encoder_dim)

        Returns:
            (batch, n_tokens, llm_dim)
        """
        # (batch, N_segments, encoder_dim) → (batch, encoder_dim, N_segments)
        x = sensor_embeddings.transpose(1, 2)
        # Pool: (batch, encoder_dim, n_tokens)
        x = self.pool(x)
        # (batch, n_tokens, encoder_dim)
        x = x.transpose(1, 2)
        # Project each token
        return self.mlp(x)
