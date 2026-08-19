"""
iTransformer Encoder for Solar Wind Features.
Based on: "iTransformer: Inverted Transformers Are Effective for Time Series
Forecasting" (Liu et al., ICLR 2024 Spotlight).

KEY DIFFERENCE from standard Transformer:
  - Standard: each TIME STEP is a token → attention over time
  - iTransformer: each FEATURE (Vsw, Bz, …) is a token → attention over features

Why this matters for our problem:
  The key physics is WHICH FEATURES correlate with flux enhancement,
  not WHEN in history they peaked (that's handled by the SSM).
  iTransformer learns "Bz and Vsw together → flux enhancement"
  which is exactly the Dungey cycle physics.
"""

import torch
import torch.nn as nn
import math


class VariateEmbedding(nn.Module):
    """Embed each feature (variate) time series into a d_model vector."""

    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(seq_len, d_model)
        self.dropout    = nn.Dropout(dropout)
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, T, n_features]

        Returns
        -------
        tokens : [B, n_features, d_model]  — one token per feature
        """
        # Transpose: [B, T, F] → [B, F, T]
        x_t = x.transpose(1, 2)
        # Each feature's full time series → one embedding
        tokens = self.norm(self.dropout(self.projection(x_t)))
        return tokens  # [B, F, d_model]


class iTransformerLayer(nn.Module):
    """
    Single iTransformer layer:
    - Self-attention OVER FEATURES (captures feature correlations)
    - FFN applied per-feature (captures temporal non-linearity)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                              batch_first=True)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x : [B, n_features, d_model]  — feature tokens
        Returns : [B, n_features, d_model], attn_weights [B, F, F]
        """
        # Self-attention over feature dimension
        attn_out, attn_weights = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN per feature
        x = self.norm2(x + self.ff(x))
        return x, attn_weights


class iTransformerEncoder(nn.Module):
    """
    Full iTransformer encoder for solar wind features.

    Input : [B, T, n_sw_features]
    Output: [B, d_model]  — pooled over feature dimension
    """

    def __init__(
        self,
        seq_len:    int,
        n_features: int,
        d_model:    int = 128,
        n_heads:    int = 4,
        n_layers:   int = 3,
        d_ff:       int = 256,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model    = d_model

        self.embedding = VariateEmbedding(seq_len, d_model, dropout)
        self.layers    = nn.ModuleList([
            iTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.pool_proj = nn.Linear(n_features * d_model, d_model)
        self.norm      = nn.LayerNorm(d_model)

        self._attn_weights = []  # Store for interpretability

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list]:
        """
        Parameters
        ----------
        x : [B, T, n_sw_features]

        Returns
        -------
        out : [B, d_model]
            Pooled SW representation.
        attn_weights : list of [B, F, F]
            Per-layer attention maps (feature correlation matrices).
            Used for interpretability analysis in paper.
        """
        # Embed: [B, T, F] → [B, F, d_model]
        tokens = self.embedding(x)

        attn_weights = []
        for layer in self.layers:
            tokens, aw = layer(tokens)
            attn_weights.append(aw)

        # Pool over features: [B, F, d_model] → [B, F*d_model] → [B, d_model]
        B, F, D = tokens.shape
        pooled  = self.pool_proj(tokens.reshape(B, F * D))
        out     = self.norm(pooled)

        return out, attn_weights
