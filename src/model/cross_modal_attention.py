"""
Cross-Modal Attention: Solar Wind ↔ Electron Flux.

Novel component: No published GEO flux paper uses cross-modal attention
between solar wind and flux modalities.

Physics motivation:
  Solar wind DRIVES electron flux — they are causally related.
  Cross-attention explicitly models:
  "Given the current flux state, which solar wind patterns are most relevant?"
  This is fundamentally different from concatenating the two streams,
  which treats them as independent.

Architecture:
  Flux tokens (query) cross-attend to SW tokens (key, value).
  The model learns "if flux is currently low, pay attention to Bz drops"
  i.e., the attention weights become condition-dependent feature selectors.
"""

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """
    Cross-attention between flux representation and solar wind tokens.

    Flux representation (query) attends to SW feature tokens (key, value).

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Cross-attention: flux queries SW
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        # Self-refinement for the fused output
        self.self_attn  = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm3   = nn.LayerNorm(d_model)

        # Learnable fusion weight
        self.alpha   = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        flux_repr: torch.Tensor,         # [B, d_model] from SSM
        sw_tokens: torch.Tensor,         # [B, n_sw_features, d_model] from iTransformer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        flux_repr : [B, d_model]
            Flux history representation from SSM encoder.
        sw_tokens : [B, n_sw_features, d_model]
            Solar wind feature tokens from iTransformer.

        Returns
        -------
        fused : [B, d_model]
            Cross-modally fused representation.
        cross_attn_weights : [B, 1, n_sw_features]
            Attention weights (interpretable: which SW features drove prediction).
        """
        # Expand flux_repr to sequence of 1 token: [B, 1, d_model]
        q = flux_repr.unsqueeze(1)

        # Cross-attention: flux queries solar wind
        cross_out, cross_weights = self.cross_attn(
            query=q, key=sw_tokens, value=sw_tokens
        )  # cross_out: [B, 1, d_model]

        # Residual + norm
        q = self.norm1(q + cross_out)    # [B, 1, d_model]

        # Self-refinement
        self_out, _ = self.self_attn(q, q, q)
        q = self.norm2(q + self_out)

        # FFN
        q = self.norm3(q + self.ff(q))

        # Squeeze back: [B, d_model]
        fused = q.squeeze(1)

        # Gated fusion: learned blend of flux-only and cross-attended
        alpha = torch.sigmoid(self.alpha)
        fused = alpha * fused + (1 - alpha) * flux_repr

        return fused, cross_weights  # [B, d_model], [B, 1, n_sw_features]
