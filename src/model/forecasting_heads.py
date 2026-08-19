"""
Multi-Horizon Forecasting Heads with Physics Residual Prediction.

Multi-horizon heads (1 h, 6 h, 12 h).
NEW: Physics residual prediction — model predicts CORRECTION over persistence.

Physics residual formula:
    ŷ(t+h) = y(t) [persistence] + f_NN(fused_repr) [learned correction]

Why this is better than direct prediction:
  1. Persistence is already a strong baseline (especially at short horizons)
  2. Model learns WHEN and HOW MUCH flux changes — a smaller, easier task
  3. During quiet time: correction ≈ 0, physics baseline dominates
  4. During storms: correction is non-zero, model learns injection/decay
  5. Training is faster/more stable (lower variance regression target)

Auxiliary heads (multi-task learning):
  - Dst prediction: forces encoder to learn ring current physics
  - Kp prediction: forces encoder to learn geomagnetic activity proxy
  - Storm onset classifier: forces encoder to learn storm precursors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForecastHead(nn.Module):
    """Single forecast head for one horizon."""

    def __init__(
        self,
        d_model:    int,
        hidden_dim: int = 64,
        dropout:    float = 0.15,
        output_dim: int = 1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHorizonHeads(nn.Module):
    """
    Three forecast heads with physics residual prediction.

    Horizons: [1 h, 6 h, 12 h]
    Each head predicts the RESIDUAL over the persistence baseline.

    Horizon conditioning: previously all three flux heads consumed the exact
    same `fused_repr` vector and were differentiated only by having separate
    (but architecturally identical) weights. Nothing in the input told a head
    "you are the 12h head" as opposed to "you are the 1h head" — the only
    signal came indirectly through gradients from different loss weights.
    We now concatenate a learned per-horizon embedding onto `fused_repr`
    before each head, so heads receive an explicit horizon identity signal.
    This is a standard trick for shared-encoder multi-horizon forecasting.
    """

    def __init__(
        self,
        d_model:    int = 128,
        hidden_dim: int = 64,
        n_horizons: int = 3,
        dropout:    float = 0.15,
        horizon_embed_dim: int = 16,
    ):
        super().__init__()
        self.n_horizons = n_horizons
        self.horizon_embed_dim = horizon_embed_dim

        # Learned embedding per horizon index (0=1h, 1=6h, 2=12h, ...)
        self.horizon_embed = nn.Embedding(n_horizons, horizon_embed_dim)

        head_in_dim = d_model + horizon_embed_dim

        # Primary flux heads (one per horizon) — now horizon-conditioned
        self.flux_heads = nn.ModuleList([
            ForecastHead(head_in_dim, hidden_dim, dropout, 1)
            for _ in range(n_horizons)
        ])

        # Auxiliary task heads (shared across horizons, predict all horizons
        # jointly — these stay on the plain fused_repr since Dst/Kp are not
        # being asked "which horizon are you")
        self.dst_head   = ForecastHead(d_model, hidden_dim // 2, dropout, n_horizons)
        self.kp_head    = ForecastHead(d_model, hidden_dim // 2, dropout, n_horizons)
        self.storm_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )  # binary: storm onset in next 6h

        # Uncertainty heads (for 6h and 12h — longer horizons need UQ)
        # also horizon-conditioned for the same reason as the flux heads
        self.log_var_heads = nn.ModuleList([
            ForecastHead(head_in_dim, hidden_dim // 2, dropout, 1)
            for _ in range(n_horizons)
        ])

    def forward(
        self,
        fused_repr:  torch.Tensor,   # [B, d_model]
        y_persist:   torch.Tensor,   # [B, n_horizons] persistence baselines
    ) -> dict:
        """
        Parameters
        ----------
        fused_repr : [B, d_model]
        y_persist  : [B, n_horizons]

        Returns
        -------
        dict with keys:
            flux_pred     : [B, n_horizons] final flux predictions
            flux_residual : [B, n_horizons] learned corrections (for loss)
            dst_pred      : [B, n_horizons]
            kp_pred       : [B, n_horizons]
            storm_prob    : [B, 1] storm onset probability
            log_var       : [B, n_horizons] aleatoric uncertainty (log variance)
        """
        B = fused_repr.size(0)
        device = fused_repr.device

        # Build horizon-conditioned inputs: [B, n_horizons, d_model + embed_dim]
        # Same fused_repr broadcast to every horizon, each concatenated with
        # that horizon's own learned embedding.
        horizon_ids  = torch.arange(self.n_horizons, device=device)          # [H]
        h_emb        = self.horizon_embed(horizon_ids)                       # [H, E]
        h_emb        = h_emb.unsqueeze(0).expand(B, -1, -1)                  # [B, H, E]
        repr_expand  = fused_repr.unsqueeze(1).expand(-1, self.n_horizons, -1)  # [B, H, D]
        conditioned  = torch.cat([repr_expand, h_emb], dim=-1)               # [B, H, D+E]

        # Physics residual prediction — each head now sees its own horizon slice
        residuals = torch.cat(
            [self.flux_heads[i](conditioned[:, i, :]) for i in range(self.n_horizons)],
            dim=1,
        )  # [B, n_horizons]

        flux_pred = y_persist + residuals  # physics + learned correction

        # Uncertainty estimates — also horizon-conditioned
        log_var = torch.cat(
            [self.log_var_heads[i](conditioned[:, i, :]) for i in range(self.n_horizons)],
            dim=1,
        )  # [B, n_horizons]

        # Auxiliary predictions
        dst_pred    = self.dst_head(fused_repr)    # [B, n_horizons]
        kp_pred     = self.kp_head(fused_repr)     # [B, n_horizons]
        storm_logits = self.storm_head(fused_repr) # [B, 1] raw logits
        storm_prob   = torch.sigmoid(storm_logits) # [B, 1] ∈ (0,1) for display

        return {
            "flux_pred":      flux_pred,
            "flux_residual":  residuals,
            "log_var":        log_var,
            "dst_pred":       dst_pred,
            "kp_pred":        kp_pred,
            "storm_logits":   storm_logits,   # for BCEWithLogitsLoss (stable)
            "storm_prob":     storm_prob,     # for display / dashboard
        }
