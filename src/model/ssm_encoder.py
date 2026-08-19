"""
Custom SSM (State Space Model) Encoder for Electron Flux History.
Simplified S4-style linear recurrence for capturing long-memory decay dynamics.

Why SSM for flux history:
  Electron flux has SLOW dynamics (hours to days decay).
  Standard Transformer struggles with very long sequences of slowly varying signal.
  SSM's linear recurrence efficiently captures exponential decay memory,
  which is exactly the physics of radiation belt particle loss processes.

Architecture: Multi-layer linear SSM with diagonal state matrix.
  h_t = A·h_{t-1} + B·x_t
  y_t = C·h_t + D·x_t

Where A is constrained to be stable (eigenvalues < 1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SSMLayer(nn.Module):
    """
    Single linear SSM layer (simplified S4/Mamba-inspired).

    State: h ∈ R^{d_state}
    Input: x ∈ R^{d_input}
    Output: y ∈ R^{d_model}
    """

    def __init__(self, d_input: int, d_state: int, d_model: int,
                 dropout: float = 0.1):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model

        # Diagonal state matrix A (log parameterization for stability)
        # A_diag = -exp(log_A) → negative diagonal → stable
        self.log_A   = nn.Parameter(torch.randn(d_state))
        self.B       = nn.Parameter(torch.randn(d_state, d_input) /
                                    math.sqrt(d_input))
        self.C       = nn.Parameter(torch.randn(d_model, d_state) /
                                    math.sqrt(d_state))
        self.D       = nn.Parameter(torch.ones(d_model, d_input) /
                                    math.sqrt(d_input))
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)

    @property
    def A_diag(self) -> torch.Tensor:
        """Stable diagonal A: eigenvalues in (-1, 0) — required for discrete-time stability.
        sigmoid maps log_A ∈ R → (0,1), negated → (-1, 0).
        This guarantees |A_diag| < 1 at all times (no blow-up).
        """
        return -torch.sigmoid(self.log_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sequential scan over time dimension.

        Parameters
        ----------
        x : [B, T, d_input]

        Returns
        -------
        out : [B, T, d_model]
        """
        B, T, _ = x.shape
        A = self.A_diag  # [d_state]

        # Initialize hidden state
        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            x_t = x[:, t, :]                              # [B, d_input]
            h   = h * A.unsqueeze(0) + x_t @ self.B.T    # [B, d_state]
            y_t = h @ self.C.T + x_t @ self.D.T           # [B, d_model]
            outputs.append(y_t)

        out = torch.stack(outputs, dim=1)  # [B, T, d_model]
        out = self.dropout(out)
        return out


class SSMEncoder(nn.Module):
    """
    Multi-layer SSM encoder for electron flux history.

    Input : [B, T, 1] (log flux time series)
    Output: [B, d_model] (pooled flux representation)
    """

    def __init__(
        self,
        d_model:  int = 128,
        d_state:  int = 64,
        n_layers: int = 2,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(1, d_model)
        self.layers     = nn.ModuleList([
            SSMLayer(d_model, d_state, d_model, dropout)
            for _ in range(n_layers)
        ])
        self.norms      = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        self.out_norm   = nn.LayerNorm(d_model)

    def forward(self, x_flux: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_flux : [B, T, 1]

        Returns
        -------
        out : [B, d_model]  — last hidden state (most recent memory)
        """
        h = self.input_proj(x_flux)           # [B, T, d_model]

        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + layer(h))            # residual connection

        # Use last timestep as the flux state representation
        out = self.out_norm(h[:, -1, :])      # [B, d_model]
        return out
