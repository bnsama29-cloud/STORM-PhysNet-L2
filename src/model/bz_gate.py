"""
Bz Physics Gate.

A soft gating mechanism that amplifies the encoder's output when IMF Bz
is strongly southward (indicating active magnetospheric driving).

Physics basis: Reconnection rate ∝ |Bz| when Bz < 0 (Dungey cycle).
Sustained southward Bz drives convection, particle injection, and
outer radiation belt electron acceleration.

When Bz << 0: gate → 1.0 (full signal passes through)
When Bz >> 0: gate → small value (quiet time, weaker modulation)

This directly connects to Stage 1 of STORM-PhysNet: the gate provides
a physics-driven attention signal for storm detection.
"""

import torch
import torch.nn as nn
import math


class BzPhysicsGate(nn.Module):
    """
    Physics-informed soft gate driven by IMF Bz history.

    The gate g(t) ∈ (0, 1] is computed as:
        Bz_south = max(0, -Bz)  [only southward component]
        raw_gate = σ(W · [Bz_south_history, Bz_south_duration] + b)
        gate     = gate_min + (1 - gate_min) * raw_gate

    Parameters
    ----------
    bz_feature_idx : int
        Column index of Bz in the solar wind feature matrix.
    bz_dur_idx : int
        Column index of bz_south_duration feature.
    d_model : int
        Model dimension (gate output dimension).
    bz_threshold : float
        Bz threshold for gate activation (nT, negative).
    gate_min : float
        Minimum gate value during quiet time (prevents complete shutdown).
    """

    def __init__(
        self,
        bz_feature_idx: int = 1,
        bz_dur_idx:     int = 5,
        d_model:        int = 128,
        bz_threshold:   float = -5.0,
        gate_min:       float = 0.1,
    ):
        super().__init__()
        self.bz_idx      = bz_feature_idx
        self.bz_dur_idx  = bz_dur_idx
        self.bz_threshold = bz_threshold
        self.gate_min    = gate_min
        self.d_model     = d_model

        # Learnable gate network: takes [mean_Bz_south, bz_dur, Bz_min] → gate scalar
        self.gate_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, d_model),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        # Initialize to produce gate ≈ 0.5 at initialization
        nn.init.xavier_uniform_(self.gate_net[0].weight)
        nn.init.zeros_(self.gate_net[0].bias)
        nn.init.xavier_uniform_(self.gate_net[2].weight)
        nn.init.constant_(self.gate_net[2].bias, -math.log(3))  # sigmoid → 0.25

    def forward(
        self,
        encoder_output: torch.Tensor,   # [B, d_model]
        x_sw: torch.Tensor,             # [B, T, n_sw_features]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply physics gate to encoder output.

        Parameters
        ----------
        encoder_output : [B, d_model]
            Fused encoder representation.
        x_sw : [B, T, n_sw_features]
            Solar wind input sequence (unscaled or scaled — gate uses relative).

        Returns
        -------
        gated_output : [B, d_model]
            Gated encoder output.
        gate_values : [B, d_model]
            Gate activation values (for interpretability / loss).
        """
        # Extract Bz features from SW input
        bz = x_sw[:, :, self.bz_idx]             # [B, T]
        bz_south = torch.clamp(-bz, min=0.0)     # southward only

        # Summarize Bz history for gate computation
        mean_bz_south = bz_south.mean(dim=1, keepdim=True)     # [B, 1]
        max_bz_south  = bz_south.max(dim=1).values.unsqueeze(1) # [B, 1]

        # Bz_south_duration from feature (if available)
        if self.bz_dur_idx < x_sw.size(-1):
            bz_dur = x_sw[:, -1, self.bz_dur_idx].unsqueeze(1)  # [B, 1]
        else:
            bz_dur = mean_bz_south  # fallback

        gate_input = torch.cat([mean_bz_south, max_bz_south, bz_dur], dim=1)  # [B, 3]
        gate_values = self.gate_net(gate_input)               # [B, d_model]

        # Ensure gate ≥ gate_min
        gate_values = self.gate_min + (1 - self.gate_min) * gate_values

        gated_output = encoder_output * gate_values
        return gated_output, gate_values

    def extra_repr(self) -> str:
        return (f"bz_threshold={self.bz_threshold}nT, "
                f"gate_min={self.gate_min}, d_model={self.d_model}")
