"""
Analogy-Inspired Gates: Radiotrophic (biology) and Cathode-Anode-Battery
(electronics) gating mechanisms.

IMPORTANT FRAMING NOTE (read before citing this in the paper):
These modules are inspired by loose structural analogies to biology and
electronics, NOT literal physical mechanisms transplanted from those
domains. There is no direct causal link between "melanin absorbing gamma
rays" or "a battery regulating voltage" and how the radiation belt
population evolves. What IS transferable is the *mathematical shape* of
the response function each analogy suggests — a saturating accumulation
curve, a rectified flow-through gate — which is a legitimate thing to
try as an architectural choice, exactly the same way "attention" or
"gating" more broadly are borrowed metaphors rather than literal claims.
Report these as "bio-inspired" / "circuit-inspired" activation choices in
the paper, ablate them against the physics-grounded BzPhysicsGate, and let
the numbers (not the metaphor) carry the argument.

── 1. RadiotrophicGate (biology-inspired) ───────────────────────────────
Real biology this draws from: melanized fungi found growing on the
Chernobyl reactor exhibit "radiotropism" (directional growth toward
radiation) and "radiosynthesis" — melanin is hypothesized to mediate an
electron-transfer reaction that gives a net energy gain from ionizing
radiation, loosely analogous to photosynthesis (Dadachova & Casadevall,
2008; Shunk et al. 2020, bioRxiv ISS radiotrophic fungus study). The
growth-response curve reported in that literature is a classic
accumulation-then-saturation curve: growth benefit rises with dose, then
plateaus (and eventually would decline at damaging doses, though that
regime isn't relevant to GEO electron flux levels).
We mimic only the *shape* of that curve: a learnable saturating gate
whose activation grows with recent flux/energy "dose" and saturates
smoothly, rather than responding linearly or with a hard threshold like
the Bz gate.

── 2. CathodeAnodeGate (electronics-inspired) ──────────────────────────
Analogy: treat the Sun / magnetosphere as an "emitter" (cathode) pushing
electrons toward the satellite ("anode" / collector), and treat the
learned gate as a "tunable voltage" controlling how much of that emitted
signal reaches the forecasting heads — similar in spirit to how a
transistor's gate voltage controls current flow. This is a relabeling of
a standard gating nonlinearity with circuit-analogy naming, not a new
physical quantity. It's included because the analogy is a genuinely
useful *design* prompt (voltage-like gates are a well-studied and
effective architecture element - c.f. LSTM gates), not because satellites
and vacuum tubes share physics.
"""

import math
import torch
import torch.nn as nn


class RadiotrophicGate(nn.Module):
    """
    Biology-inspired saturating accumulation gate.

    gate(t) = gate_min + (1 - gate_min) * tanh(k * dose_accum)

    where dose_accum is an exponentially-weighted running "dose" built
    from recent flux magnitude and solar wind driving (Vsw, Bz_south),
    analogous to how radiotrophic fungal growth response accumulates with
    cumulative radiation exposure rather than reacting only to the
    instantaneous rate. `k` (the saturation steepness) is learned.
    """

    def __init__(
        self,
        d_model:      int = 128,
        vsw_idx:      int = 0,
        bz_idx:       int = 1,
        gate_min:     float = 0.1,
        dose_halflife_steps: int = 6,  # ~ exponential memory window, in timesteps
    ):
        super().__init__()
        self.d_model = d_model
        self.vsw_idx = vsw_idx
        self.bz_idx = bz_idx
        self.gate_min = gate_min
        # decay per step for an exponentially-weighted "dose" accumulator
        self.register_buffer(
            "decay", torch.tensor(0.5 ** (1.0 / max(dose_halflife_steps, 1)))
        )

        self.dose_to_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, d_model),
        )
        self.log_k = nn.Parameter(torch.zeros(1))  # learned saturation steepness (>0 via exp)
        nn.init.xavier_uniform_(self.dose_to_gate[0].weight)
        nn.init.zeros_(self.dose_to_gate[0].bias)
        nn.init.xavier_uniform_(self.dose_to_gate[2].weight)
        nn.init.zeros_(self.dose_to_gate[2].bias)

    def _compute_dose(self, x_sw: torch.Tensor) -> torch.Tensor:
        """Exponentially-weighted 'dose' accumulated over the input window.
        x_sw: [B, T, F] -> dose: [B, 1]"""
        vsw = x_sw[:, :, self.vsw_idx].clamp(min=0.0)         # [B, T]
        bz = x_sw[:, :, self.bz_idx]                          # [B, T]
        bz_south = torch.clamp(-bz, min=0.0)                  # [B, T]
        instant = vsw * (1.0 + bz_south)                      # [B, T], driving proxy

        T = instant.size(1)
        weights = self.decay ** torch.arange(T - 1, -1, -1, device=instant.device).float()
        weights = weights / (weights.sum() + 1e-8)            # normalize
        dose = (instant * weights.unsqueeze(0)).sum(dim=1, keepdim=True)  # [B, 1]
        return dose

    def forward(self, encoder_output: torch.Tensor, x_sw: torch.Tensor):
        dose = self._compute_dose(x_sw)                        # [B, 1]
        k = torch.exp(self.log_k).clamp(max=10.0)               # steepness > 0
        raw_gate = torch.tanh(k * self.dose_to_gate(dose))       # [B, d_model], in (-1,1)
        raw_gate = 0.5 * (raw_gate + 1.0)                        # map to (0,1)
        gate_values = self.gate_min + (1 - self.gate_min) * raw_gate
        gated_output = encoder_output * gate_values
        return gated_output, gate_values

    def extra_repr(self) -> str:
        return f"gate_min={self.gate_min}, d_model={self.d_model} (bio-inspired, analogy-only)"


class CathodeAnodeGate(nn.Module):
    """
    Electronics-inspired 'tunable voltage' gate.

    Relabeling of a standard learned gate as an emitter/collector circuit:
    'emission strength' from solar wind driving plays the role of cathode
    emission current, and a learned 'voltage' parameter controls how much
    of that signal the 'anode' (forecasting head) actually collects —
    structurally similar to how LSTM/GRU gates work, renamed for
    interpretability/pedagogy in the paper rather than for new physics.
    """

    def __init__(
        self,
        d_model:  int = 128,
        vsw_idx:  int = 0,
        bz_idx:   int = 1,
        bt_idx:   int = 2,
        gate_min: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vsw_idx, self.bz_idx, self.bt_idx = vsw_idx, bz_idx, bt_idx
        self.gate_min = gate_min

        # "Emission current" estimator: how strongly the cathode (solar
        # wind / magnetosphere) is emitting, from the last timestep.
        self.emission_net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1),
        )
        # Learned "voltage" (per-channel), analogous to a bias/gain the
        # anode circuit applies to the incoming emission before it reaches
        # the rest of the network.
        self.voltage = nn.Parameter(torch.zeros(d_model))

        nn.init.xavier_uniform_(self.emission_net[0].weight)
        nn.init.zeros_(self.emission_net[0].bias)
        nn.init.xavier_uniform_(self.emission_net[2].weight)
        nn.init.zeros_(self.emission_net[2].bias)

    def forward(self, encoder_output: torch.Tensor, x_sw: torch.Tensor):
        last = x_sw[:, -1, :]                                    # [B, F]
        feats = torch.stack([
            last[:, self.vsw_idx].clamp(min=0.0),
            torch.clamp(-last[:, self.bz_idx], min=0.0),
            last[:, self.bt_idx].clamp(min=1e-3),
        ], dim=1)                                                # [B, 3]
        emission_current = torch.sigmoid(self.emission_net(feats))  # [B, 1], in (0,1)

        # "Collected current" = emission * voltage-tuned transmittance
        transmittance = torch.sigmoid(self.voltage).unsqueeze(0)     # [1, d_model]
        gate_values = emission_current * transmittance                # [B, d_model]
        gate_values = self.gate_min + (1 - self.gate_min) * gate_values

        gated_output = encoder_output * gate_values
        return gated_output, gate_values

    def extra_repr(self) -> str:
        return f"gate_min={self.gate_min}, d_model={self.d_model} (electronics-inspired, analogy-only)"
