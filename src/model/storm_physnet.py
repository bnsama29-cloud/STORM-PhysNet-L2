"""
STORM-PhysNet: The complete integrated model.
Storm-aware Physics-Informed Network for GEO Electron Flux Forecasting.

Architecture flow:
  1. Adaptive Propagation Delay — shifts SW by learnable tau
  2. iTransformer Encoder — extracts SW feature correlations
  3. SSM Encoder — captures flux history (slow decay dynamics)
  4. Cross-Modal Attention — flux queries SW (causal coupling)
  5. Bz Physics Gate — storm-driven signal amplification
  6. Multi-Horizon Heads — physics residual prediction (3 horizons)
     + Auxiliary: Dst, Kp, Storm onset (multi-task)

This file assembles the full model and supports:
  - Single model forward pass
  - Deep Ensemble (5 models) inference for uncertainty quantification
  - GRASP transfer learning (freeze encoder, fine-tune heads)
"""

import torch
import torch.nn as nn
from typing import Optional

from src.model.propagation_delay     import AdaptivePropagationDelay
from src.model.bz_gate               import BzPhysicsGate
from src.model.analogy_gates         import RadiotrophicGate, CathodeAnodeGate
from src.model.forecasting_heads     import MultiHorizonHeads
from src.model.spectral_head         import SpectralParamHead, spectral_shape_regularizer
from src.model.itransformer_encoder  import iTransformerEncoder
from src.model.ssm_encoder           import SSMEncoder
from src.model.cross_modal_attention import CrossModalAttention
from src.model.magnetopause_geometry import MagnetopauseGeometryFeatures # Optional; disabled in paper runs


class STORMPhysNet(nn.Module):
    """
    STORM-PhysNet: Full model.

    Parameters
    ----------
    n_sw_features : int
        Number of solar wind input features.
    seq_len : int
        Input sequence length in hours.
    d_model : int
        Internal model dimension.
    n_heads : int
        Attention heads for iTransformer and cross-modal attention.
    n_transformer_layers : int
        Number of iTransformer layers.
    n_ssm_layers : int
        Number of SSM layers.
    d_state : int
        SSM state dimension.
    d_ff : int
        Feed-forward dimension in iTransformer.
    hidden_dim : int
        Hidden dim for forecasting heads.
    n_horizons : int
        Number of forecast horizons (default 3).
    dropout : float
        Dropout rate.
    bz_feature_idx : int
        Column index of Bz in SW features.
    bz_dur_idx : int
        Column index of Bz south duration.
    """

    def __init__(
        self,
        n_sw_features:        int   = 14,
        seq_len:              int   = 72,
        d_model:              int   = 128,
        n_heads:              int   = 4,
        n_transformer_layers: int   = 3,
        n_ssm_layers:         int   = 2,
        d_state:              int   = 64,
        d_ff:                 int   = 256,
        hidden_dim:           int   = 64,
        n_horizons:           int   = 3,
        dropout:              float = 0.1,
        bz_feature_idx:       int   = 1,
        bz_dur_idx:           int   = 5,
        ablation:             str   = "none",
        backbone:             str   = "transformer",  # "transformer" | "hybrid"
        gate_type:            str   = "bz",  # "bz" | "radiotrophic" | "cathode_anode"
        use_spectral_head:    bool  = False,  # additive spectral-parameterization head
        use_magnetopause:     bool  = False,  # Shue (1998) magnetopause compression features
    ):
        super().__init__()
        self.d_model    = d_model
        self.n_sw_features = n_sw_features
        self.n_horizons = n_horizons
        self.ablation   = ablation
        self.backbone   = backbone
        self.gate_type  = gate_type
        self.use_spectral_head = use_spectral_head
        self.use_magnetopause  = use_magnetopause

        # ── 1. Adaptive Propagation Delay ──────────────
        self.prop_delay = AdaptivePropagationDelay(
            tau_init_hours=1.0,
            tau_min_hours=0.5,
            tau_max_hours=1.5,
            n_sw_features=n_sw_features,
        )

        # ── 2. Input Projection & Positional Encoding ───────────────────────
        # Concat SW (14 features) and Flux (1 feature) = 15 features total
        self.input_proj = nn.Linear(n_sw_features + 1, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.emb_dropout = nn.Dropout(dropout)

        # ── 3. Temporal Encoder Backbone ─────────────────────────────────────
        # Two interchangeable backbones producing the same [B, d_model] shape,
        # selected by `backbone` so both can be ablated against each other:
        #
        # "transformer" (default, matches original results): a single plain
        #   nn.TransformerEncoder sees the concatenated [SW ++ flux] sequence
        #   and attends over TIME. Simple, and what the reported baseline
        #   numbers were produced with.
        #
        # "hybrid" (previously written but never wired into forward()): SW and
        #   flux are encoded separately by architectures matched to their
        #   physics, then fused explicitly:
        #     - iTransformer attends over FEATURES (which SW variables
        #       co-vary), suited to "which drivers matter" questions.
        #     - SSM encodes flux history's slow exponential-decay dynamics,
        #       which a fixed-length attention window handles less naturally.
        #     - CrossModalAttention lets the flux state query the SW features
        #       ("given current flux, which SW pattern is relevant"), instead
        #       of just concatenating the two streams as the plain
        #       transformer does.
        #   This is a materially different, larger, and less-tested path —
        #   train/ablate it explicitly rather than assuming it's better.
        if backbone == "hybrid":
            self.itransformer = iTransformerEncoder(
                seq_len=seq_len,
                n_features=n_sw_features,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_transformer_layers,
                d_ff=d_ff,
                dropout=dropout,
            )
            self.ssm = SSMEncoder(
                d_model=d_model,
                d_state=d_state,
                n_layers=n_ssm_layers,
                dropout=dropout,
            )
            self.cross_modal = CrossModalAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
            )
            self.transformer = None
        elif backbone == "lstm":
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=n_transformer_layers,
                batch_first=True,
                dropout=dropout if n_transformer_layers > 1 else 0.0,
            )
            self.transformer = None
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_transformer_layers
            )

        # ── 5. Response Gate — physics (Bz) or analogy-inspired variant ─────
        # gate_type selects which gate is used; all three share the same
        # (encoder_output, x_sw) -> (gated_output, gate_values) interface so
        # nothing downstream needs to change based on which is active.
        # Default ("bz") is the original physics-grounded gate. The other
        # two are bio/electronics ANALOGY-inspired alternatives (see
        # analogy_gates.py docstring) — ablate them against "bz" to see
        # whether the extra structure helps, rather than assuming it does.
        if gate_type == "radiotrophic":
            self.bz_gate = RadiotrophicGate(d_model=d_model, vsw_idx=0, bz_idx=bz_feature_idx)
        elif gate_type == "cathode_anode":
            self.bz_gate = CathodeAnodeGate(d_model=d_model, vsw_idx=0, bz_idx=bz_feature_idx)
        else:
            self.bz_gate = BzPhysicsGate(
                bz_feature_idx=bz_feature_idx,
                bz_dur_idx=bz_dur_idx,
                d_model=d_model,
                bz_threshold=-5.0,
                gate_min=0.1,
            )

        # ── 6. Multi-Horizon Heads (3 horizons) ─────────
        self.heads = MultiHorizonHeads(
            d_model=d_model,
            hidden_dim=hidden_dim,
            n_horizons=n_horizons,
            dropout=dropout,
        )

        # ── 7. Optional Spectral Parameterization Head (additive) ───────────
        # Predicts real physics spectral parameters (kappa, theta) and
        # derives flux from them, as an alternative/additional way of
        # producing the same flux number the direct heads already predict.
        # Purely additive: does not replace self.heads, so both can be
        # compared and ablated independently.
        if use_spectral_head:
            self.spectral_head = SpectralParamHead(
                d_model=d_model, hidden_dim=hidden_dim, n_horizons=n_horizons,
                dropout=dropout, e_mev_default=2.0,
            )
        else:
            self.spectral_head = None

        # ── 8. Optional Magnetopause Geometry (Shue 1998) ───────────────────
        # Computes r0 (stand-off distance), alpha (flaring), and compression
        # ratio (GEO_radius / r0) from the last SW timestep, then projects
        # the 3 scalar features into d_model and ADDS them to the encoder
        # output before the gate.  Fully additive — does not change any
        # tensor shapes, so existing checkpoints remain loadable.
        if use_magnetopause:
            self.mag_geo = MagnetopauseGeometryFeatures(
                vsw_idx=0, bz_idx=bz_feature_idx, density_idx=None,
            )
            # 3 scalars (r0, alpha, compression) -> d_model
            self.mag_proj = nn.Sequential(
                nn.Linear(3, d_model),
                nn.Tanh(),
            )
        else:
            self.mag_geo  = None
            self.mag_proj = None

        # Track attention weights for interpretability
        self._last_attn_weights   = None
        self._last_gate_values    = None
        self._last_cross_weights  = None

    def forward(
        self,
        x_sw:       torch.Tensor,          # [B, T, n_sw_features]
        x_flux:     torch.Tensor,          # [B, T, 1]
        y_persist:  torch.Tensor,          # [B, n_horizons]
    ) -> dict:
        """
        Full forward pass.

        Returns dict with all predictions and intermediate values
        needed for the physics-informed loss.
        """
        # Sanitize inputs: NaN from CDF data gaps can corrupt tau_logit gradients
        # through the prop_delay linear interpolation (∂x_delayed/∂tau_frac = x_hi - x_lo)
        # Zeroing here breaks that gradient path before any computation.
        x_sw   = torch.nan_to_num(x_sw,   nan=0.0)
        x_flux = torch.nan_to_num(x_flux, nan=0.0)

        # Domain Adaptation Fix: If fine-tuning on GRASP, we might only have 1 feature (flux).
        # We zero-pad it back to n_sw_features so the pre-trained weights still match the shape.
        # Use self.n_sw_features (set in __init__) as the authoritative value, NOT a hardcoded 14.
        expected_f = self.n_sw_features
        if x_sw.shape[-1] != expected_f:
            padding = torch.zeros(*x_sw.shape[:-1], expected_f - x_sw.shape[-1], device=x_sw.device)
            x_sw = torch.cat([x_sw, padding], dim=-1)

        # ── 1. Apply adaptive propagation delay to solar wind ────────────────
        if self.ablation == "no_delay":
            x_sw_delayed = x_sw
            tau = torch.ones(x_sw.size(0), 1, device=x_sw.device) # Fixed 1.0 hr delay (already 1h shifted by OMNI ideally, but here we just pass raw)
        else:
            x_sw_delayed, tau = self.prop_delay(x_sw)

        # ── 2. Feature Fusion / Encoding ──────────────────────────────────────
        if self.backbone == "hybrid":
            # Separate SW and flux encoding, fused via cross-modal attention
            # instead of early concatenation.
            flux_repr = self.ssm(x_flux)                            # [B, d_model]

            # Run iTransformer's embedding + layers ONCE to get per-feature
            # tokens [B, F, d_model]; pool them ourselves for sw_pooled rather
            # than calling self.itransformer(...) a second time (which would
            # redundantly re-run the same embedding+layers).
            sw_tokens = self.itransformer.embedding(x_sw_delayed)   # [B, F, d_model]
            sw_attn_weights = []
            for layer in self.itransformer.layers:
                sw_tokens, aw = layer(sw_tokens)
                sw_attn_weights.append(aw)
            self._last_attn_weights = sw_attn_weights

            h, cross_weights = self.cross_modal(flux_repr, sw_tokens)  # [B, d_model]
            self._last_cross_weights = cross_weights
        else:
            # Concatenate delayed solar wind and flux along feature dimension
            x = torch.cat([x_sw_delayed, x_flux], dim=-1)         # [B, T, F+1]

            # Project to d_model
            x = self.input_proj(x)                                # [B, T, d_model]

            # Add positional embedding
            x = x + self.pos_emb[:, :x.size(1), :]
            x = self.emb_dropout(x)

            # ── 3. Temporal Transformer/LSTM Encoding ─────────────────────────────
            if hasattr(self, 'lstm') and self.lstm is not None:
                x, _ = self.lstm(x)                                   # [B, T, d_model]
            else:
                x = self.transformer(x)                               # [B, T, d_model]

            # Take the representation at the final time step
            h = x[:, -1, :]                                       # [B, d_model]

        # ── 4b. Optional Magnetopause Geometry — additive feature injection ──
        # Adds Shue (1998) compression state (r0, alpha, GEO/r0) to the
        # encoder output BEFORE the gate so the gate can see the current
        # magnetopause compression state when deciding how much to amplify
        # the signal.  Falls back gracefully if mag_geo is None.
        if self.mag_geo is not None:
            mag_feats = self.mag_geo(x_sw)          # dict of [B] tensors
            mag_vec = torch.stack([
                mag_feats["r0"],
                mag_feats["alpha"],
                mag_feats["compression"],
            ], dim=-1)                               # [B, 3]
            h = h + self.mag_proj(mag_vec)           # [B, d_model] additive

        # ── 4. Bz Physics Gate (Optional / Ablatable) ────────────────────────
        if self.ablation in ("no_bz_gate", "no_gate"):
            gated_repr = h
            gate_values = torch.ones(h.size(0), 1, device=h.device)
        else:
            gated_repr, gate_values = self.bz_gate(h, x_sw)
        
        self._last_gate_values = gate_values

        # ── 6. Multi-horizon predictions ──────────────────────────────────────────────
        outputs = self.heads(gated_repr, y_persist)
        outputs["tau"]         = tau
        outputs["gate_values"] = gate_values
        outputs["gate"]        = gate_values
        outputs["delay_loss"]  = self.prop_delay.regularization_loss()

        # ── 7. Optional spectral-parameterization prediction (additive) ─────
        # Does not affect outputs["flux_pred"] (the direct-head prediction
        # everything else in the pipeline consumes) — it's an additional,
        # independently-inspectable prediction for comparison/ablation.
        if self.spectral_head is not None:
            spec_out = self.spectral_head(gated_repr, e_mev=2.0)
            outputs["spectral_flux_pred"] = spec_out["log_flux_pred"]
            outputs["spectral_kappa"]     = spec_out["kappa"]
            outputs["spectral_theta"]     = spec_out["theta"]

        return outputs

    def get_interpretability_data(self) -> dict:
        """Return stored attention/gate data for paper figures."""
        return {
            "bz_gate_activations":     self._last_gate_values,
        }

    def freeze_encoder(self):
        """Freeze all encoder parameters for GOES→GRASP transfer learning."""
        modules = [self.prop_delay, self.bz_gate]
        if self.backbone == "hybrid":
            modules += [self.itransformer, self.ssm, self.cross_modal]
        else:
            modules += [self.input_proj, self.transformer]
        for module in modules:
            for p in module.parameters():
                p.requires_grad = False
        print(f"[STORMPhysNet] Encoder frozen for transfer learning (backbone={self.backbone}).")

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True
        print("[STORMPhysNet] All parameters unfrozen.")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Deep Ensemble Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class STORMPhysNetEnsemble(nn.Module):
    """
    Deep Ensemble of N independent STORM-PhysNet models.

    Deep ensembles are the gold standard for uncertainty quantification
    in space weather ML (Lakshminarayanan et al., NeurIPS 2017).

    Ensemble mean → point prediction (better than any single model)
    Ensemble std  → epistemic uncertainty (high during storms = physically correct)
    """

    def __init__(self, n_members: int = 5, **model_kwargs):
        super().__init__()
        self.n_members = n_members
        self.members   = nn.ModuleList([
            STORMPhysNet(**model_kwargs) for _ in range(n_members)
        ])
        print(f"[Ensemble] Created {n_members} models, "
              f"{self.members[0].count_parameters():,} params each.")

    def forward(
        self,
        x_sw:      torch.Tensor,
        x_flux:    torch.Tensor,
        y_persist: torch.Tensor,
    ) -> dict:
        """
        Forward pass through all ensemble members.

        Returns
        -------
        dict with:
            flux_pred  : [B, n_horizons] — ensemble mean
            flux_std   : [B, n_horizons] — ensemble std (uncertainty)
            all_preds  : [n_members, B, n_horizons] — individual predictions
            + other outputs from last member for loss computation
        """
        all_preds = []
        last_out  = None

        for member in self.members:
            out = member(x_sw, x_flux, y_persist)
            all_preds.append(out["flux_pred"])
            last_out = out

        stacked    = torch.stack(all_preds, dim=0)   # [M, B, H]
        ens_mean   = stacked.mean(dim=0)              # [B, H]
        ens_std    = stacked.std(dim=0)               # [B, H]

        last_out["flux_pred"]  = ens_mean
        last_out["flux_std"]   = ens_std
        last_out["all_preds"]  = stacked
        return last_out

    def predict_with_uncertainty(
        self,
        x_sw:      torch.Tensor,
        x_flux:    torch.Tensor,
        y_persist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (mean_prediction, uncertainty_std) for all horizons.
        Combines epistemic (ensemble) + aleatoric (log_var head) uncertainty.
        """
        with torch.no_grad():
            out = self.forward(x_sw, x_flux, y_persist)
            aleatoric = torch.exp(0.5 * out["log_var"])    # [B, H]
            epistemic = out["flux_std"]                     # [B, H]
            total_unc = torch.sqrt(aleatoric**2 + epistemic**2)
        return out["flux_pred"], total_unc

    def freeze_encoders(self):
        for m in self.members:
            m.freeze_encoder()
