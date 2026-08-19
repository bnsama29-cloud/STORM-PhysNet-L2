"""
Physics-Informed Loss Function — fully elaborated.

Complete loss:
    L_total = Σ_h w_h · L_primary(h)       [weighted MSE per horizon]
            + λ_dst   · L_dst               [Dst prediction auxiliary]
            + λ_kp    · L_kp                [Kp prediction auxiliary]
            + λ_storm · L_storm_cls         [storm onset classification]
            + λ_mono  · L_monotonicity      [energy transfer rate bound]
            + λ_bz    · L_bz_response       [storm-time under-prediction penalty]
            + λ_smooth· L_smooth            [temporal smoothness]
            + λ_delay · L_delay_reg         [propagation delay regularization]
            + λ_var   · L_var_calib         [uncertainty calibration]

Storm-weighted asymmetric loss:
    During storm periods (storm_flag = 1):
        Under-prediction penalized 2.5× more than over-prediction
        (Under-predicting flux is more dangerous for satellite safety)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class LossWeights:
    """Hyperparameters for the physics-informed loss."""
    # Horizon weights (short horizon should be most accurate)
    horizon_weights:  list = None  # [1.0, 0.7, 0.5] for [1h, 6h, 12h]
    # NEW: scale physics constraints per horizon (Week 1: physics hurts 6h/12h)
    physics_horizon_scale: list = None  # e.g. [1.0, 0.0, 0.0]
    # Auxiliary task weights
    lambda_dst:       float = 0.10
    lambda_kp:        float = 0.05
    lambda_storm_cls: float = 0.15
    # Physics constraint weights
    lambda_mono:      float = 0.10   # monotonicity (energy transfer bound)
    lambda_bz:        float = 0.15   # Bz response penalty
    lambda_smooth:    float = 0.05   # temporal smoothness
    lambda_delay:     float = 0.01   # delay regularization
    lambda_var:       float = 0.05   # uncertainty calibration
    lambda_epsilon:   float = 0.08   # Akasofu epsilon coupling-function consistency
    # Storm asymmetry
    storm_under_penalty: float = 2.5   # multiplier for under-prediction in storms
    smooth_threshold:    float = 1.0   # max allowed |Δflux/Δt| in log units

    def __post_init__(self):
        if self.horizon_weights is None:
            self.horizon_weights = [1.0, 0.7, 0.5]
        if self.physics_horizon_scale is None:
            # Default after Week 1: physics only on short head
            self.physics_horizon_scale = [1.0, 0.0, 0.0]


class PhysicsInformedLoss(nn.Module):
    """
    Complete physics-informed loss for STORM-PhysNet.

    Physics terms provide paper novelty over
    all existing LSTM/Transformer papers which use plain MSE.
    """

    def __init__(self, weights: LossWeights = None):
        super().__init__()
        self.w = weights or LossWeights()
        hw = torch.tensor(self.w.horizon_weights, dtype=torch.float32)
        self.register_buffer("horizon_weights", hw / hw.sum())
        phs = torch.tensor(self.w.physics_horizon_scale, dtype=torch.float32)
        self.register_buffer("physics_horizon_scale", phs)

    # ──────────────────────────────────────────────────────────────────────────
    # Primary Loss
    # ──────────────────────────────────────────────────────────────────────────

    def _primary_loss(
        self,
        flux_pred:  torch.Tensor,    # [B, H]
        y_flux:     torch.Tensor,    # [B, H]
        storm_flag: torch.Tensor,    # [B, 1]
        log_var:    torch.Tensor,    # [B, H]
    ) -> torch.Tensor:
        """
        beta-NLL loss (Stirn et al., NeurIPS 2023).
        Prevents variance branch from hijacking mean prediction gradients.
        beta=0.5 is the theoretically proven optimal value.

        Standard NLL: L = 0.5*(log_var + sq_err/var)
        beta-NLL:     L = 0.5*sg(var^beta)*(log_var + sq_err/var)
        sg = stop_gradient through weight only.
        """
        var    = torch.exp(log_var).clamp(min=1e-6, max=1e6)   # [B, H]
        sq_err = (flux_pred - y_flux).pow(2)

        # beta=0.5: weight by sqrt(var), stop grad so weight does not affect mean head
        weight = var.detach().pow(0.5)
        nll    = 0.5 * weight * (sq_err / var + log_var)

        # Asymmetric storm penalty: under-prediction is more dangerous for satellites
        under_pred = (flux_pred < y_flux).float()
        storm_mult = 1.0 + (self.w.storm_under_penalty - 1.0) * storm_flag * under_pred
        nll = nll * storm_mult

        nll_weighted = (nll * self.horizon_weights.unsqueeze(0)).sum(dim=1)
        return nll_weighted.mean()

    # ──────────────────────────────────────────────────────────────────────────
    # Physics Constraints
    # ──────────────────────────────────────────────────────────────────────────

    def _epsilon_coupling_loss(
        self,
        flux_residual: torch.Tensor,  # [B, H] predicted corrections
        y_flux:        torch.Tensor,  # [B, H]
        y_persist:     torch.Tensor,  # [B, H]
        x_sw:          torch.Tensor,  # [B, T, F]
        vsw_idx:       int = 0,
        bz_idx:        int = 1,
        bt_idx:        int = 2,
    ) -> torch.Tensor:
        """
        Akasofu epsilon coupling-function consistency loss (electromagnetism).

        The Akasofu (1981) epsilon parameter is the standard solar-wind-to-
        magnetosphere energy coupling function used throughout space weather
        physics:

            epsilon ~ Vsw * B_t^2 * sin^4(theta_c / 2)

        where theta_c = clock angle = atan2(By, Bz) is the IMF orientation in
        the GSM Y-Z plane, and B_t is the transverse (Bz-Bt plane) field
        magnitude. This differs from the existing `_monotonicity_loss` term,
        which only bounds the RATE of predicted change using Vsw^2 * |Bz| (no
        clock-angle dependence, no B_t). Epsilon instead captures the full
        directional geometry of reconnection efficiency: coupling is maximal
        for a purely southward IMF (clock angle = 180 deg -> sin^4 term = 1)
        and drops off as the field rotates toward northward or duskward/
        dawnward orientations, even at the same |Bz|.

        We don't have By in the current feature set (only bz, bt, vsw), so we
        approximate the clock angle contribution using bt as the transverse
        field proxy and southward Bz as the driving component:
            sin^2(theta_c/2) ~ Bz_south / B_t   (Bz_south = max(0, -Bz))
        which recovers the correct limits (=1 when field is purely
        southward and B_t=|Bz|; ->0 as B_t >> |Bz|, i.e. more northward/
        transverse-dominated field). If a true By feature is added later,
        replace this with the exact clock angle.

        Physics constraint: the magnitude of the model's predicted deviation
        from persistence should scale with epsilon at least loosely in rank
        order — i.e. when epsilon predicts strong coupling, the model should
        not be predicting near-zero deviation, and vice versa. We enforce
        this softly (correlation-style penalty on normalized magnitudes)
        rather than an exact functional form, since the exact proportionality
        constant is not well constrained for GEO relativistic electron flux
        (unlike for Dst, where epsilon-based coupling is well calibrated).
        """
        if x_sw.shape[-1] <= max(vsw_idx, bz_idx, bt_idx):
            return torch.tensor(0.0, device=flux_residual.device)

        vsw = x_sw[:, -1, vsw_idx].clamp(min=1.0)      # [B], km/s
        bz  = x_sw[:, -1, bz_idx]                       # [B], nT
        bt  = x_sw[:, -1, bt_idx].clamp(min=1e-3)       # [B], nT (avoid /0)

        bz_south = torch.clamp(-bz, min=0.0)             # [B]
        sin2_half_clock = (bz_south / bt).clamp(0.0, 1.0)  # approx sin^2(theta_c/2)
        sin4_half_clock = sin2_half_clock.pow(2)           # approx sin^4(theta_c/2)

        # Akasofu epsilon (unnormalized units; only relative scale matters here)
        epsilon = vsw * bt.pow(2) * sin4_half_clock         # [B]
        # Normalize per-batch so the loss is scale-free across storms/quiet time
        eps_norm = epsilon / (epsilon.mean().detach().clamp(min=1e-6) + 1e-6)
        eps_norm = eps_norm.clamp(0.0, 10.0)                # avoid outlier domination

        # True and predicted deviation-from-persistence magnitude, per horizon
        true_dev = (y_flux - y_persist).abs()               # [B, H]
        pred_dev = flux_residual.abs()                      # [B, H]

        # Soft rank-consistency: when eps_norm is high, both true and predicted
        # deviation tend to be high (that's the physics); penalize the model
        # only where it under-reacts relative to what coupling strength implies
        # the TRUE deviation would justify — i.e. don't punish the model for
        # correctly predicting small deviation under high epsilon if the true
        # deviation actually was small (that's not a physics violation, that's
        # just a hard-to-predict case). We instead penalize systematic
        # under-response: cases where epsilon is high AND true deviation is
        # high AND predicted deviation is much smaller than true deviation.
        high_coupling = (eps_norm > 1.0).float().unsqueeze(1)          # [B,1]
        under_response = F.relu(true_dev - pred_dev) * high_coupling   # [B,H]
        return under_response.mean()

    def _monotonicity_loss(
        self,
        flux_residual: torch.Tensor,  # [B, H] predicted corrections
        x_sw:          torch.Tensor,  # [B, T, F]
        bz_idx:        int = 1,
        vsw_idx:       int = 0,
    ) -> torch.Tensor:
        """
        Energy transfer rate constraint (physics-informed):
        The rate of flux change should be bounded by solar wind ram pressure
        and southward Bz (proxy for reconnection rate / energy input).

        Physically: dF/dt ≤ α · Vsw² · max(0, -Bz) / normalization
        Violation = model predicts flux increase faster than physics allows.
        """
        # Current solar wind state (last time step)
        if x_sw.shape[-1] <= bz_idx:
            return torch.tensor(0.0, device=flux_residual.device)
            
        bz_now  = x_sw[:, -1, bz_idx]                       # [B]
        vsw_now = x_sw[:, -1, vsw_idx]                      # [B]

        # Maximum physically plausible flux rate (log units / hour)
        # Based on Vsw and southward Bz driving
        max_rate = 0.001 * vsw_now.pow(2).clamp(0, 500000) * \
                   torch.clamp(-bz_now, min=0.0) / 1000.0
        max_rate = max_rate.clamp(0.01, 2.0)                # [B]

        # Violation: predicted correction exceeds max rate per horizon
        horizons = torch.tensor([1.0, 6.0, 12.0], device=flux_residual.device)
        max_per_horizon = max_rate.unsqueeze(1) * horizons.unsqueeze(0)  # [B,H]
        violation = F.relu(flux_residual.abs() - max_per_horizon)
        return violation.mean()

    def _bz_response_loss(
        self,
        flux_pred:  torch.Tensor,    # [B, H]
        y_flux:     torch.Tensor,    # [B, H]
        x_sw:       torch.Tensor,    # [B, T, F]
        y_persist:  torch.Tensor,    # [B, H]
        bz_idx:     int = 1,
        bz_dur_idx: int = 5,
    ) -> torch.Tensor:
        """
        Penalize model when it IGNORES strong southward Bz:
        If Bz < -10 nT sustained for > 6h, the model MUST predict
        flux deviation from persistence (either dropout or enhancement).

        This is the core physics the Bz gate should learn.
        """
        if x_sw.shape[-1] <= bz_idx:
            return torch.tensor(0.0, device=flux_pred.device)
            
        bz_now      = x_sw[:, -1, bz_idx]           # [B]
        bz_dur      = x_sw[:, -1, min(bz_dur_idx,
                                       x_sw.size(-1)-1)]
        strong_storm = (bz_now < -10) & (bz_dur > 6)

        if not strong_storm.any():
            return torch.tensor(0.0, device=flux_pred.device)

        # During strong storms, model should deviate from persistence
        deviation_pred = (flux_pred - y_persist).abs()     # [B, H]
        deviation_true = (y_flux   - y_persist).abs()      # [B, H]

        # Penalty: model predicts too close to persistence when storm is active
        penalty = F.relu(deviation_true - deviation_pred)[strong_storm]
        return penalty.mean() if len(penalty) > 0 else torch.tensor(0.0,
                                                         device=flux_pred.device)

    def _smooth_loss(
        self,
        flux_pred: torch.Tensor,   # [B, H]
    ) -> torch.Tensor:
        """
        Temporal smoothness: penalize unrealistic jumps between horizons.
        |ŷ(t+6h) - ŷ(t+1h)| and |ŷ(t+12h) - ŷ(t+6h)| should be bounded.
        """
        if flux_pred.size(1) < 2:
            return torch.tensor(0.0, device=flux_pred.device)
        diffs     = flux_pred[:, 1:] - flux_pred[:, :-1]
        violation = F.relu(diffs.abs() - self.w.smooth_threshold)
        return violation.mean()

    def _uncertainty_calibration_loss(
        self,
        log_var:   torch.Tensor,   # [B, H]
        sq_errors: torch.Tensor,   # [B, H]
    ) -> torch.Tensor:
        """
        Calibration: predicted variance should match actual squared errors.
        Penalize overconfident predictions.
        """
        pred_var = torch.exp(log_var)
        calib    = F.mse_loss(pred_var, sq_errors.detach())
        return calib

    # ──────────────────────────────────────────────────────────────────────────
    # Master Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        outputs:    dict,            # model output dict
        targets:    dict,            # batch dict with y_flux, y_dst, etc.
        x_sw:       torch.Tensor,    # [B, T, n_sw]
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute full physics-informed loss.

        Returns
        -------
        total_loss : torch.Tensor (scalar)
        loss_components : dict  (for logging / tensorboard)
        """
        flux_pred    = outputs["flux_pred"]       # [B, H]
        residuals    = outputs.get("flux_residual", torch.zeros_like(flux_pred))  # [B, H]
        log_var      = outputs.get("log_var", torch.zeros_like(flux_pred))        # [B, H]
        dst_pred     = outputs.get("dst_pred", torch.zeros_like(flux_pred))       # [B, H]
        kp_pred      = outputs.get("kp_pred", torch.zeros_like(flux_pred))        # [B, H]
        storm_logits = outputs.get("storm_logits", torch.zeros((flux_pred.size(0), 1), device=flux_pred.device))   # [B, 1] raw logits
        delay_loss   = outputs.get("delay_loss", torch.tensor(0.0, device=flux_pred.device))     # scalar

        # Protect against NaN propagation from any upstream component
        flux_pred = torch.nan_to_num(flux_pred, nan=0.0, posinf=6.0, neginf=-2.0)
        log_var   = torch.nan_to_num(log_var,   nan=0.0, posinf=4.0, neginf=-4.0)
        x_sw      = torch.nan_to_num(x_sw,      nan=0.0)

        y_flux     = torch.nan_to_num(targets["y_flux"],    nan=0.0)  # [B, H]
        y_dst      = torch.nan_to_num(targets["y_dst"],     nan=0.0)  # [B, H]
        y_kp       = torch.nan_to_num(targets["y_kp"],      nan=3.0)  # [B, H]
        y_storm    = targets["y_storm"]                               # [B, 1]
        storm_flag = targets["storm_flag"]                            # [B, 1]
        y_persist  = torch.nan_to_num(targets["y_persist"], nan=0.0)  # [B, H]

        sq_errors = (flux_pred - y_flux).pow(2).detach()

        # ── Primary (asymmetric storm-weighted NLL) ──────────────────────────
        L_primary = self._primary_loss(flux_pred, y_flux, storm_flag, log_var)

        # Auxiliary tasks — normalize targets so MSE scale matches primary loss (~1.0)
        # Dst ranges -300..+20 nT  → divide by 100 → scale -3..+0.2  → MSE ~ 0.1-1.0
        # Kp  ranges   0.. +9      → divide by  9  → scale  0.. 1.0  → MSE ~ 0.1-0.5
        DST_SCALE, KP_SCALE = 100.0, 9.0
        L_dst   = F.mse_loss(torch.nan_to_num(dst_pred, 0.0) / DST_SCALE,
                              y_dst / DST_SCALE)
        L_kp    = F.mse_loss(torch.nan_to_num(kp_pred,  0.0) / KP_SCALE,
                              y_kp  / KP_SCALE)
        # BCEWithLogitsLoss = sigmoid + BCE fused in log-space → numerically stable
        # Never produces NaN from large logit values (unlike BCE on sigmoid output)
        L_storm = F.binary_cross_entropy_with_logits(
            torch.nan_to_num(storm_logits, 0.0), y_storm)

        # ── Physics constraints (horizon-conditioned; Week 1 finding) ────────
        # Per-horizon residual / error used so long-horizon heads are not pulled
        # by mono/bz/eps terms that hurt 6h/12h PE.
        H = flux_pred.size(1)
        scale = self.physics_horizon_scale.to(flux_pred.device)
        if scale.numel() < H:
            pad = torch.zeros(H - scale.numel(), device=flux_pred.device)
            scale = torch.cat([scale, pad], dim=0)
        scale = scale[:H]  # [H]

        # Mask residuals/preds so physics is dominated by short horizon when
        # physics_horizon_scale = [1, 0, 0]
        res_h = residuals * scale.view(1, -1)
        L_mono   = self._monotonicity_loss(res_h, x_sw)
        L_eps    = self._epsilon_coupling_loss(res_h, y_flux, y_persist, x_sw)
        L_bz_raw = self._bz_response_loss(flux_pred, y_flux, x_sw, y_persist)
        L_bz     = L_bz_raw * scale[0]          # only short-horizon physics weight
        long_scale = scale[1:].mean() if H > 1 else scale[0]
        L_smooth = self._smooth_loss(flux_pred) * long_scale
        L_var    = self._uncertainty_calibration_loss(log_var, sq_errors)
        L_delay  = delay_loss

        # ── Total ─────────────────────────────────────────────────────────────
        total = (L_primary
                 + self.w.lambda_dst       * L_dst
                 + self.w.lambda_kp        * L_kp
                 + self.w.lambda_storm_cls * L_storm
                 + self.w.lambda_mono      * L_mono
                 + self.w.lambda_bz        * L_bz
                 + self.w.lambda_smooth    * L_smooth
                 + self.w.lambda_var       * L_var
                 + self.w.lambda_delay     * L_delay
                 + self.w.lambda_epsilon   * L_eps)

        components = {
            "primary":    L_primary.item(),
            "dst":        L_dst.item(),
            "kp":         L_kp.item(),
            "storm_cls":  L_storm.item(),
            "monotone":   L_mono.item(),
            "bz_resp":    L_bz.item() if hasattr(L_bz, "item") else float(L_bz),
            "smooth":     L_smooth.item(),
            "var_calib":  L_var.item(),
            "delay_reg":  L_delay.item() if hasattr(L_delay, "item") else float(L_delay),
            "epsilon":    L_eps.item() if hasattr(L_eps, "item") else float(L_eps),
            "total":      total.item(),
        }
        return total, components
