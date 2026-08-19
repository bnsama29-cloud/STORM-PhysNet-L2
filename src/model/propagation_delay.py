"""
Adaptive Propagation Delay Module (Fixed version)

Sample-conditioned tau: previously tau was a single global nn.Parameter
shared by the entire dataset — "adaptive" only in the sense that it was
learned rather than hand-set, not in the sense of varying per storm/sample.
Physically, L1->GEO propagation delay depends on solar wind speed (faster
wind = shorter delay), so a per-sample tau conditioned on the early part of
the input window is a closer match to the actual physics and gives the
model room to behave differently for a fast CME-driven stream vs slow
solar wind. We keep the global learned parameter as a fallback bias term
so the module degrades gracefully / matches prior behavior at init.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePropagationDelay(nn.Module):
    def __init__(
        self,
        tau_init_hours: float = 1.0,   # Start at 1 hour
        tau_min_hours:  float = 0.5,   # 30 min
        tau_max_hours:  float = 1.5,   # 90 min
        n_sw_features:  int   = 14,
        cond_window:    int   = 5,     # how many early timesteps inform tau
        cond_hidden:    int   = 16,
    ):
        super().__init__()
        self.tau_min = tau_min_hours
        self.tau_max = tau_max_hours
        self.cond_window = cond_window

        # Global learned bias (kept from the original design) — acts as the
        # dataset-average delay and as a stable init for the per-sample net.
        init_norm = (tau_init_hours - tau_min_hours) / (tau_max_hours - tau_min_hours)
        init_norm = min(max(init_norm, 0.05), 0.95)  # avoid extremes
        init_logit = torch.log(torch.tensor(init_norm / (1.0 - init_norm)))
        self.tau_logit_bias = nn.Parameter(init_logit)

        # Small per-sample conditioning network: looks at the first
        # `cond_window` timesteps of solar wind (speed, Bz, etc.) and predicts
        # a per-sample ADJUSTMENT to the delay logit. Zero-initialized final
        # layer so at the start of training tau == the old global-scalar
        # behavior exactly, and the model only deviates as it learns to.
        self.tau_cond_net = nn.Sequential(
            nn.Linear(n_sw_features * cond_window, cond_hidden),
            nn.GELU(),
            nn.Linear(cond_hidden, 1),
        )
        nn.init.zeros_(self.tau_cond_net[-1].weight)
        nn.init.zeros_(self.tau_cond_net[-1].bias)

    def compute_tau(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T, F] solar wind input (pre-delay)
        Returns per-sample tau : [B, 1], each in [tau_min, tau_max].
        """
        B, T, Feat = x.shape
        w = min(self.cond_window, T)
        early_window = x[:, :w, :]                        # [B, w, F]
        if w < self.cond_window:
            # pad on the left if the sequence is shorter than expected
            pad = self.cond_window - w
            early_window = F.pad(early_window, (0, 0, pad, 0))
        cond_in = early_window.reshape(B, -1)              # [B, w*F]
        # zero-pad/truncate feature dim if it doesn't match n_sw_features
        expected = self.tau_cond_net[0].in_features
        if cond_in.size(1) != expected:
            if cond_in.size(1) < expected:
                cond_in = F.pad(cond_in, (0, expected - cond_in.size(1)))
            else:
                cond_in = cond_in[:, :expected]

        delta_logit = self.tau_cond_net(cond_in)            # [B, 1]
        logit = self.tau_logit_bias.unsqueeze(0) + delta_logit  # [B, 1]
        logit = torch.clamp(logit, -8.0, 8.0)
        tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(logit)
        return tau                                          # [B, 1]

    @property
    def tau(self) -> torch.Tensor:
        """Global bias-only delay (for logging/backward compat). Prefer
        compute_tau(x) for the actual per-sample value used in forward()."""
        logit = torch.clamp(self.tau_logit_bias, -8.0, 8.0)
        return self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(logit)

    def forward(self, x: torch.Tensor):
        """
        x : [B, T, F]
        Returns: delayed_x, tau  (tau is now [B, 1], per-sample)
        """
        tau = self.compute_tau(x)                           # [B, 1]

        B, T, Feat = x.shape
        # Convert delay (hours) to steps. Assumes data is hourly.
        # If your data is 5-min resolution, change the scaling.
        tau_steps = tau.clamp(min=0.0, max=float(T - 2))     # [B, 1]

        tau_floor = torch.floor(tau_steps).long()            # [B, 1]
        tau_frac  = tau_steps - tau_floor.float()            # [B, 1]

        idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)  # [B, T]

        idx_lo = (idx - tau_floor).clamp(0, T - 1)            # [B, T], per-sample shift
        idx_hi = (idx_lo + 1).clamp(0, T - 1)                 # [B, T]

        x_lo = torch.gather(x, 1, idx_lo.unsqueeze(-1).expand(B, T, Feat))
        x_hi = torch.gather(x, 1, idx_hi.unsqueeze(-1).expand(B, T, Feat))

        tau_frac_expand = tau_frac.unsqueeze(-1)              # [B, 1, 1] -> broadcasts over T,F
        x_delayed = (1.0 - tau_frac_expand) * x_lo + tau_frac_expand * x_hi
        return x_delayed, tau

    def regularization_loss(self) -> torch.Tensor:
        """Very light regularization – do not force tau too hard.
        Uses the global bias term (dataset-level tendency) as before;
        per-sample deviations are implicitly bounded by the sigmoid range."""
        tau = self.tau
        return F.relu(self.tau_min - tau) + F.relu(tau - self.tau_max)
