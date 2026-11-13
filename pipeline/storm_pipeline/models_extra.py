"""Control models: residual-head Transformer and flux-only wrapper."""
import math, torch, torch.nn as nn
from .data import KEEP_FOR_FLUX_ONLY


class _PE(nn.Module):
    def __init__(self, d, n=512):
        super().__init__()
        pe = torch.zeros(n, d); pos = torch.arange(n).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x): return x + self.pe[:, : x.size(1)]


class FFNResTransformer(nn.Module):
    """Plain Transformer, d_model=128, 2 layers, 4 heads, d_ff=256 (same as STORM's encoder), last-token head,
    OPTIONAL residual-over-persistence output. No delay, no gate, no physics loss."""

    def __init__(self, n_sw_features, d_model=128, nhead=4, layers=2, d_ff=256, n_h=3, residual=True):
        super().__init__()
        self.residual = residual
        self.emb = nn.Linear(n_sw_features + 1, d_model); self.pos = _PE(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_ff, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers); self.fc = nn.Linear(d_model, n_h)

    def forward(self, x_sw, x_flux, y_persist=None, **kw):
        x = torch.cat([torch.nan_to_num(x_sw), torch.nan_to_num(x_flux)], -1)
        h = self.enc(self.pos(self.emb(x)))[:, -1]
        out = self.fc(h)
        if self.residual and y_persist is not None: out = y_persist + out
        return {"flux_pred": out, "delay_loss": torch.zeros((), device=out.device), "log_var": torch.zeros_like(out)}


class FluxOnly(nn.Module):
    """Wrap a model so it sees NO solar-wind/geomagnetic information: only flux history, the within-hour flux std and
    the observed-hour mask."""

    def __init__(self, inner):
        super().__init__(); self.inner = inner
        m = torch.zeros(1, 1, 17); m[..., KEEP_FOR_FLUX_ONLY] = 1.0; self.register_buffer("mask", m)

    def forward(self, x_sw, x_flux, y_persist=None, **kw):
        return self.inner(x_sw * self.mask, x_flux, y_persist=y_persist)


class FixedTwoPointAverage(nn.Module):
    """Non-learned replacement for STORM-PhysNet's delay stage (reviewer control, AGERS 2026).

    Reproduces exactly what the learned stage computes at its saturated value tau = 1.5 h: the output at hour t is the
    equal-weight average of hours t-1 and t (hour 0 averages hours 0 and 1, as in the learned stage). No parameters.
    Same interface as src.model.propagation_delay.AdaptivePropagationDelay."""

    TAU = 1.5

    def forward(self, x):
        B, T, F = x.shape
        idx = torch.arange(T, device=x.device)
        lo = (idx - 1).clamp(0, T - 1); hi = (lo + 1).clamp(0, T - 1)
        tau = torch.full((B, 1), self.TAU, device=x.device, dtype=x.dtype)
        return 0.5 * (x[:, lo] + x[:, hi]), tau

    @property
    def tau(self):
        return torch.tensor(self.TAU)

    def regularization_loss(self):
        return torch.zeros(())


class SmoothedInput(nn.Module):
    """Apply FixedTwoPointAverage to all auxiliary channels (x_sw) before an inner model, as STORM-PhysNet does."""

    def __init__(self, inner):
        super().__init__(); self.inner = inner; self.smooth = FixedTwoPointAverage()

    def forward(self, x_sw, x_flux, y_persist=None, **kw):
        xs, _ = self.smooth(torch.nan_to_num(x_sw))
        return self.inner(xs, x_flux, y_persist=y_persist)
