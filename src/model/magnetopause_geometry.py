"""
Magnetopause Geometry Features — a grounded version of "a sphere around
the satellite/magnetosphere that compresses during storms."

REFRAMING NOTE: the literal idea of one sphere touching both the Sun's
surface and GEO orbit isn't physical — the Sun-Earth distance (~1 AU,
~215 solar radii) and GEO orbital radius (~6.6 Earth radii) differ by
about four orders of magnitude, so no single boundary meaningfully
touches both. What IS real physics, and is exactly what you were reaching
for, is the MAGNETOPAUSE: the boundary between the solar wind and Earth's
magnetosphere, which is genuinely compressed (moves closer to Earth) by
enhanced solar wind dynamic pressure — the standard Shue et al. (1998)
empirical magnetopause model describes this boundary's shape and how its
stand-off distance shrinks during storms. Your satellite (GEO, ~6.6 Re)
can end up OUTSIDE the compressed magnetopause during extreme storms,
which is physically significant for radiation belt dynamics (a satellite
crossing the boundary experiences a qualitatively different environment).

This module computes the Shue magnetopause stand-off distance and shape
parameter from real inputs (solar wind dynamic pressure, IMF Bz) already
in your feature set, and outputs both:
  1. A scalar "compression state" feature (r0, the stand-off distance)
  2. The satellite's normalized radial position relative to the boundary
     (r_sat / r0), which flips from <1 (inside magnetosphere) to >1
     (outside, i.e. GEO is briefly in the solar wind) during extreme
     compression — a physically meaningful regime-change indicator.

The "spherical vs Cartesian/polar" idea you mentioned maps onto this
directly: the Shue model IS a polar-coordinate description
(r(theta) as a function of the angle from the Sun-Earth line), and we
also provide a simple Cartesian projection for use as a feature.
"""

import torch
import torch.nn as nn


def shue_magnetopause_r0(pdyn: torch.Tensor, bz: torch.Tensor) -> torch.Tensor:
    """
    Shue et al. (1998) empirical magnetopause stand-off distance (subsolar
    point), in Earth radii:

        r0 = (10.22 + 1.29 * tanh(0.184 * (Bz + 8.14))) * Pdyn^(-1/6.6)

    pdyn : solar wind dynamic pressure, nPa. If you don't have this
           directly, it's commonly approximated as pdyn ~ 1.6726e-6 * n *
           Vsw^2 (n in cm^-3, Vsw in km/s) — see note in shue_r0_from_sw().
    bz   : IMF Bz, nT (signed; southward = negative, matching the rest of
           this codebase's convention).
    """
    pdyn = pdyn.clamp(min=0.1)  # avoid div-by-zero / negative pressure
    r0 = (10.22 + 1.29 * torch.tanh(0.184 * (bz + 8.14))) * pdyn.pow(-1.0 / 6.6)
    return r0


def shue_alpha(pdyn: torch.Tensor, bz: torch.Tensor) -> torch.Tensor:
    """Shue et al. (1998) shape (flaring) parameter alpha."""
    pdyn = pdyn.clamp(min=0.1)
    alpha = (0.58 - 0.007 * bz) * (1.0 + 0.024 * torch.log(pdyn))
    return alpha


def shue_r_of_theta(r0: torch.Tensor, alpha: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Full Shue magnetopause shape in polar coordinates (theta = angle from
    the Sun-Earth line, 0 = subsolar point):
        r(theta) = r0 * (2 / (1 + cos(theta)))^alpha
    This is the "spherical/polar description that compresses" you were
    describing — theta=0 gives the subsolar stand-off distance r0, and
    the boundary flares out (larger r) toward the flanks/tail as theta
    increases.
    """
    return r0.unsqueeze(-1) * (2.0 / (1.0 + torch.cos(theta))).pow(alpha.unsqueeze(-1))


def shue_r0_from_sw(vsw: torch.Tensor, n_density: torch.Tensor, bz: torch.Tensor):
    """
    Convenience: derive Pdyn from Vsw + density, then r0/alpha. Use this
    if your feature set includes proton/solar-wind density; if not, see
    MagnetopauseGeometryFeatures.forward for a Vsw-only proxy fallback.
    pdyn [nPa] ~= 1.6726e-6 * n[cm^-3] * v[km/s]^2
    """
    pdyn = 1.6726e-6 * n_density.clamp(min=0.1) * vsw.clamp(min=1.0).pow(2)
    r0 = shue_magnetopause_r0(pdyn, bz)
    alpha = shue_alpha(pdyn, bz)
    return r0, alpha, pdyn


GEO_RADIUS_RE = 6.6  # geostationary orbit radius, in Earth radii


class MagnetopauseGeometryFeatures(nn.Module):
    """
    Computes real-time magnetopause compression features from the
    existing solar wind input and exposes them as an additive feature
    vector the encoder can use, plus a scalar "compression ratio"
    diagnostic useful for the physics loss / evaluation plots.

    Falls back to a Vsw^2-only dynamic-pressure proxy if no density
    feature index is provided (set density_idx=None), since this
    project's SW_FEATURES may not include measured density for all
    dataset variants — the proxy preserves the right qualitative
    behavior (faster wind -> more compression) even without density.
    """

    def __init__(
        self,
        vsw_idx: int = 0,
        bz_idx:  int = 1,
        density_idx: int = None,   # set to a real column index if available
        n_theta_samples: int = 8,  # how many angles to sample the boundary at
    ):
        super().__init__()
        self.vsw_idx, self.bz_idx = vsw_idx, bz_idx
        self.density_idx = density_idx
        thetas = torch.linspace(0.0, 2.0, n_theta_samples)  # radians, subsolar to ~115deg
        self.register_buffer("thetas", thetas)

    def forward(self, x_sw: torch.Tensor) -> dict:
        """
        x_sw : [B, T, F], uses the LAST timestep (current state).
        Returns dict of [B, ...] tensors:
          r0            : subsolar stand-off distance (Re)
          alpha         : boundary flaring parameter
          compression   : GEO_RADIUS_RE / r0  (>1 means GEO is outside the
                          magnetopause at the subsolar point — extreme
                          compression regime)
          boundary_polar: [B, n_theta_samples] boundary radius at sampled
                          angles (the "polar sphere" you asked about)
        """
        last = x_sw[:, -1, :]
        vsw = last[:, self.vsw_idx].clamp(min=1.0)
        bz = last[:, self.bz_idx]

        if self.density_idx is not None and self.density_idx < x_sw.size(-1):
            n_density = last[:, self.density_idx].clamp(min=0.1)
            pdyn = 1.6726e-6 * n_density * vsw.pow(2)
        else:
            # Density-free proxy: assume nominal solar wind density (~5
            # cm^-3) scaled by Vsw^2 only. Preserves relative compression
            # trend (faster wind -> smaller r0) even without a measured
            # density channel; do not use the absolute r0 value as exact.
            pdyn = 1.6726e-6 * 5.0 * vsw.pow(2)

        r0 = shue_magnetopause_r0(pdyn, bz)          # [B]
        alpha = shue_alpha(pdyn, bz)                  # [B]
        compression = GEO_RADIUS_RE / r0.clamp(min=1e-3)  # [B]

        theta = self.thetas.unsqueeze(0).expand(x_sw.size(0), -1)  # [B, n_theta]
        boundary_polar = shue_r_of_theta(r0, alpha, theta)          # [B, n_theta]

        return {
            "r0": r0,
            "alpha": alpha,
            "compression": compression,
            "boundary_polar": boundary_polar,
        }

    def feature_dim(self) -> int:
        """Number of scalar features this module contributes if you
        concatenate r0, alpha, compression (not boundary_polar, which is
        diagnostic-only) onto the model's input or encoder representation."""
        return 3
