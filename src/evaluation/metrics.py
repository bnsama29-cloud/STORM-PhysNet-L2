"""
Evaluation metrics for space weather forecasting.
Implements all standard space weather skill scores + storm-specific metrics.

Metrics reported in paper:
    PE   - Prediction Efficiency (standard in space weather)
    RMSE - Root Mean Square Error (log flux)
    R²   - Coefficient of determination
    HSS  - Heidke Skill Score (event-based)
    POD  - Probability of Detection
    FAR  - False Alarm Rate
    CSI  - Critical Success Index (Threat Score)
    BIAS - Frequency bias

All metrics computed separately for:
    1. All periods (aggregate)
    2. Quiet periods (Kp < 3)
    3. Storm periods (Kp ≥ 5)
    4. Intense storm periods (Kp ≥ 7)
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricsResult:
    """Container for all computed metrics."""
    # Regression metrics
    rmse:     float = 0.0
    mae:      float = 0.0
    r2:       float = 0.0
    pe:       float = 0.0   # Prediction Efficiency = 1 - MSE/Var(y)
    pe_pers:  float = 0.0   # Prediction Efficiency vs Persistence = 1 - MSE/MSE_pers
    bias:     float = 0.0   # mean(ŷ - y)
    corr:     float = 0.0   # Pearson correlation

    # Event-based metrics (flux threshold crossing)
    hss:      float = 0.0
    pod:      float = 0.0
    far:      float = 0.0
    csi:      float = 0.0

    # Uncertainty metrics
    picp:     float = 0.0   # Prediction Interval Coverage Probability
    mpiw:     float = 0.0   # Mean Prediction Interval Width

    # Sample counts
    n_total:  int = 0
    n_storm:  int = 0
    period:   str = "all"

    def __str__(self) -> str:
        return (f"[{self.period:10s}] "
                f"RMSE={self.rmse:.4f} | PE={self.pe:.4f} | PE_pers={self.pe_pers:.4f} | "
                f"R²={self.r2:.4f} | HSS={self.hss:.4f} | "
                f"POD={self.pod:.4f} | FAR={self.far:.4f} | "
                f"N={self.n_total:,}")


def prediction_efficiency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    PE = 1 - MSE(y, ŷ) / Var(y)
    Standard space weather forecasting skill metric.
    PE = 1.0 → perfect; PE = 0.0 → same as climatological mean; PE < 0 → worse than mean.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    var = np.var(y_true)
    return float(1.0 - mse / (var + 1e-10))

def prediction_efficiency_pers(y_true: np.ndarray, y_pred: np.ndarray, y_pers: np.ndarray) -> float:
    """
    PE_pers = 1 - MSE(y, ŷ) / MSE(y, y_pers)
    PE_pers = 1.0 → perfect; PE_pers = 0.0 → same as persistence; PE_pers < 0 → worse than persistence.
    """
    mse_pred = np.mean((y_true - y_pred) ** 2)
    mse_pers = np.mean((y_true - y_pers) ** 2)
    return float(1.0 - mse_pred / (mse_pers + 1e-10))


def heidke_skill_score(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    threshold: float = 4.0,  # log10(10^4) = 4 e/cm²/s/sr → radiation hazard
) -> tuple[float, float, float, float]:
    """
    Event-based contingency metrics for flux threshold exceedance.

    threshold : float
        log10 flux threshold for "event" classification.
        Default 4.0 = 10^4 pfu (moderate radiation hazard level).

    Returns
    -------
    hss, pod, far, csi : float
    """
    obs_event  = y_true >= threshold
    pred_event = y_pred >= threshold

    TP = np.sum( obs_event &  pred_event)   # Hit
    FP = np.sum(~obs_event &  pred_event)   # False Alarm
    FN = np.sum( obs_event & ~pred_event)   # Miss
    TN = np.sum(~obs_event & ~pred_event)   # Correct Rejection

    N = TP + FP + FN + TN + 1e-10

    # POD = TP / (TP + FN)
    pod = TP / max(TP + FN, 1)

    # FAR = FP / (FP + TP)
    far = FP / max(FP + TP, 1)

    # CSI = TP / (TP + FP + FN)
    csi = TP / max(TP + FP + FN, 1)

    # HSS = 2(TP·TN - FP·FN) / [(TP+FN)(FN+TN) + (TP+FP)(FP+TN)]
    numer = 2 * (TP * TN - FP * FN)
    denom = ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN) + 1e-10)
    hss = numer / denom

    return float(hss), float(pod), float(far), float(csi)


def compute_metrics(
    y_true:        np.ndarray,          # [N] true log flux
    y_pred:        np.ndarray,          # [N] predicted log flux
    storm_flags:   Optional[np.ndarray] = None,   # [N] 1=storm, 0=quiet
    kp:            Optional[np.ndarray] = None,    # [N] Kp values
    pred_std:      Optional[np.ndarray] = None,    # [N] prediction uncertainty
    y_pers:        Optional[np.ndarray] = None,    # [N] persistence baseline
    flux_threshold: float = 4.0,
    period:        str = "all",
) -> MetricsResult:
    """Compute full set of metrics for a prediction period."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t  = y_true[mask]
    y_p  = y_pred[mask]
    sf   = storm_flags[mask] if storm_flags is not None else np.zeros(len(y_t))
    ypers = y_pers[mask] if y_pers is not None else None

    rmse = float(np.sqrt(np.mean((y_t - y_p)**2)))
    mae  = float(np.mean(np.abs(y_t - y_p)))
    bias = float(np.mean(y_p - y_t))
    pe   = prediction_efficiency(y_t, y_p)
    pe_pers = prediction_efficiency_pers(y_t, y_p, ypers) if ypers is not None else 0.0
    r2   = float(np.corrcoef(y_t, y_p)[0, 1]**2) if len(y_t) > 1 else 0.0
    corr = float(stats.pearsonr(y_t, y_p)[0]) if len(y_t) > 1 else 0.0

    hss, pod, far, csi = heidke_skill_score(y_t, y_p, flux_threshold)

    # Uncertainty coverage (if std provided)
    picp, mpiw = 0.0, 0.0
    if pred_std is not None:
        std = pred_std[mask]
        lower = y_p - 1.96 * std
        upper = y_p + 1.96 * std
        picp  = float(np.mean((y_t >= lower) & (y_t <= upper)))
        mpiw  = float(np.mean(upper - lower))

    return MetricsResult(
        rmse=rmse, mae=mae, r2=r2, pe=pe, pe_pers=pe_pers, bias=bias, corr=corr,
        hss=hss, pod=pod, far=far, csi=csi,
        picp=picp, mpiw=mpiw,
        n_total=len(y_t), n_storm=int(sf.sum()),
        period=period,
    )


class StormEvaluator:
    """
    Comprehensive storm-period evaluation.
    Computes metrics separately for all / quiet / storm / intense storm periods.
    Note: The storm/quiet split here is Kp-based (Kp >= 5) by default, which differs
    from the paper's primary Dst-based definition (Dst <= -50 nT) used for PE_st,6h.
    This is the key evaluation that existing papers DON'T do.
    """

    def __init__(
        self,
        kp_storm_threshold:   float = 5.0,
        kp_intense_threshold: float = 7.0,
        flux_threshold:       float = 4.0,  # log10
    ):
        self.kp_storm    = kp_storm_threshold
        self.kp_intense  = kp_intense_threshold
        self.flux_thr    = flux_threshold

    def evaluate_all(
        self,
        y_true:    np.ndarray,           # [N, H] true log flux, H horizons
        y_pred:    np.ndarray,           # [N, H] predicted
        kp:        Optional[np.ndarray], # [N] Kp index
        storm_flags: Optional[np.ndarray],
        pred_std:  Optional[np.ndarray] = None,
        horizon_names: list = ["1h", "6h", "12h"],
        y_pers:    Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Evaluate across all periods and horizons.
        Returns a DataFrame suitable for paper Table.
        """
        results = []
        n_horizons = y_true.shape[1] if y_true.ndim > 1 else 1

        if y_true.ndim == 1:
            y_true = y_true[:, None]
            y_pred = y_pred[:, None]

        for h_idx in range(n_horizons):
            h_name = horizon_names[h_idx] if h_idx < len(horizon_names) else f"h{h_idx}"
            yt = y_true[:, h_idx]
            yp = y_pred[:, h_idx]
            ys = pred_std[:, h_idx] if pred_std is not None else None
            ypr = y_pers[:, h_idx] if y_pers is not None else None

            # All periods
            r = compute_metrics(yt, yp, storm_flags, kp, ys, ypr,
                                 self.flux_thr, "all")
            results.append({"horizon": h_name, **r.__dict__})

            # Quiet periods (Kp < 3)
            if kp is not None:
                quiet_mask = kp < 3
                if quiet_mask.sum() > 10:
                    r = compute_metrics(yt[quiet_mask], yp[quiet_mask],
                                        storm_flags[quiet_mask] if storm_flags is not None else None,
                                        kp[quiet_mask],
                                        ys[quiet_mask] if ys is not None else None,
                                        ypr[quiet_mask] if ypr is not None else None,
                                        self.flux_thr, "quiet (Kp<3)")
                    results.append({"horizon": h_name, **r.__dict__})

            # Storm periods (Kp ≥ 5)
            if kp is not None:
                storm_mask = kp >= self.kp_storm
                if storm_mask.sum() > 10:
                    r = compute_metrics(yt[storm_mask], yp[storm_mask],
                                        storm_flags[storm_mask] if storm_flags is not None else None,
                                        kp[storm_mask],
                                        ys[storm_mask] if ys is not None else None,
                                        ypr[storm_mask] if ypr is not None else None,
                                        self.flux_thr, "storm (Kp≥5)")
                    results.append({"horizon": h_name, **r.__dict__})

            # Intense storm periods (Kp ≥ 7)
            if kp is not None:
                intense_mask = kp >= self.kp_intense
                if intense_mask.sum() > 10:
                    r = compute_metrics(yt[intense_mask], yp[intense_mask],
                                        storm_flags[intense_mask] if storm_flags is not None else None,
                                        kp[intense_mask],
                                        ys[intense_mask] if ys is not None else None,
                                        ypr[intense_mask] if ypr is not None else None,
                                        self.flux_thr, "intense (Kp≥7)")
                    results.append({"horizon": h_name, **r.__dict__})

        df = pd.DataFrame(results)
        return df

    def print_paper_table(self, df: pd.DataFrame):
        """Print metrics in IEEE paper table format."""
        print("\n" + "="*90)
        print("STORM-PhysNet Performance Table (compare with Table X in existing papers)")
        print("="*90)
        for _, row in df.iterrows():
            print(f"Horizon: {row['horizon']:4s} | Period: {row['period']:20s} | "
                  f"PE={row['pe']:.3f} | PE_pers={row['pe_pers']:.3f} | RMSE={row['rmse']:.3f} | "
                  f"R²={row['r2']:.3f} | HSS={row['hss']:.3f} | "
                  f"POD={row['pod']:.3f}")
        print("="*90)
