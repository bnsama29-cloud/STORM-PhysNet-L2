"""
PyTorch Dataset and DataLoader for STORM-PhysNet.
Key feature: StormBiasedSampler oversamples storm periods during training
to fix the fundamental class imbalance (storms = ~5-8% of data).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional

from src.data.preprocessor import ALL_FEATURES, TARGET_COL

# Forecast horizons in hours
# Must match config.yaml forecast_horizons.
HORIZONS = [1.0, 6.0, 12.0]  # integer hourly leads; must match config.yaml
# Short horizon is the next hourly GOES sample (1 h lead).


class FluxDataset(Dataset):
    """
    Sliding-window dataset for GEO electron flux forecasting.

    Each sample contains:
        x_sw    : solar wind features  [seq_len, n_sw_features]
        x_flux  : flux history         [seq_len, 1]
        y       : target flux at 3 horizons [3]
        y_dst   : target Dst at 3 horizons [3]
        y_kp    : target Kp at 3 horizons [3]
        y_storm : storm onset label (binary) [1]
        storm_flag : whether current window contains storm [1]
    """

    SW_FEATURES = ["vsw", "bz", "bt", "density", "pdyn",
                   "bz_south_duration", "storm_onset_hours",
                   "vsw_roll1h", "vsw_roll6h", "vsw_roll24h",
                   "bz_roll1h", "bz_roll6h", "bz_x_vsw", "kp", "dst",
                   "log_flux_std_1h"]
    # Added "log_flux_std_1h": within-hour standard deviation of the
    # 1-minute-cadence GOES flux samples (see cdf_reader.py's
    # read_goes_directory and preprocessor.py's _clean for where this is
    # computed). This is real short-timescale variability signal that a
    # plain hourly-mean flux value discards; a quiet, stable hour and a
    # turbulent, rapidly-changing hour can have the same mean but very
    # different std. Also carries a data-quality signal: near-total
    # dropout periods (e.g. the documented GOES-15 yaw-flip artifact around
    # equinoxes) show up as anomalous values here after gap-interpolation,
    # which the model can in principle learn to partially discount.

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 72,
        horizons: list[int] = HORIZONS,
        stride: int = 1,
    ):
        self.seq_len  = seq_len
        self.horizons = horizons
        self.horizon_steps = [max(1, int(round(float(h)))) for h in self.horizons]
        self.max_h    = int(max(self.horizon_steps))
        self.stride   = stride

        # Select available SW features
        self.sw_cols  = [c for c in self.SW_FEATURES if c in df.columns]
        self.flux_col = TARGET_COL

        # Convert to numpy for fast indexing
        self.sw_data    = df[self.sw_cols].values.astype(np.float32)
        self.flux_data  = df[self.flux_col].values.astype(np.float32)
        self.dst_data   = df["dst"].values.astype(np.float32) if "dst" in df.columns else np.zeros(len(df), np.float32)
        self.kp_data    = df["kp"].values.astype(np.float32) if "kp" in df.columns else np.zeros(len(df), np.float32)
        self.storm_data = df["storm_flag"].values.astype(np.float32) if "storm_flag" in df.columns else np.zeros(len(df), np.float32)

        # Valid start indices
        self.indices = list(range(
            0,
            len(df) - seq_len - self.max_h,
            stride
        ))

        # Pre-compute storm flags per window (for sampler)
        self.window_storm_flags = np.array([
            self.storm_data[i:i + seq_len].max()
            for i in self.indices
        ], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        start = self.indices[idx]
        end   = start + self.seq_len

        x_sw   = torch.from_numpy(self.sw_data[start:end])          # [T, n_sw]
        x_flux = torch.from_numpy(
            self.flux_data[start:end].reshape(-1, 1))                # [T, 1]

        # Multi-horizon targets
        y_flux  = torch.tensor(
            [self.flux_data[end + h - 1] for h in self.horizon_steps],
            dtype=torch.float32)                                      # [3]
        y_dst   = torch.tensor(
            [self.dst_data[end + h - 1] for h in self.horizon_steps],
            dtype=torch.float32)
        y_kp    = torch.tensor(
            [self.kp_data[end + h - 1] for h in self.horizon_steps],
            dtype=torch.float32)

        # Storm onset in next 6h (binary label for auxiliary task)
        future_storm = self.storm_data[end:end + 6].max()
        y_storm = torch.tensor([future_storm], dtype=torch.float32)

        # Current storm context
        storm_flag = torch.tensor(
            [self.window_storm_flags[idx]], dtype=torch.float32)

        # Persistence baseline (current flux for residual prediction)
        y_persist = torch.tensor(
            [self.flux_data[end - 1]] * len(self.horizons),
            dtype=torch.float32)

        return {
            "x_sw":       x_sw,
            "x_flux":     x_flux,
            "y_flux":     y_flux,
            "y_dst":      y_dst,
            "y_kp":       y_kp,
            "y_storm":    y_storm,
            "storm_flag": storm_flag,
            "y_persist":  y_persist,
        }

    @property
    def n_sw_features(self) -> int:
        return len(self.sw_cols)


def make_storm_biased_sampler(
    dataset: FluxDataset,
    storm_weight: float = 12.0,
) -> WeightedRandomSampler:
    """
    Returns a WeightedRandomSampler that oversamples storm windows.

    storm_weight : float
        How many times more likely to sample a storm window vs quiet.
        Default 12x → storm fraction goes from ~5% to >50% of batches.
    """
    flags   = dataset.window_storm_flags
    weights = np.where(flags > 0, storm_weight, 1.0)
    weights = torch.from_numpy(weights.astype(np.float32))
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
    )
    return sampler


def make_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seq_len: int = 72,
    batch_size: int = 64,
    storm_weight: float = 12.0,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Training loader uses StormBiasedSampler.
    Val/Test loaders are sequential (no shuffling).
    """
    train_ds = FluxDataset(train_df, seq_len=seq_len)
    val_ds   = FluxDataset(val_df,   seq_len=seq_len)
    test_ds  = FluxDataset(test_df,  seq_len=seq_len)

    storm_sampler = make_storm_biased_sampler(train_ds, storm_weight)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=storm_sampler,
        num_workers=num_workers, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    storm_frac_train = train_ds.window_storm_flags.mean()
    print(f"[DataLoader] Train: {len(train_ds):,} windows | "
          f"Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    print(f"  Storm windows in train: {storm_frac_train:.2%} "
          f"(effective ~{storm_frac_train*storm_weight/(1+storm_frac_train*(storm_weight-1)):.0%} with sampler)")
    print(f"  SW features: {train_ds.n_sw_features}")

    return train_loader, val_loader, test_loader
