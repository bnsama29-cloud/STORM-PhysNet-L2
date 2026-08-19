"""
Preprocessing pipeline for GOES electron flux and Wind solar wind data.
Handles: spike removal, gap interpolation, feature engineering,
rolling statistics, normalization.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


# Features used for model input
SOLAR_WIND_FEATURES = ["vsw", "bz", "bt", "density", "pdyn"]
DERIVED_FEATURES    = ["bz_south_duration", "storm_onset_hours",
                       "vsw_roll1h", "vsw_roll6h", "vsw_roll24h",
                       "bz_roll1h",  "bz_roll6h",
                       "flux_roll1h", "flux_roll6h", "flux_roll24h",
                       "bz_x_vsw",   "dst", "kp",
                       "log_flux_std_1h"]
TARGET_COL          = "log_flux"
ALL_FEATURES        = SOLAR_WIND_FEATURES + DERIVED_FEATURES


class Preprocessor:
    """
    Full preprocessing pipeline. Fit on training data, transform all splits.

    Steps
    -----
    1. Spike removal (threshold-based in log-flux; Hampel filter on SW)
    2. Gap interpolation (linear up to max_gap hours)
    3. Feature engineering (rolling stats, cross-terms, derived indices)
    4. Train/Val/Test chronological split
    5. StandardScaler fit on training data
    """

    def __init__(
        self,
        spike_sigma: float = 5.0,
        max_gap_hours: int = 3,
        log_flux_min: float = -2.0,
        log_flux_max: float = 6.0,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        year_split: dict = None,
    ):
        self.spike_sigma   = spike_sigma
        self.max_gap_hours = max_gap_hours
        self.log_flux_min  = log_flux_min
        self.log_flux_max  = log_flux_max
        self.train_frac    = train_frac
        self.val_frac      = val_frac
        self.year_split    = year_split
        self.scaler        = StandardScaler()
        self._fitted       = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fit_transform(
        self, raw_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Full pipeline: clean → engineer features → split → scale.

        Returns
        -------
        train_df, val_df, test_df : pd.DataFrame
            Scaled feature DataFrames with TARGET_COL present.
        """
        df = self._clean(raw_df, is_fit=True)
        df = self._engineer_features(df)
        # Final NaN sweep after feature engineering (rolling stats may introduce NaNs)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.bfill().ffill().fillna(0.0)
        df = df.dropna()

        train_df, val_df, test_df = self._split(df)

        # Fit scaler on training features only
        train_df, val_df, test_df = self._scale(train_df, val_df, test_df)
        self._fitted = True

        print(f"[Preprocessor] Train: {len(train_df):,} | "
              f"Val: {len(val_df):,} | Test: {len(test_df):,}")
        print(f"  Storm % — Train: {train_df['storm_flag'].mean():.2%} | "
              f"Val: {val_df['storm_flag'].mean():.2%} | "
              f"Test: {test_df['storm_flag'].mean():.2%}")
        return train_df, val_df, test_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data (e.g., GRASP) using fitted scaler."""
        assert self._fitted, "Call fit_transform first."

        df = self._clean(df.copy())
        df = self._engineer_features(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.bfill().ffill().fillna(0.0)

        # Make sure every expected feature exists
        for col in ALL_FEATURES:
            if col not in df.columns:
                df[col] = 0.0

        feat_cols = list(ALL_FEATURES)
        scaled = self.scaler.transform(df[feat_cols].values)
        df[feat_cols] = scaled

        if "storm_flag" not in df.columns:
            if "dst" in df.columns:
                df["storm_flag"] = (df["dst"] <= -50.0).astype(int)
            else:
                df["storm_flag"] = 0

        return df

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal methods
    # ──────────────────────────────────────────────────────────────────────────

    def _despike(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "log_flux" in df.columns:
            df["log_flux"] = self._hampel(df["log_flux"])
        for col in ["vsw", "bz", "density"]:
            if col in df.columns:
                df[col] = self._hampel(df[col], window=5, sigma=self.spike_sigma)
        return df

    def _clean(self, df: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        df = df.copy()

        # 0. Convert raw flux to log_flux if reading real CDF data
        if "flux_gt2mev" in df.columns and "log_flux" not in df.columns:
            # clip to 1e-6 minimum to avoid log10(0) = -inf
            df["log_flux"] = np.log10(df["flux_gt2mev"].clip(lower=1e-6))
            # replace any inf/-inf that survived with NaN
            df["log_flux"] = df["log_flux"].replace([np.inf, -np.inf], np.nan)

        # 0b. Within-hour flux variability (see read_goes_directory in
        # cdf_reader.py for why this exists): std of the 1-minute samples
        # within each hour. NaN naturally occurs for hours with 0-1 valid
        # 1-minute samples (e.g. during a near-total dropout like the
        # documented GOES-15 yaw-flip artifact) — treated as "no
        # variability signal available" rather than zero, since zero would
        # falsely say "very stable" for what's actually a data gap; the
        # gap-interpolation step below (step 4) fills these consistently
        # with everything else.
        if "flux_std_1h" in df.columns:
            df["log_flux_std_1h"] = np.log10(df["flux_std_1h"].clip(lower=1e-6))
            df["log_flux_std_1h"] = df["log_flux_std_1h"].replace([np.inf, -np.inf], np.nan)
            df.drop(columns=["flux_std_1h"], inplace=True)

        # 1. Physical bounds enforcement
        if "log_flux" in df.columns:
            df["log_flux"] = df["log_flux"].clip(self.log_flux_min,
                                                  self.log_flux_max)
        if "log_flux_std_1h" in df.columns:
            df["log_flux_std_1h"] = df["log_flux_std_1h"].clip(self.log_flux_min,
                                                                  self.log_flux_max)
        if "vsw" in df.columns:
            df["vsw"] = df["vsw"].clip(200, 1000)
        if "bz" in df.columns:
            df["bz"] = df["bz"].clip(-100, 30)
        if "density" in df.columns:
            df["density"] = df["density"].clip(0.1, 100)
        if "pdyn" in df.columns:
            df["pdyn"] = df["pdyn"].clip(0.01, 50)

        # 2 & 3. Spike removal (Hampel filter)
        if is_fit:
            train, val, test = self._split(df)
            train = self._despike(train)
            val = self._despike(val)
            test = self._despike(test)
            df = pd.concat([train, val, test]).sort_index()
        else:
            df = self._despike(df)

        # 4. Gap interpolation (linear, max gap)
        df = df.interpolate(method="linear", limit=self.max_gap_hours)
        # Aggressively handle any remaining NaNs after interpolation
        # (long gaps > max_gap_hours get backfill → forwardfill → zero)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.bfill().ffill().fillna(0.0)
        return df

    def _hampel(self, series: pd.Series, window: int = 10,
                sigma: float = None) -> pd.Series:
        """Hampel filter for outlier detection and removal."""
        if sigma is None:
            sigma = self.spike_sigma
        s = series.copy()
        roll = s.rolling(window=window * 2 + 1, center=True)
        median = roll.median()
        # Fully vectorized MAD calculation to prevent memory fragmentation and OOM
        mad = (s - median).abs().rolling(window=window * 2 + 1, center=True).median()
        threshold = sigma * 1.4826 * mad  # MAD to std scale factor
        outliers = (s - median).abs() > threshold
        s[outliers] = median[outliers]
        return s

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Rolling statistics
        for col, windows in [("vsw", [1, 6, 24]), ("bz", [1, 6]),
                               ("log_flux", [1, 6, 24])]:
            if col in df.columns:
                for w in windows:
                    name = f"{col.replace('log_flux','flux')}_roll{w}h"
                    df[name] = df[col].rolling(w, min_periods=1).mean()

        # Physics cross-term: Bz × Vsw (proxy for dawn-to-dusk E-field)
        if "bz" in df.columns and "vsw" in df.columns:
            df["bz_x_vsw"] = df["bz"] * df["vsw"] / 1000  # normalised

        # Bz southward duration (hours of continuous negative Bz)
        if "bz" in df.columns and "bz_south_duration" not in df.columns:
            dur = np.zeros(len(df))
            cnt = 0
            for i, v in enumerate(df["bz"].values):
                cnt = cnt + 1 if v < -3 else 0
                dur[i] = cnt
            df["bz_south_duration"] = dur

        # Storm onset hours (hours since last Kp > 5 onset)
        if "kp" in df.columns and "storm_onset_hours" not in df.columns:
            onset_hrs = np.full(len(df), 9999.0)
            last = -9999
            kp_vals = df["kp"].values
            for i, kp in enumerate(kp_vals):
                if kp >= 5 and (i == 0 or kp_vals[i - 1] < 5):
                    last = i
                onset_hrs[i] = i - last if last >= 0 else 9999
            df["storm_onset_hours"] = onset_hrs

        # Ensure dst and kp exist
        if "dst" not in df.columns:
            df["dst"] = 0.0
        if "kp" not in df.columns:
            df["kp"] = 3.0

        if "storm_flag" not in df.columns:
            df["storm_flag"] = (df["dst"] <= -50.0).astype(int)

        return df

    def _split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.year_split is not None:
            train_years = self.year_split["train_years"]
            val_year = self.year_split["val_year"]
            test_year = self.year_split["test_year"]
            train = df[df.index.year.isin(train_years)].copy()
            val = df[df.index.year == val_year].copy()
            test = df[df.index.year == test_year].copy()
            return train, val, test

        n      = len(df)
        n_tr   = int(n * self.train_frac)
        n_val  = int(n * self.val_frac)
        train  = df.iloc[:n_tr].copy()
        val    = df.iloc[n_tr:n_tr + n_val].copy()
        test   = df.iloc[n_tr + n_val:].copy()
        return train, val, test

    def _scale(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        feat_cols = [c for c in ALL_FEATURES if c in train.columns]

        # Fit on training features
        self.scaler.fit(train[feat_cols].values)

        for split in [train, val, test]:
            split[feat_cols] = self.scaler.transform(split[feat_cols].values)

        return train, val, test

    def inverse_scale_flux(self, scaled_flux: np.ndarray) -> np.ndarray:
        """Inverse transform log_flux predictions back to original scale."""
        # log_flux is the last feature in ALL_FEATURES (or use scaler directly)
        dummy = np.zeros((len(scaled_flux), len(self.scaler.mean_)))
        flux_idx = ALL_FEATURES.index("log_flux") if "log_flux" in ALL_FEATURES else 0
        dummy[:, flux_idx] = scaled_flux
        return self.scaler.inverse_transform(dummy)[:, flux_idx]
