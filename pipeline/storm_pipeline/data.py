"""
STORM-PhysNet data pipeline (causal, leakage-free).

  1. Missing GOES hours (fill value or flux<=0) are never interpolated or filled from later data.  Model INPUT
     channels are forward-filled (past values; only the leading rows before the first observation take that first value) and an explicit `flux_obs` mask channel marks the real hours.
  2. Targets are the raw observed hourly log10 flux.  A window is used (train/val/test) only if the last lookback
     hour and every target hour (default t+1, t+6, t+12) are real observations.
     The horizons can be changed with the environment variable STORM_HSTEPS (e.g. "1,6,12,24"), set BEFORE this
     module is imported; the default reproduces the AGERS 2026 results exactly.
  3. No despiking; only physical-bound clipping.
  4. OMNI Kp is stored x10 (0..90).  It is divided by 10 and made causal: each hour only sees the last Kp 3-h
     interval that has already finished (OMNI repeats the interval's final value on all 3 hours).
  5. storm_onset_hours is computed on the causal Kp and capped at 720 h.
  6. Splits are labelled per row (train / val / test) and a window is kept only if all its 84 rows (72 lookback +
     12 lead) carry the same label, so no window straddles a boundary (automatic 84 h buffers).
  7. Scalers are fit on the training rows of each block only.
Three blocks A/B/C give three different test periods (see BLOCKS).
"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

SEQ = int(os.environ.get("STORM_SEQ", "72"))   # lookback in hours; 72 everywhere except campaign 7
assert 24 <= SEQ <= 512, f"bad STORM_SEQ {SEQ}"
HSTEPS = tuple(int(h) for h in os.environ.get("STORM_HSTEPS", "1,6,12").split(","))
assert len(HSTEPS) >= 1 and all(h >= 1 for h in HSTEPS) and list(HSTEPS) == sorted(set(HSTEPS)), f"bad STORM_HSTEPS {HSTEPS}"
SPAN = SEQ + max(HSTEPS)          # rows needed per window (84 for the default horizons)
SW_COLS = ["vsw", "bz", "bt", "density", "pdyn", "bz_south_duration", "storm_onset_hours",
           "vsw_roll1h", "vsw_roll6h", "vsw_roll24h", "bz_roll1h", "bz_roll6h", "bz_x_vsw", "kp", "dst",
           "log_flux_std_1h", "flux_obs"]
SCALED = [c for c in SW_COLS if c != "flux_obs"]
KEEP_FOR_FLUX_ONLY = [SW_COLS.index("log_flux_std_1h"), SW_COLS.index("flux_obs")]

VAL = ("2017-03-11 09:00", "2018-09-07 03:00")       # shared validation period
BLOCKS = {
    "A": ("2018-09-07 04:00", "2020-03-04 23:00"),    # declining phase -> solar minimum
    "B": ("2015-01-01 00:00", "2016-12-31 23:00"),    # incl. Mar-2015 and Jun-2015 storms
    "C": ("2012-01-01 00:00", "2013-12-31 23:00"),    # incl. Mar-2012 and Mar-2013 storms
}


def _ffill_leading(s: pd.Series) -> pd.Series:
    s = s.ffill()
    if s.isna().any() and s.notna().any():
        s = s.fillna(s.dropna().iloc[0])      # only the leading rows, before any observation exists
    return s.fillna(0.0)


def causal_kp(kp10: pd.Series) -> pd.Series:
    """OMNI hourly Kp*10 -> Kp (0-9), using only 3-h intervals that have already ended at the END of each hour row."""
    idx = kp10.index
    k = (idx.hour + 1) // 3 - 1                                    # last completed interval index (may be -1)
    ref = idx.floor("D") + pd.to_timedelta(3 * k, unit="h")        # any hour inside that interval
    return pd.Series(kp10.reindex(ref).values / 10.0, index=idx)


def build_features(goes: pd.DataFrame, omni: pd.DataFrame) -> pd.DataFrame:
    """goes: hourly frame with flux_gt2mev (+ flux_std_1h); omni: hourly frame. Returns the causal feature frame."""
    df = goes.join(omni, how="inner").copy()
    raw_flux = df["flux_gt2mev"]
    df["flux_obs"] = raw_flux.notna().astype(np.float32)
    lf = np.log10(raw_flux.clip(lower=1e-6)).clip(-2.0, 6.0)
    df["flux_raw_log"] = lf                                           # TARGET (NaN where not observed)
    df["log_flux"] = _ffill_leading(lf)                               # INPUT history (forward-filled; leading prefix = first observation)
    if "flux_std_1h" in df.columns:
        sd = np.log10(df["flux_std_1h"].clip(lower=1e-6)).clip(-2.0, 6.0)
        df["log_flux_std_1h"] = _ffill_leading(sd)
    else:
        df["log_flux_std_1h"] = 0.0
    df["vsw"] = df["vsw"].clip(200, 1000); df["bz"] = df["bz"].clip(-100, 30)
    df["density"] = df["density"].clip(0.1, 100); df["pdyn"] = df["pdyn"].clip(0.01, 50)
    for c in ["vsw", "bz", "bt", "density", "pdyn", "dst"]:
        df[c] = _ffill_leading(df[c])
    df["dst_raw"] = df["dst"].astype(np.float32)
    df["kp"] = _ffill_leading(causal_kp(df["kp"]))
    for w in (1, 6, 24): df[f"vsw_roll{w}h"] = df["vsw"].rolling(w, min_periods=1).mean()
    for w in (1, 6): df[f"bz_roll{w}h"] = df["bz"].rolling(w, min_periods=1).mean()
    df["bz_x_vsw"] = df["bz"] * df["vsw"] / 1000.0
    dur = np.zeros(len(df)); c = 0
    for i, v in enumerate(df["bz"].values):
        c = c + 1 if v < -3 else 0; dur[i] = c
    df["bz_south_duration"] = dur
    kp = df["kp"].values; onset = np.full(len(df), 720.0); last = None
    for i in range(len(df)):
        if kp[i] >= 5 and (i == 0 or kp[i - 1] < 5): last = i
        onset[i] = min(i - last, 720.0) if last is not None else 720.0
    df["storm_onset_hours"] = onset
    df["storm_flag"] = (df["dst_raw"] <= -50.0).astype(np.float32)
    return df


def label_rows(index: pd.DatetimeIndex, block: str) -> np.ndarray:
    """0 = train, 1 = val, 2 = test for each row."""
    lab = np.zeros(len(index), dtype=np.int8)
    v0, v1 = pd.Timestamp(VAL[0]), pd.Timestamp(VAL[1]); t0, t1 = [pd.Timestamp(x) for x in BLOCKS[block]]
    lab[(index >= v0) & (index <= v1)] = 1
    lab[(index >= t0) & (index <= t1)] = 2
    return lab


def valid_starts(lab: np.ndarray, obs: np.ndarray):
    """Window starts s (0-based) whose SPAN rows share one label and whose last-lookback and target rows are observed."""
    n = len(lab); starts = np.arange(0, n - SPAN + 1)
    lab_s = pd.Series(lab.astype(np.int8))
    mn = lab_s.rolling(SPAN).min().values[SPAN - 1:]; mx = lab_s.rolling(SPAN).max().values[SPAN - 1:]
    same = mn == mx
    ok = same & (obs[starts + SEQ - 1] > 0)
    for h in HSTEPS:
        ok &= obs[starts + SEQ + h - 1] > 0
    return starts[ok], lab[starts[ok]]


class Block:
    """Everything for one block: scaled arrays, window index per split, datasets/loaders."""

    def __init__(self, df: pd.DataFrame, block: str, scaler=None):
        self.block = block; self.df = df
        self.lab = label_rows(df.index, block)
        obs = df["flux_obs"].values
        tr_rows = self.lab == 0
        if scaler is None:
            mu = df.loc[tr_rows, SCALED].mean().values; sd = df.loc[tr_rows, SCALED].std().values
            sd = np.where(sd < 1e-8, 1.0, sd); self.scaler = (mu.astype(np.float64), sd.astype(np.float64))
        else:
            self.scaler = scaler
        X = df[SW_COLS].copy().astype(np.float64)
        X[SCALED] = (X[SCALED].values - self.scaler[0]) / self.scaler[1]
        self.X = torch.from_numpy(X.values.astype(np.float32))
        self.flux_in = torch.from_numpy(df["log_flux"].values.astype(np.float32))
        self.flux_tg = torch.from_numpy(np.nan_to_num(df["flux_raw_log"].values, nan=0.0).astype(np.float32))
        self.dst_s = torch.from_numpy(X["dst"].values.astype(np.float32)); self.kp_s = torch.from_numpy(X["kp"].values.astype(np.float32))
        self.storm = df["storm_flag"].values.astype(np.float32)
        self.starts, self.wlab = valid_starts(self.lab, obs)
        self.obs = obs
        # per-window metadata (lookback storm flag, strictness, raw dst at targets)
        cs = np.concatenate([[0], np.cumsum(self.storm)]); co = np.concatenate([[0], np.cumsum(obs)])
        s = self.starts
        self.win_storm = (cs[s + SEQ] - cs[s]) > 0
        self.win_strict = (co[s + SEQ] - co[s]) == SEQ
        dr = df["dst_raw"].values
        self.dst_target = np.stack([dr[s + SEQ + h - 1] for h in HSTEPS], 1)
        self.t_origin = df.index.values[s + SEQ - 1]

    def split(self, k):  # k in {0,1,2}
        m = self.wlab == k
        return WindowSet(self, self.starts[m], self.win_storm[m], self.win_strict[m], self.dst_target[m], self.t_origin[m])

    def stats(self):
        out = {}
        for k, nm in enumerate(["train", "val", "test"]):
            rows = self.lab == k
            out[nm] = dict(rows=int(rows.sum()), filled_frac=float(1 - self.obs[rows].mean()) if rows.any() else None,
                           windows=int((self.wlab == k).sum()))
        return out


class WindowSet(Dataset):
    def __init__(self, blk, starts, storm_flags, strict, dst_target, t_origin):
        self.b = blk; self.starts = starts; self.window_storm_flags = storm_flags.astype(np.float32)
        self.strict = strict; self.dst_target = dst_target; self.t_origin = t_origin
        self.horizon_steps = list(HSTEPS)

    def __len__(self): return len(self.starts)

    def __getitem__(self, i):
        b = self.b; s = int(self.starts[i]); e = s + SEQ
        tg = [e + h - 1 for h in HSTEPS]
        st6 = float(b.storm[e:e + 6].max())
        yp = b.flux_tg[e - 1]
        return {
            "x_sw": b.X[s:e], "x_flux": b.flux_in[s:e].unsqueeze(-1),
            "y_flux": b.flux_tg[tg], "y_dst": b.dst_s[tg], "y_kp": b.kp_s[tg],
            "y_storm": torch.tensor([st6]), "storm_flag": torch.tensor([float(self.window_storm_flags[i])]),
            "y_persist": yp.repeat(len(HSTEPS)),
        }


def make_loaders(blk: Block, batch_size=64, storm_weight=12.0, seed=0):
    tr, va, te = blk.split(0), blk.split(1), blk.split(2)
    w = torch.from_numpy(np.where(tr.window_storm_flags > 0, storm_weight, 1.0).astype(np.float32))
    g = torch.Generator(); g.manual_seed(seed)
    sampler = WeightedRandomSampler(w, num_samples=len(tr), replacement=True, generator=g)
    kw = dict(num_workers=0, pin_memory=False)
    return (DataLoader(tr, batch_size=batch_size, sampler=sampler, drop_last=True, **kw),
            DataLoader(va, batch_size=256, shuffle=False, **kw), DataLoader(te, batch_size=256, shuffle=False, **kw), tr, va, te)


def load_raw(repo: str):
    sys.path.insert(0, repo)
    from src.data.cdf_reader import read_goes_directory, read_wind_directory
    goes = read_goes_directory(os.path.join(repo, "datasets/goes")); omni = read_wind_directory(os.path.join(repo, "datasets/omni"))
    return goes, omni
