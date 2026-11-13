#!/usr/bin/env python
"""
Cross-satellite transfer for the journal version: GOES-15 block-A models -> GOES-16, GOES-18 and GSAT-19 GRASP.

  python transfer_journal.py --repo <clone> --ckpt_dir j24_A/ckpt_A --meta j24_A/meta_A.npz --out tj --horizons 1,6,12,24

Pretrained models: the block-A checkpoints of train_block.py (GOES-15 data up to Mar 2017, validation to Sep 2018),
so every GOES-16/18 window used here (after 7 Sep 2018) lies after the models' training and validation data.

Targets
  goes16, goes18  NOAA MPS-HI >2 MeV integral electrons (telescope 4; goesr_download.py), hourly mean + within-hour
                  std, with the same OMNI drivers and causal features as GOES-15. Standardised with the GOES-15
                  block-A scaler (as a model trained on GOES-15 would see them).
                  GOES-16 is at 75.2 W (GOES-East); GOES-15 was at 135 W and GOES-18 is at 137 W (GOES-West).
  grasp           GSAT-19 GRASP (48 E), exactly as grasp_transfer.py: no solar-wind data, every driver channel at the
                  GOES training mean; same three chronological folds.
Variants per target/fold/family/seed: zeroshot, heads (encoder frozen), full, scratch -- see storm_pipeline/transfer.py.
GOES-16/18 training windows are taken every --train_stride hours (validation and test use every hour).

Outputs in --out:  transfer_meta.npz (targets, persistence, flags, times per target-fold),
                   transfer_preds_s<seed>.npz (key <target>_<fold>_<family>_<variant>), transfer_runlog.jsonl
"""
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--ckpt_dir", required=True); ap.add_argument("--meta", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--seeds", default="42-51")
ap.add_argument("--targets", default="goes16,goes18,grasp")
ap.add_argument("--families", default="storm_bz,transformer_matched,tf_ffn_res")
ap.add_argument("--horizons", default="1,6,12,24")
ap.add_argument("--variants", default="zeroshot,heads,full,scratch", help="subset of zeroshot,heads,full,scratch")
ap.add_argument("--tag", default="", help="suffix of the output files, e.g. _zs for a zero-shot-only run")
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=8)
ap.add_argument("--train_stride", type=int, default=1, help="GOES-16/18 only: use every n-th training window")
ap.add_argument("--max_windows", type=int, default=None, help="smoke test only")
ap.add_argument("--stop_after_hours", type=float, default=None, help="start no new seed after this many hours")
args = ap.parse_args()
os.environ["STORM_HSTEPS"] = args.horizons

import numpy as np, pandas as pd, torch
from types import SimpleNamespace
from storm_pipeline.data import SW_COLS, SCALED, SEQ, HSTEPS, valid_starts, WindowSet, build_features
from storm_pipeline import transfer as T
from classical_baselines import predict_windowset
CLASSICAL = {"ridge_res", "gbm_res"}

REPO = Path(args.repo).resolve(); sys.path.insert(0, str(REPO))
from src.data.cdf_reader import read_grasp_directory, read_wind_directory

OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a, b = (args.seeds.split("-") + [None])[:2]
SEEDS = list(range(int(a), int(b) + 1)) if b else [int(s) for s in args.seeds.split(",")]
TARGETS, FAMS, VARIANTS = args.targets.split(","), args.families.split(","), args.variants.split(",")
assert set(VARIANTS) <= {"zeroshot", "heads", "full", "scratch"}, f"unknown variants {VARIANTS}"
TMPTAG = str(os.getpid()); T_START = time.time()

FOLDS = {   # train | val | test, chronological; a window never straddles two labels
    "goes16": {   # data 2017-01 .. 2025-04 (GOES-East until April 2025)
        "F1": dict(train=("2018-09-07 04:00", "2021-12-31 23:00"), val=("2022-01-01", "2022-06-30 23:00"), test=("2022-07-01", "2023-06-30 23:00")),
        "F2": dict(train=("2018-09-07 04:00", "2023-06-30 23:00"), val=("2023-07-01", "2023-12-31 23:00"), test=("2024-01-01", "2025-04-07 23:00")),
    },
    "goes18": {   # data 2022-04 .. ; OMNI drivers complete to 2026-10-14
        "F1": dict(train=("2022-04-25", "2024-12-31 23:00"), val=("2025-01-01", "2025-06-30 23:00"), test=("2025-07-01", "2026-10-14 23:00")),
    },
    "grasp": {
        "G1": dict(train=("2017-07-01", "2017-12-31 23:00"), val=("2018-01-01", "2018-01-31 23:00"), test=("2018-02-01", "2018-03-31 23:00")),
        "G2": dict(train=("2017-07-01", "2018-02-28 23:00"), val=("2018-03-01", "2018-03-31 23:00"), test=("2018-04-01", "2018-05-31 23:00")),
        "G3": dict(train=("2017-07-01", "2018-04-30 23:00"), val=("2018-05-01", "2018-05-31 23:00"), test=("2018-06-01", "2018-09-01 00:00")),
    },
}


def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)


meta = np.load(args.meta, allow_pickle=True); MU, SD = meta["scaler_mu"], meta["scaler_sd"]
assert "horizons" in meta.files and tuple(int(h) for h in meta["horizons"]) == HSTEPS, \
    f"checkpoints were trained for horizons {meta['horizons'] if 'horizons' in meta.files else (1, 6, 12)}, not {HSTEPS}"
OMNI = read_wind_directory(str(REPO / "datasets/omni"))


def ffill(s): s = s.ffill(); return s.fillna(s.dropna().iloc[0] if s.notna().any() else 0.0).fillna(0.0)


def series_goesr(sat):
    """Hourly frame for GOES-16/18 with the same causal features as GOES-15, standardised with the GOES-15 scaler."""
    h = pd.read_csv(REPO / f"datasets/goesr/{sat}_e2mev_hourly.csv", index_col="time", parse_dates=["time"])
    full = pd.date_range(h.index.min().floor("h"), h.index.max().floor("h"), freq="1h")
    goes = h[["flux_gt2mev", "flux_std_1h"]].reindex(full)                     # missing hours stay NaN (flux_obs = 0)
    df = build_features(goes, OMNI)
    X = df[SW_COLS].copy().astype(np.float64)
    X[SCALED] = (X[SCALED].values - MU) / SD
    Xs = X.values.astype(np.float32)
    return df.index, SimpleNamespace(
        X=torch.from_numpy(Xs), flux_in=torch.from_numpy(df["log_flux"].values.astype(np.float32)),
        flux_tg=torch.from_numpy(np.nan_to_num(df["flux_raw_log"].values, nan=0.0).astype(np.float32)),
        dst_s=torch.from_numpy(Xs[:, SW_COLS.index("dst")]), kp_s=torch.from_numpy(Xs[:, SW_COLS.index("kp")]),
        storm=df["storm_flag"].values.astype(np.float32)), df["flux_obs"].values.astype(np.float32), df["dst_raw"].values.astype(np.float32)


def series_grasp():
    """GRASP exactly as in grasp_transfer.py (no solar-wind data: drivers at the GOES training mean)."""
    raw = read_grasp_directory(str(REPO / "datasets/grasp"))["flux_gt2mev"].dropna()
    g = raw.resample("1h").agg(["mean", "std", "count"])
    g.loc[g["count"] < 6, ["mean", "std"]] = np.nan
    df = g.join(OMNI[["dst"]], how="left")
    obs = df["mean"].notna().values.astype(np.float32)
    lf = np.log10(df["mean"].clip(lower=1e-6)).clip(-2, 6); lsd = np.log10(df["std"].clip(lower=1e-6)).clip(-2, 6)
    i_std = SCALED.index("log_flux_std_1h")
    X = np.zeros((len(df), len(SW_COLS)), np.float32)
    X[:, SW_COLS.index("log_flux_std_1h")] = ((ffill(lsd).values - MU[i_std]) / SD[i_std]).astype(np.float32)
    X[:, SW_COLS.index("flux_obs")] = obs
    dst = ffill(df["dst"]).values.astype(np.float32)
    return df.index, SimpleNamespace(X=torch.from_numpy(X), flux_in=torch.from_numpy(ffill(lf).values.astype(np.float32)),
                                     flux_tg=torch.from_numpy(np.nan_to_num(lf.values, nan=0.0).astype(np.float32)),
                                     dst_s=torch.zeros(len(df)), kp_s=torch.zeros(len(df)), storm=(dst <= -50).astype(np.float32)), obs, dst


def fold_sets(idx, S, obs, dst, f, stride):
    lab = np.full(len(idx), -1, np.int8)
    for k, nm in enumerate(["train", "val", "test"]):
        lab[(idx >= pd.Timestamp(f[nm][0])) & (idx <= pd.Timestamp(f[nm][1]))] = k
    starts, wl = valid_starts(lab, obs)
    cs = np.concatenate([[0], np.cumsum(S.storm)]); co = np.concatenate([[0], np.cumsum(obs)])
    sets = {}
    for k, nm in enumerate(["train", "val", "test"]):
        s = starts[wl == k]
        if nm == "train" and stride > 1: s = s[::stride]
        if args.max_windows: s = s[: args.max_windows]
        st = (cs[s + SEQ] - cs[s]) > 0; strict = (co[s + SEQ] - co[s]) == SEQ
        dt = np.stack([dst[s + SEQ + h - 1] for h in HSTEPS], 1)
        sets[nm] = WindowSet(S, s, st, strict, dt, idx.values[s + SEQ - 1])
    return sets


DATA, tmeta = {}, {}
for tg in TARGETS:
    idx, S, obs, dst = series_grasp() if tg == "grasp" else series_goesr(tg)
    log(f"{tg}: {len(idx)} hourly rows, {int(obs.sum())} observed ({idx.min()} .. {idx.max()})")
    for fk, f in FOLDS[tg].items():
        sets = fold_sets(idx, S, obs, dst, f, 1 if tg == "grasp" else args.train_stride)
        n = {k: len(v) for k, v in sets.items()}
        if min(n.values()) < 50:
            log(f"  {tg} {fk}: skipped, too few windows {n}"); continue
        DATA[(tg, fk)] = sets
        t = sets["test"]; s = t.starts; key = f"{tg}_{fk}"
        tmeta[f"{key}_y"] = np.stack([S.flux_tg[s + SEQ + h - 1].numpy() for h in HSTEPS], 1)
        tmeta[f"{key}_persist"] = S.flux_tg[s + SEQ - 1].numpy(); tmeta[f"{key}_strict"] = t.strict
        tmeta[f"{key}_storm_in"] = t.window_storm_flags > 0; tmeta[f"{key}_dst"] = t.dst_target
        tmeta[f"{key}_t"] = t.t_origin.astype("datetime64[ns]").astype(np.int64)
        log(f"  {tg} {fk}: windows train/val/test = {n['train']}/{n['val']}/{n['test']}; test storm-hour windows "
            f"{int((t.window_storm_flags > 0).sum())}; test Dst<=-50 at 24 h {int((t.dst_target[:, -1] <= -50).sum())}")
tmeta["horizons"] = np.array(HSTEPS)
np.savez(OUT / f"transfer_meta.{TMPTAG}.tmp.npz", **tmeta); os.replace(OUT / f"transfer_meta.{TMPTAG}.tmp.npz", OUT / f"transfer_meta{args.tag}.npz")

M = T.Models(REPO, args.ckpt_dir, OUT / f"scratch_{TMPTAG}", DEV)
stopped = False
for seed in SEEDS:
    pf = OUT / f"transfer_preds{args.tag}_s{seed}.npz"
    if pf.exists(): continue
    if args.stop_after_hours is not None and time.time() - T_START > args.stop_after_hours * 3600:
        stopped = True; break
    t0 = time.time(); res, info = {}, []
    for (tg, fk), sets in DATA.items():
        for fam in FAMS:
            key = f"{tg}_{fk}_{fam}"
            if fam in CLASSICAL:      # ridge / gradient-boosted trees: zero-shot only, from the saved block-A models
                if "zeroshot" in VARIANTS:
                    res[f"{key}_zeroshot"] = predict_windowset(Path(args.ckpt_dir) / f"model_{fam}_s{seed}.pkl", sets["test"])
                continue
            if "zeroshot" in VARIANTS: res[f"{key}_zeroshot"] = T.predict(M.pretrained(fam, seed), sets["test"], DEV)
            for var in [v for v in ("heads", "full") if v in VARIANTS]:
                m = M.pretrained(fam, seed)
                if var == "heads": T.freeze_heads_only(m, fam)
                params = [p for p in m.parameters() if p.requires_grad]
                m, ne, bv = T.train(m, sets, 1e-4, seed, params, DEV, args.epochs, args.patience)
                res[f"{key}_{var}"] = T.predict(m, sets["test"], DEV); info.append((tg, fk, fam, var, ne, round(bv, 5)))
            if "scratch" in VARIANTS:
                torch.manual_seed(seed); m = M.build(fam)
                m, ne, bv = T.train(m, sets, 3e-4, seed, list(m.parameters()), DEV, args.epochs, args.patience)
                res[f"{key}_scratch"] = T.predict(m, sets["test"], DEV); info.append((tg, fk, fam, "scratch", ne, round(bv, 5)))
            if DEV.type == "cuda": torch.cuda.empty_cache()
    np.savez(OUT / f"{pf.stem}.{TMPTAG}.tmp.npz", **res); os.replace(OUT / f"{pf.stem}.{TMPTAG}.tmp.npz", pf)
    with open(OUT / f"transfer_runlog{args.tag}.jsonl", "a") as fh: fh.write(json.dumps(dict(seed=seed, seconds=round(time.time() - t0, 1), runs=info)) + "\n")
    log(f"seed {seed} done in {time.time() - t0:.0f}s")

import shutil; shutil.rmtree(OUT / f"scratch_{TMPTAG}", ignore_errors=True)
log(f"TIME BUDGET REACHED after {args.stop_after_hours} h: run again to continue" if stopped else "TRANSFER COMPLETE")
