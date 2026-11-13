#!/usr/bin/env python
"""
GOES -> GRASP (GSAT-19, 48E) transfer experiment, ONE script, same recipe for STORM-Bz and the matched Transformer.

Needs the block-A checkpoints from train_block.py for every family in --families (default storm_bz and
transformer_matched, seeds 42..51). Optional families: tf_ffn_res (residual-head Transformer, all four regimes) and
storm_no_delay / storm_no_gate (zero-shot, heads-only and full fine-tuning).

  python grasp_transfer.py --repo /kaggle/working/STORM-PhysNet-L2 --ckpt_dir out_A/ckpt_A --meta out_A/meta_A.npz --out out_G

GRASP input: hourly mean log10 flux (counts cm-2 s-1 sr-1) + log within-hour std + observed-hour mask; every solar-wind /
geomagnetic channel is zero in standardised space (= the GOES training mean).  GRASP has no solar-wind instrument.
Three chronological folds (train | val | test, windows never straddle a boundary):
   G1 train 2017-07-01..2017-12-31  val 2018-01  test 2018-02-01..2018-03-31
   G2 train 2017-07-01..2018-02-28  val 2018-03  test 2018-04-01..2018-05-31
   G3 train 2017-07-01..2018-04-30  val 2018-05  test 2018-06-01..2018-09-01   (contains the 25-26 Aug 2018 storm)
Variants per fold/model/seed:
   zeroshot   pretrained GOES model, no GRASP training
   heads      encoder frozen, heads trained on GRASP (the paper's recipe)
   full       all weights fine-tuned
   scratch    random init trained on GRASP only
Loss for all GRASP training: horizon-weighted MSE (1.0, 0.7, 0.5), Adam, uniform sampling, best-val checkpoint.
"""
import argparse, copy, json, os, sys, time
from pathlib import Path
os.environ.setdefault("TQDM_DISABLE", "1")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--ckpt_dir", required=True); ap.add_argument("--meta", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--seeds", default="42-51")
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=8)
ap.add_argument("--max_windows", type=int, default=None, help="smoke test only")
ap.add_argument("--families", default="storm_bz,transformer_matched")
args = ap.parse_args()

import numpy as np, pandas as pd, torch, torch.nn as nn, yaml
from types import SimpleNamespace
from torch.utils.data import DataLoader
from storm_pipeline.data import SW_COLS, SCALED, SEQ, HSTEPS, valid_starts, WindowSet
from storm_pipeline.models_extra import FFNResTransformer

REPO = Path(args.repo).resolve(); sys.path.insert(0, str(REPO))
from src.data.cdf_reader import read_grasp_directory, read_wind_directory
from src.training.trainer import Trainer

OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a, b = (args.seeds.split("-") + [None])[:2]
SEEDS = list(range(int(a), int(b) + 1)) if b else [int(s) for s in args.seeds.split(",")]
HW = torch.tensor([1.0, 0.7, 0.5], device=DEV)
FOLDS = {
    "G1": dict(train=("2017-07-01", "2017-12-31 23:00"), val=("2018-01-01", "2018-01-31 23:00"), test=("2018-02-01", "2018-03-31 23:00")),
    "G2": dict(train=("2017-07-01", "2018-02-28 23:00"), val=("2018-03-01", "2018-03-31 23:00"), test=("2018-04-01", "2018-05-31 23:00")),
    "G3": dict(train=("2017-07-01", "2018-04-30 23:00"), val=("2018-05-01", "2018-05-31 23:00"), test=("2018-06-01", "2018-09-01 00:00")),
}
def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)

# ---------------------------------------------------------------------------------------------- GRASP hourly table
raw = read_grasp_directory(str(REPO / "datasets/grasp"))["flux_gt2mev"].dropna()
g = raw.resample("1h").agg(["mean", "std", "count"])
g.loc[g["count"] < 6, ["mean", "std"]] = np.nan                     # an hour needs >= 6 of 12 five-minute samples
omni = read_wind_directory(str(REPO / "datasets/omni"))
df = g.join(omni[["dst"]], how="left")
obs = df["mean"].notna().values.astype(np.float32)
lf = np.log10(df["mean"].clip(lower=1e-6)).clip(-2, 6)
lsd = np.log10(df["std"].clip(lower=1e-6)).clip(-2, 6)
def ffill(s): s = s.ffill(); return s.fillna(s.dropna().iloc[0] if s.notna().any() else 0.0).fillna(0.0)
meta = np.load(args.meta, allow_pickle=True); mu, sd = meta["scaler_mu"], meta["scaler_sd"]
i_std = SCALED.index("log_flux_std_1h")
X = np.zeros((len(df), len(SW_COLS)), np.float32)                     # standardised space: 0 == GOES-train mean
X[:, SW_COLS.index("log_flux_std_1h")] = ((ffill(lsd).values - mu[i_std]) / sd[i_std]).astype(np.float32)
X[:, SW_COLS.index("flux_obs")] = obs
dst = ffill(df["dst"]).values.astype(np.float32)
GR = SimpleNamespace(X=torch.from_numpy(X), flux_in=torch.from_numpy(ffill(lf).values.astype(np.float32)),
                     flux_tg=torch.from_numpy(np.nan_to_num(lf.values, nan=0.0).astype(np.float32)),
                     dst_s=torch.zeros(len(df)), kp_s=torch.zeros(len(df)), storm=(dst <= -50).astype(np.float32))
idx = df.index
log(f"GRASP hourly rows={len(df)} observed={int(obs.sum())} ({idx.min()} .. {idx.max()})")

def fold_data(fk):
    f = FOLDS[fk]; lab = np.full(len(df), -1, np.int8)
    for k, nm in enumerate(["train", "val", "test"]):
        lab[(idx >= pd.Timestamp(f[nm][0])) & (idx <= pd.Timestamp(f[nm][1]))] = k
    starts, wl = valid_starts(lab, obs)
    cs = np.concatenate([[0], np.cumsum(GR.storm)]); co = np.concatenate([[0], np.cumsum(obs)])
    sets = {}
    for k, nm in enumerate(["train", "val", "test"]):
        s = starts[wl == k]
        if args.max_windows: s = s[: args.max_windows]
        st = (cs[s + SEQ] - cs[s]) > 0; strict = (co[s + SEQ] - co[s]) == SEQ
        dt = np.stack([dst[s + SEQ + h - 1] for h in HSTEPS], 1)
        sets[nm] = WindowSet(GR, s, st, strict, dt, idx.values[s + SEQ - 1])
    return sets

FD = {fk: fold_data(fk) for fk in FOLDS}
fold_meta = {}
for fk, sets in FD.items():
    t = sets["test"]; s = t.starts
    y = np.stack([GR.flux_tg[s + SEQ + h - 1].numpy() for h in HSTEPS], 1)
    fold_meta[f"{fk}_y"] = y; fold_meta[f"{fk}_persist"] = GR.flux_tg[s + SEQ - 1].numpy()
    fold_meta[f"{fk}_strict"] = t.strict; fold_meta[f"{fk}_storm_in"] = t.window_storm_flags > 0
    fold_meta[f"{fk}_dst"] = t.dst_target; fold_meta[f"{fk}_t"] = t.t_origin.astype("datetime64[ns]").astype(np.int64)
    log(f"{fk}: windows train/val/test = {len(sets['train'])}/{len(sets['val'])}/{len(sets['test'])}; test windows w/ storm hour in lookback={int((t.window_storm_flags>0).sum())}")
TMPTAG = str(os.getpid())                                            # two processes (one per GPU) may share OUT
np.savez(OUT / f"grasp_meta.{TMPTAG}.tmp.npz", **fold_meta); os.replace(OUT / f"grasp_meta.{TMPTAG}.tmp.npz", OUT / "grasp_meta.npz")
FAMS = args.families.split(",")
ABL = {"storm_bz": "none", "storm_no_delay": "no_delay", "storm_no_gate": "no_bz_gate"}
VARIANTS = {f: (["heads", "full"] if f in ("storm_no_delay", "storm_no_gate") else ["heads", "full", "scratch"]) for f in FAMS}

# ---------------------------------------------------------------------------------------------- models
cfg = yaml.safe_load(open(REPO / "configs/config.yaml"))
def build(fam):
    if fam == "tf_ffn_res": return FFNResTransformer(len(SW_COLS)).to(DEV)
    c = copy.deepcopy(cfg); c["training"]["checkpoint_dir"] = str(OUT / f"tmp_{TMPTAG}"); c["training"]["log_dir"] = str(OUT / f"tb_tmp_{TMPTAG}")
    c["model_type"] = "storm_physnet" if fam in ABL else "transformer"; c["ablation"] = ABL.get(fam, "none")
    c["match_storm_capacity"] = fam == "transformer_matched"
    tr = Trainer(c); m = tr.build_model(len(SW_COLS)); tr.writer.close(); return m

def load_pre(fam, seed):
    m = build(fam); sdict = torch.load(Path(args.ckpt_dir) / f"{fam}_s{seed}_best.pt", map_location=DEV, weights_only=True)
    m.load_state_dict(sdict); return m

def freeze_heads_only(m, fam):
    if fam in ABL:
        m.freeze_encoder()
        if hasattr(m, "pos_emb") and isinstance(m.pos_emb, torch.nn.Parameter): m.pos_emb.requires_grad = False   # freeze_encoder leaves it trainable
    else:
        for n, p in m.named_parameters(): p.requires_grad = n.startswith("fc.")

def run_fwd(m, bt):
    return m(x_sw=bt["x_sw"], x_flux=bt["x_flux"], y_persist=bt["y_persist"])["flux_pred"]

def wmse(p, y): return (((p - y) ** 2).mean(0) * HW).sum()

@torch.no_grad()
def predict(m, ds):
    m.eval(); P = []
    for bt in DataLoader(ds, batch_size=256):
        bt = {k: v.to(DEV) for k, v in bt.items()}; P.append(run_fwd(m, bt).cpu().numpy())
    return np.concatenate(P).astype(np.float32)

def train(m, sets, lr, seed, params):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator(); g.manual_seed(seed)
    tl = DataLoader(sets["train"], batch_size=64, shuffle=True, drop_last=True, generator=g)
    best, bad, best_state = 1e9, 0, None
    for ep in range(args.epochs):
        m.train()
        for bt in tl:
            bt = {k: v.to(DEV) for k, v in bt.items()}
            opt.zero_grad(); loss = wmse(run_fwd(m, bt), bt["y_flux"]); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        m.eval(); v = 0.0; n = 0
        with torch.no_grad():
            for bt in DataLoader(sets["val"], batch_size=256):
                bt = {k: v_.to(DEV) for k, v_ in bt.items()}; v += wmse(run_fwd(m, bt), bt["y_flux"]).item() * len(bt["x_sw"]); n += len(bt["x_sw"])
        v /= max(n, 1)
        if v < best - 1e-6: best, bad, best_state = v, 0, copy.deepcopy(m.state_dict())
        else:
            bad += 1
            if bad >= args.patience: break
    if best_state is not None: m.load_state_dict(best_state)          # (None only if every val loss was NaN)
    return m, ep + 1, best

# ---------------------------------------------------------------------------------------------- main loop
for seed in SEEDS:
    pf = OUT / f"grasp_preds_s{seed}.npz"
    if pf.exists(): continue
    t0 = time.time(); res = {}; info = []
    for fk, sets in FD.items():
        for fam in FAMS:
            m = load_pre(fam, seed)
            res[f"{fk}_{fam}_zeroshot"] = predict(m, sets["test"])
            for var in [v for v in VARIANTS[fam] if v != "scratch"]:
                m = load_pre(fam, seed)
                if var == "heads": freeze_heads_only(m, fam)
                params = [p for p in m.parameters() if p.requires_grad]
                n_tr = int(sum(p.numel() for p in params))
                m, ne, bv = train(m, sets, 1e-4, seed, params)
                res[f"{fk}_{fam}_{var}"] = predict(m, sets["test"]); info.append((fk, fam, var, ne, bv, n_tr))
            if "scratch" not in VARIANTS[fam]: continue
            # Note: the model is built before torch.manual_seed(seed), so its initial weights come from the RNG state
            # left by the preceding runs of this seed. That sequence is fixed, so the paper's scratch runs reproduce
            # exactly when this script is rerun as is; seeding before build() would give different (equally valid) runs.
            m = build(fam).to(DEV); torch.manual_seed(seed)
            m, ne, bv = train(m, sets, 3e-4, seed, [p for p in m.parameters()])
            res[f"{fk}_{fam}_scratch"] = predict(m, sets["test"]); info.append((fk, fam, "scratch", ne, bv))
    np.savez(OUT / f"{pf.stem}.{TMPTAG}.tmp.npz", **res); os.replace(OUT / f"{pf.stem}.{TMPTAG}.tmp.npz", pf)
    with open(OUT / "grasp_runlog.jsonl", "a") as fh: fh.write(json.dumps(dict(seed=seed, seconds=round(time.time() - t0, 1), runs=info)) + "\n")
    log(f"seed {seed} done in {time.time()-t0:.0f}s")
import shutil; shutil.rmtree(OUT / f"tmp_{TMPTAG}", ignore_errors=True); shutil.rmtree(OUT / f"tb_tmp_{TMPTAG}", ignore_errors=True)
log("GRASP COMPLETE")
