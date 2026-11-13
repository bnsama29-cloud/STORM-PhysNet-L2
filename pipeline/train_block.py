#!/usr/bin/env python
"""
Train all model families for ONE block (A, B or C) with the STORM-PhysNet pipeline.  Resumable: finished runs are skipped.

  python train_block.py --repo /kaggle/working/STORM-PhysNet-L2 --block A --out /kaggle/working/out_A

Outputs in --out:
  meta_<B>.npz                       targets, persistence, flags, times, scaler for the block's test windows
  preds_<B>_<family>_s<seed>.npz     test predictions (N,H) (+ noisy-input predictions, tau, gate where relevant;
                                     + logvar, the predicted log-variance, for the STORM-PhysNet families)
  ckpt_<B>/<family>_s<seed>_best.pt  checkpoints (block A only; needed by grasp_transfer.py)
  runlog_<B>.jsonl                   one line per run: epochs, best val MSE, seconds

Driver-robustness probes (--probes): besides Gaussian noise, each listed family is also scored with
  pred_out<L>   drivers unavailable for the last L input hours (L = 6, 24: held at their last available value;
                L = 72: no driver data at all, i.e. every driver at its training mean)
  pred_lag<L>   Kp and Dst arrive L hours late (each hour sees the value from L hours earlier)
The flux history, within-hour flux spread and observed-hour mask are never altered by these probes.
Two processes may share one --out directory (e.g. one per GPU with different --seeds).
"""
import argparse, contextlib, io, json, os, sys, time, copy
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, help="path to a clone of STORM-PhysNet-L2 (needs datasets/ + src/)")
ap.add_argument("--block", required=True, choices=["A", "B", "C"])
ap.add_argument("--out", required=True)
ap.add_argument("--families", default="lstm,transformer_matched,tf_ffn_res,tf_flux_only,storm_bz,storm_no_delay,storm_no_gate")
ap.add_argument("--seeds", default="42-51", help="e.g. 42-51 or 42,43,44")
ap.add_argument("--epochs", type=int, default=None, help="override config epochs (smoke test only)")
ap.add_argument("--max_train_windows", type=int, default=None, help="smoke test only: subsample train windows")
ap.add_argument("--noise_sigmas", default="0.5,1,2")
ap.add_argument("--keep_ckpt", default="A", help="blocks whose checkpoints are kept")
ap.add_argument("--noise_families", default="storm_bz,transformer_matched,tf_ffn_res",
                help="families that get the noise probe (and the --probes set)")
ap.add_argument("--probes", action="store_true", help="also run the driver-outage and index-latency probes")
ap.add_argument("--horizons", default="1,6,12", help="forecast horizons in hours; 1,6,12 reproduces the AGERS 2026 runs")
ap.add_argument("--stop_after_hours", type=float, default=None,
                help="start no new run after this many hours (finished runs are kept; run again later to continue)")
args = ap.parse_args()
os.environ["STORM_HSTEPS"] = args.horizons          # read by storm_pipeline.data at import time

import numpy as np, pandas as pd, torch, yaml
from storm_pipeline.data import (build_features, Block, make_loaders, load_raw, SW_COLS, HSTEPS, BLOCKS, VAL)

REPO = Path(args.repo).resolve()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
CK = OUT / f"ckpt_{args.block}"; CK.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO))
from src.training.trainer import Trainer

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.seeds.count("-") == 1 and "," not in args.seeds:
    a, b = map(int, args.seeds.split("-")); SEEDS = list(range(a, b + 1))
else:
    SEEDS = [int(s) for s in args.seeds.split(",")]
FAMILIES = args.families.split(",")
SIGMAS = [float(s) for s in args.noise_sigmas.split(",") if s]
NOISE_FAMILIES = set(args.noise_families.split(","))
N_DRV = SW_COLS.index("log_flux_std_1h")                     # channels 0..N_DRV-1 are solar-wind / geomagnetic drivers
I_KP, I_DST = SW_COLS.index("kp"), SW_COLS.index("dst")
PROBES = ([("out", 6), ("out", 24), ("out", 72), ("lag", 3)] if args.probes else [])

from storm_pipeline.families import SPEC, configure, build as build_family
for f in FAMILIES:
    assert f in SPEC, f"unknown family {f}"

base_cfg = yaml.safe_load(open(REPO / "configs/config.yaml"))
if args.epochs: base_cfg["training"]["epochs"] = args.epochs
H_WEIGHT = {1: 1.0, 6: 0.7, 12: 0.5, 24: 0.4}                      # loss weight per horizon (1/6/12 as in the paper)
assert all(h in H_WEIGHT for h in HSTEPS), f"no loss weight defined for horizons {HSTEPS}"
base_cfg["data"]["forecast_horizons"] = [float(h) for h in HSTEPS]
base_cfg["training"]["horizon_weights"] = [H_WEIGHT[h] for h in HSTEPS]
base_cfg["loss"]["physics_horizon_scale"] = [1.0] + [0.0] * (len(HSTEPS) - 1)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------------------------- data
log(f"device={DEV}  block={args.block}  horizons={HSTEPS}  families={FAMILIES}  seeds={SEEDS[0]}..{SEEDS[-1]}")
goes, omni = load_raw(str(REPO))
feat = build_features(goes, omni)
blk = Block(feat, args.block)
st = blk.stats(); log(f"rows/windows: {json.dumps(st)}")
test_set = blk.split(2)
meta_path = OUT / f"meta_{args.block}.npz"
TMPTAG = f"{os.getpid()}"                                          # per-process scratch names (two GPUs may share OUT)
if not meta_path.exists():
    y = np.stack([blk.flux_tg[test_set.starts + 72 + h - 1].numpy() for h in HSTEPS], 1)
    pers = blk.flux_tg[test_set.starts + 71].numpy()
    tmp_meta = OUT / f"meta_{args.block}.{TMPTAG}.tmp.npz"
    np.savez(tmp_meta, y=y, persist=pers, strict=test_set.strict, storm_in=test_set.window_storm_flags > 0,
             dst_target=test_set.dst_target, t_origin=test_set.t_origin.astype("datetime64[ns]").astype(np.int64),
             scaler_mu=blk.scaler[0], scaler_sd=blk.scaler[1], stats=json.dumps(st),
             block_range=json.dumps(BLOCKS[args.block]), val_range=json.dumps(VAL), horizons=np.array(HSTEPS))
    os.replace(tmp_meta, meta_path)
    log(f"wrote {meta_path.name}: {len(y)} test windows, {int(test_set.strict.sum())} strict, {int((test_set.dst_target[:, 1] <= -50).sum())} storm@6h")


def make_model(family, trainer, n_sw):
    return build_family(family, trainer, n_sw, DEV, scaler=blk.scaler)


def apply_probe(x, probe):
    """x: (B, T, 17) standardised inputs. Returns a modified copy; only driver channels are touched."""
    kind, L = probe; x = x.clone(); T = x.shape[1]
    if kind == "out":
        if L >= T: x[..., :N_DRV] = 0.0                                          # no driver data: training mean
        else: x[:, T - L:, :N_DRV] = x[:, T - L - 1:T - L, :N_DRV]                # hold the last available value
    elif kind == "lag":
        for i in (I_KP, I_DST):
            col = x[..., i].clone(); x[:, L:, i] = col[:, :T - L]; x[:, :L, i] = col[:, :1]
    return x


@torch.no_grad()
def predict(model, loader, sigma=0.0, aux=False, probe=None, logvar=False):
    model.eval(); g = torch.Generator().manual_seed(0); P, T, G, LV, SP = [], [], [], [], []
    for b in loader:
        x = b["x_sw"]
        if sigma > 0:
            noise = sigma * torch.randn(x.shape, generator=g); noise[..., -1] = 0.0   # never perturb the observed-mask channel
            x = x + noise
        if probe is not None:
            x = apply_probe(x, probe)
        x, xf, yp = x.to(DEV), b["x_flux"].to(DEV), b["y_persist"].to(DEV)
        o = model(x_sw=x, x_flux=xf, y_persist=yp)
        P.append(o["flux_pred"].cpu().numpy())
        if logvar: LV.append(o["log_var"].cpu().numpy())
        if "spectral_flux_pred" in o: SP.append(o["spectral_flux_pred"].cpu().numpy())
        if aux and "tau" in o:
            T.append(o["tau"].reshape(len(x), -1).mean(1).cpu().numpy()); G.append(o["gate_values"].reshape(len(x), -1).mean(1).cpu().numpy())
    out = {"pred": np.concatenate(P).astype(np.float32)}
    if logvar: out["logvar"] = np.concatenate(LV).astype(np.float32)
    if SP: out["spectral"] = np.concatenate(SP).astype(np.float32)
    if aux and T: out["tau"] = np.concatenate(T).astype(np.float32); out["gate"] = np.concatenate(G).astype(np.float32)
    return out


runlog = OUT / f"runlog_{args.block}.jsonl"
T_START = time.time()


def out_of_time():
    return args.stop_after_hours is not None and time.time() - T_START > args.stop_after_hours * 3600


stopped = False
for seed in SEEDS:                               # seed-major order so that partial results are balanced across families
    for fam in FAMILIES:
        pf = OUT / f"preds_{args.block}_{fam}_s{seed}.npz"
        if pf.exists():
            continue
        if out_of_time():
            stopped = True; break
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        cfg = configure(copy.deepcopy(base_cfg), fam)
        cfg["training"]["checkpoint_dir"] = str(CK); cfg["training"]["log_dir"] = str(OUT / f"tb_tmp_{TMPTAG}")
        trainer = Trainer(cfg)
        tr_loader, va_loader, te_loader, tr_ds, va_ds, te_ds = make_loaders(blk, batch_size=cfg["training"]["batch_size"],
                                                                           storm_weight=cfg["training"]["storm_weight"], seed=seed)
        if args.max_train_windows:
            tr_ds.starts = tr_ds.starts[: args.max_train_windows]; tr_ds.window_storm_flags = tr_ds.window_storm_flags[: args.max_train_windows]
            tr_ds.strict = tr_ds.strict[: args.max_train_windows]
            tr_loader, va_loader, te_loader, *_ = make_loaders(blk, 64, 12.0, seed)  # rebuilt below with truncated train set
            from torch.utils.data import DataLoader, WeightedRandomSampler
            w = torch.from_numpy(np.where(tr_ds.window_storm_flags > 0, 12.0, 1.0).astype(np.float32))
            tr_loader = DataLoader(tr_ds, batch_size=64, sampler=WeightedRandomSampler(w, len(tr_ds), True), drop_last=True)
        model = make_model(fam, trainer, len(SW_COLS))
        name = f"{fam}_s{seed}"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model = trainer.train_single(model, tr_loader, va_loader, model_name=name)
        txt = buf.getvalue()
        ep = txt.count("Epoch ") ; vals = [float(l.split("Val MSE:")[1].split("|")[0]) for l in txt.splitlines() if "Val MSE:" in l]
        trainer.writer.close()
        res = predict(model, te_loader, aux=(fam == "storm_bz"), logvar=(SPEC[fam]["model_type"] == "storm_physnet"))
        save = {"pred": res["pred"]}
        if "logvar" in res: save["logvar"] = res["logvar"]
        if "spectral" in res: save["pred_spectral"] = res["spectral"]
        if "tau" in res: save["tau"], save["gate"] = res["tau"], res["gate"]
        if fam in NOISE_FAMILIES:
            for sg in SIGMAS: save[f"pred_noise{sg:g}"] = predict(model, te_loader, sigma=sg)["pred"]
            for pr in PROBES: save[f"pred_{pr[0]}{pr[1]}"] = predict(model, te_loader, probe=pr)["pred"]
        tmp_pf = OUT / f"{pf.stem}.{TMPTAG}.tmp.npz"; np.savez(tmp_pf, **save); os.replace(tmp_pf, pf)
        if args.block not in args.keep_ckpt.split(","):
            (CK / f"{name}_best.pt").unlink(missing_ok=True)
        rec = dict(block=args.block, family=fam, seed=seed, epochs=ep, best_val_mse=min(vals) if vals else None,
                   seconds=round(time.time() - t0, 1), n_params=int(sum(p.numel() for p in model.parameters())))
        with open(runlog, "a") as fh: fh.write(json.dumps(rec) + "\n")
        log(f"done {fam} s{seed}: {rec['epochs']} epochs, val_mse={rec['best_val_mse']}, {rec['seconds']}s")
        del model, trainer
        if DEV.type == "cuda": torch.cuda.empty_cache()

    if stopped:
        break

import shutil; shutil.rmtree(OUT / f"tb_tmp_{TMPTAG}", ignore_errors=True)
log(f"TIME BUDGET REACHED after {args.stop_after_hours} h: run again to continue" if stopped else "BLOCK COMPLETE")
