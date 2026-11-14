#!/usr/bin/env python
"""Classical (non-deep) baselines on exactly the windows of train_block.py.

  python classical_baselines.py --repo .. --block A --out j4_A --horizons 1,6,12,24 [--seeds 42-51] [--families ridge_res,gbm_res]

Both families forecast the change from persistence, y(t+h) = y(t) + f_h(features), one model per lead time:
  ridge_res  ridge regression on standardised summary features; the penalty is chosen on the validation windows.
             It is deterministic, so every seed file holds the same forecast (seed spread 0).
  gbm_res    LightGBM gradient-boosted trees (squared error), early stopping on the validation windows; the seed
             sets bagging and feature subsampling. Storm windows get weight 12, as in the deep models' sampler.
Features (75) summarise the same 72 h input window the deep models see, using only hours up to the forecast origin:
the last 24 hourly log fluxes, the 24 h and 72 h flux means, the last and 6 h mean flux spread, the observed fraction
of the last 24 h, the last value and the 6 h and 24 h means of the 15 driver channels, and the 24 h minimum of Bz.
The test windows, targets and probes (noise on the 16 continuous auxiliary channels, driver outages, late Kp/Dst)
are those of train_block.py, so the outputs (meta_<B>.npz, preds_<B>_<family>_s<seed>.npz) join the same analysis.
"""
import argparse, json, os, sys, time
from pathlib import Path

if __name__ == "__main__":
    def _arg(flag, default):
        return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
    os.environ["STORM_HSTEPS"] = _arg("--horizons", "1,6,12,24")    # read by storm_pipeline.data at import time
    os.environ["STORM_SEQ"] = _arg("--lookback", "72")
    sys.path.insert(0, str(Path(_arg("--repo", ".")).resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from storm_pipeline.data import build_features, Block, load_raw, SW_COLS, SCALED, HSTEPS, SEQ, BLOCKS, VAL  # noqa: E402

N_DRV = SW_COLS.index("log_flux_std_1h")          # channels 0..14 are drivers (as in train_block.py)
I_STD, I_MASK = SW_COLS.index("log_flux_std_1h"), SW_COLS.index("flux_obs")
I_KP, I_DST, I_BZ = SW_COLS.index("kp"), SW_COLS.index("dst"), SW_COLS.index("bz")
STORM_WEIGHT = 12.0
T0 = time.time()


def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)


def windows(blk, split):
    """Inputs (n, SEQ, 17), flux history (n, SEQ), targets (n, H), persistence (n,), storm flags (n,) of a split."""
    ws = blk.split(split)
    s = ws.starts.astype(np.int64); idx = s[:, None] + np.arange(SEQ)[None]
    X = blk.X.numpy()[idx]; F = blk.flux_in.numpy()[idx]
    e = s + SEQ
    y = np.stack([blk.flux_tg.numpy()[e + h - 1] for h in HSTEPS], 1); pers = blk.flux_tg.numpy()[e - 1]
    return X, F, y, pers, ws


def features(X, F):
    """Summary features of each window from hours up to the forecast origin only. X: (n, T, 17), F: (n, T)."""
    T = X.shape[1]
    lags = F[:, T - 24:][:, ::-1]                                           # y_t, y_{t-1}, ..., y_{t-23}
    out = [lags, F[:, T - 24:].mean(1, keepdims=True), F.mean(1, keepdims=True),
           X[:, -1, I_STD:I_STD + 1], X[:, T - 6:, I_STD].mean(1, keepdims=True),
           X[:, T - 24:, I_MASK].mean(1, keepdims=True),
           X[:, -1, :N_DRV], X[:, T - 6:, :N_DRV].mean(1), X[:, T - 24:, :N_DRV].mean(1),
           X[:, T - 24:, I_BZ].min(1, keepdims=True)]
    return np.concatenate(out, 1).astype(np.float32)


def apply_probe(X, probe, rng):
    """The degradations of train_block.py: noise on the 16 continuous auxiliary channels (mask and flux history
    untouched), drivers held at their last value for the final L hours or at the training mean, Kp/Dst late."""
    X = X.copy(); T = X.shape[1]; kind, val = probe
    if kind == "noise":
        X[..., :I_MASK] += val * rng.standard_normal(X[..., :I_MASK].shape).astype(np.float32)
    elif kind == "out":
        if val >= T: X[..., :N_DRV] = 0.0
        else: X[:, T - val:, :N_DRV] = X[:, T - val - 1:T - val, :N_DRV]
    elif kind == "lag":
        for i in (I_KP, I_DST):
            col = X[..., i].copy(); X[:, val:, i] = col[:, :T - val]; X[:, :val, i] = col[:, :1]
    return X


PROBES = {"noise0.5": ("noise", 0.5), "noise1": ("noise", 1.0), "noise2": ("noise", 2.0), "out6": ("out", 6),
          "out24": ("out", 24), "out72": ("out", 72), "lag3": ("lag", 3)}


class Ridge:
    def __init__(self): self.alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    def fit(self, Xtr, ytr, Xva, yva, w, seed):
        self.mu = Xtr.mean(0); self.sd = Xtr.std(0) + 1e-6
        A = (Xtr - self.mu) / self.sd; Av = (Xva - self.mu) / self.sd
        A1 = np.hstack([A, np.ones((len(A), 1), np.float32)]); Av1 = np.hstack([Av, np.ones((len(Av), 1), np.float32)])
        sw = np.sqrt(w)[:, None]; G = (A1 * sw).T @ (A1 * sw); b = (A1 * sw).T @ (ytr * sw[:, 0])
        best = None
        for a in self.alphas:
            reg = a * np.eye(G.shape[0]); reg[-1, -1] = 0.0                  # no penalty on the intercept
            coef = np.linalg.solve(G + reg, b); err = np.mean((Av1 @ coef - yva) ** 2)
            if best is None or err < best[0]: best = (err, a, coef)
        self.alpha, self.coef = best[1], best[2]; return self

    def predict(self, X):
        A = (X - self.mu) / self.sd; return np.hstack([A, np.ones((len(A), 1), np.float32)]) @ self.coef


class GBM:
    def __init__(self, n_estimators=3000): self.n_estimators = n_estimators

    def fit(self, Xtr, ytr, Xva, yva, w, seed):
        import lightgbm as lgb
        self.m = lgb.LGBMRegressor(n_estimators=self.n_estimators, learning_rate=0.03, num_leaves=63, min_child_samples=40,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                   random_state=seed, n_jobs=os.cpu_count(), verbose=-1)
        self.m.fit(Xtr, ytr, sample_weight=w, eval_X=(Xva,), eval_y=(yva,),
                   callbacks=[lgb.early_stopping(100, verbose=False)])
        return self

    def predict(self, X): return self.m.predict(X)


MODELS = {"ridge_res": Ridge, "gbm_res": GBM}


def export(m):
    """A fitted model as a plain dict (pickle-safe across library versions for ridge; LightGBM text model)."""
    if isinstance(m, Ridge): return {"kind": "ridge", "mu": m.mu, "sd": m.sd, "coef": m.coef, "alpha": m.alpha}
    return {"kind": "gbm", "model": m.m.booster_.model_to_string(), "best_iteration": m.m.best_iteration_}


def predict_exported(d, Fx):
    if d["kind"] == "ridge":
        A = (Fx - d["mu"]) / d["sd"]; return np.hstack([A, np.ones((len(A), 1), np.float32)]) @ d["coef"]
    import lightgbm as lgb
    return lgb.Booster(model_str=d["model"]).predict(Fx, num_iteration=d["best_iteration"] or None)


def predict_windowset(model_file, ws):
    """Zero-shot forecast of a WindowSet (as built by transfer_journal.py) with a saved block-A model file."""
    import pickle
    saved = pickle.load(open(model_file, "rb"))
    s = ws.starts.astype(np.int64); idx = s[:, None] + np.arange(SEQ)[None]
    X = ws.b.X.numpy()[idx]; F = ws.b.flux_in.numpy()[idx]; pers = ws.b.flux_tg.numpy()[s + SEQ - 1]
    Fx = features(X, F)
    assert [int(h) for h in saved["horizons"]] == list(HSTEPS), "model trained for other horizons"
    return np.stack([pers + predict_exported(d, Fx) for d in saved["models"]], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--block", required=True, choices=list("ABC"))
    ap.add_argument("--out", required=True); ap.add_argument("--horizons", default="1,6,12,24")
    ap.add_argument("--seeds", default="42-51"); ap.add_argument("--families", default="ridge_res,gbm_res")
    ap.add_argument("--n_estimators", type=int, default=3000, help="upper limit; early stopping picks the number")
    ap.add_argument("--max_train_windows", type=int, default=None, help="smoke test only")
    ap.add_argument("--stop_after_hours", type=float, default=None)
    ap.add_argument("--lookback", type=int, default=72, help="input window in hours; family names get the suffix _L<h>")
    args = ap.parse_args()
    a, b = map(int, args.seeds.split("-")); SEEDS = list(range(a, b + 1))
    fams = args.families.split(","); assert set(fams) <= set(MODELS), fams
    assert args.lookback == SEQ, "--lookback must be given before the module is imported (it sets STORM_SEQ)"
    if SEQ != 72: fams = [f"{f}_L{SEQ}" for f in fams]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    goes, omni = load_raw(args.repo); blk = Block(build_features(goes, omni), args.block)
    Xtr, Ftr, ytr, ptr, wtr = windows(blk, 0); Xva, Fva, yva, pva, _ = windows(blk, 1); Xte, Fte, yte, pte, wte = windows(blk, 2)
    if args.max_train_windows: Xtr, Ftr, ytr, ptr = Xtr[:args.max_train_windows], Ftr[:args.max_train_windows], ytr[:args.max_train_windows], ptr[:args.max_train_windows]
    wt = np.where(wtr.window_storm_flags[:len(ytr)] > 0, STORM_WEIGHT, 1.0)
    log(f"block {args.block}: windows train/val/test = {len(ytr)}/{len(yva)}/{len(yte)}; horizons {HSTEPS}")
    meta = out / f"meta_{args.block}.npz"
    if not meta.exists():
        np.savez(meta, y=yte, persist=pte, strict=wte.strict, storm_in=wte.window_storm_flags > 0, dst_target=wte.dst_target,
                 t_origin=wte.t_origin, scaler_mu=blk.scaler[0], scaler_sd=blk.scaler[1], stats=json.dumps(blk.stats()),
                 block_range=json.dumps(BLOCKS[args.block]), val_range=json.dumps(VAL), horizons=np.array(HSTEPS))
    Ftr_, Fva_ = features(Xtr, Ftr), features(Xva, Fva)
    test_feats = {"pred": features(Xte, Fte)}
    for k, pr in PROBES.items():
        test_feats[f"pred_{k}"] = features(apply_probe(Xte, pr, np.random.default_rng(0)), Fte)
    for fam in fams:
        for seed in SEEDS:
            pf = out / f"preds_{args.block}_{fam}_s{seed}.npz"
            if pf.exists() and (args.block != "A" or (out / f"model_{fam}_s{seed}.pkl").exists()): continue
            if args.stop_after_hours is not None and time.time() - T0 > args.stop_after_hours * 3600:
                log("time budget reached"); return
            t0 = time.time(); preds = {k: np.zeros((len(yte), len(HSTEPS)), np.float32) for k in test_feats}; info = []; saved = []
            for i, h in enumerate(HSTEPS):
                m = (GBM(args.n_estimators) if fam.startswith("gbm_res") else Ridge()).fit(Ftr_, ytr[:, i] - ptr, Fva_, yva[:, i] - pva, wt, seed)
                for k, Fx in test_feats.items(): preds[k][:, i] = pte + m.predict(Fx)
                info.append(getattr(m, "alpha", None) or getattr(getattr(m, "m", None), "best_iteration_", None)); saved.append(export(m))
            if args.block == "A":     # block-A models are reused for zero-shot transfer (transfer_journal.py)
                import pickle
                mf = out / f"model_{fam}_s{seed}.pkl"; tmpm = out / f"{mf.stem}.tmp.pkl"
                pickle.dump({"family": fam, "seed": seed, "horizons": list(HSTEPS), "models": saved}, open(tmpm, "wb")); os.replace(tmpm, mf)
            tmp = out / f"{pf.stem}.tmp.npz"; np.savez(tmp, **preds); os.replace(tmp, pf)
            pe = 1 - ((yte - preds["pred"]) ** 2).mean(0) / yte.var(0)
            log(f"done {fam} s{seed}: PE {np.round(pe, 3).tolist()}  ({info}, {time.time() - t0:.0f}s)")
            with open(out / f"runlog_{args.block}.jsonl", "a") as f:
                f.write(json.dumps(dict(block=args.block, family=fam, seed=seed, pe=pe.tolist(), info=[str(x) for x in info],
                                        seconds=round(time.time() - t0, 1))) + "\n")
    log("BLOCK COMPLETE")


if __name__ == "__main__":
    main()
