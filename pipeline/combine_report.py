#!/usr/bin/env python
"""
Combine the outputs of the three Kaggle accounts into every table / figure / number the paper needs.

  python combine_report.py --inputs out_A out_B out_C out_G --out report

Writes:  report/tables/*.csv + *.tex,  report/figures/*.png,  report/numbers.json,  report/PAPER_NUMBERS.md
Everything is computed from the saved prediction files; nothing is hard-coded.
Conventions:
  PE_clim = 1 - MSE / Var(y)      (Var over the block's own test targets, per horizon)
  PE_pers = 1 - MSE / MSE(persistence)    (persistence = last observed log flux)
  "seed mean" = mean over seeds of the per-seed score; "bagged" = score of the mean prediction of all seeds.
  Paired tests: same seed index for both models; d_z = t / sqrt(n).
  CIs: 7-day moving-block bootstrap over test windows (2000 resamples).
"""
import argparse, glob, json, os, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--thr", type=float, default=3.0, help="event threshold in log10 flux (3.0 = 1000 pfu)")
ap.add_argument("--nboot", type=int, default=2000)
args = ap.parse_args()
OUT = Path(args.out); (OUT / "tables").mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
H = [1, 6, 12]
NAMES = {"lstm": "LSTM", "transformer_matched": "Transformer (matched)", "tf_ffn_res": "Transformer + residual head",
         "tf_flux_only": "Transformer, flux-only", "storm_bz": "STORM-PhysNet", "storm_no_delay": "STORM, no delay",
         "storm_no_gate": "STORM, no gate"}
ORDER = list(NAMES)
NUM = {}                                                         # everything that goes in numbers.json
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "serif"})
COL = {"lstm": "#888888", "transformer_matched": "#d95f02", "tf_ffn_res": "#7570b3", "tf_flux_only": "#a6761d",
       "storm_bz": "#1b9e77", "storm_no_delay": "#66a61e", "storm_no_gate": "#e7298a"}

# ------------------------------------------------------------------------------------------------ load
files = [f for d in args.inputs for f in glob.glob(str(Path(d) / "**/*"), recursive=True)]
metas = {}; P = {}; runlogs = []
for f in files:
    n = os.path.basename(f)
    m = re.match(r"meta_([ABC])\.npz", n)
    if m: metas[m.group(1)] = dict(np.load(f, allow_pickle=True))
    m = re.match(r"preds_([ABC])_(.+)_s(\d+)\.npz", n)
    if m: P.setdefault(m.group(1), {}).setdefault(m.group(2), {})[int(m.group(3))] = dict(np.load(f))
    if n.startswith("runlog_") or n == "grasp_runlog.jsonl":
        runlogs += [json.loads(l) for l in open(f) if l.strip()]
BLOCKS = [b for b in "ABC" if b in metas and b in P]
assert BLOCKS, "no block results found"
print("blocks:", BLOCKS, {b: {f: len(s) for f, s in P[b].items()} for b in BLOCKS})
# only keep seeds that exist for every family of a block (so paired tests are valid)
for b in BLOCKS:
    common = set.intersection(*[set(s) for s in P[b].values()])
    for f in P[b]: P[b][f] = {k: v for k, v in P[b][f].items() if k in common}
    NUM[f"seeds_{b}"] = sorted(common)


def pe_clim(y, p): return 1 - ((y - p) ** 2).mean(0) / y.var(0)
def pe_pers(y, p, pers): return 1 - ((y - p) ** 2).mean(0) / ((y - pers[:, None]) ** 2).mean(0)
def mse(y, p): return ((y - p) ** 2).mean(0)

def get(b, fam, key="pred"): return np.stack([P[b][fam][s][key] for s in sorted(P[b][fam])])        # (S,N,3)

def score_family(b, fam, mask=None, key="pred"):
    m = metas[b]; y, pers = m["y"], m["persist"]
    mk = np.ones(len(y), bool) if mask is None else mask
    A = get(b, fam, key)[:, mk]
    pc = np.stack([pe_clim(y[mk], a) for a in A]); pp = np.stack([pe_pers(y[mk], a, pers[mk]) for a in A])
    bag = A.mean(0)
    return dict(pc=pc, pp=pp, bag_pc=pe_clim(y[mk], bag), bag_pp=pe_pers(y[mk], bag, pers[mk]), bag=bag)

def fmt(a, d=3):
    a = np.atleast_1d(a)
    return f"{np.mean(a):.{d}f} ± {np.std(a, ddof=1):.{d}f}" if len(a) > 1 else f"{np.mean(a):.{d}f}"

def paired(a, b_):
    d = np.asarray(a) - np.asarray(b_); n = len(d)
    if n < 2 or d.std(ddof=1) == 0: return dict(diff=float(d.mean()), t=np.nan, p=np.nan, dz=np.nan, n=n)
    t, p = stats.ttest_1samp(d, 0.0); return dict(diff=float(d.mean()), t=float(t), p=float(p), dz=float(t / np.sqrt(n)), n=n)

def save(df, name, caption="", index=False, floatfmt=None):
    df.to_csv(OUT / "tables" / f"{name}.csv", index=index)
    tex = df.to_latex(index=index, escape=True, na_rep="--", column_format="l" + "c" * (df.shape[1] - (0 if index else 1)))
    (OUT / "tables" / f"{name}.tex").write_text(f"% {caption}\n{tex}")

def boot_idx(t_origin, n_boot, seed=0, block_days=7):
    day = (t_origin - t_origin.min()) // (3600 * 24 * 10 ** 9)
    wk = (day // block_days).astype(int); ids = np.unique(wk); groups = [np.where(wk == w)[0] for w in ids]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups)); yield np.concatenate([groups[i] for i in pick])

# ------------------------------------------------------------------------------------------------ 0. data / split table
rows = []
for b in BLOCKS:
    st = json.loads(str(metas[b]["stats"])); br = json.loads(str(metas[b]["block_range"]))
    rows.append(dict(Block=b, Test_period=f"{br[0][:10]} to {br[1][:10]}",
                     Train_windows=st["train"]["windows"], Val_windows=st["val"]["windows"], Test_windows=st["test"]["windows"],
                     Test_strict_windows=int(metas[b]["strict"].sum()),
                     Test_storm6h=int((metas[b]["dst_target"][:, 1] <= -50).sum()),
                     Filled_train=f"{100*st['train']['filled_frac']:.1f}%", Filled_test=f"{100*st['test']['filled_frac']:.1f}%",
                     Min_Dst_test=int(metas[b]["dst_target"].min())))
split_df = pd.DataFrame(rows); save(split_df, "T0_splits", "Block definitions and window counts"); NUM["splits"] = rows

# ------------------------------------------------------------------------------------------------ 1. main table
fam_all = [f for f in ORDER if all(f in P[b] for b in BLOCKS)]
main_rows, bag_rows = [], []
SC = {b: {f: score_family(b, f) for f in fam_all} for b in BLOCKS}
for b in BLOCKS:
    y, pers = metas[b]["y"], metas[b]["persist"]
    pp = pe_clim(y, np.repeat(pers[:, None], 3, 1))
    main_rows.append(dict(Block=b, Model="Persistence", **{f"PE_clim {h}h": f"{pp[i]:.3f}" for i, h in enumerate(H)},
                          **{f"PE_pers {h}h": "0.000" for h in H}))
    for f in fam_all:
        s = SC[b][f]
        main_rows.append(dict(Block=b, Model=NAMES[f], **{f"PE_clim {h}h": fmt(s["pc"][:, i]) for i, h in enumerate(H)},
                              **{f"PE_pers {h}h": fmt(s["pp"][:, i], 2) for i, h in enumerate(H)}))
        bag_rows.append(dict(Block=b, Model=NAMES[f], **{f"Bagged PE_clim {h}h": f"{s['bag_pc'][i]:.3f}" for i, h in enumerate(H)}))
main_df = pd.DataFrame(main_rows); save(main_df, "T1_main_per_block", "Seed mean ± sd; strict-subset in T2")
bag_df = pd.DataFrame(bag_rows); save(bag_df, "T1b_bagged", "Mean prediction over all seeds")
# average over blocks
avg = []
for f in fam_all:
    avg.append(dict(Model=NAMES[f], **{f"PE_clim {h}h": f"{np.mean([SC[b][f]['pc'][:, i].mean() for b in BLOCKS]):.3f}" for i, h in enumerate(H)},
                    **{f"PE_pers {h}h": f"{np.mean([SC[b][f]['pp'][:, i].mean() for b in BLOCKS]):.2f}" for i, h in enumerate(H)},
                    **{f"Bagged {h}h": f"{np.mean([SC[b][f]['bag_pc'][i] for b in BLOCKS]):.3f}" for i, h in enumerate(H)}))
avg_df = pd.DataFrame(avg); save(avg_df, "T1c_main_block_average", "Average of the per-block seed means")
NUM["main"] = {b: {f: dict(pe_clim_mean=SC[b][f]["pc"].mean(0).tolist(), pe_clim_sd=SC[b][f]["pc"].std(0, ddof=1).tolist() if len(SC[b][f]["pc"]) > 1 else None,
                           pe_pers_mean=SC[b][f]["pp"].mean(0).tolist(), bag_pe_clim=SC[b][f]["bag_pc"].tolist(), bag_pe_pers=SC[b][f]["bag_pp"].tolist())
               for f in fam_all} for b in BLOCKS}

# ------------------------------------------------------------------------------------------------ 2. strict subset (no filled hour in lookback)
srows = []
for b in BLOCKS:
    mk = metas[b]["strict"].astype(bool); NUM.setdefault("strict_n", {})[b] = int(mk.sum())
    for f in fam_all:
        s = score_family(b, f, mk)
        srows.append(dict(Block=b, Model=NAMES[f], N=int(mk.sum()), **{f"PE_clim {h}h": fmt(s["pc"][:, i]) for i, h in enumerate(H)}))
save(pd.DataFrame(srows), "T2_strict_subset", "Windows whose whole 72-h lookback is observed")

# ------------------------------------------------------------------------------------------------ 3. paired tests
prow = []; PT = {}
pairs = [("storm_bz", "transformer_matched"), ("storm_bz", "tf_ffn_res"), ("storm_bz", "tf_flux_only"), ("storm_bz", "lstm"),
         ("storm_bz", "storm_no_delay"), ("storm_bz", "storm_no_gate"), ("tf_ffn_res", "transformer_matched"), ("transformer_matched", "tf_flux_only")]
for a_, b_ in pairs:
    if a_ not in fam_all or b_ not in fam_all: continue
    for b in BLOCKS:
        for i, h in enumerate(H):
            r = paired(SC[b][a_]["pc"][:, i], SC[b][b_]["pc"][:, i])
            prow.append(dict(A=NAMES[a_], B=NAMES[b_], Block=b, Horizon=f"{h}h", dPE=f"{r['diff']:+.4f}", t=f"{r['t']:.2f}", p=f"{r['p']:.2e}", d_z=f"{r['dz']:.2f}", n=r["n"]))
            PT[f"{a_}__{b_}__{b}__{h}h"] = r
    for i, h in enumerate(H):                                       # all blocks pooled: paired by (block, seed)
        da = np.concatenate([SC[b][a_]["pc"][:, i] for b in BLOCKS]); db = np.concatenate([SC[b][b_]["pc"][:, i] for b in BLOCKS])
        r = paired(da, db)
        prow.append(dict(A=NAMES[a_], B=NAMES[b_], Block="pooled", Horizon=f"{h}h", dPE=f"{r['diff']:+.4f}", t=f"{r['t']:.2f}", p=f"{r['p']:.2e}", d_z=f"{r['dz']:.2f}", n=r["n"]))
        PT[f"{a_}__{b_}__pooled__{h}h"] = r
save(pd.DataFrame(prow), "T3_paired_tests", "Paired by seed. dPE = PE_clim(A) - PE_clim(B). pooled = pairs over (block,seed)"); NUM["paired"] = PT

# ------------------------------------------------------------------------------------------------ 4. storm bins, event skill
def hss(y_ev, p_ev):
    tp = (y_ev & p_ev).sum(); fn = (y_ev & ~p_ev).sum(); fp = (~y_ev & p_ev).sum(); tn = (~y_ev & ~p_ev).sum()
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return dict(HSS=float(2 * (tp * tn - fn * fp) / den) if den else np.nan, POD=float(tp / (tp + fn)) if tp + fn else np.nan,
                FAR=float(fp / (tp + fp)) if tp + fp else np.nan, events=int(y_ev.sum()), n=int(len(y_ev)))
ev_rows, st_rows = [], []
for b in BLOCKS:
    m = metas[b]; y = m["y"]
    for i, h in enumerate(H):
        yev = y[:, i] > args.thr
        cands = {"Persistence": np.repeat(m["persist"][:, None], 3, 1)[:, i]}
        for f in ["lstm", "transformer_matched", "tf_ffn_res", "storm_bz"]:
            if f in fam_all: cands[NAMES[f]] = SC[b][f]["bag"][:, i]
        for nm, pv in cands.items():
            ev_rows.append(dict(Block=b, Horizon=f"{h}h", Model=nm, **{k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in hss(yev, pv > args.thr).items()}))
    storm = m["dst_target"][:, 1] <= -50
    for lab, mk in [("storm (Dst<=-50 at t+6h)", storm), ("non-storm", ~storm)]:
        if mk.sum() < 5: continue
        for f in fam_all:
            e = SC[b][f]["bag"][mk] - y[mk]
            st_rows.append(dict(Block=b, Subset=lab, N=int(mk.sum()), Model=NAMES[f], **{f"RMSE {h}h": f"{np.sqrt((e[:, i]**2).mean()):.3f}" for i, h in enumerate(H)}))
        e = y[mk] - m["persist"][mk][:, None]
        st_rows.append(dict(Block=b, Subset=lab, N=int(mk.sum()), Model="Persistence", **{f"RMSE {h}h": f"{np.sqrt((e[:, i]**2).mean()):.3f}" for i, h in enumerate(H)}))
save(pd.DataFrame(ev_rows), "T4_event_skill", f"Event: log10 flux > {args.thr} (bagged predictions)")
save(pd.DataFrame(st_rows), "T5_storm_vs_quiet", "RMSE (log10 flux units) of bagged predictions")

# ------------------------------------------------------------------------------------------------ 5. block bootstrap CIs
ci_rows = []; NUM["ci"] = {}
for b in BLOCKS:
    m = metas[b]; y, pers = m["y"], m["persist"]; bs = list(boot_idx(m["t_origin"], args.nboot))
    for fa, fb in [("storm_bz", "transformer_matched"), ("storm_bz", "tf_ffn_res")]:
        if fa not in fam_all or fb not in fam_all: continue
        pa, pb = SC[b][fa]["bag"], SC[b][fb]["bag"]; d = np.stack([pe_clim(y[ix], pa[ix]) - pe_clim(y[ix], pb[ix]) for ix in bs])
        for i, h in enumerate(H):
            lo, hi = np.percentile(d[:, i], [2.5, 97.5])
            ci_rows.append(dict(Block=b, Comparison=f"{NAMES[fa]} - {NAMES[fb]} (bagged)", Horizon=f"{h}h", dPE=f"{pe_clim(y, pa)[i]-pe_clim(y, pb)[i]:+.4f}", CI95=f"[{lo:+.4f}, {hi:+.4f}]", excludes_0=bool(lo > 0 or hi < 0)))
            NUM["ci"][f"{fa}-{fb}-{b}-{h}h"] = [float(lo), float(hi)]
    for f in ["storm_bz", "transformer_matched"]:
        if f not in fam_all: continue
        pf = SC[b][f]["bag"]; d = np.stack([pe_clim(y[ix], pf[ix]) for ix in bs])
        for i, h in enumerate(H):
            lo, hi = np.percentile(d[:, i], [2.5, 97.5])
            ci_rows.append(dict(Block=b, Comparison=f"{NAMES[f]} PE_clim (bagged)", Horizon=f"{h}h", dPE=f"{pe_clim(y, pf)[i]:.4f}", CI95=f"[{lo:.4f}, {hi:.4f}]", excludes_0=""))
save(pd.DataFrame(ci_rows), "T6_bootstrap_ci", "7-day block bootstrap, 2000 resamples")

# ------------------------------------------------------------------------------------------------ 6. noise probe
nz = []; sig = [0.0, 0.5, 1.0, 2.0]
for f in ["storm_bz", "transformer_matched", "tf_ffn_res"]:
    if f not in fam_all: continue
    for b in BLOCKS:
        for s in sig:
            key = "pred" if s == 0 else f"pred_noise{s:g}"
            if key not in next(iter(P[b][f].values())): continue
            sc = score_family(b, f, key=key)
            nz.append(dict(Model=NAMES[f], Block=b, sigma=s, **{f"PE_clim {h}h": float(sc["pc"][:, i].mean()) for i, h in enumerate(H)}))
nz_df = pd.DataFrame(nz); nz_df.to_csv(OUT / "tables/T7_noise_raw.csv", index=False)
if len(nz_df):
    t7 = nz_df.groupby(["Model", "sigma"]).mean(numeric_only=True).reset_index().round(3); save(t7, "T7_noise_blockavg", "Gaussian noise (sigma in standardised units) on solar-wind channels at test time")
    NUM["noise"] = t7.to_dict("records")

# ------------------------------------------------------------------------------------------------ 7. gate / delay diagnostics
gd = {}
if "storm_bz" in fam_all:
    T, G, S = [], [], []
    for b in BLOCKS:
        A = P[b]["storm_bz"]; sd = sorted(A)
        if "tau" not in A[sd[0]]: continue
        tau = np.stack([A[s]["tau"] for s in sd]); gate = np.stack([A[s]["gate"] for s in sd])
        storm = metas[b]["dst_target"][:, 0] <= -50
        T.append(tau); G.append(gate); S.append(np.repeat(storm[None], len(sd), 0))
    if T:
        tau = np.concatenate([t.ravel() for t in T]); gate = np.concatenate([g.ravel() for g in G]); storm = np.concatenate([s.ravel() for s in S])
        sat = ((tau > 1.49) | (tau < 0.51)).mean()
        per_seed_sat = [float(((t > 1.49) | (t < 0.51)).mean(1).mean()) for t in T]
        gd = dict(tau_mean=float(tau.mean()), tau_sd=float(tau.std()), tau_saturated_frac=float(sat),
                  tau_saturated_range_over_blocks=[float(min(per_seed_sat)), float(max(per_seed_sat))],
                  gate_mean_storm=float(gate[storm].mean()) if storm.any() else None, gate_mean_quiet=float(gate[~storm].mean()),
                  n_storm_windows_pooled=int(storm.sum()))
        # AUC per seed (windows are not independent, so no pooled p-value is reported); mean +- sd over (block, seed)
        aucs = []
        for g_, s_ in zip(G, S):
            for gs, ss in zip(g_, s_):
                if ss.sum() > 10 and (~ss).sum() > 10:
                    aucs.append(stats.mannwhitneyu(gs[ss], gs[~ss]).statistic / (ss.sum() * (~ss).sum()))
        if aucs: gd["gate_AUC_storm_vs_quiet_mean"] = float(np.mean(aucs)); gd["gate_AUC_sd"] = float(np.std(aucs)); gd["gate_AUC_n"] = len(aucs)
        NUM["gate_delay"] = gd
        # column-width figure as in the paper (Fig. 7): tau as % of windows, gate densities for storm (Dst<=-50 nT at 1 h) vs quiet
        with plt.rc_context({"font.size": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7}):
            fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.7))
            ax[0].hist(tau, bins=np.linspace(0.5, 1.5, 41), weights=np.full(tau.shape, 100 / tau.size), color=COL["storm_bz"])
            ax[0].set_xlim(0.5, 1.52); ax[0].set_xlabel(r"learned delay $\tau$ (h)"); ax[0].set_ylabel("windows (%)")
            gb = np.linspace(min(gate.min(), .34), max(gate.max(), .5), 36)
            ax[1].hist(gate[~storm], bins=gb, alpha=.6, density=True, label="quiet", color="#888")
            if storm.any(): ax[1].hist(gate[storm], bins=gb, alpha=.6, density=True, label="storm", color="#d95f02")
            ax[1].set_xlabel("gate activation"); ax[1].set_ylabel("density"); ax[1].legend(frameon=False, loc="upper left", handlelength=1)
            for a in ax: a.grid(alpha=.3, lw=.5)
            fig.tight_layout(pad=0.3, w_pad=0.8); fig.savefig(OUT / "figures/fig_gate_delay.png", dpi=400); plt.close(fig)

# ------------------------------------------------------------------------------------------------ 8. figures: horizon, blocks, case study
fig, ax = plt.subplots(1, 3, figsize=(7, 2.4), sharey=False)
for k, b in enumerate(BLOCKS[:3]):
    for f in fam_all:
        mu = SC[b][f]["pc"].mean(0); sd_ = SC[b][f]["pc"].std(0, ddof=1) if len(SC[b][f]["pc"]) > 1 else 0 * mu
        ax[k].errorbar(H, mu, sd_, marker="o", ms=3, lw=1, capsize=2, color=COL[f], label=NAMES[f])
    y, pers = metas[b]["y"], metas[b]["persist"]; ax[k].plot(H, pe_clim(y, np.repeat(pers[:, None], 3, 1)), "k--", lw=1, label="Persistence")
    ax[k].set_title(f"Block {b}"); ax[k].set_xlabel("horizon (h)"); ax[k].set_xticks(H)
ax[0].set_ylabel(r"$PE_{clim}$"); h_, l_ = ax[-1].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=7.5, frameon=False); fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig(OUT / "figures/fig_horizon.png"); plt.close(fig)

if "storm_bz" in fam_all and "transformer_matched" in fam_all:
    fig, ax = plt.subplots(1, 3, figsize=(7, 2.3))
    for i, h in enumerate(H):
        for j, b in enumerate(BLOCKS):
            d = SC[b]["storm_bz"]["pc"][:, i] - SC[b]["transformer_matched"]["pc"][:, i]
            ax[i].bar(j, d.mean(), yerr=d.std(ddof=1) if len(d) > 1 else 0, color=["#1b9e77" if d.mean() > 0 else "#d95f02"][0], capsize=3)
        ax[i].axhline(0, color="k", lw=.8); ax[i].set_xticks(range(len(BLOCKS))); ax[i].set_xticklabels(BLOCKS); ax[i].set_title(f"{h} h"); ax[i].set_xlabel("block")
    ax[0].set_ylabel(r"$\Delta PE_{clim}$ (STORM-PhysNet − matched TF)"); fig.tight_layout(); fig.savefig(OUT / "figures/fig_block_delta.png"); plt.close(fig)

if "storm_bz" in fam_all and len(nz_df):
    fig, ax = plt.subplots(1, 3, figsize=(7, 2.3), sharey=True)
    for i, h in enumerate(H):
        for f in ["storm_bz", "transformer_matched", "tf_ffn_res"]:
            q = nz_df[nz_df.Model == NAMES[f]].groupby("sigma")[f"PE_clim {h}h"].mean()
            if len(q): ax[i].plot(q.index, q.values, marker="o", ms=3, color=COL[f], label=NAMES[f])
        ax[i].set_title(f"{h} h"); ax[i].set_xlabel(r"noise $\sigma$ (standardised units)")
    ax[0].set_ylabel(r"$PE_{clim}$"); ax[0].legend(fontsize=7.5); fig.tight_layout(); fig.savefig(OUT / "figures/fig_noise.png"); plt.close(fig)

# case study: deepest-Dst event of the pooled test blocks, 6-h lead
bb = min(BLOCKS, key=lambda b: metas[b]["dst_target"][:, 1].min())
m = metas[bb]; i0 = int(np.argmin(m["dst_target"][:, 1])); lo, hi = max(0, i0 - 120), min(len(m["y"]), i0 + 120)
t = pd.to_datetime(m["t_origin"][lo:hi]) + pd.Timedelta(hours=6)
fig, ax = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
# windows are only kept where observations exist, so time has gaps: break the lines there instead of joining across them
def gapify(tt, v):
    tt = pd.DatetimeIndex(tt); brk = np.where(np.diff(tt.values).astype("timedelta64[h]").astype(int) > 1)[0] + 1
    return np.insert(tt.values, brk, np.datetime64("NaT")), np.insert(np.asarray(v, float), brk, np.nan)
ax[0].plot(t, m["y"][lo:hi, 1], "k.", ms=2, label="GOES-15 observed")
ax[0].plot(*gapify(t, np.repeat(m["persist"][:, None], 3, 1)[lo:hi, 1]), color="#888", lw=.8, label="Persistence")
for f in ["transformer_matched", "storm_bz"]:
    if f in fam_all: ax[0].plot(*gapify(t, SC[bb][f]["bag"][lo:hi, 1]), color=COL[f], lw=1, label=NAMES[f] + " (bagged)")
ax[0].set_ylabel(r"log$_{10}$ flux (6 h lead)"); ax[0].legend(fontsize=6.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=0.8, handlelength=1.5); ax[1].plot(*gapify(t, m["dst_target"][lo:hi, 1]), color="#444"); ax[1].set_ylabel("Dst (nT)")
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT / "figures/fig_case_study.png"); plt.close(fig)
NUM["case_study"] = dict(block=bb, min_dst=float(m["dst_target"][i0, 1]), centre_time=str(pd.to_datetime(m["t_origin"][i0])))

# ------------------------------------------------------------------------------------------------ 9. GRASP
G = {}
for d in args.inputs:
    for f in glob.glob(str(Path(d) / "**/grasp_preds_s*.npz"), recursive=True): G[int(re.search(r"_s(\d+)", f).group(1))] = dict(np.load(f))
gm = [f for d in args.inputs for f in glob.glob(str(Path(d) / "**/grasp_meta.npz"), recursive=True)]
if G and gm:
    GM = dict(np.load(gm[0])); seeds = sorted(G); folds = sorted({k.split("_")[0] for k in G[seeds[0]]})
    variants = ["zeroshot", "heads", "full", "scratch"]; fams = ["storm_bz", "transformer_matched"]; grows = []; GS = {}
    for fk in folds:
        y, pers = GM[f"{fk}_y"], GM[f"{fk}_persist"]
        pp = pe_clim(y, np.repeat(pers[:, None], 3, 1)); GS[(fk, "persistence")] = pp
        grows.append(dict(Fold=fk, Model="Persistence", Variant="--", N=len(y), **{f"PE_clim {h}h": f"{pp[i]:.3f}" for i, h in enumerate(H)}))
        for fam in fams:
            for v in variants:
                k = f"{fk}_{fam}_{v}"
                if k not in G[seeds[0]]: continue
                pc = np.stack([pe_clim(y, G[s][k]) for s in seeds]); GS[(fk, fam, v)] = pc
                grows.append(dict(Fold=fk, Model=NAMES[fam], Variant=v, N=len(y), **{f"PE_clim {h}h": fmt(pc[:, i]) for i, h in enumerate(H)}))
    save(pd.DataFrame(grows), "T8_grasp", "GRASP transfer; mean ± sd over seeds")
    # fold-average + paired tests
    arows = []; NUM["grasp"] = {}
    for fam in fams:
        for v in variants:
            if all((fk, fam, v) in GS for fk in folds):
                av = np.mean([GS[(fk, fam, v)] for fk in folds], 0)             # (S,3)
                arows.append(dict(Model=NAMES[fam], Variant=v, **{f"PE_clim {h}h": fmt(av[:, i]) for i, h in enumerate(H)})); NUM["grasp"][f"{fam}_{v}"] = av.mean(0).tolist()
    save(pd.DataFrame(arows), "T8b_grasp_fold_average", "Average of the 3 chronological folds")
    trows = []
    def trow(a, bq, nm):
        for i, h in enumerate(H):
            r = paired(a[:, i], bq[:, i]); trows.append(dict(Comparison=nm, Horizon=f"{h}h", dPE=f"{r['diff']:+.4f}", t=f"{r['t']:.2f}", p=f"{r['p']:.2e}", d_z=f"{r['dz']:.2f}", n=r["n"]))
            NUM["grasp"][f"{nm}|{h}h"] = r
    avg_ = lambda fam, v: np.mean([GS[(fk, fam, v)] for fk in folds], 0)
    for fam in fams:
        if all((fk, fam, "heads") in GS for fk in folds): trow(avg_(fam, "heads"), avg_(fam, "zeroshot"), f"{NAMES[fam]}: heads-only FT vs zero-shot")
    for v in ["zeroshot", "heads", "full", "scratch"]:
        if all((fk, f, v) in GS for fk in folds for f in fams): trow(avg_("storm_bz", v), avg_("transformer_matched", v), f"STORM-Bz vs matched TF ({v})")
    save(pd.DataFrame(trows), "T8c_grasp_paired", "Paired by seed on fold-averaged PE_clim")
    fig, ax = plt.subplots(1, 3, figsize=(7, 2.4), sharey=True)
    for i, fk in enumerate(folds):
        w = .18
        for j, fam in enumerate(fams):
            for k_, v in enumerate(variants):
                if (fk, fam, v) in GS:
                    q = GS[(fk, fam, v)][:, 1]; ax[i].bar(j * 5 * w + k_ * w, q.mean(), w, yerr=q.std(ddof=1) if len(q) > 1 else 0, color=plt.cm.viridis(k_ / 3), label=v if (i == 0 and j == 0) else None, capsize=1)
        ax[i].set_xticks([1.5 * w, 5 * w + 1.5 * w]); ax[i].set_xticklabels(["STORM-PhysNet", "TF matched"], fontsize=6); ax[i].set_title(f"{fk} (6 h)")
    ax[0].set_ylabel(r"$PE_{clim}$"); ax[0].legend(fontsize=5); fig.tight_layout(); fig.savefig(OUT / "figures/fig_grasp.png"); plt.close(fig)

# ------------------------------------------------------------------------------------------------ 10. compute budget + claim checks
if runlogs:
    rl = pd.DataFrame([r for r in runlogs if "family" in r])
    if len(rl):
        s = rl.groupby(["block", "family"]).agg(runs=("seed", "count"), mean_epochs=("epochs", "mean"), mean_sec=("seconds", "mean")).round(1).reset_index()
        save(s, "T9_compute", "Training cost per run"); NUM["compute_hours_total"] = float(rl["seconds"].sum() / 3600)
CH = {}
def ok(name, cond): CH[name] = bool(cond)
if "storm_bz" in fam_all and "transformer_matched" in fam_all:
    for i, h in enumerate(H):
        ok(f"STORM-Bz > matched TF at {h}h in every block (seed means)", all(SC[b]["storm_bz"]["pc"][:, i].mean() > SC[b]["transformer_matched"]["pc"][:, i].mean() for b in BLOCKS))
        ok(f"STORM-Bz > matched TF at {h}h, pooled paired p<0.05", PT[f"storm_bz__transformer_matched__pooled__{h}h"]["diff"] > 0 and PT[f"storm_bz__transformer_matched__pooled__{h}h"]["p"] < .05)
if "tf_ffn_res" in fam_all and "storm_bz" in fam_all:
    for i, h in enumerate(H): ok(f"STORM-Bz > residual TF at {h}h, pooled paired p<0.05", PT[f"storm_bz__tf_ffn_res__pooled__{h}h"]["diff"] > 0 and PT[f"storm_bz__tf_ffn_res__pooled__{h}h"]["p"] < .05)
if "tf_flux_only" in fam_all and "transformer_matched" in fam_all:
    for i, h in enumerate(H): ok(f"solar-wind inputs help (matched TF > flux-only TF) at {h}h, pooled p<0.05", PT[f"transformer_matched__tf_flux_only__pooled__{h}h"]["diff"] > 0 and PT[f"transformer_matched__tf_flux_only__pooled__{h}h"]["p"] < .05)
NUM["claim_checks"] = CH
(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))

md = ["# PAPER NUMBERS (auto-generated)\n", "## Claim checks (True = the data supports the sentence)\n"] + [f"- {'✅' if v else '❌'} {k}" for k, v in CH.items()]
md += ["\n## Splits\n", split_df.to_markdown(index=False), "\n## Main (seed mean ± sd)\n", main_df.to_markdown(index=False),
       "\n## Block average\n", avg_df.to_markdown(index=False), "\n## Paired tests\n", pd.DataFrame(prow).to_markdown(index=False),
       "\n## Gate / delay\n", "```\n" + json.dumps(gd, indent=1) + "\n```"]
(OUT / "PAPER_NUMBERS.md").write_text("\n".join(md))
print("report written to", OUT)
