#!/usr/bin/env python
"""Figures of the journal paper.

  python journal_figures.py --repo .. --numbers journal_report/numbers.json --blocks <block dirs...> \
                            --transfer <transfer dirs...> --out figures

Reads numbers.json (journal_analysis.py), the saved predictions and the hourly data in datasets/.
Writes 300-dpi PNGs sized for a two-column Elsevier page (single column 8.8 cm, double 18.3 cm) and overlap.json.
"""
import argparse, glob, json, os, re, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from journal_analysis import load_blocks, pe_clim  # noqa: E402

COL1, COL2 = 3.46, 7.2                      # inches: one column, full width
LABEL = {"storm_bz": "STORM-PhysNet", "storm_lstm_backbone": "STORM-PhysNet (LSTM enc.)", "tf_ffn_res": "Residual TF",
         "transformer_matched": "Matched TF", "transformer": "Default TF", "tf_ffn256": "TF, FFN 256", "lstm": "LSTM",
         "mlp": "MLP", "cnn": "CNN", "tf_flux_only": "Flux-only TF", "persistence": "Persistence",
         "tf_res_fixed_ma": "Residual TF + fixed avg.", "storm_fixed_ma": "fixed average", "storm_no_delay": "no delay",
         "storm_no_gate": "no gate", "storm_no_physics": "no physics loss", "storm_cathode": "cathode-anode gate",
         "storm_cathode_spec": "cathode-anode + spectral", "storm_radiotrophic": "radiotrophic gate",
         "storm_magnetopause": "+ magnetopause"}
COL = {  # Okabe-Ito colour-blind-safe palette (line styles also differ where colours repeat)
    "storm_bz": "#D55E00", "storm_lstm_backbone": "#E69F00", "tf_ffn_res": "#0072B2", "transformer_matched": "#CC79A7",
    "lstm": "#009E73", "mlp": "#56B4E9", "tf_flux_only": "#F0E442", "persistence": "#555555", "tf_res_fixed_ma": "#0072B2",
    "transformer": "#CC79A7", "cnn": "#999999", "tf_ffn256": "#56B4E9"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, out, name):
    """Vector PDF (for the manuscript and the submission system) plus a 300-dpi PNG preview."""
    fig.savefig(out / name); fig.savefig(out / name.replace(".png", ".pdf")); plt.close(fig)


def gaps(t, *ys, max_gap="1h"):
    """Insert a NaN point after every jump longer than max_gap, so plotted lines break across data gaps."""
    t = pd.DatetimeIndex(t); cut = set((np.where(np.diff(t.values) > np.timedelta64(pd.Timedelta(max_gap)))[0]).tolist())
    tt, out = [], [[] for _ in ys]
    for j in range(len(t)):
        tt.append(t[j]); [o.append(y[j]) for o, y in zip(out, ys)]
        if j in cut:
            tt.append(t[j] + pd.Timedelta("1min")); [o.append(np.nan) for o in out]
    return pd.DatetimeIndex(tt), [np.asarray(o, float) for o in out]


# ------------------------------------------------------------------------------------------------ skill vs lead time
def fig_horizon(N, out):
    H = N["horizons"]; fig, ax = plt.subplots(figsize=(COL1, 2.6))
    ax.plot(H, N["persistence"]["pe_clim"], "--", color=COL["persistence"], label=LABEL["persistence"])
    for f in ("cnn", "lstm", "mlp", "transformer_matched", "tf_ffn_res", "storm_bz", "storm_lstm_backbone"):
        if f in N["main"]:
            m = np.array(N["main"][f]["pe_clim"]); s = np.array(N["main"][f]["sd"])
            ax.errorbar(H, m, yerr=s, marker="o", ms=3, capsize=2, lw=1.2, color=COL[f], label=LABEL[f])
    ax.set_xticks(H); ax.set_xlabel("Lead time (h)"); ax.set_ylabel(r"PE$_\mathrm{clim}$"); ax.set_ylim(0.2, 1.0)
    ax.legend(frameon=False, ncol=2, loc="lower left"); save(fig, out, "fig_horizon.png")


# ------------------------------------------------------------------------------------------------ attribution forest plot
def fig_attribution(N, out, P, meta):
    H = N["horizons"]; blocks = sorted(P)
    fams = ["storm_no_delay", "storm_fixed_ma", "storm_no_gate", "storm_no_physics", "storm_cathode", "storm_cathode_spec",
            "storm_radiotrophic", "storm_magnetopause", "storm_lstm_backbone", "tf_ffn_res"]
    fams = [f for f in fams if f in N["main"]]
    seeds = sorted(P[blocks[0]]["storm_bz"])
    ref = np.array([np.mean([pe_clim(meta[b]["y"], P[b]["storm_bz"][s]["pred"]) for b in blocks], 0) for s in seeds])
    fig, axs = plt.subplots(1, len(H), figsize=(COL2, 2.9), sharey=True)
    for i, h in enumerate(H):
        ax = axs[i]
        for k, f in enumerate(fams):
            v = np.array([np.mean([pe_clim(meta[b]["y"], P[b][f][s]["pred"]) for b in blocks], 0) for s in seeds])[:, i] - ref[:, i]
            m = v.mean(); ci = 2.262 * v.std(ddof=1) / np.sqrt(len(v))              # t(0.975, 9)
            c = COL["storm_lstm_backbone"] if f == "storm_lstm_backbone" else (COL["tf_ffn_res"] if f == "tf_ffn_res" else "#444444")
            ax.errorbar(m, k, xerr=ci, fmt="o", ms=3, color=c, capsize=2)
        ax.axvline(0, color="k", lw=0.6); ax.set_title(f"{h} h")
    axs[0].set_yticks(range(len(fams))); axs[0].set_yticklabels([LABEL[f] if f in ("storm_lstm_backbone", "tf_ffn_res") else "STORM, " + LABEL[f] for f in fams])
    axs[0].invert_yaxis(); fig.supxlabel(r"$\Delta\mathrm{PE}_\mathrm{clim}$ relative to STORM-PhysNet (mean and 95% CI over ten seeds)", fontsize=8)
    fig.tight_layout(); save(fig, out, "fig_attribution.png")


# ------------------------------------------------------------------------------------------------ noise robustness
def fig_noise(N, out):
    H = N["horizons"]; sig = [0, 0.5, 1, 2]
    fig, axs = plt.subplots(1, len(H), figsize=(COL2, 2.2))
    for i, h in enumerate(H):
        ax = axs[i]
        for f in ("storm_bz", "storm_lstm_backbone", "tf_ffn_res", "tf_res_fixed_ma", "transformer_matched", "mlp"):
            if f not in N["probes"]: continue
            v = [N["main"][f]["pe_clim"][i]] + [N["probes"][f][f"noise{s:g}"][i] for s in sig[1:]]
            ax.plot(sig, v, marker="o", ms=2.5, lw=1.1, color=COL[f], label=LABEL[f], ls="--" if f == "tf_res_fixed_ma" else "-")
        ax.set_title(f"{h} h"); ax.set_xticks(sig); ax.set_xlabel(r"noise $\sigma$")
        if i == 0: ax.set_ylabel(r"PE$_\mathrm{clim}$")
        ax.set_ylim(max(-0.2, ax.get_ylim()[0]), 1.0)
    axs[0].legend(frameon=False, fontsize=6); fig.tight_layout(); save(fig, out, "fig_noise.png")


# ------------------------------------------------------------------------------------------------ predicted vs observed
def fig_scatter(P, meta, out, h_idx=2, block="B"):
    fams = [("persistence", None), ("tf_ffn_res", "tf_ffn_res"), ("storm_lstm_backbone", "storm_lstm_backbone")]
    y = meta[block]["y"][:, h_idx]
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.4), sharex=True, sharey=True)
    for ax, (name, f) in zip(axs, fams):
        p = meta[block]["persist"] if f is None else np.mean([P[block][f][s]["pred"][:, h_idx] for s in P[block][f]], 0)
        hb = ax.hexbin(y, p, gridsize=45, bins="log", cmap="cividis", mincnt=1)
        lo, hi = 0.8, 5.4; ax.plot([lo, hi], [lo, hi], "w--", lw=0.7)
        pe = 1 - np.mean((y - p) ** 2) / np.var(y); r = np.corrcoef(y, p)[0, 1]
        ax.set_title(f"{LABEL[name]}\nPE = {pe:.3f}, r = {r:.3f}"); ax.set_xlabel(r"observed log$_{10}$ flux")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    axs[0].set_ylabel(r"forecast log$_{10}$ flux (12 h)")
    fig.colorbar(hb, ax=axs, shrink=0.8, label="windows"); save(fig, out, "fig_scatter.png")


# ------------------------------------------------------------------------------------------------ storm case study, GOES-15
def fig_case_goes15(P, meta, out, block="B", start="2015-03-14", end="2015-04-01"):
    t = pd.to_datetime(meta[block]["t_origin"]); H = [int(h) for h in meta[block]["horizons"]]
    fig, axs = plt.subplots(2, 1, figsize=(COL2, 3.6), sharex=True)
    for ax, h in zip(axs, (12, 24)):
        i = H.index(h); tv = t + pd.Timedelta(hours=h); k = (tv >= start) & (tv <= end)
        series = [meta[block]["y"][k, i], meta[block]["persist"][k]] + \
                 [np.mean([P[block][f][s]["pred"][:, i] for s in P[block][f]], 0)[k] for f in ("tf_ffn_res", "storm_bz", "storm_lstm_backbone")]
        tt, ys = gaps(tv[k], *series)
        ax.plot(tt, ys[0], color="k", lw=1.2, label="observed")
        ax.plot(tt, ys[1], color=COL["persistence"], lw=0.8, ls="--", label="persistence")
        for f, y in zip(("tf_ffn_res", "storm_bz", "storm_lstm_backbone"), ys[2:]):
            ax.plot(tt, y, color=COL[f], lw=1.0, label=LABEL[f])
        ax.axhline(3.0, color="#999999", lw=0.6, ls=":"); ax.set_ylabel(r"log$_{10}$ flux")
        ax.text(0.01, 0.92, f"{h} h ahead", transform=ax.transAxes)
    axs[0].legend(frameon=False, ncol=5, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    axs[1].xaxis.set_major_locator(mdates.DayLocator(interval=3)); axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axs[1].set_xlabel("2015 (valid time)"); fig.tight_layout(); save(fig, out, "fig_case_2015.png")


# ------------------------------------------------------------------------------------------------ storm case study, GOES-16 (May 2024)
def fig_case_goes16(tdirs, out, start="2024-05-08", end="2024-05-25", h=12):
    meta, preds = None, {}
    for d in tdirs:
        for mf in glob.glob(f"{d}/transfer_meta*.npz"):
            m = np.load(mf)
            if "goes16_F2_y" in m.files and meta is None: meta = m
            tag = re.search(r"transfer_meta(.*)\.npz", os.path.basename(mf)).group(1)
            for pf in glob.glob(f"{d}/transfer_preds{tag}_s*.npz"):
                z = np.load(pf)
                for key in z.files:
                    if key.startswith("goes16_F2_"): preds.setdefault(key[len("goes16_F2_"):], []).append(z[key])
    if meta is None: return
    i = [int(x) for x in meta["horizons"]].index(h)
    t = pd.to_datetime(meta["goes16_F2_t"]) + pd.Timedelta(hours=h); k = (t >= start) & (t <= end)
    fig, ax = plt.subplots(figsize=(COL2, 2.4))
    curves = [("obs", "GOES-16 observed", "k", "-", 1.2, meta["goes16_F2_y"][k, i]),
              ("pers", "persistence", COL["persistence"], "--", 0.8, meta["goes16_F2_persist"][k])]
    for key, lab, c, ls in (("transformer_matched_zeroshot", "Matched TF, zero-shot", COL["transformer_matched"], ":"),
                            ("storm_lstm_backbone_zeroshot", "STORM-PhysNet (LSTM enc.), zero-shot", COL["storm_lstm_backbone"], "-"),
                            ("tf_ffn_res_full", "Residual TF, fine-tuned", COL["tf_ffn_res"], "-")):
        if key in preds: curves.append((key, lab, c, ls, 1.0, np.mean(preds[key], 0)[k, i]))
    tt, ys = gaps(t[k], *[c[-1] for c in curves])
    for (key, lab, c, ls, lw, _), y in zip(curves, ys): ax.plot(tt, y, color=c, ls=ls, lw=lw, label=lab)
    ax.axhline(3.0, color="#999999", lw=0.6, ls=":"); ax.set_ylabel(r"log$_{10}$ flux (12 h ahead)"); ax.set_xlabel("2024 (valid time)")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(frameon=False, ncol=3, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); save(fig, out, "fig_case_2024.png")


# ------------------------------------------------------------------------------------------------ calibration
def fig_calibration(P, meta, out):
    levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    from scipy.stats import norm
    z = norm.ppf(0.5 + levels / 2); H = [int(h) for h in meta[sorted(meta)[0]]["horizons"]]
    fig, axs = plt.subplots(1, 2, figsize=(COL2 * 0.62, 2.4), sharey=True)
    for ax, f in zip(axs, ("storm_bz", "storm_lstm_backbone")):
        for i, h in enumerate(H):
            cov = []
            for b in sorted(P):
                y = meta[b]["y"][:, i]
                for s in P[b][f]:
                    mu = P[b][f][s]["pred"][:, i]; sg = np.exp(0.5 * P[b][f][s]["logvar"][:, i])
                    cov.append([(np.abs(y - mu) <= zz * sg).mean() for zz in z])
            ax.plot(levels, np.mean(cov, 0), marker="o", ms=2.5, label=f"{h} h")
        ax.plot([0, 1], [0, 1], "k--", lw=0.6); ax.set_title(LABEL[f]); ax.set_xlabel("nominal coverage")
    axs[0].set_ylabel("observed coverage"); axs[0].legend(frameon=False)
    fig.tight_layout(); save(fig, out, "fig_calibration.png")


# ------------------------------------------------------------------------------------------------ transfer
def fig_transfer(N, out):
    T = N["transfer"]; H = N["horizons"]; tg = [t for t in ("goes16", "goes18", "grasp") if t in T]
    name = {"goes16": "GOES-16 (75° W)", "goes18": "GOES-18 (137° W)", "grasp": "GSAT-19 GRASP (48° E)"}
    i12 = H.index(12)
    fams = [f for f in ("storm_bz", "storm_lstm_backbone", "tf_ffn_res", "transformer_matched") if f in T[tg[0]]]
    var = ("zeroshot", "heads", "full"); w = 0.8 / len(fams)
    fig, axs = plt.subplots(1, len(tg), figsize=(COL2, 2.4), sharey=True)
    for ax, t in zip(np.atleast_1d(axs), tg):
        x = np.arange(len(var))
        for k, f in enumerate(fams):
            if f in T[t]:
                ax.bar(x + (k - (len(fams) - 1) / 2) * w, [T[t][f].get(v, [np.nan] * len(H))[i12] for v in var], w,
                       color=COL[f], label=LABEL[f])
        ax.axhline(T[t]["persistence"]["-"][i12], ls="--", color=COL["persistence"], lw=1)
        ax.set_xticks(x); ax.set_xticklabels(["zero-shot", "heads", "full"]); ax.set_title(name[t]); ax.set_ylim(-0.1, 0.9)
    np.atleast_1d(axs)[0].set_ylabel(r"12 h PE$_\mathrm{clim}$"); np.atleast_1d(axs)[0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout(); save(fig, out, "fig_transfer.png")


# ------------------------------------------------------------------------------------------------ data coverage and GOES-15 vs GOES-16
def fig_coverage_and_overlap(repo, out):
    sys.path.insert(0, str(repo))
    from src.data.cdf_reader import read_goes_directory, read_grasp_directory
    g15 = read_goes_directory(str(repo / "datasets/goes"))["flux_gt2mev"]
    g16 = pd.read_csv(repo / "datasets/goesr/goes16_e2mev_hourly.csv", index_col=0, parse_dates=True)["flux_gt2mev"]
    g18 = pd.read_csv(repo / "datasets/goesr/goes18_e2mev_hourly.csv", index_col=0, parse_dates=True)["flux_gt2mev"]
    gr = read_grasp_directory(str(repo / "datasets/grasp"))["flux_gt2mev"]
    fig, ax = plt.subplots(figsize=(COL2, 1.9))
    rows = [("GOES-15 (135° W)", g15), ("GOES-16 (75° W)", g16), ("GOES-18 (137° W)", g18), ("GSAT-19 GRASP (48° E)", gr)]
    for i, (nm, s) in enumerate(rows):
        d = s.dropna().resample("1D").size(); d = d[d > 0]
        ax.scatter(d.index, np.full(len(d), i), s=0.6, color="#888888")
    tests = [(0, "2018-09-07", "2020-03-04", "A"), (0, "2015-01-01", "2016-12-31", "B"), (0, "2012-01-01", "2013-12-31", "C"),
             (1, "2022-07-01", "2023-06-30", "F1"), (1, "2024-01-01", "2025-04-07", "F2"), (2, "2025-07-01", "2026-10-14", "F1"),
             (3, "2018-02-01", "2018-09-01", "G1–3")]
    for i, a, b, lab in tests:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), ymin=(i + 0.1) / 4, ymax=(i + 0.9) / 4, color="#D55E00", alpha=0.25)
        ax.text(pd.Timestamp(a), i + 0.28, lab, fontsize=6, color="#8a3b00")
    ax.axvspan(pd.Timestamp("2017-03-11"), pd.Timestamp("2018-09-07"), ymin=0.02, ymax=0.23, color="#0072B2", alpha=0.2)
    ax.set_yticks(range(4)); ax.set_yticklabels([r[0] for r in rows]); ax.set_ylim(-0.5, 3.5)
    save(fig, out, "fig_data.png")
    j = pd.concat([np.log10(g15.clip(lower=1e-6)), np.log10(g16)], axis=1, keys=["g15", "g16"]).dropna()
    d = j.resample("1D").mean().dropna()
    fig, ax = plt.subplots(figsize=(COL1, 2.8))
    ax.scatter(d.g15, d.g16, s=3, alpha=0.5, color="#0072B2")
    lo, hi = min(d.min()), max(d.max()); ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel(r"GOES-15 daily mean log$_{10}$ flux"); ax.set_ylabel(r"GOES-16 daily mean log$_{10}$ flux")
    save(fig, out, "fig_g15_g16.png")
    return {"overlap_hours": int(len(j)), "overlap_start": str(j.index.min()), "overlap_end": str(j.index.max()),
            "median_log_ratio": float((j.g16 - j.g15).median()), "daily_corr": float(d.corr().iloc[0, 1]),
            "hourly_corr": float(j.corr().iloc[0, 1]), "g16_hours": int(g16.notna().sum()), "g18_hours": int(g18.notna().sum()),
            "g16_range": [str(g16.index.min()), str(g16.index.max())], "g18_range": [str(g18.index.min()), str(g18.index.max())]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); ap.add_argument("--numbers", required=True)
    ap.add_argument("--blocks", nargs="+", required=True); ap.add_argument("--transfer", nargs="*", default=[])
    ap.add_argument("--out", required=True); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); N = json.load(open(a.numbers))
    meta, P = load_blocks(a.blocks)
    fig_horizon(N, out); fig_noise(N, out); fig_transfer(N, out); fig_attribution(N, out, P, meta)
    fig_scatter(P, meta, out); fig_case_goes15(P, meta, out); fig_calibration(P, meta, out); fig_case_goes16(a.transfer, out)
    ov = fig_coverage_and_overlap(Path(a.repo).resolve(), out)
    json.dump(ov, open(out / "overlap.json", "w"), indent=1); print(json.dumps(ov, indent=1))


if __name__ == "__main__":
    main()
