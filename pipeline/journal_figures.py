#!/usr/bin/env python
"""Figures of the journal paper.

  python journal_figures.py --repo .. --numbers journal_report/numbers.json --out figures_journal

Reads numbers.json from journal_analysis.py and the hourly data in datasets/. Writes PNG files (300 dpi).
"""
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABEL = {"storm_bz": "STORM-PhysNet", "tf_ffn_res": "Residual TF", "transformer_matched": "Matched TF", "lstm": "LSTM",
         "tf_flux_only": "Flux-only TF", "persistence": "Persistence", "tf_res_fixed_ma": "Residual TF + fixed avg.",
         "storm_fixed_ma": "STORM-PhysNet, fixed avg."}
COL = {"storm_bz": "#c0392b", "tf_ffn_res": "#2471a3", "transformer_matched": "#7d3c98", "lstm": "#229954",
       "tf_flux_only": "#b9770e", "persistence": "#555555", "tf_res_fixed_ma": "#5dade2", "storm_fixed_ma": "#e6b0aa"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})


def fig_horizon(N, out):
    H = N["horizons"]; fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(H, N["persistence"]["pe_clim"], "--", color=COL["persistence"], label=LABEL["persistence"])
    for f in ("lstm", "tf_flux_only", "transformer_matched", "tf_ffn_res", "storm_bz"):
        if f in N["main"]:
            m = np.array(N["main"][f]["pe_clim"]); s = np.array(N["main"][f]["sd"])
            ax.errorbar(H, m, yerr=s, marker="o", ms=4, capsize=2, color=COL[f], label=LABEL[f])
    ax.set_xticks(H); ax.set_xlabel("Forecast horizon (h)"); ax.set_ylabel(r"PE$_\mathrm{clim}$"); ax.set_ylim(0.2, 1.0)
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(out / "fig_horizon.png"); plt.close(fig)


def fig_noise(N, out):
    H = N["horizons"]; sig = [0, 0.5, 1, 2]
    fig, axs = plt.subplots(1, len(H), figsize=(7.2, 2.4), sharey=False)
    for i, h in enumerate(H):
        ax = axs[i]
        for f in ("storm_bz", "tf_ffn_res", "tf_res_fixed_ma", "transformer_matched"):
            if f not in N["probes"]: continue
            v = [N["main"][f]["pe_clim"][i]] + [N["probes"][f][f"noise{s:g}"][i] for s in sig[1:]]
            ax.plot(sig, v, marker="o", ms=3, color=COL[f], label=LABEL[f])
        ax.set_title(f"{h} h"); ax.set_xticks(sig); ax.set_xlabel(r"noise $\sigma$")
        if i == 0: ax.set_ylabel(r"PE$_\mathrm{clim}$")
    axs[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(out / "fig_noise.png"); plt.close(fig)


def fig_transfer(N, out):
    T = N["transfer"]; H = N["horizons"]; tg = [t for t in ("goes16", "goes18", "grasp") if t in T]
    name = {"goes16": "GOES-16", "goes18": "GOES-18", "grasp": "GSAT-19 GRASP"}
    i12 = H.index(12)
    fig, axs = plt.subplots(1, len(tg), figsize=(7.2, 2.6), sharey=True)
    fams = ("storm_bz", "tf_ffn_res", "transformer_matched"); var = ("zeroshot", "heads", "full", "scratch")
    for ax, t in zip(np.atleast_1d(axs), tg):
        x = np.arange(len(var))
        for k, f in enumerate(fams):
            if f in T[t]: ax.bar(x + (k - 1) * 0.26, [T[t][f][v][i12] for v in var], 0.26, color=COL[f], label=LABEL[f])
        ax.axhline(T[t]["persistence"]["-"][i12], ls="--", color=COL["persistence"], lw=1)
        ax.set_xticks(x); ax.set_xticklabels(["zero-shot", "heads", "full", "scratch"], rotation=20); ax.set_title(name[t])
    np.atleast_1d(axs)[0].set_ylabel(r"12 h PE$_\mathrm{clim}$"); np.atleast_1d(axs)[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_transfer.png"); plt.close(fig)


def fig_coverage_and_overlap(repo, out):
    sys.path.insert(0, str(repo))
    from src.data.cdf_reader import read_goes_directory, read_grasp_directory
    g15 = read_goes_directory(str(repo / "datasets/goes"))["flux_gt2mev"]
    g16 = pd.read_csv(repo / "datasets/goesr/goes16_e2mev_hourly.csv", index_col=0, parse_dates=True)["flux_gt2mev"]
    g18 = pd.read_csv(repo / "datasets/goesr/goes18_e2mev_hourly.csv", index_col=0, parse_dates=True)["flux_gt2mev"]
    gr = read_grasp_directory(str(repo / "datasets/grasp"))["flux_gt2mev"]
    # data coverage with test periods
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    rows = [("GOES-15 (135° W)", g15), ("GOES-16 (75° W)", g16), ("GOES-18 (137° W)", g18), ("GSAT-19 GRASP (48° E)", gr)]
    for i, (nm, s) in enumerate(rows):
        d = s.dropna().resample("1D").size(); d = d[d > 0]
        ax.scatter(d.index, np.full(len(d), i), s=1, color="#888888")
    tests = [(0, "2018-09-07", "2020-03-04", "A"), (0, "2015-01-01", "2016-12-31", "B"), (0, "2012-01-01", "2013-12-31", "C"),
             (1, "2022-07-01", "2023-06-30", "F1"), (1, "2024-01-01", "2025-04-07", "F2"), (2, "2025-07-01", "2026-10-14", "F1"),
             (3, "2018-02-01", "2018-09-01", "G1–3")]
    for i, a, b, lab in tests:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), ymin=(i + 0.1) / 4, ymax=(i + 0.9) / 4, color="#c0392b", alpha=0.25)
        ax.text(pd.Timestamp(a), i + 0.28, lab, fontsize=6, color="#922b21")
    ax.axvspan(pd.Timestamp("2017-03-11"), pd.Timestamp("2018-09-07"), ymin=0.02, ymax=0.23, color="#2471a3", alpha=0.2)
    ax.set_yticks(range(4)); ax.set_yticklabels([r[0] for r in rows]); ax.set_ylim(-0.5, 3.5)
    fig.tight_layout(); fig.savefig(out / "fig_data.png"); plt.close(fig)
    # GOES-15 vs GOES-16 overlap
    j = pd.concat([np.log10(g15.clip(lower=1e-6)), np.log10(g16)], axis=1, keys=["g15", "g16"]).dropna()
    d = j.resample("1D").mean().dropna()
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.scatter(d.g15, d.g16, s=3, alpha=0.5, color="#2471a3")
    lo, hi = min(d.min()), max(d.max()); ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel(r"GOES-15 log$_{10}$ flux"); ax.set_ylabel(r"GOES-16 log$_{10}$ flux")
    fig.tight_layout(); fig.savefig(out / "fig_g15_g16.png"); plt.close(fig)
    return {"overlap_hours": int(len(j)), "overlap_start": str(j.index.min()), "overlap_end": str(j.index.max()),
            "median_log_ratio": float((j.g16 - j.g15).median()), "daily_corr": float(d.corr().iloc[0, 1]),
            "hourly_corr": float(j.corr().iloc[0, 1]), "g16_hours": int(g16.notna().sum()), "g18_hours": int(g18.notna().sum()),
            "g16_range": [str(g16.index.min()), str(g16.index.max())], "g18_range": [str(g18.index.min()), str(g18.index.max())]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); ap.add_argument("--numbers", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); N = json.load(open(a.numbers))
    fig_horizon(N, out); fig_noise(N, out); fig_transfer(N, out)
    ov = fig_coverage_and_overlap(Path(a.repo).resolve(), out)
    json.dump(ov, open(out / "overlap.json", "w"), indent=1); print(json.dumps(ov, indent=1))


if __name__ == "__main__":
    main()
