#!/usr/bin/env python
"""Stricter paired tests for the AGERS 2026 camera-ready (Tables III and IV), answering the reviewer point that the
30 seed-block pairs are not independent (seeds 42-51 recur in every block).

  python camera_ready_stats.py --inputs preds_blockA_and_GRASP preds_blockB preds_blockC

For each comparison and horizon: mean difference over 30 seed-block pairs; paired t-test on the 10 seed means
(each averaged over blocks), Holm-corrected over all comparisons; per-block t-tests, Bonferroni x3; and, for
STORM-PhysNet - matched Transformer, a mixed model (block fixed, seed random)."""
import argparse, glob, os
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

ap = argparse.ArgumentParser(); ap.add_argument("--inputs", nargs="+", required=True); args = ap.parse_args()
SEEDS, BLOCKS = range(42, 52), "ABC"
PAIRS = [("storm_bz", "transformer_matched"), ("storm_bz", "tf_ffn_res"), ("tf_ffn_res", "transformer_matched"),
         ("transformer_matched", "tf_flux_only"), ("storm_bz", "storm_no_delay"), ("storm_bz", "storm_no_gate"),
         ("storm_bz", "lstm")]


def find(pattern):
    hits = [h for d in args.inputs for h in glob.glob(os.path.join(d, "**", pattern), recursive=True)]
    assert hits, pattern; return hits[0]


def pe(b, fam, s):
    y = np.load(find(f"meta_{b}.npz"))["y"]; p = np.load(find(f"preds_{b}_{fam}_s{s}.npz"))["pred"]
    return 1 - ((p - y) ** 2).mean(0) / y.var(0)


rows, ps = [], []
for a, c in PAIRS:
    d = np.array([[pe(b, a, s) - pe(b, c, s) for s in SEEDS] for b in BLOCKS])     # (block, seed, horizon)
    for h, hn in enumerate(["1h", "6h", "12h"]):
        sm = d[:, :, h].mean(0); p = stats.ttest_1samp(sm, 0).pvalue; ps.append(p)
        per = [stats.ttest_1samp(d[i, :, h], 0).pvalue for i in range(3)]
        k = sum(min(1, 3 * q) < 0.05 and np.sign(d[i, :, h].mean()) == np.sign(sm.mean()) for i, q in enumerate(per))
        rows.append(dict(comparison=f"{a} - {c}", horizon=hn, dPE=d[:, :, h].mean(),
                         p_pooled30=stats.ttest_1samp(d[:, :, h].ravel(), 0).pvalue, p_seedmeans10=p,
                         blocks_sig_bonferroni=k, **{f"dPE_{b}": d[i, :, h].mean() for i, b in enumerate(BLOCKS)},
                         **{f"p_{b}_bonf": min(1, 3 * q) for b, q in zip(BLOCKS, per)}))
        if (a, c) == PAIRS[0] and h == 0:
            df = pd.DataFrame([dict(block=b, seed=s, d=d[i, j, 0]) for i, b in enumerate(BLOCKS) for j, s in enumerate(SEEDS)])
            r = smf.mixedlm("d ~ 1 + C(block, Sum)", df, groups=df["seed"]).fit(reml=True)
            print("mixed model, STORM - matched at 1 h:", round(r.params["Intercept"], 4), r.conf_int().loc["Intercept"].round(4).tolist())
out = pd.DataFrame(rows); out["p_seedmeans10_holm"] = multipletests(ps, method="holm")[1]
pd.set_option("display.width", 250); print(out.round(5).to_string(index=False))
out.to_csv("camera_ready_paired_tests.csv", index=False)
