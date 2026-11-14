#!/usr/bin/env python
"""Further analyses of the saved test predictions (no training):

  by_block    prediction efficiency per test block, i.e. per phase of solar cycle 24 (C 2012-13 near maximum,
              B 2015-16 declining phase, A 2018-20 minimum)
  daily       prediction efficiency of daily means of the hourly forecasts (comparable with daily-resolution studies)
  local_time  RMSE and skill against persistence by local time of the forecast target at GOES-15 (135 W, LT = UT - 9 h)
  rises       forecasts of rapid rises (target at least 0.5 in log10 above the flux at the origin): RMSE and the
              fraction of the observed rise that the forecast reproduces
  lookback    models trained with another input window, compared with the 72 h models on the forecast origins both
              window sets contain (each lookback has its own meta_<B>.npz)

  python extra_analyses.py --blocks DIR [DIR ...] [--lookback DIR ...] --out extra.json
"""
import argparse, json
from collections import defaultdict

import numpy as np

from journal_analysis import load_blocks, pe_clim

KEY = ["persistence", "ridge_res", "gbm_res", "lstm_res", "lstm_res_tp", "storm_lstm_backbone", "tf_ffn_res",
       "tf_ffn_res_tp", "storm_bz", "storm_bz_tp", "transformer_matched"]
LT_OFFSET_H = -9          # GOES-15 at 135 W
RISE = 0.5                # log10 units


def t_ns(m):
    """Forecast origins as int64 nanoseconds (train_block.py stores int64, classical_baselines.py datetime64)."""
    t = np.asarray(m["t_origin"])
    return t.astype("datetime64[ns]").astype(np.int64) if np.issubdtype(t.dtype, np.datetime64) else t.astype(np.int64)


def seeds_of(P, b, fam):
    return sorted(P[b][fam]) if fam in P[b] else []


def preds(meta, P, b, fam):
    """(seeds, n, H) forecasts of a family in block b; persistence is the flux at the origin for every lead time."""
    m = meta[b]
    if fam == "persistence":
        return np.repeat(m["persist"][None, :, None], len(m["horizons"]), 2)
    return np.stack([P[b][fam][s]["pred"] for s in seeds_of(P, b, fam)])


def by_block(meta, P, fams):
    out = {}
    for f in fams:
        out[f] = {b: pe_clim_seeds(meta[b]["y"], preds(meta, P, b, f)) for b in sorted(meta) if f == "persistence" or f in P[b]}
    return out


def pe_clim_seeds(y, pr):
    v = np.array([pe_clim(y, p) for p in pr])
    return dict(mean=v.mean(0).tolist(), sd=v.std(0, ddof=1).tolist() if len(v) > 1 else [0.0] * v.shape[1])


def daily(meta, P, fams, min_hours=20):
    """PE of daily means of observed and forecast flux, grouped by the UT day of the target, per block then averaged."""
    out = {}
    for f in fams:
        per_b = []
        for b in sorted(meta):
            if f != "persistence" and f not in P[b]:
                continue
            m = meta[b]; H = [int(h) for h in m["horizons"]]; pr = preds(meta, P, b, f); vals = []
            for i, h in enumerate(H):
                day = (t_ns(m) + h * 3600 * 10**9) // (86400 * 10**9)
                u, inv, cnt = np.unique(day, return_inverse=True, return_counts=True); keep = cnt >= min_hours
                yo = np.bincount(inv, m["y"][:, i]) / cnt
                vals.append([pe_clim(yo[keep][:, None], (np.bincount(inv, p[:, i]) / cnt)[keep][:, None])[0] for p in pr])
            per_b.append(np.array(vals).T)          # (seeds, H)
        v = np.mean(per_b, 0); out[f] = dict(mean=v.mean(0).tolist(), sd=v.std(0, ddof=1).tolist() if len(v) > 1 else [0.0] * v.shape[1])
    return out


def local_time(meta, P, fams, nbins=8):
    """RMSE of every family and of persistence by 3-hour bin of target local time, pooled over blocks and seeds."""
    out = {}
    for f in fams:
        se = defaultdict(lambda: np.zeros((nbins, 2)))
        for b in sorted(meta):
            if f != "persistence" and f not in P[b]:
                continue
            m = meta[b]; H = [int(h) for h in m["horizons"]]; pr = preds(meta, P, b, f)
            for i, h in enumerate(H):
                lt = ((t_ns(m) // (3600 * 10**9) + h + LT_OFFSET_H) % 24).astype(int); k = lt * nbins // 24
                for p in pr:
                    e2 = (p[:, i] - m["y"][:, i]) ** 2
                    se[h][:, 0] += np.bincount(k, e2, nbins); se[h][:, 1] += np.bincount(k, minlength=nbins)
        out[f] = {str(h): np.sqrt(a[:, 0] / a[:, 1]).tolist() for h, a in se.items()}
    return out


def rises(meta, P, fams, sign=1):
    """On targets at least RISE above (sign=1; below for sign=-1) the flux at the origin: RMSE (pooled and per seed)
    and the mean fraction of the change that the forecast reproduces."""
    out = {}
    for f in fams:
        err, frac, n = defaultdict(list), defaultdict(list), defaultdict(int)
        for b in sorted(meta):
            if f != "persistence" and f not in P[b]:
                continue
            m = meta[b]; H = [int(h) for h in m["horizons"]]; pr = preds(meta, P, b, f)
            for i, h in enumerate(H):
                d = m["y"][:, i] - m["persist"]; sel = sign * d >= RISE; n[h] += int(sel.sum())
                e2 = np.array([(p[sel, i] - m["y"][sel, i]) ** 2 for p in pr])            # (seeds, n_sel)
                fr = np.array([(p[sel, i] - m["persist"][sel]) / d[sel] for p in pr])
                err[h].append(e2); frac[h].append(fr)
        out[f] = {str(h): dict(rmse=float(np.sqrt(np.concatenate(err[h], 1).mean())),
                               rmse_seed=np.sqrt(np.concatenate(err[h], 1).mean(1)).tolist(),
                               frac=float(np.concatenate(frac[h], 1).mean()), n=n[h]) for h in n}
    return out


def lookback(meta72, P72, dirs):
    """Each lookback family against its 72 h family, on the forecast origins common to both window sets.
    Directories are grouped by the lookback in their prediction file names (each lookback has its own windows)."""
    import glob, os, re
    groups = defaultdict(list)
    for d in dirs:
        Ls = {int(m.group(1)) for f in glob.glob(f"{d}/preds_*_s*.npz") if (m := re.search(r"_L(\d+)_s\d+\.npz$", os.path.basename(f)))}
        assert len(Ls) == 1, f"{d}: expected one lookback, found {Ls}"
        groups[Ls.pop()].append(d)
    out = {}
    for L in sorted(groups):
        _lookback_group(meta72, P72, groups[L], out)
    return out


def _lookback_group(meta72, P72, dirs, out):
    import re
    meta_l, P_l = load_blocks(dirs)
    for b in sorted(meta_l):
        if b not in meta72:
            continue
        common, i72, il = np.intersect1d(t_ns(meta72[b]), t_ns(meta_l[b]), return_indices=True)
        y = meta72[b]["y"][i72]
        assert np.allclose(y, meta_l[b]["y"][il]), "targets differ on common origins"
        for fam in P_l[b]:
            base = re.sub(r"_L\d+$", "", fam); L = int(re.search(r"_L(\d+)$", fam).group(1))
            for name, pr in ((fam, [P_l[b][fam][s]["pred"][il] for s in sorted(P_l[b][fam])]),
                             (base, [P72[b][base][s]["pred"][i72] for s in sorted(P72[b].get(base, {}))])):
                if not pr:
                    continue
                key = name if name == fam else f"{base}@L{L}"      # the 72 h model on the L-hour common origins
                out.setdefault(key, {})[b] = dict(pe=np.mean([pe_clim(y, p) for p in pr], 0).tolist(),
                                                  per_seed=[pe_clim(y, p).tolist() for p in pr],
                                                  n=int(len(common)), lookback=L if name == fam else 72)


TEX_ROWS = ["persistence", "ridge_res", "gbm_res", "lstm_res", "storm_lstm_backbone", "tf_ffn_res", "tf_ffn_res_tp",
            "storm_bz", "storm_bz_tp"]


def write_tex(R, path):
    """Wide table: PE per test block at 12 and 24 h, PE of daily means at 24 h, RMSE on rises and falls at 6 and 24 h."""
    from journal_tables import LABEL
    lab = dict(LABEL, persistence="Persistence", lstm_res_tp="LSTM, residual, protocol B",
               tf_ffn_res_tp="Residual Transformer, protocol B", storm_bz_tp="STORM-PhysNet, protocol B",
               storm_lstm_backbone="STORM-PhysNet, LSTM encoder")
    f3 = lambda v: f"{v:.3f}"; f2 = lambda v: f"{v:.2f}"
    body = []
    for f in [x for x in TEX_ROWS if x in R["families"]]:
        bb = R["by_block"][f]; cells = [f3(bb[b]["mean"][2]) for b in "CBA"] + [f3(bb[b]["mean"][3]) for b in "CBA"]
        cells += [f3(R["daily"][f]["mean"][3])]
        cells += [f2(R["rises"][f][h]["rmse"]) for h in ("6", "24")] + [f2(R["falls"][f][h]["rmse"]) for h in ("6", "24")]
        body.append(lab.get(f, f).replace("\\quad ", "") + " & " + " & ".join(cells) + " \\\\")
    nr = {h: R["rises"]["persistence"][h]["n"] for h in ("6", "24")}; nf = {h: R["falls"]["persistence"][h]["n"] for h in ("6", "24")}
    tex = (r"""\begin{table*}[!t]
\centering
\caption{Skill by phase of solar cycle 24, on daily means, and on rapid changes. Prediction efficiency
\(\mathrm{PE}_{\mathrm{clim}}\) per test block (C 2012--2013, near maximum; B 2015--2016, declining phase; A
2018--2020, minimum) at 12 and 24\,h; \(\mathrm{PE}_{\mathrm{clim}}\) of daily means of the 24\,h forecasts; and RMSE
(log\(_{10}\) flux) on targets that lie at least 0.5 above (rises) or below (falls) the flux at the forecast origin,
pooled over blocks and seeds. Means over ten seeds.}
\label{tab:extra}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lccccccccccc}
\toprule
 & \multicolumn{3}{c}{PE, 12\,h} & \multicolumn{3}{c}{PE, 24\,h} & Daily & \multicolumn{2}{c}{RMSE, rises} & \multicolumn{2}{c}{RMSE, falls} \\
Model & C & B & A & C & B & A & 24\,h & 6\,h & 24\,h & 6\,h & 24\,h \\
\midrule
""" + "\n".join(body) + f"""
\\bottomrule
\\end{{tabular}}
\\end{{adjustbox}}

\\smallskip
{{\\footnotesize Rises: {nr['6']} targets at 6\\,h and {nr['24']} at 24\\,h; falls: {nf['6']} and {nf['24']}.}}
\\end{{table*}}
""")
    open(path, "w").write(tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="+", required=True); ap.add_argument("--lookback", nargs="*", default=[])
    ap.add_argument("--out", required=True); ap.add_argument("--tex", default=None)
    a = ap.parse_args()
    meta, P = load_blocks(a.blocks)
    fams = [f for f in KEY if f == "persistence" or all(f in P[b] for b in meta)]
    R = dict(families=fams, by_block=by_block(meta, P, fams), daily=daily(meta, P, fams),
             local_time=local_time(meta, P, fams), rises=rises(meta, P, fams), falls=rises(meta, P, fams, sign=-1))
    from scipy import stats
    from journal_analysis import holm
    rows = [(f, h, R["rises"]["gbm_res"][h]["rmse_seed"], R["rises"][f][h]["rmse_seed"]) for f in fams
            if f not in ("gbm_res", "persistence", "ridge_res") for h in R["rises"][f]]
    ps = [stats.ttest_rel(g, x).pvalue for _, _, g, x in rows]
    R["rises_paired_gbm"] = [dict(family=f, h=int(h), diff=float(np.mean(g) - np.mean(x)), p=float(p), p_holm=float(q))
                             for (f, h, g, x), p, q in zip(rows, ps, holm(ps))]
    if a.lookback:
        R["lookback"] = lookback(meta, P, a.lookback)
    json.dump(R, open(a.out, "w"), indent=1); print("wrote", a.out)
    if a.tex:
        write_tex(R, a.tex); print("wrote", a.tex)


if __name__ == "__main__":
    main()
