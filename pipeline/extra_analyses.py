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
  window24    the 24 h-window models scored like the main models (storm RMSE, alert HSS, every degraded-input probe,
              calibration) and in transfer, against their 72 h versions and the trees on common origins

  python extra_analyses.py --blocks DIR [DIR ...] [--lookback DIR ...] --out extra.json
"""
import argparse, json, re
from collections import defaultdict

import numpy as np

from journal_analysis import ALERT, crps_ensemble, crps_gauss, hss, holm, load_blocks, pe_clim

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


def lookback_tests(LB, against=(("tf_ffn_res_tp_L24", "gbm_res"), ("tf_ffn_res_tp_L24", "gbm_res_L24"))):
    """Paired t-tests on per-seed scores averaged over blocks, Holm-adjusted over all rows: each lookback family
    minus its 72 h family on the same origins, then each (family, reference) pair in `against`, where the reference
    is the 72 h model on that family's origins (or, if it names a window, that model). Seeds are paired in sorted
    order. Neural families only: ridge is deterministic and the tree ensembles barely vary over seeds, so a t-test
    on them measures nothing; their window effects are reported as differences."""
    from scipy import stats
    from journal_analysis import holm

    def seeds(key):
        return np.mean([LB[key][b]["per_seed"] for b in sorted(LB[key])], 0)   # seeds x horizons

    pairs = [(k, f"{re.sub(r'_L(\d+)$', '', k)}@L{re.search(r'_L(\d+)$', k).group(1)}")
             for k in sorted(LB) if "@" not in k and not k.startswith(("ridge", "gbm"))]
    pairs += [(f, ref if re.search(r"_L\d+$", ref) else f"{ref}@L{re.search(r'_L(\d+)$', f).group(1)}")
              for f, ref in against if f in LB]
    rows, ps = [], []
    for a, b in pairs:
        if b not in LB:
            continue
        x, y = seeds(a), seeds(b)
        assert x.shape == y.shape, f"{a} vs {b}: seed counts differ"
        for i in range(x.shape[1]):
            p = stats.ttest_rel(x[:, i], y[:, i]).pvalue
            rows.append((a, b, i, float((x[:, i] - y[:, i]).mean()), float(p))); ps.append(p)
    return [dict(family=a, ref=b, h_index=i, diff=d, p=p, p_holm=float(q))
            for (a, b, i, d, p), q in zip(rows, holm(ps))]


W24 = {"tf_ffn_res_tp_L24": "tf_ffn_res_tp", "storm_bz_tp_L24": "storm_bz_tp", "gbm_res_L24": "gbm_res"}
W24_ALSO = ["lstm_res", "storm_lstm_backbone"]
PROBES = ["noise0.5", "noise1", "noise2", "out6", "out24", "out72", "lag3"]
# comparisons of the 24 h networks, Holm-adjusted together (planned when campaign 8 was set up, after campaign 7)
W24_PAIRS = [("tf_ffn_res_tp_L24", "gbm_res"), ("tf_ffn_res_tp_L24", "gbm_res_L24"), ("tf_ffn_res_tp_L24", "tf_ffn_res_tp"),
             ("storm_bz_tp_L24", "storm_bz_tp"), ("storm_bz_tp_L24", "tf_ffn_res_tp_L24"), ("storm_bz_tp_L24", "gbm_res")]
W24_METRICS = ["pe", "storm_rmse", "hss", "noise1", "noise2", "out6", "out72"]   # out24 = out72 for a 24 h window


def _scores(y, dst, p):
    """PE, storm-time RMSE (Dst <= -50 nT at the target) and alert HSS of one forecast array, per lead time."""
    H = y.shape[1]; srm = []
    for i in range(H):
        k = dst[:, i] <= -50
        srm.append(np.sqrt(((y[k, i] - p[k, i]) ** 2).mean()) if k.any() else np.nan)
    return dict(pe=pe_clim(y, p), storm_rmse=np.array(srm),
                hss=np.array([hss(y[:, i] >= ALERT, p[:, i] >= ALERT) for i in range(H)]))


def window24(meta72, P72, dirs):
    """Score the 24 h-window models and their 72 h counterparts on the forecast origins both window sets contain.
    Returns per-seed arrays (seeds x lead times, averaged over blocks) for every metric, the seed means, and paired
    t-tests (W24_PAIRS x W24_METRICS x lead times, Holm-adjusted as one family). Lower is better for storm_rmse."""
    from scipy import stats
    meta_l, P_l = load_blocks(dirs)
    blocks = sorted(b for b in meta_l if b in meta72)
    idx = {}
    for b in blocks:
        _, i72, il = np.intersect1d(t_ns(meta72[b]), t_ns(meta_l[b]), return_indices=True)
        assert np.allclose(meta72[b]["y"][i72], meta_l[b]["y"][il]), "targets differ on common origins"
        idx[b] = (i72, il)
    src = lambda f: (P_l, 1) if f in P_l[blocks[0]] else (P72, 0)
    fams = [f for f in dict.fromkeys(list(W24) + list(W24.values()) + W24_ALSO) if all(f in src(f)[0][b] for b in blocks)]
    S = {}
    for f in fams:
        P, j = src(f); seeds = sorted(P[blocks[0]][f]); files = P[blocks[0]][f][seeds[0]].files
        per = defaultdict(list)
        for s in seeds:
            acc = defaultdict(list)
            for b in blocks:
                i = idx[b][j]; m = {k: meta72[b][k][idx[b][0]] for k in ("y", "dst_target")}
                for k, v in _scores(m["y"], m["dst_target"], P[b][f][s]["pred"][i]).items():
                    acc[k].append(v)
                for pk in PROBES:
                    if f"pred_{pk}" in files:
                        acc[pk].append(pe_clim(m["y"], P[b][f][s][f"pred_{pk}"][i]))
                if "logvar" in files:
                    mu = P[b][f][s]["pred"][i]; sg = np.exp(0.5 * P[b][f][s]["logvar"][i]); e = np.abs(m["y"] - mu)
                    acc["cov68"].append((e <= sg).mean(0)); acc["cov95"].append((e <= 1.96 * sg).mean(0))
                    acc["crps"].append(crps_gauss(m["y"], mu, sg).mean(0))
            for k, v in acc.items():
                per[k].append(np.mean(v, 0))
        S[f] = {k: np.array(v) for k, v in per.items()}
    ens = {}
    for f in fams:      # the ten seeds as an ensemble: PE of the mean and empirical CRPS
        P, j = src(f); pe_e, cr_e = [], []
        for b in blocks:
            E = np.stack([P[b][f][s]["pred"][idx[b][j]] for s in sorted(P[b][f])]); y = meta72[b]["y"][idx[b][0]]
            pe_e.append(pe_clim(y, E.mean(0))); cr_e.append(crps_ensemble(y, E).mean(0))
        ens[f] = dict(pe=np.mean(pe_e, 0).tolist(), crps=np.mean(cr_e, 0).tolist())
    pers = defaultdict(list)
    for b in blocks:
        m = meta72[b]; i = idx[b][0]; y = m["y"][i]
        for k, v in _scores(y, m["dst_target"][i], np.repeat(m["persist"][i][:, None], y.shape[1], 1)).items():
            pers[k].append(v)
    rows, ps = [], []
    for a, r in W24_PAIRS:
        if a not in S or r not in S:
            continue
        for k in W24_METRICS:
            if k not in S[a] or k not in S[r]:
                continue
            x, z = S[a][k], S[r][k]
            assert x.shape == z.shape, f"{a} vs {r} ({k}): seed counts differ"
            for h in range(x.shape[1]):
                p = stats.ttest_rel(x[:, h], z[:, h]).pvalue
                rows.append((a, r, k, h, float((x[:, h] - z[:, h]).mean()), float(p))); ps.append(p)
    tests = [dict(family=a, ref=r, metric=k, h_index=h, diff=d, p=p, p_holm=float(q))
             for (a, r, k, h, d, p), q in zip(rows, holm(ps))]
    return dict(blocks=blocks, n={b: int(len(idx[b][0])) for b in blocks},
                mean={f: {k: v.mean(0).tolist() for k, v in S[f].items()} for f in S},
                sd={f: {k: v.std(0, ddof=1).tolist() for k, v in S[f].items()} for f in S},
                persistence={k: np.mean(v, 0).tolist() for k, v in pers.items()}, ensemble=ens, tests=tests)


def window24_transfer(dirs72, dirs24):
    """Transfer PE of the 24 h models and of their 72 h references on the target origins both contain, per target
    satellite (mean over folds and seeds), with per-seed values for paired tests."""
    import glob, os
    from scipy import stats

    def load(dirs):
        out = {}
        for d in dirs:
            for mf in glob.glob(f"{d}/transfer_meta*.npz"):
                tag = re.search(r"transfer_meta(.*)\.npz", os.path.basename(mf)).group(1)
                files = sorted(glob.glob(f"{d}/transfer_preds{tag}_s*.npz"))
                if len(files) != 10:
                    continue
                m = np.load(mf); Ps = {int(re.search(r"_s(\d+)\.npz$", f).group(1)): np.load(f) for f in files}
                folds = sorted({k[:-2] for k in m.files if k.endswith("_y")})
                for key in Ps[min(Ps)].files:
                    fold = next(f for f in folds if key.startswith(f + "_"))
                    out[key] = (m, fold, {s: P[key] for s, P in Ps.items()})
        return out
    T24, T72 = load(dirs24), load(dirs72)
    refs_of = {"tf_ffn_res_tp_L24_zeroshot": ["tf_ffn_res_tp_zeroshot", "gbm_res_zeroshot", "ridge_res_zeroshot"],
               "storm_bz_tp_L24_zeroshot": ["storm_bz_tp_zeroshot", "gbm_res_zeroshot"],
               "tf_ffn_res_tp_L24_full": ["tf_ffn_res_full"], "tf_ffn_res_tp_L24_heads": ["tf_ffn_res_heads"],
               "storm_bz_tp_L24_full": ["storm_bz_full"], "storm_bz_tp_L24_heads": ["storm_bz_heads"]}
    seedpe = lambda y, P, i: np.array([pe_clim(y, P[s][i]) for s in sorted(P)])
    own, pairs = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for k24 in sorted(T24):
        m24, fold, p24 = T24[k24]; tg = fold.split("_")[0]; name = k24[len(fold) + 1:]
        y24 = m24[f"{fold}_y"]; allidx = np.arange(len(y24))
        own[tg][name].append(seedpe(y24, p24, allidx))
        own[tg][f"persistence@{name}"].append(pe_clim(y24, np.repeat(m24[f"{fold}_persist"][:, None], y24.shape[1], 1))[None])
        for ref in refs_of.get(name, []):
            if f"{fold}_{ref}" not in T72:
                continue
            m72, _, p72 = T72[f"{fold}_{ref}"]
            _, i24, i72 = np.intersect1d(m24[f"{fold}_t"], m72[f"{fold}_t"], return_indices=True)
            y = y24[i24]
            assert np.allclose(y, m72[f"{fold}_y"][i72]), f"{fold}: targets differ on common origins"
            pairs[tg][(name, ref)].append((seedpe(y, p24, i24), seedpe(y, p72, i72)))
    rows, ps, comp = [], [], defaultdict(dict)
    for tg in pairs:
        for (name, ref), v in pairs[tg].items():
            x = np.mean([a for a, _ in v], 0); z = np.mean([c for _, c in v], 0)      # mean over folds: seeds x H
            comp[tg][f"{name}|{ref}"] = dict(model=x.mean(0).tolist(), ref=z.mean(0).tolist(), n_folds=len(v))
            if x.shape[0] == z.shape[0] == 10:
                for h in range(x.shape[1]):
                    p = stats.ttest_rel(x[:, h], z[:, h]).pvalue
                    rows.append((tg, name, ref, h, float((x[:, h] - z[:, h]).mean()), float(p))); ps.append(p)
    tests = [dict(target=tg, family=n, ref=r, h_index=h, diff=d, p=p, p_holm=float(q))
             for (tg, n, r, h, d, p), q in zip(rows, holm(ps))]
    res = {tg: {k: np.mean(v, 0).mean(0).tolist() for k, v in d.items()} for tg, d in own.items()}
    return dict(scores=res, common=comp, tests=tests)


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


def write_tex24(R, N, path):
    """Table of the 24 h-window models against their 72 h versions and the trees, at 6-24 h, on the 72 h test origins."""
    W, T = R["window24"], R.get("window24_transfer", {}).get("common", {})
    rows = [("persistence", "Persistence", None), ("gbm_res", "Gradient-boosted trees", 72), ("gbm_res_L24", "Gradient-boosted trees", 24),
            ("lstm_res", "Residual LSTM", 72), ("tf_ffn_res_tp", "Residual Transformer, prot.\\ B", 72),
            ("tf_ffn_res_tp_L24", "Residual Transformer, prot.\\ B", 24), ("storm_bz_tp", "STORM-PhysNet, prot.\\ B", 72),
            ("storm_bz_tp_L24", "STORM-PhysNet, prot.\\ B", 24)]
    f3 = lambda v: "--" if v is None else f"{v:.3f}"
    zs_l24 = {"tf_ffn_res_tp_L24": "tf_ffn_res_tp_L24_zeroshot|tf_ffn_res_tp_zeroshot", "storm_bz_tp_L24": "storm_bz_tp_L24_zeroshot|storm_bz_tp_zeroshot"}

    def zs(f, tg):
        if f == "persistence":
            return N["transfer"][tg]["persistence"]["-"][2]
        if f in zs_l24:
            return T[tg][zs_l24[f]]["model"][2] if tg in T and zs_l24[f] in T[tg] else None
        v = N["transfer"].get(tg, {}).get(f, {}).get("zeroshot")
        return v[2] if v else None
    # the best value per column (lower is better for RMSE), in bold
    cells = []
    for f, lab, L in rows:
        m = W["persistence"] if f == "persistence" else W["mean"].get(f)
        if m is None:
            continue
        g = lambda k, i: m[k][i] if k in m else None
        cells.append((f"{lab}" + (f" & {L}" if L else " & --"),
                      [g("pe", 1), g("pe", 2), g("pe", 3), g("storm_rmse", 2), g("storm_rmse", 3), g("hss", 2),
                       g("noise1", 2), g("noise2", 2), g("out72", 2), zs(f, "goes16"), zs(f, "goes18"), zs(f, "grasp")]))
    lower = {3, 4}
    best = []
    for j in range(12):
        vals = [c[1][j] for c in cells[1:] if c[1][j] is not None]
        best.append((min(vals) if j in lower else max(vals)) if vals else None)
    body = []
    for lab, v in cells:
        out = []
        for j, x in enumerate(v):
            s = f3(x)
            out.append(f"\\textbf{{{s}}}" if x is not None and best[j] is not None and round(x, 3) == round(best[j], 3) and lab[:4] != "Pers" else s)
        body.append(lab + " & " + " & ".join(out) + " \\\\")
    tex = (r"""\begin{table*}[!t]
\centering
\caption{Models with a 24\,h input window against their 72\,h versions and the gradient-boosted trees, on the
forecast origins of the 72\,h models (all GOES-15 test windows). \(\PE_{\mathrm{clim}}\) at 6, 12 and 24\,h; storm-time
RMSE (Dst \(\le-50\)\,nT, \(\log_{10}\) flux) at 12 and 24\,h; alert HSS at 12\,h; \(\PE_{\mathrm{clim}}\) at 12\,h with noise
of \(\sigma=1\) and 2 and without driver data; zero-shot \(\PE_{\mathrm{clim}}\) at 12\,h on the other satellites. Means
over ten seeds; bold: best model in each column. The trees with a 24\,h window were not transferred.}
\label{tab:window24}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcccccccccccccc}
\toprule
 & & \multicolumn{3}{c}{PE, clean} & \multicolumn{2}{c}{Storm RMSE} & HSS & \multicolumn{3}{c}{PE, 12\,h, degraded} & \multicolumn{3}{c}{Zero-shot PE, 12\,h} \\
Model & Window (h) & 6\,h & 12\,h & 24\,h & 12\,h & 24\,h & 12\,h & \(\sigma=1\) & \(\sigma=2\) & no drivers & GOES-16 & GOES-18 & GRASP \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}
""")
    open(path, "w").write(tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="+", required=True); ap.add_argument("--lookback", nargs="*", default=[])
    ap.add_argument("--out", required=True); ap.add_argument("--tex", default=None)
    ap.add_argument("--transfer72", nargs="*", default=[]); ap.add_argument("--transfer24", nargs="*", default=[])
    ap.add_argument("--numbers", default=None, help="numbers.json of journal_analysis.py (72 h transfer scores)")
    ap.add_argument("--tex24", default=None)
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
        R["lookback_tests"] = lookback_tests(R["lookback"])
        import glob
        d24 = [d for d in a.lookback if glob.glob(f"{d}/preds_*_L24_s*.npz")]
        if d24:
            R["window24"] = window24(meta, P, d24)
    if a.transfer24:
        R["window24_transfer"] = window24_transfer(a.transfer72, a.transfer24)
    json.dump(R, open(a.out, "w"), indent=1); print("wrote", a.out)
    if a.tex:
        write_tex(R, a.tex); print("wrote", a.tex)
    if a.tex24 and "window24" in R and a.numbers:
        write_tex24(R, json.load(open(a.numbers)), a.tex24); print("wrote", a.tex24)


if __name__ == "__main__":
    main()
