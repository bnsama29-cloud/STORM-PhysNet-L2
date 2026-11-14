#!/usr/bin/env python
"""Tables and numbers of the journal paper from the saved predictions.

  python journal_analysis.py --blocks j24_A j24_B j24_C j2_A j2_B j2_C --transfer tj tj2 --out journal_report

Any family found in the block folders is included, so the same script covers the conference families
(journal24_[ABC].zip) and the journal additions (journal_l2_account*.zip). Outputs in --out:
  numbers.json      every number quoted in the paper
  tables.md         the tables, in Markdown

Conventions: PE_clim = 1 - MSE/Var(y) per block and horizon; seed means averaged over the three blocks; paired
t-tests on the ten seed means (each averaged over blocks), Holm-corrected over all comparisons with STORM-PhysNet;
storm time = Dst <= -50 nT at the target hour; alert event = log10 flux >= 3 (1000 pfu, the NOAA >2 MeV alert level).
"""
import argparse, glob, json, os, re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ALERT = 3.0
REF = "storm_bz"


# ------------------------------------------------------------------------------------------------ metrics
def pe_clim(y, p): return 1 - ((y - p) ** 2).mean(0) / y.var(0)
def pe_pers(y, p, pers): return 1 - ((y - p) ** 2).mean(0) / ((y - pers[:, None]) ** 2).mean(0)
def rmse(y, p): return np.sqrt(((y - p) ** 2).mean(0))


def contingency(obs_ev, pred_ev):
    a = np.sum(obs_ev & pred_ev); b = np.sum(~obs_ev & pred_ev); c = np.sum(obs_ev & ~pred_ev); d = np.sum(~obs_ev & ~pred_ev)
    return a, b, c, d


def hss(obs_ev, pred_ev):
    """Heidke skill score; 0 when the denominator is 0."""
    a, b, c, d = (float(v) for v in contingency(obs_ev, pred_ev))
    den = (a + c) * (c + d) + (a + b) * (b + d)
    return 2 * (a * d - b * c) / den if den else 0.0


def pod_far(obs_ev, pred_ev):
    a, b, c, _ = (float(v) for v in contingency(obs_ev, pred_ev))
    return (a / (a + c) if a + c else np.nan), (b / (a + b) if a + b else np.nan)


def crps_gauss(y, mu, sigma):
    """CRPS of N(mu, sigma^2) at y (Gneiting & Raftery 2007)."""
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_ensemble(y, ens):
    """Empirical CRPS of an ensemble ens (M, N, H) at y (N, H): E|X-y| - 0.5 E|X-X'|."""
    t1 = np.abs(ens - y[None]).mean(0)
    t2 = np.abs(ens[:, None] - ens[None]).mean((0, 1))
    return t1 - 0.5 * t2


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


# ------------------------------------------------------------------------------------------------ loading
def load_blocks(dirs):
    meta, P = {}, defaultdict(lambda: defaultdict(dict))         # P[block][family][seed] = npz
    for d in dirs:
        for mf in glob.glob(f"{d}/meta_*.npz"):
            b = re.search(r"meta_(\w)\.npz", mf).group(1); new = dict(np.load(mf, allow_pickle=True))
            if b in meta:   # runs from different campaigns must share the same test windows, or the paired tests are void
                assert meta[b]["y"].shape == new["y"].shape and np.allclose(meta[b]["y"], new["y"]), \
                    f"{mf}: test targets differ from an earlier meta_{b}.npz (different code or data version?)"
            meta.setdefault(b, new)
        for f in glob.glob(f"{d}/preds_*_s*.npz"):
            m = re.search(r"preds_(\w)_(.+)_s(\d+)\.npz", os.path.basename(f))
            P[m.group(1)][m.group(2)][int(m.group(3))] = np.load(f)
    return meta, P


def families(P):
    """Families present with all 10 seeds in every block."""
    blocks = sorted(P); fams = set.intersection(*[set(P[b]) for b in blocks]) if blocks else set()
    return sorted(f for f in fams if all(len(P[b][f]) == 10 for b in blocks)), blocks


# ------------------------------------------------------------------------------------------------ analysis
def per_seed(meta, P, fam, blocks, fn, key="pred"):
    """fn(meta_b, pred) -> (H,) ; returns (seeds, H) averaged over blocks."""
    seeds = sorted(P[blocks[0]][fam])
    return np.array([np.mean([fn(meta[b], P[b][fam][s][key]) for b in blocks], 0) for s in seeds])


LSTM_ATTR = ["lstm_res", "storm_lstm_no_gate", "storm_lstm_no_delay", "storm_lstm_no_physics", "storm_lstm_physfix",
             "storm_lstm_physfix_e1"]


def paired(S, ref, fams, H):
    """Paired t-tests of ref minus each family on the per-seed (block-averaged) scores, Holm-adjusted over all rows.
    Families absent from S are skipped; returns [] when ref or every family is missing."""
    fams = [f for f in fams if f in S and f != ref]
    if ref not in S or not fams:
        return []
    rows, ps = [], []
    for f in fams:
        d = S[ref] - S[f]
        for i, h in enumerate(H):
            p = stats.ttest_1samp(d[:, i], 0).pvalue; rows.append((f, h, float(d[:, i].mean()), float(p))); ps.append(p)
    return [dict(family=f, h=h, diff=dd, p=p, p_holm=float(a)) for (f, h, dd, p), a in zip(rows, holm(ps))]


def analyse(meta, P, transfer_dirs):
    fams, blocks = families(P)
    H = [int(h) for h in meta[blocks[0]]["horizons"]]
    N = {"horizons": H, "blocks": blocks, "families": fams, "n_test": {b: int(len(meta[b]["y"])) for b in blocks}}
    clim = lambda m, p: pe_clim(m["y"], p)
    # 1. main skill
    N["persistence"] = {"pe_clim": np.mean([pe_clim(meta[b]["y"], np.repeat(meta[b]["persist"][:, None], len(H), 1)) for b in blocks], 0).tolist()}
    S = {f: per_seed(meta, P, f, blocks, clim) for f in fams}
    N["main"] = {f: {"pe_clim": S[f].mean(0).tolist(), "sd": S[f].std(0, ddof=1).tolist(),
                     "pe_pers": per_seed(meta, P, f, blocks, lambda m, p: pe_pers(m["y"], p, m["persist"])).mean(0).tolist(),
                     "per_block": {b: np.mean([pe_clim(meta[b]["y"], P[b][f][s]["pred"]) for s in P[b][f]], 0).tolist() for b in blocks}}
                 for f in fams}
    # 2. paired tests against STORM-PhysNet
    if REF in fams:
        rows, ps = [], []
        for f in fams:
            if f == REF: continue
            d = S[REF] - S[f]
            for i, h in enumerate(H):
                p = stats.ttest_1samp(d[:, i], 0).pvalue; rows.append((f, h, float(d[:, i].mean()), float(p))); ps.append(p)
        adj = holm(ps)
        N["paired_vs_storm"] = [dict(family=f, h=h, diff=dd, p=p, p_holm=float(a)) for (f, h, dd, p), a in zip(rows, adj)]
    # 2b. attribution of the LSTM-encoder variant (campaigns 2 and 3): reference minus each one-component change
    N["paired_vs_lstm"] = paired(S, "storm_lstm_backbone", LSTM_ATTR, H)
    # width sensitivity of the residual LSTM (default width 128 minus width 64 and 256)
    N["paired_width"] = paired(S, "lstm_res", ["lstm_res_w64", "lstm_res_w256"], H)
    # physics terms at full weight from epoch 1 (campaign 5) against the ramped and the physics-free versions
    N["paired_e1"] = (paired(S, "storm_lstm_physfix_e1", ["storm_lstm_physfix", "storm_lstm_no_physics"], H)
                      + paired(S, "storm_physfix_e1", ["storm_physfix", "storm_no_physics"], H))
    # 3. storm-time RMSE (Dst <= -50 nT at the target hour)
    def storm_rmse(m, p):
        out = []
        for i in range(len(H)):
            k = m["dst_target"][:, i] <= -50
            out.append(np.sqrt(((m["y"][k, i] - p[k, i]) ** 2).mean()) if k.any() else np.nan)
        return np.array(out)
    N["storm_rmse"] = {f: per_seed(meta, P, f, blocks, storm_rmse).mean(0).tolist() for f in fams}
    N["storm_rmse"]["persistence"] = np.mean([storm_rmse(meta[b], np.repeat(meta[b]["persist"][:, None], len(H), 1)) for b in blocks], 0).tolist()
    N["n_storm_targets"] = {b: [int((meta[b]["dst_target"][:, i] <= -50).sum()) for i in range(len(H))] for b in blocks}
    # 4. alerts (>= 1000 pfu)
    def alert(m, p):
        return np.array([hss(m["y"][:, i] >= ALERT, p[:, i] >= ALERT) for i in range(len(H))])
    N["alert_hss"] = {f: per_seed(meta, P, f, blocks, alert).mean(0).tolist() for f in fams}
    N["alert_hss"]["persistence"] = np.mean([alert(meta[b], np.repeat(meta[b]["persist"][:, None], len(H), 1)) for b in blocks], 0).tolist()
    N["alert_rate"] = {b: [float((meta[b]["y"][:, i] >= ALERT).mean()) for i in range(len(H))] for b in blocks}
    # 5. robustness probes
    keys = [k for k in P[blocks[0]][fams[0]][42].files if k.startswith("pred_") and k != "pred_spectral"] if fams else []
    N["probes"] = {}
    for f in fams:
        ks = [k for k in P[blocks[0]][f][42].files if k.startswith(("pred_noise", "pred_out", "pred_lag"))]
        if ks: N["probes"][f] = {k[5:]: per_seed(meta, P, f, blocks, clim, key=k).mean(0).tolist() for k in ks}
    # 6. calibration of the predicted variance (families that output log-variance)
    N["calibration"] = {}
    for f in fams:
        if "logvar" not in P[blocks[0]][f][42].files: continue
        cov68, cov95, crps = [], [], []
        for b in blocks:
            y = meta[b]["y"]
            for s in P[b][f]:
                mu = P[b][f][s]["pred"]; sg = np.exp(0.5 * P[b][f][s]["logvar"]); e = np.abs(y - mu)
                cov68.append((e <= sg).mean(0)); cov95.append((e <= 1.96 * sg).mean(0)); crps.append(crps_gauss(y, mu, sg).mean(0))
        N["calibration"][f] = {"cov68": np.mean(cov68, 0).tolist(), "cov95": np.mean(cov95, 0).tolist(), "crps": np.mean(crps, 0).tolist()}
    # 7. seed ensembles ("bagging"): mean of the 10 seeds; empirical CRPS of the 10-member ensemble
    N["ensemble"] = {}
    for f in fams:
        pe, cr = [], []
        for b in blocks:
            ens = np.stack([P[b][f][s]["pred"] for s in sorted(P[b][f])]); y = meta[b]["y"]
            pe.append(pe_clim(y, ens.mean(0))); cr.append(crps_ensemble(y, ens).mean(0))
        N["ensemble"][f] = {"pe_clim": np.mean(pe, 0).tolist(), "crps": np.mean(cr, 0).tolist()}
    # 8. delay and gate (STORM-PhysNet)
    if REF in fams and "tau" in P[blocks[0]][REF][42].files:
        tau = np.concatenate([P[b][REF][s]["tau"].ravel() for b in blocks for s in P[b][REF]])
        N["tau"] = {"mean": float(tau.mean()), "frac_ge_1.45": float((tau >= 1.45).mean())}
    # 9. transfer
    N["transfer"] = transfer(transfer_dirs)
    return N


def transfer(dirs):
    out = {}
    for d in dirs:
        for mf in glob.glob(f"{d}/transfer_meta*.npz"):
            tag = re.search(r"transfer_meta(.*)\.npz", os.path.basename(mf)).group(1)
            m = np.load(mf); preds = sorted(glob.glob(f"{d}/transfer_preds{tag}_s*.npz"))
            if len(preds) != 10:   # an unfinished run would bias the fold means; report and skip it
                print(f"WARNING {mf}: {len(preds)} seed files, expected 10 -- skipped"); continue
            Ps = [np.load(p) for p in preds]
            folds = sorted({k[:-2] for k in m.files if k.endswith("_y")})
            for key in sorted(Ps[0].files):
                tf = next(f for f in folds if key.startswith(f + "_")); rest = key[len(tf) + 1:]
                fam, var = rest.rsplit("_", 1); tg = tf.split("_")[0]
                v = np.mean([pe_clim(m[f"{tf}_y"], P[key]) for P in Ps], 0)
                out.setdefault(tg, {}).setdefault(f"{fam}|{var}", []).append(v)
            for tf in folds:
                tg = tf.split("_")[0]; y = m[f"{tf}_y"]
                out.setdefault(tg, {}).setdefault("persistence|-", []).append(pe_clim(y, np.repeat(m[f"{tf}_persist"][:, None], y.shape[1], 1)))
    res = {}
    for tg, d in out.items():
        res[tg] = {}
        for k, v in d.items():
            fam, var = k.split("|")
            res[tg].setdefault(fam, {})[var] = np.mean(v, 0).tolist()      # mean over folds (and duplicate persistence rows)
    return res


# ------------------------------------------------------------------------------------------------ output
def table(rows, header):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


def markdown(N):
    H = N["horizons"]; f3 = lambda v: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    md = ["## Main skill (PE_clim, seed mean over 3 blocks)",
          table([["persistence"] + [f3(x) for x in N["persistence"]["pe_clim"]]] +
                [[f] + [f3(x) for x in N["main"][f]["pe_clim"]] for f in N["families"]], ["model"] + [f"{h} h" for h in H])]
    if "paired_vs_storm" in N:
        md += ["## STORM-PhysNet minus each model (Holm-adjusted p)",
               table([[r["family"], str(r["h"]), f"{r['diff']:+.4f}", f"{r['p_holm']:.1e}"] for r in N["paired_vs_storm"]], ["model", "h", "diff", "p_holm"])]
    md += ["## Storm-time RMSE (Dst <= -50 nT)", table([[f] + [f3(x) for x in v] for f, v in N["storm_rmse"].items()], ["model"] + [f"{h} h" for h in H]),
           "## Alert HSS (>= 1000 pfu)", table([[f] + [f3(x) for x in v] for f, v in N["alert_hss"].items()], ["model"] + [f"{h} h" for h in H])]
    if N["calibration"]:
        md += ["## Calibration of the predicted variance",
               table([[f, k] + [f3(x) for x in v[k]] for f, v in N["calibration"].items() for k in ("cov68", "cov95", "crps")], ["model", "metric"] + [f"{h} h" for h in H])]
    md += ["## Seed ensembles", table([[f] + [f3(x) for x in v["pe_clim"]] + [f3(x) for x in v["crps"]] for f, v in N["ensemble"].items()],
                                      ["model"] + [f"PE {h} h" for h in H] + [f"CRPS {h} h" for h in H])]
    for tg, d in N["transfer"].items():
        md += [f"## Transfer: {tg} (PE_clim, mean over folds and seeds)",
               table([[f, v] + [f3(x) for x in vals] for f, vv in d.items() for v, vals in vv.items()], ["model", "variant"] + [f"{h} h" for h in H])]
    return "\n\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--blocks", nargs="+", required=True)
    ap.add_argument("--transfer", nargs="*", default=[]); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta, P = load_blocks(a.blocks)
    N = analyse(meta, P, a.transfer)
    json.dump(N, open(out / "numbers.json", "w"), indent=1, default=float)
    (out / "tables.md").write_text(markdown(N))
    print("families:", N["families"]); print("wrote", out / "numbers.json", out / "tables.md")


if __name__ == "__main__":
    main()
