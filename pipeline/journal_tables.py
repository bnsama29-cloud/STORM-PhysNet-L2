#!/usr/bin/env python
"""LaTeX tables of the journal paper from numbers.json (journal_analysis.py). Needs \\usepackage{adjustbox,booktabs}.

  python journal_tables.py --numbers journal_report/numbers.json --out paper/tables

Families listed in ORDER but missing from numbers.json are printed as pending rows (\\pend{}), so the tables can
be regenerated unchanged once the remaining Kaggle runs are in.
"""
import argparse, json
from pathlib import Path

ORDER = [  # (family, label, group)
    ("persistence", "Persistence", "Reference"),
    ("ridge_res", "Ridge regression, residual", "Classical baselines"), ("gbm_res", "Gradient-boosted trees, residual", "Classical baselines"),
    ("lstm", "LSTM", "Baselines"), ("lstm_res", "LSTM, residual", "Baselines"),
    ("lstm_res_w64", "\\quad width 64", "Baselines"), ("lstm_res_w256", "\\quad width 256", "Baselines"),
    ("lstm_res_tp", "\\quad training protocol B", "Baselines"), ("mlp", "MLP", "Baselines"),
    ("cnn", "CNN", "Baselines"),
    ("transformer", "Transformer (default size)", "Transformer controls"),
    ("transformer_matched", "Transformer, matched", "Transformer controls"),
    ("tf_ffn256", "Transformer, FFN 256", "Transformer controls"),
    ("tf_ffn_res", "Residual Transformer", "Transformer controls"),
    ("tf_ffn_res_tp", "\\quad training protocol B", "Transformer controls"),
    ("tf_res_fixed_ma", "\\quad + fixed average", "Transformer controls"),
    ("tf_flux_only", "Flux-only Transformer", "Transformer controls"),
    ("storm_bz", "STORM-PhysNet", "STORM-PhysNet and variants"),
    ("storm_bz_tp", "\\quad training protocol B", "STORM-PhysNet and variants"),
    ("storm_fixed_ma", "\\quad fixed average", "STORM-PhysNet and variants"),
    ("storm_no_delay", "\\quad no delay", "STORM-PhysNet and variants"),
    ("storm_no_gate", "\\quad no gate", "STORM-PhysNet and variants"),
    ("storm_no_physics", "\\quad no physics loss", "STORM-PhysNet and variants"),
    ("storm_physfix", "\\quad physical-unit loss", "STORM-PhysNet and variants"),
    ("storm_physfix_e1", "\\quad physical-unit loss from epoch 1", "STORM-PhysNet and variants"),
    ("storm_cathode", "\\quad cathode--anode gate", "STORM-PhysNet and variants"),
    ("storm_cathode_spec", "\\quad cathode--anode + spectral head", "STORM-PhysNet and variants"),
    ("storm_radiotrophic", "\\quad radiotrophic gate", "STORM-PhysNet and variants"),
    ("storm_lstm_backbone", "\\quad LSTM encoder", "STORM-PhysNet and variants"),
    ("storm_lstm_backbone_tp", "\\quad LSTM encoder, training protocol B", "STORM-PhysNet and variants"),
    ("storm_lstm_no_gate", "\\quad LSTM encoder, no gate", "STORM-PhysNet and variants"),
    ("storm_lstm_no_delay", "\\quad LSTM encoder, no delay", "STORM-PhysNet and variants"),
    ("storm_lstm_no_physics", "\\quad LSTM encoder, no physics loss", "STORM-PhysNet and variants"),
    ("storm_lstm_physfix", "\\quad LSTM encoder, physical-unit loss", "STORM-PhysNet and variants"),
    ("storm_lstm_physfix_e1", "\\quad LSTM encoder, physical-unit loss from epoch 1", "STORM-PhysNet and variants"),
    ("storm_magnetopause", "\\quad + magnetopause features", "STORM-PhysNet and variants"),
]
LABEL = {f: l.replace("\\quad ", "STORM-PhysNet, " if f.startswith("storm_") else "Residual TF ") for f, l, _ in ORDER}
LABEL.update({"lstm_res_w64": "LSTM, residual, width 64", "lstm_res_w256": "LSTM, residual, width 256"})


def f3(v): return "--" if v is None or v != v else f"{v:.3f}"


def rows(N, key_fn, H, groups=True, only=None):
    out, last = [], None
    for f, lab, g in ORDER:
        if only and f not in only: continue
        if groups and g != last:
            out.append(f"\\multicolumn{{{1 + len(H)}}}{{l}}{{\\textit{{{g}}}}} \\\\"); last = g
        v = key_fn(f)
        out.append(f"{lab} & " + (" & ".join(f3(x) for x in v) if v is not None else " & ".join(["\\pend{}"] * len(H))) + " \\\\")
    return out


HIDE_PENDING = False


def tabular(cols, head, body, caption, label, note="", star=False):
    env = "table*" if star else "table"
    if HIDE_PENDING:   # drop rows of runs that have not arrived, and the note that explains them
        body = [r for r in body if "\\pend" not in r]
        note = note.replace(" \\pend{} marks runs still in progress.", "").replace("\\pend{} marks runs still in progress.", "")
    return "\n".join([f"\\begin{{{env}}}[!t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\footnotesize",
                      "\\setlength{\\tabcolsep}{4pt}", "\\begin{adjustbox}{max width=\\linewidth}",
                      f"\\begin{{tabular}}{{{cols}}}", "\\toprule", head + " \\\\", "\\midrule", *body, "\\bottomrule", "\\end{tabular}",
                      "\\end{adjustbox}",
                      (f"\n\\par\\vspace{{0.5ex}}{{\\footnotesize\\raggedright {note}\\par}}" if note else ""), f"\\end{{{env}}}", ""])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--numbers", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--hide-pending", action="store_true", help="omit rows of families without results")
    a = ap.parse_args(); global HIDE_PENDING; HIDE_PENDING = a.hide_pending; N = json.load(open(a.numbers)); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    H = N["horizons"]; hh = " & ".join(f"{h}\\,h" for h in H); cols = "l" + "c" * len(H)
    have = set(N["main"])

    def main_v(f):
        if f == "persistence": return N["persistence"]["pe_clim"]
        return N["main"][f]["pe_clim"] if f in have else None
    (out / "tab_main.tex").write_text(tabular(cols, f"Model & {hh}", rows(N, main_v, H),
        "GOES-15 test skill, $\\mathrm{PE}_\\mathrm{clim}$: mean over ten seeds and the three test blocks.", "tab:main",
        "Seed standard deviations are at most 0.011 for the other controls without residual, 0.018 for the residual Transformers, "
        "0.027 for the STORM-PhysNet variants with a Transformer encoder, 0.008 for the models with an LSTM encoder "
        "and the persistence residual, and 0.053 for the CNN. "
        "\\pend{} marks runs still in progress.", star=True))

    def blockA(f):
        if f == "persistence": return None
        return N["main"][f]["per_block"]["A"] if f in have else None
    (out / "tab_blockA.tex").write_text(tabular(cols, f"Model & {hh}", rows(N, blockA, H, only=[o[0] for o in ORDER if o[0] != "persistence"]),
        "Block A (strictly chronological test, Sep 2018--Mar 2020): $\\mathrm{PE}_\\mathrm{clim}$, mean over ten seeds.", "tab:blockA", star=True))

    # paired tests vs STORM-PhysNet
    P = {}
    for r in N.get("paired_vs_storm", []): P.setdefault(r["family"], {})[r["h"]] = r

    def cell(r):
        star = "$^{*}$" if r["p_holm"] < 0.05 else ""
        return f"{r['diff']:+.4f}{star}"
    body = []
    for f, lab, g in ORDER:
        if f in ("persistence", "storm_bz"): continue
        body.append(f"{LABEL[f]} & " + (" & ".join(cell(P[f][h]) for h in H) if f in P else " & ".join(["\\pend{}"] * len(H))) + " \\\\")
    (out / "tab_paired.tex").write_text(tabular(cols, f"STORM-PhysNet minus & {hh}", body,
        "Difference in $\\mathrm{PE}_\\mathrm{clim}$, STORM-PhysNet minus each model (positive: STORM-PhysNet better).", "tab:paired",
        "Paired $t$-test on the ten seed means (each averaged over blocks); $^{*}$ Holm-adjusted $p<0.05$ over all comparisons in the table.", star=True))

    # attribution of the LSTM-encoder variant (campaigns 2 and 3)
    PL = {}
    for r in N.get("paired_vs_lstm", []): PL.setdefault(r["family"], {})[r["h"]] = r
    lstm_rows = [("lstm_res", "Residual LSTM"), ("storm_lstm_no_gate", "no gate"),
                 ("storm_lstm_no_delay", "no delay"), ("storm_lstm_no_physics", "no physics loss"),
                 ("storm_lstm_physfix", "physical-unit loss"), ("storm_lstm_physfix_e1", "physical-unit loss, epoch 1")]
    body = [f"{lab} & " + (" & ".join(cell(PL[f][h]) for h in H) if f in PL else " & ".join(["\\pend{}"] * len(H))) + " \\\\"
            for f, lab in lstm_rows]
    (out / "tab_paired_lstm.tex").write_text(tabular(cols, f"LSTM variant minus & {hh}", body,
        "Difference in $\\mathrm{PE}_\\mathrm{clim}$, STORM-PhysNet with LSTM encoder minus each one-component change "
        "(positive: the full LSTM variant is better).", "tab:paired_lstm",
        "Paired $t$-test on the ten seed means; $^{*}$ Holm-adjusted $p<0.05$ over the comparisons in this table."))

    # storm-time RMSE and alert HSS
    def sr(f): return N["storm_rmse"].get(f) if f in N["storm_rmse"] else None
    def hs(f): return N["alert_hss"].get(f) if f in N["alert_hss"] else None
    two = ["\\multicolumn{%d}{c}{Storm-time RMSE} & \\multicolumn{%d}{c}{Alert HSS} \\\\" % (len(H), len(H))]
    body = []
    for f, lab, g in ORDER:
        a_, b_ = sr(f), hs(f)
        cells = (" & ".join(f3(x) for x in a_) if a_ else " & ".join(["\\pend{}"] * len(H))) + " & " + \
                (" & ".join(f3(x) for x in b_) if b_ else " & ".join(["\\pend{}"] * len(H)))
        body.append(f"{lab} & {cells} \\\\")
    (out / "tab_storm.tex").write_text(tabular("l" + "c" * 2 * len(H), " & " + two[0].rstrip(" \\") + " \\\\\nModel & " + hh + " & " + hh, body,
        "Storm-time RMSE (log$_{10}$ flux; Dst $\\le -50$\\,nT at the target hour) and Heidke skill score for exceeding 1000\\,pfu.",
        "tab:storm", "Seed means over the three blocks. Lower RMSE and higher HSS are better.", star=True))

    # probes
    pk = [("noise0.5", "$\\sigma=0.5$"), ("noise1", "$\\sigma=1$"), ("noise2", "$\\sigma=2$"), ("out24", "24\\,h outage"),
          ("out72", "no drivers"), ("lag3", "Kp/Dst 3\\,h late")]
    i12 = H.index(12); i24 = H.index(24) if 24 in H else len(H) - 1
    body = []
    for f, lab, g in ORDER:
        if f == "persistence": continue
        if f in N["probes"]:
            body.append(f"{lab} & " + " & ".join(f"{N['probes'][f][k][i12]:.3f}" for k, _ in pk) + " \\\\")
        elif f not in have:
            body.append(f"{lab} & " + " & ".join(["\\pend{}"] * len(pk)) + " \\\\")
    (out / "tab_probes.tex").write_text(tabular("l" + "c" * len(pk), "Model & " + " & ".join(n for _, n in pk), body,
        "Robustness probes: 12\\,h $\\mathrm{PE}_\\mathrm{clim}$ with degraded inputs (seed means over blocks).", "tab:probes",
        "Noise: Gaussian noise of standard deviation $\\sigma$ (in units of each standardised channel) on the 16 continuous "
        "auxiliary channels; the flux history and the observed-hour mask are unchanged. Outage: driver channels held at their last value for the final 24 input hours. No drivers: "
        "driver channels at their training mean for the whole input. Clean values are in Table~\\ref{tab:main}.", star=True))

    # calibration
    body = []
    for f, lab, g in ORDER:
        if not f.startswith("storm_"): continue
        c = N["calibration"].get(f)
        if c: body.append(f"{lab} & " + " & ".join(f"{c['cov68'][i]:.2f} / {c['cov95'][i]:.2f}" for i in range(len(H))) + " \\\\")
        else: body.append(f"{lab} & " + " & ".join(["\\pend{}"] * len(H)) + " \\\\")
    (out / "tab_calib.tex").write_text(tabular(cols, f"Model & {hh}", body,
        "Coverage of the predicted $1\\sigma$ / $1.96\\sigma$ intervals (nominal 0.68 / 0.95).", "tab:calib",
        "Fraction of test targets inside $\\hat y\\pm\\hat\\sigma$ and $\\hat y\\pm1.96\\hat\\sigma$, averaged over seeds and blocks.", star=True))

    # transfer: one wide table, satellites side by side
    names = {"goes16": "GOES-16 (75\\textdegree W)", "goes18": "GOES-18 (137\\textdegree W)", "grasp": "GSAT-19 GRASP (48\\textdegree E)"}
    var = [("zeroshot", "zero"), ("heads", "heads"), ("full", "full"), ("scratch", "scr.")]
    sats = [t for t in ("goes16", "goes18", "grasp") if t in N["transfer"]]
    nv = len(var)
    head = (" & " + " & ".join(f"\\multicolumn{{{nv}}}{{c}}{{{names[t]}}}" for t in sats) + " \\\\\n"
            + "".join(f"\\cmidrule(lr){{{2 + k * nv}-{1 + (k + 1) * nv}}}" for k in range(len(sats))) + "\n"
            + "Model & " + " & ".join(" & ".join(n for _, n in var) for _ in sats))
    body = ["Persistence & " + " & ".join(f"\\multicolumn{{{nv}}}{{c}}{{{N['transfer'][t]['persistence']['-'][i12]:.3f}}}" for t in sats) + " \\\\"]
    for f, lab, g in ORDER:
        if f == "persistence": continue
        if any(f in N["transfer"][t] for t in sats):
            cells = []
            for t in sats:
                T = N["transfer"][t]
                cells += [f3(T[f][v][i12]) if f in T and v in T[f] else "--" for v, _ in var]
            body.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        elif f not in have:
            body.append(f"{lab} & " + " & ".join(["\\pend{}"] * (nv * len(sats))) + " \\\\")
    (out / "tab_transfer.tex").write_text(tabular("l" + "c" * (nv * len(sats)), head, body,
        "Cross-satellite transfer from the block-A GOES-15 models: 12\\,h $\\mathrm{PE}_\\mathrm{clim}$, mean over folds and ten seeds.",
        "tab:transfer", "Zero: zero-shot, no retraining. Heads: encoder frozen, output layers retrained. Full: all weights fine-tuned. "
        "Scr.: trained from scratch on the target satellite only. ``--'': not run (zero-shot only for these models).", star=True))
    print("wrote", sorted(p.name for p in out.glob("*.tex")))


if __name__ == "__main__":
    main()
