#!/usr/bin/env python
"""Gain importance of the block-A gradient-boosted trees, grouped by physical input, per lead time.

  python gbm_importance.py --models <dir with model_gbm_res_s*.pkl> --out importance.json
"""
import argparse, glob, json, pickle
import numpy as np

DRV = ["vsw", "bz", "bt", "density", "pdyn", "bz_south_duration", "storm_onset_hours",
       "vsw_roll1h", "vsw_roll6h", "vsw_roll24h", "bz_roll1h", "bz_roll6h", "bz_x_vsw", "kp", "dst"]
DRV_GROUP = {"vsw": "speed", "vsw_roll1h": "speed", "vsw_roll6h": "speed", "vsw_roll24h": "speed",
             "bz": "imf", "bt": "imf", "bz_south_duration": "imf", "bz_roll1h": "imf", "bz_roll6h": "imf",
             "bz_x_vsw": "imf", "density": "plasma", "pdyn": "plasma",
             "kp": "geo", "dst": "geo", "storm_onset_hours": "geo"}
GROUPS = ["flux_now", "flux_hist", "flux_spread", "speed", "imf", "plasma", "geo"]


def feature_groups():
    """Group of each of the 75 features, in the order of classical_baselines.features()."""
    g = ["flux_now"] + ["flux_hist"] * 23 + ["flux_hist", "flux_hist"] + ["flux_spread"] * 3
    for _ in range(3):
        g += [DRV_GROUP[d] for d in DRV]
    g += ["imf"]
    assert len(g) == 75
    return g


def importance(model_files):
    import lightgbm as lgb
    fg = np.array(feature_groups())
    per_h = None
    for f in model_files:
        d = pickle.load(open(f, "rb"))
        rows = []
        for m in d["models"]:
            b = lgb.Booster(model_str=m["model"])
            gain = b.feature_importance("gain", iteration=m["best_iteration"] or None)
            gain = gain / gain.sum()
            rows.append([gain[fg == g].sum() for g in GROUPS])
        per_h = np.array(rows)[None] if per_h is None else np.concatenate([per_h, np.array(rows)[None]])
    return {"groups": GROUPS, "horizons": [int(h) for h in d["horizons"]],
            "mean": per_h.mean(0).tolist(), "sd": per_h.std(0).tolist(), "n_seeds": len(model_files)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.models}/model_gbm_res_s*.pkl"))
    json.dump(importance(files), open(a.out, "w"), indent=1)
