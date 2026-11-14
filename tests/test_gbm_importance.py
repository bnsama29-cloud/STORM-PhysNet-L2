import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))


def test_feature_groups_follow_classical_feature_order():
    from gbm_importance import feature_groups, GROUPS
    g = feature_groups()
    assert len(g) == 75 and set(g) <= set(GROUPS)
    assert g[0] == "flux_now" and g[1:26] == ["flux_hist"] * 25 and g[26:29] == ["flux_spread"] * 3
    # driver block: latest value of vsw, bz, kp at the documented offsets; last feature is the 24 h Bz minimum
    assert g[29] == "speed" and g[30] == "imf" and g[29 + 13] == "geo" and g[-1] == "imf"


def test_importance_groups_sum_to_one(tmp_path):
    import pickle
    import lightgbm as lgb
    from gbm_importance import importance
    rng = np.random.default_rng(0); X = rng.normal(size=(400, 75)); y = X[:, 0] + 0.5 * X[:, 40]
    b = lgb.train({"objective": "regression", "verbose": -1, "num_leaves": 7}, lgb.Dataset(X, y), 20)
    m = {"kind": "gbm", "model": b.model_to_string(), "best_iteration": 0}
    f = tmp_path / "model_gbm_res_s42.pkl"; pickle.dump({"horizons": [1, 6], "models": [m, m]}, open(f, "wb"))
    out = importance([str(f)])
    assert np.allclose(np.sum(out["mean"], 1), 1.0) and out["horizons"] == [1, 6]
    assert out["mean"][0][out["groups"].index("flux_now")] > 0.3


def test_scorecard_ranks_lower_rmse_as_better():
    from journal_figures import scorecard_table, SCORE_MODELS
    H = [1, 6, 12, 24]; v = lambda x: [x] * 4
    N = {"horizons": H, "main": {}, "probes": {}, "storm_rmse": {}, "alert_hss": {}, "ensemble": {}, "calibration": {},
         "transfer": {"goes16": {}, "grasp": {}}}
    for k, f in enumerate(SCORE_MODELS):
        N["main"][f] = {"pe_clim": v(0.8 - 0.01 * k)}; N["probes"][f] = {"noise2": v(0.5), "out72": v(0.6)}
        N["storm_rmse"][f] = v(0.6 + 0.01 * k); N["alert_hss"][f] = v(0.7); N["ensemble"][f] = {"crps": v(0.2)}
        N["transfer"]["goes16"][f] = {"zeroshot": v(0.4)}; N["transfer"]["grasp"][f] = {"zeroshot": v(0.6)}
    cols, V = scorecard_table(N)
    assert V.shape == (len(SCORE_MODELS), len(cols)) and cols[3][1] is False and V[0, 3] < V[-1, 3]
