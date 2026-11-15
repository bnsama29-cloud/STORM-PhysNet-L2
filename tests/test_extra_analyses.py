"""Checks of the further analyses on synthetic predictions."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import extra_analyses as ea  # noqa: E402

NS = 3600 * 10**9


def fake(n=24 * 30, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.int64) * NS
    y = np.cumsum(rng.normal(0, 0.2, (n, 2)), 0)
    meta = {"A": dict(y=y, persist=y[:, 0] - rng.normal(0, 0.3, n), t_origin=t, horizons=np.array([1, 24]))}
    perfect = {s: {"pred": y.copy()} for s in (42, 43)}
    noisy = {s: {"pred": y + rng.normal(0, 0.5, y.shape)} for s in (42, 43)}
    return meta, {"A": {"perfect": perfect, "noisy": noisy}}


def test_daily_perfect_forecast_scores_one_and_noise_scores_less():
    meta, P = fake()
    d = ea.daily(meta, P, ["perfect", "noisy"])
    assert np.allclose(d["perfect"]["mean"], 1.0)
    assert all(v < 1.0 for v in d["noisy"]["mean"])


def test_rises_fraction_one_for_perfect_and_zero_for_persistence():
    meta, P = fake()
    r = ea.rises(meta, P, ["perfect", "persistence"])
    for h, v in r["perfect"].items():
        if v["n"]:
            assert abs(v["frac"] - 1.0) < 1e-9 and v["rmse"] < 1e-9
    for h, v in r["persistence"].items():
        if v["n"]:
            assert abs(v["frac"]) < 1e-9
    f = ea.rises(meta, P, ["perfect"], sign=-1)
    assert all(v["n"] >= 0 for v in f["perfect"].values())


def test_local_time_bins_and_offset():
    meta, P = fake()
    lt = ea.local_time(meta, P, ["noisy", "perfect"], nbins=8)
    assert len(lt["noisy"]["1"]) == 8 and np.allclose(lt["perfect"]["24"], 0.0)
    # an origin at 09 UT with lead 1 h is 01 LT at 135 W, the first 3-hour bin
    assert ((9 + 1 + ea.LT_OFFSET_H) % 24) * 8 // 24 == 0


def test_by_block_returns_mean_and_sd_per_horizon():
    meta, P = fake()
    b = ea.by_block(meta, P, ["noisy", "persistence"])
    assert len(b["noisy"]["A"]["mean"]) == 2 and len(b["noisy"]["A"]["sd"]) == 2
    assert b["persistence"]["A"]["sd"] == [0.0, 0.0]


def test_lookback_compares_on_common_origins(tmp_path):
    rng = np.random.default_rng(1)
    t = np.arange(100, dtype=np.int64) * NS; y = rng.normal(size=(100, 2)); H = np.array([1, 24])
    meta72 = {"A": dict(y=y[10:], persist=y[10:, 0], t_origin=t[10:], horizons=H)}
    P72 = {"A": {"lstm_res": {s: {"pred": y[10:] + 0.1} for s in (42, 43)}}}
    d = tmp_path / "L24"; d.mkdir()
    np.savez(d / "meta_A.npz", y=y[:90], persist=y[:90, 0], t_origin=t[:90], horizons=H)
    for s in (42, 43):
        np.savez(d / f"preds_A_lstm_res_L24_s{s}.npz", pred=y[:90])
    out = ea.lookback(meta72, P72, [str(d)])
    assert out["lstm_res_L24"]["A"]["n"] == 80 and np.allclose(out["lstm_res_L24"]["A"]["pe"], 1.0)
    assert out["lstm_res@L24"]["A"]["lookback"] == 72 and all(v < 1 for v in out["lstm_res@L24"]["A"]["pe"])


def test_t_ns_accepts_datetime64_and_int64():
    t = np.array(["2018-09-01T00:00", "2018-09-01T01:00"], dtype="datetime64[ns]")
    assert (ea.t_ns({"t_origin": t}) == t.astype(np.int64)).all()
    assert (ea.t_ns({"t_origin": t.astype(np.int64)}) == t.astype(np.int64)).all()


def test_lookback_tests_pair_each_window_with_its_72h_model_and_holm_adjust():
    s = lambda v: {"A": {"per_seed": [[v + 0.01 * k, v] for k in range(4)]}, "B": {"per_seed": [[v + 0.01 * k, v] for k in range(4)]}}
    LB = {"tf_ffn_res_tp_L24": s(0.9), "tf_ffn_res_tp@L24": s(0.8), "gbm_res@L24": s(0.85), "gbm_res_L24": s(0.86),
          "ridge_res_L24": s(0.8), "ridge_res@L24": s(0.8), "lstm_res_L168": s(0.7)}
    T = ea.lookback_tests(LB)
    fams = {(r["family"], r["ref"]) for r in T}
    assert fams == {("tf_ffn_res_tp_L24", "tf_ffn_res_tp@L24"), ("tf_ffn_res_tp_L24", "gbm_res@L24"),
                    ("tf_ffn_res_tp_L24", "gbm_res_L24")}  # no ridge rows; lstm has no 72 h partner
    d = {(r["family"], r["ref"], r["h_index"]): r for r in T}
    assert abs(d[("tf_ffn_res_tp_L24", "tf_ffn_res_tp@L24", 0)]["diff"] - 0.1) < 1e-12
    assert all(r["p_holm"] >= r["p"] for r in T)


def _w24_dirs(tmp_path, n=120, shift=10, seeds=range(42, 52)):
    """72 h windows on origins [shift, n), 24 h windows on [0, n): the 24 h model is perfect, its 72 h version noisy."""
    rng = np.random.default_rng(2)
    t = np.arange(n, dtype=np.int64) * NS; y = rng.normal(3, 1, size=(n, 2)); H = np.array([1, 24])
    dst = np.where(np.arange(n)[:, None] % 5 == 0, -80.0, 0.0) * np.ones((1, 2))
    meta72 = {"A": dict(y=y[shift:], persist=y[shift:, 0], t_origin=t[shift:], horizons=H, dst_target=dst[shift:])}
    P72 = {"A": {"tf_ffn_res_tp": {s: {"pred": y[shift:] + rng.normal(0, 0.3, (n - shift, 2))} for s in seeds},
                 "gbm_res": {s: {"pred": y[shift:] + 0.2} for s in seeds}}}
    for k in P72["A"]:
        for s in seeds:
            P72["A"][k][s] = _Npz(P72["A"][k][s])
    d = tmp_path / "L24"; d.mkdir()
    np.savez(d / "meta_A.npz", y=y, persist=y[:, 0], t_origin=t, horizons=H, dst_target=dst)
    for s in seeds:
        np.savez(d / f"preds_A_tf_ffn_res_tp_L24_s{s}.npz", pred=y, pred_noise1=y + 0.1, logvar=np.zeros_like(y))
    return meta72, P72, d


class _Npz(dict):
    @property
    def files(self):
        return list(self)


def test_window24_scores_on_common_origins_and_tests_pairs(tmp_path):
    meta72, P72, d = _w24_dirs(tmp_path)
    R = ea.window24(meta72, P72, [str(d)])
    assert R["n"]["A"] == 110                                    # only the origins both window sets contain
    m = R["mean"]["tf_ffn_res_tp_L24"]
    assert np.allclose(m["pe"], 1.0) and np.allclose(m["storm_rmse"], 0.0) and np.allclose(m["hss"], 1.0)
    assert all(v < 1 for v in m["noise1"]) and np.allclose(m["cov68"], 1.0)   # |error| 0 <= sigma 1
    assert all(v < 1 for v in R["mean"]["tf_ffn_res_tp"]["pe"])
    pairs = {(r["family"], r["ref"], r["metric"]) for r in R["tests"]}
    assert ("tf_ffn_res_tp_L24", "gbm_res", "pe") in pairs and ("tf_ffn_res_tp_L24", "tf_ffn_res_tp", "storm_rmse") in pairs
    assert ("tf_ffn_res_tp_L24", "gbm_res", "noise1") not in pairs                # the reference has no probe
    assert all(r["p_holm"] >= r["p"] for r in R["tests"])
    assert all(r["diff"] < 0 for r in R["tests"] if r["metric"] == "storm_rmse")  # perfect model: lower error
    assert np.allclose(R["ensemble"]["tf_ffn_res_tp_L24"]["crps"], 0.0) and np.allclose(R["ensemble"]["tf_ffn_res_tp_L24"]["pe"], 1.0)


def test_window24_transfer_pairs_on_common_target_origins(tmp_path):
    rng = np.random.default_rng(3); n = 60; t = np.arange(n, dtype=np.int64) * NS; y = rng.normal(size=(n, 2))
    d24 = tmp_path / "t24"; d72 = tmp_path / "t72"; d24.mkdir(); d72.mkdir()
    np.savez(d24 / "transfer_meta_zs.npz", grasp_G1_y=y, grasp_G1_persist=y[:, 0], grasp_G1_t=t, horizons=np.array([1, 24]))
    np.savez(d72 / "transfer_meta_zs.npz", grasp_G1_y=y[5:], grasp_G1_persist=y[5:, 0], grasp_G1_t=t[5:], horizons=np.array([1, 24]))
    for s in range(42, 52):
        np.savez(d24 / f"transfer_preds_zs_s{s}.npz", grasp_G1_tf_ffn_res_tp_L24_zeroshot=y)
        np.savez(d72 / f"transfer_preds_zs_s{s}.npz", grasp_G1_tf_ffn_res_tp_zeroshot=y[5:] + rng.normal(0, .5, (n - 5, 2)))
    R = ea.window24_transfer([str(d72)], [str(d24)])
    assert np.allclose(R["scores"]["grasp"]["tf_ffn_res_tp_L24_zeroshot"], 1.0)
    c = R["common"]["grasp"]["tf_ffn_res_tp_L24_zeroshot|tf_ffn_res_tp_zeroshot"]
    assert np.allclose(c["model"], 1.0) and all(v < 1 for v in c["ref"])
    assert len(R["tests"]) == 2 and all(r["diff"] > 0 for r in R["tests"])
