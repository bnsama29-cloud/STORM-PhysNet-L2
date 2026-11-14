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
