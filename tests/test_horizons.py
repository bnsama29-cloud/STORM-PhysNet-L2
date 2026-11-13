"""Configurable forecast horizons (journal extension: 24 h).

The horizon set is read from STORM_HSTEPS when storm_pipeline.data is imported, so each case runs in a fresh
interpreter. The default must reproduce the AGERS 2026 window selection exactly."""
import json, os, subprocess, sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "pipeline"))


def run(code, hsteps=None):
    env = dict(os.environ); env.pop("STORM_HSTEPS", None)
    if hsteps is not None: env["STORM_HSTEPS"] = hsteps
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT / "pipeline", env=env, capture_output=True, text=True)
    return r


PREP = """
import sys, json, numpy as np
sys.path.insert(0, '..')
from storm_pipeline import data as D
rng = np.random.default_rng(0)
n = 2000
lab = np.zeros(n, dtype=np.int8); lab[800:1300] = 1; lab[1300:] = 2
obs = (rng.random(n) > 0.05).astype(float)
"""


def test_default_horizons_match_agers_window_rule():
    code = PREP + """
s, _ = D.valid_starts(lab, obs)
# the AGERS 2026 rule, written out explicitly
SEQ, SPAN = 72, 84
st = np.arange(0, n - SPAN + 1); ok = []
for x in st:
    same = lab[x:x + SPAN].min() == lab[x:x + SPAN].max()
    ok.append(same and obs[x + 71] > 0 and obs[x + 72] > 0 and obs[x + 77] > 0 and obs[x + 83] > 0)
ref = st[np.array(ok)]
print(json.dumps(dict(hsteps=D.HSTEPS, span=D.SPAN, equal=bool(np.array_equal(s, ref)), n=int(len(s)))))
"""
    r = run(code); assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["hsteps"] == [1, 6, 12] and out["span"] == 84
    assert out["equal"] is True and out["n"] > 100


def test_24h_horizon_windows_and_items():
    code = PREP + """
s, _ = D.valid_starts(lab, obs)
SPAN = 96
bad = [x for x in s if lab[x:x + SPAN].min() != lab[x:x + SPAN].max() or not all(obs[x + 72 + h - 1] > 0 for h in (1, 6, 12, 24)) or obs[x + 71] == 0]
print(json.dumps(dict(hsteps=D.HSTEPS, span=D.SPAN, n=int(len(s)), bad=len(bad), last=int(s.max()))))
"""
    r = run(code, "1,6,12,24"); assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["hsteps"] == [1, 6, 12, 24] and out["span"] == 96
    assert out["bad"] == 0 and out["n"] > 100 and out["last"] <= 2000 - 96


def test_bad_horizon_list_is_rejected():
    r = run("import sys; sys.path.insert(0, '..'); from storm_pipeline import data", "12,6")
    assert r.returncode != 0 and "bad STORM_HSTEPS" in r.stderr


def test_physics_loss_with_four_horizons():
    from src.training.physics_loss import PhysicsInformedLoss, LossWeights
    w = LossWeights(horizon_weights=[1.0, 0.7, 0.5, 0.4], physics_horizon_scale=[1.0, 0, 0, 0], horizon_hours=[1, 6, 12, 24])
    fn = PhysicsInformedLoss(w)
    B, H = 8, 4
    torch.manual_seed(0)
    out = {"flux_pred": torch.randn(B, H), "flux_residual": torch.randn(B, H), "log_var": torch.zeros(B, H)}
    tg = {"y_flux": torch.randn(B, H), "y_dst": torch.zeros(B, H), "y_kp": torch.zeros(B, H), "y_storm": torch.zeros(B, 1),
          "storm_flag": torch.zeros(B, 1), "y_persist": torch.randn(B, H)}
    total, comps = fn(out, tg, torch.randn(B, 72, 17))
    assert torch.isfinite(total) and set(comps) >= {"primary", "monotone", "total"}
    assert fn.horizon_hours.tolist() == [1.0, 6.0, 12.0, 24.0]


def test_physics_loss_rejects_mismatched_horizon_hours():
    from src.training.physics_loss import PhysicsInformedLoss, LossWeights
    fn = PhysicsInformedLoss(LossWeights())                      # default hours 1, 6, 12
    with pytest.raises(AssertionError, match="one entry per forecast horizon"):
        fn._monotonicity_loss(torch.randn(4, 4), torch.randn(4, 72, 17))


def test_default_loss_weights_unchanged():
    from src.training.physics_loss import LossWeights
    w = LossWeights()
    assert w.horizon_weights == [1.0, 0.7, 0.5] and w.physics_horizon_scale == [1.0, 0.0, 0.0] and w.horizon_hours == [1.0, 6.0, 12.0]


def test_residual_transformer_four_outputs():
    from storm_pipeline.models_extra import FFNResTransformer, SmoothedInput
    m = SmoothedInput(FFNResTransformer(17, n_h=4)).eval()
    o = m(torch.randn(2, 72, 17), torch.randn(2, 72, 1), y_persist=torch.zeros(2, 4))
    assert tuple(o["flux_pred"].shape) == (2, 4)


def test_time_budget_stops_before_any_new_run(tmp_path):
    """--stop_after_hours 0 must exit cleanly (code 0) without training, so a Kaggle session can end and resume."""
    r = subprocess.run([sys.executable, "train_block.py", "--repo", "..", "--block", "B", "--out", str(tmp_path),
                        "--families", "lstm", "--seeds", "42", "--stop_after_hours", "0", "--horizons", "1,6,12,24"],
                       cwd=ROOT / "pipeline", capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "TIME BUDGET REACHED" in r.stdout
    assert not list(tmp_path.glob("preds_*.npz")) and (tmp_path / "meta_B.npz").exists()
