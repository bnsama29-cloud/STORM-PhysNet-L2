"""Classical baselines: features use only the input window; probes degrade only the intended channels."""
import sys
from pathlib import Path

import numpy as np, pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "pipeline"))
from classical_baselines import features, apply_probe, Ridge, N_DRV, I_MASK, I_BZ  # noqa: E402
from storm_pipeline.data import SW_COLS  # noqa: E402


def _win(n=3, T=72, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T, len(SW_COLS))).astype(np.float32), rng.normal(size=(n, T)).astype(np.float32)


def test_feature_layout_and_values():
    X, F = _win()
    f = features(X, F)
    assert f.shape == (3, 75)
    assert np.allclose(f[:, 0], F[:, -1]) and np.allclose(f[:, 23], F[:, -24])          # y_t ... y_{t-23}
    assert np.allclose(f[:, 24], F[:, -24:].mean(1)) and np.allclose(f[:, 25], F.mean(1))
    assert np.allclose(f[:, -1], X[:, -24:, I_BZ].min(1))


def test_features_ignore_hours_before_the_window_summaries():
    """Changing hours older than 72 h is impossible by construction; changing hour t-30 only moves the 72 h means."""
    X, F = _win(); f0 = features(X, F)
    X2, F2 = X.copy(), F.copy(); F2[:, -31] += 5.0; X2[:, -31, :N_DRV] += 5.0
    d = np.abs(features(X2, F2) - f0).max(0) > 1e-6
    assert d.sum() == 1 and d[25]                                                      # only the 72 h flux mean


def test_noise_probe_leaves_mask_and_flux_history_untouched():
    X, _ = _win()
    Xn = apply_probe(X, ("noise", 1.0), np.random.default_rng(0))
    assert np.array_equal(Xn[..., I_MASK], X[..., I_MASK])
    assert not np.allclose(Xn[..., :I_MASK], X[..., :I_MASK])


def test_outage_probes_hold_or_zero_only_the_drivers():
    X, _ = _win()
    Xo = apply_probe(X, ("out", 24), None)
    assert np.allclose(Xo[:, -24:, :N_DRV], X[:, -25:-24, :N_DRV]) and np.array_equal(Xo[..., N_DRV:], X[..., N_DRV:])
    Xz = apply_probe(X, ("out", 72), None)
    assert np.all(Xz[..., :N_DRV] == 0) and np.array_equal(Xz[..., N_DRV:], X[..., N_DRV:])


def test_ridge_recovers_a_linear_change():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(500, 5)).astype(np.float32); y = A @ np.array([1., -2., 0., 0.5, 0.]) + 0.3
    m = Ridge().fit(A[:400], y[:400], A[400:], y[400:], np.ones(400), 42)
    assert np.allclose(m.predict(A[400:]), y[400:], atol=0.02)


def test_exported_ridge_predicts_like_the_fitted_model():
    from classical_baselines import export, predict_exported
    rng = np.random.default_rng(2)
    A = rng.normal(size=(300, 4)).astype(np.float32); y = A[:, 0] - A[:, 2]
    m = Ridge().fit(A[:200], y[:200], A[200:], y[200:], np.ones(200), 42)
    assert np.allclose(predict_exported(export(m), A[200:]), m.predict(A[200:]))
