"""Metrics used in the journal tables."""
import sys
from pathlib import Path

import numpy as np, pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from journal_analysis import pe_clim, pe_pers, hss, pod_far, crps_gauss, crps_ensemble, holm  # noqa: E402


def test_pe_clim_and_pers():
    y = np.array([[0.0], [2.0]]); assert pe_clim(y, y)[0] == 1.0
    assert pe_clim(y, np.full_like(y, 1.0))[0] == pytest.approx(0.0)            # predicting the mean: no skill
    assert pe_pers(y, y, np.array([1.0, 1.0]))[0] == 1.0
    assert pe_pers(y, np.full_like(y, 1.0), np.array([1.0, 1.0]))[0] == pytest.approx(0.0)


def test_hss_perfect_random_and_empty():
    o = np.array([1, 1, 0, 0, 1, 0], bool)
    assert hss(o, o) == pytest.approx(1.0)
    assert hss(o, ~o) == pytest.approx(-1.0)
    assert hss(np.zeros(4, bool), np.zeros(4, bool)) == 0.0                     # no events at all: defined as 0
    pod, far = pod_far(o, np.array([1, 0, 0, 1, 1, 0], bool))
    assert pod == pytest.approx(2 / 3) and far == pytest.approx(1 / 3)


def test_crps_gauss_known_values():
    # CRPS of N(0,1) at 0 is (sqrt(2)-1)/sqrt(pi)
    assert crps_gauss(np.array(0.0), np.array(0.0), np.array(1.0)) == pytest.approx((np.sqrt(2) - 1) / np.sqrt(np.pi))
    # sharper correct forecasts score lower
    assert crps_gauss(np.array(0.0), np.array(0.0), np.array(0.1)) < crps_gauss(np.array(0.0), np.array(0.0), np.array(1.0))


def test_crps_ensemble_reduces_to_abs_error_for_one_member():
    y = np.array([[1.0, 2.0]]); ens = np.array([[[0.0, 4.0]]])
    assert crps_ensemble(y, ens).tolist() == [[1.0, 2.0]]


def test_holm_matches_hand_calculation():
    adj = holm([0.01, 0.04, 0.03])
    assert adj.tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_gaps_breaks_lines_only_at_long_jumps():
    import pandas as pd
    from journal_figures import gaps
    t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 05:00", "2024-01-01 06:00"])
    tt, (y,) = gaps(t, np.array([1.0, 2.0, 3.0, 4.0]))
    assert len(tt) == 5 and np.isnan(y[2]) and y[[0, 1, 3, 4]].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_paired_against_lstm_reference_skips_missing_families():
    import numpy as np
    from journal_analysis import paired
    rng = np.random.default_rng(0)
    ref = rng.normal(0.8, 0.01, (10, 2))
    S = {"storm_lstm_backbone": ref, "storm_lstm_no_gate": ref - 0.05}
    rows = paired(S, "storm_lstm_backbone", ["storm_lstm_no_gate", "lstm_res"], [1, 6])
    assert [r["family"] for r in rows] == ["storm_lstm_no_gate", "storm_lstm_no_gate"]
    assert rows[0]["diff"] == pytest.approx(0.05)
    assert paired({}, "storm_lstm_backbone", ["lstm_res"], [1]) == []


def test_load_blocks_rejects_mismatched_test_windows(tmp_path):
    from journal_analysis import load_blocks
    for d, y in (("a", [[1.0, 2.0]]), ("b", [[1.0, 2.5]])):
        (tmp_path / d).mkdir(); np.savez(tmp_path / d / "meta_A.npz", y=np.array(y))
    with pytest.raises(AssertionError, match="test targets differ"):
        load_blocks([tmp_path / "a", tmp_path / "b"])
    (tmp_path / "c").mkdir(); np.savez(tmp_path / "c" / "meta_A.npz", y=np.array([[1.0, 2.0]]))
    meta, _ = load_blocks([tmp_path / "a", tmp_path / "c"])
    assert meta["A"]["y"].tolist() == [[1.0, 2.0]]
