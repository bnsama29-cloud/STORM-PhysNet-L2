"""GOES-16/18 download helpers and the shared transfer pieces (journal extension)."""
import io, sys
from pathlib import Path

import h5py, numpy as np, pandas as pd, pytest, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "pipeline"))
from goesr_download import omni_year_to_lst, hourly, five_min_from_nc, TELESCOPE  # noqa: E402
from src.data.cdf_reader import read_wind_txt  # noqa: E402


def omni_line(year=2024, doy=1, hour=0, bt=5.3, bz=-1.3, n=7.4, v=306.0, p=1.35, kp=7, dst=-12):
    w = ["0"] * 55
    w[0], w[1], w[2] = str(year), str(doy), str(hour)
    w[8], w[16], w[23], w[24], w[28], w[38], w[40] = str(bt), str(bz), str(n), str(v), str(p), str(kp), str(dst)
    return " ".join(w)


def test_omni_conversion_reads_back_with_fill_values(tmp_path):
    txt = omni_line() + "\n" + omni_line(hour=1, bt=999.9, bz=999.9, n=999.9, v=9999.0, p=99.99, kp=99, dst=99999) + "\n"
    f = tmp_path / "o.lst"; f.write_text(omni_year_to_lst(txt))
    df = read_wind_txt(str(f))
    assert list(df.index) == [pd.Timestamp("2024-01-01 00:00"), pd.Timestamp("2024-01-01 01:00")]
    r = df.iloc[0]
    assert (r.bt, r.bz, r.density, r.vsw, r.pdyn, r.kp, r.dst) == (5.3, -1.3, 7.4, 306.0, 1.35, 7, -12)
    assert df.iloc[1].isna().all(), f"fill values must become NaN, got {df.iloc[1].to_dict()}"


def test_omni_conversion_skips_short_lines():
    assert omni_year_to_lst("garbage line\n") == "\n"


def test_hourly_needs_six_valid_samples():
    t = pd.date_range("2024-01-01", periods=24, freq="5min")             # 12 samples in hour 0, 12 in hour 1
    s = pd.Series(np.r_[np.full(12, 100.0), np.full(5, 300.0)], index=t[:17])
    h = hourly(s)
    assert list(h.index) == [pd.Timestamp("2024-01-01 00:00")]           # hour 1 has only 5 samples
    assert h.iloc[0]["flux_gt2mev"] == 100.0 and h.iloc[0]["flux_std_1h"] == 0.0 and h.iloc[0]["n5min"] == 12


def make_nc(times, flux):
    b = io.BytesIO()
    with h5py.File(b, "w") as h:
        h["time"] = np.asarray(times, dtype=np.float64)
        h["AvgIntElectronFlux"] = np.asarray(flux, dtype=np.float32)
    return b.getvalue()


def test_five_min_reader_uses_telescope_4_and_drops_invalid():
    sec0 = (pd.Timestamp("2024-05-10 00:00") - pd.Timestamp("2000-01-01 12:00")).total_seconds()
    flux = np.ones((4, 5)) * 7.0
    flux[:, TELESCOPE] = [10.0, -1e31, 0.0, 40.0]                        # fill and non-positive values are dropped
    s = five_min_from_nc(make_nc([sec0, sec0 + 300, sec0 + 600, sec0 + 900], flux))
    assert TELESCOPE == 3
    assert list(s.values) == [10.0, 40.0]
    assert list(s.index) == [pd.Timestamp("2024-05-10 00:00"), pd.Timestamp("2024-05-10 00:15")]


def test_transfer_horizon_weights_and_heads_freezing():
    from storm_pipeline import transfer as T
    from storm_pipeline.models_extra import FFNResTransformer
    assert T.horizon_weights("cpu").tolist() == pytest.approx([1.0, 0.7, 0.5])     # default horizons 1, 6, 12
    m = FFNResTransformer(17, n_h=3); T.freeze_heads_only(m, "tf_ffn_res")
    trainable = sorted(n for n, p in m.named_parameters() if p.requires_grad)
    assert trainable == ["fc.bias", "fc.weight"]


def test_transfer_train_returns_best_validation_state():
    from storm_pipeline import transfer as T
    from storm_pipeline.models_extra import FFNResTransformer
    torch.manual_seed(0)

    class DS(torch.utils.data.Dataset):
        def __len__(self): return 128
        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return {"x_sw": torch.randn(72, 17, generator=g), "x_flux": torch.randn(72, 1, generator=g),
                    "y_persist": torch.zeros(3), "y_flux": torch.full((3,), 0.5)}
    m = FFNResTransformer(17, n_h=3)
    sets = {"train": DS(), "val": DS()}
    before = T.wmse(torch.from_numpy(T.predict(m, sets["val"], "cpu")), torch.full((128, 3), 0.5), T.horizon_weights("cpu")).item()
    m, ep, best = T.train(m, sets, 1e-3, 0, list(m.parameters()), "cpu", epochs=3, patience=2)
    after = T.wmse(torch.from_numpy(T.predict(m, sets["val"], "cpu")), torch.full((128, 3), 0.5), T.horizon_weights("cpu")).item()
    assert 1 <= ep <= 3 and after == pytest.approx(best, rel=1e-4) and after < before
