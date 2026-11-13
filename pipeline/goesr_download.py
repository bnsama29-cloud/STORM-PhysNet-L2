#!/usr/bin/env python
"""Download NOAA GOES-16/18 MPS-HI >2 MeV electron fluxes and OMNI2 hourly drivers for 2021 onwards.

  python goesr_download.py --out ../datasets            (needs internet; about 1.5 GB is streamed, ~3 MB is kept)

GOES-R:  NCEI L2 5-minute averages (mpsh-l2-avg5m_science, v2), variable AvgIntElectronFlux (E11, >2 MeV integral,
         electrons cm-2 s-1 sr-1, the same unit as the GOES-15 >2 MeV channel). We use telescope 4: the middle of the
         five-telescope fan (north-south order 3, 1, 4, 2, 5), i.e. the one closest to the equatorial plane, also when
         the spacecraft is yaw-flipped. Fill values and non-positive fluxes are dropped. An hour is kept when at least
         6 of its 12 five-minute samples are valid (the GRASP rule); we store the hourly mean, within-hour std and count.
         -> goesr/goes16_e2mev_hourly.csv, goesr/goes18_e2mev_hourly.csv
OMNI2:   SPDF low-resolution yearly files (omni2_YYYY.dat), rewritten in the 10-column OMNIWeb layout read by
         src.data.cdf_reader.read_wind_txt (year doy hour |B| Bz_GSM N V Pdyn Kp*10 Dst), with the same fill values.
         -> omni/omni2_2021_2026.lst
"""
import argparse, io, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np, pandas as pd

BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/{sat}/l2/data/mpsh-l2-avg5m_science/"
OMNI = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
TELESCOPE = 3                    # 0-based index of telescope 4
FILL = -1e30
EPOCH = pd.Timestamp("2000-01-01 12:00:00")
# OMNI2 word numbers (1-based) -> our columns, and the OMNI2 fill value of each
OMNI_WORDS = {"bt": (9, 999.9), "bz": (17, 999.9), "density": (24, 999.9), "vsw": (25, 9999.0), "pdyn": (29, 99.99),
              "kp": (39, 99), "dst": (41, 99999)}


def get(url, tries=4, timeout=120):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout).read()
        except Exception:
            if k == tries - 1: raise
            time.sleep(3 * (k + 1))


def links(url):
    return [h for h in re.findall(r'href="([^"?#]+)"', get(url).decode("utf-8", "ignore")) if not h.startswith(("/", "http", ".."))]


def five_min_from_nc(raw: bytes) -> pd.Series:
    """One daily L2 file -> >2 MeV flux of telescope 4 at 5-min cadence (NaN where invalid)."""
    import h5py
    with h5py.File(io.BytesIO(raw), "r") as h:
        t = h["time"][:]; f = h["AvgIntElectronFlux"][:, TELESCOPE].astype(np.float64)
    ok = (t > FILL / 10) & np.isfinite(f) & (f > 0) & (f > FILL / 10)
    idx = EPOCH + pd.to_timedelta(t[ok], unit="s")
    return pd.Series(f[ok], index=idx)


def hourly(s: pd.Series, min_count=6) -> pd.DataFrame:
    g = s.resample("1h").agg(["mean", "std", "count"])
    g = g[g["count"] >= min_count]
    return g.rename(columns={"mean": "flux_gt2mev", "std": "flux_std_1h", "count": "n5min"})


def goesr(sat, years, workers=8):
    base = BASE.format(sat=sat); urls = []
    for y in [h for h in links(base) if h.strip("/").isdigit() and int(h.strip("/")) in years]:
        for m in [h for h in links(base + y) if h.strip("/").isdigit()]:
            urls += [base + y + m + f for f in links(base + y + m) if f.endswith(".nc")]
    urls = sorted(urls); print(sat, len(urls), "daily files", flush=True)

    def one(u):
        try: return five_min_from_nc(get(u))
        except Exception as e: print("  failed", u, e, flush=True); return None
    parts, t0 = [], time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, s in enumerate(ex.map(one, urls)):
            if s is not None and len(s): parts.append(s)
            if i % 200 == 0: print(f"  {sat} {i}/{len(urls)} {time.time() - t0:.0f}s", flush=True)
    s = pd.concat(parts).sort_index(); s = s[~s.index.duplicated()]
    return hourly(s)


def omni_year_to_lst(text: str) -> str:
    """OMNI2 yearly .dat text -> the 10-column OMNIWeb .lst layout used in datasets/omni."""
    out = []
    for line in text.splitlines():
        w = line.split()
        if len(w) < 41: continue
        v = {k: float(w[i - 1]) for k, (i, _) in OMNI_WORDS.items()}
        out.append(f"{int(w[0]):4d}{int(w[1]):4d}{int(w[2]):3d}{v['bt']:6.1f}{v['bz']:6.1f}{v['density']:6.1f}"
                   f"{v['vsw']:5.0f}.{v['pdyn']:6.2f}{int(v['kp']):3d}{int(v['dst']):6d}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--years", default="2017-2026"); ap.add_argument("--omni_years", default="2021-2026")
    a = ap.parse_args(); out = Path(a.out)
    y0, y1 = map(int, a.years.split("-")); o0, o1 = map(int, a.omni_years.split("-"))
    (out / "goesr").mkdir(parents=True, exist_ok=True); (out / "omni").mkdir(parents=True, exist_ok=True)
    txt = "".join(omni_year_to_lst(get(OMNI.format(year=y)).decode()) for y in range(o0, o1 + 1))
    (out / f"omni/omni2_{o0}_{o1}.lst").write_text(txt); print("OMNI rows", txt.count("\n"))
    for sat in ("goes16", "goes18"):
        h = goesr(sat, set(range(y0, y1 + 1)))
        h.to_csv(out / f"goesr/{sat}_e2mev_hourly.csv", float_format="%.6g", index_label="time")
        print(sat, len(h), "hours", h.index.min(), h.index.max())


if __name__ == "__main__":
    main()
