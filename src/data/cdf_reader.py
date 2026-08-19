"""
CDF Reader for GOES electron flux and Wind/ACE solar wind data.
Handles real CDF files from CDAWeb archives.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import warnings

try:
    import cdflib
    CDF_AVAILABLE = True
except ImportError as e:
    print(f'CRITICAL CDFLIB ERROR: {e}')
    CDF_AVAILABLE = False
    warnings.warn("cdflib not installed. Real CDF reading unavailable. Use synthetic data.")


# ─────────────────────────────────────────────────────────────────────────────
# Variable name mappings for different GOES satellites
# ─────────────────────────────────────────────────────────────────────────────
GOES_VAR_MAP = {
    # GOES-15 and earlier
    "GOES-15": {
        "time": "Epoch",
        "flux_gt2mev": "E2W_COR_FLUX",
        "quality_flag": "E1_QUAL_FLAG",
    },
    # GOES-16/17 (GOES-R series)
    "GOES-16": {
        "time": "L2_SciData_TimeStamp",
        "flux_gt2mev": "AvgDiffProtonFlux",  # adapt based on actual GOES-R variable
        "quality_flag": "DataQuality",
    },
}

WIND_VAR_MAP = {
    "time": "Epoch",
    "vsw": "Proton_V_nonlin",          # solar wind speed (km/s)
    "density": "Proton_Np_nonlin",     # proton density (cm^-3)
    "bx": "BGSE_0",                    # IMF Bx (nT)
    "by": "BGSE_1",                    # IMF By (nT)
    "bz": "BGSE_2",                    # IMF Bz (nT)
}

ACE_VAR_MAP = {
    "time": "Epoch",
    "vsw": "V_GSE",
    "density": "Np",
    "bx": "BGSEc_0",
    "by": "BGSEc_1",
    "bz": "BGSEc_2",
}


def _cdf_epoch_to_datetime(epoch):
    """Convert CDF Epoch (milliseconds since J2000) to pandas DatetimeIndex.
    Handles both old cdflib (to_np kwarg) and new cdflib (no kwarg) APIs.
    """
    if not CDF_AVAILABLE:
        raise RuntimeError("cdflib not available")
    try:
        # cdflib >= 1.0: to_np argument removed
        times = cdflib.epochs.CDFepoch.to_datetime(epoch)
    except TypeError:
        # cdflib < 1.0: fallback with to_np
        times = cdflib.epochs.CDFepoch.to_datetime(epoch, to_np=True)
    return pd.DatetimeIndex(times)


def read_goes_cdf(file_path: str, satellite: str = "GOES-15") -> pd.DataFrame:
    """
    Read GOES electron flux data from a CDF file.

    Parameters
    ----------
    file_path : str
        Path to the GOES CDF file.
    satellite : str
        Satellite identifier ('GOES-15', 'GOES-16', etc.)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['time', 'flux_gt2mev', 'quality_flag']
    """
    if not CDF_AVAILABLE:
        raise RuntimeError("cdflib not installed. Cannot read CDF files.")

    var_map = GOES_VAR_MAP.get(satellite, GOES_VAR_MAP["GOES-15"])

    cdf = cdflib.CDF(file_path)
    info = cdf.cdf_info()

    # Extract time
    epoch = cdf.varget(var_map["time"])
    times = _cdf_epoch_to_datetime(epoch)

    # Extract flux
    flux = cdf.varget(var_map["flux_gt2mev"])
    if flux.ndim > 1:
        flux = flux[:, 0]  # take first energy channel if multiple

    # Extract quality flag
    try:
        quality = cdf.varget(var_map["quality_flag"])
    except Exception:
        quality = np.zeros(len(times), dtype=int)

    df = pd.DataFrame({
        "time": times,
        "flux_gt2mev": flux,
        "quality_flag": quality,
    })
    df.set_index("time", inplace=True)

    # Mask bad quality data
    bad_mask = (df["quality_flag"] > 0) | (df["flux_gt2mev"] <= 0)
    df.loc[bad_mask, "flux_gt2mev"] = np.nan

    return df[["flux_gt2mev"]]


def read_goes_directory(directory: str, satellite: str = "GOES-15",
                        resample_freq: str = "1h") -> pd.DataFrame:
    """
    Read and concatenate all GOES CDF files in a directory, then resample
    to hourly resolution.

    IMPORTANT FIX: this function used to return the raw 1-minute-cadence
    dataframe unchanged. Every caller in this project then did
    `goes_df.join(wind_df, how="inner")` against hourly OMNI data — an
    inner join on mismatched frequencies keeps only rows where both
    indices exactly coincide, which for 1-minute-vs-hourly data means only
    the single sample exactly on each hour boundary survives; the other 59
    minutes of GOES data per hour were silently discarded (verified
    directly with a synthetic 1-min/hourly join: 180 minute rows joined
    against 3 hourly rows produced exactly 3 output rows, not a mean of
    60 each). This function now does the resampling itself, using a
    genuine hourly MEAN of the 1-minute flux, so every caller downstream
    gets the intended smoothed hourly value rather than one noisy 1-minute
    snapshot per hour.

    Also computes `flux_std_1h`: the standard deviation of the 1-minute
    flux samples within each hour. This is real signal that a plain
    hourly mean discards — short-timescale variability is physically
    meaningful (a turbulent, rapidly-changing hour vs. a quiet stable one)
    and, as a side benefit, spikes during GOES-15's known yaw-flip data
    dropout periods (documented in the literature and independently
    verified in this project's own GOES-15 file — e.g. near-total NaN
    fraction on 2015-03-17), giving the model and any later analysis a
    concrete signal for data-quality-degraded hours.

    Parameters
    ----------
    directory : str
        Path to directory containing GOES CDF files.
    satellite : str
        Satellite identifier.
    resample_freq : str
        Target resolution for the returned dataframe (default hourly,
        matching the OMNI/Wind data this gets joined against).

    Returns
    -------
    pd.DataFrame
        Hourly-resampled dataframe with columns ['flux_gt2mev', 'flux_std_1h'].
    """
    cdf_dir = Path(directory)
    files = sorted(cdf_dir.glob("*.cdf")) + sorted(cdf_dir.glob("*.CDF"))

    if not files:
        raise FileNotFoundError(f"No CDF files found in {directory}")

    dfs = []
    for f in files:
        try:
            df = read_goes_cdf(str(f), satellite=satellite)
            dfs.append(df)
        except Exception as e:
            warnings.warn(f"Failed to read {f}: {e}")

    if not dfs:
        raise RuntimeError("No CDF files could be read successfully.")

    combined = pd.concat(dfs)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.sort_index(inplace=True)

    resampled = combined.resample(resample_freq).agg(
        flux_gt2mev=("flux_gt2mev", "mean"),
        flux_std_1h=("flux_gt2mev", "std"),
    )
    n_before, n_after = len(combined), len(resampled)
    print(f"[GOES] Resampled {n_before:,} raw 1-minute-cadence samples -> "
          f"{n_after:,} rows at '{resample_freq}' resolution "
          f"(mean + within-hour std)")
    return resampled


def read_wind_cdf(file_path: str) -> pd.DataFrame:
    """
    Read Wind spacecraft solar wind data from a CDF file.
    Combines SWE (plasma) and MFI (magnetic field) data.

    Parameters
    ----------
    file_path : str
        Path to the Wind CDF file.

    Returns
    -------
    pd.DataFrame
        DataFrame with solar wind parameters.
    """
    if not CDF_AVAILABLE:
        raise RuntimeError("cdflib not installed.")

    cdf = cdflib.CDF(file_path)
    var_names = [v["Variable"] for v in cdf.cdf_info()["rVariables"]] + \
                [v["Variable"] for v in cdf.cdf_info()["zVariables"]]

    data = {}
    epoch = cdf.varget("Epoch")
    times = _cdf_epoch_to_datetime(epoch)

    for key, var in WIND_VAR_MAP.items():
        if var in var_names and key != "time":
            try:
                vals = cdf.varget(var)
                if vals.ndim > 1:
                    vals = vals.mean(axis=1)
                data[key] = vals
            except Exception:
                pass

    df = pd.DataFrame(data, index=times)

    # Compute derived quantities
    if "bx" in df and "by" in df and "bz" in df:
        df["bt"] = np.sqrt(df["bx"]**2 + df["by"]**2 + df["bz"]**2)

    if "vsw" in df and "density" in df:
        # Dynamic pressure (nPa): 0.5 * m_p * n * v^2, with unit conversions
        mp = 1.6726e-27   # proton mass (kg)
        df["pdyn"] = 0.5 * mp * (df["density"] * 1e6) * (df["vsw"] * 1e3)**2 * 1e9

    # Mask fill values
    for col in df.columns:
        df[col] = df[col].where(df[col].abs() < 9e30, np.nan)

    return df


def read_wind_txt(file_path: str) -> pd.DataFrame:
    """
    Read OMNI/Wind solar wind and geomagnetic data from a .lst file.
    Columns: YEAR, DOY, Hour, bt, bz, density, vsw, pdyn, kp, dst
    """
    names = ["year", "doy", "hour", "bt", "bz", "density", "vsw", "pdyn", "kp", "dst"]
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=names, on_bad_lines="skip")
    
    # Calculate datetime
    # DOY 1 is Jan 1. We add DOY-1 days and hour hours
    df["time"] = pd.to_datetime(df["year"], format="%Y") + \
                 pd.to_timedelta(df["doy"] - 1, unit="D") + \
                 pd.to_timedelta(df["hour"], unit="h")
                 
    df.set_index("time", inplace=True)
    df.drop(columns=["year", "doy", "hour"], inplace=True)
    
    # Replace OMNI missing values (e.g. 999.9 or 9999) with NaN
    df.replace({
        "bt": {999.9: np.nan},
        "bz": {999.9: np.nan},
        "density": {999.9: np.nan},
        "vsw": {9999.0: np.nan},
        "pdyn": {99.99: np.nan, 999.99: np.nan},
        "kp": {99: np.nan},
        "dst": {99999: np.nan}
    }, inplace=True)
    
    return df


def read_wind_directory(directory: str) -> pd.DataFrame:
    """
    Read and concatenate all Wind/OMNI files in a directory (.cdf, .lst, .txt).
    """
    wind_dir = Path(directory)
    files = sorted(list(wind_dir.glob("*.cdf")) + list(wind_dir.glob("*.CDF")) + 
                   list(wind_dir.glob("*.lst")) + list(wind_dir.glob("*.txt")))
# skip format/readme files
    files = [f for f in files if f.suffix.lower() != ".fmt" and "fmt" not in f.name.lower()]

    if not files:
        raise FileNotFoundError(f"No valid Wind/OMNI data found in {directory}")

    dfs = []
    for f in files:
        try:
            if f.suffix.lower() in [".lst", ".txt"]:
                df = read_wind_txt(str(f))
            else:
                df = read_wind_cdf(str(f))
            dfs.append(df)
        except Exception as e:
            warnings.warn(f"Failed to read {f}: {e}")

    combined = pd.concat(dfs)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.sort_index(inplace=True)
    return combined


import re

def read_grasp_txt(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    m = re.search(r"(20\d{2})", path.stem)
    year = int(m.group(1)) if m else 2018

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        skiprows=1,
        header=None,
        names=["doy", "electron_flux", "proton_flux", "extra", "extra2"],
        on_bad_lines="skip",
        engine="python",
    )
    if df.empty:
        return pd.DataFrame(columns=["flux_gt2mev"])

    start_of_year = pd.Timestamp(year=year, month=1, day=1)
    df["time"] = start_of_year + pd.to_timedelta(df["doy"] - 1, unit="D")
    df = df.set_index("time")

    df["flux_gt2mev"] = pd.to_numeric(df["electron_flux"], errors="coerce")
    df["flux_gt2mev"] = df["flux_gt2mev"].where(df["flux_gt2mev"] > 0, np.nan)
    return df[["flux_gt2mev"]]

def read_grasp_directory(directory: str) -> pd.DataFrame:
    """
    Read and concatenate all GRASP .txt files in a directory.
    """
    grasp_dir = Path(directory)
    files = sorted(grasp_dir.glob("*.txt"))
    
    if not files:
        # Return empty dataframe if no GRASP data downloaded yet
        return pd.DataFrame(columns=["flux_gt2mev"])
        
    dfs = []
    for f in files:
        try:
            df = read_grasp_txt(str(f))
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            warnings.warn(f"Failed to read GRASP file {f}: {e}")
            
    if not dfs:
        return pd.DataFrame(columns=["flux_gt2mev"])
        
    combined = pd.concat(dfs)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.sort_index(inplace=True)
    return combined


def merge_goes_wind(
    goes_df: pd.DataFrame,
    wind_df: pd.DataFrame,
    resample_freq: str = "1h",
    propagation_delay_min: int = 60,
) -> pd.DataFrame:
    """
    DEPRECATED / unused by current notebooks.

    Prefer: read_goes_directory (already hourly mean + flux_std_1h) then
    goes.join(wind, how="inner"). Calling this on already-hourly GOES can
    drop flux_std_1h if only .mean() is applied. Propagation delay is handled
    by AdaptivePropagationDelay in the model, not here.

    Merge GOES and Wind data at a common time resolution.
    Applies propagation delay to solar wind data (L1 → GEO).

    Parameters
    ----------
    goes_df : pd.DataFrame
        GOES electron flux.
    wind_df : pd.DataFrame
        Wind solar wind parameters.
    resample_freq : str
        Target time resolution (default '1H' for hourly).
    propagation_delay_min : int
        Solar wind propagation delay from L1 to GEO (minutes).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame at target resolution.
    """
    # Shift solar wind by propagation delay
    delay = pd.Timedelta(minutes=propagation_delay_min)
    wind_shifted = wind_df.copy()
    wind_shifted.index = wind_df.index + delay

    # Resample to common hourly resolution
    goes_resampled = goes_df.resample(resample_freq).mean()
    wind_resampled = wind_shifted.resample(resample_freq).mean()

    merged = goes_resampled.join(wind_resampled, how="inner")
    return merged
