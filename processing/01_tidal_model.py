"""
processing/01_tidal_model.py

Computes per-pixel tidal variability (sigma^2_tidal) from tide gauge data.

Strategy per site:
  sundarbans / venice : GESLA-3 CSV files → harmonic analysis → R_T + eps_T
  narrabeen           : fallback datums (Sydney Fort Denison, BOM)
  florida             : NOAA CO-OPS live API (Naples, Fort Myers, Key West)

Falls back to FALLBACK_TIDAL_DATUMS for any gauge whose GESLA-3 file
cannot be found or parsed.

Run:
    python processing/01_tidal_model.py --site sundarbans
    python processing/01_tidal_model.py --site venice
    python processing/01_tidal_model.py --site narrabeen
    python processing/01_tidal_model.py --site florida
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import requests
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

from processing.utils.constants import (
    AOI,
    TARGET_CRS,
    TARGET_RES_M,
    TIDE_GAUGES,
    FALLBACK_TIDAL_DATUMS,
    GESLA3_DIR,
    NOAA_API,
)

# ---------------------------------------------------------------------------
# Tidal constituent frequencies (rad/hr)
# ---------------------------------------------------------------------------

TIDAL_CONSTITUENTS: dict[str, float] = {
    "M2": 2 * np.pi / 12.4206,
    "S2": 2 * np.pi / 12.0000,
    "N2": 2 * np.pi / 12.6583,
    "K1": 2 * np.pi / 23.9345,
    "O1": 2 * np.pi / 25.8194,
}

REQUEST_PAUSE = 2   # seconds between NOAA API calls


# ===========================================================================
# GESLA-3 methods (Sundarbans, Venice)
# ===========================================================================

def find_gesla_file(gesla_dir: Path, keywords: list[str]) -> Path | None:
    if not gesla_dir.exists():
        return None
    for f in gesla_dir.glob("*.csv"):
        if all(kw.lower() in f.name.lower() for kw in keywords):
            return f
    for kw in keywords:
        matches = [f for f in gesla_dir.glob("*.csv") if kw.lower() in f.name.lower()]
        if matches:
            return matches[0]
    return None


def load_gesla_file(file_path: Path):
    import pandas as pd
    header_lines = 0
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                header_lines += 1
            else:
                break
    try:
        df = pd.read_csv(
            file_path, skiprows=header_lines, sep=r"\s+", header=None,
            names=["date", "time", "sea_level", "qc_flag", "use_flag"],
            dtype={"date": str, "time": str, "sea_level": float,
                   "qc_flag": float, "use_flag": float},
            na_values=["-99.9999", "9999"], on_bad_lines="skip",
        )
    except Exception as exc:
        print(f"  Failed to parse {file_path.name}: {exc}")
        return None
    try:
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"],
                                        format="%Y/%m/%d %H:%M:%S", errors="coerce")
    except Exception:
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"],
                                        infer_datetime_format=True, errors="coerce")
    df = df[(df["use_flag"] == 1) & (~df["qc_flag"].isin([3, 4, 5]))]
    df = df[df["sea_level"].notna() & df["datetime"].notna()]
    df = df[df["sea_level"].abs() < 20]
    if len(df) < 24 * 30:
        return None
    return df[["datetime", "sea_level"]].sort_values("datetime").reset_index(drop=True)


def fit_tidal_constituents(df):
    import pandas as pd
    t0 = df["datetime"].iloc[0]
    t_hours = (df["datetime"] - t0).dt.total_seconds() / 3600.0
    sl = df["sea_level"].values
    mean_sl = float(np.mean(sl))
    sl_dm = sl - mean_sl
    freqs = list(TIDAL_CONSTITUENTS.values())
    n = len(freqs)
    A = np.ones((len(t_hours), 1 + 2 * n))
    for i, w in enumerate(freqs):
        A[:, 1 + 2*i]     = np.cos(w * t_hours)
        A[:, 1 + 2*i + 1] = np.sin(w * t_hours)
    coeffs, _, _, _ = np.linalg.lstsq(A, sl_dm, rcond=None)
    amps = np.array([np.sqrt(coeffs[1+2*i]**2 + coeffs[2+2*i]**2) for i in range(n)])
    phases = np.array([np.arctan2(coeffs[2+2*i], coeffs[1+2*i]) for i in range(n)])
    return amps, phases, mean_sl


def compute_tidal_metrics_gesla(df) -> dict[str, float]:
    amps, phases, mean_sl = fit_tidal_constituents(df)
    R_T = 2.0 * float(np.sum(amps))
    t0 = df["datetime"].iloc[0]
    t_h = (df["datetime"] - t0).dt.total_seconds() / 3600.0
    freqs = list(TIDAL_CONSTITUENTS.values())
    pred = np.full(len(df), mean_sl)
    for i, w in enumerate(freqs):
        pred += amps[i] * np.cos(w * t_h - phases[i])
    eps_T = max(float(np.std(df["sea_level"].values - pred)), 0.02)
    return {"R_T": R_T, "eps_T": eps_T}


GESLA_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "sundarbans": {
        "diamond_harbour": ["diamond"],
        "haldia":          ["haldia"],
    },
    "venice": {
        "venezia_ps": ["venezia"],
        "chioggia":   ["chioggia"],
    },
}


# ===========================================================================
# NOAA CO-OPS methods (Florida)
# ===========================================================================

def fetch_noaa_datums(station_id: int, name: str) -> dict[str, float] | None:
    url = (f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/"
           f"stations/{station_id}/datums.json?units=metric&datum=NAVD")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        datums = {d["name"]: d["value"] for d in data.get("datums", [])}
        mhw  = datums.get("MHW")
        mllw = datums.get("MLLW")
        if mhw is not None and mllw is not None:
            print(f"  {name:16s} MHW={mhw:.3f}m MLLW={mllw:.3f}m range={mhw-mllw:.3f}m")
            return {"MHW": mhw, "MLLW": mllw}
    except Exception as exc:
        print(f"  NOAA datums fetch failed for {name}: {exc}")
    return None


def fetch_noaa_wl(station_id: int, name: str, product: str,
                  start: str = "20230101", end: str = "20231231") -> list[float]:
    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt   = datetime.strptime(end,   "%Y%m%d")
    values, cursor = [], start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=30), end_dt)
        params = {
            "begin_date": cursor.strftime("%Y%m%d"),
            "end_date":   chunk_end.strftime("%Y%m%d"),
            "station":    station_id, "product": product,
            "datum": "NAVD", "time_zone": "GMT", "interval": "h",
            "units": "metric", "application": "cmui_research", "format": "json",
        }
        try:
            resp = requests.get(NOAA_API, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("data") or data.get("predictions") or []
            for r in records:
                try:
                    values.append(float(r["v"]))
                except (KeyError, ValueError, TypeError):
                    pass
        except Exception as exc:
            print(f"    NOAA fetch chunk failed: {exc}")
        cursor = chunk_end + timedelta(days=1)
        time.sleep(REQUEST_PAUSE)
    return values


def compute_noaa_metrics(station_id: int, name: str,
                         fallback: dict) -> dict[str, float]:
    """Try NOAA live API; fall back to published datums + standard RMSE."""
    datums = fetch_noaa_datums(station_id, name)
    if datums:
        R_T = abs(datums["MHW"] - datums["MLLW"])
    else:
        R_T = abs(fallback["MHW"] - fallback["MLLW"])
        print(f"  {name}: using fallback R_T={R_T:.3f}m")

    # Fetch 1 year water levels for RMSE (optional — skip if API slow)
    try:
        print(f"  {name}: fetching observed water levels...")
        obs  = fetch_noaa_wl(station_id, name, "water_level")
        pred = fetch_noaa_wl(station_id, name, "predictions")
        n = min(len(obs), len(pred))
        if n >= 24:
            residuals = np.array(obs[:n]) - np.array(pred[:n])
            residuals = residuals[np.isfinite(residuals)]
            eps_T = max(float(np.std(residuals)), 0.02)
            print(f"  {name}: eps_T={eps_T:.4f}m (from {n} pairs)")
        else:
            eps_T = 0.05
            print(f"  {name}: using default eps_T=0.05m")
    except Exception:
        eps_T = 0.05
        print(f"  {name}: API error — using default eps_T=0.05m")

    return {"R_T": R_T, "eps_T": eps_T}


# ===========================================================================
# IDW interpolation and raster output
# ===========================================================================

def idw_interpolate(gauge_lons, gauge_lats, gauge_values,
                    grid_lons, grid_lats, power=2.0):
    out = np.zeros_like(grid_lons, dtype=np.float32)
    w_sum = np.zeros_like(grid_lons, dtype=np.float32)
    for lon, lat, val in zip(gauge_lons, gauge_lats, gauge_values):
        dist = np.sqrt((grid_lons - lon)**2 + (grid_lats - lat)**2)
        dist = np.maximum(dist, 1e-10)
        w = 1.0 / dist**power
        out   += w * val
        w_sum += w
    return out / w_sum


def write_raster(bands, aoi_bounds, crs_string, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = list(bands.values())
    height, width = arrays[0].shape
    transform = from_bounds(aoi_bounds[0], aoi_bounds[1],
                            aoi_bounds[2], aoi_bounds[3], width, height)
    epsg = int(crs_string.split(":")[-1])
    profile = {"driver": "GTiff", "dtype": "float32", "count": len(arrays),
               "height": height, "width": width,
               "crs": CRS.from_epsg(epsg), "transform": transform,
               "compress": "lzw"}
    with rasterio.open(output_path, "w", **profile) as dst:
        for idx, (name, arr) in enumerate(bands.items(), start=1):
            dst.write(arr.astype(np.float32), idx)
            dst.update_tags(idx, name=name)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Written: {output_path}  ({size_mb:.1f} MB)")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute sigma^2_tidal from tide gauge data"
    )
    parser.add_argument(
        "--site",
        choices=list(AOI.keys()),
        default="sundarbans",
        help="Study site",
    )
    args = parser.parse_args()
    site = args.site

    output_path = Path("D:/cmui/data/processed/tidal_variance") / f"tidal_variance_{site}.tif"
    aoi_bounds  = AOI[site]
    site_crs    = TARGET_CRS[site]
    gauge_config  = TIDE_GAUGES[site]
    fallback_data = FALLBACK_TIDAL_DATUMS[site]

    print(f"Site : {site}  |  CRS: {site_crs}  |  AOI: {aoi_bounds}")

    gauge_names, lons, lats, R_T_list, eps_list = [], [], [], [], []

    if site == "florida":
        # NOAA CO-OPS live API
        print("\nFetching tidal data from NOAA CO-OPS...")
        for gauge_name, gauge_info in gauge_config.items():
            print(f"\nGauge: {gauge_name}")
            metrics = compute_noaa_metrics(
                station_id=gauge_info["noaa_id"],
                name=gauge_name,
                fallback=fallback_data.get(gauge_name, {"MHW": 0.3, "MLLW": -0.3}),
            )
            gauge_names.append(gauge_name)
            lons.append(gauge_info["lon"])
            lats.append(gauge_info["lat"])
            R_T_list.append(metrics["R_T"])
            eps_list.append(metrics["eps_T"])

    else:
        # GESLA-3 with fallback
        gesla_dir = Path(GESLA3_DIR)
        keyword_map = GESLA_KEYWORDS.get(site, {})
        print(f"\nGESLA-3 directory: {gesla_dir}")

        for gauge_name, gauge_info in gauge_config.items():
            print(f"\nGauge: {gauge_name}  ({gauge_info['lon']}E, {gauge_info['lat']}N)")
            keywords = keyword_map.get(gauge_name, [gauge_name])
            file_path = find_gesla_file(gesla_dir, keywords)
            R_T, eps_T = None, None

            if file_path is not None:
                df = load_gesla_file(file_path)
                if df is not None:
                    try:
                        m = compute_tidal_metrics_gesla(df)
                        R_T, eps_T = m["R_T"], m["eps_T"]
                        print(f"  R_T={R_T:.3f}m  eps_T={eps_T:.4f}m")
                    except Exception as exc:
                        print(f"  Harmonic analysis failed: {exc}")

            if R_T is None:
                fb = fallback_data.get(gauge_name)
                if fb:
                    R_T   = abs(fb["MHW"] - fb["MLLW"])
                    eps_T = 0.08
                    print(f"  Fallback: R_T={R_T:.3f}m  eps_T={eps_T:.3f}m")
                else:
                    print(f"  No data for {gauge_name} — skipping")
                    continue

            gauge_names.append(gauge_name)
            lons.append(gauge_info["lon"])
            lats.append(gauge_info["lat"])
            R_T_list.append(R_T)
            eps_list.append(eps_T)

    if not gauge_names:
        raise RuntimeError(f"No gauge data for site '{site}'.")

    lons_a    = np.array(lons)
    lats_a    = np.array(lats)
    R_T_a     = np.array(R_T_list)
    eps_a     = np.array(eps_list)
    sigma2_a  = (R_T_a * eps_a) ** 2

    print(f"\nGauge summary ({len(gauge_names)} gauges):")
    for n, r, e, s in zip(gauge_names, R_T_a, eps_a, sigma2_a):
        print(f"  {n:20s}  R_T={r:.3f}m  eps_T={e:.4f}m  sigma^2={s:.6f} m^2")

    # Build output grid
    w, s, e, n = aoi_bounds
    lat_mid = (s + n) / 2
    n_rows = int(round((n - s) * 111000 / TARGET_RES_M))
    n_cols = int(round((e - w) * 111000 * np.cos(np.radians(lat_mid)) / TARGET_RES_M))
    print(f"\nOutput grid: {n_rows} rows x {n_cols} cols")

    lat_grid = np.linspace(n, s, n_rows)
    lon_grid = np.linspace(w, e, n_cols)
    grid_lons, grid_lats = np.meshgrid(lon_grid, lat_grid)

    print("Interpolating (IDW)...")
    sigma2_grid = idw_interpolate(lons_a, lats_a, sigma2_a,  grid_lons, grid_lats)
    range_grid  = idw_interpolate(lons_a, lats_a, R_T_a,    grid_lons, grid_lats)
    rmse_grid   = idw_interpolate(lons_a, lats_a, eps_a,    grid_lons, grid_lats)

    write_raster(
        bands={"tidal_variance": sigma2_grid,
               "tidal_range":    range_grid,
               "tidal_rmse":     rmse_grid},
        aoi_bounds=aoi_bounds,
        crs_string=site_crs,
        output_path=output_path,
    )

    print(f"\nDone. sigma^2_tidal saved to:\n  {output_path}")
    print("Bands: 1=tidal_variance (m^2)  2=tidal_range (m)  3=tidal_rmse (m)")
    print(f"Next step: python processing/02_zone_classifier.py --site {site}")


if __name__ == "__main__":
    main()
