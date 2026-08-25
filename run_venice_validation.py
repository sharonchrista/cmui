"""
run_venice_validation.py

Venice CMUI validation — no pyproj dependency.
Uses pure numpy UTM projection (WGS84 -> EPSG:32632).

Run from D:\\cmui:
    python run_venice_validation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import rasterio
from scipy import stats
from sklearn.metrics import mean_squared_error

CMUI_PATH = Path("D:/cmui/outputs/cmui_rasters/cmui_venice.tif")
ZONE_PATH = Path("D:/cmui/data/processed/zone_maps/zone_map_venice.tif")
GT_PATH   = Path("D:/cmui/data/raw/groundtruth/ground_truth_venice.csv")
OUT_DIR   = Path("D:/cmui/outputs/validation")

ZONE_NAMES = {
    0: "unclassified", 1: "sandy_beach", 2: "rocky_shore",
    3: "mangrove",     4: "tidal_flat",  5: "coral_reef",
}


def wgs84_to_utm32n(lons: np.ndarray, lats: np.ndarray):
    """
    Pure numpy WGS84 -> UTM Zone 32N (EPSG:32632).
    Accurate to <1m for Venice area. No external dependencies.
    """
    a   = 6378137.0
    f   = 1 / 298.257223563
    b   = a * (1 - f)
    e2  = 1 - (b / a) ** 2
    lon0    = np.radians(9.0)   # central meridian UTM Zone 32
    k0      = 0.9996
    E0      = 500000.0

    phi = np.radians(lats)
    lam = np.radians(lons)

    N   = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    T   = np.tan(phi) ** 2
    C   = (e2 / (1 - e2)) * np.cos(phi) ** 2
    A_  = np.cos(phi) * (lam - lon0)

    e4 = e2 ** 2
    e6 = e2 ** 3
    M   = a * (
        (1 - e2/4 - 3*e4/64 - 5*e6/256)  * phi
      - (3*e2/8 + 3*e4/32 + 45*e6/1024)  * np.sin(2*phi)
      + (15*e4/256 + 45*e6/1024)          * np.sin(4*phi)
      - (35*e6/3072)                       * np.sin(6*phi)
    )

    xs = k0 * N * (
        A_
        + (1 - T + C)          * A_**3 / 6
        + (5 - 18*T + T**2 + 72*C - 58*(e2/(1-e2))) * A_**5 / 120
    ) + E0

    ys = k0 * (
        M + N * np.tan(phi) * (
            A_**2 / 2
          + (5 - T + 9*C + 4*C**2)     * A_**4 / 24
          + (61 - 58*T + T**2 + 600*C - 330*(e2/(1-e2))) * A_**6 / 720
        )
    )

    return xs, ys


def sample_band(path: Path, xs: np.ndarray, ys: np.ndarray, band: int = 1):
    with rasterio.open(path) as src:
        nodata = src.nodata
        values = []
        for v in src.sample(zip(xs, ys), indexes=band):
            val = float(v[0])
            if (nodata is not None and val == nodata) or not np.isfinite(val):
                values.append(np.nan)
            else:
                values.append(val)
    return np.array(values)


def stats_row(label, x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return {"label": label, "n": n, "r": np.nan, "r2": np.nan,
                "p_value": np.nan, "rmse": np.nan}
    r, p   = stats.pearsonr(x, y)
    rmse   = float(np.sqrt(mean_squared_error(y, x)))
    return {"label": label, "n": n,
            "r": round(r, 4), "r2": round(r**2, 4),
            "p_value": round(p, 6), "rmse": round(rmse, 6)}


def main():
    print("Venice CMUI Validation")
    print("=" * 50)

    df = pd.read_csv(GT_PATH)
    print(f"Ground truth points : {len(df)}")
    print(f"Lon: {df['lon'].min():.3f} – {df['lon'].max():.3f}")
    print(f"Lat: {df['lat'].min():.3f} – {df['lat'].max():.3f}")

    print("\nProjecting WGS84 -> UTM Zone 32N (EPSG:32632)...")
    xs, ys = wgs84_to_utm32n(df["lon"].values, df["lat"].values)
    print(f"Easting : {xs.min():.0f} – {xs.max():.0f}")
    print(f"Northing: {ys.min():.0f} – {ys.max():.0f}")

    print("\nSampling CMUI...")
    cmui_vals = sample_band(CMUI_PATH, xs, ys, band=1)
    n_valid = int(np.sum(np.isfinite(cmui_vals)))
    print(f"  Valid samples : {n_valid}/{len(cmui_vals)}")
    if n_valid > 0:
        print(f"  CMUI range    : {np.nanmin(cmui_vals):.4f} – "
              f"{np.nanmax(cmui_vals):.4f} m")

    print("Sampling zone map...")
    zone_raw  = sample_band(ZONE_PATH, xs, ys, band=1)
    zone_vals = np.where(np.isfinite(zone_raw), zone_raw.astype(int), 0)

    df["cmui_m"]    = cmui_vals
    df["zone_code"] = zone_vals
    df["zone_label"] = [ZONE_NAMES.get(int(z), "unknown") for z in zone_vals]
    df["is_proxy"]   = False

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "validation_venice.csv", index=False)

    valid = df.dropna(subset=["cmui_m", "observed_error_m"])
    valid = valid[valid["cmui_m"] > 0]

    lines = ["CMUI Validation Statistics — venice", "=" * 50]
    ov = stats_row("overall", valid["cmui_m"].values,
                   valid["observed_error_m"].values)
    lines += [
        f"\nOverall (n={ov['n']}):",
        f"  Pearson r  : {ov['r']}",
        f"  R^2        : {ov['r2']}",
        f"  p-value    : {ov['p_value']}",
        f"  RMSE       : {ov['rmse']} m",
        "\nBy geomorphic zone:",
    ]
    for code, name in ZONE_NAMES.items():
        sub = valid[valid["zone_code"] == code]
        if len(sub) < 3:
            continue
        s = stats_row(name, sub["cmui_m"].values, sub["observed_error_m"].values)
        lines.append(
            f"  {s['label']:15s}  n={s['n']:3d}  r={s['r']:6.3f}  "
            f"R²={s['r2']:5.3f}  p={s['p_value']:.4f}  RMSE={s['rmse']:.6f}m"
        )
    lines += [
        "\nNOTE: observed_error_m = SHYFEM tidal model RMSE at 26 tide gauge",
        "stations (Madricardo et al. 2017, Sci. Data). Validates sigma^2_tidal.",
    ]

    txt = "\n".join(lines)
    print("\n" + txt)
    stats_path = OUT_DIR / "validation_stats_venice.txt"
    stats_path.write_text(txt, encoding="utf-8")
    print(f"\nSaved: {stats_path}")


if __name__ == "__main__":
    main()
