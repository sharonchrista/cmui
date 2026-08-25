"""
run_narrabeen_gps_validation.py

Validates CMUI against 40+ years of monthly GPS beach profile surveys
at Narrabeen-Collaroy Beach, Sydney, Australia.

Reference:
  Turner et al. (2016) A multi-decade dataset of monthly beach profile
  surveys and inshore wave forcing at Narrabeen, Australia.
  Nature Scientific Data. doi:10.1038/sdata.2016.24

Ground truth: standard deviation of shoreline position across all
monthly surveys at each of 5 transects (PF1, PF2, PF4, PF6, PF8).
This temporal variability represents real measured positional uncertainty
that is completely independent of CMUI's input data.

Validation: CMUI values extracted at each transect origin are correlated
against the observed shoreline position std dev.

Run after run_narrabeen_processing.py:
    python run_narrabeen_gps_validation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import rasterio
from scipy import stats
from sklearn.metrics import mean_squared_error

CMUI_PATH  = Path("D:/cmui/outputs/cmui_rasters/cmui_narrabeen.tif")
ZONE_PATH  = Path("D:/cmui/data/processed/zone_maps/zone_map_narrabeen.tif")
PROFILES   = Path("D:/cmui/data/raw/groundtruth/narrabeen/narrabeen_profiles.csv")
OUT_DIR    = Path("D:/cmui/outputs/validation")

ZONE_NAMES = {0:"unclassified",1:"sandy_beach",2:"rocky_shore",
              3:"mangrove",4:"tidal_flat",5:"coral_reef"}

# Transect landward-end coordinates in WGS84
# Source: NARRABEEN_transects.geojson (SDS_Benchmark, Vos et al. 2019)
TRANSECTS = {
    "PF1": (151.304527, -33.705739),
    "PF2": (151.302870, -33.709297),
    "PF4": (151.299674, -33.717098),
    "PF6": (151.299616, -33.724952),
    "PF8": (151.301793, -33.732205),
}


# ---------------------------------------------------------------------------
# UTM Zone 56S projection (no pyproj needed)
# ---------------------------------------------------------------------------

def wgs84_to_utm56s(lons, lats):
    """
    Pure numpy WGS84 -> UTM Zone 56S (EPSG:32756).
    Southern hemisphere: N_false = 10,000,000m.
    """
    a   = 6378137.0
    f   = 1 / 298.257223563
    b   = a * (1 - f)
    e2  = 1 - (b/a)**2
    lon0    = np.radians(153.0)   # Central meridian UTM Zone 56
    k0      = 0.9996
    E0      = 500000.0
    N_false = 10_000_000.0

    phi = np.radians(lats)
    lam = np.radians(lons)
    N   = a / np.sqrt(1 - e2 * np.sin(phi)**2)
    T   = np.tan(phi)**2
    C   = (e2/(1-e2)) * np.cos(phi)**2
    A_  = np.cos(phi) * (lam - lon0)
    e4  = e2**2; e6 = e2**3
    M   = a * (
        (1 - e2/4 - 3*e4/64 - 5*e6/256) * phi
      - (3*e2/8 + 3*e4/32 + 45*e6/1024) * np.sin(2*phi)
      + (15*e4/256 + 45*e6/1024)         * np.sin(4*phi)
      - (35*e6/3072)                      * np.sin(6*phi)
    )
    xs = k0 * N * (
        A_ + (1-T+C)*A_**3/6
        + (5-18*T+T**2+72*C-58*(e2/(1-e2)))*A_**5/120
    ) + E0
    ys = k0 * (
        M + N*np.tan(phi) * (
            A_**2/2
          + (5-T+9*C+4*C**2)*A_**4/24
          + (61-58*T+T**2+600*C-330*(e2/(1-e2)))*A_**6/720
        )
    ) + N_false
    return xs, ys


# ---------------------------------------------------------------------------
# Compute shoreline std dev from GPS profiles
# ---------------------------------------------------------------------------

def compute_shoreline_std(profiles_path: Path) -> pd.DataFrame:
    """
    For each transect, compute the standard deviation of shoreline position
    across all monthly surveys. Shoreline = chainage where elevation = 0 (MSL).
    """
    df = pd.read_csv(profiles_path)
    df['Date'] = pd.to_datetime(df['Date'])

    records = []
    for pid, (lon, lat) in TRANSECTS.items():
        sub = df[df['Profile ID'] == pid].copy()
        shorelines = []
        for date, grp in sub.groupby('Date'):
            grp = grp.sort_values('Chainage')
            above = grp[grp['Elevation'] > 0]
            below = grp[grp['Elevation'] <= 0]
            if len(above) > 0 and len(below) > 0:
                x1, y1 = above.iloc[-1][['Chainage','Elevation']]
                x2, y2 = below.iloc[0][['Chainage','Elevation']]
                if y1 != y2:
                    sl = x1 - y1 * (x2 - x1) / (y2 - y1)
                    shorelines.append(sl)
        if len(shorelines) >= 12:
            records.append({
                'profile_id':       pid,
                'lon':              lon,
                'lat':              lat,
                'n_surveys':        len(shorelines),
                'shoreline_std_m':  float(np.std(shorelines)),
                'shoreline_mean_m': float(np.mean(shorelines)),
            })
            print(f"  {pid}: n={len(shorelines)}  std={np.std(shorelines):.2f}m")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Sample raster at projected coordinates
# ---------------------------------------------------------------------------

def sample_band(raster_path: Path, xs: np.ndarray, ys: np.ndarray,
                band: int = 1) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        values = []
        for v in src.sample(zip(xs, ys), indexes=band):
            val = float(v[0])
            if (nodata is not None and val == nodata) or not np.isfinite(val):
                values.append(np.nan)
            else:
                values.append(val)
    return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Validation statistics
# ---------------------------------------------------------------------------

def compute_stats(label, x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return {"label": label, "n": n, "r": np.nan, "r2": np.nan,
                "p": np.nan, "rmse": np.nan}
    r, p   = stats.pearsonr(x, y)
    rmse   = float(np.sqrt(mean_squared_error(y, x)))
    return {"label": label, "n": n,
            "r":    round(r, 4), "r2": round(r**2, 4),
            "p":    round(p, 6), "rmse": round(rmse, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for p in [CMUI_PATH, ZONE_PATH, PROFILES]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    print("=== Narrabeen GPS Validation ===\n")

    # Step 1: compute shoreline std dev from GPS profiles
    print("Computing shoreline position variability from GPS surveys...")
    gt = compute_shoreline_std(PROFILES)
    print(f"  Ground truth computed for {len(gt)} transects\n")

    # Step 2: project coordinates to UTM Zone 56S
    print("Projecting transect coordinates to UTM Zone 56S...")
    lons = gt['lon'].values
    lats = gt['lat'].values
    xs, ys = wgs84_to_utm56s(lons, lats)
    print(f"  Easting:  {xs.min():.0f} – {xs.max():.0f}")
    print(f"  Northing: {ys.min():.0f} – {ys.max():.0f}\n")

    # Step 3: sample CMUI and zone
    print("Sampling CMUI at transect locations...")
    cmui_vals = sample_band(CMUI_PATH, xs, ys, band=1)
    zone_vals = sample_band(ZONE_PATH, xs, ys, band=1)
    n_valid   = int(np.sum(np.isfinite(cmui_vals)))
    print(f"  Valid CMUI samples: {n_valid}/{len(cmui_vals)}")
    if n_valid > 0:
        print(f"  CMUI range at transects: {np.nanmin(cmui_vals):.4f} – "
              f"{np.nanmax(cmui_vals):.4f} m\n")

    gt['cmui_m']    = cmui_vals
    gt['zone_code'] = np.where(np.isfinite(zone_vals), zone_vals.astype(int), 0)
    gt['zone_label']= [ZONE_NAMES.get(int(z), "unknown") for z in gt['zone_code']]
    gt['is_proxy']  = False

    # Step 4: validation statistics
    valid = gt.dropna(subset=['cmui_m', 'shoreline_std_m'])
    valid = valid[valid['cmui_m'] > 0]

    # Print per-transect table
    print("Per-transect results:")
    print(f"  {'Transect':8s} {'CMUI (m)':10s} {'GPS std (m)':12s} {'Zone':15s}")
    print("  " + "-" * 50)
    for _, row in valid.iterrows():
        print(f"  {row['profile_id']:8s} {row['cmui_m']:10.4f} "
              f"{row['shoreline_std_m']:12.2f} {row['zone_label']:15s}")

    # Statistics
    ov = compute_stats("overall",
                       valid['cmui_m'].values,
                       valid['shoreline_std_m'].values)

    lines = [
        "CMUI Validation Statistics — narrabeen",
        "=" * 55,
        f"\nSite: Narrabeen-Collaroy Beach, Sydney, Australia",
        f"Ground truth: shoreline position std dev (GPS surveys 1976–2019)",
        f"Reference: Turner et al. (2016) Nature Scientific Data",
        f"Transects: {len(valid)} (PF1, PF2, PF4, PF6, PF8)",
        "",
        f"Overall (n={ov['n']}):",
        f"  Pearson r  : {ov['r']}",
        f"  R^2        : {ov['r2']}",
        f"  p-value    : {ov['p']}",
        f"  RMSE       : {ov['rmse']} m",
        "",
        "Per-transect:",
    ]
    for _, row in valid.iterrows():
        lines.append(
            f"  {row['profile_id']:5s}  CMUI={row['cmui_m']:.4f}m  "
            f"GPS_std={row['shoreline_std_m']:.2f}m  zone={row['zone_label']}"
        )
    lines += [
        "",
        "NOTE: GPS shoreline std dev (independent of CMUI inputs) used as",
        "observed positional uncertainty. Higher std dev = higher real-world",
        "positional uncertainty at that transect.",
    ]

    txt = "\n".join(lines)
    print("\n" + txt)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt.to_csv(OUT_DIR / "validation_narrabeen.csv", index=False)
    (OUT_DIR / "validation_stats_narrabeen.txt").write_text(txt, encoding="utf-8")
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()