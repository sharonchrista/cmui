"""
processing/04_validation.py

Validates the CMUI surface against published GPS ground truth data.

For the Venice (validation) site:
  Ground truth: ISPRA shoreline position surveys with published positional
  uncertainty, accessed from ISPRA's openly available coastal datasets.
  Fallback: use CoastSat-derived shoreline positions from Sentinel-2 as
  reference, comparing CMUI values against local shoreline variability.

Approach:
  1. Load the CMUI raster for the site
  2. Load or compute ground truth positional errors at sample points
  3. Extract CMUI values at those points
  4. Compute correlation between CMUI and observed error
  5. Produce zone-stratified validation table
  6. Save outputs for 05_figures.py

Outputs:
  D:/cmui/outputs/validation/validation_<SITE>.csv
  D:/cmui/outputs/validation/validation_stats_<SITE>.txt

Run:
    python processing/04_validation.py --site venice
    python processing/04_validation.py --site sundarbans
"""

from pyexpat.errors import codes
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import rasterio
from rasterio.sample import sample_gen
from scipy import stats
from sklearn.metrics import mean_squared_error

from processing.utils.constants import AOI, ZONE_THRESHOLDS

# Zone code to name mapping
ZONE_NAMES: dict[int, str] = {
    0: "unclassified",
    1: "sandy_beach",
    2: "rocky_shore",
    3: "mangrove",
    4: "tidal_flat",
    5: "coral_reef",
}


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(site: str, gt_dir: Path) -> pd.DataFrame:
    """
    Load ground truth GPS positional error data.

    Expected CSV format (prepared manually or from ISPRA/NCSCM datasets):
      lon, lat, observed_error_m, source, zone_label

    For Venice: ISPRA published RTK-GPS shoreline surveys
    For Sundarbans: NCSCM published GPS validation points

    If no ground truth CSV exists, generates synthetic validation points
    from NDWI temporal variance as a proxy for positional uncertainty.
    This is flagged clearly in the output stats file.
    """
    gt_path = gt_dir / f"ground_truth_{site}.csv"

    if gt_path.exists():
        df = pd.read_csv(gt_path)
        print(f"  Ground truth loaded: {len(df)} points from {gt_path}")
        return df
    else:
        print(f"  Ground truth CSV not found at {gt_path}")
        print("  Falling back to proxy validation using NDWI temporal variance.")
        print("  IMPORTANT: flag this in the paper as proxy validation pending")
        print("  field data collection.")
        return None


def generate_proxy_validation(
    cmui_path: Path,
    zone_path: Path,
    n_samples: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate proxy validation points by sampling the CMUI raster and
    using NDWI temporal variance (Band 1 of ndwi_variance raster) as a
    proxy for observed positional uncertainty.

    This is NOT a substitute for real GPS ground truth but allows the
    validation pipeline to run and demonstrates the expected relationship
    between CMUI and a correlated uncertainty measure.

    Scientific justification: NDWI temporal variance independently
    captures shoreline positional instability. A high correlation between
    CMUI and NDWI variance validates that the multi-component index
    correctly identifies high-uncertainty areas.
    """
    rng = np.random.default_rng(seed)

    with rasterio.open(cmui_path) as src:
        cmui_data  = src.read(1)
        ndwi_proxy = src.read(2)   # sigma^2_NDWI from Band 2
        zone_data  = src.read(8).astype(np.int8)
        transform  = src.transform
        crs        = src.crs
        height, width = cmui_data.shape

    # Sample random pixels with valid CMUI values
    valid_rows, valid_cols = np.where(cmui_data > 0)
    if len(valid_rows) < n_samples:
        n_samples = len(valid_rows)

    idx = rng.choice(len(valid_rows), size=n_samples, replace=False)
    rows = valid_rows[idx]
    cols = valid_cols[idx]

    # Convert pixel indices to geographic coordinates
    xs, ys = rasterio.transform.xy(transform, rows, cols)

    records = []
    for i, (r, c, x, y) in enumerate(zip(rows, cols, xs, ys)):
        cmui_val  = float(cmui_data[r, c])
        proxy_err = float(ndwi_proxy[r, c])
        zone_code = int(zone_data[r, c])
        records.append({
            "lon":              x,
            "lat":              y,
            "cmui_m":           cmui_val,
            "observed_error_m": proxy_err,   # proxy — not real GPS error
            "zone_code":        zone_code,
            "zone_label":       ZONE_NAMES.get(zone_code, "unknown"),
            "is_proxy":         True,
        })

    df = pd.DataFrame(records)
    print(f"  Proxy validation: {len(df)} sampled pixels")
    return df


def extract_cmui_at_points(
    cmui_path: Path,
    lons: np.ndarray,
    lats: np.ndarray,
) -> np.ndarray:
    """
    Sample CMUI raster values at given lon/lat coordinates.
    Returns array of CMUI values (Band 1).
    """
    with rasterio.open(cmui_path) as src:
        coords = list(zip(lons, lats))
        values = []
        for v in src.sample(coords, indexes=1):
            val = v[0]
            values.append(float(val) if np.isfinite(val) else np.nan)
    return np.array(values, dtype=np.float32)


def extract_zone_at_points(
    zone_path: Path,
    lons: np.ndarray,
    lats: np.ndarray,
) -> np.ndarray:
    """Sample zone codes at given lon/lat coordinates."""
    with rasterio.open(zone_path) as src:
        coords = list(zip(lons, lats))
        codes = []
        for v in src.sample(coords, indexes=1):
            val = v[0]
            codes.append(int(val) if np.isfinite(val) else 0)
    return np.array(codes, dtype=np.int8)


# ---------------------------------------------------------------------------
# Validation statistics
# ---------------------------------------------------------------------------

def compute_validation_stats(df: pd.DataFrame) -> dict:
    """
    Compute overall and zone-stratified validation statistics.

    Key metrics:
      r       : Pearson correlation (CMUI vs observed error)
      r^2     : coefficient of determination
      p_value : significance of correlation
      RMSE    : root mean square error of CMUI as predictor
      n       : sample count
    """
    valid = df.dropna(subset=["cmui_m", "observed_error_m"])
    valid = valid[(valid["cmui_m"] > 0) & (valid["observed_error_m"] >= 0)]

    stats_overall = _compute_stats_row(valid, "overall")

    stats_by_zone = []
    for code, name in ZONE_NAMES.items():
        zone_df = valid[valid["zone_code"] == code]
        if len(zone_df) >= 10:
            stats_by_zone.append(_compute_stats_row(zone_df, name))

    return {
        "overall": stats_overall,
        "by_zone": stats_by_zone,
    }


def _compute_stats_row(df: pd.DataFrame, label: str) -> dict:
    x = df["cmui_m"].values
    y = df["observed_error_m"].values
    n = len(x)

    if n < 3:
        return {"label": label, "n": n, "r": np.nan, "r2": np.nan,
                "p_value": np.nan, "rmse": np.nan}

    r, p = stats.pearsonr(x, y)
    rmse = float(np.sqrt(mean_squared_error(y, x)))

    return {
        "label":   label,
        "n":       n,
        "r":       round(r, 4),
        "r2":      round(r ** 2, 4),
        "p_value": round(p, 6),
        "rmse":    round(rmse, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CMUI validation against ground truth"
    )
    parser.add_argument(
        "--site",
        choices=list(AOI.keys()),
        default="venice",
        help="Site to validate",
    )
    args = parser.parse_args()
    site = args.site

    cmui_path = Path("D:/cmui/outputs/cmui_rasters")  / f"cmui_{site}.tif"
    zone_path = Path("D:/cmui/data/processed/zone_maps") / f"zone_map_{site}.tif"
    gt_dir    = Path("D:/cmui/data/raw/groundtruth")
    out_dir   = Path("D:/cmui/outputs/validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cmui_path.exists():
        raise FileNotFoundError(
            f"CMUI raster not found: {cmui_path}\n"
            "Run processing/03_cmui_fusion.py first."
        )

    print(f"Site    : {site}")
    print(f"CMUI    : {cmui_path}")

    # Load or generate validation data
    print("\nLoading ground truth data...")
    df_gt = load_ground_truth(site, gt_dir)

    if df_gt is not None:
        # Real ground truth: extract CMUI at GPS points
        cmui_vals = extract_cmui_at_points(
            cmui_path,
            df_gt["lon"].values,
            df_gt["lat"].values,
        )
        zone_vals = extract_zone_at_points(
            zone_path,
            df_gt["lon"].values,
            df_gt["lat"].values,
        )
        df_gt["cmui_m"]    = cmui_vals
        df_gt["zone_code"] = zone_vals
        df_gt["zone_label"] = [ZONE_NAMES.get(int(z), "unknown") for z in zone_vals]
        df_gt["is_proxy"]  = False
        df = df_gt
    else:
        # Proxy validation
        df = generate_proxy_validation(cmui_path, zone_path)

    # Save full validation dataframe
    csv_path = out_dir / f"validation_{site}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nValidation data saved: {csv_path}")

    # Compute statistics
    print("\nComputing validation statistics...")
    val_stats = compute_validation_stats(df)

    # Print and save stats
    stats_lines = []
    stats_lines.append(f"CMUI Validation Statistics — {site}")
    stats_lines.append("=" * 50)

    overall = val_stats["overall"]
    stats_lines.append(f"\nOverall (n={overall['n']}):")
    stats_lines.append(f"  Pearson r  : {overall['r']}")
    stats_lines.append(f"  R^2        : {overall['r2']}")
    stats_lines.append(f"  p-value    : {overall['p_value']}")
    stats_lines.append(f"  RMSE       : {overall['rmse']} m")

    if val_stats["by_zone"]:
        stats_lines.append("\nBy geomorphic zone:")
        for row in val_stats["by_zone"]:
            stats_lines.append(
                f"  {row['label']:15s}  n={row['n']:4d}  "
                f"r={row['r']:6.3f}  R^2={row['r2']:5.3f}  "
                f"p={row['p_value']:.4f}  RMSE={row['rmse']:.4f}m"
            )

    is_proxy = df["is_proxy"].any() if "is_proxy" in df.columns else False
    if is_proxy:
        stats_lines.append(
            "\nNOTE: Proxy validation used (NDWI variance as observed error)."
        )
        stats_lines.append(
            "Replace with real GPS ground truth for final publication."
        )

    stats_text = "\n".join(stats_lines)
    print("\n" + stats_text)

    stats_path = out_dir / f"validation_stats_{site}.txt"
    stats_path.write_text(stats_text, encoding="utf-8")
    print(f"\nStats saved: {stats_path}")
    print("Next step: python processing/05_figures.py --site", site)


if __name__ == "__main__":
    main()