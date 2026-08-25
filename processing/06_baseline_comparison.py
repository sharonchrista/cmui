"""
06_baseline_comparison.py
Compares CMUI against three baselines using the existing CMUI raster.
Run: python processing/06_baseline_comparison.py --site sundarbans
"""
import argparse
import numpy as np
import rasterio
from pathlib import Path
from scipy import stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="sundarbans")
    args = parser.parse_args()

    path = Path(f"D:/cmui/outputs/cmui_rasters/cmui_{args.site}.tif")
    with rasterio.open(path) as src:
        cmui      = src.read(1).astype(np.float32)
        sigma_dem = src.read(2).astype(np.float32)
        sigma_ndwi= src.read(3).astype(np.float32)
        zones     = src.read(8).astype(np.int8)

    # Baselines
    baseline_dem  = np.sqrt(sigma_dem)
    baseline_ndwi = np.sqrt(np.abs(sigma_ndwi))

    # Uniform weight baseline (equal weighting, no zone adaptation)
    with rasterio.open(path) as src:
        s2 = src.read(2); s3 = src.read(3)
        s4 = src.read(4); s5 = src.read(5); s6 = src.read(6)
    baseline_uniform = np.sqrt(0.2*(s2+s3+s4+s5+s6))

    # Mask invalid
    mask = cmui > 0

    # Stats
    print(f"\nBaseline Comparison — {args.site}")
    print("=" * 60)

    for name, baseline in [
        ("sigma_DEM only",    baseline_dem),
        ("sigma_NDWI only",   baseline_ndwi),
        ("Uniform weights",   baseline_uniform),
    ]:
        b = baseline[mask]; c = cmui[mask]
        r, p = stats.pearsonr(b.flatten(), c.flatten())
        diff  = np.mean(np.abs(c - b))
        pct_higher = 100 * np.mean(c > b)
        print(f"\n{name}:")
        print(f"  Pearson r vs CMUI : {r:.4f}  (p={p:.2e})")
        print(f"  Mean abs diff     : {diff:.4f} m")
        print(f"  CMUI > baseline   : {pct_higher:.1f}% of pixels")

    # Zone-stratified comparison (DEM baseline vs CMUI)
    ZONE_NAMES = {1:"sandy_beach", 2:"rocky_shore", 3:"mangrove",
                  4:"tidal_flat",  5:"coral_reef"}
    print("\nZone-stratified: CMUI vs sigma_DEM baseline")
    for code, name in ZONE_NAMES.items():
        m = (zones == code) & mask
        if np.sum(m) < 10:
            continue
        cmui_z = cmui[m]; dem_z = baseline_dem[m]
        print(f"  {name:15s}  CMUI={np.mean(cmui_z):.3f}m  "
              f"DEM={np.mean(dem_z):.3f}m  "
              f"diff={np.mean(cmui_z-dem_z):+.3f}m")

if __name__ == "__main__":
    main()