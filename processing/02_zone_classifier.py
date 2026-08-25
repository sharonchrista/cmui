"""
processing/02_zone_classifier.py

Rule-based geomorphic zone classification for the CMUI pipeline.

Reads the GEE-exported rasters (DEM, NDWI variance, veg canopy) and
assigns each pixel to one of five coastal geomorphic zones using
spectral and morphological thresholds from constants.ZONE_THRESHOLDS.

Inputs (from D:/cmui/data/gee_exports/):
  dem_variance/<SITE>.tif      -> Band 1: elevation, Band 3: slope
  ndwi_variance/<SITE>.tif     -> Band 1: NDWI variance (used as proxy
                                   for tidal inundation frequency)
  veg_canopy/<SITE>.tif        -> Band 1: NDVI median

Output: D:/cmui/data/processed/zone_maps/zone_map_<SITE>.tif
  Band 1 - zone_code : integer zone label
             1 = sandy_beach
             2 = rocky_shore
             3 = mangrove
             4 = tidal_flat
             5 = coral_reef
             0 = unclassified (land, open water)
  Band 2 - zone_confidence : fraction of pixels in a 3x3 neighbourhood
             sharing the same zone code (spatial homogeneity indicator)

Run after all 4 GEE scripts have completed:
    python processing/02_zone_classifier.py --site sundarbans
    python processing/02_zone_classifier.py --site venice
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum
from scipy.ndimage import uniform_filter

from processing.utils.constants import (
    AOI,
    TARGET_CRS,
    TARGET_RES_M,
    ZONE_THRESHOLDS,
)

# ---------------------------------------------------------------------------
# Zone codes
# ---------------------------------------------------------------------------

ZONE_CODES: dict[str, int] = {
    "sandy_beach": 1,
    "rocky_shore": 2,
    "mangrove":    3,
    "tidal_flat":  4,
    "coral_reef":  5,
    "unclassified": 0,
}

ZONE_NAMES: dict[int, str] = {v: k for k, v in ZONE_CODES.items()}


# ---------------------------------------------------------------------------
# Raster alignment
# ---------------------------------------------------------------------------

def load_and_align(
    src_path: Path,
    band: int,
    ref_profile: dict,
) -> np.ndarray:
    """
    Load one band from src_path and reproject it to match ref_profile
    (CRS, transform, width, height). Returns a 2D float32 array.
    """
    with rasterio.open(src_path) as src:
        data = src.read(band).astype(np.float32)
        src_crs       = src.crs
        src_transform = src.transform

    dst = np.zeros(
        (ref_profile["height"], ref_profile["width"]),
        dtype=np.float32,
    )

    reproject(
        source=data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_profile["transform"],
        dst_crs=ref_profile["crs"],
        resampling=Resampling.bilinear,
    )

    return dst


def load_reference_profile(dem_path: Path) -> dict:
    """Load the DEM raster profile as the alignment reference grid."""
    with rasterio.open(dem_path) as src:
        return src.profile.copy()


# ---------------------------------------------------------------------------
# Zone classification
# ---------------------------------------------------------------------------

def classify_zones(
    elevation: np.ndarray,
    slope: np.ndarray,
    ndvi: np.ndarray,
    ndwi_variance: np.ndarray,
    thresholds: dict,
) -> np.ndarray:
    """
    Assign each pixel an integer zone code using rule-based thresholds.

    Classification order matters: more specific rules are applied last
    and override earlier assignments. Mangrove (NDVI + elevation) is
    the most distinctive and applied last.

    Returns a 2D int8 array of zone codes.
    """
    codes = np.zeros(elevation.shape, dtype=np.int8)

    t = thresholds

    # Tidal flat: very low slope, high NDWI variance, low vegetation
    tidal_mask = (
        (slope      <= t["tidal_flat"]["slope_max_deg"])
        & (ndvi     <= t["tidal_flat"]["ndvi_max"])
        & (ndwi_variance > np.nanpercentile(ndwi_variance, 40))
    )
    codes[tidal_mask] = ZONE_CODES["tidal_flat"]

    # Sandy beach: low slope, low NDVI, some NDWI variance
    sandy_mask = (
        (ndvi  <= t["sandy_beach"]["ndvi_max"])
        & (slope <= t["sandy_beach"]["slope_max_deg"])
        & (~tidal_mask)
    )
    codes[sandy_mask] = ZONE_CODES["sandy_beach"]

    # Rocky shore: low NDVI, high slope
    rocky_mask = (
        (ndvi  <= t["rocky_shore"]["ndvi_max"])
        & (slope >= t["rocky_shore"]["slope_min_deg"])
    )
    codes[rocky_mask] = ZONE_CODES["rocky_shore"]

    # Coral reef: very low NDVI, near-zero elevation
    # Typically submerged — elevation near 0 or slightly negative
    coral_mask = (
    (ndvi      <= t["coral_reef"]["ndvi_max"])
    & (elevation >= t["coral_reef"]["bath_min_m"])
    & (elevation <= t["coral_reef"]["bath_max_m"])
    & (slope    <= 1.0)
    & (~sandy_mask)
    & (~tidal_mask)
)
    codes[coral_mask] = ZONE_CODES["coral_reef"]

    # Mangrove: high NDVI, low elevation (overrides other classes)
    mangrove_mask = (
        (ndvi      >= t["mangrove"]["ndvi_min"])
        & (elevation <= t["mangrove"]["elev_max_m"])
    )
    codes[mangrove_mask] = ZONE_CODES["mangrove"]

    return codes


def compute_spatial_confidence(
    zone_codes: np.ndarray,
    window_size: int = 3,
) -> np.ndarray:
    """
    Compute per-pixel zone confidence as the fraction of pixels in a
    3x3 neighbourhood sharing the same zone code.
    A confidence of 1.0 means all 9 neighbours agree on the zone.
    """
    confidence = np.zeros_like(zone_codes, dtype=np.float32)
    for code in ZONE_CODES.values():
        if code == 0:
            continue
        binary = (zone_codes == code).astype(np.float32)
        local_frac = uniform_filter(binary, size=window_size)
        mask = zone_codes == code
        confidence[mask] = local_frac[mask]
    return confidence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rule-based geomorphic zone classifier for CMUI"
    )
    parser.add_argument(
        "--site",
        choices=list(AOI.keys()),
        default="sundarbans",
        help="Study site to classify",
    )
    args = parser.parse_args()
    site = args.site

    gee_dir  = Path("D:/cmui/data/gee_exports")
    proc_dir = Path("D:/cmui/data/processed/zone_maps")

    dem_path   = gee_dir / "dem_variance"   / f"dem_variance_{site}.tif"
    ndwi_path  = gee_dir / "ndwi_variance"  / f"ndwi_variance_{site}.tif"
    veg_path   = gee_dir / "veg_canopy"     / f"veg_canopy_{site}.tif"
    output_path = proc_dir / f"zone_map_{site}.tif"

    for p in [dem_path, ndwi_path, veg_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required input not found: {p}\n"
                "Run all 4 GEE scripts first."
            )

    print(f"Site     : {site}")
    print(f"Inputs   : {gee_dir}")

    # Load reference grid from DEM
    ref_profile = load_reference_profile(dem_path)
    ref_profile.update(dtype="float32", count=1)

    print("Loading and aligning input rasters...")
    elevation     = load_and_align(dem_path,  1, ref_profile)
    slope         = load_and_align(dem_path,  3, ref_profile)
    ndwi_variance = load_and_align(ndwi_path, 1, ref_profile)
    ndvi          = load_and_align(veg_path,  1, ref_profile)

    print("Classifying zones...")
    zone_codes = classify_zones(
        elevation=elevation,
        slope=slope,
        ndvi=ndvi,
        ndwi_variance=ndwi_variance,
        thresholds=ZONE_THRESHOLDS,
    )

    print("Computing spatial confidence...")
    confidence = compute_spatial_confidence(zone_codes)

    # Zone distribution summary
    total_pixels = zone_codes.size
    print("\nZone distribution:")
    for code, name in ZONE_NAMES.items():
        count = int(np.sum(zone_codes == code))
        pct   = 100.0 * count / total_pixels
        print(f"  {name:15s} (code={code})  {count:8d} px  {pct:5.1f}%")

    # Write output
    proc_dir.mkdir(parents=True, exist_ok=True)
    out_profile = ref_profile.copy()
    out_profile.update(dtype="float32", count=2, compress="lzw")

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(zone_codes.astype(np.float32), 1)
        dst.write(confidence, 2)
        dst.update_tags(1, name="zone_code")
        dst.update_tags(2, name="zone_confidence")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nZone map written: {output_path}  ({size_mb:.1f} MB)")
    print("Next step: python processing/03_cmui_fusion.py --site", site)


if __name__ == "__main__":
    main()