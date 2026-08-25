"""
processing/03_cmui_fusion.py

Fuses all five uncertainty components into the Coastal Measurement
Uncertainty Index (CMUI) surface.

CMUI(x, t) = sqrt( sum_i [ w_i(x) * sigma^2_i(x) ] + tau(t) )

Components:
  sigma^2_DEM    : from gee_exports/dem_variance/
  sigma^2_NDWI   : from gee_exports/ndwi_variance/
  sigma^2_tidal  : from processed/tidal_variance/
  sigma^2_veg    : computed here from NDVI + canopy_height + beta
  sigma^2_sensor : from gee_exports/sensor_variance/

Weights w_i(x) are zone-adaptive, read from ZONE_WEIGHTS in constants.py
and applied per-pixel using the zone map from 02_zone_classifier.py.

tau(t) temporal staleness is computed using the DEM acquisition year
relative to the current year.

Outputs: D:/cmui/data/processed/cmui_components/ (intermediate)
         D:/cmui/outputs/cmui_rasters/cmui_<SITE>.tif (final)

Run:
    python processing/03_cmui_fusion.py --site sundarbans
    python processing/03_cmui_fusion.py --site venice
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from processing.utils.constants import (
    AOI,
    ZONE_WEIGHTS,
    ZONE_THRESHOLDS,
    LAMBDA_STALENESS,
    BETA_MANGROVE,
    BETA_DEFAULT,
    COPERNICUS_RMSE_M,
)
from processing.utils.weights import get_weights

# ---------------------------------------------------------------------------
# Zone codes (must match 02_zone_classifier.py)
# ---------------------------------------------------------------------------

ZONE_CODE_MAP: dict[int, str] = {
    1: "sandy_beach",
    2: "rocky_shore",
    3: "mangrove",
    4: "tidal_flat",
    5: "coral_reef",
    0: "unclassified",
}

# DEM acquisition year for tau(t) computation
# Copernicus GLO-30 is based on TanDEM-X acquisitions 2011-2015, product 2021
DEM_ACQUISITION_YEAR: int = 2021
CURRENT_YEAR: int = 2025


# ---------------------------------------------------------------------------
# Raster alignment
# ---------------------------------------------------------------------------

def load_and_align(
    src_path: Path,
    band: int,
    ref_profile: dict,
    nodata_fill: float = 0.0,
) -> np.ndarray:
    """
    Load one band from src_path and reproject to match ref_profile.
    Fills nodata with nodata_fill. Returns 2D float32 array.
    """
    with rasterio.open(src_path) as src:
        data = src.read(band).astype(np.float32)
        nodata = src.nodata
        src_crs       = src.crs
        src_transform = src.transform

    if nodata is not None:
        data[data == nodata] = np.nan

    dst = np.full(
        (ref_profile["height"], ref_profile["width"]),
        fill_value=np.nan,
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
    # Replace remaining NaNs with nodata_fill after reprojection
    dst = np.where(np.isfinite(dst), dst, nodata_fill)
    return dst


def load_reference_profile(dem_path: Path) -> dict:
    with rasterio.open(dem_path) as src:
        return src.profile.copy()


# ---------------------------------------------------------------------------
# sigma^2_veg computation
# ---------------------------------------------------------------------------

def compute_sigma2_veg(
    ndvi: np.ndarray,
    canopy_height: np.ndarray,
    zone_codes: np.ndarray,
) -> np.ndarray:
    """
    sigma^2_veg(x) = (beta * NDVI(x) * h_c(x))^2

    beta is zone-dependent:
      - mangrove zones: BETA_MANGROVE (0.4)
      - all other zones: BETA_DEFAULT (0.2)

    Reference: Gaveau & Hill (2003) Canadian Journal of Remote Sensing
    """
    beta = np.where(zone_codes == 3, BETA_MANGROVE, BETA_DEFAULT)
    sigma2_veg = (beta * ndvi * canopy_height) ** 2
    return sigma2_veg.astype(np.float32)


# ---------------------------------------------------------------------------
# tau(t) temporal staleness
# ---------------------------------------------------------------------------

def compute_tau(
    delta_t: float,
    sigma2_0: np.ndarray,
    lambda_rate: float,
) -> np.ndarray:
    """
    tau(t) = sigma^2_0 * (exp(lambda * delta_t) - 1)

    Where:
      delta_t   : measurement age in years
      sigma^2_0 : baseline variance (DEM RMSE squared at acquisition)
      lambda    : staleness decay rate from LAMBDA_STALENESS

    Reference: CMUI paper Equation 6
    """
    tau = sigma2_0 * (np.exp(lambda_rate * delta_t) - 1.0)
    return tau.astype(np.float32)


# ---------------------------------------------------------------------------
# Zone-adaptive weighted fusion
# ---------------------------------------------------------------------------

def fuse_cmui(
    sigma2_dem:    np.ndarray,
    sigma2_ndwi:   np.ndarray,
    sigma2_tidal:  np.ndarray,
    sigma2_veg:    np.ndarray,
    sigma2_sensor: np.ndarray,
    zone_codes:    np.ndarray,
    tau:           np.ndarray,
) -> np.ndarray:
    """
    CMUI(x, t) = sqrt( w1*s2_dem + w2*s2_ndwi + w3*s2_tidal
                       + w4*s2_veg + w5*s2_sensor + tau )

    Weights w1-w5 are looked up per-pixel from the zone code.
    Zones with code 0 (unclassified) use uniform weights [0.2, 0.2, 0.2, 0.2, 0.2].

    Returns a 2D float32 array with CMUI values in metres.
    """
    h, w = sigma2_dem.shape
    cmui = np.zeros((h, w), dtype=np.float32)

    # Build weight arrays per zone
    components = np.stack([
        sigma2_dem,
        sigma2_ndwi,
        sigma2_tidal,
        sigma2_veg,
        sigma2_sensor,
    ], axis=0)   # shape: (5, H, W)

    for code, zone_name in ZONE_CODE_MAP.items():
        mask = (zone_codes == code)
        if not np.any(mask):
            continue

        if zone_name == "unclassified":
            weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
        else:
            weights = get_weights(zone_name)

        # Weighted sum of variance components at this zone's pixels
        weighted_sum = np.zeros((h, w), dtype=np.float32)
        for i, wt in enumerate(weights):
            weighted_sum += wt * components[i]

        inner = weighted_sum + tau
        inner = np.maximum(inner, 0.0)   # guard against tiny negatives from float ops
        cmui[mask] = np.sqrt(inner[mask])

    return cmui


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_raster_from_profile(
    bands: dict[str, np.ndarray],
    ref_profile: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = ref_profile.copy()
    profile.update(
        dtype="float32",
        count=len(bands),
        compress="lzw",
        nodata=-9999.0,
    )
    with rasterio.open(output_path, "w", **profile) as dst:
        for idx, (name, arr) in enumerate(bands.items(), start=1):
            dst.write(arr.astype(np.float32), idx)
            dst.update_tags(idx, name=name)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Written: {output_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CMUI weighted fusion pipeline"
    )
    parser.add_argument(
        "--site",
        choices=list(AOI.keys()),
        default="sundarbans",
        help="Study site",
    )
    args = parser.parse_args()
    site = args.site

    gee_dir   = Path("D:/cmui/data/gee_exports")
    proc_dir  = Path("D:/cmui/data/processed")
    out_dir   = Path("D:/cmui/outputs/cmui_rasters")

    # Input paths
    dem_path     = gee_dir  / "dem_variance"   / f"dem_variance_{site}.tif"
    ndwi_path    = gee_dir  / "ndwi_variance"  / f"ndwi_variance_{site}.tif"
    veg_path     = gee_dir  / "veg_canopy"     / f"veg_canopy_{site}.tif"
    sensor_path  = gee_dir  / "sensor_variance"/ f"sensor_variance_{site}.tif"
    tidal_path   = proc_dir / "tidal_variance" / f"tidal_variance_{site}.tif"
    zone_path    = proc_dir / "zone_maps"      / f"zone_map_{site}.tif"
    output_path  = out_dir  / f"cmui_{site}.tif"

    required = [dem_path, ndwi_path, veg_path, sensor_path, tidal_path, zone_path]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"Required input not found: {p}\n"
                "Run all GEE scripts and processing/01_tidal_model.py and "
                "processing/02_zone_classifier.py first."
            )

    print(f"Site: {site}")
    print("Loading reference grid from DEM...")
    ref_profile = load_reference_profile(dem_path)
    ref_profile.update(dtype="float32", count=1)

    print("Loading and aligning all input rasters...")
    elevation     = load_and_align(dem_path,    1, ref_profile)
    dem_roughness = load_and_align(dem_path,    2, ref_profile)
    sigma2_ndwi   = load_and_align(ndwi_path,   1, ref_profile)
    ndvi          = load_and_align(veg_path,    1, ref_profile)
    canopy_height = load_and_align(veg_path,    2, ref_profile)
    sigma2_tidal  = load_and_align(tidal_path,  1, ref_profile)
    sigma2_sensor = load_and_align(sensor_path, 1, ref_profile)
    zone_codes    = load_and_align(zone_path,   1, ref_profile, nodata_fill=0.0).astype(np.int8)

    # sigma^2_DEM: scale roughness by Copernicus RMSE
    # roughness (local std dev) + published RMSE form combined DEM uncertainty
    sigma2_dem = (dem_roughness ** 2 + COPERNICUS_RMSE_M ** 2)

    print("Computing sigma^2_veg from NDVI and canopy height...")
    sigma2_veg = compute_sigma2_veg(ndvi, canopy_height, zone_codes)

    print("Computing temporal staleness tau(t)...")
    delta_t  = float(CURRENT_YEAR - DEM_ACQUISITION_YEAR)
    sigma2_0 = dem_roughness ** 2   # baseline variance in m^2
    tau      = compute_tau(delta_t, np.full_like(elevation, sigma2_0), LAMBDA_STALENESS)

    print(f"  DEM age: {delta_t:.0f} years  |  tau (mean) = {float(np.mean(sigma2_0) * (np.exp(LAMBDA_STALENESS * delta_t) - 1)):.4f} m^2")

    print("Fusing CMUI surface...")
    cmui = fuse_cmui(
        sigma2_dem=sigma2_dem,
        sigma2_ndwi=sigma2_ndwi,
        sigma2_tidal=sigma2_tidal,
        sigma2_veg=sigma2_veg,
        sigma2_sensor=sigma2_sensor,
        zone_codes=zone_codes,
        tau=tau,
    )

    # Summary statistics
    valid = cmui[cmui > 0]
    print(f"\nCMUI statistics ({site}):")
    print(f"  Mean   : {np.mean(valid):.4f} m")
    print(f"  Std    : {np.std(valid):.4f} m")
    print(f"  Min    : {np.min(valid):.4f} m")
    print(f"  Max    : {np.max(valid):.4f} m")
    print(f"  P95    : {np.percentile(valid, 95):.4f} m")

    print("\nWriting CMUI raster and component rasters...")
    write_raster_from_profile(
        bands={
            "CMUI":           cmui,
            "sigma2_DEM":     sigma2_dem,
            "sigma2_NDWI":    sigma2_ndwi,
            "sigma2_tidal":   sigma2_tidal,
            "sigma2_veg":     sigma2_veg,
            "sigma2_sensor":  sigma2_sensor,
            "tau":            tau,
            "zone_codes":     zone_codes.astype(np.float32),
        },
        ref_profile=ref_profile,
        output_path=output_path,
    )

    print(f"\nDone. CMUI surface saved to:\n  {output_path}")
    print("Bands: 1=CMUI  2-6=sigma^2 components  7=tau  8=zone_codes")
    print("Next step: python processing/04_validation.py --site", site)


if __name__ == "__main__":
    main()