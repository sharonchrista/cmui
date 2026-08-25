"""
run_narrabeen_processing.py

Runs the full local processing pipeline for Narrabeen:
  1. Tidal model (fallback datums — Narrabeen is micro-tidal, ~0.5m range)
  2. Zone classifier
  3. CMUI fusion

Run after run_narrabeen_gee.py:
    python run_narrabeen_processing.py
"""

import sys, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Add Narrabeen to constants before running
# (patching constants at runtime to avoid editing the file)
import processing.utils.constants as C

C.AOI["narrabeen"]          = [151.2958, -33.7390, 151.3122, -33.7013]
C.TARGET_CRS["narrabeen"]   = "EPSG:32756"
C.TIDE_GAUGES["narrabeen"]  = {
    "sydney_fort_denison": {"gesla_id": "sydney-fort-denison-australia",
                            "lon": 151.225, "lat": -33.854},
}
C.FALLBACK_TIDAL_DATUMS["narrabeen"] = {
    "sydney_fort_denison": {"MHW": 0.65, "MLLW": -0.65},  # micro-tidal ~1.3m range
}

# Now run each processing step
import numpy as np, rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling
from pathlib import Path

SITE      = "narrabeen"
AOI       = C.AOI[SITE]
SITE_CRS  = C.TARGET_CRS[SITE]
GEE_DIR   = Path("D:/cmui/data/gee_exports")
PROC_DIR  = Path("D:/cmui/data/processed")
OUT_DIR   = Path("D:/cmui/outputs/cmui_rasters")

DEM_PATH    = GEE_DIR / "dem_variance"   / f"dem_variance_{SITE}.tif"
NDWI_PATH   = GEE_DIR / "ndwi_variance"  / f"ndwi_variance_{SITE}.tif"
VEG_PATH    = GEE_DIR / "veg_canopy"     / f"veg_canopy_{SITE}.tif"
SENSOR_PATH = GEE_DIR / "sensor_variance"/ f"sensor_variance_{SITE}.tif"
TIDAL_PATH  = PROC_DIR / "tidal_variance" / f"tidal_variance_{SITE}.tif"
ZONE_PATH   = PROC_DIR / "zone_maps"     / f"zone_map_{SITE}.tif"
CMUI_PATH   = OUT_DIR  / f"cmui_{SITE}.tif"


def load_and_align(src_path, band, ref_profile, fill=0.0):
    with rasterio.open(src_path) as src:
        data = src.read(band).astype(np.float32)
        nd = src.nodata
        src_crs = src.crs
        src_t   = src.transform
    if nd is not None:
        data[data == nd] = np.nan
    dst = np.full((ref_profile["height"], ref_profile["width"]), np.nan, np.float32)
    reproject(source=data, destination=dst,
              src_transform=src_t, src_crs=src_crs,
              dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
              resampling=Resampling.bilinear)
    return np.where(np.isfinite(dst), dst, fill)


def step1_tidal():
    """Compute tidal variance using fallback datums (micro-tidal site)."""
    print("[1/3] Tidal variance (fallback datums — micro-tidal)...")
    w, s, e, n = AOI
    lat_mid = (s + n) / 2
    n_rows = int(round((n - s) * 111000 / 30))
    n_cols = int(round((e - w) * 111000 * np.cos(np.radians(lat_mid)) / 30))

    # Sydney Fort Denison: R_T=1.3m, eps_T=0.05m (GESLA-3 value)
    R_T   = abs(0.65 - (-0.65))   # = 1.30m
    eps_T = 0.05
    sigma2_tidal = (R_T * eps_T) ** 2   # = 0.00423 m^2
    print(f"  R_T={R_T:.2f}m  eps_T={eps_T:.3f}m  sigma^2_tidal={sigma2_tidal:.5f} m^2")

    tidal_grid = np.full((n_rows, n_cols), sigma2_tidal, dtype=np.float32)
    transform  = from_bounds(w, s, e, n, n_cols, n_rows)
    TIDAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver":"GTiff","dtype":"float32","count":3,
               "height":n_rows,"width":n_cols,
               "crs":CRS.from_epsg(int(SITE_CRS.split(":")[-1])),
               "transform":transform,"compress":"lzw"}
    with rasterio.open(TIDAL_PATH, "w", **profile) as dst:
        dst.write(tidal_grid, 1)
        dst.write(np.full_like(tidal_grid, R_T), 2)
        dst.write(np.full_like(tidal_grid, eps_T), 3)
    print(f"  Written: {TIDAL_PATH}")


def step2_zone():
    """Rule-based zone classification."""
    print("[2/3] Zone classification...")
    from scipy.ndimage import uniform_filter

    with rasterio.open(DEM_PATH) as src:
        ref = src.profile.copy()
    ref.update(dtype="float32", count=1)

    elevation = load_and_align(DEM_PATH,  1, ref)
    slope     = load_and_align(DEM_PATH,  3, ref)
    ndwi_var  = load_and_align(NDWI_PATH, 1, ref)
    ndvi      = load_and_align(VEG_PATH,  1, ref)

    t = C.ZONE_THRESHOLDS
    codes = np.zeros(elevation.shape, dtype=np.int8)

    # Narrabeen is entirely sandy beach (micro-tidal, low-gradient, low NDVI)
    tidal_mask  = (slope <= t["tidal_flat"]["slope_max_deg"]) & (ndvi <= t["tidal_flat"]["ndvi_max"]) & (ndwi_var > np.nanpercentile(ndwi_var, 40))
    sandy_mask  = (ndvi <= t["sandy_beach"]["ndvi_max"]) & (slope <= t["sandy_beach"]["slope_max_deg"]) & (~tidal_mask)
    rocky_mask  = (ndvi <= t["rocky_shore"]["ndvi_max"]) & (slope >= t["rocky_shore"]["slope_min_deg"])
    mangrove_mask = (ndvi >= t["mangrove"]["ndvi_min"]) & (elevation <= t["mangrove"]["elev_max_m"])

    codes[tidal_mask]    = 4
    codes[sandy_mask]    = 1
    codes[rocky_mask]    = 2
    codes[mangrove_mask] = 3

    conf = np.zeros_like(codes, dtype=np.float32)
    for code in [1,2,3,4,5]:
        b = (codes == code).astype(np.float32)
        lf = uniform_filter(b, size=3)
        conf[codes == code] = lf[codes == code]

    total = codes.size
    print(f"  Zone distribution:")
    for code, name in {0:"unclassified",1:"sandy_beach",2:"rocky_shore",3:"mangrove",4:"tidal_flat"}.items():
        n = int(np.sum(codes == code))
        print(f"    {name:15s} {n:8d} px  {100*n/total:.1f}%")

    ZONE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_p = ref.copy()
    out_p.update(dtype="float32", count=2, compress="lzw")
    with rasterio.open(ZONE_PATH, "w", **out_p) as dst:
        dst.write(codes.astype(np.float32), 1)
        dst.write(conf, 2)
    print(f"  Written: {ZONE_PATH}")


def step3_fusion():
    """CMUI weighted fusion."""
    print("[3/3] CMUI fusion...")
    from processing.utils.weights import get_weights

    ZONE_NAMES = {0:"unclassified",1:"sandy_beach",2:"rocky_shore",3:"mangrove",4:"tidal_flat",5:"coral_reef"}
    DEM_ACQ_YEAR = 2021
    CURRENT_YEAR = 2025

    with rasterio.open(DEM_PATH) as src:
        ref = src.profile.copy()
    ref.update(dtype="float32", count=1)

    elevation     = load_and_align(DEM_PATH,    1, ref)
    dem_roughness = load_and_align(DEM_PATH,    2, ref)
    sigma2_ndwi   = load_and_align(NDWI_PATH,   1, ref)
    ndvi          = load_and_align(VEG_PATH,    1, ref)
    canopy_height = load_and_align(VEG_PATH,    2, ref)
    sigma2_tidal  = load_and_align(TIDAL_PATH,  1, ref)
    sigma2_sensor = load_and_align(SENSOR_PATH, 1, ref)
    zone_codes    = load_and_align(ZONE_PATH,   1, ref, fill=0.0).astype(np.int8)

    sigma2_dem = dem_roughness**2 + C.COPERNICUS_RMSE_M**2
    beta = np.where(zone_codes == 3, C.BETA_MANGROVE, C.BETA_DEFAULT)
    sigma2_veg = (beta * ndvi * canopy_height)**2

    delta_t  = float(CURRENT_YEAR - DEM_ACQ_YEAR)
    sigma2_0 = dem_roughness**2
    tau = sigma2_0 * (np.exp(C.LAMBDA_STALENESS * delta_t) - 1.0)
    print(f"  DEM age: {delta_t:.0f}yr  tau mean={float(np.mean(sigma2_0)*(np.exp(C.LAMBDA_STALENESS*delta_t)-1)):.4f} m^2")

    h, w = sigma2_dem.shape
    cmui = np.zeros((h, w), dtype=np.float32)
    components = np.stack([sigma2_dem, sigma2_ndwi, sigma2_tidal, sigma2_veg, sigma2_sensor])

    for code, name in ZONE_NAMES.items():
        mask = zone_codes == code
        if not np.any(mask): continue
        weights = np.array([0.2]*5) if name == "unclassified" else get_weights(name)
        ws = np.zeros((h, w), np.float32)
        for i, wt in enumerate(weights):
            ws += wt * components[i]
        inner = np.maximum(ws + tau, 0.0)
        cmui[mask] = np.sqrt(inner[mask])

    valid = cmui[cmui > 0]
    print(f"  CMUI statistics ({SITE}):")
    print(f"    Mean={np.mean(valid):.4f}m  Std={np.std(valid):.4f}m  "
          f"Min={np.min(valid):.4f}m  Max={np.max(valid):.4f}m  P95={np.percentile(valid,95):.4f}m")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_p = ref.copy()
    out_p.update(dtype="float32", count=8, compress="lzw", nodata=-9999.0)
    with rasterio.open(CMUI_PATH, "w", **out_p) as dst:
        dst.write(cmui, 1)
        dst.write(sigma2_dem.astype(np.float32), 2)
        dst.write(sigma2_ndwi.astype(np.float32), 3)
        dst.write(sigma2_tidal.astype(np.float32), 4)
        dst.write(sigma2_veg.astype(np.float32), 5)
        dst.write(sigma2_sensor.astype(np.float32), 6)
        dst.write(tau.astype(np.float32), 7)
        dst.write(zone_codes.astype(np.float32), 8)
    print(f"  Written: {CMUI_PATH}")


def main():
    print(f"=== Narrabeen Processing Pipeline ===\n")
    for p in [DEM_PATH, NDWI_PATH, VEG_PATH, SENSOR_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"GEE export not found: {p}\nRun run_narrabeen_gee.py first.")
    step1_tidal()
    step2_zone()
    step3_fusion()
    print("\nDone. Run: python run_narrabeen_gps_validation.py")


if __name__ == "__main__":
    main()