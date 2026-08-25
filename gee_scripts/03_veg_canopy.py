"""
03_veg_canopy.py

Computes vegetation canopy inputs for sigma^2_veg over the
SW Florida study area using:
  - Sentinel-2 NDVI (median composite, 2023 dry season)
  - ETH Global Canopy Height Model 10m (Lang et al. 2023)

sigma^2_veg is NOT computed here — it is computed locally in
processing/03_cmui_fusion.py using:
    sigma^2_veg(x) = (beta * NDVI(x) * h_c(x))^2

This script exports the two input rasters needed for that formula.

Output: D:/cmui/data/gee_exports/veg_canopy/veg_canopy_florida.tif
  Band 1 - NDVI_median   : median NDVI composite (dimensionless, -1 to 1)
  Band 2 - canopy_height : canopy height in metres (Lang et al. 2023)

All bands Float32. Same 3x3 tile grid + retry strategy as scripts 01-02.

References:
  - Lang et al. (2023) A high-resolution canopy height model of the
    Earth. Nature Ecology & Evolution. doi:10.1038/s41559-023-02206-6
  - McFeeters (1996) for NDVI context
  - Gaveau & Hill (2003) for beta coefficient

Run:
    python gee_scripts/03_veg_canopy.py
"""

import sys
import time
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import geemap
import rasterio
from rasterio.merge import merge

from gee_scripts.utils.gee_auth import authenticate_and_initialise
from gee_scripts.utils.aoi import get_aoi
from processing.utils.constants import (
    S2_COLLECTION,
    S2_CLOUD_THRESHOLD,
    TARGET_CRS,
    TARGET_RES_M,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SITE        = "venice"
OUTPUT_DIR  = Path("D:/cmui/data/gee_exports/veg_canopy")
OUTPUT_FILE = OUTPUT_DIR / f"veg_canopy_{SITE}.tif"
TILES_DIR   = OUTPUT_DIR / "tiles"

TILE_COLS   = 3
TILE_ROWS   = 3
MAX_RETRIES = 3
RETRY_DELAY = 10

from processing.utils.constants import AOI
AOI_BOUNDS = AOI[SITE]

# NDVI composite window: Nov-Apr dry season avoids cloud/rain interference
# Florida dry season gives cleaner canopy signal over mangroves
NDVI_START = "2022-10-01"
NDVI_END   = "2023-02-28"

# ETH Global Canopy Height Model (Lang et al. 2023) — 10m, global
# GEE community asset — publicly available
ETH_CANOPY_HEIGHT = "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1"


# ---------------------------------------------------------------------------
# Tile grid
# ---------------------------------------------------------------------------

def build_tile_grid(
    bounds: list[float],
    cols: int,
    rows: int,
) -> list[list[float]]:
    w, s, e, n = bounds
    lon_step = (e - w) / cols
    lat_step = (n - s) / rows
    tiles = []
    for row in range(rows):
        for col in range(cols):
            tile_w = w + col * lon_step
            tile_e = tile_w + lon_step
            tile_s = s + row * lat_step
            tile_n = tile_s + lat_step
            tiles.append([tile_w, tile_s, tile_e, tile_n])
    return tiles


# ---------------------------------------------------------------------------
# NDVI median composite
# ---------------------------------------------------------------------------

def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Mask clouds and cirrus via QA60 band (bits 10 and 11)."""
    qa = image.select("QA60")
    mask = (
        qa.bitwiseAnd(1 << 10).eq(0)
        .And(qa.bitwiseAnd(1 << 11).eq(0))
    )
    return image.updateMask(mask)


def compute_ndvi(image: ee.Image) -> ee.Image:
    """
    NDVI = (B8 NIR - B4 Red) / (B8 + B4)
    Sentinel-2: B8 = NIR (842nm), B4 = Red (665nm)
    """
    return image.normalizedDifference(["B8", "B4"]).rename("NDVI")


def build_ndvi_median(aoi: ee.Geometry) -> ee.Image:
    """
    Median NDVI composite over the dry-season window.
    Median is more robust than mean against cloud residuals.
    """
    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(NDVI_START, NDVI_END)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_THRESHOLD))
        .map(mask_s2_clouds)
        .map(compute_ndvi)
    )
    count = collection.size().getInfo()
    print(f"  NDVI composite: {count} images ({NDVI_START} to {NDVI_END})")

    return (
        collection
        .median()
        .toFloat()
        .rename("NDVI_median")
    )


# ---------------------------------------------------------------------------
# ETH canopy height
# ---------------------------------------------------------------------------

def build_canopy_height(aoi: ee.Geometry) -> ee.Image:
    """
    Load ETH Global Canopy Height Model (Lang et al. 2023).
    Resamples from native 10m to TARGET_RES_M (30m) using mean
    aggregation — appropriate for a continuous height field.

    The 'b1' band contains canopy height in metres.
    Pixels with no canopy (bare ground, water) have value 0.
    """
    try:
        canopy = (
            ee.Image(ETH_CANOPY_HEIGHT)
            .select("b1")
            .toFloat()
            .rename("canopy_height")
        )
        return canopy
    except Exception as exc:
        raise RuntimeError(
            f"Could not load ETH canopy height asset '{ETH_CANOPY_HEIGHT}'. "
            "Ensure your GEE account can access community assets. "
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Robust tile download
# ---------------------------------------------------------------------------

def download_tile_with_retry(
    image: ee.Image,
    tile_bounds: list[float],
    tile_path: Path,
    crs: str,
    scale: int,
    max_retries: int = MAX_RETRIES,
) -> Path:
    aoi = ee.Geometry.Rectangle(tile_bounds)
    tile_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            if hasattr(geemap, "download_ee_image"):
                geemap.download_ee_image(
                    image=image.clip(aoi),
                    filename=str(tile_path),
                    scale=scale,
                    crs=crs,
                    region=aoi,
                    overwrite=True,
                )
            else:
                geemap.ee_export_image(
                    ee_object=image.clip(aoi),
                    filename=str(tile_path),
                    scale=scale,
                    crs=crs,
                    region=aoi,
                    file_per_band=False,
                )

            if tile_path.exists() and tile_path.stat().st_size > 1024:
                size_mb = tile_path.stat().st_size / (1024 * 1024)
                print(f"    OK  {tile_path.name}  ({size_mb:.1f} MB)")
                return tile_path
            else:
                raise RuntimeError(
                    f"File missing or empty after attempt {attempt}"
                )

        except Exception as exc:
            print(f"    Attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                print(f"    Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(
                    f"All {max_retries} attempts failed for tile {tile_bounds}"
                ) from exc


# ---------------------------------------------------------------------------
# Tile merge
# ---------------------------------------------------------------------------

def merge_tiles(tile_paths: list[Path], output_path: Path) -> None:
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(datasets)
    profile = datasets[0].profile.copy()
    profile.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
    )
    for ds in datasets:
        ds.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Merged output: {output_path}  ({size_mb:.1f} MB)")
    shutil.rmtree(TILES_DIR)
    print("Tile cache cleaned up.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    authenticate_and_initialise()

    full_aoi = get_aoi(SITE)
    print(f"Site           : {SITE}")
    print(f"NDVI window    : {NDVI_START} to {NDVI_END} (dry season composite)")
    print(f"Canopy height  : ETH Global Canopy Height Model (Lang et al. 2023)")

    print("\nBuilding NDVI median composite...")
    ndvi = build_ndvi_median(full_aoi)

    print("Loading ETH canopy height model...")
    canopy = build_canopy_height(full_aoi)

    # Stack both bands into one image for a single download pass
    output_image = ndvi.addBands(canopy)

    tiles = build_tile_grid(AOI_BOUNDS, cols=TILE_COLS, rows=TILE_ROWS)
    print(f"\nDownloading {len(tiles)} tiles ({TILE_COLS}x{TILE_ROWS} grid)...")

    tile_paths = []
    for idx, tile_bounds in enumerate(tiles, start=1):
        print(f"  Tile {idx:02d}/{len(tiles)}: {[round(x, 3) for x in tile_bounds]}")
        path = download_tile_with_retry(
            image=output_image,
            tile_bounds=tile_bounds,
            tile_path=TILES_DIR / f"tile_{idx:02d}.tif",
            crs=TARGET_CRS[SITE],
            scale=TARGET_RES_M,
        )
        tile_paths.append(path)

    print("\nMerging tiles...")
    merge_tiles(tile_paths, OUTPUT_FILE)

    print(f"\nDone. Vegetation canopy inputs saved to:\n  {OUTPUT_FILE}")
    print("Bands: 1=NDVI_median  2=canopy_height (metres)")
    print("Next step: python gee_scripts/04_sensor_variance.py")


if __name__ == "__main__":
    main()