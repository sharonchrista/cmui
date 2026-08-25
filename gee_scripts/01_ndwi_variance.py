"""
01_ndwi_variance.py

Computes per-pixel NDWI temporal variance (sigma^2_NDWI) over the
SW Florida study area using Sentinel-2 SR time series via GEE.

Downloads DIRECTLY to D:/cmui/data/gee_exports/ndwi_variance/
No Google Drive required.

Uses a 3x3 tile grid (~6 MB per tile) with 3 retry attempts per tile
to reliably stay under the GEE download size limit.

Run:
    python gee_scripts/01_ndwi_variance.py
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
    AOI,
    S2_COLLECTION,
    S2_CLOUD_THRESHOLD,
    TARGET_CRS,
    TARGET_RES_M,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SITE       = "venice"
START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"

OUTPUT_DIR  = Path("D:/cmui/data/gee_exports/ndwi_variance")
OUTPUT_FILE = OUTPUT_DIR / f"ndwi_variance_{SITE}.tif"
TILES_DIR   = OUTPUT_DIR / "tiles"

# 3x3 grid -> ~6 MB per tile, comfortably under the 48 MB GEE limit
TILE_COLS   = 3
TILE_ROWS   = 3
MAX_RETRIES = 3
RETRY_DELAY = 10   # seconds between retry attempts


# ---------------------------------------------------------------------------
# Tile grid builder
# ---------------------------------------------------------------------------

def build_tile_grid(
    bounds: list[float],
    cols: int,
    rows: int,
) -> list[list[float]]:
    """
    Split [W, S, E, N] into a cols x rows grid.
    Returns list of [W, S, E, N] sub-bounds.
    """
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
# GEE image computation
# ---------------------------------------------------------------------------

def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Mask clouds and cirrus via QA60 band (bits 10 and 11)."""
    qa = image.select("QA60")
    mask = (
        qa.bitwiseAnd(1 << 10).eq(0)
        .And(qa.bitwiseAnd(1 << 11).eq(0))
    )
    return image.updateMask(mask)


def compute_ndwi(image: ee.Image) -> ee.Image:
    """NDWI = (B3 Green - B8 NIR) / (B3 + B8)  [McFeeters 1996]."""
    return image.normalizedDifference(["B3", "B8"]).rename("NDWI")


def build_ndwi_collection(aoi: ee.Geometry) -> ee.ImageCollection:
    """Filter, cloud-mask, and compute NDWI for each S2 image."""
    return (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_THRESHOLD))
        .map(mask_s2_clouds)
        .map(compute_ndwi)
    )


def compute_variance_and_count(collection: ee.ImageCollection) -> ee.Image:
    """
    Returns a 2-band Float32 image.
      Band 1 - NDWI_variance : temporal variance across all acquisitions
      Band 2 - NDWI_count    : number of valid observations per pixel
    Both bands cast to Float32 to avoid Float64/UInt32 type mismatch.
    """
    variance = (
        collection.reduce(ee.Reducer.variance())
        .toFloat()
        .rename("NDWI_variance")
    )
    count = (
        collection.reduce(ee.Reducer.count())
        .toFloat()
        .rename("NDWI_count")
    )
    return variance.addBands(count)


# ---------------------------------------------------------------------------
# Robust tile download with retry
# ---------------------------------------------------------------------------

def download_tile_with_retry(
    image: ee.Image,
    tile_bounds: list[float],
    tile_path: Path,
    crs: str,
    scale: int,
    max_retries: int = MAX_RETRIES,
) -> Path:
    """
    Download one tile using geemap.download_ee_image() with retry logic.

    geemap.download_ee_image() uses the GEE REST getPixels API internally
    and is more robust than ee_export_image() for local downloads.

    Falls back to geemap.ee_export_image() if download_ee_image unavailable.
    """
    aoi = ee.Geometry.Rectangle(tile_bounds)
    tile_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            # Primary method: download_ee_image (geemap >= 0.20)
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
                # Fallback for older geemap versions
                geemap.ee_export_image(
                    ee_object=image.clip(aoi),
                    filename=str(tile_path),
                    scale=scale,
                    crs=crs,
                    region=aoi,
                    file_per_band=False,
                )

            # Verify the file exists and is non-empty
            if tile_path.exists() and tile_path.stat().st_size > 1024:
                size_mb = tile_path.stat().st_size / (1024 * 1024)
                print(f"    OK  {tile_path.name}  ({size_mb:.1f} MB)")
                return tile_path
            else:
                raise RuntimeError(
                    f"File missing or empty after download attempt {attempt}"
                )

        except Exception as exc:
            print(f"    Attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                print(f"    Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(
                    f"All {max_retries} download attempts failed for tile "
                    f"{tile_bounds}. Check internet connection and GEE quota."
                ) from exc


# ---------------------------------------------------------------------------
# Tile merge
# ---------------------------------------------------------------------------

def merge_tiles(tile_paths: list[Path], output_path: Path) -> None:
    """
    Merge tile GeoTIFFs into a single output GeoTIFF with rasterio.
    Removes the tiles directory after successful merge.
    """
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
    print(f"Site       : {SITE}")
    print(f"Date range : {START_DATE} to {END_DATE}")
    print(f"Tile grid  : {TILE_COLS} x {TILE_ROWS} = {TILE_COLS * TILE_ROWS} tiles")

    print("\nBuilding Sentinel-2 NDWI collection...")
    collection = build_ndwi_collection(full_aoi)
    image_count = collection.size().getInfo()
    print(f"Images after cloud filter: {image_count}")

    if image_count == 0:
        print("No images found. Check AOI and date range.")
        sys.exit(1)

    print("Computing variance and count on GEE cloud...")
    output_image = compute_variance_and_count(collection)

    from processing.utils.constants import AOI
    AOI_BOUNDS = AOI[SITE]
    tiles = build_tile_grid(AOI_BOUNDS, cols=TILE_COLS, rows=TILE_ROWS)

    print(f"\nDownloading {len(tiles)} tiles to {TILES_DIR}")
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

    print(f"\nDone. sigma^2_NDWI saved to:\n  {OUTPUT_FILE}")
    print("Next step: python gee_scripts/02_dem_variance.py")


if __name__ == "__main__":
    main()
