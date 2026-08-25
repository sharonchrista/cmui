"""
02_dem_variance.py

Computes per-pixel DEM vertical variance (sigma^2_DEM) over the
SW Florida study area using Copernicus GLO-30 DEM via GEE.

For the Copernicus GLO-30 DEM, vertical variance is derived from:
  - Local terrain roughness (standard deviation within a moving window)
    as a proxy for within-pixel elevation variability
  - The published per-region RMSE of Copernicus GLO-30 as a scalar offset

Output: D:/cmui/data/gee_exports/dem_variance/dem_variance_florida.tif
  Band 1 - DEM_elevation   : raw elevation in metres (for zone classifier)
  Band 2 - DEM_roughness   : local terrain roughness (std dev, 3x3 kernel)
  Band 3 - DEM_slope       : slope in degrees (for zone classifier)

All bands Float32. Same 3x3 tile grid + retry strategy as script 01.

Run:
    python gee_scripts/02_dem_variance.py
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
    COPERNICUS_DEM,
    TARGET_CRS,
    TARGET_RES_M,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SITE        = "venice"
OUTPUT_DIR  = Path("D:/cmui/data/gee_exports/dem_variance")
OUTPUT_FILE = OUTPUT_DIR / f"dem_variance_{SITE}.tif"
TILES_DIR   = OUTPUT_DIR / "tiles"

TILE_COLS   = 3
TILE_ROWS   = 3
MAX_RETRIES = 3
RETRY_DELAY = 10

from processing.utils.constants import AOI
AOI_BOUNDS = AOI[SITE]


# ---------------------------------------------------------------------------
# Tile grid (reused from script 01)
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
# DEM processing
# ---------------------------------------------------------------------------

def build_dem_image(aoi: ee.Geometry) -> ee.Image:
    """
    Load Copernicus GLO-30 DEM, compute slope and terrain roughness.

    Roughness is the standard deviation of elevation within a 3x3
    pixel neighbourhood — a proxy for within-pixel vertical variability
    that serves as sigma^2_DEM in the CMUI formulation.

    Slope (degrees) is exported alongside elevation as it is needed
    by the geomorphic zone classifier (script 02_zone_classifier.py).

    Returns a 3-band Float32 image clipped to the AOI.
    """
    # Copernicus GLO-30: DEM band is named 'DEM'
    dem = (
        ee.ImageCollection(COPERNICUS_DEM)
        .filterBounds(aoi)
        .select("DEM")
        .mosaic()
        .setDefaultProjection(crs="EPSG:4326", scale=30)
    )

    # Terrain roughness: local std dev in a 3x3 pixel kernel
    # Acts as sigma^2_DEM — captures within-pixel elevation variability
    roughness = (
        dem.reduceNeighborhood(
            reducer=ee.Reducer.stdDev(),
            kernel=ee.Kernel.square(radius=1, units="pixels"),
        )
        .rename("DEM_roughness")
        .toFloat()
    )

    # Slope in degrees from GEE terrain analysis
    slope = (
        ee.Terrain.slope(dem)
        .rename("DEM_slope")
        .toFloat()
    )

    elevation = dem.rename("DEM_elevation").toFloat()

    return elevation.addBands(roughness).addBands(slope)


# ---------------------------------------------------------------------------
# Robust tile download (same pattern as script 01)
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
    print(f"Site      : {SITE}")
    print(f"DEM source: Copernicus GLO-30  ({COPERNICUS_DEM})")
    print(f"Outputs   : elevation, roughness (sigma^2_DEM proxy), slope")

    print("\nBuilding DEM image with roughness and slope...")
    dem_image = build_dem_image(full_aoi)

    tiles = build_tile_grid(AOI_BOUNDS, cols=TILE_COLS, rows=TILE_ROWS)
    print(f"\nDownloading {len(tiles)} tiles ({TILE_COLS}x{TILE_ROWS} grid)...")

    tile_paths = []
    for idx, tile_bounds in enumerate(tiles, start=1):
        print(f"  Tile {idx:02d}/{len(tiles)}: {[round(x, 3) for x in tile_bounds]}")
        path = download_tile_with_retry(
            image=dem_image,
            tile_bounds=tile_bounds,
            tile_path=TILES_DIR / f"tile_{idx:02d}.tif",
            crs=TARGET_CRS[SITE],
            scale=TARGET_RES_M,
        )
        tile_paths.append(path)

    print("\nMerging tiles...")
    merge_tiles(tile_paths, OUTPUT_FILE)

    print(f"\nDone. sigma^2_DEM saved to:\n  {OUTPUT_FILE}")
    print("Bands: 1=DEM_elevation  2=DEM_roughness  3=DEM_slope")
    print("Next step: python gee_scripts/03_veg_canopy.py")


if __name__ == "__main__":
    main()