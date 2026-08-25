"""
04_sensor_variance.py

Computes per-pixel inter-sensor NDWI disagreement (sigma^2_sensor).

Run:
    python gee_scripts/04_sensor_variance.py
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
    L9_COLLECTION,
    L9_START,
    TARGET_CRS,
    TARGET_RES_M,
    AOI,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SITE        = "venice"      # change to "venice" for validation site
OUTPUT_DIR  = Path("D:/cmui/data/gee_exports/sensor_variance")
OUTPUT_FILE = OUTPUT_DIR / f"sensor_variance_{SITE}.tif"
TILES_DIR   = OUTPUT_DIR / "tiles"

TILE_COLS   = 3
TILE_ROWS   = 3
MAX_RETRIES = 3
RETRY_DELAY = 10

AOI_BOUNDS  = AOI[SITE]

START_DATE  = "2019-01-01"
END_DATE    = "2024-12-31"
L9_END      = "2024-12-31"

# Resolve CRS string for this site
SITE_CRS = TARGET_CRS[SITE]


# ---------------------------------------------------------------------------
# Tile grid
# ---------------------------------------------------------------------------

def build_tile_grid(bounds: list[float], cols: int, rows: int) -> list[list[float]]:
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
# Sentinel-2 NDWI
# ---------------------------------------------------------------------------

def mask_s2_clouds(image: ee.Image) -> ee.Image:
    qa = image.select("QA60")
    mask = (
        qa.bitwiseAnd(1 << 10).eq(0)
        .And(qa.bitwiseAnd(1 << 11).eq(0))
    )
    return image.updateMask(mask)


def s2_ndwi(image: ee.Image) -> ee.Image:
    return image.normalizedDifference(["B3", "B8"]).rename("NDWI")


def build_s2_ndwi_mean(aoi: ee.Geometry) -> ee.Image:
    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_THRESHOLD))
        .map(mask_s2_clouds)
        .map(s2_ndwi)
    )
    count = collection.size().getInfo()
    print(f"  Sentinel-2 images: {count}")
    return (
        collection.mean()
        .reproject(crs=SITE_CRS, scale=TARGET_RES_M)
        .toFloat()
        .rename("s2_ndwi_mean")
    )


# ---------------------------------------------------------------------------
# Landsat 9 NDWI
# ---------------------------------------------------------------------------

def mask_l9_clouds(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    mask = (
        qa.bitwiseAnd(1 << 3).eq(0)
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    )
    return image.updateMask(mask)


def l9_scale_sr(image: ee.Image) -> ee.Image:
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    return image.addBands(optical, overwrite=True)


def l9_ndwi(image: ee.Image) -> ee.Image:
    return image.normalizedDifference(["SR_B3", "SR_B5"]).rename("NDWI")


def build_l9_ndwi_mean(aoi: ee.Geometry) -> ee.Image:
    collection = (
        ee.ImageCollection(L9_COLLECTION)
        .filterBounds(aoi)
        .filterDate(L9_START, L9_END)
        .map(mask_l9_clouds)
        .map(l9_scale_sr)
        .map(l9_ndwi)
    )
    count = collection.size().getInfo()
    print(f"  Landsat 9 images : {count}")
    return (
        collection.mean()
        .toFloat()
        .rename("l9_ndwi_mean")
    )


# ---------------------------------------------------------------------------
# Cross-sensor variance
# ---------------------------------------------------------------------------

def compute_sensor_variance(s2_mean: ee.Image, l9_mean: ee.Image) -> ee.Image:
    combined = ee.ImageCollection([
        s2_mean.rename("NDWI"),
        l9_mean.rename("NDWI"),
    ])
    variance = (
        combined.reduce(ee.Reducer.variance())
        .toFloat()
        .rename("sensor_variance")
    )
    return variance.addBands(s2_mean).addBands(l9_mean)


# ---------------------------------------------------------------------------
# Download with retry
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
                raise RuntimeError(f"File missing or empty after attempt {attempt}")

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
# Merge tiles
# ---------------------------------------------------------------------------

def merge_tiles(tile_paths: list[Path], output_path: Path) -> None:
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(datasets)
    profile = datasets[0].profile.copy()
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform)
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
    print(f"CRS       : {SITE_CRS}")
    print(f"S2 window : {START_DATE} to {END_DATE}")
    print(f"L9 window : {L9_START} to {L9_END}")

    print("\nBuilding Sentinel-2 NDWI mean composite...")
    s2_mean = build_s2_ndwi_mean(full_aoi)

    print("Building Landsat 9 NDWI mean composite...")
    l9_mean = build_l9_ndwi_mean(full_aoi)

    print("Computing cross-sensor NDWI variance...")
    output_image = compute_sensor_variance(s2_mean, l9_mean)

    tiles = build_tile_grid(AOI_BOUNDS, cols=TILE_COLS, rows=TILE_ROWS)
    print(f"\nDownloading {len(tiles)} tiles ({TILE_COLS}x{TILE_ROWS} grid)...")

    tile_paths = []
    for idx, tile_bounds in enumerate(tiles, start=1):
        print(f"  Tile {idx:02d}/{len(tiles)}: {[round(x, 3) for x in tile_bounds]}")
        path = download_tile_with_retry(
            image=output_image,
            tile_bounds=tile_bounds,
            tile_path=TILES_DIR / f"tile_{idx:02d}.tif",
            crs=SITE_CRS,        # string, not dict
            scale=TARGET_RES_M,
        )
        tile_paths.append(path)

    print("\nMerging tiles...")
    merge_tiles(tile_paths, OUTPUT_FILE)

    print(f"\nDone. sigma^2_sensor saved to:\n  {OUTPUT_FILE}")
    print("Bands: 1=sensor_variance  2=s2_ndwi_mean  3=l9_ndwi_mean")
    print("Next step: python processing/01_tidal_model.py")


if __name__ == "__main__":
    main()