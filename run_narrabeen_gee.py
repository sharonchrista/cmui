"""
run_narrabeen_gee.py

Computes all 4 GEE variance rasters for Narrabeen, Australia.
Downloads directly to D:/cmui/data/gee_exports/

AOI: [151.2958, -33.7390, 151.3122, -33.7013]  (W S E N, WGS84)
CRS: EPSG:32756  (UTM Zone 56S, covers Sydney area)

Narrabeen is micro-tidal (~0.5m range), sandy beach.
5 transects with 40+ years of monthly GPS surveys (Harley et al. 2016,
Nature Scientific Data). Used as real GPS validation for CMUI.

Run:
    python run_narrabeen_gee.py
"""

import sys, time, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ee, geemap, rasterio
from rasterio.merge import merge
from gee_scripts.utils.gee_auth import authenticate_and_initialise
from processing.utils.constants import (
    S2_COLLECTION, S2_CLOUD_THRESHOLD,
    L9_COLLECTION, L9_START,
    COPERNICUS_DEM, ETH_CANOPY_HEIGHT,
    TARGET_RES_M,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SITE       = "narrabeen"
AOI_BOUNDS = [151.2958, -33.7390, 151.3122, -33.7013]
SITE_CRS   = "EPSG:32756"   # UTM Zone 56S
START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"
L9_END     = "2024-12-31"
# Narrabeen dry season: Oct-Apr (Southern Hemisphere summer)
NDVI_START = "2022-10-01"
NDVI_END   = "2023-04-30"

GEE_DIR    = Path("D:/cmui/data/gee_exports")
MAX_RETRIES = 3
RETRY_DELAY = 10


def mask_s2(image):
    qa = image.select("QA60")
    return image.updateMask(
        qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0))
    )

def ndwi(image):
    return image.normalizedDifference(["B3","B8"]).rename("NDWI")

def ndvi(image):
    return image.normalizedDifference(["B8","B4"]).rename("NDVI")

def mask_l9(image):
    qa = image.select("QA_PIXEL")
    return image.updateMask(qa.bitwiseAnd(1<<3).eq(0).And(qa.bitwiseAnd(1<<4).eq(0)))

def l9_scale(image):
    return image.addBands(image.select("SR_B.").multiply(0.0000275).add(-0.2), overwrite=True)

def l9_ndwi(image):
    return image.normalizedDifference(["SR_B3","SR_B5"]).rename("NDWI")


def download(image, aoi, out_path, crs, scale, description):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_RETRIES+1):
        try:
            geemap.download_ee_image(
                image=image.clip(aoi), filename=str(out_path),
                scale=scale, crs=crs, region=aoi, overwrite=True,
            )
            if out_path.exists() and out_path.stat().st_size > 1024:
                print(f"  OK {description}: {out_path.stat().st_size/1e6:.1f} MB")
                return
            raise RuntimeError("file empty")
        except Exception as e:
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES: time.sleep(RETRY_DELAY)
            else: raise


def main():
    authenticate_and_initialise()
    aoi = ee.Geometry.Rectangle(AOI_BOUNDS)
    print(f"Site: {SITE}  |  CRS: {SITE_CRS}  |  AOI: {AOI_BOUNDS}")

    # 1. sigma^2_NDWI
    print("\n[1/4] NDWI temporal variance...")
    s2_coll = (ee.ImageCollection(S2_COLLECTION)
               .filterBounds(aoi).filterDate(START_DATE, END_DATE)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_THRESHOLD))
               .map(mask_s2).map(ndwi))
    print(f"  S2 images: {s2_coll.size().getInfo()}")
    ndwi_var = (s2_coll.reduce(ee.Reducer.variance()).toFloat().rename("NDWI_variance")
                .addBands(s2_coll.reduce(ee.Reducer.count()).toFloat().rename("NDWI_count")))
    download(ndwi_var, aoi,
             GEE_DIR/"ndwi_variance"/f"ndwi_variance_{SITE}.tif",
             SITE_CRS, TARGET_RES_M, "NDWI variance")

    # 2. sigma^2_DEM
    print("\n[2/4] DEM variance (Copernicus GLO-30)...")
    dem = (ee.ImageCollection(COPERNICUS_DEM).filterBounds(aoi)
           .select("DEM").mosaic()
           .setDefaultProjection(crs="EPSG:4326", scale=30))
    roughness = (dem.reduceNeighborhood(
        reducer=ee.Reducer.stdDev(),
        kernel=ee.Kernel.square(radius=1, units="pixels"))
        .toFloat().rename("DEM_roughness"))
    slope = ee.Terrain.slope(dem).toFloat().rename("DEM_slope")
    elev  = dem.toFloat().rename("DEM_elevation")
    dem_image = elev.addBands(roughness).addBands(slope)
    download(dem_image, aoi,
             GEE_DIR/"dem_variance"/f"dem_variance_{SITE}.tif",
             SITE_CRS, TARGET_RES_M, "DEM variance")

    # 3. Veg canopy (NDVI + ETH canopy height)
    print("\n[3/4] Vegetation canopy...")
    ndvi_coll = (ee.ImageCollection(S2_COLLECTION)
                 .filterBounds(aoi).filterDate(NDVI_START, NDVI_END)
                 .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_THRESHOLD))
                 .map(mask_s2).map(ndvi))
    print(f"  NDVI images: {ndvi_coll.size().getInfo()}")
    ndvi_med = ndvi_coll.median().toFloat().rename("NDVI_median")
    canopy   = ee.Image(ETH_CANOPY_HEIGHT).select("b1").toFloat().rename("canopy_height")
    veg_image = ndvi_med.addBands(canopy)
    download(veg_image, aoi,
             GEE_DIR/"veg_canopy"/f"veg_canopy_{SITE}.tif",
             SITE_CRS, TARGET_RES_M, "Veg canopy")

    # 4. sigma^2_sensor (S2 vs Landsat 9)
    print("\n[4/4] Cross-sensor variance...")
    s2_mean = (ee.ImageCollection(S2_COLLECTION)
               .filterBounds(aoi).filterDate(START_DATE, END_DATE)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_THRESHOLD))
               .map(mask_s2).map(ndwi).mean()
               .reproject(crs=SITE_CRS, scale=TARGET_RES_M)
               .toFloat().rename("s2_ndwi_mean"))
    l9_mean = (ee.ImageCollection(L9_COLLECTION)
               .filterBounds(aoi).filterDate(L9_START, L9_END)
               .map(mask_l9).map(l9_scale).map(l9_ndwi).mean()
               .toFloat().rename("l9_ndwi_mean"))
    l9_count = ee.ImageCollection(L9_COLLECTION).filterBounds(aoi).filterDate(L9_START, L9_END).size().getInfo()
    print(f"  L9 images: {l9_count}")
    combined = ee.ImageCollection([s2_mean.rename("NDWI"), l9_mean.rename("NDWI")])
    sensor_var = (combined.reduce(ee.Reducer.variance()).toFloat().rename("sensor_variance")
                  .addBands(s2_mean).addBands(l9_mean))
    download(sensor_var, aoi,
             GEE_DIR/"sensor_variance"/f"sensor_variance_{SITE}.tif",
             SITE_CRS, TARGET_RES_M, "Sensor variance")

    print(f"\nAll 4 GEE rasters downloaded for {SITE}.")
    print("Next: python run_narrabeen_processing.py")


if __name__ == "__main__":
    main()