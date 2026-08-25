"""
raster_ops.py

Reusable raster utility functions: reproject, clip, resample, align.
All functions operate on file paths and return file paths (GeoTIFF).
"""

from pathlib import Path
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


def reproject_raster(
    src_path: Path,
    dst_path: Path,
    target_crs: str,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """Reproject a raster to target_crs and write to dst_path."""
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
        )
        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                )
    return dst_path


def clip_raster_to_bounds(
    src_path: Path,
    dst_path: Path,
    bounds: tuple[float, float, float, float],
) -> Path:
    """
    Clip a raster to the given bounds (left, bottom, right, top) in the
    raster's native CRS. Writes result to dst_path.
    """
    from rasterio.mask import mask
    from shapely.geometry import box
    import geopandas as gpd

    aoi_geom = [box(*bounds).__geo_interface__]
    with rasterio.open(src_path) as src:
        clipped, clipped_transform = mask(src, aoi_geom, crop=True)
        profile = src.profile.copy()
        profile.update(
            transform=clipped_transform,
            height=clipped.shape[1],
            width=clipped.shape[2],
        )
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(clipped)
    return dst_path
