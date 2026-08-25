"""
aoi.py

Study area bounding box definitions for GEE scripts.
Returns ee.Geometry objects ready to pass to GEE API calls.
"""

import ee
from processing.utils.constants import AOI


def get_aoi(site: str) -> ee.Geometry:
    """
    Return an ee.Geometry.Rectangle for the named study site.
    Coordinates are [West, South, East, North] in WGS84.
    """
    if site not in AOI:
        raise ValueError(f"Unknown site '{site}'. Available: {list(AOI.keys())}")
    coords = AOI[site]
    return ee.Geometry.Rectangle(coords)
