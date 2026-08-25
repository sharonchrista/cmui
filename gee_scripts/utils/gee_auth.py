"""
gee_auth.py

Authenticate with Google Earth Engine.
Run once before executing any GEE script:
    python gee_scripts/utils/gee_auth.py
"""

import ee


def authenticate_and_initialise(project: str = "") -> None:
    """
    Authenticate with GEE and initialise the API.

    On first run this opens a browser for OAuth consent.
    On subsequent runs the stored credentials are reused.

    Args:
        project: GEE cloud project ID. Leave empty to use the default.
    """
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        print("GEE initialised successfully.")
    except ee.EEException:
        print("GEE credentials not found. Starting authentication flow...")
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        print("GEE authenticated and initialised.")


if __name__ == "__main__":
    authenticate_and_initialise()
"""
gee_auth.py

Authenticate and initialise Google Earth Engine for the CMUI project.

Run once before executing any GEE script:
    python gee_scripts/utils/gee_auth.py

On first run this opens a browser for OAuth consent.
On subsequent runs stored credentials are reused automatically.
"""

import ee


GEE_PROJECT = "black-octagon-291810"


def authenticate_and_initialise(project: str = GEE_PROJECT) -> None:
    """
    Authenticate with GEE and initialise the API.

    Args:
        project: GEE cloud project ID. Defaults to the CMUI project.
    """
    try:
        ee.Initialize(project=project)
        print(f"GEE initialised. Project: {project}")
    except ee.EEException:
        print("GEE credentials not found. Starting authentication flow...")
        ee.Authenticate()
        ee.Initialize(project=project)
        print(f"GEE authenticated and initialised. Project: {project}")


if __name__ == "__main__":
    authenticate_and_initialise()
    # Quick sanity check — print the Sentinel-2 collection size for Florida AOI
    aoi = ee.Geometry.Rectangle([-81.8, 25.6, -81.0, 26.2])
    count = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate("2023-01-01", "2023-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .size()
        .getInfo()
    )
    print(f"Sanity check — Sentinel-2 images over Florida AOI in 2023: {count}")
    print("GEE connection confirmed. You are ready to run the pipeline scripts.")