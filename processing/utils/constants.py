"""
constants.py

Central configuration for the CMUI project.

Study design:
  PRIMARY site   : Sundarbans mangrove delta, West Bengal, India
                   Demonstrates CMUI across all five geomorphic zones
                   Macro-tidal (3-5m range), data-sparse, high uncertainty

  VALIDATION site: Venice Lagoon + North Adriatic coast, Italy
                   Cross-validates CMUI predictions against published
                   European ground truth (ISPRA, GESLA-3)
                   Micro-tidal (0.3-0.7m range), data-rich, low uncertainty

The contrast between these two sites — different tidal regimes, data
availability, and geomorphic complexity — is the paper's core argument
for CMUI's cross-regional generalisability.
"""

# ---------------------------------------------------------------------------
# Study site bounding boxes [West, South, East, North] in WGS84
# ---------------------------------------------------------------------------

AOI: dict[str, list[float]] = {
    # Sundarbans delta, West Bengal, India
    # Covers Sagar Island, Hooghly estuary mouth, major tidal creek networks
    # and the Indian Sundarbans mangrove forest
    "sundarbans": [88.0, 21.5, 89.5, 22.5],

    # Venice Lagoon + northern Adriatic coast, Italy
    # Covers the Venice Lagoon, Po delta tidal flats, and Adriatic sandy coast
    "venice": [12.0, 45.0, 13.2, 45.8],
}

# Convenience aliases used in scripts
PRIMARY_SITE    = "sundarbans"
VALIDATION_SITE = "venice"

# ---------------------------------------------------------------------------
# Output coordinate reference systems
# (one per site — different UTM zones)
# ---------------------------------------------------------------------------

TARGET_CRS: dict[str, str] = {
    "sundarbans": "EPSG:32645",   # UTM Zone 45N covers West Bengal
    "venice":     "EPSG:32632",   # UTM Zone 32N covers northern Italy
}

TARGET_RES_M: int = 30   # spatial resolution in metres (Copernicus DEM)

# ---------------------------------------------------------------------------
# CMUI parameters
# ---------------------------------------------------------------------------

# Temporal staleness decay rate
# lambda = ln(2) / 5 -> staleness contribution doubles every 5 years
# consistent with median global shoreline change rates (Luijendijk et al. 2018)
LAMBDA_STALENESS: float = 0.1386

# Vegetation canopy penetration bias coefficient
# initialised from published mangrove values (Gaveau & Hill 2003)
BETA_MANGROVE: float = 0.4
BETA_DEFAULT:  float = 0.2

# Copernicus GLO-30 published vertical RMSE (metres)
# Used to scale sigma^2_DEM where airborne LiDAR is unavailable
COPERNICUS_RMSE_M: float = 4.0

# ---------------------------------------------------------------------------
# Zone weight matrix [w_DEM, w_NDWI, w_tidal, w_veg, w_sensor]
# Weights within each zone sum to 1.0
# Source: Table 2, CMUI paper (SIGSPATIAL 2026)
# ---------------------------------------------------------------------------

ZONE_WEIGHTS: dict[str, list[float]] = {
    "sandy_beach": [0.30, 0.30, 0.20, 0.05, 0.15],
    "rocky_shore": [0.40, 0.15, 0.10, 0.05, 0.30],
    "mangrove":    [0.20, 0.15, 0.30, 0.25, 0.10],
    "tidal_flat":  [0.20, 0.35, 0.30, 0.05, 0.10],
    "coral_reef":  [0.25, 0.25, 0.20, 0.05, 0.25],
}

# ---------------------------------------------------------------------------
# Zone classifier thresholds (rule-based, from Sentinel-2 + DEM)
# ---------------------------------------------------------------------------

ZONE_THRESHOLDS: dict[str, dict] = {
    "sandy_beach": {"ndvi_max": 0.15, "slope_max_deg": 2.0},
    "rocky_shore": {"ndvi_max": 0.15, "slope_min_deg": 10.0},
    "mangrove":    {"ndvi_min": 0.40, "elev_max_m": 5.0},
    "tidal_flat":  {"slope_max_deg": 1.0, "ndvi_max": 0.15},
    "coral_reef":  {"ndvi_max": 0.10, "bath_min_m": 0.0, "bath_max_m": 20.0},
}

# ---------------------------------------------------------------------------
# GEE collection IDs
# ---------------------------------------------------------------------------

S2_COLLECTION:     str = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_THRESHOLD: int = 20

L9_COLLECTION: str = "LANDSAT/LC09/C02/T1_L2"
L9_START:      str = "2022-01-01"

GEDI_COLLECTION: str = "LARSE/GEDI/GEDI02_A_002_MONTHLY"
COPERNICUS_DEM:  str = "COPERNICUS/DEM/GLO30"

# ---------------------------------------------------------------------------
# GEE image asset paths
# ---------------------------------------------------------------------------

# ETH Global Canopy Height Model 10m 2020 (Lang et al. 2023)
# Verified path: GEE Community Catalog, 2024
ETH_CANOPY_HEIGHT:    str = "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1"
ETH_CANOPY_HEIGHT_SD: str = "users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1"

# ---------------------------------------------------------------------------
# Tide gauge stations
# Source: GESLA-3 (Global Extreme Sea Level Analysis, version 3)
# https://gesla787883612.wordpress.com/
#
# For Sundarbans: nearest quality-controlled GESLA-3 gauges
# For Venice: ISPRA national tide gauge network via GESLA-3
# ---------------------------------------------------------------------------

TIDE_GAUGES: dict[str, dict] = {
    "sundarbans": {
        # Diamond Harbour (Hooghly River, 80km from Sundarbans)
        # GESLA-3 station ID: DIAMOND_HARBOUR_INDIA
        # Coords: 88.19E, 22.19N
        "diamond_harbour": {
            "gesla_id": "diamond-harbour-india",
            "lon": 88.19,
            "lat": 22.19,
        },
        # Haldia (Hooghly River mouth)
        # GESLA-3 station ID: HALDIA_INDIA
        # Coords: 88.07E, 22.02N
        "haldia": {
            "gesla_id": "haldia-india",
            "lon": 88.07,
            "lat": 22.02,
        },
    },
    "venice": {
        # Punta della Salute, Venice (primary ISPRA gauge)
        # GESLA-3 station ID: VENEZIA-PS
        # Coords: 12.338E, 45.433N
        "venezia_ps": {
            "gesla_id": "venezia-punta-salute-italy",
            "lon": 12.338,
            "lat": 45.433,
        },
        # Chioggia (southern Venice Lagoon entrance)
        # Coords: 12.278E, 45.218N
        "chioggia": {
            "gesla_id": "chioggia-italy",
            "lon": 12.278,
            "lat": 45.218,
        },
    },
}

# ---------------------------------------------------------------------------
# Published tidal datums for Sundarbans and Venice
# (fallback values if GESLA-3 API is unavailable)
# Sources:
#   Sundarbans: Unnikrishnan & Shankar (2007) Est. Coast. Shelf Sci.
#   Venice:     ISPRA Annuario dei Dati Ambientali 2022
# ---------------------------------------------------------------------------

FALLBACK_TIDAL_DATUMS: dict[str, dict] = {
    "sundarbans": {
        "diamond_harbour": {"MHW": 2.50,  "MLLW": -2.30},  # metres, spring range ~4.8m
        "haldia":          {"MHW": 2.30,  "MLLW": -2.10},
    },
    "venice": {
        "venezia_ps": {"MHW": 0.35, "MLLW": -0.35},   # metres, range ~0.70m
        "chioggia":   {"MHW": 0.30, "MLLW": -0.30},
    },
}

# ---------------------------------------------------------------------------
# GESLA-3 data directory (download once from gesla787883612.wordpress.com)
# Place extracted files here before running processing/01_tidal_model.py
# ---------------------------------------------------------------------------

GESLA3_DIR: str = "D:/cmui/data/raw/tide_gauge/gesla3"
NOAA_API: str = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

AOI["narrabeen"] = [151.2958, -33.7390, 151.3122, -33.7013]
AOI["florida"]   = [-81.8, 25.6, -81.0, 26.2]
TARGET_CRS["narrabeen"] = "EPSG:32756"
TARGET_CRS["florida"]   = "EPSG:32617"
FALLBACK_TIDAL_DATUMS["narrabeen"] = {"sydney_fort_denison": {"MHW": 0.65, "MLLW": -0.65}}
FALLBACK_TIDAL_DATUMS["florida"]   = {"naples": {"MHW": 0.47, "MLLW": -0.47}, "fort_myers": {"MHW": 0.44, "MLLW": -0.44}, "key_west": {"MHW": 0.25, "MLLW": -0.25}}
TIDE_GAUGES["narrabeen"] = {"sydney_fort_denison": {"gesla_id": "sydney-fort-denison-australia", "lon": 151.225, "lat": -33.854}}
TIDE_GAUGES["florida"]   = {"naples": {"noaa_id": 8725110, "lon": -81.808, "lat": 26.132}, "fort_myers": {"noaa_id": 8725520, "lon": -81.867, "lat": 26.647}, "key_west": {"noaa_id": 8724580, "lon": -81.808, "lat": 24.555}}
