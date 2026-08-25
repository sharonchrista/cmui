# CMUI — Coastal Measurement Uncertainty Index

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-ready-green.svg)](https://earthengine.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A spatially explicit, per-pixel coastal measurement uncertainty index that fuses five Earth observation variance components through a geomorphic-zone-adaptive weighting scheme.

> **Paper:** CMUI: A Spatially Explicit Coastal Measurement Uncertainty Index for Multi-Source Earth Observation Data Fusion  

---

## What CMUI Does

CMUI computes the total positional measurement uncertainty (metres) at each 30 m coastal pixel by fusing:

| Component | Source | What it captures |
|-----------|--------|-----------------|
| σ²_DEM | Copernicus GLO-30 | Elevation error and terrain roughness |
| σ²_NDWI | Sentinel-2 time series | Spectral water index temporal variability |
| σ²_tidal | GESLA-3 / NOAA CO-OPS | Tidal prediction residuals |
| σ²_veg | ETH Canopy Height + NDVI | Vegetation canopy penetration bias |
| σ²_sensor | Sentinel-2 vs Landsat 9 | Cross-sensor NDWI disagreement |

Weights are assigned per-pixel based on geomorphic zone (mangrove, tidal flat, sandy beach, rocky shore, coral reef), and a temporal staleness penalty τ(t) captures the declining reliability of older DEM acquisitions.

---

## Study Sites

| Site | Role | Tidal regime |
|------|------|-------------|
| Sundarbans, India | Primary | Macro-tidal (4.8 m) |
| Venice Lagoon, Italy | Validation 1 | Micro-tidal (0.7 m) |
| Narrabeen, Australia | Validation 2 | Micro-tidal (1.3 m) |
| SW Florida, USA | Validation 3 | Micro-meso-tidal (0.5–1.0 m) |

---

## Repository Structure

```
cmui/
├── gee_scripts/                 # Google Earth Engine Python API scripts
│   ├── utils/
│   │   ├── gee_auth.py          # GEE authentication (project: black-octagon-291810)
│   │   └── aoi.py               # AOI geometry helpers
│   ├── 01_ndwi_variance.py      # σ²_NDWI — Sentinel-2 time series variance
│   ├── 02_dem_variance.py       # σ²_DEM — Copernicus GLO-30 roughness + slope
│   ├── 03_veg_canopy.py         # NDVI + ETH canopy height
│   └── 04_sensor_variance.py    # σ²_sensor — S2 vs Landsat 9 cross-sensor variance
│
├── processing/                  # Local Python processing pipeline
│   ├── utils/
│   │   ├── constants.py         # All site configs, weights, thresholds
│   │   ├── weights.py           # Zone weight lookup
│   │   └── raster_ops.py        # Reproject, clip, resample helpers
│   ├── 01_tidal_model.py        # σ²_tidal — GESLA-3 harmonic analysis / NOAA API
│   ├── 02_zone_classifier.py    # Rule-based geomorphic zone classification
│   ├── 03_cmui_fusion.py        # Zone-adaptive weighted CMUI fusion
│   ├── 04_validation.py         # CMUI vs ground truth correlation
│   ├── 05_figures.py            # Publication figures (CMUI maps, boxplot, scatter)
│   └── 06_baseline_comparison.py # CMUI vs DEM-only, NDWI-only, uniform weights
│
├── paper/                       # LaTeX paper files
│   ├── sec_abstract.tex
│   ├── sec_methodology.tex
│   ├── sec_results.tex
│   ├── sec_discussion.tex
│   └── cmui_references.bib
│
├── run_narrabeen_gee.py         # All 4 GEE exports for Narrabeen in one script
├── run_narrabeen_processing.py  # Tidal + zone + fusion for Narrabeen
├── run_narrabeen_gps_validation.py  # GPS validation (Turner et al. 2016)
├── run_venice_validation.py     # Venice validation with UTM reprojection
├── generate_fig1_plotly.py      # 4-site location map (plotly, no matplotlib needed)
├── generate_pipeline_and_cmui_map.py  # Pipeline diagram + conceptual CMUI schematic
├── 07_shap_analysis.py          # SHAP explainability — all 4 sites
├── environment/
│   ├── environment.yml          # Conda environment spec
│   └── requirements.txt         # pip fallback
└── README.md
```

---

## Quick Start

### 1. Environment

```bash
# Conda (recommended)
conda env create -f environment/environment.yml
conda activate cmui
pip install geedim geemap pyTMD geedim shap

# Or pip only
pip install -r environment/requirements.txt
```

### 2. GEE Authentication

```bash
python gee_scripts/utils/gee_auth.py
```

Uses GEE project `black-octagon-291810`. Replace with your own project ID in `gee_scripts/utils/gee_auth.py` if needed.

### 3. Run the Pipeline (example: Sundarbans)

```bash
# Step 1 — Download GEE variance rasters (runs on GEE cloud, ~40 min)
python gee_scripts/01_ndwi_variance.py      # change SITE = "sundarbans"
python gee_scripts/02_dem_variance.py
python gee_scripts/03_veg_canopy.py
python gee_scripts/04_sensor_variance.py

# Step 2 — Local processing
python processing/01_tidal_model.py --site sundarbans
python processing/02_zone_classifier.py --site sundarbans
python processing/03_cmui_fusion.py --site sundarbans

# Step 3 — Validation and figures
python processing/04_validation.py --site sundarbans
python processing/05_figures.py --site sundarbans
python processing/06_baseline_comparison.py --site sundarbans

# Step 4 — SHAP analysis (all sites)
py -3.11 07_shap_analysis.py

# Step 5 — Location map (plotly, no matplotlib needed)
py -3.11 generate_fig1_plotly.py
```

Repeat Steps 1–3 with `--site venice`, `--site narrabeen`, `--site florida`.

---

## Data Sources

All inputs are freely available without registration:

| Dataset | Source | Access |
|---------|--------|--------|
| Copernicus GLO-30 DEM | ESA | Google Earth Engine |
| Sentinel-2 SR | ESA | Google Earth Engine |
| Landsat 9 | USGS | Google Earth Engine |
| ETH Canopy Height (Lang et al. 2023) | ETH Zürich | GEE community asset |
| GESLA-3 tide gauges | Haigh et al. 2023 | gesla787883612.wordpress.com |
| NOAA CO-OPS (Florida) | NOAA | api.tidesandcurrents.noaa.gov |
| Narrabeen GPS surveys | Turner et al. 2016 | narrabeen.wrl.unsw.edu.au |

---

## Key Results

| Site | CMUI Mean | CMUI Max | Top SHAP predictor |
|------|-----------|----------|--------------------|
| Sundarbans | 2.034 m | 9.981 m | σ²_DEM (0.138 m) |
| Venice | 1.935 m | 19.239 m | σ²_NDWI (0.067 m) |
| Narrabeen | 2.445 m | 14.426 m | σ²_DEM (0.580 m) |
| Florida | 2.122 m | 8.278 m | σ²_DEM (0.191 m) |

DEM-only approaches overestimate uncertainty by **80–122%** on tidal flats.  
NDWI-only approaches underestimate at **100%** of pixels across all sites.

---

## Citation

If you use CMUI in your research, please cite:

```
Christa, S.; Jayaram, R.; Sharma, R. CMUI: A Spatially Explicit Coastal Measurement
Uncertainty Index for Multi-Source Earth Observation Data Fusion.
Geomatics 2026. [Under Review]
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

Sharon Christa — sharonchrista@gmail.com  
ORCID: [0000-0001-6717-2200](https://orcid.org/0000-0001-6717-2200)  
MIT Art, Design and Technology University, Pune, India
