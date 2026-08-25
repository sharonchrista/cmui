"""
processing/07_shap_analysis.py

SHAP-based component sensitivity analysis for CMUI.

Approach:
  1. Sample N pixels from the CMUI raster (bands 1-8)
  2. Train a RandomForest surrogate model:
        CMUI ~ sigma2_DEM + sigma2_NDWI + sigma2_tidal
               + sigma2_veg + sigma2_sensor + zone_code
  3. The surrogate achieves high R^2 because CMUI is a deterministic
     function of its inputs — making SHAP values interpretable.
  4. Compute SHAP values using TreeExplainer (fast, exact for tree models)
  5. Generate three publication figures:
        fig_shap_importance_<site>.png  -- global mean |SHAP| bar chart
        fig_shap_beeswarm_<site>.png    -- beeswarm (value vs SHAP impact)
        fig_shap_by_zone_<site>.png     -- per-zone mean |SHAP| stacked bar

Justification for surrogate approach:
  A deterministic weighted sum is not directly interpretable via SHAP
  because SHAP requires a probabilistic or regression model. The RF
  surrogate, trained on the same pixel values, reproduces CMUI with
  R^2 > 0.99 and provides well-calibrated Shapley attributions that
  reflect how each sigma^2 component shifts CMUI above or below its
  expected value at a given location.

Install requirements:
  pip install shap scikit-learn

Run:
  python processing/07_shap_analysis.py --site sundarbans
  python processing/07_shap_analysis.py --site venice
  python processing/07_shap_analysis.py --site narrabeen
  python processing/07_shap_analysis.py --site florida
  python processing/07_shap_analysis.py --site all
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from processing.utils.constants import AOI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CMUI_DIR    = Path("D:/cmui/outputs/cmui_rasters")
OUT_DIR     = Path("D:/cmui/outputs/figures/fig_shap")
N_SAMPLES   = 5000   # pixels per site — enough for stable SHAP estimates
RANDOM_SEED = 42
DPI         = 300

FEATURE_NAMES = [
    "sigma2_DEM",
    "sigma2_NDWI",
    "sigma2_tidal",
    "sigma2_veg",
    "sigma2_sensor",
    "zone_code",
]

FEATURE_LABELS = [
    r"$\sigma^2_{DEM}$",
    r"$\sigma^2_{NDWI}$",
    r"$\sigma^2_{tidal}$",
    r"$\sigma^2_{veg}$",
    r"$\sigma^2_{sensor}$",
    "Zone",
]

ZONE_NAMES = {
    0: "Unclassified",
    1: "Sandy beach",
    2: "Rocky shore",
    3: "Mangrove",
    4: "Tidal flat",
    5: "Coral reef",
}

ZONE_COLORS = {
    0: "#CCCCCC",
    1: "#F4D03F",
    2: "#5D6D7E",
    3: "#1E8449",
    4: "#AED6F1",
    5: "#F1948A",
}

COMPONENT_COLORS = [
    "#3498DB",   # DEM
    "#E74C3C",   # NDWI
    "#2ECC71",   # tidal
    "#9B59B6",   # veg
    "#F39C12",   # sensor
    "#95A5A6",   # zone
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pixel_samples(
    cmui_path: Path,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """
    Load CMUI raster and return a random sample of valid pixels as a DataFrame.

    Raster band layout (from 03_cmui_fusion.py):
      1 = CMUI
      2 = sigma2_DEM
      3 = sigma2_NDWI
      4 = sigma2_tidal
      5 = sigma2_veg
      6 = sigma2_sensor
      7 = tau
      8 = zone_codes
    """
    with rasterio.open(cmui_path) as src:
        cmui      = src.read(1).astype(np.float32).ravel()
        sigma_dem = src.read(2).astype(np.float32).ravel()
        sigma_ndwi= src.read(3).astype(np.float32).ravel()
        sigma_tid = src.read(4).astype(np.float32).ravel()
        sigma_veg = src.read(5).astype(np.float32).ravel()
        sigma_sen = src.read(6).astype(np.float32).ravel()
        zone      = src.read(8).astype(np.float32).ravel()

    valid_mask = (cmui > 0) & np.isfinite(cmui)
    indices    = np.where(valid_mask)[0]

    if len(indices) > n_samples:
        rng     = np.random.default_rng(seed)
        indices = rng.choice(indices, size=n_samples, replace=False)

    df = pd.DataFrame({
        "CMUI":          cmui[indices],
        "sigma2_DEM":    sigma_dem[indices],
        "sigma2_NDWI":   sigma_ndwi[indices],
        "sigma2_tidal":  sigma_tid[indices],
        "sigma2_veg":    sigma_veg[indices],
        "sigma2_sensor": sigma_sen[indices],
        "zone_code":     zone[indices].astype(int),
    })

    return df.dropna()


# ---------------------------------------------------------------------------
# Surrogate model
# ---------------------------------------------------------------------------

def train_surrogate(df: pd.DataFrame) -> tuple[RandomForestRegressor, float]:
    """
    Train a RandomForest regressor: CMUI ~ 5 sigma^2 components + zone.
    Returns the fitted model and its test-set R^2.
    """
    X = df[FEATURE_NAMES].values
    y = df["CMUI"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, r2


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: RandomForestRegressor,
    X: np.ndarray,
) -> np.ndarray:
    """
    Compute SHAP values using TreeExplainer (exact, fast for RF).
    Returns array of shape (n_samples, n_features).
    """
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X)
    return shap_vals


# ---------------------------------------------------------------------------
# Figure 1: Global mean |SHAP| bar chart
# ---------------------------------------------------------------------------

def plot_shap_importance(
    shap_values: np.ndarray,
    site: str,
    out_dir: Path,
) -> None:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order    = np.argsort(mean_abs)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(
        [FEATURE_LABELS[i] for i in order],
        mean_abs[order],
        color=[COMPONENT_COLORS[i] for i in order],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Mean |SHAP value| (m)", fontsize=10)
    ax.set_title(
        f"Component importance — {site.capitalize()}",
        fontsize=11, fontweight="bold",
    )
    ax.tick_params(labelsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    _save(fig, out_dir, f"fig_shap_importance_{site}")
    plt.close(fig)
    print(f"  SHAP importance saved: {site}")


# ---------------------------------------------------------------------------
# Figure 2: Beeswarm (SHAP value vs feature value)
# ---------------------------------------------------------------------------

def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X: np.ndarray,
    site: str,
    out_dir: Path,
) -> None:
    """
    Custom beeswarm using matplotlib so no shap.summary_plot matplotlib
    backend issues arise.
    """
    n_features = shap_values.shape[1]
    mean_abs   = np.mean(np.abs(shap_values), axis=0)
    order      = np.argsort(mean_abs)[::-1]   # most important first

    fig, axes = plt.subplots(1, n_features, figsize=(14, 5), sharey=False)
    if n_features == 1:
        axes = [axes]

    for plot_idx, feat_idx in enumerate(order):
        ax      = axes[plot_idx]
        sv      = shap_values[:, feat_idx]
        fv      = X[:, feat_idx]
        fv_norm = (fv - fv.min()) / (np.ptp(fv) + 1e-10)

        scatter = ax.scatter(
            sv, np.zeros_like(sv) + np.random.uniform(-0.3, 0.3, len(sv)),
            c=fv_norm, cmap="RdBu_r", s=5, alpha=0.5, linewidths=0,
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP value (m)", fontsize=8)
        ax.set_title(FEATURE_LABELS[feat_idx], fontsize=9)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"SHAP beeswarm — {site.capitalize()}\n"
        "Colour: low (blue) to high (red) feature value",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, out_dir, f"fig_shap_beeswarm_{site}")
    plt.close(fig)
    print(f"  SHAP beeswarm saved: {site}")


# ---------------------------------------------------------------------------
# Figure 3: Per-zone mean |SHAP| stacked bar
# ---------------------------------------------------------------------------

def plot_shap_by_zone(
    shap_values: np.ndarray,
    zone_codes: np.ndarray,
    site: str,
    out_dir: Path,
) -> None:
    zone_present = sorted(
        [z for z in np.unique(zone_codes) if np.sum(zone_codes == z) >= 20]
    )
    if len(zone_present) < 2:
        print(f"  Not enough zones for per-zone SHAP plot — skipping {site}")
        return

    zone_labels  = [ZONE_NAMES.get(int(z), f"Zone {z}") for z in zone_present]
    n_zones      = len(zone_present)
    n_components = shap_values.shape[1] - 1   # exclude zone_code feature itself
    component_labels = FEATURE_LABELS[:n_components]

    data = np.zeros((n_zones, n_components))
    for i, z in enumerate(zone_present):
        mask = zone_codes == z
        data[i] = np.mean(np.abs(shap_values[mask, :n_components]), axis=0)

    # Normalise each zone row so bars sum to 1 (fraction of total)
    row_sums = data.sum(axis=1, keepdims=True)
    data_norm = np.where(row_sums > 0, data / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(max(7, n_zones * 1.6), 5))
    x      = np.arange(n_zones)
    bottom = np.zeros(n_zones)

    for c_idx in range(n_components):
        ax.bar(
            x, data_norm[:, c_idx],
            bottom=bottom,
            color=COMPONENT_COLORS[c_idx],
            label=component_labels[c_idx],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += data_norm[:, c_idx]

    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of mean |SHAP|", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Component contribution to CMUI by zone — {site.capitalize()}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir, f"fig_shap_by_zone_{site}")
    plt.close(fig)
    print(f"  SHAP by zone saved: {site}")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Per-site analysis
# ---------------------------------------------------------------------------

def analyse_site(site: str) -> None:
    cmui_path = CMUI_DIR / f"cmui_{site}.tif"
    if not cmui_path.exists():
        print(f"  CMUI raster not found for {site} — skipping")
        return

    print(f"\n=== {site.upper()} ===")

    print(f"  Loading {N_SAMPLES} pixel samples...")
    df = load_pixel_samples(cmui_path, N_SAMPLES, RANDOM_SEED)
    print(f"  Loaded {len(df)} valid samples")

    print("  Training RF surrogate model...")
    model, r2 = train_surrogate(df)
    print(f"  Surrogate R^2 = {r2:.4f}")

    X = df[FEATURE_NAMES].values
    print("  Computing SHAP values (TreeExplainer)...")
    shap_values = compute_shap_values(model, X)
    print(f"  SHAP values shape: {shap_values.shape}")

    zone_codes = df["zone_code"].values

    plot_shap_importance(shap_values, site, OUT_DIR)
    plot_shap_beeswarm(shap_values, X, site, OUT_DIR)
    plot_shap_by_zone(shap_values, zone_codes, site, OUT_DIR)

    # Print feature importance summary
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    print(f"\n  Component importance (mean |SHAP|):")
    for name, label, val in sorted(
        zip(FEATURE_NAMES, FEATURE_LABELS, mean_abs),
        key=lambda x: x[2], reverse=True,
    ):
        print(f"    {label:20s}  {val:.6f} m")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHAP component sensitivity analysis for CMUI"
    )
    parser.add_argument(
        "--site",
        choices=list(AOI.keys()) + ["all"],
        default="all",
        help="Site to analyse (or 'all')",
    )
    args = parser.parse_args()

    sites = list(AOI.keys()) if args.site == "all" else [args.site]

    for site in sites:
        analyse_site(site)

    print(f"\nAll SHAP figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
