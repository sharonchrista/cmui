"""
processing/05_figures.py

Generates all publication-quality figures for the CMUI paper.

Figure list:
  Fig 1: CMUI pipeline diagram       (already done as SVG — skipped here)
  Fig 2: CMUI spatial map            (primary site: Sundarbans)
  Fig 3: CMUI spatial map            (validation site: Venice)
  Fig 4: Zone-stratified CMUI boxplot (both sites)
  Fig 5: Validation scatter plot      (CMUI vs observed error, Venice)
  Fig 6: Component contribution chart (stacked bar per zone)

All figures saved as PDF (vector) + PNG (300 dpi) for paper submission.

Run:
    python processing/05_figures.py --site sundarbans
    python processing/05_figures.py --site venice
    python processing/05_figures.py --site all
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/laptop
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from processing.utils.constants import AOI, ZONE_WEIGHTS

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

FIGURE_DPI = 300
FONT_SIZE   = 9
LABEL_SIZE  = 10

ZONE_COLORS: dict[int, str] = {
    0: "#CCCCCC",   # unclassified
    1: "#F4D03F",   # sandy_beach
    2: "#5D6D7E",   # rocky_shore
    3: "#1E8449",   # mangrove
    4: "#AED6F1",   # tidal_flat
    5: "#F1948A",   # coral_reef
}

ZONE_LABELS: dict[int, str] = {
    0: "Unclassified",
    1: "Sandy beach",
    2: "Rocky shore",
    3: "Mangrove",
    4: "Tidal flat",
    5: "Coral reef",
}

CMUI_CMAP = "RdYlBu_r"   # red=high uncertainty, blue=low


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_cmui_raster(site: str) -> tuple[np.ndarray, rasterio.transform.Affine, tuple]:
    """Load CMUI band and return (array, transform, bounds)."""
    path = Path("D:/cmui/outputs/cmui_rasters") / f"cmui_{site}.tif"
    if not path.exists():
        raise FileNotFoundError(f"CMUI raster not found: {path}")
    with rasterio.open(path) as src:
        cmui      = src.read(1).astype(np.float32)
        zones     = src.read(8).astype(np.int8)
        transform = src.transform
        bounds    = src.bounds
        components = {
            "DEM":    src.read(2),
            "NDWI":   src.read(3),
            "Tidal":  src.read(4),
            "Veg":    src.read(5),
            "Sensor": src.read(6),
        }
    cmui[cmui <= 0] = np.nan
    return cmui, zones, transform, bounds, components


def load_validation_csv(site: str) -> pd.DataFrame | None:
    path = Path("D:/cmui/outputs/validation") / f"validation_{site}.csv"
    if not path.exists():
        print(f"  Validation CSV not found for {site} — skipping Fig 5")
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Figure 2/3: CMUI spatial map
# ---------------------------------------------------------------------------

def plot_cmui_map(
    site: str,
    cmui: np.ndarray,
    zones: np.ndarray,
    bounds: rasterio.coords.BoundingBox,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Determine vmin/vmax from 2nd–98th percentile for robust colormap
    valid = cmui[np.isfinite(cmui)]
    vmin = float(np.percentile(valid, 2))
    vmax = float(np.percentile(valid, 98))

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    # Left panel: CMUI surface
    ax = axes[0]
    im = ax.imshow(
        cmui,
        extent=extent,
        cmap=CMUI_CMAP,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("CMUI (m)", fontsize=FONT_SIZE)
    cb.ax.tick_params(labelsize=FONT_SIZE - 1)
    ax.set_title(f"CMUI — {site.capitalize()}", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=FONT_SIZE)
    ax.set_ylabel("Latitude", fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE - 1)

    # Right panel: geomorphic zone map
    ax2 = axes[1]
    zone_rgb = np.zeros((*zones.shape, 4), dtype=np.float32)
    for code, hex_color in ZONE_COLORS.items():
        rgba = mcolors.to_rgba(hex_color)
        mask = zones == code
        zone_rgb[mask] = rgba

    ax2.imshow(zone_rgb, extent=extent, aspect="auto")
    legend_patches = [
        Patch(color=ZONE_COLORS[c], label=ZONE_LABELS[c])
        for c in sorted(ZONE_COLORS.keys())
        if np.any(zones == c)
    ]
    ax2.legend(
        handles=legend_patches,
        loc="lower left",
        fontsize=FONT_SIZE - 1,
        framealpha=0.8,
    )
    ax2.set_title("Geomorphic zones", fontsize=LABEL_SIZE, fontweight="bold")
    ax2.set_xlabel("Longitude", fontsize=FONT_SIZE)
    ax2.tick_params(labelsize=FONT_SIZE - 1)

    plt.tight_layout()
    _save_figure(fig, out_dir, f"fig_cmui_map_{site}")
    plt.close(fig)
    print(f"  Fig 2/3 saved: fig_cmui_map_{site}")


# ---------------------------------------------------------------------------
# Figure 4: Zone-stratified CMUI boxplot
# ---------------------------------------------------------------------------

def plot_zone_boxplot(
    sites: list[str],
    out_dir: Path,
) -> None:
    data_by_site: dict[str, dict] = {}

    for site in sites:
        try:
            cmui, zones, _, _, _ = load_cmui_raster(site)
        except FileNotFoundError:
            continue

        zone_data: dict[str, list] = {}
        for code, label in ZONE_LABELS.items():
            if code == 0:
                continue
            vals = cmui[zones == code]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 10:
                zone_data[label] = vals.tolist()
        data_by_site[site] = zone_data

    if not data_by_site:
        print("  No data for boxplot — skipping Fig 4")
        return

    zone_labels_present = sorted(
        set().union(*[set(d.keys()) for d in data_by_site.values()])
    )

    n_zones = len(zone_labels_present)
    n_sites = len(data_by_site)
    fig, ax = plt.subplots(figsize=(max(8, n_zones * 2), 5))

    colors   = ["#3498DB", "#E67E22", "#27AE60", "#8E44AD"]
    width    = 0.35
    x_pos    = np.arange(n_zones)

    for s_idx, (site, zone_data) in enumerate(data_by_site.items()):
        bp_data   = [zone_data.get(z, [0]) for z in zone_labels_present]
        positions = x_pos + (s_idx - (n_sites - 1) / 2) * width
        bp = ax.boxplot(
            bp_data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"facecolor": colors[s_idx % len(colors)], "alpha": 0.7},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
            showfliers=False,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(zone_labels_present, rotation=20, ha="right",
                       fontsize=FONT_SIZE)
    ax.set_ylabel("CMUI (m)", fontsize=FONT_SIZE)
    ax.set_title("CMUI by geomorphic zone", fontsize=LABEL_SIZE, fontweight="bold")
    ax.tick_params(labelsize=FONT_SIZE - 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    legend_patches = [
        Patch(color=colors[i], alpha=0.7, label=s.capitalize())
        for i, s in enumerate(data_by_site.keys())
    ]
    ax.legend(handles=legend_patches, fontsize=FONT_SIZE)

    plt.tight_layout()
    _save_figure(fig, out_dir, "fig_zone_boxplot")
    plt.close(fig)
    print("  Fig 4 saved: fig_zone_boxplot")


# ---------------------------------------------------------------------------
# Figure 5: Validation scatter plot
# ---------------------------------------------------------------------------

def plot_validation_scatter(
    site: str,
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    if "observed_error_m" not in df.columns and "shoreline_std_m" in df.columns:
        df = df.rename(columns={"shoreline_std_m": "observed_error_m"})
    valid = df.dropna(subset=["cmui_m", "observed_error_m"])
    valid = valid[(valid["cmui_m"] > 0) & (valid["observed_error_m"] >= 0)]

    if len(valid) < 5:
        print(f"  Too few validation points for scatter ({len(valid)}) — skipping")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    # Color by zone
    for code, label in ZONE_LABELS.items():
        if code == 0:
            continue
        subset = valid[valid["zone_code"] == code]
        if len(subset) == 0:
            continue
        ax.scatter(
            subset["cmui_m"],
            subset["observed_error_m"],
            label=label,
            color=ZONE_COLORS[code],
            s=20,
            alpha=0.6,
            edgecolors="none",
        )

    # Regression line
    from scipy.stats import pearsonr, linregress
    x = valid["cmui_m"].values
    y = valid["observed_error_m"].values
    slope, intercept, r, p, _ = linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=1.2,
            label=f"Fit  r={r:.2f}, p={p:.3f}")

    ax.set_xlabel("CMUI (m)", fontsize=FONT_SIZE)
    ax.set_ylabel("Observed positional error (m)", fontsize=FONT_SIZE)
    ax.set_title(
        f"CMUI vs positional error — {site.capitalize()}",
        fontsize=LABEL_SIZE, fontweight="bold",
    )
    ax.legend(fontsize=FONT_SIZE - 1, loc="upper left")
    ax.tick_params(labelsize=FONT_SIZE - 1)
    ax.grid(linestyle="--", alpha=0.3)

    is_proxy = valid["is_proxy"].any() if "is_proxy" in valid.columns else False
    if is_proxy:
        ax.text(
            0.98, 0.02,
            "Proxy validation (NDWI variance)",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=FONT_SIZE - 2,
            color="gray",
        )

    plt.tight_layout()
    _save_figure(fig, out_dir, f"fig_validation_scatter_{site}")
    plt.close(fig)
    print(f"  Fig 5 saved: fig_validation_scatter_{site}")


# ---------------------------------------------------------------------------
# Figure 6: Component contribution stacked bar
# ---------------------------------------------------------------------------

def plot_component_contributions(
    site: str,
    cmui: np.ndarray,
    zones: np.ndarray,
    components: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """
    Stacked bar chart showing mean contribution of each sigma^2 component
    to total CMUI^2 per geomorphic zone.
    """
    zone_labels_present = []
    contributions: dict[str, list] = {k: [] for k in components}
    total_per_zone: list[float] = []

    for code, label in ZONE_LABELS.items():
        if code == 0:
            continue
        mask = zones == code
        if np.sum(mask) < 10:
            continue
        zone_labels_present.append(label)
        zone_cmui2 = np.mean(cmui[mask & np.isfinite(cmui)] ** 2)
        total_per_zone.append(zone_cmui2 if zone_cmui2 > 0 else 1.0)
        for cname, cdata in components.items():
            contributions[cname].append(float(np.nanmean(cdata[mask])))

    if not zone_labels_present:
        print("  No zones for component chart — skipping Fig 6")
        return

    n = len(zone_labels_present)
    x = np.arange(n)
    comp_colors = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6", "#F39C12"]
    comp_names  = list(components.keys())

    fig, ax = plt.subplots(figsize=(max(7, n * 1.5), 5))
    bottom = np.zeros(n)
    for i, (cname, color) in enumerate(zip(comp_names, comp_colors)):
        vals = np.array(contributions[cname])
        # Normalise to fraction of total per zone
        fracs = vals / np.array(total_per_zone)
        ax.bar(x, fracs, bottom=bottom, color=color, label=f"σ²_{cname}",
               edgecolor="white", linewidth=0.5)
        bottom += fracs

    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels_present, rotation=20, ha="right",
                       fontsize=FONT_SIZE)
    ax.set_ylabel("Fraction of CMUI²", fontsize=FONT_SIZE)
    ax.set_title(
        f"Component contributions to CMUI² — {site.capitalize()}",
        fontsize=LABEL_SIZE, fontweight="bold",
    )
    ax.legend(fontsize=FONT_SIZE - 1, loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=FONT_SIZE - 1)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, out_dir, f"fig_components_{site}")
    plt.close(fig)
    print(f"  Fig 6 saved: fig_components_{site}")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=FIGURE_DPI, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CMUI publication figures"
    )
    parser.add_argument(
        "--site",
        choices=list(AOI.keys()) + ["all"],
        default="all",
        help="Site to generate figures for (or 'all')",
    )
    args = parser.parse_args()

    out_dir  = Path("D:/cmui/outputs/figures")
    sites    = list(AOI.keys()) if args.site == "all" else [args.site]

    print(f"Generating figures for: {sites}")

    for site in sites:
        print(f"\n--- {site} ---")
        try:
            cmui, zones, transform, bounds, components = load_cmui_raster(site)
        except FileNotFoundError as e:
            print(f"  Skipping {site}: {e}")
            continue

        plot_cmui_map(site, cmui, zones, bounds, out_dir / f"fig2_cmui_map")
        plot_component_contributions(site, cmui, zones, components, out_dir / "fig6_components")

    # Zone boxplot across all available sites
    print("\n--- Multi-site boxplot ---")
    plot_zone_boxplot(sites, out_dir / "fig4_zone_boxplot")

    # Validation scatter — Venice only
    for site in sites:
        print(f"\n--- Validation scatter: {site} ---")
        df_val = load_validation_csv(site)
        if df_val is not None:
            plot_validation_scatter(site, df_val, out_dir / "fig5_validation")

    print(f"\nAll figures saved to: {out_dir}")
    print("Paper figures ready for LaTeX inclusion.")


if __name__ == "__main__":
    main()