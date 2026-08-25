"""
generate_location_maps.py

Generates publication-quality study area location maps for the CMUI paper.

Figure 1a: World map showing both study site locations
Figure 1b: Sundarbans regional map with AOI bounding box
Figure 1c: Venice Lagoon regional map with AOI bounding box

All three panels are combined into a single figure suitable for
a journal paper or conference submission.

Outputs:
  D:/cmui/outputs/figures/fig1_location_maps/fig1_location_maps.pdf
  D:/cmui/outputs/figures/fig1_location_maps/fig1_location_maps.png

Requirements:
  pip install cartopy
  (cartopy handles basemaps and coastlines — no API key needed)

Run:
  python generate_location_maps.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("cartopy not found. Install with: pip install cartopy")
    print("Falling back to simple matplotlib basemap.")

# ---------------------------------------------------------------------------
# Study site definitions
# ---------------------------------------------------------------------------

SITES = {
    "sundarbans": {
        "label":    "Sundarbans,\nIndia",
        "lon":       88.75,
        "lat":       22.00,
        "aoi":      [88.0, 21.5, 89.5, 22.5],   # [W, S, E, N]
        "color":    "#E74C3C",
        "role":     "Primary site",
        "zoom_extent": [85.0, 19.5, 92.0, 25.0],
    },
    "venice": {
        "label":    "Venice Lagoon,\nItaly",
        "lon":       12.35,
        "lat":       45.43,
        "aoi":      [12.0, 45.0, 13.2, 45.8],
        "color":    "#2980B9",
        "role":     "Validation site",
        "zoom_extent": [10.0, 43.5, 15.0, 47.5],
    },
}

OUTPUT_DIR  = Path("D:/cmui/outputs/figures/fig1_location_maps")
DPI         = 300
FONT_SIZE   = 9
LABEL_SIZE  = 10


# ---------------------------------------------------------------------------
# Cartopy-based map generation
# ---------------------------------------------------------------------------

def make_world_panel(ax):
    """Panel A: world map with both site locations marked."""
    ax.set_global()
    ax.stock_img()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color="#444444")
    ax.add_feature(cfeature.BORDERS,   linewidth=0.2, color="#666666")
    ax.add_feature(cfeature.OCEAN,     color="#D6EAF8", zorder=0)
    ax.add_feature(cfeature.LAND,      color="#ECF0F1", zorder=0)

    for site_name, site in SITES.items():
        ax.plot(
            site["lon"], site["lat"],
            marker="*",
            markersize=12,
            color=site["color"],
            transform=ccrs.PlateCarree(),
            zorder=5,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        ax.text(
            site["lon"] + 5,
            site["lat"] - 3,
            site["label"],
            fontsize=7,
            color=site["color"],
            transform=ccrs.PlateCarree(),
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
        )

    ax.set_title("(a) Study site locations", fontsize=FONT_SIZE,
                 fontweight="bold", pad=4)


def make_regional_panel(ax, site_name: str, panel_label: str):
    """Panel B/C: regional map with AOI bounding box."""
    site   = SITES[site_name]
    extent = site["zoom_extent"]

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.6, color="#444444")
    ax.add_feature(cfeature.BORDERS,    linewidth=0.3, color="#888888")
    ax.add_feature(cfeature.OCEAN,      color="#D6EAF8", zorder=0)
    ax.add_feature(cfeature.LAND,       color="#ECF0F1", zorder=0)
    ax.add_feature(cfeature.RIVERS,     linewidth=0.4, color="#85C1E9",
                   zorder=1)
    ax.add_feature(cfeature.LAKES,      color="#D6EAF8", zorder=1)

    # AOI bounding box
    aoi = site["aoi"]
    w, s, e, n = aoi
    aoi_patch = mpatches.Rectangle(
        xy=(w, s),
        width=e - w,
        height=n - s,
        linewidth=2,
        edgecolor=site["color"],
        facecolor=site["color"],
        alpha=0.15,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    ax.add_patch(aoi_patch)

    # AOI border (solid line on top)
    for spine_w, spine_s, spine_e, spine_n in [
        (w, s, e, s), (w, n, e, n),
        (w, s, w, n), (e, s, e, n),
    ]:
        ax.plot(
            [spine_w, spine_e], [spine_s, spine_n],
            color=site["color"], linewidth=2,
            transform=ccrs.PlateCarree(), zorder=4,
        )

    # AOI label
    ax.text(
        (w + e) / 2, n + 0.1,
        f"AOI: {w}°E–{e}°E\n{s}°N–{n}°N",
        fontsize=7,
        ha="center", va="bottom",
        color=site["color"],
        transform=ccrs.PlateCarree(),
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
        zorder=5,
    )

    # Site marker
    ax.plot(
        site["lon"], site["lat"],
        marker="*", markersize=10,
        color=site["color"],
        transform=ccrs.PlateCarree(),
        zorder=6,
        markeredgecolor="white", markeredgewidth=0.8,
    )

    # Gridlines
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    # Scale bar (approximate)
    _add_scale_bar(ax, extent)

    role_color = {"Primary site": "#E74C3C", "Validation site": "#2980B9"}
    ax.set_title(
        f"{panel_label} {site['label'].replace(chr(10), ', ')} "
        f"({site['role']})",
        fontsize=FONT_SIZE,
        fontweight="bold",
        pad=4,
        color=role_color.get(site["role"], "black"),
    )


def _add_scale_bar(ax, extent: list[float]) -> None:
    """Add a simple scale bar to the lower-left corner of a regional panel."""
    w, s, e, n = extent
    lon_span = e - w

    # Choose a round scale bar length
    span_km = lon_span * 111   # rough km per degree longitude
    if span_km > 400:
        bar_deg = 1.0;  bar_label = "~111 km"
    elif span_km > 100:
        bar_deg = 0.5;  bar_label = "~55 km"
    else:
        bar_deg = 0.1;  bar_label = "~11 km"

    bar_x = w + 0.05 * lon_span
    bar_y = s + 0.05 * (n - s)

    ax.plot(
        [bar_x, bar_x + bar_deg], [bar_y, bar_y],
        color="black", linewidth=2,
        transform=ccrs.PlateCarree(), zorder=10,
    )
    ax.text(
        bar_x + bar_deg / 2, bar_y + 0.02 * (n - s),
        bar_label,
        ha="center", va="bottom",
        fontsize=6,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )


# ---------------------------------------------------------------------------
# Fallback: simple matplotlib map (no cartopy)
# ---------------------------------------------------------------------------

def make_simple_location_map(output_dir: Path) -> None:
    """Simple location map using only matplotlib — no cartopy needed."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: world context (simplified)
    ax = axes[0]
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_facecolor("#D6EAF8")
    ax.set_aspect("equal")
    ax.set_title("(a) World overview", fontsize=FONT_SIZE, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=FONT_SIZE - 1)
    ax.set_ylabel("Latitude",  fontsize=FONT_SIZE - 1)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.tick_params(labelsize=FONT_SIZE - 2)

    for site_name, site in SITES.items():
        ax.plot(site["lon"], site["lat"], marker="*", markersize=14,
                color=site["color"], zorder=5,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.annotate(
            site["label"],
            xy=(site["lon"], site["lat"]),
            xytext=(site["lon"] + 8, site["lat"] - 5),
            fontsize=7, color=site["color"], fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": site["color"],
                        "lw": 0.8},
        )

    # Panels B and C: regional zoom
    for idx, (site_name, site) in enumerate(SITES.items()):
        ax = axes[idx + 1]
        ze = site["zoom_extent"]
        aoi = site["aoi"]

        ax.set_xlim(ze[0], ze[2])
        ax.set_ylim(ze[1], ze[3])
        ax.set_facecolor("#D6EAF8")
        ax.set_aspect("equal")

        # AOI box
        rect = mpatches.Rectangle(
            (aoi[0], aoi[1]), aoi[2] - aoi[0], aoi[3] - aoi[1],
            linewidth=2, edgecolor=site["color"],
            facecolor=site["color"], alpha=0.2,
        )
        ax.add_patch(rect)

        ax.plot(site["lon"], site["lat"], marker="*", markersize=12,
                color=site["color"], zorder=5,
                markeredgecolor="white", markeredgewidth=0.8)

        panel = "(b)" if idx == 0 else "(c)"
        ax.set_title(
            f"{panel} {site['label'].replace(chr(10), ', ')}\n"
            f"({site['role']})",
            fontsize=FONT_SIZE, fontweight="bold", color=site["color"],
        )
        ax.set_xlabel("Longitude", fontsize=FONT_SIZE - 1)
        ax.set_ylabel("Latitude",  fontsize=FONT_SIZE - 1)
        ax.tick_params(labelsize=FONT_SIZE - 2)
        ax.grid(linestyle="--", alpha=0.4, linewidth=0.4)

        aoi_label = (
            f"AOI: {aoi[0]}°–{aoi[2]}°E\n{aoi[1]}°–{aoi[3]}°N"
        )
        ax.text(
            (aoi[0] + aoi[2]) / 2, aoi[3] + 0.1,
            aoi_label,
            ha="center", va="bottom", fontsize=7,
            color=site["color"], fontweight="bold",
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color=s["color"], label=f"{n.capitalize()} ({s['role']})")
        for n, s in SITES.items()
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=2,
        fontsize=FONT_SIZE,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, output_dir, "fig1_location_maps")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main cartopy figure
# ---------------------------------------------------------------------------

def make_cartopy_location_map(output_dir: Path) -> None:
    """Full publication-quality map using cartopy."""
    fig = plt.figure(figsize=(14, 5))

    # Panel A: world
    ax_world = fig.add_subplot(
        1, 3, 1,
        projection=ccrs.Robinson(),
    )
    make_world_panel(ax_world)

    # Panel B: Sundarbans
    ax_sun = fig.add_subplot(
        1, 3, 2,
        projection=ccrs.PlateCarree(),
    )
    make_regional_panel(ax_sun, "sundarbans", "(b)")

    # Panel C: Venice
    ax_ven = fig.add_subplot(
        1, 3, 3,
        projection=ccrs.PlateCarree(),
    )
    make_regional_panel(ax_ven, "venice", "(c)")

    # Legend
    legend_patches = [
        mpatches.Patch(
            color=s["color"],
            label=f"{s['label'].replace(chr(10), ', ')} — {s['role']}",
        )
        for s in SITES.values()
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=2,
        fontsize=FONT_SIZE,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    _save(fig, output_dir, "fig1_location_maps")
    plt.close(fig)
    print("Cartopy location map saved.")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = OUTPUT_DIR

    if HAS_CARTOPY:
        print("cartopy found — generating high-quality map...")
        make_cartopy_location_map(output_dir)
    else:
        print("cartopy not found — generating simple map...")
        print("Install cartopy for publication-quality output:")
        print("  conda install -c conda-forge cartopy")
        make_simple_location_map(output_dir)


if __name__ == "__main__":
    main()