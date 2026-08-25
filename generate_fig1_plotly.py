"""
generate_fig1_plotly.py

Generates publication-quality Figure 1 location maps using plotly.
No matplotlib or cartopy required — uses plotly's built-in Natural Earth basemap.

Requirements (install with pip — no conda needed):
    pip install plotly kaleido

Run:
    python generate_fig1_plotly.py

Outputs:
    D:/cmui/outputs/figures/fig1_location_maps/fig1_location_maps.png
    D:/cmui/outputs/figures/fig1_location_maps/fig1_location_maps.pdf
    D:/cmui/outputs/figures/fig1_location_maps/fig1_location_maps.html
"""

from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

OUT_DIR = Path(r"D:\cmui\outputs\figures\fig1_location_maps")

# ---------------------------------------------------------------------------
# Study site definitions
# ---------------------------------------------------------------------------

SITES = [
    {
        "name":     "Sundarbans, India",
        "role":     "Primary site",
        "lon":       88.75, "lat": 22.00,
        "aoi":      [88.0, 21.5, 89.5, 22.5],
        "color":    "#E74C3C",
        "zoom_lon": [85.0, 92.0], "zoom_lat": [19.5, 25.0],
        "panel":    "(b)",
    },
    {
        "name":     "Venice Lagoon, Italy",
        "role":     "Validation site 1",
        "lon":       12.35, "lat": 45.43,
        "aoi":      [12.0, 45.0, 13.2, 45.8],
        "color":    "#2980B9",
        "zoom_lon": [10.0, 15.0], "zoom_lat": [43.5, 47.5],
        "panel":    "(c)",
    },
    {
        "name":     "Narrabeen, Australia",
        "role":     "Validation site 2",
        "lon":       151.30, "lat": -33.72,
        "aoi":      [151.2958, -33.7390, 151.3122, -33.7013],
        "color":    "#27AE60",
        "zoom_lon": [150.8, 151.8], "zoom_lat": [-34.2, -33.2],
        "panel":    "(d)",
    },
    {
        "name":     "SW Florida, USA",
        "role":     "Validation site 3",
        "lon":      -81.40, "lat": 25.90,
        "aoi":      [-81.8, 25.6, -81.0, 26.2],
        "color":    "#8E44AD",
        "zoom_lon": [-83.0, -80.0], "zoom_lat": [24.5, 27.5],
        "panel":    "(e)",
    },
]


def aoi_rect(aoi):
    w, s, e, n = aoi
    return [w, e, e, w, w], [s, s, n, n, s]


def build_figure() -> go.Figure:
    titles = ["(a) World overview"] + [
        f"{s['panel']} {s['name'].split(',')[0]}<br>"
        f"<sup>{s['name'].split(',')[1].strip()} — {s['role']}</sup>"
        for s in SITES
    ]

    fig = make_subplots(
        rows=1, cols=5,
        subplot_titles=titles,
        specs=[[{"type": "geo"}] * 5],
        horizontal_spacing=0.02,
    )

    # Panel A: world — all 4 stars
    for site in SITES:
        fig.add_trace(
            go.Scattergeo(
                lon=[site["lon"]], lat=[site["lat"]],
                mode="markers+text",
                marker={"size": 14, "color": site["color"],
                        "symbol": "star", "line": {"color": "white", "width": 1}},
                text=[site["name"].split(",")[0]],
                textposition="bottom right",
                textfont={"size": 9, "color": site["color"]},
                name=f"{site['name']} ({site['role']})",
                showlegend=True,
            ),
            row=1, col=1,
        )

    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor="#ECF0F1",
        showocean=True, oceancolor="#D6EAF8",
        showcoastlines=True, coastlinecolor="#555555", coastlinewidth=0.5,
        showcountries=True, countrycolor="#AAAAAA", countrywidth=0.3,
        showlakes=True, lakecolor="#D6EAF8",
        row=1, col=1,
    )

    # Panels B-E: regional zoom
    for col_idx, site in enumerate(SITES, start=2):
        aoi = site["aoi"]
        rl, rla = aoi_rect(aoi)
        r, g, b = int(site["color"][1:3],16), int(site["color"][3:5],16), int(site["color"][5:7],16)

        fig.add_trace(go.Scattergeo(
            lon=rl, lat=rla, mode="lines",
            line={"color": site["color"], "width": 2},
            fill="toself", fillcolor=f"rgba({r},{g},{b},0.12)",
            showlegend=False,
        ), row=1, col=col_idx)

        fig.add_trace(go.Scattergeo(
            lon=[site["lon"]], lat=[site["lat"]], mode="markers",
            marker={"size": 10, "color": site["color"], "symbol": "star",
                    "line": {"color": "white", "width": 0.8}},
            showlegend=False,
        ), row=1, col=col_idx)

        cx = (aoi[0] + aoi[2]) / 2
        hemi = 'S' if aoi[3] < 0 else 'N'
        fig.add_trace(go.Scattergeo(
            lon=[cx], lat=[aoi[3] + (aoi[3]-aoi[1])*0.08], mode="text",
            text=[f"{aoi[0]:.1f}°–{aoi[2]:.1f}° / {aoi[1]:.1f}°–{aoi[3]:.1f}°{hemi}"],
            textfont={"size": 7, "color": site["color"]}, showlegend=False,
        ), row=1, col=col_idx)

        fig.update_geos(
            projection_type="mercator",
            lonaxis_range=site["zoom_lon"], lataxis_range=site["zoom_lat"],
            showland=True, landcolor="#ECF0F1",
            showocean=True, oceancolor="#D6EAF8",
            showcoastlines=True, coastlinecolor="#555555", coastlinewidth=0.8,
            showcountries=True, countrycolor="#AAAAAA", countrywidth=0.4,
            showrivers=True, rivercolor="#85C1E9", riverwidth=0.5,
            showlakes=True, lakecolor="#D6EAF8",
            row=1, col=col_idx,
        )

    fig.update_layout(
        width=1800, height=440, paper_bgcolor="white",
        font={"family": "Arial", "size": 10},
        legend={"orientation": "h", "y": -0.18, "x": 0.5,
                "xanchor": "center", "yanchor": "bottom",
                "bgcolor": "white", "bordercolor": "#CCC",
                "borderwidth": 1, "font": {"size": 9}},
        margin={"t": 60, "b": 100, "l": 5, "r": 5},
        title={"text": "Fig. 1.  Study area location map", "x": 0.5,
               "font": {"size": 13}},
    )
    return fig


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_figure(fig: go.Figure, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # HTML — always works, no kaleido needed
    html_path = out_dir / "fig1_location_maps.html"
    fig.write_html(str(html_path))
    print(f"HTML saved: {html_path}")

    # PNG — requires kaleido
    try:
        png_path = out_dir / "fig1_location_maps.png"
        fig.write_image(str(png_path), scale=3)   # 3x scale = 300 dpi equivalent
        print(f"PNG saved:  {png_path}")
    except Exception as exc:
        print(f"PNG export failed (install kaleido): {exc}")
        print("  pip install kaleido")

    # PDF — requires kaleido
    try:
        pdf_path = out_dir / "fig1_location_maps.pdf"
        fig.write_image(str(pdf_path))
        print(f"PDF saved:  {pdf_path}")
    except Exception as exc:
        print(f"PDF export failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building Figure 1 location map...")
    fig = build_figure()
    save_figure(fig, OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()