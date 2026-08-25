"""
generate_pipeline_and_cmui_map.py

Generates two remaining paper figures using plotly only:
  1. CMUI computation pipeline diagram
  2. Conceptual CMUI spatial map (schematic transect)

Requirements:
    pip install plotly kaleido

Run:
    python generate_pipeline_and_cmui_map.py
"""

from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio

OUT_DIR = Path("D:/cmui/outputs/figures")


# ---------------------------------------------------------------------------
# Helper: save figure
# ---------------------------------------------------------------------------

def save(fig: go.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    html = out_dir / f"{stem}.html"
    fig.write_html(str(html))
    print(f"HTML : {html}")
    try:
        png = out_dir / f"{stem}.png"
        fig.write_image(str(png), scale=3, width=1400, height=600)
        print(f"PNG  : {png}")
    except Exception as e:
        print(f"PNG failed (kaleido?): {e}")
    try:
        pdf = out_dir / f"{stem}.pdf"
        fig.write_image(str(pdf))
        print(f"PDF  : {pdf}")
    except Exception as e:
        print(f"PDF failed: {e}")


# ---------------------------------------------------------------------------
# Figure A: CMUI Pipeline Diagram
# ---------------------------------------------------------------------------

def make_pipeline_figure() -> go.Figure:
    """
    Four-tier flowchart rendered as an SVG-style plotly figure.
    Tier 1: 5 input boxes
    Tier 2: 5 variance component boxes
    Tier 3: Fusion layer (zone classifier + CMUI fusion + tau staleness)
    Tier 4: Output CMUI map
    """
    fig = go.Figure()

    # Canvas
    fig.update_layout(
        width=1200, height=560,
        xaxis={"range": [0, 12], "showgrid": False, "zeroline": False,
               "showticklabels": False},
        yaxis={"range": [0, 8],  "showgrid": False, "zeroline": False,
               "showticklabels": False, "scaleanchor": "x"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        title={"text": "CMUI Computation Pipeline",
               "x": 0.5, "font": {"size": 16}},
        showlegend=False,
    )

    # Color palette
    C_INPUT   = ("#E6F1FB", "#185FA5")   # fill, border
    C_VAR     = ("#E1F5EE", "#0F6E56")
    C_FUSE    = ("#EEEDFE", "#534AB7")
    C_SIDE    = ("#FAEEDA", "#854F0B")
    C_OUTPUT  = ("#F1EFE8", "#5F5E5A")

    def box(x0, y0, x1, y1, fill, border, label, sublabel="", fontsize=10):
        # Rectangle
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      fillcolor=fill, line={"color": border, "width": 1.5})
        # Main label
        cy = (y0 + y1) / 2 + (0.1 if sublabel else 0)
        fig.add_annotation(x=(x0+x1)/2, y=cy, text=label,
                           showarrow=False, font={"size": fontsize, "color": border},
                           xanchor="center", yanchor="middle")
        if sublabel:
            fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2 - 0.22, text=sublabel,
                               showarrow=False,
                               font={"size": fontsize - 1.5, "color": "#555555"},
                               xanchor="center", yanchor="middle")

    def arrow(x0, y0, x1, y1):
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0,
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1.2,
                           arrowcolor="#888888", arrowwidth=1.5)

    def hline(x0, x1, y):
        fig.add_shape(type="line", x0=x0, y0=y, x1=x1, y1=y,
                      line={"color": "#AAAAAA", "width": 1.2})

    def vline(x, y0, y1):
        fig.add_shape(type="line", x0=x, y0=y0, x1=x, y1=y1,
                      line={"color": "#AAAAAA", "width": 1.2})

    # ---- Tier 1 — Inputs (y=6.5–7.5) ----
    inputs = [
        ("LiDAR / DEM", ""),
        ("Sentinel-2", "time series"),
        ("Tidal model", "FES2014/OTPS"),
        ("NDVI / canopy", "height"),
        ("Multi-source", "S2 + Landsat"),
    ]
    xs = [0.4, 2.6, 4.8, 7.0, 9.2]
    for (lbl, sub), x in zip(inputs, xs):
        box(x, 6.4, x+2.0, 7.4, *C_INPUT, lbl, sub, fontsize=9)

    # ---- Arrows Tier 1 -> Tier 2 ----
    for x in xs:
        arrow(x+1.0, 6.4, x+1.0, 5.7)

    # ---- Tier 2 — Variance components (y=4.8–5.7) ----
    var_labels = [
        ("σ²_DEM", "DEM error"),
        ("σ²_NDWI", "Water index"),
        ("σ²_tidal", "Tidal range"),
        ("σ²_veg", "Canopy bias"),
        ("σ²_sensor", "Cross-sensor"),
    ]
    for (lbl, sub), x in zip(var_labels, xs):
        box(x, 4.8, x+2.0, 5.7, *C_VAR, f"<b>{lbl}</b>", sub, fontsize=9)

    # ---- Convergence bus ----
    for x in xs:
        vline(x+1.0, 4.8, 4.3)
    hline(xs[0]+1.0, xs[-1]+1.0, 4.3)
    arrow(6.2, 4.3, 6.2, 3.65)

    # ---- Tier 3 — Fusion layer (y=2.8–3.65) ----
    # Zone classifier (left side)
    box(0.3, 2.8, 2.8, 3.65, *C_SIDE, "<b>Zone classifier</b>", "5 coastal types", fontsize=9)
    arrow(2.8, 3.2, 3.8, 3.2)

    # CMUI fusion (centre)
    box(3.8, 2.8, 8.6, 3.65, *C_FUSE, "<b>CMUI weighted fusion</b>",
        "Equation 1 · per-pixel output", fontsize=10)

    # Tau staleness (right side)
    box(9.2, 2.8, 11.7, 3.65, *C_SIDE, "<b>τ(t) staleness</b>", "Temporal decay", fontsize=9)
    arrow(9.2, 3.2, 8.6, 3.2)

    # ---- Arrow Tier 3 -> Tier 4 ----
    arrow(6.2, 2.8, 6.2, 2.1)

    # ---- Tier 4 — Output ----
    box(3.5, 1.2, 8.9, 2.1, *C_OUTPUT, "<b>CMUI uncertainty surface</b>",
        "per-pixel · metres", fontsize=11)

    # ---- Tier labels (right side) ----
    for y, lbl in [(6.9, "Input"), (5.2, "Variance"), (3.2, "Fusion"), (1.6, "Output")]:
        fig.add_annotation(x=11.9, y=y, text=f"<i>{lbl}</i>",
                           showarrow=False,
                           font={"size": 9, "color": "#AAAAAA"},
                           xanchor="left", yanchor="middle")

    return fig


# ---------------------------------------------------------------------------
# Figure B: Conceptual CMUI spatial map (schematic transect)
# ---------------------------------------------------------------------------

def make_cmui_schematic() -> go.Figure:
    """
    Horizontal schematic plan-view transect from land to sea,
    colored by expected CMUI uncertainty level.
    """
    fig = go.Figure()

    fig.update_layout(
        width=1100, height=480,
        xaxis={"range": [0, 11], "showgrid": False, "zeroline": False,
               "showticklabels": False},
        yaxis={"range": [0, 6],  "showgrid": False, "zeroline": False,
               "showticklabels": False},
        plot_bgcolor="white", paper_bgcolor="white",
        margin={"l": 10, "r": 10, "t": 60, "b": 80},
        title={"text": "Conceptual CMUI Spatial Map — Sundarbans Transect (Land → Sea)",
               "x": 0.5, "font": {"size": 14}},
        showlegend=False,
    )

    # Zone definitions: (x0, fill_color, zone_name, cmui_level, text_color)
    zones = [
        (0.0,  "#C6BDA0",  "Land",        "Not mapped",  "#7A6E50"),
        (1.0,  "#4A8FCA",  "Rocky shore", "Low",         "#FFFFFF"),
        (2.5,  "#5DB87A",  "Sandy beach", "Low–Mod",     "#FFFFFF"),
        (4.2,  "#D04828",  "Mangrove",    "Very High",   "#FFFFFF"),
        (6.5,  "#E8962A",  "Tidal flat",  "High",        "#FFFFFF"),
        (8.5,  "#A8D4E8",  "Open sea",    "N/A",         "#3A7A9A"),
        (10.0, None,       None,          None,          None),  # sentinel
    ]

    for i, (x0, fill, name, lvl, tcol) in enumerate(zones[:-1]):
        x1 = zones[i+1][0]
        if fill is None:
            continue
        fig.add_shape(type="rect", x0=x0, y0=0.5, x1=x1, y1=4.5,
                      fillcolor=fill, line={"color": "white", "width": 1})

        cx = (x0 + x1) / 2
        fig.add_annotation(x=cx, y=4.9, text=f"<b>{name}</b>",
                           showarrow=False, font={"size": 10, "color": "#333333"},
                           xanchor="center")
        fig.add_annotation(x=cx, y=4.55, text=f"<i>{lvl}</i>",
                           showarrow=False,
                           font={"size": 9, "color": zones[i][4]
                                 if lvl != "Not mapped" else "#7A6E50"},
                           xanchor="center")

    # Tidal creek through mangrove zone (winding white + dark red)
    import numpy as np
    t = np.linspace(0, 1, 60)
    creek_x = 4.2 + 2.3 * t
    creek_y = 2.5 + 0.8 * np.sin(t * 3.5 * np.pi)

    fig.add_trace(go.Scatter(x=creek_x, y=creek_y, mode="lines",
                             line={"color": "white", "width": 10},
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=creek_x, y=creek_y, mode="lines",
                             line={"color": "#7A1A08", "width": 4},
                             hoverinfo="skip"))
    fig.add_annotation(x=5.4, y=3.55, text="<b>Tidal creek<br>CMUI: maximum</b>",
                       showarrow=True, arrowhead=2,
                       ax=5.4, ay=4.0,
                       font={"size": 9, "color": "#FFFFFF"},
                       bgcolor="#7A1A08", borderpad=3,
                       xanchor="center")

    # "Peak uncertainty" label above mangrove zone
    fig.add_annotation(x=5.35, y=5.3, text="▲ Peak uncertainty",
                       showarrow=False,
                       font={"size": 10, "color": "#A82A10", "family": "Arial"},
                       xanchor="center")

    # Colorbar legend (vertical gradient)
    # Draw 5 stacked rectangles as a manual gradient
    grad_colors = ["#C8391C", "#D96A20", "#E8962A", "#A8C87A", "#4A8FCA"]
    grad_labels = ["High", "", "Med", "", "Low"]
    for i, (gc, gl) in enumerate(zip(grad_colors, grad_labels)):
        fig.add_shape(type="rect", x0=10.2, y0=0.5 + i*0.8,
                      x1=10.55, y1=0.5 + (i+1)*0.8,
                      fillcolor=gc, line={"width": 0})
        if gl:
            fig.add_annotation(x=10.65, y=0.5 + (i+0.5)*0.8, text=gl,
                               showarrow=False,
                               font={"size": 9, "color": "#666666"},
                               xanchor="left", yanchor="middle")
    fig.add_annotation(x=10.375, y=5.1, text="CMUI",
                       showarrow=False,
                       font={"size": 9, "color": "#666666"},
                       xanchor="center")

    # Bottom axis label
    fig.add_annotation(x=5.0, y=0.1,
                       text="← Land    ·    Coastal zone    ·    Sea →    (schematic plan view)",
                       showarrow=False, font={"size": 9, "color": "#AAAAAA"},
                       xanchor="center")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pipeline_dir = OUT_DIR / "fig_pipeline"
    schematic_dir = OUT_DIR / "fig_cmui_schematic"

    print("Generating pipeline diagram...")
    fig_pipe = make_pipeline_figure()
    save(fig_pipe, pipeline_dir, "fig_cmui_pipeline")

    print("\nGenerating conceptual CMUI spatial map...")
    fig_map = make_cmui_schematic()
    save(fig_map, schematic_dir, "fig_cmui_schematic")

    print("\nDone.")


if __name__ == "__main__":
    main()