# =============================================================================
# Richmond Spatial Decision Intelligence Explorer
# A working instance of the "Dashboard" component of the Spatial Decision
# Intelligence Framework (Health Equity Through Sustainable Urban Development,
# PhD proposal, University of Cambridge). Case study: Richmond, California,
# within Contra Costa County, at census-tract resolution.
#
# Author:   Genie Tang
# Script:   richmond_explorer.py
# Version:  v33
#
# Version log
#   v1 (2026-10-24): First version. Exposure, burden, vulnerability and health
#       layers from the CDC/ATSDR Environmental Justice Index 2024 and CDC
#       PLACES 2023, by census tract, with the City of Richmond outlined.
#       Tabs: Map, Clusters (global Moran's I and LISA, PySAL), Bivariate map,
#       Priority tracts (top tercile of exposure and of burden), Table,
#       How to use, About the data. Reproduces Figures 10 to 12 of the
#       proposal (global Moran's I for asthma = 0.69, queen contiguity, 999
#       permutations).
#   v2 (2026-10-24): Map usability. The maps now open on an explicit view of
#       the county instead of an automatic fit, so the toolbar's reset button
#       returns there; the toolbar is always visible; and each map has a
#       Reset map view button.
#
# How to run
#   pip install -r requirements.txt
#   streamlit run richmond_explorer.py
#   The app reads data/contra_costa_tracts.geojson and data/richmond.geojson,
#   built once by prepare_data.py. No network access is needed.
# =============================================================================

import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from esda.moran import Moran, Moran_Local
from libpysal.weights import KNN, Queen, Rook

VERSION = "v33"
INK = "#1f2933"
INK_SOFT = "#52514e"
GRID = "#e6e6e2"
RICHMOND_LINE = "#14212b"
# Plotly loads a world outline file to set up any geographic plot; it is served
# from the app's own static folder so the maps work without outside scripts.
# Map toolbar: always visible, so the reset (home) button is there after a
# zoom, and the clutter buttons are removed. The topojson line points Plotly
# at the outline file shipped in static/ when that folder is present.
MAPCFG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    # An explicit list, not a remove-list: only these buttons can appear. Each
    # acts on the click itself, so nothing on the toolbar looks unresponsive,
    # and Plotly's "Share chart" button, which uploads the plotted data to an
    # outside service, cannot come back.
    "modeBarButtons": [["toImage", "zoomInGeo", "zoomOutGeo", "resetGeo"]],
    "toImageButtonOptions": {"format": "png", "scale": 2,
                             "filename": "richmond_explorer_map"},
}
CHARTCFG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": False,
    "modeBarButtons": [["toImage", "zoomIn2d", "zoomOut2d",
                        "autoScale2d", "resetScale2d"]],
    "toImageButtonOptions": {"format": "png", "scale": 2,
                             "filename": "richmond_explorer_chart"},
}
MAP_OUTLINES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "static", "world_110m.json")
if os.path.exists(MAP_OUTLINES):
    MAPCFG["topojsonURL"] = "app/static/"

# ---- Layer catalogue ---------------------------------------------------------
# key: (label, group, unit label, decimals, hover format, colour family)
LAYERS = {
    "pm25_days": ("PM2.5: days above the national standard per year", "Exposure",
                  "days per year (3-year mean)", 2, "blues"),
    "ozone_days": ("Ozone: days above the national standard per year", "Exposure",
                   "days per year (3-year mean)", 2, "blues"),
    "diesel_pm": ("Diesel particulate matter", "Exposure", "µg/m³", 3, "blues"),
    "air_toxics_cancer_risk": ("Air toxics cancer risk", "Exposure",
                               "cases per million", 1, "blues"),
    "near_road": ("Share of tract within 1 mile of a high-volume road", "Exposure",
                  "proportion", 2, "blues"),
    "near_tri": ("Share of tract within 1 mile of a Toxic Release Inventory site",
                 "Exposure", "proportion", 2, "blues"),
    "pre1980_housing": ("Housing built before 1980", "Built environment", "%", 1, "blues"),
    "near_park": ("Share of tract within 1 mile of a park", "Built environment",
                  "proportion", 2, "greens"),
    "walkability": ("National Walkability Index", "Built environment", "index", 1, "greens"),
    "eji_pct": ("Environmental Justice Index (overall percentile)", "Index",
                "percentile rank 0 to 1", 2, "greens"),
    "ebm_pct": ("Environmental burden module (percentile)", "Index",
                "percentile rank 0 to 1", 2, "greens"),
    "svm_pct": ("Social vulnerability module (percentile)", "Index",
                "percentile rank 0 to 1", 2, "greens"),
    "asthma_eji": ("Adults with asthma (CDC PLACES, in EJI 2024)", "Health",
                   "% of adults", 1, "purples"),
    "asthma_places2023": ("Adults with asthma (CDC PLACES 2023 release)", "Health",
                          "% of adults", 1, "purples"),
    "diabetes": ("Adults with diabetes", "Health", "% of adults", 1, "purples"),
    "heart_disease": ("Adults with coronary heart disease", "Health", "% of adults", 1, "purples"),
    "poor_mental_health": ("Adults reporting poor mental health", "Health", "% of adults", 1, "purples"),
    "poverty200": ("Income below 200% of the poverty level", "Social", "% of persons", 1, "oranges"),
    "housing_burden": ("Housing cost burden (income under $75,000)", "Social", "% of households", 1, "oranges"),
    "renter": ("Renter-occupied housing", "Social", "% of units", 1, "oranges"),
    "uninsured": ("Uninsured", "Social", "% of persons", 1, "oranges"),
    "disability": ("Persons with a disability", "Social", "% of persons", 1, "oranges"),
    "age65": ("Persons aged 65 and older", "Social", "% of persons", 1, "oranges"),
    "minority": ("Persons of colour", "Social", "% of persons", 1, "oranges"),
    "limited_english": ("Limited English proficiency", "Social", "% of persons", 1, "oranges"),
    "no_internet": ("Households without internet", "Social", "% of households", 1, "oranges"),
}
SCALES = {
    "blues": [[0, "#eef3fb"], [0.5, "#7fb0ea"], [1, "#1b4f9c"]],
    "greens": [[0, "#edf7ef"], [0.5, "#74c69d"], [1, "#1b6b3a"]],
    "purples": [[0, "#f3effa"], [0.5, "#a78bd6"], [1, "#4b2c84"]],
    "oranges": [[0, "#fdf1e6"], [0.5, "#f0a35e"], [1, "#b4531a"]],
}
CLUSTER_COLORS = {"High-High": "#b03030", "Low-Low": "#1f63a8", "High-Low": "#e8a0a0",
                  "Low-High": "#9dc3ea", "Not significant": "#e9ecef"}
CLUSTER_ORDER = ["High-High", "Low-Low", "High-Low", "Low-High", "Not significant"]
# Bivariate 3x3 palette (x = exposure low->high, y = outcome low->high), as in Figure 12.
BIV = {(0, 0): "#e8e8e8", (1, 0): "#b5c0da", (2, 0): "#6c83b5",
       (0, 1): "#b8d6be", (1, 1): "#90b2b3", (2, 1): "#567994",
       (0, 2): "#73ae80", (1, 2): "#5a9178", (2, 2): "#2a5a5b"}


# ---- Data ----------------------------------------------------------------------
@st.cache_data(show_spinner="Loading tract data...")
def load():
    here = os.path.dirname(os.path.abspath(__file__))
    # Data files sit next to the script (or in data/ if that folder exists).
    sub = "data" if os.path.exists(os.path.join(here, "data", "contra_costa_tracts.geojson")) else ""
    tracts = gpd.read_file(os.path.join(here, sub, "contra_costa_tracts.geojson"))
    richmond = gpd.read_file(os.path.join(here, sub, "richmond.geojson"))
    tracts = tracts.set_index("GEOID", drop=False)
    geo = json.loads(tracts.to_json())
    rgeo = json.loads(richmond.to_json())
    # Bounding box of the county, with a small margin. The maps use this as an
    # explicit view instead of letting Plotly fit the bounds, so the reset
    # button in the map toolbar returns to the county and not to the world.
    minx, miny, maxx, maxy = tracts.total_bounds
    padx = (maxx - minx) * 0.04
    pady = (maxy - miny) * 0.06
    bounds = {"lon": [minx - padx, maxx + padx], "lat": [miny - pady, maxy + pady]}
    return tracts, geo, rgeo, bounds


@st.cache_resource(show_spinner=False)
def weights(kind, k):
    """Spatial weights on the full tract set, row-standardised."""
    tracts, _, _, _ = load()
    if kind == "Queen contiguity":
        w = Queen.from_dataframe(tracts, use_index=False)
    elif kind == "Rook contiguity":
        w = Rook.from_dataframe(tracts, use_index=False)
    else:
        pts = tracts.geometry.representative_point()
        w = KNN.from_array(np.c_[pts.x, pts.y], k=k)
    w.transform = "r"
    return w


@st.cache_data(show_spinner="Running Moran's I and LISA...")
def lisa(col, kind, k, permutations):
    tracts, _, _, _ = load()
    w = weights(kind, k)
    y = tracts[col].astype(float)
    # Fill the one tract with no value (a small group-quarters tract) with the
    # county median so the weights matrix keeps every neighbour relation.
    y = y.fillna(y.median()).to_numpy()
    gm = Moran(y, w, permutations=permutations)
    lm = Moran_Local(y, w, permutations=permutations, seed=12345)
    labels = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
    out = pd.DataFrame(index=tracts.index)
    out["z"] = (y - y.mean()) / y.std()
    out["lag"] = w.sparse @ out["z"].to_numpy()
    out["p"] = lm.p_sim
    out["cluster"] = [labels[q] if p < 0.05 else "Not significant"
                      for q, p in zip(lm.q, lm.p_sim)]
    return out, float(gm.I), float(gm.p_sim), float(gm.EI)


def map_with_reset(fig, name, config=None, label="\u27f2  Reset map view"):
    """Draw a map with a Reset button above it. The button bumps a counter that
    is part of the chart's key, so Streamlit draws a fresh chart and the view
    returns to the county."""
    cnt_key = f"_mapreset_{name}"
    if cnt_key not in st.session_state:
        st.session_state[cnt_key] = 0
    _, right = st.columns([3, 1.6])
    with right:
        if st.button(label, key=f"btn_{name}", width="stretch",
                     help="Zoom back out to the whole county"):
            st.session_state[cnt_key] += 1
    # Plotly holds on to the zoom and pan a user has set for as long as
    # uirevision keeps the same value, which is what made the Reset button
    # appear to do nothing. Changing it discards that state and falls back to
    # the ranges set on the layout.
    fig.update_layout(uirevision=f"{name}-{st.session_state[cnt_key]}")
    st.plotly_chart(fig, width="stretch", config=config or MAPCFG,
                    key=f"{name}_{st.session_state[cnt_key]}")
    st.caption("Scroll or drag to zoom and pan. Press Reset map view above to "
               "return to the whole county.")


def fmt(v, d):
    return "n/a" if pd.isna(v) else f"{v:,.{d}f}"


def base_map(fig, rgeo, show_richmond=True, height=560, title=""):
    """Shared map styling: no base tiles (works offline), Richmond outline on top."""
    if show_richmond:
        for feat in rgeo["features"]:
            geom = feat["geometry"]
            polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
            for poly in polys:
                ring = poly[0]
                fig.add_trace(go.Scattergeo(
                    lon=[p[0] for p in ring], lat=[p[1] for p in ring], mode="lines",
                    line=dict(color=RICHMOND_LINE, width=2.2), name="City of Richmond",
                    hoverinfo="skip", showlegend=False))
    _, _, _, bnds = load()
    fig.update_geos(visible=False, bgcolor="white", showframe=False,
                    projection_type="mercator",
                    lonaxis=dict(range=bnds["lon"]), lataxis=dict(range=bnds["lat"]))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=40 if title else 0, b=0),
                      paper_bgcolor="white", font=dict(color=INK),
                      title=dict(text=title, font=dict(size=16, color=INK), x=0) if title else None)
    return fig


def choropleth(tracts, geo, rgeo, col, label, unit, decimals, scale, classes, title,
               zoom_richmond=False, height=560):
    vals = tracts[col].astype(float)
    custom = np.c_[tracts["name"], tracts["pop"].fillna(0), vals, tracts["in_richmond"]]
    if classes == "Quintiles":
        # Equal-count classes, like the proposal figures.
        ranks = vals.rank(pct=True)
        z = np.ceil(ranks * 5).clip(1, 5)
        qs = vals.quantile([0, .2, .4, .6, .8, 1]).to_numpy()
        ticktext = [f"{fmt(qs[i], decimals)} to {fmt(qs[i + 1], decimals)}" for i in range(5)]
        trace = go.Choropleth(
            geojson=geo, locations=tracts.index, featureidkey="properties.GEOID",
            z=z, zmin=0.5, zmax=5.5, colorscale=scale,
            colorbar=dict(title=dict(text=unit, side="right"), thickness=12, len=0.7,
                          tickvals=[1, 2, 3, 4, 5], ticktext=ticktext),
            marker_line_color="#ffffff", marker_line_width=0.6, customdata=custom,
            hovertemplate="<b>%{customdata[0]}</b><br>" + label + ": %{customdata[2]:,." +
                          str(decimals) + "f} " + unit + "<br>Population %{customdata[1]:,.0f}"
                          "<extra></extra>")
    else:
        trace = go.Choropleth(
            geojson=geo, locations=tracts.index, featureidkey="properties.GEOID",
            z=vals, colorscale=scale,
            colorbar=dict(title=dict(text=unit, side="right"), thickness=12, len=0.7),
            marker_line_color="#ffffff", marker_line_width=0.6, customdata=custom,
            hovertemplate="<b>%{customdata[0]}</b><br>" + label + ": %{z:,." +
                          str(decimals) + "f} " + unit + "<br>Population %{customdata[1]:,.0f}"
                          "<extra></extra>")
    fig = go.Figure(trace)
    if zoom_richmond:
        sub = tracts[tracts["in_richmond"]]
        fig.add_trace(go.Choropleth(geojson=geo, locations=sub.index,
                                    featureidkey="properties.GEOID", z=[0] * len(sub),
                                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                                    showscale=False, marker_line_width=0, hoverinfo="skip"))
    return base_map(fig, rgeo, height=height, title=title)


def category_map(tracts, geo, rgeo, cats, colors, order, title, hover_extra=None, height=560):
    fig = go.Figure()
    for cat in order:
        idx = cats.index[cats == cat]
        if len(idx) == 0:
            continue
        sub = tracts.loc[idx]
        custom = np.c_[sub["name"], hover_extra.loc[idx] if hover_extra is not None else [""] * len(idx)]
        fig.add_trace(go.Choropleth(
            geojson=geo, locations=idx, featureidkey="properties.GEOID",
            z=[order.index(cat)] * len(idx), name=cat, showscale=False,
            colorscale=[[0, colors[cat]], [1, colors[cat]]],
            marker_line_color="#ffffff", marker_line_width=0.6, customdata=custom,
            hovertemplate="<b>%{customdata[0]}</b><br>" + cat + "<br>%{customdata[1]}<extra></extra>"))
    return base_map(fig, rgeo, height=height, title=title)


def legend_html(items):
    return "".join(
        f"<span style='display:inline-flex;align-items:center;margin:0 14px 6px 0;font-size:0.85rem'>"
        f"<span style='width:14px;height:14px;background:{c};border:1px solid #c8ccd0;"
        f"display:inline-block;margin-right:6px'></span>{t}</span>" for t, c in items)


# ---- Page ------------------------------------------------------------------------
st.set_page_config(page_title="Richmond Spatial Decision Intelligence Explorer",
                   page_icon="🗺️", layout="wide")
st.markdown("""<style>
/* Century Gothic where it is installed, Didact Gothic as the free stand-in,
   the same pair the portfolio site uses. */
@import url('https://fonts.googleapis.com/css2?family=Didact+Gothic&display=swap');
html, body, [class*="st-"], .stApp, button, input, select, textarea {
    font-family: "Century Gothic", CenturyGothic, AppleGothic, "Didact Gothic", sans-serif;
}
.block-container {padding-top: 2.2rem;}
/* Cards in the portfolio site's style: a pastel fill, a border a shade deeper,
   and a 6px edge in the card's own colour. min-height keeps a row level when
   one caption wraps further than the others. */
.tile {border:1px solid transparent;border-left-width:6px;border-radius:10px;
       padding:14px 16px 12px 16px;height:100%;min-height:132px;box-sizing:border-box;}
.tile .name {font-size:0.9rem;font-weight:600;}
.tile .big {font-size:1.7rem;font-weight:600;color:#14212B;line-height:1.15;margin-top:3px;}
.tile .sub {font-size:0.8rem;color:#4B5963;margin-top:6px;line-height:1.35;}
</style>""", unsafe_allow_html=True)

# Card colours lifted from the portfolio site: fill, border, accent.
CARD = {
    "pink":   ("#FBE7E6", "#F0CBC8", "#A33E50"),
    "teal":   ("#E3F2F3", "#C3E0E2", "#16707E"),
    "yellow": ("#FFF7D1", "#F7E08A", "#8C6D00"),
    "purple": ("#F1EBF8", "#DACDEC", "#6A4C9C"),
    "green":  ("#E4F3EA", "#C6E3D1", "#276B43"),
    "blue":   ("#E6F0FA", "#C9DCF0", "#2F6FB0"),
    # White, for the card that stays the same whatever is chosen.
    "fixed":  ("#FFFFFF", "#D9E0DC", "#9FB3CC"),
}

tracts, geo, rgeo, bounds = load()

with st.sidebar:
    st.header("Layer")
    groups = ["Exposure", "Built environment", "Index", "Health", "Social"]
    group = st.radio("Group", groups, horizontal=True, key="group")
    options = [k for k, v in LAYERS.items() if v[1] == group]
    layer = st.selectbox("Indicator", options, format_func=lambda k: LAYERS[k][0], key="layer")
    classes = st.radio("Colour classes", ["Quintiles", "Continuous"], horizontal=True, key="classes",
                       help="Quintiles put one fifth of tracts in each class, as in the "
                            "proposal figures. Continuous shows the raw scale.")
    show_richmond = st.toggle("Outline the City of Richmond", value=True, key="outline")

    st.header("Spatial statistics")
    cluster_layer = st.selectbox("Cluster variable", list(LAYERS),
                                 index=list(LAYERS).index("asthma_eji"),
                                 format_func=lambda k: LAYERS[k][0], key="cluster_layer")
    wkind = st.radio("Spatial weights", ["Queen contiguity", "Rook contiguity", "k nearest"],
                     key="wkind", help="Queen: tracts that share a border or a corner. "
                                       "Rook: a border only. k nearest: by centroid distance.")
    k = st.slider("k (nearest-neighbour weights only)", 4, 10, 6, key="k",
                  disabled=wkind != "k nearest")
    perms = st.select_slider("Permutations", [199, 499, 999, 4999], value=999, key="perms")

    st.header("Bivariate and priority")
    x_layer = st.selectbox("Exposure (x)", [k for k, v in LAYERS.items() if v[1] in ("Exposure", "Built environment", "Index")],
                           index=0, format_func=lambda k: LAYERS[k][0], key="x_layer")
    y_opts = [k for k, v in LAYERS.items() if v[1] in ("Health", "Social", "Index")]
    y_layer = st.selectbox("Burden or outcome (y)", y_opts, index=y_opts.index("asthma_eji"),
                           format_func=lambda k: LAYERS[k][0], key="y_layer")

label, _, unit, dec, fam = LAYERS[layer]

st.title("Richmond Spatial Decision Intelligence Explorer")
st.markdown(
    f"<p style='color:{INK_SOFT};font-size:1.05rem;margin-top:-0.4rem'>Where environmental "
    "exposure, social vulnerability and health burden co-locate across the 241 census tracts "
    "of Contra Costa County, California, with the City of Richmond outlined. Data: CDC/ATSDR "
    "Environmental Justice Index 2024 and CDC PLACES 2023. A working instance of the Dashboard "
    f"component of the Spatial Decision Intelligence Framework. App {VERSION}.</p>",
    unsafe_allow_html=True)

# Headline tiles: Richmond against the county.
rich = tracts[tracts["in_richmond"]]
cnty = tracts


def tile(col, name, big, sub, colour="teal"):
    fill, edge, accent = CARD[colour]
    col.markdown(
        f"<div class='tile' style='background:{fill};border-color:{edge};"
        f"border-left-color:{accent}'>"
        f"<div class='name' style='color:{accent}'>{name}</div>"
        f"<div class='big'>{big}</div>"
        f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def pop_mean(df, c):
    d = df.dropna(subset=[c])
    return np.average(d[c], weights=d["pop"]) if len(d) else np.nan


# The indicator is named once, above the row, so the card titles can stay
# short and the five cards keep the same height. The first card is the
# constant, Richmond's size inside the county; the other four follow the
# indicator chosen in the sidebar.
r_mean, c_mean = pop_mean(rich, layer), pop_mean(cnty, layer)
if pd.notna(r_mean) and pd.notna(c_mean) and c_mean:
    ratio = r_mean / c_mean
    ratio_txt = f"{ratio:.2f}x"
    ratio_sub = ("Richmond higher" if ratio > 1.02 else
                 "Richmond lower" if ratio < 0.98 else "About the same")
else:
    ratio_txt, ratio_sub = "n/a", "No value here"
cut = cnty[layer].quantile(0.8)
top_n = int((rich[layer] >= cut).sum()) if pd.notna(cut) else 0

st.markdown(
    f"<p style='color:{INK_SOFT};font-size:0.95rem;margin:0 0 6px 0'>"
    f"Showing <b style='color:{INK}'>{label}</b>, {unit}. The green card stays "
    "the same; the other four follow the indicator chosen in the sidebar.</p>",
    unsafe_allow_html=True)

tiles = st.columns(5)
tile(tiles[0], "Richmond tracts", f"{len(rich)} of {len(cnty)}",
     f"Population {rich['pop'].sum():,.0f}<br>The same on every view", "green")
tile(tiles[1], "Richmond", fmt(r_mean, dec),
     "Population-weighted mean<br>of the 23 tracts", "pink")
tile(tiles[2], "County", fmt(c_mean, dec),
     f"Population-weighted mean<br>of all {len(cnty)} tracts", "teal")
tile(tiles[3], "Richmond against county", ratio_txt,
     f"{ratio_sub}<br>Richmond divided by county", "yellow")
tile(tiles[4], "In the county top fifth", f"{top_n} of {len(rich)}",
     f"Richmond tracts at or<br>above {fmt(cut, dec)}", "purple")
st.markdown("")

(tab_map, tab_cluster, tab_biv, tab_priority, tab_table, tab_help, tab_about) = st.tabs(
    ["Map", "Clusters", "Bivariate Map", "Priority Tracts", "Table",
     "User Guide", "About the Data"])

# ---- Map tab ------------------------------------------------------------------
with tab_map:
    left, right = st.columns([3, 1])
    with left:
        fig = choropleth(tracts, geo, rgeo, layer, label, unit, dec, SCALES[fam], classes,
                         f"{label}, by census tract")
        if not show_richmond:
            fig.data = [t for t in fig.data if not isinstance(t, go.Scattergeo)]
        map_with_reset(fig, "layer")
    with right:
        vals = tracts[layer].astype(float)
        st.markdown(f"**{label}**")
        st.caption(unit)
        st.metric("Richmond, population-weighted mean", fmt(pop_mean(rich, layer), dec))
        st.metric("County, population-weighted mean", fmt(pop_mean(cnty, layer), dec))
        top = tracts.dropna(subset=[layer]).nlargest(5, layer)
        st.markdown("**Highest five tracts**")
        st.dataframe(pd.DataFrame({"Tract": top["name"].str.replace("Tract ", ""),
                                   "Value": top[layer].round(dec),
                                   "Richmond": np.where(top["in_richmond"], "yes", "")}),
                     hide_index=True, width="stretch")
    st.caption("Zoom with the scroll wheel, or use the toolbar above the map to zoom, pan "
               "and reset the view (the last two buttons return it to the full county). "
               "Source: CDC/ATSDR Environmental Justice Index 2024 (indicators from EPA, "
               "ACS 2018-2022 and CDC PLACES); CDC PLACES 2023 release for the second asthma "
               "layer. Census tract boundaries: 2020 cartographic files. Hover a tract for "
               "its values.")

# ---- Clusters tab -------------------------------------------------------------
with tab_cluster:
    clabel, _, cunit, cdec, _ = LAYERS[cluster_layer]
    res, I, pI, EI = lisa(cluster_layer, wkind, k, perms)
    counts = res["cluster"].value_counts()
    hh = int(counts.get("High-High", 0))
    hh_rich = int(res.loc[tracts["in_richmond"], "cluster"].eq("High-High").sum())
    p_txt = "< 0.001" if pI < 0.001 and perms >= 999 else f"{pI:.3f}"
    # The cards name the variable and the weights, so switching either one is
    # visible even when Moran's I barely moves.
    m1, m2, m3, m4 = st.columns(4)
    tile(m1, "Global Moran's I", f"{I:.2f}",
         f"{clabel}<br>Expected {EI:.3f} under no pattern", "pink")
    tile(m2, "Permutation p-value", p_txt,
         f"{perms} permutations<br>{wkind} weights", "teal")
    tile(m3, "High-High tracts", f"{hh}",
         f"{hh_rich} of them in Richmond<br>High values beside high values", "yellow")
    tile(m4, "Significant tracts", f"{int((res['p'] < 0.05).sum())} of {len(res)}",
         f"p &lt; 0.05<br>{clabel}", "purple")
    lcol, rcol = st.columns([3, 2])
    with lcol:
        hover = res["p"].map(lambda p: f"p = {p:.3f}") + "<br>" + clabel + ": " + \
            tracts[cluster_layer].map(lambda v: fmt(v, cdec)) + " " + cunit
        fig = category_map(tracts, geo, rgeo, res["cluster"], CLUSTER_COLORS, CLUSTER_ORDER,
                           f"Local Moran's I clusters: {clabel} (p < 0.05)", hover)
        map_with_reset(fig, "clusters")
        st.markdown(legend_html([(c, CLUSTER_COLORS[c]) for c in CLUSTER_ORDER]),
                    unsafe_allow_html=True)
    with rcol:
        fig2 = go.Figure()
        for cat in CLUSTER_ORDER:
            sub = res[res["cluster"] == cat]
            if sub.empty:
                continue
            fig2.add_trace(go.Scatter(
                x=sub["z"], y=sub["lag"], mode="markers", name=cat,
                text=tracts.loc[sub.index, "name"],
                marker=dict(color=CLUSTER_COLORS[cat], size=9,
                            line=dict(color="#6b7280", width=0.5),
                            symbol=np.where(tracts.loc[sub.index, "in_richmond"], "diamond", "circle")),
                hovertemplate="<b>%{text}</b><br>z = %{x:.2f}<br>lag = %{y:.2f}<extra></extra>"))
        xs = np.array([res["z"].min(), res["z"].max()])
        fig2.add_trace(go.Scatter(x=xs, y=I * xs, mode="lines",
                                  line=dict(color=INK, width=1.5, dash="dot"),
                                  name=f"Slope = Moran's I ({I:.2f})", hoverinfo="skip"))
        fig2.add_hline(y=0, line_color=GRID)
        fig2.add_vline(x=0, line_color=GRID)
        fig2.update_layout(height=560, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10, r=10, t=40, b=10),
                           title=dict(text="Moran scatterplot (diamonds: Richmond tracts)",
                                      font=dict(size=16, color=INK), x=0),
                           xaxis=dict(title="Standardised value (z)", gridcolor=GRID, zeroline=False),
                           yaxis=dict(title="Spatial lag of z", gridcolor=GRID, zeroline=False),
                           legend=dict(orientation="h", y=-0.22, font=dict(size=11)))
        st.plotly_chart(fig2, width="stretch", config=CHARTCFG)
    st.caption("The map toolbar zooms, pans and resets the view. "
               f"Method: {wkind.lower()} weights" + (f" (k = {k})" if wkind == "k nearest" else "") +
               f", row-standardised; global and local Moran's I from PySAL esda with {perms} "
               "conditional permutations; clusters shown at p < 0.05. High-High: a tract with a "
               "high value surrounded by high-value tracts; Low-Low: the reverse; High-Low and "
               "Low-High are spatial outliers. With queen contiguity and 999 permutations, adults "
               "with asthma gives global Moran's I = 0.69, as reported in the proposal (Section "
               "06.05, Figure 11).")

# ---- Bivariate tab -------------------------------------------------------------
with tab_biv:
    xl, _, xu, xd, _ = LAYERS[x_layer]
    yl, _, yu, yd, _ = LAYERS[y_layer]
    xt = pd.qcut(tracts[x_layer].rank(method="first"), 3, labels=[0, 1, 2]).astype("Int64")
    yt = pd.qcut(tracts[y_layer].rank(method="first"), 3, labels=[0, 1, 2]).astype("Int64")
    names = {0: "Low", 1: "Mid", 2: "High"}
    cats = pd.Series([f"{names[a]} {xl.split(':')[0]} / {names[b]} {yl.split(' (')[0]}"
                      if pd.notna(a) and pd.notna(b) else "No data"
                      for a, b in zip(xt, yt)], index=tracts.index)
    colors = {}
    for (a, b), c in BIV.items():
        colors[f"{names[a]} {xl.split(':')[0]} / {names[b]} {yl.split(' (')[0]}"] = c
    colors["No data"] = "#f4f4f1"
    order = [key for key in colors]
    hover = tracts[x_layer].map(lambda v: f"{xl.split(':')[0]}: {fmt(v, xd)}") + "<br>" + \
        tracts[y_layer].map(lambda v: f"{yl.split(' (')[0]}: {fmt(v, yd)}")
    lcol, rcol = st.columns([3, 1])
    with lcol:
        fig = category_map(tracts, geo, rgeo, cats, colors, order,
                           f"{yl.split(' (')[0]} and {xl.split(':')[0]}: terciles", hover)
        map_with_reset(fig, "bivariate")
    with rcol:
        st.markdown("**Legend** (rows: y terciles, columns: x terciles)")
        grid = "<table style='border-collapse:collapse;font-size:0.8rem'>"
        for b in (2, 1, 0):
            grid += f"<tr><td style='padding:2px 6px;color:{INK_SOFT}'>{names[b]}</td>"
            for a in (0, 1, 2):
                grid += f"<td style='width:34px;height:34px;background:{BIV[(a, b)]}'></td>"
            grid += "</tr>"
        grid += f"<tr><td></td>" + "".join(
            f"<td style='text-align:center;color:{INK_SOFT};padding-top:4px'>{names[a]}</td>" for a in (0, 1, 2)) + "</tr></table>"
        st.markdown(grid, unsafe_allow_html=True)
        st.caption(f"Up and right: {yl.split(' (')[0]} (y) · {xl.split(':')[0]} (x)")
        hh = tracts[(xt == 2) & (yt == 2)]
        st.metric("Tracts high on both", f"{len(hh)}",
                  f"{int(hh['in_richmond'].sum())} in Richmond", delta_color="off")
    st.caption("The map toolbar zooms, pans and resets the view. Terciles across the 241 "
               "county tracts, as in Figure 12 of the proposal. The "
               "darkest cells are tracts in the top third on both measures.")

# ---- Priority tracts tab -------------------------------------------------------
with tab_priority:
    st.markdown(f"**Priority tracts: top tercile on both {xl.split(':')[0]} (x) and "
                f"{yl.split(' (')[0]} (y).** This is the operation the framework "
                "performs: crossing an exposure surface with a burden or need measure to find "
                "where both concentrate (proposal, Section 06.04).")
    pri = tracts[(xt == 2) & (yt == 2)].copy()
    cats_p = pd.Series(np.where(tracts.index.isin(pri.index), "Priority tract",
                                np.where(tracts["in_richmond"], "Richmond, not priority", "Other")),
                       index=tracts.index)
    pcol = {"Priority tract": "#b03030", "Richmond, not priority": "#dfe6ee", "Other": "#f1f1ee"}
    hover = tracts[x_layer].map(lambda v: f"{xl.split(':')[0]}: {fmt(v, xd)}") + "<br>" + \
        tracts[y_layer].map(lambda v: f"{yl.split(' (')[0]}: {fmt(v, yd)}")
    lcol, rcol = st.columns([3, 2])
    with lcol:
        fig = category_map(tracts, geo, rgeo, cats_p, pcol,
                           ["Priority tract", "Richmond, not priority", "Other"],
                           f"{len(pri)} priority tracts, {int(pri['in_richmond'].sum())} in Richmond", hover)
        map_with_reset(fig, "priority")
    with rcol:
        show = pri.sort_values(y_layer, ascending=False)
        st.dataframe(pd.DataFrame({
            "Tract": show["name"].str.replace("Tract ", ""),
            "Richmond": np.where(show["in_richmond"], "yes", ""),
            xl.split(":")[0]: show[x_layer].round(xd),
            yl.split(" (")[0]: show[y_layer].round(yd),
            "EJI pct": show["eji_pct"].round(2),
            "Population": show["pop"].astype(int),
        }), hide_index=True, width="stretch", height=520)
        st.download_button("Download priority tracts (CSV)",
                           show.drop(columns="geometry").to_csv(index=False),
                           file_name="priority_tracts.csv", mime="text/csv")
    st.caption(f"Population in priority tracts: {pri['pop'].sum():,.0f} "
               f"({pri['pop'].sum() / tracts['pop'].sum():.0%} of the county). Change the x and "
               "y layers in the sidebar to test other pairings.")

# ---- Table tab -------------------------------------------------------------------
with tab_table:
    cols = ["name", "in_richmond", "pop", "pm25_days", "ozone_days", "diesel_pm", "eji_pct",
            "ebm_pct", "svm_pct", "asthma_eji", "asthma_places2023", "poverty200",
            "housing_burden", "pre1980_housing"]
    tdf = tracts[cols].copy()
    tdf["in_richmond"] = np.where(tdf["in_richmond"], "yes", "")
    tdf = tdf.rename(columns={"name": "Tract", "in_richmond": "Richmond", "pop": "Population",
                              "pm25_days": "PM2.5 days", "ozone_days": "Ozone days",
                              "diesel_pm": "Diesel PM", "eji_pct": "EJI pct", "ebm_pct": "EBM pct",
                              "svm_pct": "SVM pct", "asthma_eji": "Asthma % (EJI)",
                              "asthma_places2023": "Asthma % (PLACES 2023)",
                              "poverty200": "Poverty <200% FPL %", "housing_burden": "Housing burden %",
                              "pre1980_housing": "Pre-1980 housing %"})
    only_r = st.toggle("Richmond tracts only", value=False, key="only_r")
    if only_r:
        tdf = tdf[tdf["Richmond"] == "yes"]
    st.dataframe(tdf.round(2), hide_index=True, width="stretch", height=520)
    st.download_button("Download the table (CSV)", tdf.to_csv(index=False),
                       file_name="contra_costa_tracts.csv", mime="text/csv")

# ---- Help tab --------------------------------------------------------------------
with tab_help:
    # Written as paragraphs with the numbers typed in, not as a markdown
    # ordered list: Streamlit renumbers nested list items, which showed up
    # as "1.4" and "2.0" instead of 4 and 5.
    st.markdown("""
**1. Map.** Pick a group and an indicator in the sidebar. Quintile classes match the proposal
figures; Continuous shows the raw scale. Hover any tract for its values. The black outline
is the City of Richmond.

**2. Clusters.** Choose the cluster variable and the spatial weights. Global Moran's I says
whether similar values sit together; the map shows where (High-High, Low-Low, outliers).
Richmond tracts are diamonds in the scatterplot. Raise the permutations for a finer p-value.

**3. Bivariate Map.** Choose an exposure (x) and a burden or outcome (y). The 3 by 3 legend
reads up and right; the darkest cells are high on both.

**4. Priority Tracts.** The tracts in the top tercile on both x and y, with a table and a CSV
download. Change x and y to test other pairings.

**5. Table.** Every tract with the main indicators; switch to Richmond only.

**Maps.** Every map has a Reset map view button above it that returns the view to the whole
county, and a toolbar with download, zoom and reset.
""")

# ---- About tab -------------------------------------------------------------------
with tab_about:
    st.markdown(f"""
**Purpose.** This app is a working instance of the Dashboard component of the Spatial Decision
Intelligence Framework (Figure 1 of the proposal *Health Equity Through Sustainable Urban
Development*): descriptive and exploratory analysis of current patterns at the resolution at
which planning decisions are made. It reproduces the preliminary analysis in Section 06.05 and
Figures 10 to 12, and lets a reader change the variables, the weights and the thresholds.

**Data.**
- CDC/ATSDR Environmental Justice Index 2024, California file, census tracts (2020 boundaries).
  Environmental burden indicators come from EPA (PM2.5 and ozone as the annual mean number of
  days above the national standard, averaged over three years; diesel PM; air toxics cancer
  risk; proximity to roads, Toxic Release Inventory and National Priority List sites; parks;
  walkability), social vulnerability from the ACS 2018-2022, and health from CDC PLACES.
  Percentile ranks (0 to 1) are national ranks across all US tracts; 1 is the most burdened.
- CDC PLACES 2023 release, current asthma among adults, crude prevalence by tract, added as a
  second, newer asthma layer. It agrees closely with the PLACES estimate inside the EJI.
- Census 2020 cartographic boundary files (1:500,000) for tracts and for the City of Richmond.
  A tract counts as a Richmond tract when its representative point falls inside the city
  boundary ({int(tracts['in_richmond'].sum())} tracts).

**Preparation.** `prepare_data.py` filters the state file to Contra Costa County
({len(tracts)} tracts), replaces the EJI code -999 with missing, joins PLACES and the
boundaries, and writes one GeoJSON. One tract (a small group-quarters tract) has no EJI
percentile and is filled with the county median for the cluster analysis only.

**Methods.** Global Moran's I and local Moran (LISA) from PySAL `esda` with `libpysal`
weights (queen, rook or k nearest), row-standardised, conditional permutations. The bivariate
map and priority tracts use terciles across county tracts.

**Limits.** PLACES health measures are model-based small-area estimates, not surveys of each
tract. EJI percentiles are national ranks, so a Contra Costa tract at 0.9 is in the top 10% of
US tracts, not of county tracts. Tract boundaries are 2020; EJI 2024 uses 2018-2022 ACS data.
Spatial clustering of reported outcomes also reflects where people are diagnosed and
surveyed. These are patterns to take to a planning decision, not causal estimates.

**Built with** Python, Streamlit, pandas, GeoPandas, Plotly and PySAL. Code:
`richmond_explorer.py` {VERSION}.

**Author.** Genie Tang, epidemiologist and DrPH student, Johns Hopkins Bloomberg School of
Public Health; master's in spatial analysis for public health. Portfolio:
[spatialhealthinformatics.github.io](https://spatialhealthinformatics.github.io).
""")
