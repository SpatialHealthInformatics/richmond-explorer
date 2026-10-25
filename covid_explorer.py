# =============================================================================
# COVID-19 Surveillance Explorer (Streamlit web app)
# Author:   Genie Tang
# Script:   covid_explorer.py
# Version:  v48
#
# A web app for comparing reported COVID-19 cases and deaths across countries,
# built on the Johns Hopkins CSSE global time series. It aggregates provinces
# to national totals, derives daily counts from running totals, and offers raw
# or per-100,000 views, a 7-day average, choropleth and bubble maps, global
# Moran's I with a LISA cluster map (PySAL), and EARS C2 aberration alerts.
#
# How to run locally
#   pip install -r requirements.txt
#   streamlit run covid_explorer.py
#
# Version note
#   Spatial statistics is the first tab here, so the app opens on the cluster
#   map. The class copy opens on the chart instead.
#
#   The number jumps from v4 to v48 so this app, the class app and the write-up
#   all carry one version number. Nothing was skipped.
#
# Maps
#   Plotly downloads the base map for every geo figure from its CDN. Where that
#   download is blocked the maps come up empty while the other charts still
#   draw. A copy of the outlines ships in static/world_110m.json and Streamlit
#   serves it, so keep that file and .streamlit/config.toml next to this script.
#
# The app reads the CSV files directly from the CSSE GitHub repository each
# time the server starts (results are cached for 12 hours), so no local data
# files are needed. CSSE stopped updating on March 10, 2023, so March 9, 2023
# is the last date in every series.
# =============================================================================

# ---- Clean start -------------------------------------------------------------
# A Streamlit script is re-run from the top on every user interaction, so
# there is no leftover environment to clear. All state lives in the cached
# data and in st.session_state.

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local

VERSION = "v48"

BASE = ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
        "csse_covid_19_data/")
URLS = {
    "Cases": BASE + "csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
    "Deaths": BASE + "csse_covid_19_time_series/time_series_covid19_deaths_global.csv",
}
LOOKUP_URL = BASE + "UID_ISO_FIPS_LookUp_Table.csv"
REPO_URL = ("https://github.com/CSSEGISandData/COVID-19/tree/master/"
            "csse_covid_19_data/csse_covid_19_time_series")

# Entries in the CSSE "Country/Region" column that are not countries. They stay
# selectable, but they have no population, so per-capita values are blank.
NON_COUNTRIES = {"Diamond Princess", "MS Zaandam", "Summer Olympics 2020",
                 "Winter Olympics 2022", "Antarctica"}

# One fixed color per country, assigned in the order countries are added and
# kept for the rest of the session. The palette passes a colorblind
# separation check for adjacent pairs. A seventh country and beyond reuse
# colors, so the app caps the selection at six.
# Map outlines: Plotly fetches a small world file from its CDN. If a copy is
# shipped in static/ (and static serving is on), use that instead so the maps
# also work where the CDN is blocked. Either way the app runs.
import os
# Map toolbar: always visible, so the reset button is there after a zoom, with
# the clutter buttons removed. The topojson line points Plotly at the outline
# file shipped in static/ when that folder is present.
MAPCFG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtons": [["toImage", "zoomInGeo", "zoomOutGeo", "resetGeo"]],
    "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "covid_map"},
}

# Toolbar for the line charts and the scatterplot. The box-zoom, select and
# lasso buttons only arm a drag tool, so a click on one looks like nothing
# happened. They are removed, leaving buttons that act on the click itself.
# Dragging across a chart still zooms to the box, which is how most people do it.
CHARTCFG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": False,
    # An explicit list, not a remove-list: only these buttons can appear. Each
    # acts on the click itself, so nothing on the toolbar looks unresponsive,
    # and Plotly's "Share chart" button, which uploads the plotted data to an
    # outside service, cannot come back.
    "modeBarButtons": [["toImage", "zoomIn2d", "zoomOut2d",
                        "autoScale2d", "resetScale2d"]],
    "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "covid_chart"},
}
# Plotly downloads the base map for every geo figure from its CDN. Where that
# download is blocked the maps come up empty while the other charts still draw,
# which is hard to spot. A copy of the outlines ships in static/, and Streamlit
# serves it (enableStaticServing in .streamlit/config.toml), so the app can
# supply its own base map. Both files have to sit next to this script.
MAP_OUTLINES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "static", "world_110m.json")
MAPS_SERVED_LOCALLY = os.path.exists(MAP_OUTLINES)
if MAPS_SERVED_LOCALLY:
    MAPCFG["topojsonURL"] = "app/static/"

FONT_STACK = "Century Gothic, CenturyGothic, AppleGothic, Didact Gothic, sans-serif"
import plotly.io as pio
pio.templates["appfont"] = go.layout.Template(
    layout=dict(font=dict(family=FONT_STACK)))
pio.templates.default = "plotly_white+appfont"

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
MAX_COUNTRIES = len(PALETTE)
INK = "#1f2933"
INK_SOFT = "#52514e"
GRID = "#e6e6e2"


# ---- Data loading and cleaning ----------------------------------------------
@st.cache_data(ttl=12 * 60 * 60, show_spinner="Loading CSSE data from GitHub...")
def load_data():
    """Return (long data frame, country table).

    Long frame: one row per country, date, and measure with the cumulative
    count, the daily count, and the population.
    Country table: population, latitude and longitude per country.

    Cleaning steps:
      1. Read the wide CSSE file (one column per date).
      2. Sum the province/state rows to one row per country. China, Canada,
         Australia, the UK, France, the Netherlands, Denmark, and New Zealand
         are reported by province or territory; the CSSE global files do not
         also carry a country-total row for them, so summing does not double
         count.
      3. Convert the date columns (format m/d/yy) to real dates and reshape
         to long format.
      4. Daily counts = difference of the cumulative series. The first day
         (1/22/20) has no prior day, so its daily value is the cumulative
         value on that day.
      5. Negative daily values are kept. These come from reporting
         corrections (for example, France removed about 349,000 cases on
         5/20/21). The user chooses how to display them.
    """
    frames = []
    for measure, url in URLS.items():
        wide = pd.read_csv(url)
        date_cols = wide.columns[4:]  # Province/State, Country/Region, Lat, Long come first
        by_country = wide.groupby("Country/Region")[date_cols].sum()
        long = (by_country.T
                .rename_axis("date")
                .reset_index()
                .melt(id_vars="date", var_name="country", value_name="cumulative"))
        long["date"] = pd.to_datetime(long["date"], format="%m/%d/%y")
        long = long.sort_values(["country", "date"]).reset_index(drop=True)
        long["daily"] = long.groupby("country")["cumulative"].diff()
        first_day = long["daily"].isna()
        long.loc[first_day, "daily"] = long.loc[first_day, "cumulative"]
        long["measure"] = measure
        frames.append(long)
    data = pd.concat(frames, ignore_index=True)

    # Population and a map position: the lookup table has one country-level
    # row (no province) for every Country_Region, matching the time series
    # names exactly.
    lookup = pd.read_csv(LOOKUP_URL)
    countries = (lookup[lookup["Province_State"].isna()]
                 .set_index("Country_Region")[["Population", "Lat", "Long_", "iso3"]]
                 .rename(columns={"Population": "population", "Lat": "lat", "Long_": "lon"}))
    data["population"] = data["country"].map(countries["population"])
    return data, countries


def prepare_view(data, countries, measure, count_type, per_capita,
                 smooth, negatives, start, end):
    """Filter and transform the data for the chart and tiles."""
    view = data[(data["measure"] == measure)
                & (data["country"].isin(countries))].copy()
    view = view.sort_values(["country", "date"])

    if count_type == "Daily new":
        if negatives == "Hide (show as a gap)":
            view.loc[view["daily"] < 0, "daily"] = float("nan")
        elif negatives == "Set to zero":
            view["daily"] = view["daily"].clip(lower=0)
        # "Keep as reported" leaves the corrections in place.
        if smooth:
            # Trailing 7-day mean, computed before the date filter so the
            # first days of a chosen window still use a full week of data.
            view["daily"] = (view.groupby("country")["daily"]
                             .transform(lambda s: s.rolling(7, min_periods=1).mean()))
        view["value"] = view["daily"]
    else:
        view["value"] = view["cumulative"]

    if per_capita:
        view["value"] = view["value"] / view["population"] * 100_000

    view = view[(view["date"] >= pd.Timestamp(start))
                & (view["date"] <= pd.Timestamp(end))]
    return view


def window_total(data, country, measure, start, end):
    """New counts inside the window: cumulative at the end minus cumulative
    the day before the window starts. Corrections are kept, as reported."""
    full = data[(data["country"] == country) & (data["measure"] == measure)]
    end_val = full[full["date"] <= pd.Timestamp(end)]["cumulative"].iloc[-1]
    before = full[full["date"] < pd.Timestamp(start)]["cumulative"]
    return end_val - (before.iloc[-1] if len(before) else 0)


@st.cache_data(show_spinner=False)
def world_window(data, measure, start, end):
    """Count and rate per 100,000 for every country over the date window."""
    sub = data[data["measure"] == measure]
    at_end = (sub[sub["date"] <= pd.Timestamp(end)]
              .groupby("country")["cumulative"].last())
    before = (sub[sub["date"] < pd.Timestamp(start)]
              .groupby("country")["cumulative"].last())
    pop = sub.groupby("country")["population"].first()
    out = pd.DataFrame({"count": at_end - before.reindex(at_end.index).fillna(0),
                        "population": pop})
    out["rate"] = out["count"] / out["population"] * 100_000
    return out.reset_index()


@st.cache_data(show_spinner="Running Moran's I and LISA...")
def spatial_stats(world, k, permutations=999):
    """Global Moran's I and local Moran (LISA) on log10 rate per 100,000.

    Weights: k nearest neighbours on great-circle distance between the CSSE
    country positions, row-standardised. Rates are log-transformed because
    they are right-skewed. Significance by permutation (999 draws).
    Returns the data frame with LISA categories and the global statistic.
    """
    df = world.dropna(subset=["rate", "lat", "lon"]).copy()
    df = df[df["population"] > 0]
    coords = df[["lon", "lat"]].to_numpy()
    w = KNN.from_array(coords, k=k, radius=6371.0)  # radius -> arc distance in km
    w.transform = "r"
    y = np.log10(df["rate"].to_numpy() + 1)
    gm = Moran(y, w, permutations=permutations)
    lm = Moran_Local(y, w, permutations=permutations, seed=12345)
    labels = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
    df["z"] = (y - y.mean()) / y.std()
    df["lag"] = w.sparse @ df["z"].to_numpy()
    df["p"] = lm.p_sim
    df["cluster"] = [labels[q] if p < 0.05 else "Not significant"
                     for q, p in zip(lm.q, lm.p_sim)]
    return df, float(gm.I), float(gm.p_sim), float(gm.EI)


def ears_c2(series, baseline=7, lag=2, threshold=3.0):
    """EARS C2: compare each day with the mean and SD of a 7-day baseline that
    ends two days earlier. Alert when (x - mean) / sd exceeds the threshold.
    This is the aberration-detection method used in CDC's EARS and in many
    syndromic surveillance systems. A floor of 0.2 on the SD avoids dividing
    by zero on flat stretches, as in the EARS implementation."""
    x = series.clip(lower=0).astype(float)
    shifted = x.shift(lag)
    mean = shifted.rolling(baseline, min_periods=baseline).mean()
    sd = shifted.rolling(baseline, min_periods=baseline).std(ddof=1).clip(lower=0.2)
    c2 = (x - mean) / sd
    return c2, c2 > threshold


def summary_table(data, countries, start, end):
    """Totals over the selected date window, for both measures."""
    rows = []
    for c in countries:
        pop = data.loc[data["country"] == c, "population"].iloc[0]
        new_cases = window_total(data, c, "Cases", start, end)
        new_deaths = window_total(data, c, "Deaths", start, end)
        rows.append({
            "Country": c,
            "Population": pop,
            "Cases in window": new_cases,
            "Deaths in window": new_deaths,
            "Cases per 100k": new_cases / pop * 100_000 if pd.notna(pop) else None,
            "Deaths per 100k": new_deaths / pop * 100_000 if pd.notna(pop) else None,
            "Deaths per 100 cases": new_deaths / new_cases * 100 if new_cases else None,
        })
    return pd.DataFrame(rows)


def map_with_reset(fig, name, label="\u27f2  Reset map view"):
    """Draw a map with a Reset button above it and a hint below it.

    Plotly's own toolbar only appears while the pointer is over the figure, and
    its reset icon is small, so zooming in and not finding the way back is easy.
    This button is always visible. It bumps a counter that forms part of the
    chart's key, so Streamlit draws a fresh chart and the view returns to the
    whole world."""
    cnt_key = f"_mapreset_{name}"
    if cnt_key not in st.session_state:
        st.session_state[cnt_key] = 0
    _, right = st.columns([3, 1.6])
    with right:
        if st.button(label, key=f"btn_{name}", width="stretch",
                     help="Zoom back out to the whole world"):
            st.session_state[cnt_key] += 1
    # Plotly keeps the zoom and pan a user has set for as long as uirevision
    # holds the same value, which is what made an earlier Reset button appear
    # to do nothing. Changing uirevision tells Plotly to discard that state and
    # fall back to the ranges set on the layout, which is the whole world.
    fig.update_layout(uirevision=f"{name}-{st.session_state[cnt_key]}")
    st.plotly_chart(fig, width="stretch", config=MAPCFG,
                    key=f"{name}_{st.session_state[cnt_key]}")
    st.caption("Scroll or drag to zoom and pan. Press Reset map view above to "
               "return to the whole world.")


def fmt(value, decimals):
    if pd.isna(value):
        return "n/a"
    return f"{value:,.{decimals}f}"


# ---- Page layout -------------------------------------------------------------
st.set_page_config(page_title="COVID-19 by Country", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    /* Century Gothic where it is installed; Didact Gothic, a free look-alike,
       for everyone else. The same pair the portfolio site uses. */
    @import url('https://fonts.googleapis.com/css2?family=Didact+Gothic&display=swap');
    html, body, [class*="st-"], .stApp, button, input, select, textarea {
        font-family: "Century Gothic", CenturyGothic, AppleGothic, "Didact Gothic", sans-serif;
    }
    .block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
    .tile {border: 1px solid #e6e6e2; border-radius: 12px; padding: 14px 16px 12px 16px;
           background: #ffffff; height: 100%;}
    .tile .name {font-size: 0.95rem; font-weight: 600; color: #1f2933; display: flex;
                 align-items: center; gap: 8px;}
    .tile .swatch {width: 12px; height: 12px; border-radius: 3px; display: inline-block;}
    .tile .big {font-size: 1.9rem; font-weight: 600; color: #1f2933; margin-top: 4px;
                line-height: 1.1;}
    .tile .sub {font-size: 0.8rem; color: #52514e; margin-top: 6px; line-height: 1.35;}
    /* Cards in the same style as the portfolio site: a pastel fill, a border
       a shade deeper, and a 6px edge in the card's own colour. */
    /* min-height keeps the four cards level when one line of the caption
       wraps and the others do not. */
    .stile {border-radius: 10px; padding: 16px 18px 14px 18px;
            height: 100%; min-height: 152px; box-sizing: border-box;
            border: 1px solid transparent; border-left-width: 6px;}
    .stile .name {font-size: 0.95rem; font-weight: 600;}
    .stile .big {font-size: 1.9rem; font-weight: 600; margin-top: 4px;
                 line-height: 1.1; color: #14212B;}
    .stile .sub {font-size: 0.8rem; color: #4B5963; margin-top: 6px;
                 line-height: 1.35;}
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    data, country_info = load_data()
except Exception as err:  # network or file-format problem
    st.error(f"Could not load the CSSE data from GitHub: {err}")
    st.stop()

all_countries = sorted(data["country"].unique())
min_date = data["date"].min().date()
max_date = data["date"].max().date()

# Countries the user has added, and the color each one was given when it was
# first added. Both live in session state so they survive reruns.
if "countries" not in st.session_state:
    st.session_state.countries = ["US", "United Kingdom", "Korea, South", "Japan"]
if "colors" not in st.session_state:
    st.session_state.colors = {}


def assign_colors(selected):
    """Give each selected country a palette color it keeps for the session."""
    colors = st.session_state.colors
    for c in list(colors):
        if c not in selected:
            del colors[c]           # free the color when a country is removed
    for c in selected:
        if c not in colors:
            used = set(colors.values())
            colors[c] = next(p for p in PALETTE if p not in used)
    return {c: colors[c] for c in selected}


def add_country():
    """Button callback: move the picked country into the selected list."""
    pick = st.session_state.get("to_add")
    if pick and pick not in st.session_state.countries:
        if len(st.session_state.countries) >= MAX_COUNTRIES:
            st.session_state.limit_hit = True
        else:
            st.session_state.countries = st.session_state.countries + [pick]
    st.session_state.to_add = None


with st.sidebar:
    st.header("Countries")
    st.selectbox("Add a country",
                 [c for c in all_countries if c not in st.session_state.countries],
                 index=None, placeholder="Type or pick a country", key="to_add")
    st.button("Add", width="stretch", on_click=add_country, key="add_country")
    chosen = st.multiselect("Selected countries (click x to remove)",
                            all_countries, key="countries",
                            max_selections=MAX_COUNTRIES)
    if st.session_state.pop("limit_hit", False):
        st.warning(f"Up to {MAX_COUNTRIES} countries can be compared at once. "
                   "Remove one first.")

    st.header("Display options")
    measure = st.radio("Measure", ["Cases", "Deaths"], horizontal=True, key="measure")
    count_type = st.radio("Count type", ["Daily new", "Cumulative"], horizontal=True,
                          key="count_type")
    per_capita = st.toggle("Per 100,000 people", value=False,
                           help="Divides by the CSSE population estimate so "
                                "countries of different sizes can be compared.")
    smooth = st.toggle("7-day average", value=True,
                       disabled=count_type != "Daily new",
                       help="Trailing 7-day mean. Many countries reported only "
                            "on weekdays or once a week, so raw daily counts "
                            "jump between zero and large spikes.")
    negatives = st.selectbox(
        "Negative daily values (data corrections)",
        ["Hide (show as a gap)", "Set to zero", "Keep as reported"],
        disabled=count_type != "Daily new",
        help="A negative daily value means the country revised its total "
             "downward. It is not a real count of new cases.")
    layout_choice = st.radio("Chart layout", ["One chart", "One panel per country"],
                             horizontal=True, key="layout",
                             help="Separate panels help when one country's "
                                  "counts dwarf the others.")
    log_scale = st.toggle("Log scale (y-axis)", value=False)
    start, end = st.slider("Date range", min_value=min_date, max_value=max_date,
                           value=(min_date, max_date), format="MMM YYYY")

# ---- Header ------------------------------------------------------------------
st.title("Daily COVID-19 Cases and Deaths by Country")
st.markdown(
    f"<p style='color:{INK_SOFT}; font-size:1.05rem; margin-top:-0.4rem'>"
    "Pick countries in the sidebar and compare how reported cases and deaths "
    "moved from January 2020 to March 2023. Source: Johns Hopkins CSSE "
    f"COVID-19 time series (<a href='{REPO_URL}'>GitHub</a>). App {VERSION}.</p>",
    unsafe_allow_html=True)

(tab_spatial, tab_chart, tab_map, tab_world, tab_alerts, tab_table,
 tab_help, tab_about) = st.tabs(
    ["Spatial Statistics", "Chart", "Map", "World Map", "Alerts",
     "Summary Table", "User Guide", "About the Data"])

if not chosen:
    tab_chart.info("Add at least one country in the sidebar to see the chart.")
    tab_map.info("Add at least one country in the sidebar to see the map.")
    tab_world.info("Add at least one country in the sidebar to see the world map.")
    tab_spatial.info("Add at least one country in the sidebar to see the spatial statistics.")
    tab_alerts.info("Add at least one country in the sidebar to see the alerts.")
    tab_table.info("Add at least one country in the sidebar to see the table.")
else:
    colors = assign_colors(chosen)
    view = prepare_view(data, chosen, measure, count_type, per_capita,
                        smooth, negatives, start, end)

    unit = "per 100,000 people" if per_capita else "count"
    label = f"{count_type} {measure.lower()}"
    if count_type == "Daily new" and smooth:
        label += " (7-day average)"
    y_title = f"{label}, {unit}"
    decimals = 2 if per_capita else 0

    # ---- Chart tab ------------------------------------------------------------
    with tab_chart:
        missing_pop = [c for c in chosen if c in NON_COUNTRIES]
        if per_capita and missing_pop:
            st.warning("No population is available for: "
                       + ", ".join(missing_pop) + ". Per-capita values are blank.")

        # Headline tiles: one per country, in the order they were added.
        tiles = st.columns(len(chosen))
        for col, c in zip(tiles, chosen):
            s = view[view["country"] == c].dropna(subset=["value"])
            if s.empty:
                col.markdown(f"<div class='tile'><div class='name'>"
                             f"<span class='swatch' style='background:{colors[c]}'></span>"
                             f"{c}</div><div class='sub'>No values in this window.</div></div>",
                             unsafe_allow_html=True)
                continue
            last = s.iloc[-1]
            peak = s.loc[s["value"].idxmax()]
            total = window_total(data, c, measure, start, end)
            pop = s["population"].iloc[0]
            if per_capita and pd.notna(pop):
                total_txt = f"{fmt(total / pop * 100_000, 1)} per 100k in window"
            else:
                total_txt = f"{fmt(total, 0)} {measure.lower()} in window"
            if count_type == "Daily new":
                sub = (f"Peak {fmt(peak['value'], decimals)} on {peak['date']:%b %d, %Y}"
                       f"<br>{total_txt}")
                big_caption = f"on {last['date']:%b %d, %Y}"
            else:
                sub = f"{total_txt}<br>Population {fmt(pop, 0)}"
                big_caption = f"by {last['date']:%b %d, %Y}"
            col.markdown(
                f"<div class='tile'><div class='name'>"
                f"<span class='swatch' style='background:{colors[c]}'></span>{c}</div>"
                f"<div class='big'>{fmt(last['value'], decimals)}</div>"
                f"<div class='sub'>{label.capitalize()} {big_caption}<br>{sub}</div></div>",
                unsafe_allow_html=True)

        st.markdown("")
        hover = f"%{{y:,.{decimals}f}}"

        if layout_choice == "One chart":
            fig = go.Figure()
            for c in chosen:
                s = view[view["country"] == c]
                fig.add_trace(go.Scatter(
                    x=s["date"], y=s["value"], name=c, mode="lines",
                    line=dict(color=colors[c], width=2),
                    hovertemplate=hover, connectgaps=False))
            fig.update_layout(
                title=None,
                height=520, hovermode="x unified",
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", y=-0.14, title=None),
                margin=dict(l=10, r=20, t=50, b=10),
                xaxis=dict(title=None, showgrid=False, linecolor=GRID),
                yaxis=dict(title=None, gridcolor=GRID, zeroline=False,
                           type="log" if log_scale else "linear",
                           tickformat=",.0f" if decimals == 0 else ",.2~f"),
                font=dict(color=INK),
            )
            st.markdown(f"**{y_title.capitalize()}**")
            st.plotly_chart(fig, width="stretch", config=CHARTCFG)
        else:
            n = len(chosen)
            cols = 2 if n > 1 else 1
            rows = -(-n // cols)
            fig = make_subplots(rows=rows, cols=cols, subplot_titles=chosen,
                                shared_xaxes=True, vertical_spacing=0.12,
                                horizontal_spacing=0.08)
            for i, c in enumerate(chosen):
                s = view[view["country"] == c]
                r, k = divmod(i, cols)
                fig.add_trace(go.Scatter(
                    x=s["date"], y=s["value"], name=c, mode="lines",
                    line=dict(color=colors[c], width=2), hovertemplate=hover,
                    showlegend=False, connectgaps=False), row=r + 1, col=k + 1)
            fig.update_layout(
                title=None,
                height=max(420, 280 * rows), plot_bgcolor="white",
                paper_bgcolor="white", hovermode="x unified",
                margin=dict(l=10, r=10, t=70, b=10), font=dict(color=INK))
            fig.update_xaxes(showgrid=False, linecolor=GRID)
            fig.update_yaxes(gridcolor=GRID, zeroline=False,
                             type="log" if log_scale else "linear",
                             tickformat=",.0f" if decimals == 0 else ",.2~f")
            for ann in fig.layout.annotations:
                ann.font.size = 14
                ann.font.color = INK
            st.markdown(f"**{y_title.capitalize()}**")
            st.plotly_chart(fig, width="stretch", config=CHARTCFG)

        st.caption("Hover for every country's value on a date. Drag on the chart "
                   "to zoom; double-click to reset.")
        csv = view[["country", "date", "measure", "value"]].to_csv(index=False)
        st.download_button("Download the plotted data (CSV)", csv,
                           file_name="covid_selected_countries.csv", mime="text/csv")

    # ---- Map tab --------------------------------------------------------------
    with tab_map:
        if not MAPS_SERVED_LOCALLY:
            st.caption("Map outlines are being fetched from the Plotly CDN. "
                       "If the map is blank, static/world_110m.json is missing "
                       "from the folder next to this script.")
        table = summary_table(data, chosen, start, end)
        geo = table.merge(country_info[["lat", "lon"]], left_on="Country",
                          right_index=True, how="left")
        size_col = "Cases per 100k" if measure == "Cases" else "Deaths per 100k"
        count_col = "Cases in window" if measure == "Cases" else "Deaths in window"
        geo = geo.dropna(subset=["lat", "lon"])
        if geo.empty:
            st.info("None of the selected entries has a map position.")
        else:
            size = geo[size_col].fillna(geo[size_col].min() if geo[size_col].notna().any() else 1)
            fig = go.Figure(go.Scattergeo(
                lat=geo["lat"], lon=geo["lon"], text=geo["Country"],
                mode="markers+text", textposition="top center",
                textfont=dict(color=INK, size=12),
                marker=dict(size=size, sizemode="area",
                            sizeref=2.0 * size.max() / (46 ** 2) if size.max() else 1,
                            sizemin=6, color=[colors[c] for c in geo["Country"]],
                            line=dict(width=1.5, color="white"), opacity=0.85),
                customdata=geo[[count_col, size_col]].values,
                hovertemplate="<b>%{text}</b><br>" + count_col +
                              ": %{customdata[0]:,.0f}<br>" + size_col +
                              ": %{customdata[1]:,.1f}<extra></extra>"))
            # An explicit view, so the toolbar's reset button returns to it.
            fig.update_geos(projection_type="natural earth", showcountries=True,
                            countrycolor="#d9d9d4", landcolor="#f3f3ef",
                            oceancolor="#ffffff", showocean=True, coastlinecolor="#d9d9d4",
                            showframe=False, lonaxis_range=[-180, 180],
                            lataxis_range=[-58, 84])
            fig.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0),
                              paper_bgcolor="white",
                              title=None)
            st.markdown(f"**{measure} per 100,000 people** · "
                        f"{start:%b %d, %Y} to {end:%b %d, %Y} · bubble area")
            map_with_reset(fig, "bubble")
            st.caption("Bubble area is proportional to the rate per 100,000 in the "
                       "selected date window. Colors match the chart.")

    # ---- World map tab --------------------------------------------------------
    with tab_world:
        world = world_window(data, measure, start, end)
        world = world.merge(country_info[["iso3"]], left_on="country",
                            right_index=True, how="left").dropna(subset=["iso3"])
        world_metric = st.radio("Shade countries by", ["Rate per 100,000", "Count"],
                                horizontal=True, key="world_metric")
        if world_metric == "Rate per 100,000":
            z, z_label, z_fmt = world["rate"], f"{measure} per 100,000", ",.0f"
        else:
            z, z_label, z_fmt = world["count"], measure, ",.0f"
        # Sequential scale: one hue, light to dark. The top of the scale is
        # the 98th percentile so a few extreme values do not wash out the rest.
        zmax = float(z.quantile(0.98)) if z.notna().any() else None
        fig = go.Figure(go.Choropleth(
            locations=world["iso3"], z=z, text=world["country"],
            colorscale=[[0, "#eef3fb"], [0.5, "#7fb0ea"], [1, "#1b4f9c"]],
            zmin=0, zmax=zmax, marker_line_color="#ffffff", marker_line_width=0.4,
            colorbar=dict(title=dict(text=z_label, side="right"), thickness=12,
                          len=0.6, tickformat=z_fmt),
            customdata=world[["count", "rate", "population"]].values,
            hovertemplate="<b>%{text}</b><br>" + measure + ": %{customdata[0]:,.0f}"
                          "<br>Per 100,000: %{customdata[1]:,.1f}"
                          "<br>Population: %{customdata[2]:,.0f}<extra></extra>"))
        # Outline the selected countries in their own colors.
        sel = world[world["country"].isin(chosen)]
        if not sel.empty:
            fig.add_trace(go.Choropleth(
                locations=sel["iso3"], z=[1] * len(sel), text=sel["country"],
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False, marker_line_color="#eb6834", marker_line_width=2.2,
                hoverinfo="skip"))
            pts = sel.merge(country_info[["lat", "lon"]], left_on="country",
                            right_index=True, how="left")
            fig.add_trace(go.Scattergeo(
                lat=pts["lat"], lon=pts["lon"], text=pts["country"], mode="text",
                textposition="top center", textfont=dict(size=12, color=INK),
                hoverinfo="skip", showlegend=False))
        fig.update_geos(projection_type="natural earth", showcountries=False,
                        landcolor="#f3f3ef", showocean=True, oceancolor="#ffffff",
                        coastlinecolor="#d9d9d4", showframe=False,
                        lonaxis_range=[-180, 180], lataxis_range=[-58, 84])
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=40, b=0),
                          paper_bgcolor="white",
                          title=None)
        st.markdown(f"**{z_label} in every country** · "
                    f"{start:%b %d, %Y} to {end:%b %d, %Y}")
        map_with_reset(fig, "world")
        st.caption("Shading uses one blue scale from light to dark; the darkest step "
                   "starts at the 98th percentile so a few extreme values do not "
                   "flatten the rest. Selected countries are outlined in orange. "
                   "Hover any country for its numbers.")

        ranked = (world.sort_values("rate", ascending=False)
                  .reset_index(drop=True))
        ranked.index = ranked.index + 1
        ranked = ranked.rename(columns={"country": "Country", "count": measure,
                                        "rate": f"{measure} per 100k",
                                        "population": "Population"})
        st.markdown(f"**All countries ranked by {measure.lower()} per 100,000 in the window**")
        st.dataframe(ranked[["Country", measure, f"{measure} per 100k", "Population"]],
                     width="stretch", height=320,
                     column_config={
                         measure: st.column_config.NumberColumn(format="localized"),
                         f"{measure} per 100k": st.column_config.NumberColumn(format="%.1f"),
                         "Population": st.column_config.NumberColumn(format="localized")})

    # ---- Spatial statistics tab -----------------------------------------------
    with tab_spatial:
        st.markdown(f"**Do countries with high {measure.lower()[:-1]} rates sit next to each "
                    "other?** Global Moran's I measures that for the whole map; local "
                    "Moran (LISA) finds the clusters.")
        k = st.slider("Neighbours per country (k nearest, great-circle distance)",
                      4, 10, 6, key="knn_k")
        world_s = world_window(data, measure, start, end)
        world_s = world_s.merge(country_info[["iso3", "lat", "lon"]], left_on="country",
                                right_index=True, how="left").dropna(subset=["iso3"])
        sp, I, p_I, EI = spatial_stats(world_s, k)

        # Same tile styling as the Chart tab, so the two tabs read as one app,
        # but the numbers are the spatial ones rather than a repeat per country.
        p_txt = "< 0.001" if p_I < 0.001 else f"{p_I:.3f}"
        verdict = ("Similar rates sit together" if I > 0.1 and p_I < 0.05
                   else "No clear spatial pattern")
        hh = int((sp["cluster"] == "High-High").sum())
        # Four tinted cards. The fill is the card's own colour at low opacity,
        # so the row reads as a set and stays distinct from the white per-country
        # tiles on the Chart tab.
        # Fill, border and accent are the portfolio site's own card colours.
        tiles = [
            ("Global Moran's I", f"{I:.3f}", "#FBE7E6", "#F0CBC8", "#A33E50",
             f"{verdict}<br>Expected {EI:.3f} under no pattern"),
            ("Permutation p-value", p_txt, "#E3F2F3", "#C3E0E2", "#16707E",
             "999 random re-assignments<br>of the rates to countries"),
            ("High-High countries", f"{hh}", "#FFF7D1", "#F7E08A", "#8C6D00",
             "High rates whose neighbours<br>also have high rates"),
            ("Countries in analysis", f"{len(sp)}", "#F1EBF8", "#DACDEC", "#6A4C9C",
             f"With a rate and a map position<br>{k} nearest neighbours each"),
        ]
        for col, (name, big, fill, edge, accent, sub) in zip(st.columns(4), tiles):
            col.markdown(
                f"<div class='stile' style='background:{fill}; border-color:{edge}; "
                f"border-left-color:{accent}'>"
                f"<div class='name' style='color:{accent}'>{name}</div>"
                f"<div class='big'>{big}</div>"
                f"<div class='sub'>{sub}</div></div>",
                unsafe_allow_html=True)
        st.write("")

        cluster_colors = {"High-High": "#b03030", "Low-Low": "#1f63a8",
                          "High-Low": "#e8a0a0", "Low-High": "#9dc3ea",
                          "Not significant": "#e9ecef"}
        order = ["High-High", "Low-Low", "High-Low", "Low-High", "Not significant"]
        left, right = st.columns([3, 2])
        with left:
            fig = go.Figure()
            for cat in order:
                sub = sp[sp["cluster"] == cat]
                if sub.empty:
                    continue
                fig.add_trace(go.Choropleth(
                    locations=sub["iso3"], z=[order.index(cat)] * len(sub),
                    text=sub["country"], name=cat, showscale=False,
                    colorscale=[[0, cluster_colors[cat]], [1, cluster_colors[cat]]],
                    marker_line_color="#ffffff", marker_line_width=0.4,
                    customdata=sub[["rate", "p"]].values,
                    hovertemplate="<b>%{text}</b><br>" + cat +
                                  "<br>Rate per 100,000: %{customdata[0]:,.0f}"
                                  "<br>p = %{customdata[1]:.3f}<extra></extra>"))
            sel = sp[sp["country"].isin(chosen)]
            if not sel.empty:
                fig.add_trace(go.Choropleth(
                    locations=sel["iso3"], z=[1] * len(sel),
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                    showscale=False, marker_line_color="#eb6834",
                    marker_line_width=2.2, hoverinfo="skip"))
            fig.update_geos(projection_type="natural earth", showcountries=False,
                            landcolor="#f3f3ef", showocean=True, oceancolor="#ffffff",
                            coastlinecolor="#d9d9d4", showframe=False,
                            lonaxis_range=[-180, 180], lataxis_range=[-58, 84])
            fig.update_layout(height=460, margin=dict(l=0, r=0, t=40, b=0),
                              paper_bgcolor="white",
                              title=None)
            st.markdown(f"**LISA clusters** · {measure.lower()} per 100,000 · "
                        "p below 0.05")
            map_with_reset(fig, "lisa")
            counts = sp["cluster"].value_counts().reindex(order).fillna(0).astype(int)
            st.caption("  ·  ".join(f"{c}: {n}" for c, n in counts.items())
                       + ". Selected countries are outlined in orange.")
        with right:
            fig2 = go.Figure()
            for cat in order:
                sub = sp[sp["cluster"] == cat]
                if sub.empty:
                    continue
                fig2.add_trace(go.Scatter(
                    x=sub["z"], y=sub["lag"], mode="markers", name=cat,
                    text=sub["country"],
                    marker=dict(color=cluster_colors[cat], size=9,
                                line=dict(color="#6b7280", width=0.5)),
                    hovertemplate="<b>%{text}</b><br>z = %{x:.2f}<br>"
                                  "spatial lag = %{y:.2f}<extra></extra>"))
            sel = sp[sp["country"].isin(chosen)]
            for _, r in sel.iterrows():
                fig2.add_annotation(x=r["z"], y=r["lag"], text=r["country"],
                                    showarrow=True, arrowhead=0, ax=0, ay=-22,
                                    font=dict(size=11, color=INK))
            slope = I
            xs = np.array([sp["z"].min(), sp["z"].max()])
            fig2.add_trace(go.Scatter(x=xs, y=slope * xs, mode="lines",
                                      line=dict(color=INK, width=1.5, dash="dot"),
                                      name=f"Slope = Moran's I ({I:.2f})",
                                      hoverinfo="skip"))
            fig2.add_hline(y=0, line_color=GRID)
            fig2.add_vline(x=0, line_color=GRID)
            fig2.update_layout(height=460, plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=10, r=10, t=40, b=10),
                               title=None,
                               xaxis=dict(title="Standardised log rate (z)",
                                          gridcolor=GRID, zeroline=False),
                               yaxis=dict(title="Spatial lag of z (neighbours' mean)",
                                          gridcolor=GRID, zeroline=False),
                               legend=dict(orientation="h", y=-0.25, font=dict(size=11)))
            st.markdown("**Moran scatterplot**")
            st.plotly_chart(fig2, width="stretch", config=CHARTCFG)

        st.caption(
            "Method: rates per 100,000 for the selected date window, log10-transformed "
            "because they are right-skewed. Spatial weights: each country's k nearest "
            "neighbours by great-circle distance between the CSSE country positions, "
            "row-standardised. Global Moran's I and local Moran statistics from PySAL "
            "esda with 999 permutations. High-High: a high-rate country surrounded by "
            "high-rate neighbours; Low-Low: the reverse; High-Low and Low-High are spatial "
            "outliers. Expected I under no spatial pattern: "
            f"{EI:.3f}. Reported rates reflect testing and reporting practices, which "
            "themselves cluster regionally, so the clusters describe the reported data, "
            "not necessarily infection.")

    # ---- Alerts tab -------------------------------------------------------------
    with tab_alerts:
        st.markdown(f"**When would a surveillance system have raised an alert?** "
                    f"EARS C2 compares each day's {measure.lower()} with the mean and "
                    "standard deviation of the previous 7 days (after a 2-day gap). "
                    "A day more than 3 standard deviations above that baseline is an alert.")
        thr = st.slider("Alert threshold (standard deviations above baseline)",
                        2.0, 5.0, 3.0, 0.5, key="ears_thr")
        base = data[(data["measure"] == measure) & (data["country"].isin(chosen))].copy()
        base = base.sort_values(["country", "date"])
        n = len(chosen)
        cols = 2 if n > 1 else 1
        rows = -(-n // cols)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=chosen,
                            shared_xaxes=True, vertical_spacing=0.12,
                            horizontal_spacing=0.08)
        summary_rows = []
        for i, c in enumerate(chosen):
            s_c = base[base["country"] == c].set_index("date")["daily"]
            c2, flagged = ears_c2(s_c, threshold=thr)
            win = (s_c.index >= pd.Timestamp(start)) & (s_c.index <= pd.Timestamp(end))
            s_w, f_w, c2_w = s_c[win], flagged[win], c2[win]
            r, kcol = divmod(i, cols)
            fig.add_trace(go.Scatter(x=s_w.index, y=s_w.clip(lower=0), mode="lines",
                                     line=dict(color=colors[c], width=1.5), name=c,
                                     showlegend=False,
                                     hovertemplate="%{y:,.0f}<extra></extra>"),
                          row=r + 1, col=kcol + 1)
            al = s_w[f_w]
            fig.add_trace(go.Scatter(x=al.index, y=al.clip(lower=0), mode="markers",
                                     marker=dict(color="#b03030", size=6, symbol="circle"),
                                     name="Alert", showlegend=(i == 0),
                                     customdata=c2_w[f_w].values,
                                     hovertemplate="Alert: %{y:,.0f} (C2 = %{customdata:.1f})"
                                                   "<extra></extra>"),
                          row=r + 1, col=kcol + 1)
            summary_rows.append({
                "Country": c, "Alert days in window": int(f_w.sum()),
                "Days in window": int(win.sum()),
                "First alert": al.index.min().strftime("%b %d, %Y") if len(al) else "none",
                "Largest C2": round(float(c2_w.max()), 1) if c2_w.notna().any() else None,
                "Day of largest C2": (c2_w.idxmax().strftime("%b %d, %Y")
                                      if c2_w.notna().any() else "none"),
            })
        fig.update_layout(title=None,
                          height=max(420, 280 * rows), plot_bgcolor="white",
                          paper_bgcolor="white", hovermode="x unified",
                          margin=dict(l=10, r=10, t=70, b=10), font=dict(color=INK),
                          legend=dict(orientation="h", y=-0.08))
        fig.update_xaxes(showgrid=False, linecolor=GRID)
        fig.update_yaxes(gridcolor=GRID, zeroline=False, tickformat=",.0f")
        for ann in fig.layout.annotations:
            ann.font.size = 14
            ann.font.color = INK
        st.markdown(f"**Daily {measure.lower()} with EARS C2 alerts** · "
                    f"threshold {thr:g} SD")
        st.plotly_chart(fig, width="stretch", config=CHARTCFG)
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")
        st.caption(
            "Method: EARS C2 (CDC Early Aberration Reporting System). Baseline = the 7 "
            "days ending 2 days before the day being tested; C2 = (count - baseline mean) / "
            "baseline SD, with the SD floored at 0.2. Negative daily values (corrections) "
            "are set to zero before testing. Weekly reporting, where a country posts a "
            "whole week on one day, produces alerts on reporting days; the first alerts of "
            "each wave are the ones worth reading. The same method runs in the author's "
            "automated influenza alert report for Japan's IDWR data.")

    # ---- Table tab ------------------------------------------------------------
    with tab_table:
        st.write(f"Totals between {start:%b %d, %Y} and {end:%b %d, %Y}. "
                 "Counts are as reported, including corrections.")
        table = summary_table(data, chosen, start, end)
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={
                         "Population": st.column_config.NumberColumn(format="localized"),
                         "Cases in window": st.column_config.NumberColumn(format="localized"),
                         "Deaths in window": st.column_config.NumberColumn(format="localized"),
                         "Cases per 100k": st.column_config.NumberColumn(format="%.1f"),
                         "Deaths per 100k": st.column_config.NumberColumn(format="%.1f"),
                         "Deaths per 100 cases": st.column_config.NumberColumn(format="%.2f"),
                     })
        st.caption("Deaths per 100 cases is a case fatality ratio over the window. "
                   "It depends on how much testing each country did, so it is "
                   "not a measure of how deadly the virus was.")

with tab_help:
    st.markdown("""
**Getting started**
1. In the sidebar, pick a country in **Add a country** and press **Add**. You can also type
   straight into **Selected countries**. Remove a country with the **x** on its tag.
   Up to six countries can be compared at once; each keeps its color while selected.
2. Choose **Cases** or **Deaths**, then **Daily new** or **Cumulative**.
3. Turn on **Per 100,000 people** to compare countries of different sizes.
4. For daily counts, the **7-day average** smooths out weekday and weekly reporting patterns.
   Choose how to show **negative daily values** (downward corrections to a country's total).
5. **One panel per country** gives each country its own y-axis, which helps when one
   country's numbers dwarf the others. **Log scale** is another way to handle that.
6. Drag the **Date range** slider to zoom in on a period. The tiles, map and table all
   follow the selected range.

**Reading the results**
- The tiles above the chart show each country's most recent value in the window, its
  peak and the peak date (daily view), and the total for the window.
- Hover over the chart to see every selected country's value on that date.
- The **Map** tab places the selected countries with bubble area proportional to the
  rate per 100,000 in the window.
- The **World map** tab shades every country by its rate (or count) for the window,
  outlines the selected countries, and ranks all countries in a table below.
- The **Spatial statistics** tab tests whether high rates cluster geographically
  (global Moran's I) and maps the clusters (LISA). Change the number of neighbours
  to see how stable the result is.
- The **Alerts** tab runs EARS C2 aberration detection on each selected country's
  daily counts and marks the days a surveillance system would have raised an alert.
- The **Summary table** tab gives totals, per-capita rates and deaths per 100 cases.
- **Download the plotted data** saves exactly what is on the chart as a CSV file.
""")

with tab_about:
    st.markdown(f"""
**Source.** Johns Hopkins University Center for Systems Science and Engineering (CSSE),
COVID-19 Data Repository, global time series of confirmed cases and deaths
([link]({REPO_URL})). Population, ISO3 codes and map positions come from the CSSE
`UID_ISO_FIPS_LookUp_Table.csv`. CSSE stopped collecting data on March 10, 2023, so
every series ends on **{max_date:%B %d, %Y}**.

**How the data were prepared**
- Countries reported by province or territory (for example China, Canada, Australia,
  the United Kingdom, and France, including overseas territories) were summed to
  one national total.
- The files report running totals. Daily new counts are the change from one day to
  the next.
- Some daily changes are negative because a country revised its total downward
  (France removed about 349,000 cases on May 20, 2021). These are corrections, not
  real counts, so by default they are hidden. Cumulative totals, the window totals in
  the tiles, the map and the summary table are left as reported, so corrections still
  count there.
- Five entries are not countries: Diamond Princess, MS Zaandam, Summer Olympics 2020,
  Winter Olympics 2022, and Antarctica. They can be viewed as counts but have no
  population, so their per-capita values are blank.
- Recovered counts are not shown because CSSE stopped tracking them in August 2021.

**Limits of the data.** Confirmed cases depend on how much testing each country did,
which changed over time and differs across countries. After home testing became
common in 2022, many cases were never reported. Many countries also moved to weekly
reporting, so a daily chart shows long flat stretches followed by spikes; the
7-day average spreads these out. Comparisons across countries should be read with
these limits in mind.

**Spatial statistics.** Global Moran's I and local Moran (LISA) from PySAL `esda`,
with k-nearest-neighbour weights (`libpysal`) on great-circle distance between the CSSE
country positions, 999 permutations. **Alerts.** EARS C2 (7-day baseline, 2-day lag,
3 SD threshold), the aberration-detection method used in syndromic surveillance.

**Built with** Python, Streamlit, pandas, Plotly, PySAL (esda, libpysal). Code: `covid_explorer.py` {VERSION}.

Genie Tang, research scientist, spatial epidemiologist. Portfolio: [spatialhealthinformatics.github.io](https://spatialhealthinformatics.github.io).
""")
