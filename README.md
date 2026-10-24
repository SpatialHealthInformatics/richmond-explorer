# Richmond Spatial Decision Intelligence Explorer (v2)

Genie Tang. A working instance of the Dashboard component of the Spatial Decision Intelligence Framework from the PhD proposal *Health Equity Through Sustainable Urban Development* (University of Cambridge). Case study: Richmond, California, within Contra Costa County, at census-tract resolution.

## What it shows

- **Map**: 26 indicators from the CDC/ATSDR Environmental Justice Index 2024 and CDC PLACES 2023 (exposure, built environment, index, health, social) by tract, in quintiles or continuous, with the City of Richmond outlined.
- **Clusters**: global Moran's I and local Moran (LISA) clusters for any indicator, with queen, rook or k-nearest weights and 199 to 4,999 permutations. Adults with asthma, queen contiguity, 999 permutations gives I = 0.69, as in the proposal (Section 06.05, Figure 11).
- **Bivariate map**: terciles of an exposure (x) against a burden or outcome (y), as in Figure 12.
- **Priority tracts**: tracts in the top tercile on both x and y, with a table and CSV download (the operation described in Section 06.04).
- **Table**, **How to use**, **About the data**.

Every map has a toolbar (zoom, pan, reset, download as PNG) and a **Reset map view** button that returns it to the whole county.

## Run it

```
pip install -r requirements.txt
streamlit run richmond_explorer.py
```

The app reads `data/contra_costa_tracts.geojson` and `data/richmond.geojson`, which are included, so no network access or downloads are needed. Keep `.streamlit/config.toml` (theme) and `static/world_110m.json` (map outlines) next to the script.

## Rebuild the data

`prepare_data.py` builds the two GeoJSON files from the raw downloads (not included in the repository because of their size):

- `EJI_2024_California.csv` from the CDC/ATSDR EJI data download (also archived at github.com/oedp/cdc-ej-index)
- `PLACES_2023_CA_tract_asthma.csv` from data.cdc.gov (dataset cwsq-ngmh, filtered to California and CASTHMA)
- `contra_costa_tracts_2020.geojson` and `richmond_place_2020.geojson`, Census 2020 cartographic boundaries (1:500k)

Place them in `data_raw/` and run `python prepare_data.py`.

## Methods and limits

See the About the data tab in the app. Built with Python, Streamlit, pandas, GeoPandas, Plotly and PySAL.
