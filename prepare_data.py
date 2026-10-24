# =============================================================================
# Richmond Spatial Decision Intelligence Explorer: data preparation
# Script:   prepare_data.py   Version: v1 (2026-10-24)
# Author:   Genie Tang
#
# Builds data/contra_costa_tracts.geojson (241 tracts with the EJI 2024
# indicators and CDC PLACES 2023 asthma) and data/richmond.geojson (city
# boundary) from the raw downloads in data_raw/. Run once; the app reads
# only the two output files, so it needs no network access.
#
# Inputs (data_raw/)
#   EJI_2024_California.csv            CDC/ATSDR Environmental Justice Index 2024
#   PLACES_2023_CA_tract_asthma.csv    CDC PLACES 2023 release, CASTHMA crude prevalence
#   contra_costa_tracts_2020.geojson   Census 2020 cartographic tracts (1:500k), Contra Costa
#   richmond_place_2020.geojson        Census 2020 cartographic place boundary, Richmond city
# =============================================================================
import json
import pandas as pd
import geopandas as gpd

RAW = "data_raw/"
OUT = "data/"

KEEP = {
    "GEOID": "GEOID", "COUNTY": "COUNTY", "E_TOTPOP": "pop",
    "E_PM": "pm25_days", "E_OZONE": "ozone_days", "E_DSLPM": "diesel_pm",
    "E_ROAD": "near_road", "E_TRI": "near_tri", "E_NPL": "near_npl",
    "E_PARK": "near_park", "E_HOUAGE": "pre1980_housing", "E_WLKIND": "walkability",
    "E_TOTCR": "air_toxics_cancer_risk", "E_NOINT": "no_internet", "E_LIMENG": "limited_english",
    "E_ASTHMA": "asthma_eji", "E_CHD": "heart_disease", "E_DIABETES": "diabetes",
    "E_MHLTH": "poor_mental_health", "E_CANCER": "cancer",
    "E_POV200": "poverty200", "E_UNEMP": "unemployed", "E_RENTER": "renter",
    "E_HOUBDN": "housing_burden", "E_UNINSUR": "uninsured", "E_AGE65": "age65",
    "E_DISABL": "disability", "E_MINRTY": "minority",
    "RPL_EJI": "eji_pct", "RPL_EBM": "ebm_pct", "RPL_SVM": "svm_pct", "RPL_HVM": "hvm_score",
    "EPL_PM": "pm25_pct", "EPL_ASTHMA": "asthma_pct",
}

eji = pd.read_csv(RAW + "EJI_2024_California.csv", low_memory=False, dtype={"GEOID": str})
eji = eji[eji["COUNTY"].str.contains("Contra Costa")][list(KEEP)].rename(columns=KEEP)
eji = eji.replace(-999, float("nan"))
for c in eji.columns:
    if c not in ("GEOID", "COUNTY"):
        eji[c] = pd.to_numeric(eji[c], errors="coerce")
eji["GEOID"] = eji["GEOID"].str.zfill(11)

places = pd.read_csv(RAW + "PLACES_2023_CA_tract_asthma.csv", dtype={"locationid": str})
places = places[places["countyname"] == "Contra Costa"][["locationid", "data_value"]]
places = places.rename(columns={"locationid": "GEOID", "data_value": "asthma_places2023"})

tracts = gpd.read_file(RAW + "contra_costa_tracts_2020.geojson")[["GEOID", "NAME", "ALAND", "geometry"]]
tracts = tracts.merge(eji, on="GEOID", how="left").merge(places, on="GEOID", how="left")
tracts["name"] = "Tract " + tracts["NAME"]

richmond = gpd.read_file(RAW + "richmond_place_2020.geojson")[["NAME", "geometry"]]
# Tracts whose centroid falls inside the city boundary count as Richmond tracts.
cent = tracts.copy()
cent["geometry"] = cent.geometry.representative_point()
inside = gpd.sjoin(cent, richmond, predicate="within", how="inner")["GEOID"]
tracts["in_richmond"] = tracts["GEOID"].isin(inside)

tracts = tracts.drop(columns=["NAME"])
tracts.to_file(OUT + "contra_costa_tracts.geojson", driver="GeoJSON")
richmond.to_file(OUT + "richmond.geojson", driver="GeoJSON")
print(len(tracts), "tracts;", int(tracts["in_richmond"].sum()), "in Richmond;",
      "missing EJI:", int(tracts["eji_pct"].isna().sum()),
      "missing PLACES:", int(tracts["asthma_places2023"].isna().sum()))
