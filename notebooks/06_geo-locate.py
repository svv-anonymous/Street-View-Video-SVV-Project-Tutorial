#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCR-Depth Integration and Observer Geolocation.

Processes OCR results and metric depth maps to estimate the observer's geographic position through POI matching and spatial optimization.

1. OCR Matching: Fuzzy-match detected text to POI database.
2. Spatial Filtering: Filter outlier POIs based on cluster density.
3. Depth Mapping: Associate OCR regions with metric depth values (.npy).
4. Edge Selection: Identify road edges via distance and direction consistency.
5. Optimization: Localize observer by minimizing depth projection errors.

Expected Structure:
- Input: data/image/{series}/ocr/*_depth_results.csv
- Output: data/image/{series}/location/{series}_location.csv

Prepation: Please prepare your own OSM POI dataset as gpkg before running this script.

"""

import os
import glob
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox
from rapidfuzz import process, fuzz
from shapely.geometry import Point
from geopy.distance import geodesic


# POI LOADING & CLEANING

def load_poi_gdf(gpkg_path):
    """
    Load POI GeoDataFrame from a GPKG file and perform basic cleaning.

    - Reprojects to WGS84 (EPSG:4326)
    - Drops POIs without valid names (OCR matching relies on names)
    """
    gdf = gpd.read_file(gpkg_path).to_crs(epsg=4326)
    gdf["name"] = gdf["name"].replace("nan", np.nan)
    return gdf.dropna(subset=["name"])


# OCR - POI Fuzzy matching

def fuzzy_match_all(df_places, gdf_poi, threshold=75, limit=20):
    """
    Fuzzy-match OCR-recognized text against POI names.

    For each OCR text entry:
    - Compute string similarity against all POI names
    - Retain candidates above a similarity threshold
    - Return all matched POIs with geometry and similarity score

    Parameters:
    - threshold: minimum string similarity score to accept a match
    - limit: maximum number of fuzzy matches per OCR text
    """
    poi_map = {n.lower().strip(): n for n in gdf_poi["name"].unique()}
    keys = list(poi_map.keys())

    results = []
    for _, row in df_places.iterrows():
        text = row.get("text", "")
        if pd.isna(text):
            results.append([])
            continue

        matches = process.extract(
            str(text).lower(), keys, scorer=fuzz.ratio, limit=limit
        )

        recs = []
        for key, score, _ in matches:
            if score < threshold:
                continue

            name = poi_map[key]
            for _, poi in gdf_poi[gdf_poi["name"] == name].iterrows():
                recs.append({
                    "osm_id": poi["osm_id"],
                    "name": name,
                    "lon": poi.geometry.x,
                    "lat": poi.geometry.y,
                    "score": score
                })

        results.append(recs)

    out = df_places.copy()
    out["matched_pois"] = results
    return out


def retrieve_all_matched_pois(df_matched, gdf_poi):
    """
    Retrieve the full POI records that were matched by OCR text.

    This step limits downstream processing to only POIs that appear at least once in fuzzy matching.
    """
    names = {item["name"] for row in df_matched["matched_pois"] for item in row}
    return gdf_poi[gdf_poi["name"].isin(names)].copy()


# Spatial coexistence filtering

def spatial_coexistence_filter_with_fuzzy(
    pois_gdf,
    radius_meter=50,
    fuzzy_threshold=90,
    min_neighbors=2
):
    """
    Filter POIs based on spatial coexistence constraints.

    A POI is kept only if:
    - It has at least `min_neighbors` nearby POIs within `radius_meter`
    - Neighbor POIs are not trivially similar in name

    This removes isolated or spurious OCR-POI matches.
    """
    if len(pois_gdf) < min_neighbors + 1:
        return pois_gdf.iloc[[]]

    gdf = pois_gdf.to_crs(epsg=3857).copy()
    gdf["name_clean"] = gdf["name"].str.lower().str.strip()
    gdf["geom"] = gdf.geometry

    valid_ids = set()

    for i in gdf.index:
        ni = gdf.at[i, "name_clean"]
        pi = gdf.at[i, "geom"]
        neighbors = []

        for j in gdf.index:
            if j == i:
                continue

            nj = gdf.at[j, "name_clean"]
            if ni == nj or fuzz.ratio(ni, nj) >= fuzzy_threshold:
                continue

            pj = gdf.at[j, "geom"]
            if pi.distance(pj) <= radius_meter:
                neighbors.append(j)

        if len(neighbors) >= min_neighbors:
            valid_ids.add(gdf.at[i, "osm_id"])
            for j in neighbors:
                valid_ids.add(gdf.at[j, "osm_id"])

    return pois_gdf[pois_gdf["osm_id"].isin(valid_ids)].copy()


# OCR + DEPTH aggregation

def attach_ocr_info_with_avg_depth(filtered_pois, df_matched):
    """
    Attach OCR text and average depth values to filtered POIs.

    For each POI:
    - Aggregate depth values across frames
    - Aggregate OCR text 

    Output is a POI-level table suitable for spatial reasoning.
    """
    records = []

    for row in df_matched.itertuples():
        for poi in row.matched_pois or []:
            if poi["osm_id"] in set(filtered_pois["osm_id"]):
                records.append({
                    "osm_id": poi["osm_id"],
                    "name": poi["name"],
                    "lon": poi["lon"],
                    "lat": poi["lat"],
                    "avg_depth": row.avg_depth,
                    "ocr_text": row.text
                })

    if not records:
        return pd.DataFrame(
            columns=["osm_id", "name", "lon", "lat", "avg_depth", "ocr_text"]
        )

    df_temp = pd.DataFrame(records)

    return df_temp.groupby(
        ["osm_id", "name", "lon", "lat"],
        as_index=False
    ).agg({
        "avg_depth": "mean",
        "ocr_text": lambda x: ", ".join(sorted(set(x)))
    })


# Road network edge selection

def load_network_edges(graphml_path):
    """
    Load road network edges from a GraphML file using OSMnx.

    Only edges are needed; nodes are discarded.
    """
    G = ox.load_graphml(
        graphml_path,
        edge_dtypes={"oneway": str, "reversed": str}
    )
    return ox.graph_to_gdfs(G, nodes=False)


def select_best_edge(edges, poi_gdf):
    """
    Select the most plausible road edge based on:
    - Mean distance to POIs
    - Directional consistency with POI depth ordering

    The intuition:
    - POIs closer/farther in depth should align with the road direction
    """
    edges_proj = edges.to_crs(epsg=3857)
    pois_proj = poi_gdf.to_crs(epsg=3857)

    edges_proj["mean_dist"] = edges_proj.geometry.apply(
        lambda seg: pois_proj.distance(seg).mean()
    )

    top2 = edges_proj.nsmallest(2, "mean_dist").copy()

    sorted_p = pois_proj.sort_values("avg_depth")
    if len(sorted_p) < 2:
        return None

    p0, p1 = sorted_p.geometry.iloc[0], sorted_p.geometry.iloc[-1]
    v = np.array([p1.x - p0.x, p1.y - p0.y])

    if np.linalg.norm(v) == 0:
        return None

    v /= np.linalg.norm(v)

    def dir_sim(seg):
        coords = np.array(seg.coords)
        ve = coords[-1] - coords[0]
        if np.linalg.norm(ve) == 0:
            return np.nan
        ve /= np.linalg.norm(ve)
        return abs(np.dot(ve, v))

    top2["dir_sim"] = top2.geometry.apply(dir_sim)

    if top2["dir_sim"].isna().all():
        return None

    idx = top2["dir_sim"].idxmax()
    return edges.loc[idx].geometry


# Obeserver position estimation

def find_best_observer(edge_geom, poi_gdf, samples=100):
    """
    Estimate observer position along a road edge.

    - Uniformly sample points along the edge
    - Choose the point minimizing squared error between geometric distance and estimated depth
    """
    line = (
        gpd.GeoSeries([edge_geom], crs="EPSG:4326")
        .to_crs(epsg=3857)
        .iloc[0]
    )

    pois = poi_gdf.to_crs(epsg=3857)

    best_pt = None
    best_score = float("inf")

    for frac in np.linspace(0, 1, samples):
        pt = line.interpolate(frac, normalized=True)
        score = sum(
            (pt.distance(row.geometry) - row["avg_depth"]) ** 2
            for _, row in pois.iterrows()
        )

        if score < best_score:
            best_pt = pt
            best_score = score

    if best_pt is None:
        return None

    return (
        gpd.GeoSeries([best_pt], crs="EPSG:3857")
        .to_crs(epsg=4326)
        .iloc[0]
    )


# Folder level processing

def process_folder(folder, gpkg, graphml):
    """
    Process one OCR-result folder and estimate observer locations over time.

    Each CSV file corresponds to one time step.
    """
    poi_gdf = load_poi_gdf(gpkg)
    edges = load_network_edges(graphml)

    results = []

    for csv_file in sorted(glob.glob(os.path.join(folder, "*_depth_results.csv"))):
        df = pd.read_csv(csv_file)

        matched = fuzzy_match_all(df, poi_gdf)
        retrieved = retrieve_all_matched_pois(matched, poi_gdf)
        filtered = spatial_coexistence_filter_with_fuzzy(retrieved)
        agg = attach_ocr_info_with_avg_depth(filtered, matched)

        if agg.empty:
            continue

        agg_gdf = gpd.GeoDataFrame(
            agg,
            geometry=gpd.points_from_xy(agg.lon, agg.lat),
            crs="EPSG:4326"
        )

        edge = select_best_edge(edges, agg_gdf)
        if edge is None:
            continue

        observer = find_best_observer(edge, agg_gdf)
        if observer is None:
            continue

        seq = os.path.basename(csv_file).split("_")[1]
        results.append({
            "time": seq,
            "lat": observer.y,
            "lng": observer.x
        })

    df_res = pd.DataFrame(results)

    if df_res.empty:
        return df_res

    # Remove isolated estimates and extreme outliers
    if len(df_res) > 2:
        coords = df_res[["lat", "lng"]].values

        keep_idx = []
        for i, p in enumerate(coords):
            neigh = []
            if i > 0:
                neigh.append(coords[i - 1])
            if i < len(coords) - 1:
                neigh.append(coords[i + 1])

            if len(neigh) == 2 and all(
                geodesic(p, n).meters > 150 for n in neigh
            ):
                continue

            keep_idx.append(i)

        df_res = df_res.iloc[keep_idx].reset_index(drop=True)

        keep2 = []
        for i, p in enumerate(df_res[["lat", "lng"]].values):
            dists = [
                geodesic(p, q).meters
                for j, q in enumerate(df_res[["lat", "lng"]].values)
                if i != j
            ]

            if len(dists) <= 1 or np.mean(dists[1:]) <= 500:
                keep2.append(i)

        df_res = df_res.iloc[keep2].reset_index(drop=True)

    return df_res



def main():

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    IMAGE_ROOT = PROJECT_ROOT / "data" / "image"
    POI_GPKG = PROJECT_ROOT / "geo-files" / "POI.gpkg"
    GRAPH_PATH = PROJECT_ROOT / "geo-files-processed" / "road_network_unsplit.graphml"

    if not IMAGE_ROOT.exists():
        print(f"Error: {IMAGE_ROOT} not found.")
        return

    for subdir in IMAGE_ROOT.iterdir():
        if not subdir.is_dir():
            continue

        series_name = subdir.name
        ocr_dir = subdir / "ocr"

        if not ocr_dir.is_dir():
            print(f"{series_name}: no ocr directory -> skipped")
            continue

        try:
            df_out = process_folder(ocr_dir, POI_GPKG, GRAPH_PATH)

            if df_out.empty:
                print(f"{series_name}: no localization results -> skipped")
                continue

            df_out.insert(0, "agent_id", "a1")

            loc_dir = subdir / "location"
            loc_dir.mkdir(parents=True, exist_ok=True)

            out_file = loc_dir / f"{series_name}_location.csv"
            df_out.to_csv(out_file, index=False)

            print(f"{series_name}: success -> {out_file.relative_to(PROJECT_ROOT)}")

        except Exception as ex:
            print(f"{series_name}: processing error: {ex} -> skipped")


if __name__ == "__main__":
    main()
