#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Link-time mapping.

Maps continuous time sequences to specific road network links by interpolating observer trajectories and spatial joining with matched road segments.

Expected Structure:
- Input: data/image/{series}/location/*.csv & data/image/{series}/map-matching/*.geojson
- Output: data/image/{series}/road_with_results/road.csv
"""
import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString


def process_folder(subfolder):
    """
    Process a single video series folder to create link-time mapping.
    
    Args:
        subfolder: Path object pointing to the video series folder
    """
    location_dir = subfolder / 'location'
    map_dir = subfolder / 'map-matching'
    out_dir = subfolder / 'road_with_results'
    out_dir.mkdir(parents=True, exist_ok=True)

    node_csvs = list(location_dir.glob('*.csv'))
    lines_geojsons = list(map_dir.glob('*match_link.geojson'))

    if not node_csvs or not lines_geojsons:
        print(f"Skip {subfolder.name} - missing required files in location/ or map-matching/")
        return

    nodes_csv = node_csvs[0]
    lines_geojson = lines_geojsons[0]

    # Read node and line data
    nodes_df = pd.read_csv(nodes_csv)
    gdf_nodes = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(nodes_df['lng'], nodes_df['lat']),
        crs='EPSG:4326'
    )
    gdf_lines = gpd.read_file(lines_geojson)

    # Project to metric CRS (3857) for accurate distance calculations
    gdf_nodes_m = gdf_nodes.to_crs(epsg=3857)
    gdf_lines_m = gdf_lines.to_crs(epsg=3857)

    # Merge all matched links into a single route for projection
    route = gdf_lines_m.geometry.union_all()
    gdf_nodes_m['dist_on_route'] = gdf_nodes_m.geometry.apply(lambda pt: route.project(pt))

    # Extract (time, distance_along_route) and sort
    seq_dist = (
        gdf_nodes_m[['time', 'dist_on_route']]
        .dropna(subset=['time'])
        .drop_duplicates('time')
        .sort_values('time')
        .reset_index(drop=True)
    )

    # Linearly interpolate distances for missing intermediate time steps
    filled = []
    records = seq_dist.to_dict('records')
    for i in range(len(records) - 1):
        s0, d0 = int(records[i]['time']), records[i]['dist_on_route']
        s1, d1 = int(records[i+1]['time']), records[i+1]['dist_on_route']
        filled.append({'time': s0, 'dist_on_route': d0})
        for s in range(s0 + 1, s1):
            frac = (s - s0) / (s1 - s0)
            d = d0 + frac * (d1 - d0)
            filled.append({'time': s, 'dist_on_route': d})
    filled.append({'time': int(records[-1]['time']), 'dist_on_route': records[-1]['dist_on_route']})

    gdf_filled_m = gpd.GeoDataFrame(
        filled,
        geometry=[route.interpolate(r['dist_on_route']) for r in filled],
        crs='EPSG:3857'
    )
    gdf_filled = gdf_filled_m.to_crs(epsg=4326)

    # Build line segments between consecutive points
    pts = gdf_filled.sort_values('time').reset_index(drop=True)
    segs = [
        {
            'time_start': int(pts.loc[i, 'time']),
            'time_end': int(pts.loc[i+1, 'time']),
            'geometry': LineString([pts.loc[i, 'geometry'], pts.loc[i+1, 'geometry']])
        }
        for i in range(len(pts) - 1)
    ]
    gdf_segs = gpd.GeoDataFrame(segs, crs='EPSG:4326')

    # Assign a link_id to each segment (best overlap / nearest)
    gdf_segs_m = gdf_segs.to_crs(epsg=3857)
    sidx = gdf_lines_m.sindex
    assigned_link_ids = []

    for seg in gdf_segs_m.geometry:
        cand = list(sidx.intersection(seg.bounds))
        if cand:
            lengths = [seg.intersection(gdf_lines_m.geometry.iloc[j]).length for j in cand]
            j_best = cand[int(np.argmax(lengths))]
        else:
            j_best = list(sidx.nearest(seg, 1))[0]
        assigned_link_ids.append(gdf_lines_m.iloc[j_best]['link_id'])
    
    gdf_segs['link_id'] = assigned_link_ids

    # Aggregate time indices per link_id
    road_segments = (
        gdf_segs
        .groupby('link_id')
        .apply(lambda df: df.time_start.astype(int).tolist())
        .to_dict()
    )

    # Save aggregated times to CSV
    out_csv = out_dir / 'road.csv'
    rows = []
    for lid, times in road_segments.items():
        row = [lid] + times
        rows.append(row)
    
    maxlen = max(len(r) for r in rows) if rows else 0
    columns = ['link_id'] + [f'time_{i+1}' for i in range(maxlen - 1)]
    pd.DataFrame(rows, columns=columns).to_csv(out_csv, index=False)
    
    print(f"Processed {subfolder.name}, saved -> {out_csv.name}")

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    IMAGE_ROOT = PROJECT_ROOT / "data" / "image"

    if not IMAGE_ROOT.exists():
        print(f"Error: {IMAGE_ROOT} not found.")
        return

    error_folders = []
    
    for subdir in IMAGE_ROOT.iterdir():
        if not subdir.is_dir():
            continue
            
        try:
            process_folder(subdir)
        except Exception as e:
            print(f"Error processing {subdir.name}: {e}")
            error_folders.append(subdir.name)

    if error_folders:
        print(f"\nFolders with errors: {error_folders}")

if __name__ == '__main__':
    main()