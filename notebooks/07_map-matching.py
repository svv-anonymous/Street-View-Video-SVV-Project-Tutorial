#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Map-Matching.

Aligns estimated observer locations to the 50m segmented road network using the GoTrackIt map-matching algorithm.

"""
import os
import glob
from pathlib import Path
import pandas as pd
import geopandas as gpd
from gotrackit.map.Net import Net
from gotrackit.MapMatch import MapMatch


def main():
    """Match GPS trajectories to road network for all video series."""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    IMAGE_ROOT = PROJECT_ROOT / "data" / "image"
    link_path = PROJECT_ROOT / "geo-files-processed" / "road_network_50m_edges.shp"
    node_path = PROJECT_ROOT / "geo-files-processed" / "road_network_50m_nodes.shp"

    # Load road network and initialize Net
    link = gpd.read_file(link_path)
    node = gpd.read_file(node_path)
    my_net = Net(link_gdf=link, node_gdf=node, not_conn_cost=1200, cut_off=1200)
    my_net.init_net()

    # Iterate over each series subfolder
    for series_dir in IMAGE_ROOT.iterdir():
        if not series_dir.is_dir():
            continue
            
        series_name = series_dir.name
        csv_loc_dir = series_dir / 'location'
        
        if not csv_loc_dir.exists():
            print(f"Skipping '{series_name}': no 'location' folder")
            continue

        csv_files = list(csv_loc_dir.glob('*.csv'))
        if not csv_files:
            print(f"Skipping '{series_name}': no .csv found")
            continue
        
        csv_file = csv_files[0]
        gps_df = pd.read_csv(csv_file)

        out_folder = series_dir / 'map-matching'
        out_folder.mkdir(parents=True, exist_ok=True)

        # Build MapMatch object
        mpm = MapMatch(
            net=my_net,
            use_sub_net=True,
            flag_name='sparse_sample', 
            time_format='%Y-%m-%d %H:%M:%S',
            dense_gps=False,
            gps_buffer=700,
            top_k=20,
            use_heading_inf=True,
            export_html=False,
            export_geo_res=True, 
            out_fldr=str(out_folder), 
            gps_radius=15.0
        )

        try:
            match_res, warn_info, error_info = mpm.execute(gps_df=gps_df)
            
            csv_out = out_folder / 'match_res.csv'
            match_res.to_csv(csv_out, encoding='utf_8_sig', index=False)
            print(f"'{series_name}' done -> {out_folder.relative_to(PROJECT_ROOT)}")
        except Exception as e:
            print(f"'{series_name}' failed: {e}")


if __name__ == '__main__':
    main()
