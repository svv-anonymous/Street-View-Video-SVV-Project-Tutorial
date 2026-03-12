#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate pedestrian and audio features by road link with day/night split.

This script aggregates extracted pedestrian and audio features from video clips and maps them to road network links. Only video series listed in day_list.txt or night_list.txt are included. 
Features are aggregated at three levels:

1. Overall: all valid series in day_list ∪ night_list
2. Day: only series listed in day_list.txt
3. Night: only series listed in night_list.txt

Expected Structure:
- Input:  data/image/{series}/road_with_results/road.csv
- Input:  data/image/{series}/audio_result/*.csv
- Input:  data/image/{series}/video_result/*.csv
- Input:  day_list.txt
- Input:  night_list.txt
- Input:  geo-files-processed/road_network_50m_edges.shp

Outputs:
- Output: data/results/link_audio_mean_daynight.csv
- Output: data/results/link_ped_mean_daynight.csv
- Output: geo-files-output/road_network_50m_edges_with_results.shp
"""

import os
from pathlib import Path
from collections import defaultdict

import pandas as pd
import geopandas as gpd
from tqdm import tqdm


try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path(os.getcwd()).parent


# =========================================================
# PATH CONFIGURATION
# =========================================================

PARENT_DIR = PROJECT_ROOT / "data" / "image"
RESULT_DIR = PROJECT_ROOT / "data" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DAY_LIST = PROJECT_ROOT / "day_list.txt"
NIGHT_LIST = PROJECT_ROOT / "night_list.txt"

OUTPUT_CSV_AUDIO = RESULT_DIR / "link_audio_mean_daynight.csv"
OUTPUT_CSV_PED = RESULT_DIR / "link_ped_mean_daynight.csv"

EDGES_SHP = PROJECT_ROOT / "geo-files-processed" / "road_network_50m_edges.shp"
OUT_SHP = PROJECT_ROOT / "geo-files-output" / "road_network_50m_edges_with_results.shp"
OUT_SHP.parent.mkdir(parents=True, exist_ok=True)


TOPK = 5
EXCLUDE_TAGS_FROM_TOPK = {"total"}

ACTIONS_TO_AGG = ["walking", "standing", "sitting"]
TOTAL_ACTION_NAME = "total_unique_people"



def read_list(path: Path):
    """Read a txt list file into a set."""
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def mean_or_none(values):
    """Return mean of a list, or None if empty."""
    if not values:
        return None
    return sum(values) / len(values)


def safe_read_csv(path: Path):
    """Read csv safely, return DataFrame or None."""
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def get_series_group(series_name, day_series, night_series):
    """Return series group: day / night / None."""
    if series_name in day_series:
        return "day"
    if series_name in night_series:
        return "night"
    return None


def collect_time_values_from_road_row(row):
    """
    Extract mapped clip times from a road.csv row.
    Assumes first column is link_id and the rest are time columns.
    """
    return [v for v in row.iloc[1:].dropna().values]


def update_nested_dict(store, link_id, key, value):
    """Append value into nested defaultdict structure."""
    store[link_id][key].append(value)


def get_all_keys_from_nested_dict(nested_dict):
    """Collect all second-level keys from nested dict."""
    keys = set()
    for subdict in nested_dict.values():
        keys.update(subdict.keys())
    return sorted(keys)


day_series = read_list(DAY_LIST)
night_series = read_list(NIGHT_LIST)
valid_series = day_series.union(night_series)

print(f"Day series: {len(day_series)}")
print(f"Night series: {len(night_series)}")
print(f"Total valid series: {len(valid_series)}")


# =========================================================
# AUDIO AGGREGATION
# =========================================================

def aggregate_audio():
    """
    Aggregate audio tag values by link for overall / day / night.
    """
    all_link_tag_values_all = defaultdict(lambda: defaultdict(list))
    all_link_tag_values_day = defaultdict(lambda: defaultdict(list))
    all_link_tag_values_night = defaultdict(lambda: defaultdict(list))

    print("\nProcessing audio information...")

    for folder in tqdm(sorted(os.listdir(PARENT_DIR)), desc="Audio"):
        folder_path = PARENT_DIR / folder

        if not folder_path.is_dir():
            continue

        if folder not in valid_series:
            continue

        road_csv = folder_path / "road_with_results" / "road.csv"
        audio_dir = folder_path / "audio_result"

        if not (road_csv.exists() and audio_dir.exists()):
            continue

        road_df = safe_read_csv(road_csv)
        if road_df is None or "link_id" not in road_df.columns:
            continue

        series_group = get_series_group(folder, day_series, night_series)

        for _, row in road_df.iterrows():
            link_id = row["link_id"]
            times = collect_time_values_from_road_row(row)

            for t in times:
                try:
                    clip_name = f"clip_{int(t):04d}_audio.csv"
                except Exception:
                    continue

                clip_path = audio_dir / clip_name
                if not clip_path.exists():
                    continue

                clip_df = safe_read_csv(clip_path)
                if clip_df is None:
                    continue

                if not {"tag", "value"}.issubset(clip_df.columns):
                    continue

                for tag, val in zip(clip_df["tag"].values, clip_df["value"].values):
                    try:
                        v = float(val)
                    except Exception:
                        continue

                    update_nested_dict(all_link_tag_values_all, link_id, tag, v)

                    if series_group == "day":
                        update_nested_dict(all_link_tag_values_day, link_id, tag, v)
                    elif series_group == "night":
                        update_nested_dict(all_link_tag_values_night, link_id, tag, v)

    if not all_link_tag_values_all:
        print("Warning: No audio data found.")
        return pd.DataFrame(columns=["link_id", "audio_mean", "audio_day", "audio_night"])

    all_tags = get_all_keys_from_nested_dict(all_link_tag_values_all)

    rows_audio = []
    all_link_ids = sorted(set(all_link_tag_values_all.keys()) |
                          set(all_link_tag_values_day.keys()) |
                          set(all_link_tag_values_night.keys()))

    for link_id in all_link_ids:
        row = {"link_id": link_id}

        # overall tag means
        for tag in all_tags:
            row[tag] = mean_or_none(all_link_tag_values_all[link_id].get(tag, []))

        # total sound intensity
        row["audio_mean"] = mean_or_none(all_link_tag_values_all[link_id].get("total", []))
        row["audio_day"] = mean_or_none(all_link_tag_values_day[link_id].get("total", []))
        row["audio_night"] = mean_or_none(all_link_tag_values_night[link_id].get("total", []))

        rows_audio.append(row)

    df_audio = pd.DataFrame(rows_audio)

    # pick global top-K tags from overall means
    tag_cols = [c for c in df_audio.columns if c not in {"link_id", "audio_mean", "audio_day", "audio_night"}]
    means = df_audio[tag_cols].mean(numeric_only=True)
    means = means.drop(labels=[t for t in EXCLUDE_TAGS_FROM_TOPK if t in means.index], errors="ignore")
    top_tags = means.sort_values(ascending=False).head(TOPK).index.tolist()

    print("Top audio tags:", top_tags)

    out_cols = ["link_id", "audio_mean", "audio_day", "audio_night"] + [t for t in top_tags if t in df_audio.columns]
    out_df_audio = df_audio[out_cols].copy()
    out_df_audio.to_csv(OUTPUT_CSV_AUDIO, index=False)

    print(f"Done. Output to {OUTPUT_CSV_AUDIO}")
    return out_df_audio


# =========================================================
# PEDESTRIAN AGGREGATION
# =========================================================

def aggregate_pedestrian():
    """
    Aggregate pedestrian counts by link for overall / day / night.
    """
    all_link_action_values_all = defaultdict(lambda: defaultdict(list))
    all_link_action_values_day = defaultdict(lambda: defaultdict(list))
    all_link_action_values_night = defaultdict(lambda: defaultdict(list))

    print("\nProcessing pedestrian information...")

    for folder in tqdm(sorted(os.listdir(PARENT_DIR)), desc="Pedestrian"):
        folder_path = PARENT_DIR / folder

        if not folder_path.is_dir():
            continue

        if folder not in valid_series:
            continue

        road_csv = folder_path / "road_with_results" / "road.csv"
        video_dir = folder_path / "video_result"

        if not (road_csv.exists() and video_dir.exists()):
            continue

        road_df = safe_read_csv(road_csv)
        if road_df is None or "link_id" not in road_df.columns:
            continue

        series_group = get_series_group(folder, day_series, night_series)

        for _, row in road_df.iterrows():
            link_id = row["link_id"]
            times = collect_time_values_from_road_row(row)

            for t in times:
                try:
                    clip_name = f"clip_{int(t):04d}.csv"
                except Exception:
                    continue

                clip_path = video_dir / clip_name
                if not clip_path.exists():
                    continue

                clip_df = safe_read_csv(clip_path)
                if clip_df is None:
                    continue

                if not {"action", "count"}.issubset(clip_df.columns):
                    continue

                for action, cnt in zip(clip_df["action"].values, clip_df["count"].values):
                    if action != TOTAL_ACTION_NAME and action not in ACTIONS_TO_AGG:
                        continue

                    try:
                        v = float(cnt)
                    except Exception:
                        continue

                    update_nested_dict(all_link_action_values_all, link_id, action, v)

                    if series_group == "day":
                        update_nested_dict(all_link_action_values_day, link_id, action, v)
                    elif series_group == "night":
                        update_nested_dict(all_link_action_values_night, link_id, action, v)

    if not all_link_action_values_all:
        print("Warning: No pedestrian data found.")
        return pd.DataFrame(columns=[
            "link_id", "ped_mean", "ped_day", "ped_night",
            "walking", "standing", "sitting"
        ])

    rows_ped = []
    all_link_ids = sorted(set(all_link_action_values_all.keys()) |
                          set(all_link_action_values_day.keys()) |
                          set(all_link_action_values_night.keys()))

    for link_id in all_link_ids:
        row = {"link_id": link_id}

        # total pedestrian count
        row["ped_mean"] = mean_or_none(all_link_action_values_all[link_id].get(TOTAL_ACTION_NAME, []))
        row["ped_day"] = mean_or_none(all_link_action_values_day[link_id].get(TOTAL_ACTION_NAME, []))
        row["ped_night"] = mean_or_none(all_link_action_values_night[link_id].get(TOTAL_ACTION_NAME, []))

        # action means (overall)
        for act in ACTIONS_TO_AGG:
            row[act] = mean_or_none(all_link_action_values_all[link_id].get(act, []))
            row[f"{act}_day"] = mean_or_none(all_link_action_values_day[link_id].get(act, []))
            row[f"{act}_night"] = mean_or_none(all_link_action_values_night[link_id].get(act, []))

        rows_ped.append(row)

    out_df_ped = pd.DataFrame(rows_ped)
    out_df_ped.to_csv(OUTPUT_CSV_PED, index=False)

    print(f"Done. Output to {OUTPUT_CSV_PED}")
    return out_df_ped



def merge_with_shapefile(df_ped, df_audio):
    """
    Merge aggregated results with road network shapefile.
    """
    print("\nMerging aggregated results with shapefile...")

    if not EDGES_SHP.exists():
        print(f"Error: Shapefile not found at {EDGES_SHP}")
        return

    gdf = gpd.read_file(EDGES_SHP)
    gdf["link_id"] = gdf["link_id"].astype(int)

    if not df_ped.empty and "link_id" in df_ped.columns:
        df_ped = df_ped.copy()
        df_ped["link_id"] = df_ped["link_id"].astype(int)
        gdf = gdf.merge(df_ped, on="link_id", how="left")

    if not df_audio.empty and "link_id" in df_audio.columns:
        df_audio = df_audio.copy()
        df_audio["link_id"] = df_audio["link_id"].astype(int)
        gdf = gdf.merge(df_audio, on="link_id", how="left")

    gdf.to_file(OUT_SHP)

    print(f"Done. Final output saved to {OUT_SHP}")
    print(f"Columns in final shapefile: {list(gdf.columns)}")

def main():
    df_audio = aggregate_audio()
    df_ped = aggregate_pedestrian()
    merge_with_shapefile(df_ped, df_audio)
    print("\nAll aggregation tasks complete.")


if __name__ == "__main__":
    main()