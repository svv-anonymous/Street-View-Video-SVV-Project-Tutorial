#!/usr/bin/env python3
"""
Route Validation and Accuracy Assessment.

Evaluates the quality of map-matching by comparing the matched road names against ground-truth timestamps (POI sequences). It uses a fuzzy Longest 
Common Subsequence (LCS) algorithm to handle OCR and mapping noise.

1. Name Mapping: Link internal link_ids to human-readable street names.
2. Route Extraction: Extract the sequence of road names from map-matching GeoJSON.
3. Fuzzy Validation: Compare the extracted sequence with ground truth (timestamp.txt).
4. Reporting: Generate a summary CSV with matching scores (LCS ratio).

- Input: data/image/{series}/map-matching/*.geojson
- Output: data/image/{series}/map-matching/timestamp.txt (Ground Truth)
- Output: geo-files-processed/map_match_check_summary.csv
"""

import os
import re
import json
import difflib
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import List
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEO_PROCESSED = PROJECT_ROOT / 'geo-files-processed'
IMAGE_PARENT = PROJECT_ROOT / 'data' / 'image'
VIDEO_PARENT = PROJECT_ROOT / 'data' / 'videos'

ROAD_GDF_PATH = GEO_PROCESSED / 'road_network_50m_edges.shp'
OUT_CSV = GEO_PROCESSED / 'map_match_check_summary.csv'

# Config
MAP_SUBDIR = "map-matching"
FUZZY_THRESHOLD = 70
BLACKLIST = {"introduction", "end", "subscribe", "outroduction", "thanks"}
# Regex to match "00:00 Location Name"
TS_LINE_RE = re.compile(r'^\s*(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+?)\s*$')

def normalize_text(text: str) -> str:
    """Clean text for comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s&/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def extract_pois_from_desc(desc_path: Path) -> List[str]:
    """Extract location names from .description file using regex."""
    locs = []
    if not desc_path.exists():
        return locs
    try:
        with open(desc_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = TS_LINE_RE.match(line)
                if m:
                    locs.append(m.group(2).strip())
    except Exception as e:
        print(f"Error reading {desc_path.name}: {e}")
    return locs

def prepare_ground_truth():
    """Scan video directories and generate timestamp.txt in the image directory."""
    print("Step 1: Extracting Ground Truth (POIs) from .description files...")
    
    # Iterate through folders in data/videos/
    for video_folder in VIDEO_PARENT.iterdir():
        if not video_folder.is_dir():
            continue
            
        series_name = video_folder.name
        desc_file = video_folder / f"{series_name}.description"
        target_mm_dir = IMAGE_PARENT / series_name / MAP_SUBDIR
        
        if desc_file.exists():
            pois = extract_pois_from_desc(desc_file)
            if pois:
                target_mm_dir.mkdir(parents=True, exist_ok=True)
                ts_out = target_mm_dir / "timestamp.txt"
                with open(ts_out, "w", encoding="utf-8") as f:
                    for p in pois:
                        f.write(p + "\n")
                # print(f" [OK] {series_name}: {len(pois)} POIs extracted.")

def lcs_ratio(seq_a: List[str], seq_b: List[str], threshold: float) -> float:
    """Fuzzy Longest Common Subsequence ratio."""
    if not seq_b: return 0.0
    A = [normalize_text(x) for x in seq_a]
    B = [normalize_text(x) for x in seq_b]
    
    prev = [0] * (len(B) + 1)
    for ai in A:
        curr = [0] * (len(B) + 1)
        for j, bj in enumerate(B):
            score = difflib.SequenceMatcher(None, ai, bj).ratio() * 100.0
            if score >= threshold:
                curr[j+1] = prev[j] + 1
            else:
                curr[j+1] = max(prev[j+1], curr[j])
        prev = curr
    return prev[len(B)] / len(B)

def main():
    # Generate timestamp.txt automatically
    prepare_ground_truth()

    # Load Road Network
    if not ROAD_GDF_PATH.exists():
        print(f"Error: Shapefile not found at {ROAD_GDF_PATH}")
        return
    print(f"Step 2: Loading road network...")
    gdf_road = gpd.read_file(ROAD_GDF_PATH)
    link2name = pd.Series(gdf_road.name.values, index=gdf_road.link_id.astype(int)).to_dict()

    # Validation Loop
    print(f"Step 3: Evaluating route accuracy...")
    results = []
    series_folders = [p for p in IMAGE_PARENT.iterdir() if p.is_dir()]

    for folder in tqdm(series_folders, desc="Validating Series"):
        base = folder.name
        mm_dir = folder / MAP_SUBDIR
        ts_path = mm_dir / "timestamp.txt"
        geojson_files = list(mm_dir.glob("*match_link.geojson"))

        if not ts_path.exists() or not geojson_files:
            continue

        # Extract names from Map-Matching GeoJSON
        try:
            with open(geojson_files[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            
            raw_names = []
            for ft in data.get("features", []):
                lid = ft.get("properties", {}).get("link_id")
                if lid:
                    name = link2name.get(int(float(lid)), "")
                    if name: raw_names.append(name)
            
            # De-duplicate consecutive names
            matched_names = []
            for n in raw_names:
                if not matched_names or matched_names[-1] != n:
                    matched_names.append(n)
            
            # Save the processed route names for debugging
            with open(mm_dir / "mapped-route-name.txt", "w", encoding="utf-8") as f:
                for nm in matched_names: f.write(nm + "\n")

            # Load POI Ground Truth
            with open(ts_path, "r", encoding="utf-8") as f:
                gt_names = [ln.strip() for ln in f if ln.strip()]
            
            gt_filtered = [n for n in gt_names if normalize_text(n) not in BLACKLIST]
            
            # Calculate LCS Score
            score = lcs_ratio(gt_filtered, matched_names, FUZZY_THRESHOLD)
            
            results.append({
                "series": base,
                "gt_poi_count": len(gt_filtered),
                "matched_road_count": len(matched_names),
                "accuracy_score": round(score, 4)
            })
        except Exception as e:
            print(f" Error processing {base}: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUT_CSV, index=False)
        print(f"\n[DONE] Summary saved to: {OUT_CSV.relative_to(PROJECT_ROOT)}")
        print(f"Mean Accuracy Score: {df['accuracy_score'].mean():.2%}")
    else:
        print("No validation results generated. Check if geojson/description files exist.")

if __name__ == "__main__":
    main()
