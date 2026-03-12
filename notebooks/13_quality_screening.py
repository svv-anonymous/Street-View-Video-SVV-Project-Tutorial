#!/usr/bin/env python3
"""
Video Quality Screening for Street Walking Videos.

This script performs automatic quality filtering for collected walking videos.
Two filtering rules are applied:

1. Distance anomaly filtering
   - Detect abnormal movement using trajectory points.
   - If any consecutive segment exceeds the allowed walking distance threshold, the video will be flagged and removed.

2. Audio filtering
   - Based on audio tagging results produced by the audio tagging script.
   - If any clip contains high probability of "music" or "speech" (above the defined threshold), the video will be removed.

Expected directory structure:

data/videos/{series}/
    location/*.csv
    audio_result/*.csv

Outputs:
    removed_videos.txt

"""

import math
import re
from pathlib import Path

import pandas as pd
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_ROOT / "data" / "videos"
OUTPUT_TXT = PROJECT_ROOT / "removed_videos.txt"

DIST_THRESHOLD_PER_10S = 50
AUDIO_THRESHOLD = 0.8

NUMBER_RE = re.compile(r"\d+")


def natural_sort(files):
    """Sort filenames with numeric order."""
    return sorted(
        files,
        key=lambda x: int(NUMBER_RE.findall(str(x))[-1]) if NUMBER_RE.findall(str(x)) else -1,
    )


def build_transformer(lon, lat):
    """Build coordinate transformer from WGS84 to local metric CRS."""
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def dist_m(tf, lon1, lat1, lon2, lat2):
    """Compute Euclidean distance (meters) between two lon/lat points."""
    x1, y1 = tf.transform(lon1, lat1)
    x2, y2 = tf.transform(lon2, lat2)
    return math.hypot(x2 - x1, y2 - y1)


def pick_location_csv(video_dir):
    """Pick the largest location CSV in location/ folder."""
    loc_dir = video_dir / "location"
    if not loc_dir.exists():
        return None
    csvs = list(loc_dir.glob("*.csv"))
    return max(csvs, key=lambda p: p.stat().st_size) if csvs else None


def check_distance(video_dir):
    """Return True if any adjacent trajectory segment is too long."""
    loc_csv = pick_location_csv(video_dir)
    if loc_csv is None:
        return False

    try:
        df = pd.read_csv(loc_csv)
        lats = df["lat"].astype(float).tolist()
        lngs = df["lng"].astype(float).tolist()
    except Exception:
        return False

    if len(lats) < 2:
        return False

    tf = build_transformer(lngs[0], lats[0])
    return any(
        dist_m(tf, lngs[i - 1], lats[i - 1], lngs[i], lats[i]) > DIST_THRESHOLD_PER_10S
        for i in range(1, len(lats))
    )


def read_tag_prob(csv_path, tag_name):
    """Read probability value for a specific audio tag."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if "tag" not in df.columns or "value" not in df.columns:
        return None

    row = df[df["tag"].str.lower() == tag_name.lower()]
    return float(row.iloc[0]["value"]) if len(row) else None


def check_audio(video_dir):
    """Return True if music/speech score exceeds threshold."""
    audio_dir = video_dir / "audio_result"
    if not audio_dir.exists():
        return False

    for csv_path in natural_sort(audio_dir.glob("*.csv")):
        music = read_tag_prob(csv_path, "Music")
        speech = read_tag_prob(csv_path, "Speech")
        if (music is not None and music >= AUDIO_THRESHOLD) or (
            speech is not None and speech >= AUDIO_THRESHOLD
        ):
            return True
    return False


def main():
    removed = []
    video_dirs = sorted((p for p in ROOT_DIR.iterdir() if p.is_dir()), key=lambda x: x.name)

    for video_dir in video_dirs:
        video_name = video_dir.name
        reasons = [
            name
            for name, flagged in (
                ("distance", check_distance(video_dir)),
                ("audio", check_audio(video_dir)),
            )
            if flagged
        ]

        if reasons:
            removed.append(video_name)
            print(f"REMOVE: {video_name} ({','.join(reasons)})")
        else:
            print(f"KEEP: {video_name}")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        if removed:
            f.write("\n".join(removed) + "\n")

    print("\nFiltering complete.")
    print(f"Removed videos: {len(removed)}")
    print(f"Saved to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()