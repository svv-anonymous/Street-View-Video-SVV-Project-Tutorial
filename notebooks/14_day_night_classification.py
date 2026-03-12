#!/usr/bin/env python3
"""
Lighting Screening for SVV Dataset.

This script classifies lighting conditions for each series based on extracted image frames and separates them into day and night groups.

Pipeline:
1. Read removed video list produced by the quality screening step and skip those series.
2. Run lighting classification on series folders under data/image and aggregate results.
3. Export two lists:
   - day_list.txt
   - night_list.txt

"""

import shutil
import tempfile
from pathlib import Path
from collections import Counter

import pandas as pd
from zensvi.cv import ClassifierLighting



PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PARENT = PROJECT_ROOT / "data" / "image"

REMOVED_LIST = PROJECT_ROOT / "removed_videos.txt"

OUTPUT_DAY = PROJECT_ROOT / "day_list.txt"
OUTPUT_NIGHT = PROJECT_ROOT / "night_list.txt"


def read_removed_list(path: Path):
    """Load removed video list."""
    if not path.exists():
        return set()

    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def find_label_column(df: pd.DataFrame):
    """Find the lighting label column from classifier output."""
    candidates = [
        "label",
        "pred_label",
        "prediction",
        "class",
        "lighting",
        "lighting_label"
    ]

    lower_map = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c in lower_map:
            return lower_map[c]

    # fallback: first object column
    for c in df.columns:
        if df[c].dtype == "object":
            return c

    return None


def normalize_label(x):
    """Normalize lighting label."""
    if pd.isna(x):
        return None
    return str(x).strip().lower()


def map_day_night(label):
    """Map raw label to day/night group."""
    if label == "day":
        return "day"

    if label in {"night", "dawn/dusk", "dawn", "dusk"}:
        return "night"

    return None


def mode_label(labels):
    """Return most frequent label."""
    labels = [x for x in labels if x is not None]

    if not labels:
        return None

    counter = Counter(labels)
    return counter.most_common(1)[0][0]


def main():

    removed_videos = read_removed_list(REMOVED_LIST)
    print(f"Removed videos loaded: {len(removed_videos)}")

    classifier = ClassifierLighting()

    day_list = []
    night_list = []

    if not IMAGE_PARENT.exists():
        print(f"Image parent not found: {IMAGE_PARENT}")
        return

    for series_folder in IMAGE_PARENT.iterdir():

        if not series_folder.is_dir():
            continue

        series_name = series_folder.name

        if series_name in removed_videos:
            print(f"Skip removed: {series_name}")
            continue

        image_dir = series_folder / "image"

        if not image_dir.exists():
            print(f"Skip (no image): {series_name}")
            continue

        print(f"Processing: {series_name}")

        tmp_dir = Path(tempfile.mkdtemp())

        try:

            classifier.classify(
                str(image_dir),
                dir_summary_output=str(tmp_dir),
            )

            csv_files = list(tmp_dir.glob("*.csv"))

            if not csv_files:
                print(f"No result csv: {series_name}")
                continue

            df = pd.read_csv(csv_files[0])

            label_col = find_label_column(df)

            if label_col is None:
                print(f"Cannot find label column: {series_name}")
                continue

            labels = [normalize_label(x) for x in df[label_col].tolist()]

            final_label = mode_label(labels)

            group = map_day_night(final_label)

            if group == "day":
                day_list.append(series_name)

            elif group == "night":
                night_list.append(series_name)

        except Exception as e:

            print(f"Error in {series_name}: {e}")

        finally:

            shutil.rmtree(tmp_dir, ignore_errors=True)


    with open(OUTPUT_DAY, "w") as f:
        for name in sorted(day_list):
            f.write(name + "\n")

    with open(OUTPUT_NIGHT, "w") as f:
        for name in sorted(night_list):
            f.write(name + "\n")


    print("\nLighting screening complete.")
    print(f"Day videos: {len(day_list)}")
    print(f"Night videos: {len(night_list)}")
    print(f"Saved to:")
    print(f"  {OUTPUT_DAY}")
    print(f"  {OUTPUT_NIGHT}")


if __name__ == "__main__":
    main()