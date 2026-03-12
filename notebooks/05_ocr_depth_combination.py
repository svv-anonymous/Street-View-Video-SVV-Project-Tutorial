#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combine OCR text recognition with depth estimation results.

For each detected text region, computes average depth from the corresponding depth map region and saves combined results as CSV files.

- Input: data/image/{series}/ocr/*_res.json & data/image/{series}/depth/*.npy
- Output: data/image/{series}/ocr/*_depth_results.csv

"""

import os
from pathlib import Path
import numpy as np
import json
import csv
from tqdm import tqdm


def compute_avg_depth_for_text(depth_map, bbox):
    """
    Compute average depth for a text bounding box region.
    
    Args:
        depth_map: 2D numpy array of depth values
        bbox: [x0, y0, x1, y1] bounding box coordinates
        
    Returns:
        Average depth value or None if no valid depths found
    """
    height, width = depth_map.shape
    x0, y0, x1, y1 = map(int, bbox)
    x0, x1 = np.clip([x0, x1], 0, width - 1)
    y0, y1 = np.clip([y0, y1], 0, height - 1)

    region = depth_map[y0:y1, x0:x1]
    valid_depths = region[np.isfinite(region) & (region > 0)]
    return float(np.mean(valid_depths)) if valid_depths.size > 0 else None




def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    IMAGE_ROOT = PROJECT_ROOT / "data" / "image"

    if not IMAGE_ROOT.exists():
        print(f"Error: {IMAGE_ROOT} not found.")
        return

    # Iterating through video series folders
    for subdir in IMAGE_ROOT.iterdir():
        if not subdir.is_dir():
            continue

        series_name = subdir.name
        ocr_dir = subdir / "ocr"
        depth_dir = subdir / "depth"

        if not ocr_dir.exists() or not depth_dir.exists():
            print(f"{series_name}: OCR or Depth folder missing -> skipped")
            continue

        ocr_files = list(ocr_dir.glob("*_res.json"))
        if not ocr_files:
            print(f"{series_name}: no OCR results found -> skipped")
            continue
        
        print(f"\nProcessing series: {series_name}")
        
        for ocr_path in tqdm(ocr_files, desc=f"  ↳ {series_name}"):
            stem = ocr_path.name.replace('_res.json', '')
            depth_path = depth_dir / f"{stem}.npy"
            csv_path = ocr_dir / f"{stem}_depth_results.csv"

            if csv_path.exists():
                continue  

            if not depth_path.exists():
                continue

            try:
                # Load OCR data
                with open(ocr_path, "r", encoding="utf-8") as f:
                    ocr_data = json.load(f)
                texts = ocr_data.get("rec_texts", [])
                boxes = ocr_data.get("rec_boxes", [])
                
                # Load Depth map
                depth_map = np.load(depth_path)
                height, width = depth_map.shape
            except Exception as e:
                print(f"Error loading {stem}: {e}")
                continue

            # Process individual text regions
            combined_results = []
            for txt, box in zip(texts, boxes):
                avg_depth = compute_avg_depth_for_text(depth_map, box)
                x0, y0, x1, y1 = map(int, box)
                
                combined_results.append({
                    "text": txt,
                    "avg_depth": avg_depth,
                    "bbox": [x0, y0, x1, y1]
                })

            # Save combined results for Geolocation stage
            try:
                with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["text", "avg_depth", "bbox"])
                    for item in combined_results:
                        writer.writerow([item["text"], item["avg_depth"], item["bbox"]])
            except Exception as e:
                print(f"Failed to write CSV {csv_path.name}: {e}")


if __name__ == "__main__":
    main()
