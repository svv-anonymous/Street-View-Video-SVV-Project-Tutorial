#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCR recognition for image frames using PaddleOCR.

Notes:
- Each video corresponds to one folder under `data/image/{series_name}/`
- Input images are expected under `{series_name}/image/`
- OCR results are saved as JSON files under `{series_name}/ocr/`
- Each image produces exactly one `*_res.json` file
- The script supports safe re-running:
    - If all images already have OCR results, the folder is skipped
    - If partially processed, only missing images are re-processed

Expected Structure:
- Input: data/image/{series}/image/*.png (or .jpg, .jpeg)
- Output: data/image/{series}/ocr/*_res.json
"""

import os
from pathlib import Path

from paddleocr import PaddleOCR
from tqdm import tqdm


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    IMAGE_ROOT = PROJECT_ROOT / "data" / "image"

    if not IMAGE_ROOT.exists():
        print(f"Error: Image root folder not found at {IMAGE_ROOT}")
        return

    # --- PaddleOCR Initialization ---
    # use_doc_orientation_classify=False: Frames are assumed upright
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",)

    # Discover all series folders
    subfolders = [p for p in IMAGE_ROOT.iterdir() if p.is_dir()]

    if not subfolders:
        print(f"Warning: No subfolders found under {IMAGE_ROOT}")
        return

    for folder in subfolders:
        series_name = folder.name
        image_dir = folder / "image"
        output_dir = folder / "ocr"

        if not image_dir.exists():
            print(f"Skipping {series_name}: No 'image' folder found.")
            continue

        # Collect supported image formats
        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_files:
            print(f"Skipping {series_name}: No images found.")
            continue

        # Resume logic: Skip complete folders, process missing parts
        if output_dir.exists():
            output_files = list(output_dir.glob("*_res.json"))

            if len(output_files) == len(image_files):
                print(f"Skipping {series_name}: OCR already complete.")
                continue
            else:
                print(f"Resuming {series_name}: {len(output_files)}/{len(image_files)} complete.")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing series: {series_name} ({len(image_files)} images)")

        for fname in tqdm(image_files, desc=f"  ↳ {series_name}"):
            img_path = image_dir / fname

            # Output JSON name: <image_stem>_res.json
            json_name = Path(fname).stem + "_res.json"
            output_path = output_dir / json_name

            # Skip already processed images
            if output_path.exists():
                continue

            try:
                # Run OCR inference
                result = ocr.predict(input=str(img_path))

                # Save results into one JSON file per image
                for res in result:
                    res.save_to_json(str(output_path))

            except Exception as e:
                print(f"Failed to process {fname} in {series_name}: {e}")

    print("\nOCR processing complete.")


if __name__ == "__main__":
    main()
