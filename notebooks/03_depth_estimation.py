#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Depth estimation for images using Depth Anything V2 (Metric Depth).

This script processes all image frames extracted from videos and generates metric depth maps for each frame. Depth maps are saved as NumPy arrays (.npy files) for later use in geolocation and OCR-depth combination steps.

Expected Structure:
- Input:
- models/Depth-Anything-V2/metric_depth/
  - depth_anything_v2/...
  - checkpoints/
- data/image/{series_name}/image/*.png

- Output:
- data/image/{series_name}/depth/*.npy

Notes:
- Official metric checkpoints are hosted on Hugging Face.
- If checkpoint is missing locally, the script can auto-download it if AUTO_DOWNLOAD=1.

Preparation:
# ==============================================================================
# Please ensure the model repository is cloned before running:
# $ cd <PROJECT_ROOT>/models
# $ git clone https://github.com/DepthAnything/Depth-Anything-V2.git 
# ==============================================================================
"""

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the Depth-Anything-V2 repo is placed inside your project
MODEL_REPO_DIR = PROJECT_ROOT / "models" / "Depth-Anything-V2"
METRIC_DIR = MODEL_REPO_DIR / "metric_depth"

IMAGE_ROOT = PROJECT_ROOT / "data" / "image"

# Model configuration
ENCODER = "vitl"     # 'vits', 'vitb', 'vitl'
DATASET = "vkitti"   # metric calibration dataset
MAX_DEPTH = 80       # meters, as suggested for VKITTI
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

# Official checkpoint URLs (HF)
# You can extend this dict if later you want other encoder/dataset combos.
CKPT_URLS = {
    ("vkitti", "vitl"): "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth?download=true",
}

# If set to 1, missing checkpoint will be auto-downloaded
AUTO_DOWNLOAD = True

# Optional: allow overriding checkpoint path explicitly
OVERRIDE_CKPT = os.environ.get("DEPTH_ANYTHING_CKPT", "").strip()


def add_metric_repo_to_syspath(metric_dir: Path):
    """
    Ensure we can import depth_anything_v2 from the metric_depth folder
    without requiring pip install.
    """
    if not metric_dir.exists():
        raise FileNotFoundError(f"Metric directory not found: {metric_dir}")
    sys.path.insert(0, str(metric_dir))


def get_checkpoint_path(metric_dir: Path, dataset: str, encoder: str) -> Path:
    """
    Return the intended checkpoint path inside metric_depth/checkpoints/.
    """
    ckpt_dir = metric_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"depth_anything_v2_metric_{dataset}_{encoder}.pth"
    return ckpt_dir / ckpt_name


def ensure_checkpoint(ckpt_path: Path, dataset: str, encoder: str):
    """
    Ensure checkpoint exists; optionally download.
    """
    if ckpt_path.exists():
        return

    url = CKPT_URLS.get((dataset, encoder))
    msg = (
        f"Model checkpoint not found:\n  {ckpt_path}\n\n"
        "Official metric checkpoints are hosted on Hugging Face.\n"
    )

    if not url:
        raise FileNotFoundError(
            msg + f"No download URL configured for (dataset={dataset}, encoder={encoder}).\n"
                  f"Please download the correct checkpoint manually and place it at:\n  {ckpt_path}\n"
        )

    if not AUTO_DOWNLOAD:
        raise FileNotFoundError(
            msg
            + f"Download from:\n  {url}\n\n"
              "Then place it into the path above.\n"
              "Or enable auto-download by running:\n"
              " AUTO_DOWNLOAD=1 python your_script.py\n"
        )

    print(msg)
    print(f"AUTO_DOWNLOAD=1 -> downloading checkpoint to:\n  {ckpt_path}")
    try:
        urlretrieve(url, ckpt_path)
        print("Download complete.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download checkpoint from:\n  {url}\nError: {e}"
        ) from e


def main():
    add_metric_repo_to_syspath(METRIC_DIR)
    from depth_anything_v2.dpt import DepthAnythingV2

    # Resolve checkpoint
    if OVERRIDE_CKPT:
        ckpt_path = Path(OVERRIDE_CKPT).expanduser().resolve()
    else:
        ckpt_path = get_checkpoint_path(METRIC_DIR, DATASET, ENCODER)
        ensure_checkpoint(ckpt_path, DATASET, ENCODER)

    # Initialize model
    print(f"Loading Depth Anything V2: encoder={ENCODER}, dataset={DATASET}")
    model = DepthAnythingV2(**{**MODEL_CONFIGS[ENCODER], "max_depth": MAX_DEPTH})
    model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE))
    model.to(DEVICE).eval()

    if not IMAGE_ROOT.exists():
        print(f"Error: {IMAGE_ROOT} not found.")
        return

    # Process series folders
    subfolders = sorted([p for p in IMAGE_ROOT.iterdir() if p.is_dir()])
    
    for series_dir in subfolders:
        image_dir = series_dir / "image"
        output_dir = series_dir / "depth"

        if not image_dir.exists():
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        image_files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() == ".png"])

        if not image_files:
            continue

        print(f"\nProcessing series: {series_dir.name} ({len(image_files)} frames)")

        for img_path in tqdm(image_files, desc=f"  ↳ {series_dir.name}"):
            raw_img = cv2.imread(str(img_path))
            if raw_img is None:
                continue

            try:
                depth = model.infer_image(raw_img)
                out_path = output_dir / (img_path.stem + ".npy")
                np.save(str(out_path), depth)
            except Exception as e:
                print(f"Error in {series_dir.name}/{img_path.name}: {e}")

    print("\nDepth estimation complete.")


if __name__ == "__main__":
    main()
