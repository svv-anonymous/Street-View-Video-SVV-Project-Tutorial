#!/usr/bin/env python3
"""
Action recognition for video clips using YOLO tracking and video classification.

Detects and tracks pedestrians in video clips, then classifies their actions (walking, running, standing, sitting) using video classification models.
Results are aggregated per person and saved as CSV files.

Expected Structure:
- Input: data/videos/{series}/video_split/*.mp4 (from 09_video_postprocessing)
- Output: data/image/{series}/video_result/*.csv

Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
import argparse
import os
import glob
from collections import defaultdict
from pathlib import Path
from typing import List, Union, Tuple
import cv2
import numpy as np
import torch
import pandas as pd
from numpy.linalg import LinAlgError
from transformers import AutoModel, AutoProcessor

from ultralytics import YOLO
from ultralytics.data.loaders import get_best_youtube_url
from ultralytics.utils.plotting import Annotator
from ultralytics.utils.torch_utils import select_device


def crop_and_pad(frame: np.ndarray, box: List[float], margin_percent: int) -> np.ndarray:
    """Crop bounding box from frame with margin, pad to square, and resize to 224x224."""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * margin_percent / 100), int(h * margin_percent / 100)
    x1, y1 = x1 - mx, y1 - my
    x2, y2 = x2 + mx, y2 + my

    # boundary check
    y1c, y2c = max(0, y1), min(frame.shape[0], y2)
    x1c, x2c = max(0, x1), min(frame.shape[1], x2)
    crop = frame[y1c:y2c, x1c:x2c]

    # if empty, return black image
    if crop.size == 0:
        return np.zeros((224, 224, 3), dtype=frame.dtype)

    # pad to square
    ch, cw = crop.shape[:2]
    size = max(ch, cw)
    square = np.zeros((size, size, 3), dtype=frame.dtype)
    dy, dx = (size - ch) // 2, (size - cw) // 2
    square[dy:dy+ch, dx:dx+cw] = crop

    # resize
    return cv2.resize(square, (224, 224), interpolation=cv2.INTER_LINEAR)


class TorchVisionVideoClassifier:
    """Video classifier using TorchVision pre-trained models."""
    from torchvision.models.video import (
        MViT_V1_B_Weights, MViT_V2_S_Weights,
        R3D_18_Weights, S3D_Weights,
        Swin3D_B_Weights, Swin3D_T_Weights,
        mvit_v1_b, mvit_v2_s,
        r3d_18, s3d,
        swin3d_b, swin3d_t,
    )
    model_name_to_model_and_weights = {
        "s3d": (s3d, S3D_Weights.DEFAULT),
        "r3d_18": (r3d_18, R3D_18_Weights.DEFAULT),
        "swin3d_t": (swin3d_t, Swin3D_T_Weights.DEFAULT),
        "swin3d_b": (swin3d_b, Swin3D_B_Weights.DEFAULT),
        "mvit_v1_b": (mvit_v1_b, MViT_V1_B_Weights.DEFAULT),
        "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights.DEFAULT),
    }

    def __init__(self, model_name: str, device: Union[str, torch.device] = ""):
        """Initialize TorchVision video classifier model."""
        if model_name not in self.model_name_to_model_and_weights:
            raise ValueError(f"Invalid model name '{model_name}'")
        fn, self.weights = self.model_name_to_model_and_weights[model_name]
        self.device = select_device(device)
        self.model = fn(weights=self.weights).to(self.device).eval()

    @staticmethod
    def available_model_names() -> List[str]:
        """Return list of available model names."""
        return list(TorchVisionVideoClassifier.model_name_to_model_and_weights.keys())

    def preprocess(self, crops: List[np.ndarray]) -> torch.Tensor:
        """Preprocess image crops for video classification."""
        from torchvision.transforms import v2
        ts = self.weights.transforms()
        transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize([224, 224], antialias=True),
            v2.Normalize(mean=ts.mean, std=ts.std),
        ])
        proc = [transform(torch.from_numpy(c).permute(2, 0, 1)) for c in crops]
        # shape (1, T, C, H, W)
        return torch.stack(proc).unsqueeze(0).permute(0, 2, 1, 3, 4).to(self.device)

    def __call__(self, seq: torch.Tensor) -> torch.Tensor:
        """Run inference on video sequence."""
        with torch.inference_mode():
            return self.model(seq)

    def postprocess(self, outputs: torch.Tensor) -> Tuple[List[str], List[float]]:
        """Postprocess model outputs to get action labels and confidence scores."""
        labels, confs = [], []
        for out in outputs:
            idx = out.argmax().item()
            labels.append(self.weights.meta["categories"][idx])
            confs.append(out.softmax(0)[idx].item())
        return labels, confs


class HuggingFaceVideoClassifier:
    """Video classifier using HuggingFace transformer models."""
    
    def __init__(self, labels: List[str], model_name: str, device: Union[str, torch.device], fp16: bool):
        """Initialize HuggingFace video classifier model."""
        self.fp16 = fp16
        self.labels = labels
        self.device = select_device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        m = AutoModel.from_pretrained(model_name).to(self.device)
        self.model = m.half() if fp16 else m.eval()

    def preprocess(self, crops: List[np.ndarray]) -> torch.Tensor:
        """Preprocess image crops for video classification."""
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.float()/255.0),
            transforms.Resize([224,224]),
            transforms.Normalize(mean=self.processor.image_processor.image_mean,
                                 std=self.processor.image_processor.image_std),
        ])
        proc = [transform(torch.from_numpy(c).permute(2,0,1)) for c in crops]
        b = torch.stack(proc).unsqueeze(0).to(self.device)
        return b.half() if self.fp16 else b

    def __call__(self, seq: torch.Tensor) -> torch.Tensor:
        """Run inference on video sequence with text labels."""
        ids = self.processor(text=self.labels, return_tensors="pt", padding=True)["input_ids"].to(self.device)
        with torch.inference_mode():
            out = self.model(pixel_values=seq, input_ids=ids)
        return out.logits_per_video

    def postprocess(self, outputs: torch.Tensor) -> Tuple[List[List[str]], List[List[float]]]:
        """Postprocess model outputs to get action labels and confidence scores."""
        probs = outputs.softmax(-1)
        all_labels, all_confs = [], []
        for p in probs:
            topk = p.topk(1)
            all_labels.append([self.labels[i] for i in topk.indices.tolist()])
            all_confs.append(topk.values.tolist())
        return all_labels, all_confs


def process_mp4(
    mp4_path: str,
    csv_path: str,
    yolo_model,
    video_classifier,
    labels: List[str],
    crop_margin: int,
    seq_len: int,
    skip_frame: int,
    overlap: float
):
    """
    Process a single video clip: track pedestrians and classify their actions.
    
    Uses YOLO to track people across frames, collects sequences of cropped images,
    and classifies actions using a video classifier. Aggregates results per person
    and saves to CSV.
    """
    cap = cv2.VideoCapture(mp4_path)
    per_track = defaultdict(lambda: defaultdict(float))
    history = defaultdict(list)
    counter = 0
    to_inf_ids, to_inf_crops = [], []
    pred_lbls, pred_confs = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        counter += 1

        # Catch LinAlgError from the tracker and skip the current frame
        try:
            res = yolo_model.track(
                frame, persist=True, classes=[0],
                tracker="botsort.yaml", conf=0.3, iou=0.5
            )
        except LinAlgError:
            # If tracking fails, skip this frame and continue
            continue

        if res[0].boxes.is_track:
            bxs = res[0].boxes.xyxy.cpu().numpy()
            tids = res[0].boxes.id.cpu().numpy()

            if counter % skip_frame == 0:
                to_inf_ids.clear()
                to_inf_crops.clear()

            for bx, tid in zip(bxs, tids):
                if counter % skip_frame == 0:
                    history[tid].append(crop_and_pad(frame, bx, crop_margin))
                if len(history[tid]) > seq_len:
                    history[tid].pop(0)
                if len(history[tid]) == seq_len and counter % skip_frame == 0:
                    to_inf_crops.append(video_classifier.preprocess(history[tid]))
                    to_inf_ids.append(tid)

            if to_inf_crops and (
                not pred_lbls or
                counter % int(seq_len * skip_frame * (1 - overlap)) == 0
            ):
                batch = torch.cat(to_inf_crops, dim=0)
                out = video_classifier(batch)
                pred_lbls, pred_confs = video_classifier.postprocess(out)

            if to_inf_ids and to_inf_crops:
                for bx, tid, lbl_list, conf_list in zip(
                    bxs, to_inf_ids, pred_lbls, pred_confs
                ):
                    label = lbl_list[0]
                    conf  = conf_list[0]
                    per_track[int(tid)][label] += conf

    cap.release()

    # Aggregate per-track results and save to CSV
    total = len(per_track)
    counts = defaultdict(int)
    for scores in per_track.values():
        dom = max(scores.items(), key=lambda x: x[1])[0]
        counts[dom] += 1

    rows = [("total_unique_people", total)]
    for lbl in labels:
        rows.append((lbl, counts.get(lbl, 0)))

    df = pd.DataFrame(rows, columns=["action", "count"])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def main():   
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    VIDEO_PARENT = PROJECT_ROOT / "data" / "videos"
    RESULT_PARENT = PROJECT_ROOT / "data" / "image"

    weights = "yolo11n.pt"
    device_arg = ""             
    crop_margin = 10            
    seq_len = 8                 
    skip_frame = 2              
    overlap = 0.25              
    fp16 = False                
    classifier_model = "microsoft/xclip-base-patch32" 
    labels = ["walking", "standing", "sitting"]

    device = select_device(device_arg)
    yolo = YOLO(weights).to(device)
    

    if "clf" not in locals(): 
        clf = HuggingFaceVideoClassifier(labels, classifier_model, device, fp16)

    if not VIDEO_PARENT.exists():
        print(f"Error: {VIDEO_PARENT} not found.")
        return

    for series_folder in VIDEO_PARENT.iterdir():
        if not series_folder.is_dir():
            continue

        video_name = series_folder.name
        vd_dir = series_folder / "video_split" 
        
        if not vd_dir.is_dir():
            continue
            
        mp4s = list(vd_dir.glob("*.mp4"))
        if not mp4s:
            continue

        out_dir = RESULT_PARENT / video_name / "video_result"
        
        csvs = list(out_dir.glob("*.csv")) if out_dir.exists() else []
        if len(mp4s) == len(csvs) and len(mp4s) > 0:
            print(f"[SKIP] {video_name}: already complete")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        for mp4_path in mp4s:
            base = mp4_path.stem
            csv_path = out_dir / f"{base}.csv"
            
            print(f"Processing: {video_name}/{base}.mp4")
            
            process_mp4(
                str(mp4_path), 
                str(csv_path), 
                yolo, 
                clf, 
                labels,
                crop_margin, 
                seq_len, 
                skip_frame, 
                overlap
            )

    print("Pipeline finished.")


if __name__ == "__main__":
    main()
