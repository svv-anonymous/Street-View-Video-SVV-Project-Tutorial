#!/usr/bin/env python3
"""
Audio tagging for video clip audio tracks using PANNs.

Processes audio files extracted from video clips and predicts audio tags (e.g., speech, music, traffic sounds) using PANNs (CNN14) model.
Also computes RMS dB level. Results are saved as CSV files.

Expected Structure:
- Input: data/videos/{series}/audio_split/*.wav
- Output: data/image/{series}/audio_result/*.csv
"""
import os
import glob
from pathlib import Path
import numpy as np
import librosa
import pandas as pd
import torch
from panns_inference import AudioTagging, labels


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PARENT = PROJECT_ROOT / "data" / "videos"
RESULT_PARENT = PROJECT_ROOT / "data" / "image"

# Audio parameters
MIN_DURATION_SEC = 1.0
TOP_K = 10

def compute_rms_db(audio_signal):
    """Compute RMS (Root Mean Square) in decibels from audio signal."""
    rms = np.sqrt(np.mean(audio_signal**2))
    return 20 * np.log10(rms + 1e-9)


def process_audio_clip(wav_path, at_model, top_k=10, min_duration=1.0):
    """
    Process a single audio clip: compute RMS dB and predict audio tags.
    
    Args:
        wav_path: Path to WAV audio file
        at_model: AudioTagging model instance
        top_k: Number of top tags to return (None for all)
        min_duration: Minimum clip duration in seconds to process
        
    Returns:
        DataFrame with 'tag' and 'value' columns, or None if skipped
    """
    try:
        # Load audio
        y, sr = librosa.load(wav_path, sr=32000, mono=True)

        # Skip clips that are too short
        if len(y) < sr * min_duration:
            return None

        # Compute RMS dB
        db = compute_rms_db(y)

        # Run audio tagging model
        audio_batch = y[None, :]
        clipwise_output, _ = at_model.inference(audio_batch)
        scores = clipwise_output[0]

        # Select top_k tags
        idxs = np.argsort(scores)[::-1]
        if top_k is not None:
            idxs = idxs[:top_k]

        # Build result DataFrame
        rows = [('total', round(db, 3))]
        for i in idxs:
            rows.append((labels[i], round(float(scores[i]), 3)))
        
        return pd.DataFrame(rows, columns=['tag', 'value'])
    
    except RuntimeError as e:
        print(f" Inference failed for {wav_path}: {e}")
        return None


def main():
    print("Initializing PANNs Audio Tagging model")
    at = AudioTagging(checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu')

    if not VIDEO_PARENT.exists():
        print(f"Error: {VIDEO_PARENT} not found.")
        return

    for series_folder in VIDEO_PARENT.iterdir():
        if not series_folder.is_dir():
            continue

        series_name = series_folder.name
        audio_dir = series_folder / "audio_split" 
        
        if not audio_dir.is_dir():
            continue

        out_dir = RESULT_PARENT / series_name / "audio_result"
        
        wav_paths = list(audio_dir.glob("*.wav"))
        if not wav_paths:
            continue

        existing_csvs = list(out_dir.glob("*.csv")) if out_dir.exists() else []
        if len(wav_paths) == len(existing_csvs) and len(wav_paths) > 0:
            print(f"{series_name}: Audio tagging already complete.")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing audio for: {series_name}")

        for wav_path in wav_paths:
            csv_path = out_dir / f"{wav_path.stem}_audio.csv"
            if csv_path.exists():
                continue

            df_out = process_audio_clip(wav_path, at, top_k=TOP_K, min_duration=MIN_DURATION_SEC)
            
            if df_out is not None:
                df_out.to_csv(csv_path, index=False)
                
    print("\nAudio tagging task complete.")


if __name__ == "__main__":
    main()
