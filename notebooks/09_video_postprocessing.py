#!/usr/bin/env python3
"""
Video postprocessing: split videos into clips and extract audio.

Synchronizes the original video files with the geolocated time sequences. 
It clips the videos into 10-second segments corresponding to each geolocated point and extracts their audio tracks for further multi-modal analysis.

1. Sync: Read geolocation CSVs to identify valid time steps.
2. Clip: Use FFmpeg to cut 10s segments starting from the calibrated offset.
3. Audio: Extract high-quality WAV (PCM 16-bit) audio from each clip.

Expected Structure:
- Input: data/image/{series}/location/*.csv
- Input: data/videos/{series}/*.mp4
- Output: data/videos/{series}/video_split/*.mp4
- Output: data/videos/{series}/audio_split/*.wav

Preparation:
- ffmpeg must be installed and available in PATH
"""

import os
import glob
import subprocess
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PARENT = PROJECT_ROOT / 'data' / 'image'
VIDEO_PARENT = PROJECT_ROOT / 'data' / 'videos'

# Video parameters
BASE_OFFSET = 95.0   # Start offset: time=1 maps to 95s
INTERVAL = 10.0      # Duration per clip
HEIGHT = 720
USE_GPU = False
VCODEC = 'h264_nvenc' if USE_GPU else 'libx264'

# Audio parameters
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2
AUDIO_CODEC = 'pcm_s16le'

def split_videos_into_clips():
    """Extract continuous 10s clips from t_min to t_max."""
    print("Step 1: Splitting videos into continuous clips")
    
    series_folders = [p for p in CSV_PARENT.iterdir() if p.is_dir()]
    
    for folder in series_folders:
        series_name = folder.name
        loc_dir = folder / 'location'
        
        loc_files = list(loc_dir.glob('*.csv'))
        if not loc_files:
            continue
            
        video_dir = VIDEO_PARENT / series_name
        mp4_files = list(video_dir.glob('*.mp4'))
        if not mp4_files:
            print(f" Original video not found for {series_name}, skipping.")
            continue
            
        video_file = mp4_files[0]
        output_dir = video_dir / 'video_split'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine time range from CSV
        df = pd.read_csv(loc_files[0])
        t_min = int(df['time'].min())
        t_max = int(df['time'].max())
        
        print(f" Processing '{series_name}': Range t={t_min} to t={t_max}")
        
        for t in tqdm(range(t_min, t_max + 1), desc=f"    ↳ clipping", leave=False):
            # Formula: start = offset + (t-1) * interval
            start_sec = BASE_OFFSET + (t - 1) * INTERVAL
            output_file = output_dir / f"clip_{t:04d}.mp4"
            
            if output_file.exists():
                continue
                
            cmd = [
                'ffmpeg',
                '-ss', str(start_sec),
                '-i', str(video_file),
                '-t', str(INTERVAL),
                '-vf', f'scale=-2:{HEIGHT}',
                '-c:v', VCODEC,
                '-crf', '28',
                '-preset', 'fast',
                '-threads', '0',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                '-y', str(output_file)
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extract_audio_from_clips():
    """Extract WAV files and verify sync with csv_split directory."""
    print("\nStep 2: Extracting audio tracks (Sync check with csv_split)")
    
    for folder in VIDEO_PARENT.iterdir():
        if not folder.is_dir(): continue
        
        video_split_dir = folder / 'video_split'
        audio_split_dir = folder / 'audio_split'
        csv_split_dir = folder / 'csv_split'

        if not video_split_dir.exists():
            continue

        mp4_files = sorted(list(video_split_dir.glob('*.mp4')))
        if not mp4_files:
            continue

        # Sync Validation: check if wav count matches csv count
        wav_files = list(audio_split_dir.glob('*.wav')) if audio_split_dir.exists() else []
        csv_files = list(csv_split_dir.glob('*.csv')) if csv_split_dir.exists() else []

        if wav_files and csv_files and len(wav_files) == len(csv_files):
            print(f" [SKIP] '{folder.name}': Already synced ({len(wav_files)} files).")
            continue

        # Reset audio folder if sync fails
        if audio_split_dir.exists():
            for f in audio_split_dir.glob('*.wav'): f.unlink()
        else:
            audio_split_dir.mkdir(parents=True, exist_ok=True)

        for video_file in tqdm(mp4_files, desc=f"    ↳ {folder.name} extracting"):
            audio_file = audio_split_dir / f"{video_file.stem}.wav"
            
            cmd = [
                'ffmpeg', '-i', str(video_file),
                '-vn', '-acodec', AUDIO_CODEC,
                '-ar', str(AUDIO_SAMPLE_RATE), 
                '-ac', str(AUDIO_CHANNELS), 
                '-y', str(audio_file)
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                print(f" [ERR] Failed: {video_file.name}")

def main():
    """Execute pipeline."""
    print("=" * 60)
    print("Video Postprocessing: Range-based Clipping & Audio Sync")
    print("=" * 60)
    
    split_videos_into_clips()
    extract_audio_from_clips()
    
    print("\nAll tasks completed successfully.")

if __name__ == "__main__":
    main()

