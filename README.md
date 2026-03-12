## Street View Video (SVV) Project Tutorial

This repository contains a 15-step pipeline for downloading, geolocating walking videos and extracting multimodal information. 

### Folder layout

- **`notebooks/`**: Main tutorial steps `01`–`13`.
- **`data/`**:
  - `videos/`: Raw videos and extracted audio/video splits.
  - `image/`: Extracted frames, depth, OCR, and map-matching results.
  - `results/`: Final per-link CSV outputs.
- **`geo-files/`**: Prepared OSM road network and POI geo-files.
- **`geo-files-processed/`**: Cleaned networks and route validation reports.
- **`geo-files-output/`**: Final network shapefile with attached indicators.
- **`models/`**: External model code and weights (Depth Anything, PANNs, etc.).
- **`envs/`**: Conda environment YAML files for each processing stage.


You can adapt the folder names, but then update the corresponding `Path` definitions at the top of each step.

### Pipeline overview (steps 1–13)

- **01 – Road network preprocessing (`01_road_network_preprocessing.py`)**
  - Clean and simplify the OSM road network, export an unsplit GraphML network and a 50 m split version (GraphML + SHP).

- **02 – Video preprocessing (`02_video_preprocessing.ipynb`)**
  - Use `yt-dlp` command-line tool to search and download walking videos.
  - Split raw videos into frames for geo-localizaiton.

- **03 – Depth estimation (`03_depth_estimation.py`)**
  - Run Depth Anything V2 on frames to generate per-frame depth maps (`.npy`) under `depth/` subfolder.

- **04 – OCR recognition (`04_ocr_recognition.py`)**
  - Run PaddleOCR on all image frames and save JSON results under each series’ `ocr/` subfolder.

- **05 – OCR + depth combination (`05_ocr_depth_combination.py`)**
  - Match OCR bounding boxes with depth maps.

- **06 – Geo-locate (`06_geo-locate.py`)**
  - Fuzzy-match OCR text to POIs, filter spatially, and infer observer locations along the road network, and export per-series `location/*.csv` with inferred positions.

- **07 – Map matching (`07_map-matching.py`)**
  - Map-match the inferred locations to the 50 m link network using GoTrackIt, and save results under `map-matching/`.

- **08 – Link time mapping (`08_link_time_mapping.py`)**
  - Interpolate positions along the matched route.

- **09 – Video postprocessing (`09_video_postprocessing.py`)**
  - Split original videos into 10-second clips (`video_split/`), and extract audio tracks (`audio_split/`) aligned with the same time indices.

- **10 – Route validation (`10_route_validation.py`)**
  - Compare map-matching results with ground truth timestamps from video descriptions using LCS method.

- **11 – Action recognition (`11_action_recognition.py`)**
  - Run YOLO-based tracking plus a video classifier over the video clips.

- **12 – Audio tagging (`12_audio_tagging.py`)**
  - Tag audio clips with PANNs.

- **13 – Quality screening (`13_quality_screening.py`)**
  - Remove low-quality videos based on:  trajectory anomaly filtering, and excessive speech or music probability

- **14 – Lighting screening (`14_lighting_screening.py`)**
  - Classify lighting conditions of videos and split them into: day_list.txt and night_list.txt

- **15 – Aggregate (`15_aggregate.py`)**
  - Aggregate audio tags and pedestrian counts per `link_id`, at overall, daytime, and nighttime level.
  - Join aggregated CSVs back to the 50 m edge shapefile and export a final network with attributes.


## Recommended environments

To reduce setup overhead, you can use the following compact 5-environment setup.
This still keeps heavy dependencies separated while covering all 15 steps.

- **Env A - `svv-geo` (network + geolocation + map matching)**  
  - **Files**: `01_road_network_preprocessing.py`, `06_geo-locate.py`, `07_map-matching.py`, `08_link_time_mapping.py`, `10_route_validation.py`  
  - **Env file**: `envs/environment-svv-geo.yml`

```bash
conda env create -f envs/environment-svv-geo.yml
conda activate svv-geo
```

- **Env B - `svv-vision` (depth + OCR + fusion)**  
  - **Files**: `03_depth_estimation.py`, `04_ocr_recognition.py`, `05_ocr_depth_combination.py`  
  - **Env file**: `envs/environment-svv-vision.yml`

```bash
conda env create -f envs/environment-svv-vision.yml
conda activate svv-vision
```

- **Env C - `svv-video-audio` (video I/O + audio tagging)**  
  - **Files**: `02_video_preprocessing.ipynb`, `09_video_postprocessing.py`, `12_audio_tagging.py`  
  - **Env file**: `envs/environment-svv-video-audio.yml`

```bash
conda env create -f envs/environment-svv-video-audio.yml
conda activate svv-video-audio
```

- **Env D - `svv-action` (action + lighting)**  
  - **Files**: `11_action_recognition.py`, `14_lighting_screening.py`  
  - **Env file**: `envs/environment-svv-action.yml`

```bash
conda env create -f envs/environment-svv-action.yml
conda activate svv-action
```

- **Env E - `svv-agg` (screening + final aggregation)**  
  - **Files**: `13_quality_screening.py`, `15_aggregate.py`  
  - **Env file**: `envs/environment-svv-agg.yml`

```bash
conda env create -f envs/environment-svv-agg.yml
conda activate svv-agg
```





