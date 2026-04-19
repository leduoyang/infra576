# Video Segmentation Project

A simplified, extensible video segmentation starter codebase using `PySceneDetect`.

## Overview
This repository provides a modular, multimodal segmentation pipeline:
1. **Ingest (`src/ingest.py`)**: Common Infra: Hardware-accelerated metadata and media extraction.
2. **Segmentation (`src/segmentation.py`)**: **ORCHESTRATOR**: Coordinates shot detection and feature extraction.
3. **Features (`src/features/`)**: Specialized research modules:
    - `visual.py`: Motion analysis, color variance, and **Global Visual Profiling**.
    - `audio.py`: **Spectral Centroid**, **Bandwidth**, and **Universal Audio Profiling**.
4. **Classification (`src/classification.py`)**: **DECISION MAKER**: Distance-based logic that identifies "Main Content" by comparing segments against the video's **Universal Baseline**.

5. **Export (`src/export.py`)**: Common Infra: Formats results into the Web Player JSON schema.


## Setup

1. Install system dependency:
```bash
brew install ffmpeg
```

2. Install Python packages:
```bash
pip3 install -r requirements.txt
```

## How to use the CLI
Run the main pipeline:
```bash
python3 src/main.py --input /path/to/test_001.mp4 --output result.json
```

## Developer Guide: Extending the Pipeline

This project uses an **Orchestrated Architecture** designed for collaborative research.

### 1. Adding New Features
To add a new analysis capability (e.g., OCR, Face Detection, Sentiment):
1.  Create a new file in `src/features/` (e.g., `src/features/text.py`).
2.  Implement your extraction function.
3.  Import and call it within the loop in `src/segmentation.py:run_segmentation_pipeline`.
4.  The results will automatically be available to the classifier.

### 2. Modifying Segmentation Strategy
Teammates can replace `PySceneDetect` with their own logic in `src/segmentation.py`. The `run_segmentation_pipeline` should always return a list of segments with their attached features.

### 3. Tuning Classification
Update `src/classification.py` to use the features (motion, audio energy, etc.) to define what constitutes an "Ad" vs "Main Content".

---

## Algorithm Explanation: The Multimodal Engine

We use a deterministic pipeline that relies on classical machine learning and broadcast standard constraints to isolate ads without expensive AI inference.

### 1. Delta Features (The K-Means Inputs)
To prevent the algorithm from confusing "loud/fast movie scenes" with "loud/fast commercials", we evaluate relative feature spikes instead of raw absolute values.

We calculate a **5-Minute Rolling Median** across the video timeline. The K-Means engine is then fed the **Deltas** (Difference from the Median). The 6-feature matrix includes:
1. **Delta Audio Energy**: Loudness spike relative to the surrounding 5 minutes.
2. **ZCR (Zero-Crossing Rate)**: High in commercials due to excited speech and compressed pop music.
3. **Delta Motion Energy Variance**: How frenetic/fast-cutting the visual scene is relative to the baseline.
4. **Spatial Edge Density**: Captures on-screen text, graphics, and hard logos common in ads.
5. **Pacing Score**: Shot boundary frequency (cuts per minute).
6. **Delta RMS Variance**: Compressed audio dynamic range.

### 2. K-Means Clustering (K=10)
We pass the delta features into a pure numpy **K-Means clustering algorithm configured with K=10**. 
This slices the video into 10 highly distinct behavioral states (e.g., quiet dialogue, fast action, silent black frames, loud commercial graphics).

### 3. The Two-Tier Modulo-15 Engine (Decision Maker)
Once the 10 clusters are generated, we must identify which ones represent Advertisements.
Because broadcast television strictly enforces commercial runtimes (15s, 30s, 45s, 60s), we evaluate the **Hit Rate** of each cluster—how often sequences inside the cluster perfectly match a broadcast runtime.

We evaluate all 10 clusters against a Two-Tier ruleset in `_label_clusters`:
- **Rule A (Strong Match)**: If a cluster's blocks match Modulo-15 lengths `≥ 40%` of the time, the entire cluster is flagged as an Ad.
- **Rule B (Moderate Match + High Contrast)**: If a cluster's blocks match Modulo-15 lengths `≥ 25%` of the time, AND the cluster's centroid exhibits a huge Feature Contrast Score (massive spike in motion/audio/edges), it is flagged as an Ad.

---

## How to Run Tests

This project uses `pytest`. Ensure you have dependencies installed.

### Run All Tests
```bash
python3 -m pytest
```

### Run Specific Test Suites
```bash
# Unit tests only
python3 -m pytest tests/unit/ -v

# End-to-end tests only
python3 -m pytest tests/e2e/ -v
```

**What the E2E test does:**
The End-to-End test ensures the full pipeline is functional by:
1. **Creating a synthetic video**: Generates a 30s `.mp4` with a known structure (Red 5s, Blue 20s, Black 5s).
2. **Running the Pipeline**: Invokes `run_pipeline` to process this video from Ingest to Export.
3. **Validating Schema**: Checks that the output JSON strictly matches the required web player schema.

## Visualization
Open `player/index.html` in your browser. You can upload your video file and the corresponding `result.json` output to visualize the segments, thumbnails, and multimodal features in the interactive player dashboard.

4. **Verifying Logic**: Confirms that the baseline algorithm correctly classifies the short segments as "ads" and the long segment as "content".
5. **CLI Check**: Runs the pipeline via `subprocess` to ensure the command-line interface is fully working.
