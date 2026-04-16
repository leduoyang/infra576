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
