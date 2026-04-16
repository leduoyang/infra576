# AI Context - Video Segmentation Project

## Project Goal
Refactor a legacy video segmentation pipeline into a modular, research-friendly multimodal system. The system must synchronize frames and audio perfectly and allow for rapid experimentation with features and classification rules.

## Core Capabilities
- **Synchronized Ingestion**: VFR-safe frame extraction (`POS_MSEC`) and calibrated audio offset extraction.
- **Orchestrated Segmentation**: `src/segmentation.py` acts as the workflow manager.
- **Modular Features**: `src/features/` contains independent visual and audio analyzers.
- **Extensible Classification**: `src/classification.py` allows for multimodal rule definitions (e.g., motion + energy).

## Research Logic (Branch: main)
- **Segmentation**: Currently uses PySceneDetect `ContentDetector` with a merge/split post-processing layer.
- **Features**: 
    - `visual.py`: Motion score and color histogram variance.
    - `audio.py`: RMS spectral energy.
- **Classification**: Heuristic based on segment duration, multimodal dynamism, and loudness.

## How to Assist
- **Adding Features**: Create a new file in `src/features/` and register the call in `src/segmentation.py`.
- **Replacing Detectors**: Swap the `detect_scenes_scenedetect` call in `src/segmentation.py` with any custom logic (e.g., ML shot boundary models).
- **Tuning Thresholds**: Adjust `classification.py` based on feature distribution analysis.

---
*This document serves as the primary context for AI tools to understand the architecture and research patterns.*
