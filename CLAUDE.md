# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

USC CSC 576 (Spring 2026) video segmentation project. Given a video that has ads spliced into it, the pipeline detects the ad regions and emits a JSON description of the timeline so the web player can mark ads vs. original content.

## Commands

```bash
# Setup
brew install ffmpeg              # Hard requirement; tests skip without it (see conftest.py)
pip3 install -r requirements.txt

# Run the pipeline on one video
python3 src/main.py --input videos_with_ads/test_001.mp4 --output result.json
# Optional: --scene-threshold (PySceneDetect ContentDetector threshold, default 15.0)

# Tests
python3 -m pytest                    # all
python3 -m pytest tests/unit/ -v
python3 -m pytest tests/e2e/ -v      # builds a synthetic 30s video via ffmpeg and runs the full pipeline
python3 -m pytest tests/unit/test_classification.py::TestName::test_x -v
```

The web player (`player/index.html`) is static — open it in a browser and upload both the source video and its `*_segments.json` to inspect results.

## Architecture: the Orchestrator pattern

The pipeline is intentionally split into **common infra** (ingest/export) vs. **research code** (segmentation/features/classification). `src/main.py:run_pipeline` wires them together — read it first when orienting.

```
ingest → segmentation (orchestrator) → classification → export
```

Two non-obvious points that drive the rest of the code:

**1. `run_segmentation_pipeline` is the orchestrator, not just a shot detector.**
It loads audio + frames once, then runs `detect_scenes_scenedetect` *and* attaches every per-segment feature in one pass. Classification receives feature-rich segments and never re-reads media. Its signature is `(segments, global_profile)` — both are needed downstream. To add a new modality (OCR, faces, transcripts), drop a module in `src/features/` and add a single line inside the per-scene loop in `src/segmentation.py`; it propagates to classification automatically. Don't re-open the video from `classification.py`.

**2. "Universal Baseline" classification.**
`compute_global_visual_profile` and `compute_global_audio_profile` produce a per-video baseline (mean motion, mean spectral centroid, mean bandwidth, etc.) that `classify_segments` compares each segment against. The idea: the dominant content of a video defines what's "normal", and ads are detected as deviations from that baseline — not by absolute thresholds. When tuning, change *deviation ratios* in `classification.py`, not absolute numbers. After labeling, `merge_consecutive_segments` collapses adjacent same-type runs, so don't expect the timeline length to match `len(scenes)`.

## Critical invariants

- **VFR-safe frame extraction.** `extract_frames` walks `CAP_PROP_POS_MSEC`, not frame indices. Any new visual feature must consume the `[{"timestamp_seconds", "frame"}]` shape and use timestamps to align with segment boundaries.
- **Audio/video stream offsets.** `get_video_metadata` reads `audio_start_time` from ffprobe; `analyze_audio_features` subtracts it before indexing into the librosa array. Real-world videos often have non-zero audio start; never index audio with raw segment seconds.
- **Output schema is fixed.** The `final_video_*` fields refer to the input video's timeline (which already has ads in it); `original_video_*` fields are the would-be timeline with ads removed. `_build_timeline` and `_build_inserted_ads` in `src/export.py` keep `acc_ad_dur` to map between the two — preserve that bookkeeping when changing the exporter. Reference outputs live in `video_info/*.json`; e2e tests in `tests/e2e/test_pipeline_e2e.py` enforce the schema.
- **Detector fallback.** `detect_scenes_scenedetect` swallows exceptions and returns one scene covering the whole video. If you suspect detection is silently failing, check that PySceneDetect actually imported — don't assume one giant scene means a single-shot video.

## Local data

- `videos_with_ads/test_00{1..5}.mp4` — sample inputs (untracked, ~400 MB total).
- `video_info/test_00{1..5}.json` — reference outputs in the expected schema; useful for diffing when changing the exporter.
- `tmp_audio.wav` is written by `run_segmentation_pipeline` in the CWD and is not auto-cleaned. Don't commit it.
