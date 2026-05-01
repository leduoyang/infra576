"""
segmentation.py – Orchestrates the full multimodal segmentation workflow.

Pipeline:
  1. Extract audio → WAV
  2. Extract frames at sample FPS
  3. (Optional) Transcribe audio via Whisper
  4. Compute global profiles (audio + visual + transcript)
  5. Detect shot boundaries (PySceneDetect)
  6. Force small chunks near intro/outro boundary regions
  7. Attach per-segment multimodal features
"""

import os
import librosa
import subprocess
import json
import numpy as np

from src.features.visual import (
    detect_scenes_scenedetect,
    analyze_motion_for_segment,
    analyze_color_variance,
    analyze_edge_density_for_segment,
    analyze_frame_self_similarity,
    analyze_layout_for_segment,
    compute_global_visual_profile,
)
from src.features.audio import analyze_audio_features, compute_global_audio_profile
from src.features.transcript import (
    transcribe_audio,
    analyze_transcript_features,
    compute_global_transcript_profile,
)
from src.ingest import extract_audio, extract_frames


def run_segmentation_pipeline(
    video_path: str,
    metadata: dict,
    scene_threshold: float = 15.0,
    use_transcript: bool = True,
    whisper_model: str = "base",
) -> tuple[list[dict], dict]:
    """
    ORCHESTRATOR: Executes the full segmentation workflow.

    1. Data loading (audio + frames)
    2. Global profiling
    3. Shot boundary detection
    4. Boundary-region splitting
    5. Per-segment multimodal feature attachment
    6. Optional transcript feature attachment

    Returns:
        (segments, global_profile)
    """
    # 1. DATA LOADING
    audio_path = _temp_audio_path(video_path)
    extract_audio(video_path, audio_path)
    y, sr = librosa.load(audio_path, sr=16000)

    frames = extract_frames(video_path, sample_fps=2.0)

    # 2. TRANSCRIPTION (optional, runs before segmentation for overlap)
    transcript = []
    if use_transcript:
        try:
            transcript = transcribe_audio(audio_path, model_size=whisper_model)
        except Exception as e:
            print(f"[segmentation] Transcript extraction failed: {e}")
            transcript = []

    # 3. GLOBAL PROFILING
    global_visual = compute_global_visual_profile(frames)
    global_audio = compute_global_audio_profile(y, sr)
    global_transcript = compute_global_transcript_profile(
        transcript, metadata["duration_seconds"]
    )
    global_profile = {**global_visual, **global_audio, **global_transcript}

    # 4. RAW SEGMENTATION (shot boundary detection)
    scenes = detect_scenes_scenedetect(video_path, threshold=scene_threshold)
    scenes = merge_short_scenes(scenes, min_duration=2.0)

    # Important:
    # Classification can only label whole segments. If PySceneDetect returns a scene
    # that contains both intro + main content, the classifier cannot cut inside it.
    # So we force small chunks near beginning/end where intro/outro usually occur.
    boundary_sec = min(140.0, max(30.0, metadata["duration_seconds"] * 0.12))
    scenes = split_boundary_regions(
        scenes,
        total_duration=metadata["duration_seconds"],
        boundary_sec=boundary_sec,
        step_sec=0.5,
    )

    # 5. FEATURE ATTACHMENT
    audio_offset = metadata.get("audio_start_time", 0.0)

    for scene in scenes:
        start = scene["start_seconds"]
        end = scene["end_seconds"]

        # Audio features
        audio_feats = analyze_audio_features(y, sr, start, end, audio_offset)
        scene.update(audio_feats)

        # Visual features
        scene["motion_score"] = analyze_motion_for_segment(frames, start, end)
        scene["color_variance"] = analyze_color_variance(frames, start, end)
        scene["edge_density"] = analyze_edge_density_for_segment(frames, start, end)
        scene["frame_self_similarity"] = analyze_frame_self_similarity(frames, start, end)

        # Layout / black-border features
        # These are consumed by classification.py to detect whether intro/outro
        # has a different active area / black-frame pattern from main content.
        layout_feats = analyze_layout_for_segment(frames, start, end)
        scene.update(layout_feats)

        # Transcript features
        if transcript:
            tx_feats = analyze_transcript_features(transcript, start, end)
            scene.update(tx_feats)

    # Cleanup temp audio
    try:
        os.remove(audio_path)
    except OSError:
        pass

    return scenes, global_profile


def _temp_audio_path(video_path: str) -> str:
    """Generate a temp audio path next to the video file."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(os.path.dirname(video_path) or ".", f".tmp_{base}_audio.wav")


def merge_short_scenes(scenes: list[dict], min_duration: float = 2.0) -> list[dict]:
    """
    Merge tiny scenes so random one-frame cuts do not produce unstable features.
    """
    if not scenes:
        return []

    result = []
    for scene in scenes:
        if not result:
            result.append(scene.copy())
        else:
            if (
                result[-1]["duration_seconds"] < min_duration
                or scene["duration_seconds"] < min_duration
            ):
                result[-1]["end_seconds"] = scene["end_seconds"]
                result[-1]["duration_seconds"] = (
                    result[-1]["end_seconds"] - result[-1]["start_seconds"]
                )
            else:
                result.append(scene.copy())
    return result


def split_boundary_regions(
    scenes: list[dict],
    total_duration: float,
    boundary_sec: float = 140.0,
    step_sec: float = 3.0,
) -> list[dict]:
    """
    Force smaller segments near the beginning and ending.

    Reason:
      PySceneDetect may return one scene that contains both intro + content.
      Classification can only label whole segments, so intro/outro regions need
      smaller chunks.

    Example:
      [0, 58] may become [0,3], [3,6], ..., [54,57], [57,58]
      near the opening boundary.
    """
    if not scenes:
        return []

    result = []

    intro_end = min(boundary_sec, total_duration)
    outro_start = max(0.0, total_duration - boundary_sec)

    for scene in scenes:
        s0 = float(scene["start_seconds"])
        e0 = float(scene["end_seconds"])

        if e0 <= s0:
            continue

        split_points = {s0, e0}

        # Split intro-side region.
        if s0 < intro_end and e0 > 0.0:
            intro_split_start = s0
            intro_split_end = min(e0, intro_end)
            x = intro_split_start
            while x < intro_split_end:
                split_points.add(float(x))
                x += step_sec
            split_points.add(float(intro_split_end))

        # Split outro-side region.
        if e0 > outro_start and s0 < total_duration:
            outro_split_start = max(s0, outro_start)
            outro_split_end = e0
            x = outro_split_start
            split_points.add(float(outro_split_start))
            while x < outro_split_end:
                split_points.add(float(x))
                x += step_sec
            split_points.add(float(outro_split_end))

        points = sorted(p for p in split_points if s0 <= p <= e0)

        for a, b in zip(points[:-1], points[1:]):
            if b - a <= 0.05:
                continue
            result.append({
                "start_seconds": float(a),
                "end_seconds": float(b),
                "duration_seconds": float(b - a),
            })

    return result


def split_long_scenes(
    scenes: list[dict],
    max_duration: float = 120.0,
) -> list[dict]:
    """
    Optional helper for breaking very long scenes.
    Currently not called by the pipeline, but kept for compatibility.
    """
    result = []
    for scene in scenes:
        dur = scene["duration_seconds"]
        if dur <= max_duration:
            result.append(scene)
        else:
            n_splits = int(np.ceil(dur / max_duration))
            sub_dur = dur / n_splits
            for i in range(n_splits):
                s = scene["start_seconds"] + i * sub_dur
                e = min(
                    scene["start_seconds"] + (i + 1) * sub_dur,
                    scene["end_seconds"],
                )
                result.append({
                    "start_seconds": s,
                    "end_seconds": e,
                    "duration_seconds": e - s,
                })
    return result
