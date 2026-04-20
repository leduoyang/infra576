import librosa
import subprocess
import json
import numpy as np
from src.features.visual import (
    detect_scenes_scenedetect, 
    analyze_motion_for_segment, 
    analyze_color_variance,
    compute_global_visual_profile
)
from src.features.audio import analyze_audio_features, compute_global_audio_profile
from src.features.speech import transcribe, speech_features_for_segment
from src.ingest import extract_audio, extract_frames

def run_segmentation_pipeline(
    video_path: str,
    metadata: dict,
    scene_threshold: float = 15.0,
    transcript_cache: str = None,
) -> tuple[list[dict], dict]:
    """
    ORCHESTRATOR: Executes the full segmentation workflow.
    
    1. Detect raw scenes (shots).
    2. Extract multimodal features (Audio Energy, Color Variance, Motion).
    3. Return feature-rich segments.
    """
    # 1. DATA LOADING
    # Extract audio for features
    audio_path = "tmp_audio.wav"
    extract_audio(video_path, audio_path)
    y, sr = librosa.load(audio_path, sr=16000)

    # 1b. Speech transcription (Whisper, cached per video)
    transcript = transcribe(audio_path, transcript_cache)
    
    # Extract frames for visual features
    frames = extract_frames(video_path, sample_fps=2.0)
    
    # 2. GLOBAL PROFILING (Universal Feature Baseline)
    global_visual = compute_global_visual_profile(frames)
    global_audio = compute_global_audio_profile(y, sr)
    global_profile = {**global_visual, **global_audio}
    
    # 3. RAW SEGMENTATION
    # Teammates: You can replace detect_scenes_scenedetect with other techniques here
    scenes = detect_scenes_scenedetect(video_path, threshold=scene_threshold)
    scenes = merge_short_scenes(scenes, min_duration=2.0)
    # scenes = split_long_scenes(scenes, max_duration=120.0)

    # 4. FEATURE ATTACHMENT
    audio_offset = metadata.get("audio_start_time", 0.0)
    
    for scene in scenes:
        start = scene["start_seconds"]
        end = scene["end_seconds"]
        
        # Audio features (Energy, Centroid, Bandwidth)
        audio_feats = analyze_audio_features(y, sr, start, end, audio_offset)
        scene.update(audio_feats)
        
        # Visual features (Motion and Color)
        scene["motion_score"] = analyze_motion_for_segment(frames, start, end)
        scene["color_variance"] = analyze_color_variance(frames, start, end)

        # Speech features (Whisper)
        scene.update(speech_features_for_segment(transcript, start, end))

    return scenes, global_profile

def merge_short_scenes(scenes: list[dict], min_duration: float = 2.0) -> list[dict]:
    if not scenes: return []
    result = []
    for scene in scenes:
        if not result:
            result.append(scene.copy())
        else:
            if result[-1]["duration_seconds"] < min_duration or scene["duration_seconds"] < min_duration:
                result[-1]["end_seconds"] = scene["end_seconds"]
                result[-1]["duration_seconds"] = result[-1]["end_seconds"] - result[-1]["start_seconds"]
            else:
                result.append(scene.copy())
    return result

def split_long_scenes(scenes: list[dict], max_duration: float = 120.0) -> list[dict]:
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
                e = min(scene["start_seconds"] + (i + 1) * sub_dur, scene["end_seconds"])
                result.append({"start_seconds": s, "end_seconds": e, "duration_seconds": e - s})
    return result
