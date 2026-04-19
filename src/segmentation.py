"""
segmentation.py – 5-Step K-Means pipeline orchestrator.

Data flow per the MoviePy integration spec:
  Step 1  PySceneDetect → list of (start, end) timestamps
  Step 1b MoviePy loads the full video once → audio array for dropout detection
  Step 2  For each segment:
            subclip = clip.subclip(start, end)
            Branch B: subclip.audio.to_soundarray() → Librosa (audio features)
            Branch A: subclip.iter_frames()        → OpenCV (visual features)
  Steps 3-5 handled by classify_segments() in classification.py
"""

import numpy as np
import librosa

from moviepy.editor import VideoFileClip

from src.features.visual import (
    detect_scenes_scenedetect,
    _bgr_frames_from_moviepy,
    analyze_motion_from_frames,
    analyze_motion_energy_variance_from_frames,
    analyze_spatial_edge_density_from_frames,
    analyze_color_variance_from_frames,
    analyze_sharpness_from_frames,
    analyze_hsv_from_frames,
    analyze_letterbox_from_frames,
    analyze_watermark_from_frames,
    analyze_solid_color_from_frames,
)
from src.features.audio import (
    analyze_audio_from_array,
    compute_rms_variance_from_array,
    compute_zcr_from_array,
)
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Audio signal constants
# ---------------------------------------------------------------------------
_AUDIO_SR = 16_000       # sample rate; librosa default for speech/music features
_SILENCE_DB = 40.0       # top_db for librosa silence split
_MIN_SILENCE_DUR = 0.5   # minimum gap (seconds) to count as an audio-dropout bookend
_FRAME_SAMPLE_FPS = 2.0  # frames per second to sample from iter_frames
_RESIZE = (320, 180)     # resize visual frames for speed


# ---------------------------------------------------------------------------
# Step 1 helper – audio-dropout bookend detection
# ---------------------------------------------------------------------------

def _detect_audio_dropouts(y: np.ndarray, sr: int,
                            top_db: float = _SILENCE_DB,
                            min_silence_dur: float = _MIN_SILENCE_DUR) -> list[float]:
    """
    Returns timestamps (seconds) of audio-silence gaps ≥ min_silence_dur.
    Used as bookend markers to supplement PySceneDetect visual cuts.
    """
    intervals = librosa.effects.split(y, top_db=top_db)
    gaps = []
    for k in range(1, len(intervals)):
        gap_start_s = intervals[k - 1][1]
        gap_end_s   = intervals[k][0]
        if (gap_end_s - gap_start_s) / sr >= min_silence_dur:
            gaps.append(float(gap_start_s / sr))
    return gaps


# ---------------------------------------------------------------------------
# Step 1 helper – merge boundary lists
# ---------------------------------------------------------------------------

def _merge_boundary_lists(scene_boundaries: list[dict],
                           dropout_times: list[float],
                           total_duration: float,
                           min_dur: float = 2.0) -> list[dict]:
    """
    Merges visual scene-cut timestamps and audio-dropout timestamps into one
    sorted segment list, then fans very short segments into their neighbours.
    """
    cuts = sorted({0.0}
                  | {s["start_seconds"] for s in scene_boundaries}
                  | set(dropout_times))
    cuts = [t for t in cuts if t < total_duration]
    cuts.append(total_duration)

    segments = [{"start_seconds": cuts[i],
                 "end_seconds":   cuts[i + 1],
                 "duration_seconds": cuts[i + 1] - cuts[i]}
                for i in range(len(cuts) - 1)]

    return merge_short_scenes(segments, min_duration=min_dur)


# ---------------------------------------------------------------------------
# Step 2 helpers – per-segment extraction via MoviePy subclip
# ---------------------------------------------------------------------------

def _get_audio_array(subclip, sr: int = _AUDIO_SR) -> np.ndarray:
    """
    Extract a mono float32 audio array from a MoviePy subclip.
    Works around MoviePy 1.0.3 + NumPy 2.x np.vstack generator TypeError
    by manually coercing iter_chunks into a list before stacking.
    """
    if subclip.audio is None:
        return np.zeros(0, dtype=np.float32)
    # manual to_soundarray to avoid generator np.vstack error
    chunks = list(subclip.audio.iter_chunks(fps=sr, quantize=False, chunksize=16000))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    raw = np.vstack(chunks)
    mono = raw.mean(axis=1) if raw.ndim > 1 else raw
    return mono.astype(np.float32)


def _get_bgr_frames(subclip, sample_fps: float = _FRAME_SAMPLE_FPS,
                    resize: tuple = _RESIZE) -> list[np.ndarray]:
    """
    Sample frames from a MoviePy subclip at sample_fps and convert RGB→BGR.
    Uses iter_frames() which yields raw numpy arrays — passed directly to OpenCV.
    """
    fps    = subclip.fps or sample_fps
    stride = max(1, int(round(fps / sample_fps)))
    rgb_frames = [f for i, f in enumerate(subclip.iter_frames()) if i % stride == 0]
    if not rgb_frames:
        return []
    import cv2
    bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in rgb_frames]
    if resize:
        bgr = [cv2.resize(f, resize) for f in bgr]
    return bgr


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------

def run_segmentation_pipeline(
    video_path: str,
    metadata: dict,
    scene_threshold: float = 15.0,
) -> list[dict]:
    """
    ORCHESTRATOR: 5-Step K-Means Overlapping Sliding Window Pipeline.

    Step 1: Raw Timestamp Generation (No Merging)
    Step 2/3: Fixed 15-second Overlapping Windows & Feature Extraction
    """
    # ── Step 1a: Load video with MoviePy ────────────────────────────────────
    print(" - Step 1a: Loading video with MoviePy...")
    clip = VideoFileClip(str(video_path))
    duration = clip.duration

    # ── Step 1b: PySceneDetect for visual bookends ───────────────────────────
    print(" - Step 1b: PySceneDetect (raw visual cuts)...")
    scenes = detect_scenes_scenedetect(video_path, threshold=scene_threshold)
    raw_scene_cuts = [s["start_seconds"] for s in scenes]

    # ── Step 1c: Audio dropout detection for audio bookends ─────────────────
    print(" - Step 1c: Building full audio array for raw dropout detection...")
    if clip.audio is not None:
        chunks = list(clip.audio.iter_chunks(fps=_AUDIO_SR, quantize=False, chunksize=16000))
        if chunks:
            full_raw = np.vstack(chunks)
            full_mono = full_raw.mean(axis=1) if full_raw.ndim > 1 else full_raw
            full_mono = full_mono.astype(np.float32)
        else:
            full_mono = np.zeros(0, dtype=np.float32)
    else:
        full_mono = np.zeros(0, dtype=np.float32)

    audio_offset = metadata.get("audio_start_time", 0.0)
    raw_dropouts = _detect_audio_dropouts(full_mono, _AUDIO_SR)
    raw_dropouts = [max(0.0, t + audio_offset) for t in raw_dropouts]

    # Combine strict timestamps (No merging into arbitrary segment boundaries)
    all_cuts = sorted(set(raw_scene_cuts) | set(raw_dropouts))

    # ── Step 2 & 3: Sliding Window Generator & Extraction ────────────────────────
    window_size = 15.0
    step_size = 5.0
    num_windows = int(np.ceil(duration / step_size))

    print(f" - Step 3: Extracting features for {num_windows} overlapping {window_size}s windows...")

    segments = []
    
    for i in tqdm(range(num_windows), desc="Windows", unit="win"):
        start = i * step_size
        end = min(start + window_size, duration)

        if start >= duration:
            break

        scene = {
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": end - start
        }

        # Pacing/Freneticism: Count raw cuts strictly inside this window
        cuts_in_window = [t for t in all_cuts if start <= t < end]
        scene["pacing_score"] = float(len(cuts_in_window))

        # Isolate this window via MoviePy subclip
        subclip = clip.subclip(start, end)

        # ── Branch B: Audio → Librosa ────────────────────────────────────
        audio_y = _get_audio_array(subclip)

        audio_feats = analyze_audio_from_array(audio_y, _AUDIO_SR)
        scene.update(audio_feats)
        scene["audio_rms_variance"] = compute_rms_variance_from_array(audio_y, _AUDIO_SR)
        scene["zcr_mean"]           = compute_zcr_from_array(audio_y)

        # ── Branch A: Frames → OpenCV ────────────────────────────────────
        bgr_frames = _get_bgr_frames(subclip)

        scene["motion_score"]           = analyze_motion_from_frames(bgr_frames)
        scene["motion_energy_variance"] = analyze_motion_energy_variance_from_frames(bgr_frames)
        scene["spatial_edge_density"]   = analyze_spatial_edge_density_from_frames(bgr_frames)
        scene["color_variance"]         = analyze_color_variance_from_frames(bgr_frames)
        scene["avg_sharpness"]          = analyze_sharpness_from_frames(bgr_frames)

        sat, val = analyze_hsv_from_frames(bgr_frames)
        scene["avg_saturation"]         = sat
        scene["avg_luminance"]          = val
        scene["letterbox_variance"]     = analyze_letterbox_from_frames(bgr_frames)
        scene["watermark_density"]      = analyze_watermark_from_frames(bgr_frames)

        is_solid = analyze_solid_color_from_frames(bgr_frames)
        scene["is_transition_marker"]   = is_solid and scene["duration_seconds"] < 2.0

        subclip.close()
        segments.append(scene)

    clip.close()
    return segments
