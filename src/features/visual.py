import cv2
import numpy as np
import subprocess
import json
from typing import List


def detect_scenes_scenedetect(video_path: str, threshold: float = 15.0) -> List[dict]:
    """
    Finds shot boundaries (cuts) in the video using the PySceneDetect library.
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        scenes = []
        for start, end in scene_list:
            s = start.get_seconds()
            e = end.get_seconds()
            scenes.append({
                "start_seconds": s,
                "end_seconds": e,
                "duration_seconds": e - s,
            })
        return scenes

    except Exception:
        # Fallback: return single scene covering the full video using ffprobe.
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = 0.0
        if result.returncode == 0:
            info = json.loads(result.stdout)
            duration = float(info.get("format", {}).get("duration", 0))
        return [{"start_seconds": 0.0, "end_seconds": duration, "duration_seconds": duration}]


# ---------------------------------------------------------------------------
# Basic visual features
# ---------------------------------------------------------------------------

def compute_frame_diff(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compute mean absolute difference between two frames."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(gray1, gray2)))


def analyze_motion_for_segment(
    frames_with_ts: List[dict],
    start_sec: float,
    end_sec: float,
) -> float:
    """Returns the average motion score for a specific segment."""
    segment_frames = [
        f["frame"]
        for f in frames_with_ts
        if start_sec <= f["timestamp_seconds"] < end_sec
    ]
    if len(segment_frames) < 2:
        return 0.0

    diffs = []
    for i in range(1, len(segment_frames)):
        diffs.append(compute_frame_diff(segment_frames[i - 1], segment_frames[i]))
    return float(np.mean(diffs))


def analyze_color_variance(
    frames_with_ts: List[dict],
    start_sec: float,
    end_sec: float,
) -> float:
    """
    Computes the variance of color histograms across frames in a specific segment.
    """
    segment_frames = [
        f["frame"]
        for f in frames_with_ts
        if start_sec <= f["timestamp_seconds"] < end_sec
    ]
    if not segment_frames:
        return 0.0

    hists = []
    for frame in segment_frames:
        hist = cv2.calcHist(
            [frame], [0, 1, 2], None,
            [8, 8, 8], [0, 256, 0, 256, 0, 256],
        )
        hists.append(hist.flatten())

    if len(hists) < 2:
        return 0.0

    hists_array = np.array(hists)
    variance = np.mean(np.var(hists_array, axis=0))
    return float(variance)


def analyze_edge_density(frame: np.ndarray) -> float:
    """
    Compute edge density (fraction of edge pixels) using Canny.
    High edge density suggests text overlays, title cards, or graphics.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.mean(edges > 0))


def analyze_edge_density_for_segment(
    frames_with_ts: List[dict],
    start_sec: float,
    end_sec: float,
) -> float:
    """Average edge density across frames in a segment."""
    segment_frames = [
        f["frame"]
        for f in frames_with_ts
        if start_sec <= f["timestamp_seconds"] < end_sec
    ]
    if not segment_frames:
        return 0.0
    return float(np.mean([analyze_edge_density(f) for f in segment_frames]))


def analyze_frame_self_similarity(
    frames_with_ts: List[dict],
    start_sec: float,
    end_sec: float,
) -> float:
    """
    Measures how static/repetitive a segment is by computing average histogram
    correlation between consecutive frames.

    Returns value in [0, 1], where higher means more static/similar frames.
    Useful for detecting title cards, holding screens, and end cards.
    """
    segment_frames = [
        f["frame"]
        for f in frames_with_ts
        if start_sec <= f["timestamp_seconds"] < end_sec
    ]
    if len(segment_frames) < 2:
        return 1.0

    correlations = []
    for i in range(1, len(segment_frames)):
        h1 = cv2.calcHist(
            [segment_frames[i - 1]], [0, 1, 2], None,
            [8, 8, 8], [0, 256, 0, 256, 0, 256],
        ).flatten().astype(np.float32)
        h2 = cv2.calcHist(
            [segment_frames[i]], [0, 1, 2], None,
            [8, 8, 8], [0, 256, 0, 256, 0, 256],
        ).flatten().astype(np.float32)
        corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        correlations.append(corr)

    return float(np.mean(correlations))


# ---------------------------------------------------------------------------
# Active-layout / black-border features
# ---------------------------------------------------------------------------

def _median_float(values, default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return float(default)
    return float(np.median(vals))


def analyze_active_layout_for_frame(
    frame: np.ndarray,
    black_thresh: int = 24,
    min_active_ratio: float = 0.01,
) -> dict:
    """
    Detect the active non-black visual area of one frame.

    This helps distinguish:
      - main content with stable black border / letterbox frame
      - intro/outro with a different active area or full-screen title card

    The input frames may already be resized. That is OK because the returned
    values are ratios, not absolute pixel coordinates.
    """
    if frame is None or frame.size == 0:
        return {
            "frame_width": 0.0,
            "frame_height": 0.0,
            "active_width_ratio": 0.0,
            "active_height_ratio": 0.0,
            "active_aspect_ratio": 0.0,
            "border_left_ratio": 0.0,
            "border_right_ratio": 0.0,
            "border_top_ratio": 0.0,
            "border_bottom_ratio": 0.0,
            "black_border_ratio": 1.0,
        }

    h, w = frame.shape[:2]

    # BGR frame. A pixel is black if all channels are very dark.
    black_mask = np.all(frame <= black_thresh, axis=2)
    active_mask = ~black_mask

    active_ratio = float(np.mean(active_mask))
    black_ratio = float(np.mean(black_mask))

    # All-black / mostly-black title card.
    if active_ratio < min_active_ratio:
        return {
            "frame_width": float(w),
            "frame_height": float(h),
            "active_width_ratio": 0.0,
            "active_height_ratio": 0.0,
            "active_aspect_ratio": 0.0,
            "border_left_ratio": 0.5,
            "border_right_ratio": 0.5,
            "border_top_ratio": 0.5,
            "border_bottom_ratio": 0.5,
            "black_border_ratio": black_ratio,
        }

    ys, xs = np.where(active_mask)

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    active_w = max(1, x_max - x_min + 1)
    active_h = max(1, y_max - y_min + 1)

    active_w_ratio = active_w / max(w, 1)
    active_h_ratio = active_h / max(h, 1)

    return {
        "frame_width": float(w),
        "frame_height": float(h),
        "active_width_ratio": float(active_w_ratio),
        "active_height_ratio": float(active_h_ratio),
        "active_aspect_ratio": float(active_w_ratio / max(active_h_ratio, 1e-6)),
        "border_left_ratio": float(x_min / max(w, 1)),
        "border_right_ratio": float((w - 1 - x_max) / max(w, 1)),
        "border_top_ratio": float(y_min / max(h, 1)),
        "border_bottom_ratio": float((h - 1 - y_max) / max(h, 1)),
        "black_border_ratio": black_ratio,
    }


def analyze_layout_for_segment(
    frames_with_ts: List[dict],
    start_sec: float,
    end_sec: float,
) -> dict:
    """
    Compute median active-layout / black-border features for a segment.

    These fields are consumed by classification.py:
      - frame_width
      - frame_height
      - active_width_ratio
      - active_height_ratio
      - active_aspect_ratio
      - border_left_ratio
      - border_right_ratio
      - border_top_ratio
      - border_bottom_ratio
      - black_border_ratio
    """
    segment_frames = [
        f["frame"]
        for f in frames_with_ts
        if start_sec <= f["timestamp_seconds"] < end_sec
    ]

    if not segment_frames:
        return {
            "frame_width": 0.0,
            "frame_height": 0.0,
            "active_width_ratio": 0.0,
            "active_height_ratio": 0.0,
            "active_aspect_ratio": 0.0,
            "border_left_ratio": 0.0,
            "border_right_ratio": 0.0,
            "border_top_ratio": 0.0,
            "border_bottom_ratio": 0.0,
            "black_border_ratio": 0.0,
            "layout_frame_count": 0,
        }

    layouts = [analyze_active_layout_for_frame(frame) for frame in segment_frames]

    return {
        "frame_width": _median_float([x["frame_width"] for x in layouts]),
        "frame_height": _median_float([x["frame_height"] for x in layouts]),
        "active_width_ratio": _median_float([x["active_width_ratio"] for x in layouts]),
        "active_height_ratio": _median_float([x["active_height_ratio"] for x in layouts]),
        "active_aspect_ratio": _median_float([x["active_aspect_ratio"] for x in layouts]),
        "border_left_ratio": _median_float([x["border_left_ratio"] for x in layouts]),
        "border_right_ratio": _median_float([x["border_right_ratio"] for x in layouts]),
        "border_top_ratio": _median_float([x["border_top_ratio"] for x in layouts]),
        "border_bottom_ratio": _median_float([x["border_bottom_ratio"] for x in layouts]),
        "black_border_ratio": _median_float([x["black_border_ratio"] for x in layouts]),
        "layout_frame_count": len(segment_frames),
    }


# ---------------------------------------------------------------------------
# Global visual profile
# ---------------------------------------------------------------------------

def compute_global_visual_profile(frames_with_ts: List[dict]) -> dict:
    """
    Computes the 'Universal Visual Profile' for the entire video.
    """
    if len(frames_with_ts) < 2:
        return {
            "avg_motion": 0.0,
            "avg_color_variance": 0.0,
            "avg_edge_density": 0.0,
            "std_motion": 0.0,
        }

    # 1. Motion stats
    diffs = []
    for i in range(1, len(frames_with_ts)):
        diffs.append(
            compute_frame_diff(frames_with_ts[i - 1]["frame"], frames_with_ts[i]["frame"])
        )
    avg_motion = float(np.mean(diffs))
    std_motion = float(np.std(diffs))

    # 2. Avg color variance, sampled
    sampled_frames = frames_with_ts[::max(1, len(frames_with_ts) // 20)]
    hists = []
    for f in sampled_frames:
        hist = cv2.calcHist(
            [f["frame"]], [0, 1, 2], None,
            [8, 8, 8], [0, 256, 0, 256, 0, 256],
        )
        hists.append(hist.flatten())
    avg_color_variance = float(np.mean(np.var(np.array(hists), axis=0))) if hists else 0.0

    # 3. Avg edge density, sampled
    edge_densities = [analyze_edge_density(f["frame"]) for f in sampled_frames]
    avg_edge_density = float(np.mean(edge_densities)) if edge_densities else 0.0

    return {
        "avg_motion": avg_motion,
        "std_motion": std_motion,
        "avg_color_variance": avg_color_variance,
        "avg_edge_density": avg_edge_density,
    }
