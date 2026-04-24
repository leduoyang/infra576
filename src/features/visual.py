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
        # Fallback: return single scene covering the full video using ffprobe
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


def compute_frame_diff(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compute mean absolute difference between two frames."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(gray1, gray2)))

def analyze_motion_for_segment(frames_with_ts: List[dict], start_sec: float, end_sec: float) -> float:
    """Returns the average motion score for a specific segment."""
    segment_frames = [f["frame"] for f in frames_with_ts if start_sec <= f["timestamp_seconds"] < end_sec]
    if len(segment_frames) < 2: return 0.0
    
    diffs = []
    for i in range(1, len(segment_frames)):
        diffs.append(compute_frame_diff(segment_frames[i-1], segment_frames[i]))
    return float(np.mean(diffs))

def analyze_color_variance(frames_with_ts: List[dict], start_sec: float, end_sec: float) -> float:
    """
    Computes the variance of color histograms across frames in a specific segment.
    """
    segment_frames = [f["frame"] for f in frames_with_ts if start_sec <= f["timestamp_seconds"] < end_sec]
    if not segment_frames:
        return 0.0
    
    hists = []
    for frame in segment_frames:
        # Compute histogram for B, G, R channels
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hists.append(hist.flatten())
    
    if len(hists) < 2:
        return 0.0
        
    hists_array = np.array(hists)
    variance = np.mean(np.var(hists_array, axis=0))
    return float(variance)

def compute_active_aspect_ratio(frame: np.ndarray, dark_thr: int = 18) -> float:
    """Aspect ratio of the non-letterbox/pillarbox region.

    Crops rows/cols whose max luminance is below dark_thr (treated as black bars).
    Returns NaN for fade-to-black frames where active region is degenerate.
    Show vs ad often differ here: cinematic shows are 2.35:1 letterboxed,
    inserted ads typically fill 16:9.
    """
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows = np.where(g.max(axis=1) > dark_thr)[0]
    cols = np.where(g.max(axis=0) > dark_thr)[0]
    if len(rows) < 2 or len(cols) < 2:
        return float("nan")
    aw = int(cols[-1] - cols[0] + 1)
    ah = int(rows[-1] - rows[0] + 1)
    return float(aw) / float(ah)


def analyze_aspect_ratio_for_segment(frames_with_ts: List[dict], start_sec: float, end_sec: float) -> float:
    """Median active aspect ratio across frames in [start_sec, end_sec). NaN if none."""
    seg = [f["frame"] for f in frames_with_ts if start_sec <= f["timestamp_seconds"] < end_sec]
    if not seg:
        return float("nan")
    ars = [compute_active_aspect_ratio(f) for f in seg]
    ars = [a for a in ars if not np.isnan(a)]
    if not ars:
        return float("nan")
    return float(np.median(ars))


def compute_global_visual_profile(frames_with_ts: List[dict]) -> dict:
    """
    Computes the 'Universal Visual Profile' for the entire video.
    """
    if len(frames_with_ts) < 2:
        return {"avg_motion": 0.0, "avg_color_variance": 0.0, "avg_aspect_ratio": float("nan")}

    # 1. Avg Motion
    diffs = []
    for i in range(1, len(frames_with_ts)):
        diffs.append(compute_frame_diff(frames_with_ts[i-1]["frame"], frames_with_ts[i]["frame"]))
    avg_motion = float(np.mean(diffs))

    # 2. Avg Color Variance (sampled every 10th frame for speed if many)
    sampled_frames = frames_with_ts[::max(1, len(frames_with_ts)//20)] # Sample ~20 frames
    hists = []
    for f in sampled_frames:
        hist = cv2.calcHist([f["frame"]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hists.append(hist.flatten())
    avg_color_variance = float(np.mean(np.var(np.array(hists), axis=0))) if hists else 0.0

    # 3. Baseline aspect ratio (median of active region across sampled frames)
    ars = [compute_active_aspect_ratio(f["frame"]) for f in sampled_frames]
    ars = [a for a in ars if not np.isnan(a)]
    avg_aspect_ratio = float(np.median(ars)) if ars else float("nan")

    return {
        "avg_motion": avg_motion,
        "avg_color_variance": avg_color_variance,
        "avg_aspect_ratio": avg_aspect_ratio,
    }

