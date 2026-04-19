import cv2
import numpy as np
import subprocess
import json
import logging
from typing import List, Tuple

def compute_hsv_metrics(frame: np.ndarray) -> Tuple[float, float]:
    """Computes median Saturation and Value (Brightness) in HSV space."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    med_s = np.median(hsv[:, :, 1])
    med_v = np.median(hsv[:, :, 2])
    return float(med_s), float(med_v)


def compute_letterbox_variance(frame: np.ndarray) -> float:
    """Computes the mean variance of the top and bottom 10% rows to detect letterboxing."""
    h, w = frame.shape[:2]
    border_h = max(1, int(h * 0.10))
    top_rows    = frame[:border_h, :, :]
    bottom_rows = frame[h - border_h:, :, :]
    top_var    = np.mean(np.var(top_rows,    axis=(0, 1)))
    bottom_var = np.mean(np.var(bottom_rows, axis=(0, 1)))
    return float((top_var + bottom_var) / 2.0)


def compute_spatial_edge_density(frame: np.ndarray) -> float:
    """
    Spatial Edge Density (Text & Graphics Metric).
    Converts to grayscale, applies Canny, returns fraction of edge pixels.
    Ads filled with on-screen text / motion-graphics score much higher than
    naturally-lit cinematic content.
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(np.count_nonzero(edges)) / float(edges.size)


def compute_corner_edge_density(frame: np.ndarray) -> float:
    """Computes Canny edge density in corners to detect static watermarks/bugs."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    h, w = edges.shape
    q = 0.15  # Corner quadrant size
    tl = np.mean(edges[:int(h * q), :int(w * q)])
    tr = np.mean(edges[:int(h * q), int(w * (1 - q)):])
    bl = np.mean(edges[int(h * (1 - q)):, :int(w * q)])
    br = np.mean(edges[int(h * (1 - q)):, int(w * (1 - q)):])
    return float((tl + tr + bl + br) / 4.0)

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
    """Returns the mean SAD motion score for a specific segment."""
    segment_frames = [f["frame"] for f in frames_with_ts if start_sec <= f["timestamp_seconds"] < end_sec]
    if len(segment_frames) < 2:
        return 0.0
    diffs = [compute_frame_diff(segment_frames[i - 1], segment_frames[i])
             for i in range(1, len(segment_frames))]
    return float(np.mean(diffs))


def analyze_motion_energy_variance_for_segment(
        frames_with_ts: List[dict], start_sec: float, end_sec: float) -> float:
    """
    Temporal Motion Energy Variance (Freneticism Metric).
    Returns the VARIANCE of per-frame SAD scores within the segment.
    High variance → fast-cutting, shaky-cam, kinetic ads.
    Low variance  → static camera, calm dialogue (typical content).
    """
    segment_frames = [f["frame"] for f in frames_with_ts if start_sec <= f["timestamp_seconds"] < end_sec]
    if len(segment_frames) < 3:
        return 0.0
    diffs = [compute_frame_diff(segment_frames[i - 1], segment_frames[i])
             for i in range(1, len(segment_frames))]
    return float(np.var(diffs))


def analyze_spatial_edge_density_for_segment(
        frames_with_ts: List[dict], start_sec: float, end_sec: float) -> float:
    """
    Returns the mean spatial edge density across frames in the segment.
    Ads filled with on-screen text / motion-graphics will score much higher
    than naturally-lit cinematic content.
    """
    segment_frames = [f["frame"] for f in frames_with_ts if start_sec <= f["timestamp_seconds"] < end_sec]
    if not segment_frames:
        return 0.0
    return float(np.mean([compute_spatial_edge_density(f) for f in segment_frames]))


# ---------------------------------------------------------------------------
# Frame-list API (MoviePy integration)
# These functions receive a plain List[np.ndarray] in BGR format, exactly as
# produced by iterating MoviePy's clip.iter_frames() and converting RGB→BGR.
# They are the primary API used by the new segmentation.py pipeline.
# ---------------------------------------------------------------------------

def _bgr_frames_from_moviepy(rgb_frames) -> List[np.ndarray]:
    """Convert a sequence of RGB numpy arrays (from MoviePy) to BGR for OpenCV."""
    return [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in rgb_frames]


def analyze_motion_from_frames(bgr_frames: List[np.ndarray]) -> float:
    """Mean SAD motion score across a list of BGR frames."""
    if len(bgr_frames) < 2:
        return 0.0
    diffs = [compute_frame_diff(bgr_frames[i - 1], bgr_frames[i])
             for i in range(1, len(bgr_frames))]
    return float(np.mean(diffs))


def analyze_motion_energy_variance_from_frames(bgr_frames: List[np.ndarray]) -> float:
    """Temporal Motion Energy Variance from a list of BGR frames."""
    if len(bgr_frames) < 3:
        return 0.0
    diffs = [compute_frame_diff(bgr_frames[i - 1], bgr_frames[i])
             for i in range(1, len(bgr_frames))]
    return float(np.var(diffs))


def analyze_spatial_edge_density_from_frames(bgr_frames: List[np.ndarray]) -> float:
    """Mean Canny edge density (Text & Graphics metric) from a list of BGR frames."""
    if not bgr_frames:
        return 0.0
    return float(np.mean([compute_spatial_edge_density(f) for f in bgr_frames]))


def analyze_color_variance_from_frames(bgr_frames: List[np.ndarray]) -> float:
    """Color histogram variance across a list of BGR frames."""
    if not bgr_frames:
        return 0.0
    hists = [cv2.calcHist([f], [0, 1, 2], None, [8, 8, 8],
                          [0, 256, 0, 256, 0, 256]).flatten()
             for f in bgr_frames]
    if len(hists) < 2:
        return 0.0
    return float(np.mean(np.var(np.array(hists), axis=0)))


def analyze_sharpness_from_frames(bgr_frames: List[np.ndarray]) -> float:
    """Mean Laplacian sharpness across a list of BGR frames."""
    if not bgr_frames:
        return 0.0
    return float(np.mean([compute_sharpness(f) for f in bgr_frames]))


def analyze_hsv_from_frames(bgr_frames: List[np.ndarray]) -> Tuple[float, float]:
    """Median saturation and luminance (HSV Color Volume) from a list of BGR frames."""
    if not bgr_frames:
        return 0.0, 0.0
    s_vals, v_vals = zip(*[compute_hsv_metrics(f) for f in bgr_frames])
    return float(np.mean(s_vals)), float(np.mean(v_vals))


def analyze_letterbox_from_frames(bgr_frames: List[np.ndarray]) -> float:
    """Mean letterbox variance from a list of BGR frames."""
    if not bgr_frames:
        return 0.0
    return float(np.mean([compute_letterbox_variance(f) for f in bgr_frames]))


def analyze_watermark_from_frames(bgr_frames: List[np.ndarray]) -> float:
    """Mean corner edge density (watermark detection) from a list of BGR frames."""
    if not bgr_frames:
        return 0.0
    return float(np.mean([compute_corner_edge_density(f) for f in bgr_frames]))


def analyze_solid_color_from_frames(bgr_frames: List[np.ndarray]) -> bool:
    """True if >90% of frames are near-solid black or white."""
    if not bgr_frames:
        return False
    solid = sum(
        1 for f in bgr_frames
        if np.mean(f) < 5.0 or np.mean(f) > 250.0
    )
    return (solid / len(bgr_frames)) > 0.9


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

def compute_sharpness(frame) -> float:
    """
    Computes the focus measure (sharpness) using the variance of the Laplacian.
    Higher values mean more sharpness/focus.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def analyze_sharpness_for_segment(frames_with_ts: List[dict], start_sec: float, end_sec: float) -> float:
    """
    Computes average sharpness for frames in a segment.
    """
    segment_frames = [f["frame"] for f in frames_with_ts if start_sec <= f["timestamp_seconds"] < end_sec]
    if not segment_frames:
        return 0.0
    
    sharpness_values = [compute_sharpness(f) for f in segment_frames]
    return float(np.mean(sharpness_values))

def analyze_hsv_for_segment(frames_with_ts: List[dict], start_sec: float, end_sec: float) -> Tuple[float, float]:
    segment_frames = [f["frame"] for f in frames_with_ts if start_sec <= f["timestamp_seconds"] < end_sec]
    if not segment_frames: return 0.0, 0.0
    
    s_vals, v_vals = zip(*[compute_hsv_metrics(f) for f in segment_frames])
    return float(np.mean(s_vals)), float(np.mean(v_vals))

def analyze_letterbox_for_segment(frames_with_ts: List[dict], start_sec: float, end_sec: float) -> float:
    segment_frames = [f["frame"] for f in frames_with_ts if start_sec <= f["timestamp_seconds"] < end_sec]
    if not segment_frames: return 0.0
    return float(np.mean([compute_letterbox_variance(f) for f in segment_frames]))

def analyze_watermark_for_segment(frames_with_ts: List[dict], start_sec: float, end_sec: float) -> float:
    segment_frames = [f["frame"] for f in frames_with_ts if start_sec <= f["timestamp_seconds"] < end_sec]
    if not segment_frames: return 0.0
    return float(np.mean([compute_corner_edge_density(f) for f in segment_frames]))

def analyze_solid_color_for_segment(frames_with_ts: List[dict], start_sec: float, end_sec: float) -> bool:
    """
    Returns True if more than 90% of frames in the segment are almost solid black or white.
    """
    segment_frames = [f["frame"] for f in frames_with_ts if start_sec <= f["timestamp_seconds"] < end_sec]
    if not segment_frames:
        return False
    
    solid_count = 0
    for frame in segment_frames:
        avg_bgr = np.mean(frame, axis=(0, 1))
        brightness = np.mean(avg_bgr)
        # Black: brightness < 5, White: brightness > 250
        if brightness < 5.0 or brightness > 250.0:
            solid_count += 1
            
    return (solid_count / len(segment_frames)) > 0.9

def compute_global_visual_profile(frames_with_ts: List[dict]) -> dict:
    """
    Computes the 'Universal Visual Profile' (Mean and Std Dev) for the entire video.
    """
    if len(frames_with_ts) < 2:
        return {
            "avg_motion": 0.0, "std_motion": 0.0,
            "avg_color_variance": 0.0, "std_color_variance": 0.0,
            "avg_sharpness": 0.0, "std_sharpness": 0.0
        }
    
    # 1. Motion Distribution
    diffs = []
    for i in range(1, len(frames_with_ts)):
        diffs.append(compute_frame_diff(frames_with_ts[i-1]["frame"], frames_with_ts[i]["frame"]))
    avg_motion = float(np.mean(diffs))
    std_motion = float(np.std(diffs))
    
    # 2. Color Variance Distribution
    sampled_frames = frames_with_ts[::max(1, len(frames_with_ts)//20)]
    hists = []
    for f in sampled_frames:
        hist = cv2.calcHist([f["frame"]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hists.append(hist.flatten())
    
    avg_color_var = 0.0
    std_color_var = 0.0
    if hists:
        vars_hist = np.var(np.array(hists), axis=0)
        avg_color_var = float(np.mean(vars_hist))
        std_color_var = float(np.std(vars_hist))

    # 3. Sharpness Distribution
    sharpness_scores = [compute_sharpness(f["frame"]) for f in sampled_frames]
    avg_sharpness = float(np.mean(sharpness_scores)) if sharpness_scores else 0.0
    std_sharpness = float(np.std(sharpness_scores)) if sharpness_scores else 0.0
    
    # 4. HSV Distribution
    hsv_data = [compute_hsv_metrics(f["frame"]) for f in sampled_frames]
    avg_sat = float(np.mean([x[0] for x in hsv_data])) if hsv_data else 0.0
    std_sat = float(np.std([x[0] for x in hsv_data])) if hsv_data else 0.0
    avg_val = float(np.mean([x[1] for x in hsv_data])) if hsv_data else 0.0
    std_val = float(np.std([x[1] for x in hsv_data])) if hsv_data else 0.0

    # 5. Letterbox Distribution
    lb_scores = [compute_letterbox_variance(f["frame"]) for f in sampled_frames]
    avg_lb = float(np.mean(lb_scores)) if lb_scores else 0.0
    std_lb = float(np.std(lb_scores)) if lb_scores else 0.0

    # 6. Watermark Distribution
    wm_scores = [compute_corner_edge_density(f["frame"]) for f in sampled_frames]
    avg_wm = float(np.mean(wm_scores)) if wm_scores else 0.0
    std_wm = float(np.std(wm_scores)) if wm_scores else 0.0

    return {
        "avg_motion": avg_motion, "std_motion": std_motion,
        "avg_color_variance": avg_color_var, "std_color_variance": std_color_var,
        "avg_sharpness": avg_sharpness, "std_sharpness": std_sharpness,
        "avg_saturation": avg_sat, "std_saturation": std_sat,
        "avg_luminance": avg_val, "std_luminance": std_val,
        "avg_letterbox": avg_lb, "std_letterbox": std_lb,
        "avg_watermark": avg_wm, "std_watermark": std_wm
    }
