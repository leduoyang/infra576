import numpy as np
import pytest
import cv2
from src.features.visual import (
    compute_frame_diff, 
    analyze_motion_for_segment, 
    analyze_color_variance,
    detect_scenes_scenedetect,
    compute_global_visual_profile
)


def test_compute_frame_diff_identical():
    f = np.zeros((100, 100, 3), dtype=np.uint8)
    assert compute_frame_diff(f, f) == 0.0

def test_analyze_color_variance_static():
    # Identical frames should have 0 variance in their histograms
    f = np.zeros((100, 100, 3), dtype=np.uint8)
    frames = [
        {"timestamp_seconds": 1.0, "frame": f},
        {"timestamp_seconds": 2.0, "frame": f}
    ]
    assert analyze_color_variance(frames, 0.0, 3.0) == 0.0

def test_analyze_color_variance_dynamic():
    # Different frames should have non-zero variance
    f1 = np.zeros((100, 100, 3), dtype=np.uint8)
    f2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    frames = [
        {"timestamp_seconds": 1.0, "frame": f1},
        {"timestamp_seconds": 2.0, "frame": f2}
    ]
    var = analyze_color_variance(frames, 0.0, 3.0)
    assert var > 0.0

def test_analyze_motion_basic():
    f1 = np.zeros((10, 10, 3), dtype=np.uint8)
    f2 = np.ones((10, 10, 3), dtype=np.uint8) * 10
    frames = [
        {"timestamp_seconds": 0.0, "frame": f1},
        {"timestamp_seconds": 1.0, "frame": f2}
    ]
    # mean(abs(10-0)) = 10
    assert analyze_motion_for_segment(frames, 0.0, 2.0) == pytest.approx(10.0)

def test_detect_scenes_scenedetect_fallback():
    scenes = detect_scenes_scenedetect("invalid_path.mp4")
    assert isinstance(scenes, list)
    assert len(scenes) >= 1

def test_compute_global_visual_profile():
    f1 = np.zeros((100, 100, 3), dtype=np.uint8)
    f2 = np.ones((100, 100, 3), dtype=np.uint8) * 10
    frames = [
        {"timestamp_seconds": 0.0, "frame": f1},
        {"timestamp_seconds": 1.0, "frame": f2},
        {"timestamp_seconds": 2.0, "frame": f1},
    ]
    profile = compute_global_visual_profile(frames)
    assert "avg_motion" in profile
    assert "avg_color_variance" in profile
    assert profile["avg_motion"] == pytest.approx(10.0)


