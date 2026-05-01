"""Unit tests for src/features/visual.py"""

import numpy as np
import pytest
import cv2
from src.features.visual import (
    compute_frame_diff,
    analyze_motion_for_segment,
    analyze_color_variance,
    analyze_edge_density,
    analyze_edge_density_for_segment,
    analyze_frame_self_similarity,
    detect_scenes_scenedetect,
    compute_global_visual_profile,
)


def test_compute_frame_diff_identical():
    f = np.zeros((100, 100, 3), dtype=np.uint8)
    assert compute_frame_diff(f, f) == 0.0


def test_analyze_color_variance_static():
    f = np.zeros((100, 100, 3), dtype=np.uint8)
    frames = [
        {"timestamp_seconds": 1.0, "frame": f},
        {"timestamp_seconds": 2.0, "frame": f},
    ]
    assert analyze_color_variance(frames, 0.0, 3.0) == 0.0


def test_analyze_color_variance_dynamic():
    f1 = np.zeros((100, 100, 3), dtype=np.uint8)
    f2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    frames = [
        {"timestamp_seconds": 1.0, "frame": f1},
        {"timestamp_seconds": 2.0, "frame": f2},
    ]
    assert analyze_color_variance(frames, 0.0, 3.0) > 0.0


def test_analyze_motion_basic():
    f1 = np.zeros((10, 10, 3), dtype=np.uint8)
    f2 = np.ones((10, 10, 3), dtype=np.uint8) * 10
    frames = [
        {"timestamp_seconds": 0.0, "frame": f1},
        {"timestamp_seconds": 1.0, "frame": f2},
    ]
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
    assert "avg_edge_density" in profile
    assert "std_motion" in profile
    assert profile["avg_motion"] == pytest.approx(10.0)


# ── New feature tests ────────────────────────────────────────────────────────

class TestEdgeDensity:
    def test_blank_frame_low_density(self):
        f = np.zeros((100, 100, 3), dtype=np.uint8)
        assert analyze_edge_density(f) == 0.0

    def test_edges_detected(self):
        f = np.zeros((100, 100, 3), dtype=np.uint8)
        # Draw a white rectangle to create edges
        cv2.rectangle(f, (20, 20), (80, 80), (255, 255, 255), 2)
        density = analyze_edge_density(f)
        assert density > 0.0

    def test_segment_edge_density(self):
        f = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(f, (10, 10), (90, 90), (255, 255, 255), 2)
        frames = [
            {"timestamp_seconds": 0.0, "frame": f},
            {"timestamp_seconds": 1.0, "frame": f},
        ]
        density = analyze_edge_density_for_segment(frames, 0.0, 2.0)
        assert density > 0.0

    def test_empty_segment(self):
        assert analyze_edge_density_for_segment([], 0.0, 1.0) == 0.0


class TestFrameSelfSimilarity:
    def test_identical_frames_high_similarity(self):
        f = np.ones((100, 100, 3), dtype=np.uint8) * 128
        frames = [
            {"timestamp_seconds": 0.0, "frame": f},
            {"timestamp_seconds": 1.0, "frame": f},
            {"timestamp_seconds": 2.0, "frame": f},
        ]
        sim = analyze_frame_self_similarity(frames, 0.0, 3.0)
        assert sim > 0.99

    def test_different_frames_lower_similarity(self):
        f1 = np.zeros((100, 100, 3), dtype=np.uint8)
        f2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        frames = [
            {"timestamp_seconds": 0.0, "frame": f1},
            {"timestamp_seconds": 1.0, "frame": f2},
        ]
        sim = analyze_frame_self_similarity(frames, 0.0, 2.0)
        assert sim < 0.5

    def test_single_frame_max_similarity(self):
        f = np.zeros((100, 100, 3), dtype=np.uint8)
        frames = [{"timestamp_seconds": 0.0, "frame": f}]
        assert analyze_frame_self_similarity(frames, 0.0, 1.0) == 1.0

    def test_empty_segment(self):
        assert analyze_frame_self_similarity([], 0.0, 1.0) == 1.0
