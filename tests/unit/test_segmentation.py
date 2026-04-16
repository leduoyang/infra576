"""
Unit tests for src/preprocessing.py
"""

import os
import sys
import numpy as np
import pytest
import cv2

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.segmentation import (
    merge_short_scenes,
    split_long_scenes,
    run_segmentation_pipeline
)

from src.features.visual import (
    compute_frame_diff,
    analyze_motion_for_segment
)

# Mocking scenedetect since it's hard to test without ffmpeg/real video in a unit test
# or I can use the same technique as test_read.py for creating dummy videos if needed.
# But for merge/split/motion, I can test logic with mock data.

def test_merge_short_scenes():
    scenes = [
        {"start_seconds": 0.0, "end_seconds": 1.0, "duration_seconds": 1.0},
        {"start_seconds": 1.0, "end_seconds": 5.0, "duration_seconds": 4.0},
        {"start_seconds": 5.0, "end_seconds": 5.5, "duration_seconds": 0.5},
        {"start_seconds": 5.5, "end_seconds": 10.0, "duration_seconds": 4.5},
    ]
    # min_duration=2.0
    merged = merge_short_scenes(scenes, min_duration=2.0)
    
    # Scene 0 (1s) should merge into Scene 1
    # Resulting Scene 0: 0.0 to 5.0 (5s)
    # Scene 2 (0.5s) should merge into Scene 3
    # Resulting Scene 1: 5.0 to 10.0 (5s)
    
    assert len(merged) == 2
    assert merged[0]["start_seconds"] == 0.0
    assert merged[0]["end_seconds"] == 5.5
    assert merged[0]["duration_seconds"] == 5.5
    assert merged[1]["start_seconds"] == 5.5
    assert merged[1]["end_seconds"] == 10.0
    assert merged[1]["duration_seconds"] == 4.5

def test_merge_short_scenes_empty():
    assert merge_short_scenes([]) == []

def test_split_long_scenes():
    scenes = [
        {"start_seconds": 0.0, "end_seconds": 10.0, "duration_seconds": 10.0},
    ]
    # max_duration=4.0
    # 10 / 4 = 2.5 -> ceil(2.5) = 3 splits
    # 10 / 3 = 3.333 per split
    splits = split_long_scenes(scenes, max_duration=4.0)
    
    assert len(splits) == 3
    for s in splits:
        assert s["duration_seconds"] == pytest.approx(3.333, abs=0.01)

def test_compute_frame_diff():
    f1 = np.zeros((100, 100, 3), dtype=np.uint8)
    f2 = np.ones((100, 100, 3), dtype=np.uint8) * 255 # White
    
    # Gray conversion: 0 -> 0, 255 -> 255
    # diff = 255, mean(255) = 255
    diff = compute_frame_diff(f1, f2)
    assert diff == pytest.approx(255.0)

def test_analyze_motion_for_segment():
    # Create mock frames
    f1 = np.zeros((100, 100, 3), dtype=np.uint8)
    f2 = np.ones((100, 100, 3), dtype=np.uint8) * 10
    f3 = np.ones((100, 100, 3), dtype=np.uint8) * 20
    
    frames_with_ts = [
        {"timestamp_seconds": 0.0, "frame": f1},
        {"timestamp_seconds": 0.5, "frame": f2},
        {"timestamp_seconds": 1.0, "frame": f3},
        {"timestamp_seconds": 2.0, "frame": f1}, # Outside 0-1 range
    ]
    
    # Segment 0.0 to 1.1
    motion = analyze_motion_for_segment(frames_with_ts, 0.0, 1.1)
    # Frames included: f1, f2, f3
    # Diff(f1, f2) = 10
    # Diff(f2, f3) = 10
    # Mean = 10
    
    assert motion == pytest.approx(10.0)

def test_analyze_motion_no_frames():
    assert analyze_motion_for_segment([], 0.0, 1.0) == 0.0

def test_run_segmentation_pipeline_return_type(tmp_path):
    # This is a structural test. We mock the dependencies to check the return type.
    # In a real environment, we'd use a small dummy video.
    # For now, we just want to ensure that if we ever mock it, we expect a tuple.
    import unittest.mock as mock
    
    with mock.patch('src.segmentation.extract_audio') as m_audio, \
         mock.patch('src.segmentation.extract_frames') as m_frames, \
         mock.patch('src.segmentation.librosa.load') as m_load, \
         mock.patch('src.segmentation.compute_global_visual_profile') as m_gv, \
         mock.patch('src.segmentation.compute_global_audio_profile') as m_ga, \
         mock.patch('src.segmentation.detect_scenes_scenedetect') as m_ds:
        
        m_load.return_value = (np.zeros(16000), 16000)
        m_frames.return_value = []
        m_gv.return_value = {"avg_motion": 0.0}
        m_ga.return_value = {"avg_centroid": 0.0}
        m_ds.return_value = [{"start_seconds": 0.0, "end_seconds": 1.0, "duration_seconds": 1.0}]
        
        segments, profile = run_segmentation_pipeline("fake.mp4", {"audio_start_time": 0.0})
        
        assert isinstance(segments, list)
        assert isinstance(profile, dict)
        assert len(segments) == 1
        assert "avg_motion" in profile

