"""
Unit tests for src/process.py
"""

import os
import sys
import pytest

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.classification import classify_segments

def test_classify_segments_with_profile():
    global_profile = {
        "avg_centroid": 1000.0,
        "avg_bandwidth": 1000.0,
        "avg_motion": 5.0
    }
    segments = [
        {
            "start_seconds": 0.0, 
            "end_seconds": 10.0, 
            "duration_seconds": 10.0,
            "spectral_centroid": 1050.0, # Close to avg
            "spectral_bandwidth": 950.0,  # Close to avg
            "motion_score": 5.5          # Close to avg
        },
        {
            "start_seconds": 10.0, 
            "end_seconds": 15.0, 
            "duration_seconds": 5.0,     # Short + Outlier
            "spectral_centroid": 2500.0, # Far from avg
            "spectral_bandwidth": 2000.0, 
            "motion_score": 20.0
        },
    ]
    classified = classify_segments(segments, duration=15.0, global_profile=global_profile)
    
    assert len(classified) == 2
    assert classified[0]["is_content"] is True
    assert classified[1]["is_content"] is False
    assert classified[1]["segment_type"] == "ad"

def test_classify_segments_empty():
    assert classify_segments([], 0.0, {}) == []

def test_merge_consecutive_segments():
    # Mock segments of same type
    segments = [
        {"start_seconds": 0.0, "end_seconds": 2.0, "duration_seconds": 2.0, "segment_type": "video_content", "confidence": 0.9},
        {"start_seconds": 2.0, "end_seconds": 4.0, "duration_seconds": 2.0, "segment_type": "video_content", "confidence": 0.8},
        {"start_seconds": 4.0, "end_seconds": 6.0, "duration_seconds": 2.0, "segment_type": "ad", "confidence": 1.0},
        {"start_seconds": 6.0, "end_seconds": 8.0, "duration_seconds": 2.0, "segment_type": "ad", "confidence": 0.5},
        {"start_seconds": 8.0, "end_seconds": 10.0, "duration_seconds": 2.0, "segment_type": "video_content", "confidence": 1.0},
    ]

    
    from src.classification import merge_consecutive_segments
    merged = merge_consecutive_segments(segments)
    
    # Expected: 3 segments (content, ad, content)
    assert len(merged) == 3
    assert merged[0]["duration_seconds"] == 4.0
    assert merged[0]["segment_type"] == "video_content"
    assert merged[0]["confidence"] == 0.8  # Min confidence
    
    assert merged[1]["duration_seconds"] == 4.0
    assert merged[1]["segment_type"] == "ad"
    assert merged[1]["confidence"] == 0.5
    
    assert merged[2]["duration_seconds"] == 2.0
    assert merged[2]["segment_type"] == "video_content"


