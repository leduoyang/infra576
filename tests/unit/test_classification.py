"""
Unit tests for src/classification.py – intro/outro-only segment classifier.

The current classifier only detects: intro, outro, content.
It does NOT detect ad, dead_air, transition, self_promotion, or recap.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.classification import (
    classify_segments,
    merge_consecutive_segments,
    CONTENT, INTRO, OUTRO,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_seg(start, end, **overrides):
    """Build a minimal segment dict."""
    seg = {
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": end - start,
    }
    seg.update(overrides)
    return seg


GLOBAL_PROFILE = {
    "avg_centroid": 1000.0,
    "avg_bandwidth": 1000.0,
    "avg_energy": 0.05,
    "avg_motion": 5.0,
    "std_motion": 3.0,
    "avg_flatness": 0.10,
    "avg_edge_density": 0.05,
    "avg_color_variance": 0.01,
    "avg_zcr": 0.05,
}


# ── Intro Detection ─────────────────────────────────────────────────────────

class TestIntroDetection:
    def test_intro_silent_black_card_at_start(self):
        """A silent black card at position 0 should be classified as intro."""
        segments = [
            _make_seg(0.0, 8.0,
                      spectral_centroid=1500.0,
                      spectral_bandwidth=1500.0,
                      audio_energy=0.001,
                      motion_score=0.2,
                      silence_ratio=0.95,
                      spectral_flatness=0.20,
                      edge_density=0.08,
                      frame_self_similarity=0.97,
                      color_variance=0.001,
                      word_count=0,
                      words_per_second=0.0,
                      transcript_text="",
                      black_border_ratio=0.85,
                      active_width_ratio=0.5,
                      active_height_ratio=0.5),
            _make_seg(8.0, 300.0,
                      spectral_centroid=1000.0,
                      spectral_bandwidth=1000.0,
                      audio_energy=0.05,
                      motion_score=5.0,
                      word_count=50,
                      words_per_second=2.5,
                      transcript_text="today we discuss important things"),
        ]
        result = classify_segments(segments, duration=300.0, global_profile=GLOBAL_PROFILE)
        assert result[0]["segment_type"] == INTRO
        assert result[0]["is_content"] is False

    def test_speech_at_start_stays_content(self):
        """A segment with speech at start should stay as content."""
        segments = [
            _make_seg(0.0, 20.0,
                      spectral_centroid=1000.0,
                      spectral_bandwidth=1000.0,
                      audio_energy=0.05,
                      motion_score=5.0,
                      silence_ratio=0.0,
                      spectral_flatness=0.10,
                      edge_density=0.05,
                      frame_self_similarity=0.5,
                      word_count=30,
                      words_per_second=2.0,
                      transcript_text="hello everyone welcome to the show"),
        ]
        result = classify_segments(segments, duration=200.0, global_profile=GLOBAL_PROFILE)
        assert result[0]["segment_type"] == CONTENT

    def test_intro_too_long_stays_content(self):
        """A segment > 4% of video at start should not be labeled intro."""
        segments = [
            _make_seg(0.0, 150.0,
                      spectral_centroid=1500.0,
                      spectral_bandwidth=1500.0,
                      audio_energy=0.001,
                      motion_score=0.2,
                      silence_ratio=0.95,
                      spectral_flatness=0.20,
                      edge_density=0.10,
                      frame_self_similarity=0.95,
                      word_count=0,
                      words_per_second=0.0,
                      transcript_text=""),
            _make_seg(150.0, 300.0),
        ]
        result = classify_segments(segments, duration=300.0, global_profile=GLOBAL_PROFILE)
        assert result[0]["segment_type"] != INTRO


# ── Outro Detection ──────────────────────────────────────────────────────────

class TestOutroDetection:
    def test_outro_silent_black_card_at_end(self):
        """A silent black card at the end should be classified as outro."""
        segments = [
            _make_seg(0.0, 290.0,
                      spectral_centroid=1000.0,
                      spectral_bandwidth=1000.0,
                      audio_energy=0.05,
                      motion_score=5.0,
                      word_count=100,
                      words_per_second=2.5,
                      transcript_text="content content content"),
            _make_seg(290.0, 300.0,
                      spectral_centroid=1400.0,
                      spectral_bandwidth=1400.0,
                      audio_energy=0.001,
                      motion_score=0.2,
                      silence_ratio=0.95,
                      spectral_flatness=0.18,
                      edge_density=0.08,
                      frame_self_similarity=0.97,
                      color_variance=0.001,
                      word_count=0,
                      words_per_second=0.0,
                      transcript_text="",
                      black_border_ratio=0.85,
                      active_width_ratio=0.5,
                      active_height_ratio=0.5),
        ]
        result = classify_segments(segments, duration=300.0, global_profile=GLOBAL_PROFILE)
        assert result[-1]["segment_type"] == OUTRO
        assert result[-1]["is_content"] is False

    def test_content_at_end_not_misclassified(self):
        """Normal content at the end should remain content."""
        segments = [
            _make_seg(0.0, 280.0),
            _make_seg(280.0, 300.0,
                      spectral_centroid=1000.0,
                      spectral_bandwidth=1000.0,
                      audio_energy=0.05,
                      motion_score=5.0,
                      silence_ratio=0.0,
                      spectral_flatness=0.10,
                      edge_density=0.05,
                      frame_self_similarity=0.5,
                      word_count=20,
                      words_per_second=2.0,
                      transcript_text="thanks for watching bye"),
        ]
        result = classify_segments(segments, duration=300.0, global_profile=GLOBAL_PROFILE)
        assert result[-1]["segment_type"] == CONTENT

    def test_animation_with_audio_stays_content(self):
        """Colorful animation with active audio near end should not be outro."""
        segments = [
            _make_seg(0.0, 280.0,
                      spectral_centroid=1000.0,
                      spectral_bandwidth=1000.0,
                      audio_energy=0.05,
                      motion_score=5.0),
            _make_seg(280.0, 300.0,
                      spectral_centroid=1100.0,
                      spectral_bandwidth=1100.0,
                      audio_energy=0.06,
                      motion_score=3.0,
                      silence_ratio=0.1,
                      spectral_flatness=0.12,
                      edge_density=0.06,
                      frame_self_similarity=0.85,
                      color_variance=0.02,
                      word_count=0,
                      words_per_second=0.0,
                      transcript_text=""),
        ]
        result = classify_segments(segments, duration=300.0, global_profile=GLOBAL_PROFILE)
        # Active audio + non-black visuals = content, not outro
        assert result[-1]["segment_type"] == CONTENT


# ── Merge Logic ──────────────────────────────────────────────────────────────

class TestMerge:
    def test_merge_consecutive_same_type(self):
        segments = [
            {"start_seconds": 0.0, "end_seconds": 2.0, "duration_seconds": 2.0,
             "segment_type": CONTENT, "is_content": True, "confidence": 0.9},
            {"start_seconds": 2.0, "end_seconds": 4.0, "duration_seconds": 2.0,
             "segment_type": CONTENT, "is_content": True, "confidence": 0.8},
            {"start_seconds": 4.0, "end_seconds": 6.0, "duration_seconds": 2.0,
             "segment_type": INTRO, "is_content": False, "confidence": 1.0},
        ]
        merged = merge_consecutive_segments(segments)
        assert len(merged) == 2
        assert merged[0]["duration_seconds"] == 4.0
        assert merged[0]["confidence"] == 0.8

    def test_empty(self):
        assert classify_segments([], 0.0, {}) == []

    def test_merge_empty(self):
        assert merge_consecutive_segments([]) == []
