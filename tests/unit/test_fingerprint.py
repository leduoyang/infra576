"""Unit tests for src/features/fingerprint.py – audio fingerprinting."""

import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.features.fingerprint import (
    compute_audio_fingerprint,
    fingerprint_similarity,
    find_repeated_segments,
)


class TestFingerprintSimilarity:
    def test_identical(self):
        fp = np.array([1.0, 2.0, 3.0, 4.0])
        assert fingerprint_similarity(fp, fp) == 1.0

    def test_orthogonal(self):
        fp1 = np.array([1.0, 0.0])
        fp2 = np.array([0.0, 1.0])
        assert abs(fingerprint_similarity(fp1, fp2)) < 0.01

    def test_zero_vector(self):
        fp1 = np.zeros(4)
        fp2 = np.array([1.0, 2.0, 3.0, 4.0])
        assert fingerprint_similarity(fp1, fp2) == 0.0


class TestFindRepeatedSegments:
    def test_finds_cross_episode_repeats(self):
        shared_fp = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        unique_fp1 = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        unique_fp2 = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])

        episodes = [
            {
                "video_filename": "ep01.mp4",
                "segments": [
                    {"start_seconds": 0, "end_seconds": 10, "fingerprint": shared_fp, "segment_label": "intro"},
                    {"start_seconds": 10, "end_seconds": 200, "fingerprint": unique_fp1, "segment_label": "content"},
                ],
            },
            {
                "video_filename": "ep02.mp4",
                "segments": [
                    {"start_seconds": 0, "end_seconds": 10, "fingerprint": shared_fp.copy(), "segment_label": "intro"},
                    {"start_seconds": 10, "end_seconds": 200, "fingerprint": unique_fp2, "segment_label": "content"},
                ],
            },
        ]

        repeats = find_repeated_segments(episodes, similarity_threshold=0.90)
        assert len(repeats) >= 1
        assert repeats[0]["num_episodes"] == 2

    def test_no_repeats_different_content(self):
        episodes = [
            {
                "video_filename": "ep01.mp4",
                "segments": [
                    {"start_seconds": 0, "end_seconds": 10, "fingerprint": np.array([1.0, 0, 0, 0])},
                ],
            },
            {
                "video_filename": "ep02.mp4",
                "segments": [
                    {"start_seconds": 0, "end_seconds": 10, "fingerprint": np.array([0, 0, 0, 1.0])},
                ],
            },
        ]

        repeats = find_repeated_segments(episodes, similarity_threshold=0.90)
        assert len(repeats) == 0

    def test_same_video_not_matched(self):
        """Segments within the same video should NOT match."""
        fp = np.array([1.0, 2.0, 3.0])
        episodes = [
            {
                "video_filename": "ep01.mp4",
                "segments": [
                    {"start_seconds": 0, "end_seconds": 10, "fingerprint": fp},
                    {"start_seconds": 100, "end_seconds": 110, "fingerprint": fp.copy()},
                ],
            },
        ]
        repeats = find_repeated_segments(episodes, similarity_threshold=0.90)
        assert len(repeats) == 0

    def test_empty_input(self):
        assert find_repeated_segments([]) == []
