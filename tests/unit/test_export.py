"""
Unit tests for src/output_generator.py
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.export import build_output
from src.ingest import seconds_to_formatted


def make_classified_segments() -> list[dict]:
    """Minimal classified segment list: intro, content, ad, content, outro."""
    base = {
        "mean_rms": 0.1, "silence_ratio": 0.1, "music_ratio": 0.0,
        "is_mostly_silent": False, "is_mostly_music": False,
        "mean_brightness": 100.0, "mean_color_variance": 20.0,
        "black_frame_ratio": 0.0, "is_static_visual": False,
        "mean_motion": 10.0, "is_low_motion": False, "has_scene_cut": False,
        "speech_ratio": 0.7, "has_no_speech": False, "is_repeat_phrase": False,
        "speech_type_hints": [], "transcript_text": "",
    }

    return [
        {**base, "start_seconds": 0.0,   "end_seconds": 30.0,   "duration_seconds": 30.0,
         "position_ratio": 0.0,  "segment_type": "intro",   "is_content": False},
        {**base, "start_seconds": 30.0,  "end_seconds": 400.0,  "duration_seconds": 370.0,
         "position_ratio": 0.1,  "segment_type": "content", "is_content": True},
        {**base, "start_seconds": 400.0, "end_seconds": 430.0,  "duration_seconds": 30.0,
         "position_ratio": 0.5,  "segment_type": "ad",      "is_content": False},
        {**base, "start_seconds": 430.0, "end_seconds": 900.0,  "duration_seconds": 470.0,
         "position_ratio": 0.55, "segment_type": "content", "is_content": True},
        {**base, "start_seconds": 900.0, "end_seconds": 950.0,  "duration_seconds": 50.0,
         "position_ratio": 0.95, "segment_type": "outro",   "is_content": False},
    ]


VIDEO_META = {
    "duration_seconds": 950.0,
    "resolution": "1280x720",
    "width": 1280,
    "height": 720,
    "fps": 30.0,
    "has_audio": True,
    "video_codec": "h264",
    "audio_codec": "aac",
}


class TestBuildOutput:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.segments = make_classified_segments()
        self.result = build_output("test_video.mp4", VIDEO_META, self.segments)

    def test_returns_dict(self):
        assert isinstance(self.result, dict)

    def test_required_top_level_keys(self):
        required = [
            "video_filename", "original_video_duration_seconds",
            "original_video_duration_formatted", "original_video_resolution",
            "timeline_segments", "inserted_ads",
            "num_ads_inserted", "total_ads_duration_seconds",
        ]
        for key in required:
            assert key in self.result, f"Missing key: {key}"

    def test_video_filename(self):
        assert self.result["video_filename"] == "test_video.mp4"

    def test_resolution(self):
        assert self.result["original_video_resolution"] == "1280x720"

    def test_duration(self):
        # 950.0 output - 110.0 ads = 840.0 original
        assert self.result["original_video_duration_seconds"] == pytest.approx(840.0, abs=0.1)
        assert self.result["output_duration_seconds"] == pytest.approx(950.0, abs=0.1)

    def test_duration_formatted(self):
        fmt = self.result["original_video_duration_formatted"]
        assert ":" in fmt  # HH:MM:SS.mmm

    def test_timeline_segment_count(self):
        # 5 segments total
        assert len(self.result["timeline_segments"]) == 5

    def test_non_content_count(self):
        # intro + ad + outro = 3 non-content
        assert self.result["num_ads_inserted"] == 3

    def test_total_non_content_duration(self):
        expected = 30.0 + 30.0 + 50.0  # intro + ad + outro
        assert self.result["total_ads_duration_seconds"] == pytest.approx(expected, abs=0.1)

    def test_timeline_coverage(self):
        """Verify timeline segments cover the full video monotonically."""
        segs = self.result["timeline_segments"]
        # Check monotonic start times
        starts = [s["final_video_start_seconds"] for s in segs]
        for i in range(1, len(starts)):
            assert starts[i] >= starts[i - 1], f"Non-monotonic at index {i}"

    def test_timeline_segment_types(self):
        valid_types = {"video_content", "ad"}
        for s in self.result["timeline_segments"]:
            assert s["type"] in valid_types

    def test_content_segments_have_original_timestamps(self):
        content_segs = [s for s in self.result["timeline_segments"] if s["type"] == "video_content"]
        for s in content_segs:
            assert "original_video_start_seconds" in s
            assert "original_video_end_seconds" in s

    # ── Non-content validation ────────────────────────────────────────────────

    def test_non_content_is_list(self):
        assert isinstance(self.result["inserted_ads"], list)

    def test_num_nc_matches_list_length(self):
        assert self.result["num_ads_inserted"] == len(
            self.result["inserted_ads"]
        )

    def test_nc_segment_fields(self):
        for seg in self.result["inserted_ads"]:
            assert "ad_index" in seg
            assert "ad_duration_seconds" in seg
