"""
Unit tests for src/export.py – backward-compatible + new taxonomy.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.export import build_output
from src.ingest import seconds_to_formatted


def make_classified_segments() -> list[dict]:
    """Minimal classified segment list: intro, content, ad, content, outro."""
    return [
        {"start_seconds": 0.0, "end_seconds": 30.0, "duration_seconds": 30.0,
         "segment_type": "intro", "segment_label": "intro", "is_content": False, "confidence": 0.85},
        {"start_seconds": 30.0, "end_seconds": 400.0, "duration_seconds": 370.0,
         "segment_type": "content", "segment_label": "content", "is_content": True, "confidence": 0.95},
        {"start_seconds": 400.0, "end_seconds": 430.0, "duration_seconds": 30.0,
         "segment_type": "ad", "segment_label": "ad", "is_content": False, "confidence": 0.90},
        {"start_seconds": 430.0, "end_seconds": 900.0, "duration_seconds": 470.0,
         "segment_type": "content", "segment_label": "content", "is_content": True, "confidence": 0.92},
        {"start_seconds": 900.0, "end_seconds": 950.0, "duration_seconds": 50.0,
         "segment_type": "outro", "segment_label": "outro", "is_content": False, "confidence": 0.80},
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

    def test_required_legacy_keys(self):
        required = [
            "video_filename", "original_video_duration_seconds",
            "original_video_duration_formatted", "original_video_resolution",
            "timeline_segments", "inserted_ads",
            "num_ads_inserted", "total_ads_duration_seconds",
        ]
        for key in required:
            assert key in self.result, f"Missing key: {key}"

    def test_required_new_keys(self):
        new_keys = [
            "non_content_summary", "chapters",
            "content_duration_seconds", "total_noncontent_duration_seconds",
            "num_segments",
        ]
        for key in new_keys:
            assert key in self.result, f"Missing key: {key}"

    def test_video_filename(self):
        assert self.result["video_filename"] == "test_video.mp4"

    def test_resolution(self):
        assert self.result["original_video_resolution"] == "1280x720"

    def test_duration(self):
        nc_dur = 30.0 + 30.0 + 50.0  # intro + ad + outro
        assert self.result["output_duration_seconds"] == pytest.approx(950.0, abs=0.1)
        assert self.result["original_video_duration_seconds"] == pytest.approx(950.0 - nc_dur, abs=0.1)

    def test_timeline_segment_count(self):
        assert len(self.result["timeline_segments"]) == 5

    def test_non_content_count(self):
        assert self.result["num_ads_inserted"] == 3

    def test_total_non_content_duration(self):
        expected = 30.0 + 30.0 + 50.0
        assert self.result["total_ads_duration_seconds"] == pytest.approx(expected, abs=0.1)

    def test_timeline_coverage(self):
        segs = self.result["timeline_segments"]
        starts = [s["final_video_start_seconds"] for s in segs]
        for i in range(1, len(starts)):
            assert starts[i] >= starts[i - 1]

    def test_content_segments_have_original_timestamps(self):
        content_segs = [s for s in self.result["timeline_segments"] if s["type"] == "video_content"]
        for s in content_segs:
            assert "original_video_start_seconds" in s
            assert "original_video_end_seconds" in s

    def test_segment_labels_present(self):
        for s in self.result["timeline_segments"]:
            assert "segment_label" in s
            assert "is_content" in s
            assert "confidence" in s

    def test_skip_suggestions(self):
        tl = self.result["timeline_segments"]
        # Non-content should have skip_suggestion=True
        for s in tl:
            if not s["is_content"]:
                assert s["skip_suggestion"] is True
                assert "skip_reason" in s
            else:
                assert s["skip_suggestion"] is False

    def test_chapters_count(self):
        assert len(self.result["chapters"]) == 5

    def test_chapters_titles(self):
        chapters = self.result["chapters"]
        assert chapters[0]["title"] == "Intro"
        assert chapters[1]["title"] == "Content part 1"
        assert chapters[2]["title"] == "Ad break"
        assert chapters[3]["title"] == "Content part 2"
        assert chapters[4]["title"] == "Outro"

    def test_noncontent_summary(self):
        summary = self.result["non_content_summary"]
        assert "intro" in summary
        assert summary["intro"]["count"] == 1
        assert "ad" in summary
        assert summary["ad"]["count"] == 1
        assert "outro" in summary
        assert summary["outro"]["count"] == 1

    def test_inserted_ads_fields(self):
        for seg in self.result["inserted_ads"]:
            assert "ad_index" in seg
            assert "ad_duration_seconds" in seg
            assert "ad_type" in seg
            assert "final_video_ad_start_seconds" in seg
            assert "final_video_ad_end_seconds" in seg
