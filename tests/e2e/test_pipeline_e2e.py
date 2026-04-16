"""
End-to-end pipeline test.

Creates a synthetic test video with known structure, runs the full pipeline,
and validates the output JSON matches expected schema and constraints.

The synthetic video has 3 distinct visual segments:
  - 0-5s:   Red frame (loud sine tone) → should classify as something
  - 5-25s:  Blue frame (normal audio) → likely content
  - 25-30s: Black frame (silence) → likely dead_air or transition
"""

import sys
import os
import json
import subprocess
import tempfile
import pytest
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.main import run_pipeline
from src.ingest import seconds_to_formatted


# ─── Fixture: synthetic test video ──────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory) -> str:
    """
    Create a 30-second synthetic video with 3 visual segments:
      0-5s:   solid red, 440Hz sine
      5-25s:  solid blue, 440Hz sine
      25-30s: solid black, silence
    """
    tmpdir = tmp_path_factory.mktemp("e2e_videos")
    out_path = str(tmpdir / "synthetic_test.mp4")

    # Create segment 1: red + audio
    seg1 = str(tmpdir / "seg1.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=red:s=320x240:r=25:d=5",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", seg1,
    ], check=True)

    # Create segment 2: blue + audio
    seg2 = str(tmpdir / "seg2.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=25:d=20",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", seg2,
    ], check=True)

    # Create segment 3: black + silence
    seg3 = str(tmpdir / "seg3.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=320x240:r=25:d=5",
        "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=5",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", seg3,
    ], check=True)

    # Concatenate segments
    concat_file = str(tmpdir / "concat.txt")
    with open(concat_file, "w") as f:
        f.write(f"file '{seg1}'\n")
        f.write(f"file '{seg2}'\n")
        f.write(f"file '{seg3}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c", "copy", out_path,
    ], check=True)

    assert os.path.exists(out_path), "Failed to create synthetic test video"
    return out_path


# ─── E2E Tests ───────────────────────────────────────────────────────────────

class TestPipelineE2E:
    """Full pipeline integration tests."""

    @pytest.fixture(autouse=True)
    def run_once(self, synthetic_video):
        """Run pipeline once per class."""
        if not hasattr(self.__class__, "_result"):
            # New run_pipeline signature: (video_path, output_path, scene_threshold)
            self.__class__._result = run_pipeline(
                video_path=synthetic_video,
                scene_threshold=15.0,
            )
        self.result = self.__class__._result

    # ── Schema validation ────────────────────────────────────────────────────

    def test_result_is_dict(self):
        assert isinstance(self.result, dict)

    def test_required_top_level_keys(self):
        keys = [
            "video_filename", "original_video_duration_seconds",
            "original_video_duration_formatted", "original_video_resolution",
            "timeline_segments", "inserted_ads", "num_ads_inserted",
        ]
        for k in keys:
            assert k in self.result, f"Missing key: {k}"

    def test_video_filename_is_string(self):
        assert isinstance(self.result["video_filename"], str)

    def test_resolution_format(self):
        res = self.result["original_video_resolution"]
        assert "x" in res, f"Bad resolution format: {res}"
        w, h = res.split("x")
        assert int(w) > 0 and int(h) > 0

    # ── Duration validation ──────────────────────────────────────────────────

    def test_duration_is_number(self):
        dur = self.result["original_video_duration_seconds"]
        assert isinstance(dur, (int, float))
        assert dur > 0

    def test_duration_approx_30s(self):
        dur = self.result["output_duration_seconds"]
        assert 25 <= dur <= 35, f"Expected ~30s, got {dur}s"

    def test_duration_formatted_is_string(self):
        fmt = self.result["original_video_duration_formatted"]
        assert isinstance(fmt, str)
        assert ":" in fmt

    # ── Timeline validation ───────────────────────────────────────────────────

    def test_timeline_is_list(self):
        assert isinstance(self.result["timeline_segments"], list)

    def test_has_at_least_one_segment(self):
        assert len(self.result["timeline_segments"]) >= 1

    def test_timeline_segment_required_fields(self):
        required = [
            "type", "final_video_start_seconds", "final_video_end_seconds",
            "duration_seconds", "final_video_start_formatted", "final_video_end_formatted",
        ]
        for seg in self.result["timeline_segments"]:
            for k in required:
                assert k in seg, f"Segment missing key: {k}"

    def test_timeline_start_times_monotonic(self):
        segs = self.result["timeline_segments"]
        starts = [s["final_video_start_seconds"] for s in segs]
        for i in range(1, len(starts)):
            assert starts[i] >= starts[i-1] - 0.001, \
                f"Non-monotonic at index {i}: {starts[i-1]} > {starts[i]}"

    def test_timeline_types_valid(self):
        valid = {"video_content", "ad"}
        for s in self.result["timeline_segments"]:
            assert s["type"] in valid, f"Invalid segment type: {s['type']}"

    def test_segment_durations_positive(self):
        for s in self.result["timeline_segments"]:
            assert s["duration_seconds"] >= 0, f"Negative duration: {s}"

    def test_timeline_covers_full_video(self):
        segs = self.result["timeline_segments"]
        if not segs:
            return
        last_end = segs[-1]["final_video_end_seconds"]
        total_dur = self.result["output_duration_seconds"]
        # Allow ±2s tolerance (detection isn't frame-perfect)
        assert last_end >= total_dur - 2.0, \
            f"Timeline ends at {last_end}s but video is {total_dur}s"

    def test_first_segment_starts_at_zero(self):
        first = self.result["timeline_segments"][0]
        assert first["final_video_start_seconds"] == pytest.approx(0.0, abs=2.0)

    # ── Non-content validation (Ads) ──────────────────────────────────────────

    def test_inserted_ads_is_list(self):
        assert isinstance(self.result["inserted_ads"], list)

    def test_num_ads_matches_list_length(self):
        assert self.result["num_ads_inserted"] == len(
            self.result["inserted_ads"]
        )

    def test_ad_segment_fields(self):
        for seg in self.result["inserted_ads"]:
            assert "ad_index" in seg
            assert "ad_duration_seconds" in seg

    # ── JSON serialization ────────────────────────────────────────────────────

    def test_json_serializable(self):
        json_str = json.dumps(self.result)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_json_roundtrip_preserves_duration(self):
        json_str = json.dumps(self.result)
        parsed = json.loads(json_str)
        orig = self.result["output_duration_seconds"]
        parsed_dur = parsed["output_duration_seconds"]
        assert orig == pytest.approx(parsed_dur, rel=0.001)


class TestPipelineCLI:
    """Test the CLI entry point via subprocess."""

    def test_cli_produces_json(self, synthetic_video, tmp_path):
        out_json = str(tmp_path / "result.json")
        result = subprocess.run(
            [
                sys.executable, "src/main.py",
                "--input", synthetic_video,
                "--output", out_json,
            ],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(out_json), "JSON output not created"
        with open(out_json) as f:
            data = json.load(f)
        assert "timeline_segments" in data
        assert "video_filename" in data

    def test_cli_invalid_input_fails(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable, "src/main.py",
                "--input", "/nonexistent/fake_video.mp4",
                "--output", str(tmp_path / "out.json"),
            ],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert result.returncode != 0
