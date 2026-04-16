"""
Unit tests for src/ingest.py
"""

import sys
import os
import subprocess
import tempfile
import numpy as np
import pytest

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ingest import seconds_to_formatted, get_video_metadata, extract_frames, extract_audio
from conftest import requires_ffmpeg


# ─── Helpers ────────────────────────────────────────────────────────────────

def make_test_video(path: str, duration: float = 5.0, width: int = 320, height: int = 240,
                    has_audio: bool = True):
    """Generate a short synthetic test video using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=red:s={width}x{height}:r=25:d={duration}",
    ]
    if has_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
        cmd += ["-c:v", "libx264", "-c:a", "aac", "-shortest", str(path)]
    else:
        cmd += ["-c:v", "libx264", "-an", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"ffmpeg failed to create test video: {result.stderr}"


# ─── seconds_to_formatted ────────────────────────────────────────────────────

class TestSecondsToFormatted:
    def test_zero(self):
        assert seconds_to_formatted(0.0) == "00:00:00.000"

    def test_one_second(self):
        assert seconds_to_formatted(1.0) == "00:00:01.000"

    def test_one_minute(self):
        assert seconds_to_formatted(60.0) == "00:01:00.000"

    def test_one_hour(self):
        assert seconds_to_formatted(3600.0) == "01:00:00.000"

    def test_fractional(self):
        result = seconds_to_formatted(106.159)
        assert result == "00:01:46.159"

    def test_large_value(self):
        result = seconds_to_formatted(1279.86)
        assert result == "00:21:19.860"

    def test_format_structure(self):
        result = seconds_to_formatted(12.5)
        parts = result.split(":")
        assert len(parts) == 3
        assert "." in parts[2]

    def test_millisecond_precision(self):
        assert seconds_to_formatted(0.001) == "00:00:00.001"
        assert seconds_to_formatted(0.999) == "00:00:00.999"


# ─── get_video_metadata ───────────────────────────────────────────────────────

@requires_ffmpeg
class TestGetVideoMetadata:
    @pytest.fixture(scope="class")
    def test_video(self, tmp_path_factory):
        path = tmp_path_factory.mktemp("videos") / "test.mp4"
        make_test_video(str(path), duration=5.0, width=320, height=240, has_audio=True)
        return str(path)

    def test_returns_dict(self, test_video):
        meta = get_video_metadata(test_video)
        assert isinstance(meta, dict)

    def test_resolution(self, test_video):
        meta = get_video_metadata(test_video)
        assert meta["resolution"] == "320x240"
        assert meta["width"] == 320
        assert meta["height"] == 240

    def test_duration_approximate(self, test_video):
        meta = get_video_metadata(test_video)
        assert abs(meta["duration_seconds"] - 5.0) < 1.0, f"Duration: {meta['duration_seconds']}"

    def test_has_audio(self, test_video):
        meta = get_video_metadata(test_video)
        assert meta["has_audio"] is True

    def test_fps_positive(self, test_video):
        meta = get_video_metadata(test_video)
        assert meta["fps"] > 0

    def test_invalid_path_raises(self):
        with pytest.raises(RuntimeError):
            get_video_metadata("/nonexistent/path/video.mp4")


# ─── extract_frames ───────────────────────────────────────────────────────────

@requires_ffmpeg
class TestExtractFrames:
    @pytest.fixture(scope="class")
    def test_video(self, tmp_path_factory):
        path = tmp_path_factory.mktemp("videos") / "frames_test.mp4"
        make_test_video(str(path), duration=5.0)
        return str(path)

    def test_returns_list(self, test_video):
        frames = extract_frames(test_video, sample_fps=1.0)
        assert isinstance(frames, list)
        assert len(frames) > 0

    def test_frame_structure(self, test_video):
        frames = extract_frames(test_video, sample_fps=1.0)
        assert "timestamp_seconds" in frames[0]
        assert "frame" in frames[0]
        assert frames[0]["frame"].shape[2] == 3  # BGR

    def test_sample_fps_affects_count(self, test_video):
        frames_1fps = extract_frames(test_video, sample_fps=1.0)
        frames_2fps = extract_frames(test_video, sample_fps=2.0)
        assert len(frames_2fps) >= len(frames_1fps)

    def test_timestamps_monotonic(self, test_video):
        frames = extract_frames(test_video, sample_fps=1.0)
        ts = [f["timestamp_seconds"] for f in frames]
        assert all(ts[i] <= ts[i+1] for i in range(len(ts)-1))

    def test_resize(self, test_video):
        frames = extract_frames(test_video, sample_fps=1.0, resize=(160, 90))
        h, w = frames[0]["frame"].shape[:2]
        assert w == 160
        assert h == 90


# ─── extract_audio ───────────────────────────────────────────────────────────

@requires_ffmpeg
class TestExtractAudio:
    @pytest.fixture(scope="class")
    def test_video(self, tmp_path_factory):
        path = tmp_path_factory.mktemp("videos") / "audio_test.mp4"
        make_test_video(str(path), duration=3.0, has_audio=True)
        return str(path)

    def test_creates_wav(self, test_video, tmp_path):
        wav_path = str(tmp_path / "out.wav")
        result = extract_audio(test_video, wav_path)
        assert result == wav_path
        assert os.path.exists(wav_path)
        assert os.path.getsize(wav_path) > 0
