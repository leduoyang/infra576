"""
conftest.py – pytest configuration for video-segmentation tests.

Provides shared fixtures and skip markers.
"""

import shutil
import subprocess
import pytest


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


# Global skip marker for tests requiring ffmpeg
requires_ffmpeg = pytest.mark.skipif(
    not _has_ffmpeg(),
    reason="ffmpeg/ffprobe not found in PATH – install with: brew install ffmpeg",
)
