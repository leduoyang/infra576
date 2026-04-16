"""
ingest.py – Video ingestion and metadata extraction.
Uses ffprobe/ffmpeg (via subprocess) to get video metadata,
extract audio to WAV, and sample frames as numpy arrays.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def get_video_metadata(video_path: str) -> dict:
    """
    Extracts high-level metadata from the video file using ffprobe.

    INPUT:
        video_path (str): Path to the video file.

    OUTPUT:
        dict: A dictionary containing:
            - duration_seconds (float)
            - resolution (str) e.g., "1920x1080"
            - fps (float)
            - has_audio (bool)
            - video_codec (str)
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on '{video_path}': {result.stderr}")

    info = json.loads(result.stdout)
    video_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    audio_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "audio"),
        None,
    )

    duration = float(info["format"].get("duration", 0))
    width = int(video_stream.get("width", 0)) if video_stream else 0
    height = int(video_stream.get("height", 0)) if video_stream else 0

    # Parse fps
    fps_raw = video_stream.get("r_frame_rate", "0/1") if video_stream else "0/1"
    try:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except Exception:
        fps = 0.0

    # Extract offsets to handle stream alignment
    video_start = float(video_stream.get("start_time", 0)) if video_stream else 0.0
    audio_start = float(audio_stream.get("start_time", 0)) if audio_stream else 0.0

    return {
        "duration_seconds": duration,
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
        "fps": fps,
        "has_audio": audio_stream is not None,
        "video_codec": video_stream.get("codec_name", "unknown") if video_stream else "unknown",
        "audio_codec": audio_stream.get("codec_name", "unknown") if audio_stream else "none",
        "video_start_time": video_start,
        "audio_start_time": audio_start,
    }


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, out_wav_path: str, sample_rate: int = 16000) -> str:
    """Extract audio from a video file to a mono WAV file."""
    # We use -start_at_zero to ensure the output WAV starts exactly at T=0 of the video container
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vn",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "wav",
        str(out_wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")
    return out_wav_path


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: str,
    sample_fps: float = 1.0,
    resize: Optional[tuple] = (320, 180),
) -> list[dict]:
    """
    Extract frames from video at `sample_fps`.
    Uses CAP_PROP_POS_MSEC for high-precision timestamps (Handles VFR).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    
    # We'll sample based on timestamps rather than frame indices for better VFR support
    last_sampled_ms = -10000.0  # Force first frame
    interval_ms = 1000.0 / sample_fps

    while True:
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ret, frame = cap.read()
        if not ret:
            break
        
        if pos_ms >= last_sampled_ms + interval_ms:
            ts = pos_ms / 1000.0
            if resize:
                frame = cv2.resize(frame, resize)
            frames.append({"timestamp_seconds": ts, "frame": frame})
            last_sampled_ms = pos_ms

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def seconds_to_formatted(seconds: float) -> str:
    """Convert float seconds to HH:MM:SS.mmm string."""
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    s = (total_ms // 1000) % 60
    m = (total_ms // 60000) % 60
    h = total_ms // 3600000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
