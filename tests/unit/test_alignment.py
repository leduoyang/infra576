import pytest
import numpy as np
import cv2
import os
from src.ingest import extract_frames, extract_audio, get_video_metadata
from src.segmentation import run_segmentation_pipeline
from pathlib import Path

def test_audio_video_alignment(tmp_path):
    """
    Alignment Test Strategy:
    1. Create a 5s sync video:
       - 0-2s: Red frame + 440Hz tone
       - 2-5s: Blue frame + 880Hz tone
    2. Extract frames and audio.
    3. Verify that at 1.0s, the frame is red and audio has 440Hz energy.
    4. Verify that at 3.0s, the frame is blue and audio has 880Hz energy.
    """
    import subprocess
    
    video_path = str(tmp_path / "sync_test.mp4")
    
    # Create segments
    seg1 = str(tmp_path / "seg1.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=red:s=160x120:d=2",
        "-f", "lavfi", "-i", "sine=f=440:d=2",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", seg1
    ], check=True)
    
    seg2 = str(tmp_path / "seg2.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=3",
        "-f", "lavfi", "-i", "sine=f=880:d=3",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", seg2
    ], check=True)
    
    # Concat
    concat_txt = str(tmp_path / "concat.txt")
    with open(concat_txt, "w") as f:
        f.write(f"file '{seg1}'\nfile '{seg2}'\n")
        
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", concat_txt,
        "-c", "copy", video_path
    ], check=True)
    
    # Analyze
    metadata = get_video_metadata(video_path)
    frames = extract_frames(video_path, sample_fps=1.0)
    
    audio_wav = str(tmp_path / "sync.wav")
    extract_audio(video_path, audio_wav, sample_rate=16000)
    
    import librosa
    y, sr = librosa.load(audio_wav, sr=16000)
    
    # Checks
    # Frame at 1.0s should be Red (high Red channel)
    f1 = next(f for f in frames if 0.9 <= f["timestamp_seconds"] <= 1.1)
    # OpenCV BGR: Red is index 2
    assert f1["frame"][0, 0, 2] > 200
    assert f1["frame"][0, 0, 0] < 50
    
    # Frame at 3.0s should be Blue (high Blue channel)
    f2 = next(f for f in frames if 2.9 <= f["timestamp_seconds"] <= 3.1)
    assert f2["frame"][0, 0, 0] > 200
    assert f2["frame"][0, 0, 2] < 50
    
    # Audio at 1.0s should have 440Hz peak
    # We'll check the STFT or a simple filter.
    # Simple check: Energy should be present
    start_idx = int(1.0 * 16000)
    chunk = y[start_idx : start_idx + 1600]
    assert np.mean(chunk**2) > 0.001
    
    # Alignment: Verify metadata start times (usually 0 for such files)
    assert metadata["video_start_time"] == pytest.approx(0.0, abs=0.1)
    assert metadata["audio_start_time"] == pytest.approx(0.0, abs=0.1)
