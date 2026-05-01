"""Unit tests for src/features/audio.py"""

import numpy as np
import pytest
from src.features.audio import analyze_audio_features, compute_global_audio_profile


def test_compute_global_audio_profile():
    sr = 16000
    y = np.random.uniform(-1, 1, sr * 5).astype(np.float32)
    profile = compute_global_audio_profile(y, sr)
    assert "avg_centroid" in profile
    assert "avg_bandwidth" in profile
    assert "avg_energy" in profile
    assert "avg_zcr" in profile
    assert "avg_flatness" in profile
    assert "avg_rolloff" in profile
    assert profile["avg_centroid"] > 0.0


def test_compute_global_audio_profile_short():
    sr = 16000
    y = np.zeros(100)
    profile = compute_global_audio_profile(y, sr)
    assert profile["avg_centroid"] == 0.0
    assert profile["avg_energy"] == 0.0


def test_analyze_audio_features_silence():
    sr = 16000
    y = np.zeros(sr * 2)
    feats = analyze_audio_features(y, sr, 0.0, 1.0)
    assert feats["audio_energy"] == 0.0
    assert feats["spectral_centroid"] == 0.0
    assert feats["silence_ratio"] == 1.0


def test_analyze_audio_features_noise():
    sr = 16000
    y = np.random.uniform(-1, 1, sr * 2).astype(np.float32)
    feats = analyze_audio_features(y, sr, 0.0, 1.0)
    assert feats["audio_energy"] > 0.0
    assert feats["spectral_centroid"] > 0.0
    assert feats["zcr"] > 0.0
    assert feats["spectral_flatness"] > 0.0
    assert feats["spectral_rolloff"] > 0.0


def test_analyze_audio_features_new_keys():
    sr = 16000
    y = np.random.uniform(-1, 1, sr * 2).astype(np.float32)
    feats = analyze_audio_features(y, sr, 0.0, 1.0)
    expected_keys = [
        "audio_energy", "spectral_centroid", "spectral_bandwidth",
        "zcr", "silence_ratio", "spectral_flatness", "spectral_rolloff",
    ]
    for key in expected_keys:
        assert key in feats, f"Missing key: {key}"


def test_analyze_audio_features_offset():
    sr = 16000
    y = np.zeros(sr * 2)
    y[sr:] = 1.0
    f1 = analyze_audio_features(y, sr, 0.0, 0.5, offset=0.0)
    assert f1["audio_energy"] == 0.0
    f2 = analyze_audio_features(y, sr, 0.0, 0.5, offset=-1.0)
    assert f2["audio_energy"] > 0.0


def test_analyze_short_segment():
    sr = 16000
    y = np.zeros(500)  # too short
    feats = analyze_audio_features(y, sr, 0.0, 0.01)
    assert feats["audio_energy"] == 0.0
    # Too-short fragments return silence_ratio=0.0 to avoid false silence signals
    assert feats["silence_ratio"] == 0.0
    assert feats.get("audio_too_short") is True
