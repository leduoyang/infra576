import numpy as np
import pytest
from src.features.audio import analyze_audio_features, compute_global_audio_profile

def test_compute_global_audio_profile():
    sr = 16000
    y = np.random.uniform(-1, 1, sr * 5)
    profile = compute_global_audio_profile(y, sr)
    assert "avg_centroid" in profile
    assert "avg_bandwidth" in profile
    assert profile["avg_centroid"] > 0.0


def test_analyze_audio_features_silence():
    sr = 16000
    y = np.zeros(sr * 2) # 2 seconds of silence
    feats = analyze_audio_features(y, sr, 0.0, 1.0)
    assert feats["audio_energy"] == 0.0
    assert feats["spectral_centroid"] == 0.0

def test_analyze_audio_features_noise():
    sr = 16000
    y = np.random.uniform(-1, 1, sr * 2)
    feats = analyze_audio_features(y, sr, 0.0, 1.0)
    assert feats["audio_energy"] > 0.0
    assert feats["spectral_centroid"] > 0.0

def test_analyze_audio_features_offset():
    sr = 16000
    y = np.zeros(sr * 2)
    y[sr:] = 1.0 # 1s silence, 1s noise
    
    # Analyze first second with offset 0
    f1 = analyze_audio_features(y, sr, 0.0, 0.5, offset=0.0)
    assert f1["audio_energy"] == 0.0
    
    # Analyze with offset so that "time 0" points to the second half
    f2 = analyze_audio_features(y, sr, 0.0, 0.5, offset=-1.0)
    assert f2["audio_energy"] > 0.0

