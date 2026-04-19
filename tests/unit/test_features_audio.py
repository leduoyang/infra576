"""
test_features_audio.py – Updated to reflect the new audio.py API.
compute_global_audio_profile was removed; compute_audio_rms_variance and
compute_zcr_mean were added.
"""
import numpy as np
import pytest
from src.features.audio import (
    analyze_audio_features,
    compute_audio_rms_variance,
    compute_zcr_mean,
)


def test_analyze_audio_features_silence():
    sr = 16000
    y = np.zeros(sr * 2)  # 2 seconds of silence
    feats = analyze_audio_features(y, sr, 0.0, 1.0)
    assert feats["audio_energy"] == 0.0
    assert feats["spectral_centroid"] == 0.0


def test_analyze_audio_features_noise():
    sr = 16000
    y = np.random.default_rng(0).uniform(-1, 1, sr * 2).astype(np.float32)
    feats = analyze_audio_features(y, sr, 0.0, 1.0)
    assert feats["audio_energy"] > 0.0
    assert feats["spectral_centroid"] > 0.0


def test_analyze_audio_features_offset():
    sr = 16000
    y = np.zeros(sr * 2, dtype=np.float32)
    y[sr:] = 1.0  # first 1s silence, second 1s constant signal

    # Analyse first second → silence
    f1 = analyze_audio_features(y, sr, 0.0, 0.5, offset=0.0)
    assert f1["audio_energy"] == 0.0

    # Shift by -1s so "time 0" maps to the second half
    f2 = analyze_audio_features(y, sr, 0.0, 0.5, offset=-1.0)
    assert f2["audio_energy"] > 0.0


def test_compute_audio_rms_variance_noise_vs_silence():
    sr = 16000
    silence = np.zeros(sr * 2, dtype=np.float32)
    noise   = np.random.default_rng(1).uniform(-1, 1, sr * 2).astype(np.float32)

    var_silence = compute_audio_rms_variance(silence, sr, 0.0, 2.0)
    var_noise   = compute_audio_rms_variance(noise,   sr, 0.0, 2.0)

    assert var_silence == pytest.approx(0.0, abs=1e-10)
    assert var_noise > 0.0


def test_compute_zcr_mean_noise_vs_silence():
    sr = 16000
    silence = np.zeros(sr * 2, dtype=np.float32)
    noise   = np.random.default_rng(2).uniform(-1, 1, sr * 2).astype(np.float32)

    zcr_silence = compute_zcr_mean(silence, sr, 0.0, 2.0)
    zcr_noise   = compute_zcr_mean(noise,   sr, 0.0, 2.0)

    assert zcr_silence == pytest.approx(0.0, abs=1e-6)
    assert zcr_noise > zcr_silence
