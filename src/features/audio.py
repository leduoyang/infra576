import numpy as np
import librosa


def _extract_segment(y: np.ndarray, sr: int, start_sec: float, end_sec: float, offset: float = 0.0) -> np.ndarray:
    """Slice audio array for a time range, clamped to valid indices."""
    start_idx = max(0, int((start_sec - offset) * sr))
    end_idx = min(len(y), int((end_sec - offset) * sr))
    return y[start_idx:end_idx]


def analyze_audio_features(y: np.ndarray, sr: int, start_sec: float, end_sec: float, offset: float = 0.0) -> dict:
    """
    Computes spectral + temporal features for a segment.

    Returns dict with keys:
        audio_energy, spectral_centroid, spectral_bandwidth,
        zcr, silence_ratio, spectral_flatness, spectral_rolloff
    """
    segment_audio = _extract_segment(y, sr, start_sec, end_sec, offset)

    empty = {
        "audio_energy": 0.0, "spectral_centroid": 0.0,
        "spectral_bandwidth": 0.0, "zcr": 0.0,
        "silence_ratio": 1.0, "spectral_flatness": 0.0,
        "spectral_rolloff": 0.0,
    }

    if len(segment_audio) < 2048:
        empty["audio_too_short"] = True
        empty["silence_ratio"] = 0.0   # do not let tiny fragments look like silence
        return empty

    rms = librosa.feature.rms(y=segment_audio)
    centroid = librosa.feature.spectral_centroid(y=segment_audio, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=segment_audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=segment_audio)
    flatness = librosa.feature.spectral_flatness(y=segment_audio)
    rolloff = librosa.feature.spectral_rolloff(y=segment_audio, sr=sr)

    # Silence ratio: fraction of RMS frames below a threshold
    rms_vals = rms.flatten()
    silence_thresh = 0.005  # roughly -46 dBFS
    silence_ratio = float(np.mean(rms_vals < silence_thresh)) if len(rms_vals) > 0 else 1.0

    return {
        "audio_energy": float(np.mean(rms)),
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
        "zcr": float(np.mean(zcr)),
        "silence_ratio": silence_ratio,
        "spectral_flatness": float(np.mean(flatness)),
        "spectral_rolloff": float(np.mean(rolloff)),
        "audio_too_short": False,
    }


def compute_global_audio_profile(y: np.ndarray, sr: int) -> dict:
    """
    Computes the 'Universal Audio Profile' for the entire video.
    Includes averages and standard deviations for adaptive thresholding.
    """
    if len(y) < 2048:
        return {
            "avg_centroid": 0.0, "avg_bandwidth": 0.0,
            "avg_energy": 0.0, "avg_zcr": 0.0,
            "std_centroid": 0.0, "std_bandwidth": 0.0,
            "avg_flatness": 0.0, "avg_rolloff": 0.0,
        }

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    flatness = librosa.feature.spectral_flatness(y=y)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    return {
        "avg_centroid": float(np.mean(centroid)),
        "avg_bandwidth": float(np.mean(bandwidth)),
        "avg_energy": float(np.mean(rms)),
        "avg_zcr": float(np.mean(zcr)),
        "std_centroid": float(np.std(centroid)),
        "std_bandwidth": float(np.std(bandwidth)),
        "avg_flatness": float(np.mean(flatness)),
        "avg_rolloff": float(np.mean(rolloff)),
    }

