import numpy as np
import librosa


def _get_segment_audio(y: np.ndarray, sr: int,
                       start_sec: float, end_sec: float,
                       offset: float = 0.0) -> np.ndarray:
    """Slice the raw audio array to the requested segment window."""
    start_idx = max(0, int((start_sec - offset) * sr))
    end_idx   = min(len(y), int((end_sec - offset) * sr))
    return y[start_idx:end_idx]


def analyze_audio_features(y: np.ndarray, sr: int,
                           start_sec: float, end_sec: float,
                           offset: float = 0.0) -> dict:
    """
    Computes spectral features for a segment.
    Returns: {audio_energy, spectral_centroid, spectral_bandwidth}
    """
    seg = _get_segment_audio(y, sr, start_sec, end_sec, offset)
    if len(seg) < 1024:
        return {"audio_energy": 0.0, "spectral_centroid": 0.0, "spectral_bandwidth": 0.0}

    rms       = librosa.feature.rms(y=seg)
    centroid  = librosa.feature.spectral_centroid(y=seg, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=seg, sr=sr)

    return {
        "audio_energy":        float(np.mean(rms)),
        "spectral_centroid":   float(np.mean(centroid)),
        "spectral_bandwidth":  float(np.mean(bandwidth)),
    }


def compute_audio_rms_variance(y: np.ndarray, sr: int,
                               start_sec: float, end_sec: float,
                               offset: float = 0.0) -> float:
    """
    Step 2 Feature: Audio RMS Variance (dynamic range / compression).
    Returns the variance of the per-frame RMS energy within the segment.
    Ads are heavily compressed → uniformly loud → LOW variance.
    TV content has whispers and bangs → HIGH variance.
    """
    seg = _get_segment_audio(y, sr, start_sec, end_sec, offset)
    if len(seg) < 1024:
        return 0.0
    rms = librosa.feature.rms(y=seg)[0]   # shape: (n_frames,)
    return float(np.var(rms))


def compute_zcr_mean(y: np.ndarray, sr: int,
                     start_sec: float, end_sec: float,
                     offset: float = 0.0) -> float:
    """
    Zero-Crossing Rate (ZCR) — noisiness metric.
    Counts how many times the waveform crosses 0 per frame, then returns
    the mean across all frames in the segment.
    Hard rock, explosions, excited announcers (common in ads) score much
    higher than ambient room tone or normal conversational dialogue.
    """
    seg = _get_segment_audio(y, sr, start_sec, end_sec, offset)
    if len(seg) < 1024:
        return 0.0
    zcr = librosa.feature.zero_crossing_rate(seg)[0]  # shape: (n_frames,)
    return float(np.mean(zcr))


# ---------------------------------------------------------------------------
# Array-based API (MoviePy integration)
# These functions receive a pre-extracted segment audio array directly from
# MoviePy's clip.audio.to_soundarray(), bypassing the need to slice a global
# audio array.  They are the primary API used by segmentation.py.
# ---------------------------------------------------------------------------

def analyze_audio_from_array(y: np.ndarray, sr: int) -> dict:
    """
    Computes all per-segment spectral features from a pre-extracted audio array.
    
    Workflow (MoviePy integration):
        subclip = clip.subclip(start, end)
        raw = subclip.audio.to_soundarray(fps=sr)     # shape: (N,) or (N, 2)
        mono = raw.mean(axis=1) if raw.ndim > 1 else raw
        feats = analyze_audio_from_array(mono.astype(np.float32), sr)
    
    Returns: {audio_energy, spectral_centroid, spectral_bandwidth}
    """
    if len(y) < 1024:
        return {"audio_energy": 0.0, "spectral_centroid": 0.0, "spectral_bandwidth": 0.0}
    rms       = librosa.feature.rms(y=y)
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return {
        "audio_energy":       float(np.mean(rms)),
        "spectral_centroid":  float(np.mean(centroid)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
    }


def compute_rms_variance_from_array(y: np.ndarray, sr: int) -> float:
    """
    Audio RMS Variance (dynamic range / compression) from a pre-extracted array.
    Ads are compressed → uniformly loud → LOW variance.
    TV content has whispers and bangs → HIGH variance.
    """
    if len(y) < 1024:
        return 0.0
    rms = librosa.feature.rms(y=y)[0]
    return float(np.var(rms))


def compute_zcr_from_array(y: np.ndarray) -> float:
    """
    Zero-Crossing Rate mean from a pre-extracted audio array.
    High ZCR → frenetic audio (ads with excited speech / rock music).
    """
    if len(y) < 1024:
        return 0.0
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    return float(np.mean(zcr))
