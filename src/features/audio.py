import numpy as np
import librosa

def analyze_audio_features(y: np.ndarray, sr: int, start_sec: float, end_sec: float, offset: float = 0.0) -> dict:
    """
    Computes spectral features for a segment to distinguish content types.
    Returns: {rms, spectral_centroid, spectral_bandwidth}
    """
    start_idx = int((start_sec - offset) * sr)
    end_idx = int((end_sec - offset) * sr)
    
    start_idx = max(0, start_idx)
    end_idx = min(len(y), end_idx)
    
    segment_audio = y[start_idx:end_idx]
    if len(segment_audio) < 1024:
        return {"audio_energy": 0.0, "spectral_centroid": 0.0, "spectral_bandwidth": 0.0, "mfcc": [0.0] * 12}
        
    rms = librosa.feature.rms(y=segment_audio)
    centroid = librosa.feature.spectral_centroid(y=segment_audio, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=segment_audio, sr=sr)
    mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
    
    return {
        "audio_energy": float(np.mean(rms)),
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
        "mfcc": [float(x) for x in np.mean(mfcc[1:], axis=1)],
    }

def compute_global_audio_profile(y: np.ndarray, sr: int) -> dict:
    """
    Computes the 'Universal Audio Profile' for the entire video.
    Also returns strong onset times (seconds) for boundary snapping.
    """
    if len(y) < 2048:
        return {"avg_centroid": 0.0, "avg_bandwidth": 0.0, "onset_times": [], "avg_mfcc": [0.0] * 12}

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    all_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    strengths = onset_env[all_frames]
    if len(strengths) > 0:
        cutoff = float(np.percentile(strengths, 80))
        keep = strengths >= cutoff
        frames = all_frames[keep]
        kept_strengths = strengths[keep]
    else:
        frames = all_frames
        kept_strengths = strengths
    onset_times = librosa.frames_to_time(frames, sr=sr).tolist()

    return {
        "avg_centroid": float(np.median(centroid)),
        "avg_bandwidth": float(np.median(bandwidth)),
        "avg_mfcc": [float(x) for x in np.median(mfcc[1:], axis=1)],
        "onset_times": onset_times,
        "onset_strengths": kept_strengths.tolist(),
    }

