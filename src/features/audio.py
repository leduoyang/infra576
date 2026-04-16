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
        return {"audio_energy": 0.0, "spectral_centroid": 0.0, "spectral_bandwidth": 0.0}
        
    rms = librosa.feature.rms(y=segment_audio)
    centroid = librosa.feature.spectral_centroid(y=segment_audio, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=segment_audio, sr=sr)
    
    return {
        "audio_energy": float(np.mean(rms)),
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
    }

def compute_global_audio_profile(y: np.ndarray, sr: int) -> dict:
    """
    Computes the 'Universal Audio Profile' for the entire video.
    """
    if len(y) < 2048:
        return {"avg_centroid": 0.0, "avg_bandwidth": 0.0}
        
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    
    return {
        "avg_centroid": float(np.mean(centroid)),
        "avg_bandwidth": float(np.mean(bandwidth)),
    }

