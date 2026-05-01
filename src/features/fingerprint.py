"""
fingerprint.py – Audio fingerprinting for cross-episode repeat detection.

Uses chromagram-based fingerprinting to detect segments that are
acoustically similar across multiple video files (recurring intros,
outros, sponsor reads, boilerplate).
"""

from __future__ import annotations
import numpy as np
from typing import Optional

try:
    import librosa
except ImportError:
    librosa = None


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------

def compute_audio_fingerprint(
    y: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    n_chroma: int = 12,
    hop_length: int = 512,
) -> Optional[np.ndarray]:
    """
    Compute a compact audio fingerprint for a time segment.
    Uses mean + std of chroma features → 24-dim vector.
    """
    if librosa is None:
        return None

    start_idx = max(0, int(start_sec * sr))
    end_idx = min(len(y), int(end_sec * sr))
    segment = y[start_idx:end_idx]

    if len(segment) < 2048:
        return None

    chroma = librosa.feature.chroma_stft(y=segment, sr=sr, hop_length=hop_length)
    # Compact: mean + std per chroma bin
    fp = np.concatenate([np.mean(chroma, axis=1), np.std(chroma, axis=1)])
    return fp


def fingerprint_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Cosine similarity between two fingerprints."""
    norm1 = np.linalg.norm(fp1)
    norm2 = np.linalg.norm(fp2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return float(np.dot(fp1, fp2) / (norm1 * norm2))


# ---------------------------------------------------------------------------
# Cross-episode matching
# ---------------------------------------------------------------------------

def find_repeated_segments(
    episodes: list[dict],
    similarity_threshold: float = 0.92,
) -> list[dict]:
    """
    Given fingerprinted segments from multiple episodes, find segments
    that repeat across episodes.

    Each episode dict: {
        "video_filename": str,
        "segments": [{"start_seconds", "end_seconds", "fingerprint": np.ndarray, ...}]
    }

    Returns list of repeat groups:
        [{"segments": [...], "similarity": float, "likely_type": str}]
    """
    all_fps = []
    for ep in episodes:
        for seg in ep.get("segments", []):
            fp = seg.get("fingerprint")
            if fp is not None:
                all_fps.append({
                    "video": ep["video_filename"],
                    "start": seg["start_seconds"],
                    "end": seg["end_seconds"],
                    "duration": seg.get("duration_seconds", seg["end_seconds"] - seg["start_seconds"]),
                    "fp": fp,
                    "label": seg.get("segment_label", "unknown"),
                })

    if len(all_fps) < 2:
        return []

    # Pairwise cross-episode comparison
    repeats = []
    used = set()

    for i in range(len(all_fps)):
        if i in used:
            continue
        group = [all_fps[i]]
        for j in range(i + 1, len(all_fps)):
            if j in used:
                continue
            # Only match across different videos
            if all_fps[j]["video"] == all_fps[i]["video"]:
                continue
            sim = fingerprint_similarity(all_fps[i]["fp"], all_fps[j]["fp"])
            if sim >= similarity_threshold:
                group.append(all_fps[j])
                used.add(j)

        if len(group) > 1:
            used.add(i)
            avg_sim = np.mean([
                fingerprint_similarity(group[0]["fp"], g["fp"])
                for g in group[1:]
            ])
            # Infer type from position
            likely_type = _infer_repeat_type(group)
            repeats.append({
                "segments": [
                    {"video": g["video"], "start": g["start"], "end": g["end"],
                     "duration": g["duration"]}
                    for g in group
                ],
                "similarity": round(float(avg_sim), 4),
                "likely_type": likely_type,
                "num_episodes": len(set(g["video"] for g in group)),
            })

    return repeats


def _infer_repeat_type(group: list[dict]) -> str:
    """Guess what type of repeated segment this is based on position."""
    avg_start = np.mean([g["start"] for g in group])
    # If it's in the first 30 seconds of most episodes, likely an intro
    if avg_start < 30.0:
        return "recurring_intro"
    # If it's labeled outro or appears late, likely outro
    if all(g.get("label") == "outro" for g in group):
        return "recurring_outro"
    return "recurring_boilerplate"
