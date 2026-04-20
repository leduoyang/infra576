"""
classification.py – Hybrid ad detector.

Combines:
  (a) absolute deviation from the video's universal baseline
  (b) adaptive MAD z-score across segments in this video

A shot is flagged as an ad if either channel fires strongly, or both fire
moderately. Then post-process (bridge fragments, drop shorts, drop boundaries).
"""

import numpy as np

BRIDGE_GAP_SECONDS = 20.0
MIN_AD_DURATION = 20.0
BOUNDARY_MARGIN = 3.0

ABS_THRESHOLD = 1.5       # absolute combined score threshold
Z_STRONG = 3.0            # any single-feature z above this = ad
Z_SOFT = 1.5              # z threshold used for combined vote


def _robust_z(values: np.ndarray) -> np.ndarray:
    med = np.median(values)
    mad = np.median(np.abs(values - med)) + 1e-6
    return (values - med) / (1.4826 * mad)


def _absolute_score(seg: dict, gp: dict) -> float:
    avg_c = gp.get("avg_centroid", 1000.0)
    avg_b = gp.get("avg_bandwidth", 1000.0)
    avg_m = gp.get("avg_motion", 5.0)

    c = seg.get("spectral_centroid", avg_c)
    b = seg.get("spectral_bandwidth", avg_b)
    m = seg.get("motion_score", avg_m)

    freq = abs(c - avg_c) / (avg_c + 1e-6)
    rich = abs(b - avg_b) / (avg_b + 1e-6)
    mot = max(0.0, m / (avg_m + 1e-6) - 1.0)

    music = 0.4 if (b > avg_b * 1.2 and c > avg_c) else 0.0
    return freq * 1.5 + rich * 0.8 + mot * 0.8 + music


def _speech_score(seg: dict) -> float:
    """Directional speech-based ad score. Higher = more ad-like."""
    ns = float(seg.get("no_speech_prob", 0.0))           # 0..1, high = music/noise
    wr = float(seg.get("word_rate", 0.0))                # words/sec
    lp = float(seg.get("avg_logprob", 0.0))              # <=0, very negative = transcription confused
    sponsor = float(seg.get("sponsor_hit", 0.0))          # 0 or 1

    score = 0.0
    score += ns * 1.5
    if wr < 1.0:
        score += 0.6
    elif wr < 2.0:
        score += 0.3
    if lp < -1.0:
        score += 0.7 * min(1.0, (-lp - 1.0))
    score += sponsor * 1.0
    return score


def classify_segments(segments: list[dict], duration: float, global_profile: dict = None) -> list[dict]:
    if not segments:
        return []

    gp = global_profile or {}

    # Adaptive: z-score per feature across all segments
    keys = ("spectral_centroid", "spectral_bandwidth", "motion_score", "audio_energy", "color_variance")
    feat = {k: np.array([s.get(k, 0.0) for s in segments], dtype=float) for k in keys}
    z = {k: np.abs(_robust_z(v)) for k, v in feat.items()}
    z_combined = (
        1.2 * z["spectral_centroid"]
        + 0.8 * z["spectral_bandwidth"]
        + 0.9 * z["motion_score"]
        + 0.7 * z["audio_energy"]
        + 0.3 * z["color_variance"]
    ) / 3.9

    # Color variance alone is noisy, exclude from "strong single signal" gate
    strong_keys = ("spectral_centroid", "spectral_bandwidth", "motion_score", "audio_energy")
    z_max_per_seg = np.maximum.reduce([z[k] for k in strong_keys])

    classified = []
    for i, seg in enumerate(segments):
        s = seg.copy()
        abs_s = _absolute_score(s, gp)
        sp_s = _speech_score(s)

        strong_adaptive = z_max_per_seg[i] >= Z_STRONG
        combined_adaptive = z_combined[i] >= Z_SOFT
        absolute_hit = abs_s >= ABS_THRESHOLD
        speech_hit = sp_s >= 1.2

        # Ad if: strong adaptive, OR (adaptive + absolute), OR very strong absolute,
        # OR speech indicates non-content, OR speech combines with any other signal
        is_ad = (
            strong_adaptive
            or (combined_adaptive and absolute_hit)
            or abs_s >= ABS_THRESHOLD * 1.5
        )

        s["ad_score_abs"] = round(float(abs_s), 3)
        s["ad_score_z"] = round(float(z_combined[i]), 3)
        s["ad_score_zmax"] = round(float(z_max_per_seg[i]), 3)
        s["ad_score_speech"] = round(float(sp_s), 3)
        s["is_content"] = not is_ad
        s["segment_type"] = "ad" if is_ad else "video_content"
        s["confidence"] = float(min(1.0, max(
            abs_s / (ABS_THRESHOLD * 2),
            z_combined[i] / (Z_STRONG * 2),
            sp_s / 2.5,
        )))
        classified.append(s)

    classified = _merge_consecutive(classified)
    classified = _bridge_ad_fragments(classified, BRIDGE_GAP_SECONDS)
    classified = _drop_short_ads(classified, MIN_AD_DURATION)
    classified = _drop_boundary_ads(classified, duration, BOUNDARY_MARGIN)
    classified = _merge_consecutive(classified)

    # Speech-based rescue: find sustained low-speech / high no-speech runs missed above
    classified = _speech_rescue(segments, classified, duration)
    classified = _merge_consecutive(classified)
    return classified


def _speech_rescue(raw_segments: list[dict], classified: list[dict], duration: float) -> list[dict]:
    """Mark as ad any contiguous run of raw shots with sustained non-speech signal
    (median no_speech_prob >= 0.35, median word_rate <= 1.0, total duration >= 30s)."""
    if not raw_segments:
        return classified

    # Mark raw segs
    flags = []
    for s in raw_segments:
        ns = float(s.get("no_speech_prob", 0.0))
        wr = float(s.get("word_rate", 0.0))
        lp = float(s.get("avg_logprob", 0.0))
        flags.append(ns >= 0.35 and wr <= 1.5 and lp <= -0.8)

    # Find runs
    runs = []
    i = 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j + 1 < len(flags) and flags[j + 1]:
                j += 1
            start = raw_segments[i]["start_seconds"]
            end = raw_segments[j]["end_seconds"]
            if end - start >= 30.0:
                runs.append((start, end))
            i = j + 1
        else:
            i += 1

    if not runs:
        return classified

    # Build content mask, flip runs to ad
    for start, end in runs:
        classified = _force_ad_region(classified, start, end)

    # Re-apply boundary filter
    classified = _drop_boundary_ads(classified, duration, BOUNDARY_MARGIN)
    return classified


def _force_ad_region(segments: list[dict], ad_start: float, ad_end: float) -> list[dict]:
    """Split/flip any content segment overlapping [ad_start, ad_end] to ad."""
    out = []
    for s in segments:
        ss, se = s["start_seconds"], s["end_seconds"]
        if se <= ad_start or ss >= ad_end or s["segment_type"] == "ad":
            out.append(s.copy())
            continue
        # Overlaps: split
        parts = []
        if ss < ad_start:
            parts.append((ss, ad_start, "video_content"))
        parts.append((max(ss, ad_start), min(se, ad_end), "ad"))
        if se > ad_end:
            parts.append((ad_end, se, "video_content"))
        for st, en, typ in parts:
            if en - st < 0.01:
                continue
            clone = s.copy()
            clone["start_seconds"] = st
            clone["end_seconds"] = en
            clone["duration_seconds"] = en - st
            clone["segment_type"] = typ
            clone["is_content"] = typ == "video_content"
            out.append(clone)
    return out


def _merge_consecutive(segments: list[dict]) -> list[dict]:
    if not segments:
        return []
    merged = [segments[0].copy()]
    for nxt in segments[1:]:
        cur = merged[-1]
        if nxt["segment_type"] == cur["segment_type"]:
            cur["end_seconds"] = nxt["end_seconds"]
            cur["duration_seconds"] = cur["end_seconds"] - cur["start_seconds"]
            cur["confidence"] = min(cur.get("confidence", 1.0), nxt.get("confidence", 1.0))
        else:
            merged.append(nxt.copy())
    return merged


def _bridge_ad_fragments(segments: list[dict], gap_seconds: float) -> list[dict]:
    if len(segments) < 3:
        return segments
    out = [segments[0].copy()]
    i = 1
    while i < len(segments):
        cur = segments[i]
        prev = out[-1]
        has_next = i + 1 < len(segments)
        if (
            cur["segment_type"] == "video_content"
            and prev["segment_type"] == "ad"
            and has_next
            and segments[i + 1]["segment_type"] == "ad"
            and cur["duration_seconds"] <= gap_seconds
        ):
            nxt = segments[i + 1]
            prev["end_seconds"] = nxt["end_seconds"]
            prev["duration_seconds"] = prev["end_seconds"] - prev["start_seconds"]
            i += 2
        else:
            out.append(cur.copy())
            i += 1
    return out


def _drop_short_ads(segments: list[dict], min_dur: float) -> list[dict]:
    out = []
    for s in segments:
        if s["segment_type"] == "ad" and s["duration_seconds"] < min_dur:
            f = s.copy()
            f["segment_type"] = "video_content"
            f["is_content"] = True
            out.append(f)
        else:
            out.append(s.copy())
    return out


def _drop_boundary_ads(segments: list[dict], duration: float, margin: float) -> list[dict]:
    out = []
    for s in segments:
        near_start = s["start_seconds"] <= margin
        near_end = s["end_seconds"] >= duration - margin
        if s["segment_type"] == "ad" and (near_start or near_end) and s["duration_seconds"] < 20.0:
            f = s.copy()
            f["segment_type"] = "video_content"
            f["is_content"] = True
            out.append(f)
        else:
            out.append(s.copy())
    return out


def merge_consecutive_segments(segments: list[dict]) -> list[dict]:
    return _merge_consecutive(segments)
