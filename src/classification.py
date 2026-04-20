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
BOUNDARY_MARGIN = 50.0
BOUNDARY_MAX_DURATION = 45.0

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

    # Audio onset boundary refinement: snap ad start/end to strongest onset (±window)
    onset_times = gp.get("onset_times")
    onset_strengths = gp.get("onset_strengths")
    if onset_times:
        classified = _snap_ad_boundaries(classified, onset_times, onset_strengths, window=6.0)
    return classified


def _snap_ad_boundaries(segments: list[dict], onset_times, onset_strengths=None, window: float = 6.0) -> list[dict]:
    """Snap each ad's start/end to the strongest audio onset within ±window.
    Adjust the adjacent content segment so the timeline stays contiguous."""
    if not onset_times:
        return segments
    times_arr = np.asarray(onset_times, dtype=float)
    order = np.argsort(times_arr)
    onsets = times_arr[order]
    if onset_strengths is not None and len(onset_strengths) == len(onset_times):
        strengths = np.asarray(onset_strengths, dtype=float)[order]
    else:
        strengths = np.ones_like(onsets)
    segs = [s.copy() for s in segments]

    def strongest(t: float, lo: float, hi: float):
        mask = (onsets >= lo) & (onsets <= hi)
        if not mask.any():
            return None
        cand_t = onsets[mask]
        cand_s = strengths[mask]
        # Prefer strong onsets close to t: score = strength / (1 + distance)
        score = cand_s / (1.0 + np.abs(cand_t - t))
        return float(cand_t[np.argmax(score)])

    for i, s in enumerate(segs):
        if s["segment_type"] != "ad":
            continue
        # snap start
        t = s["start_seconds"]
        lo = t - window
        hi = t + window
        if i > 0:
            lo = max(lo, segs[i - 1]["start_seconds"] + 1.0)
            hi = min(hi, s["end_seconds"] - 1.0)
        new_start = strongest(t, lo, hi)
        if new_start is not None:
            s["start_seconds"] = new_start
            if i > 0:
                segs[i - 1]["end_seconds"] = new_start
                segs[i - 1]["duration_seconds"] = (
                    segs[i - 1]["end_seconds"] - segs[i - 1]["start_seconds"]
                )
        # snap end
        t = s["end_seconds"]
        lo = t - window
        hi = t + window
        if i + 1 < len(segs):
            hi = min(hi, segs[i + 1]["end_seconds"] - 1.0)
            lo = max(lo, s["start_seconds"] + 1.0)
        new_end = strongest(t, lo, hi)
        if new_end is not None:
            s["end_seconds"] = new_end
            if i + 1 < len(segs):
                segs[i + 1]["start_seconds"] = new_end
                segs[i + 1]["duration_seconds"] = (
                    segs[i + 1]["end_seconds"] - segs[i + 1]["start_seconds"]
                )
        s["duration_seconds"] = s["end_seconds"] - s["start_seconds"]
    return segs


def _speech_rescue(raw_segments: list[dict], classified: list[dict], duration: float) -> list[dict]:
    """Mark as ad any contiguous run of raw shots with sustained non-speech signal
    (median no_speech_prob >= 0.35, median word_rate <= 1.0, total duration >= 30s).

    Intro/outro suppression: runs touching the first 10% or last 8% of the video
    are hard-suppressed. GT only labels inserted ads; original intros/outros are
    not ads even if they look like ads acoustically (silent/musical)."""
    if not raw_segments:
        return classified

    intro_cutoff = duration * 0.10
    outro_cutoff = duration * 0.92

    flags = []
    for s in raw_segments:
        ns = float(s.get("no_speech_prob", 0.0))
        wr = float(s.get("word_rate", 0.0))
        lp = float(s.get("avg_logprob", 0.0))
        flags.append(ns >= 0.35 and wr <= 2.0 and lp <= -0.8)

    # Allow up to MAX_GAP consecutive non-flag shots inside a run (intermittent
    # music/talk in fragmented ad blocks).
    MAX_GAP = 2
    runs = []
    i = 0
    while i < len(flags):
        if flags[i]:
            j = i
            gap = 0
            k = i
            while k + 1 < len(flags):
                if flags[k + 1]:
                    j = k + 1
                    gap = 0
                    k = k + 1
                elif gap + 1 <= MAX_GAP:
                    gap += 1
                    k = k + 1
                else:
                    break
            start = raw_segments[i]["start_seconds"]
            end = raw_segments[j]["end_seconds"]
            if end - start >= 30.0:
                if start <= intro_cutoff or end >= outro_cutoff:
                    i = j + 1
                    continue
                # Reject pure-silence runs (content transitions/interludes):
                # real ads have mixed speech, not 100% ns=1.0, wr=0.
                run_shots = raw_segments[i : j + 1]
                pure = sum(
                    1
                    for s in run_shots
                    if float(s.get("no_speech_prob", 0.0)) >= 0.95
                    and float(s.get("word_rate", 0.0)) < 0.1
                )
                if pure / max(1, len(run_shots)) > 0.85:
                    i = j + 1
                    continue
                runs.append((start, end))
            i = j + 1
        else:
            i += 1

    if not runs:
        return classified

    for start, end in runs:
        classified = _force_ad_region(classified, start, end)

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
        if s["segment_type"] == "ad" and (near_start or near_end) and s["duration_seconds"] < BOUNDARY_MAX_DURATION:
            f = s.copy()
            f["segment_type"] = "video_content"
            f["is_content"] = True
            out.append(f)
        else:
            out.append(s.copy())
    return out


def merge_consecutive_segments(segments: list[dict]) -> list[dict]:
    return _merge_consecutive(segments)
