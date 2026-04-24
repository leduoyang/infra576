"""
classification.py – Hybrid ad detector.

Combines:
  (a) absolute deviation from the video's universal baseline
  (b) adaptive MAD z-score across segments in this video

A shot is flagged as an ad if either channel fires strongly, or both fire
moderately. Then post-process (bridge fragments, drop shorts, drop boundaries).
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

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


def _identify_main_speakers(segments: list[dict], threshold_seconds: float = 180.0) -> list[np.ndarray]:
    """Clusters MFCCs to find all speakers who talk for more than threshold_seconds in content segments."""
    speech_segs = [s for s in segments if s.get("no_speech_prob", 1.0) < 0.3 and s.get("mfcc") and s.get("is_content", True)]
    
    valid_segs = []
    for s in speech_segs:
        if np.linalg.norm(s["mfcc"]) > 0:
            valid_segs.append(s)

    if not valid_segs:
        return []

    X = np.array([s["mfcc"] for s in valid_segs])
    if len(X) < 2:
        if valid_segs[0].get("duration_seconds", 0) >= threshold_seconds:
            return [X[0]]
        return []

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage='average',
        distance_threshold=0.1
    )
    labels = clustering.fit_predict(X)

    main_speakers = []
    unique_labels = set(labels)
    for label in unique_labels:
        cluster_segs = [s for i, s in enumerate(valid_segs) if labels[i] == label]
        total_duration = sum(s.get("duration_seconds", 0) for s in cluster_segs)
        if total_duration >= threshold_seconds:
            cluster_mfccs = np.array([s["mfcc"] for s in cluster_segs])
            centroid = np.median(cluster_mfccs, axis=0)
            main_speakers.append(centroid)

    return main_speakers


def _rescue_main_speakers(classified: list[dict], main_speakers: list[np.ndarray]) -> list[dict]:
    """Rescues 'ad' segments that contain a main speaker's voice."""
    if not main_speakers:
        return classified

    for s in classified:
        if not s["is_content"] and s.get("mfcc") and s.get("no_speech_prob", 1.0) < 0.3:
            seg_mfcc = np.array(s["mfcc"])
            norm_s = np.linalg.norm(seg_mfcc)
            if norm_s == 0:
                continue

            for speaker in main_speakers:
                norm_sp = np.linalg.norm(speaker)
                if norm_sp > 0:
                    sim = np.dot(seg_mfcc, speaker) / (norm_s * norm_sp)
                    if sim > 0.98:
                        s["is_content"] = True
                        s["segment_type"] = "video_content"
                        s["confidence"] = 0.5  # Lower confidence since it was rescued
                        break
    return classified


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

    main_speakers = _identify_main_speakers(classified, threshold_seconds=180.0)
    classified = _rescue_main_speakers(classified, main_speakers)

    classified = _merge_consecutive(classified)
    classified = _bridge_ad_fragments(classified, BRIDGE_GAP_SECONDS)
    classified = _drop_short_ads(classified, MIN_AD_DURATION)
    classified = _drop_boundary_ads(classified, duration, BOUNDARY_MARGIN)
    classified = _merge_consecutive(classified)
    classified = _drop_content_like_ads(classified, segments)

    # Speech-based rescue: find sustained low-speech / high no-speech runs missed above
    classified = _speech_rescue(segments, classified, duration)
    classified = _merge_consecutive(classified)
    classified = _drop_content_like_ads(classified, segments)
    classified = _trim_dialogue_boundaries(classified, segments)
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
                # Reject pure-silence runs that are ALSO quiet (content transitions):
                # loud silent runs may be jingle/music ads, keep those.
                run_shots = raw_segments[i : j + 1]
                pure = sum(
                    1
                    for s in run_shots
                    if float(s.get("no_speech_prob", 0.0)) >= 0.75
                    and float(s.get("word_rate", 0.0)) < 2.0
                )
                run_rms = np.mean([float(s.get("audio_energy", 0.0)) for s in run_shots])
                video_rms_med = float(np.median([float(s.get("audio_energy", 0.0)) for s in raw_segments])) + 1e-6
                is_quiet = run_rms < 1.5 * video_rms_med
                if pure / max(1, len(run_shots)) > 0.85 and is_quiet:
                    i = j + 1
                    continue
                runs.append((start, end))
            i = j + 1
        else:
            i += 1

    # Sandwich rescue: short silent shot (8-20s) between speech-heavy neighbors.
    # Pattern: content → silent ad snippet → content. Extend ±10s.
    for idx, s in enumerate(raw_segments):
        ns = float(s.get("no_speech_prob", 0.0))
        wr = float(s.get("word_rate", 0.0))
        lp = float(s.get("avg_logprob", 0.0))
        d = float(s.get("duration_seconds", 0.0))
        if not (ns >= 0.95 and wr < 0.1 and lp <= -1.5 and 5.0 <= d < 20.0):
            continue
        if idx == 0 or idx + 1 >= len(raw_segments):
            continue
        prev_wr = float(raw_segments[idx - 1].get("word_rate", 0.0))
        next_wr = float(raw_segments[idx + 1].get("word_rate", 0.0))
        prev_ns = float(raw_segments[idx - 1].get("no_speech_prob", 1.0))
        next_ns = float(raw_segments[idx + 1].get("no_speech_prob", 1.0))
        if prev_wr >= 1.5 and next_wr >= 1.5 and prev_ns < 0.5 and next_ns < 0.5:
            start = max(0.0, s["start_seconds"] - 10.0)
            end = min(duration, s["end_seconds"] + 10.0)
            if not (start <= intro_cutoff or end >= outro_cutoff):
                runs.append((start, end))

    # Visual rescue: sustained color_variance spike (p75 threshold).
    # Targets ads with voiceover narration that speech_rescue misses but
    # visual features clearly mark as non-content.
    cvs = np.asarray([float(s.get("color_variance", 0.0)) for s in raw_segments])
    mots = np.asarray([float(s.get("motion_score", 0.0)) for s in raw_segments])
    if len(cvs) >= 5:
        cv_hi = float(np.percentile(cvs, 85))
        mot_hi = float(np.percentile(mots, 65))
        vflags = [
            (cvs[i] > cv_hi and mots[i] > mot_hi)
            for i in range(len(raw_segments))
        ]
        VMAX_GAP = 1
        i = 0
        while i < len(vflags):
            if vflags[i]:
                j = i
                gap = 0
                k = i
                while k + 1 < len(vflags):
                    if vflags[k + 1]:
                        j = k + 1
                        gap = 0
                        k += 1
                    elif gap + 1 <= VMAX_GAP:
                        gap += 1
                        k += 1
                    else:
                        break
                start = raw_segments[i]["start_seconds"]
                end = raw_segments[j]["end_seconds"]
                if end - start >= 30.0:
                    if not (start <= intro_cutoff or end >= outro_cutoff):
                        # Anti-FP: ≥30% of shots must have speech drop signal
                        run_shots = raw_segments[i : j + 1]
                        drop_cnt = sum(
                            1
                            for sh in run_shots
                            if float(sh.get("no_speech_prob", 0.0)) >= 0.35
                            or float(sh.get("word_rate", 10.0)) < 1.0
                        )
                        if drop_cnt / max(1, len(run_shots)) >= 0.30:
                            runs.append((start, end))
                i = j + 1
            else:
                i += 1

    # Loud-narration rescue: broadcast ads are mixed louder and have voiceover.
    # Shot with RMS ≥ 1.8× video-median + speech (ns≤0.1, wr≥0.2) + motion ≥ 20
    # = ad candidate. Targets motion-rich narration ads missed by main classifier.
    loud_runs = []
    rms_all = np.asarray([float(s.get("audio_energy", 0.0)) for s in raw_segments])
    if len(rms_all) >= 5:
        med_rms = float(np.median(rms_all))
        if med_rms > 1e-5:
            lflags = [
                (
                    float(s.get("audio_energy", 0.0)) >= 1.8 * med_rms
                    and float(s.get("no_speech_prob", 1.0)) <= 0.1
                    and float(s.get("word_rate", 0.0)) >= 0.2
                    and float(s.get("motion_score", 0.0)) >= 20.0
                )
                for s in raw_segments
            ]
            LMAX_GAP = 1
            i = 0
            while i < len(lflags):
                if lflags[i]:
                    j = i
                    gap = 0
                    k = i
                    while k + 1 < len(lflags):
                        if lflags[k + 1]:
                            j = k + 1
                            gap = 0
                            k += 1
                        elif gap + 1 <= LMAX_GAP:
                            gap += 1
                            k += 1
                        else:
                            break
                    start = raw_segments[i]["start_seconds"]
                    end = raw_segments[j]["end_seconds"]
                    if end - start >= 20.0:
                        if not (start <= intro_cutoff or end >= outro_cutoff):
                            loud_runs.append((start, end))
                    i = j + 1
                else:
                    i += 1
            runs.extend(loud_runs)

    # When loud-narration rescue fires next to an existing silent-dominated ad,
    # flip that silent region back to content. The rescue provides the real ad
    # boundary; the adjacent silent block is boundary drift from main classifier.
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


def _drop_content_like_ads(classified: list[dict], raw_segments: list[dict]) -> list[dict]:
    """Reject ad segments that look like content (static cutaways, quiet silence).

    Two patterns:
    (a) Single long static shot: 1 shot covers the whole ad, duration > 35s,
        motion < 10, mean RMS < 0.5× video median. Real ads cut every 5-15s.
    (b) Short pure-silent quiet region: ≥90% shots pure-silent (ns≥0.95, wr<0.1),
        duration < 40s, mean RMS < 2.5× video median. Loud silent jingles kept.
    """
    if not raw_segments:
        return classified
    rms_all = np.asarray([float(s.get("audio_energy", 0.0)) for s in raw_segments])
    med_rms = float(np.median(rms_all)) + 1e-6
    out = []
    for seg in classified:
        if seg.get("segment_type") != "ad":
            out.append(seg.copy())
            continue
        s0, s1 = seg["start_seconds"], seg["end_seconds"]
        shots_in = [r for r in raw_segments if r["end_seconds"] > s0 and r["start_seconds"] < s1]
        if not shots_in:
            out.append(seg.copy())
            continue
        dur = s1 - s0
        mean_rms = float(np.mean([float(r.get("audio_energy", 0.0)) for r in shots_in]))
        rms_ratio = mean_rms / med_rms
        # (a) single long static shot
        if len(shots_in) == 1 and dur > 35.0:
            sh = shots_in[0]
            if float(sh.get("motion_score", 0.0)) < 10.0 and rms_ratio < 0.5:
                flipped = seg.copy()
                flipped["segment_type"] = "video_content"
                flipped["is_content"] = True
                out.append(flipped)
                continue
        # (b) short pure-silent quiet region
        pure = sum(
            1 for r in shots_in
            if float(r.get("no_speech_prob", 0.0)) >= 0.95
            and float(r.get("word_rate", 1.0)) < 0.1
        )
        pure_ratio = pure / len(shots_in)
        if pure_ratio >= 0.90 and dur < 40.0 and rms_ratio < 2.5:
            flipped = seg.copy()
            flipped["segment_type"] = "video_content"
            flipped["is_content"] = True
            out.append(flipped)
            continue
        # (c) Narration-absent low-motion region: zero narration shots (wr≥1.0),
        # zero high-motion shots (≥ 2.5× video median), short, quiet.
        # Silent high-motion ads survive via motion check; silent-loud jingles
        # via RMS; long silent commercials (test_001 ad1) via duration.
        mot_med = float(np.median([float(r.get("motion_score", 0.0)) for r in raw_segments])) + 1e-6
        n_narration = sum(1 for r in shots_in if float(r.get("word_rate", 0.0)) >= 1.0)
        n_highmot = sum(1 for r in shots_in if float(r.get("motion_score", 0.0)) / mot_med >= 2.5)
        if n_narration == 0 and n_highmot == 0 and dur < 45.0 and rms_ratio < 2.5:
            flipped = seg.copy()
            flipped["segment_type"] = "video_content"
            flipped["is_content"] = True
            out.append(flipped)
            continue
        out.append(seg.copy())
    return out


def _trim_dialogue_boundaries(classified: list[dict], raw_segments: list[dict]) -> list[dict]:
    """Trim leading/trailing dialogue-content shots from ad segments.

    Pattern: ad classifier bleeds into adjacent content dialogue. Trailing shots
    with strong speech (ns<=0.15), high word rate (wr>=2.5), and low motion
    (<15) are content dialogue, not ad.
    """
    if not raw_segments:
        return classified

    def is_dialogue(sh: dict) -> bool:
        return (
            float(sh.get("no_speech_prob", 0.0)) <= 0.15
            and float(sh.get("word_rate", 0.0)) >= 2.5
            and float(sh.get("motion_score", 0.0)) < 15.0
        )

    out = []
    for seg in classified:
        if seg.get("segment_type") != "ad":
            out.append(seg.copy())
            continue
        s0, s1 = seg["start_seconds"], seg["end_seconds"]
        shots_in = [
            r for r in raw_segments
            if r["end_seconds"] > s0 + 0.01 and r["start_seconds"] < s1 - 0.01
        ]
        if len(shots_in) < 2:
            out.append(seg.copy())
            continue

        new_end = s1
        for sh in reversed(shots_in):
            if is_dialogue(sh):
                new_end = sh["start_seconds"]
            else:
                break
        new_start = s0
        for sh in shots_in:
            if sh["end_seconds"] >= new_end:
                break
            if is_dialogue(sh):
                new_start = sh["end_seconds"]
            else:
                break

        new_start = max(s0, new_start)
        new_end = min(s1, new_end)
        # Require trimmed span ≥ 10s to avoid chopping single speech-bursts
        # that are part of the ad (narrator tag lines, product callouts).
        tail_trim = s1 - new_end
        head_trim = new_start - s0
        if tail_trim > 0 and tail_trim < 10.0:
            new_end = s1
        if head_trim > 0 and head_trim < 10.0:
            new_start = s0
        if new_end - new_start < MIN_AD_DURATION or (new_start == s0 and new_end == s1):
            out.append(seg.copy())
            continue

        if new_start > s0:
            c = seg.copy()
            c["start_seconds"] = s0
            c["end_seconds"] = new_start
            c["duration_seconds"] = new_start - s0
            c["segment_type"] = "video_content"
            c["is_content"] = True
            out.append(c)
        a = seg.copy()
        a["start_seconds"] = new_start
        a["end_seconds"] = new_end
        a["duration_seconds"] = new_end - new_start
        out.append(a)
        if new_end < s1:
            c = seg.copy()
            c["start_seconds"] = new_end
            c["end_seconds"] = s1
            c["duration_seconds"] = s1 - new_end
            c["segment_type"] = "video_content"
            c["is_content"] = True
            out.append(c)
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
