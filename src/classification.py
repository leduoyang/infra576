"""
classification.py – Intro/Outro segment classifier.

Taxonomy:
  - content : core material the viewer wants
  - intro   : opening sequence / title card / logo / bumper
  - outro   : closing sequence / end card / credits / logo / bumper

This version intentionally ignores all other non-content classes and only detects:
  - intro
  - outro
  - content

Major cues:
  1. Beginning/end position
  2. Speech should block intro/outro
  3. Static/title/logo/screen-like visual behavior
  4. Audio difference, silence, or repeated bumper music
  5. Layout anomaly:
       main content often has one stable resolution/active-area/black-frame pattern,
       while intro/outro often has a different layout.

Important policy:
  - BGM starting near the end is NOT enough to be outro.
  - Colorful animation / spoken footage near the end is NOT outro.
  - Documentary PPT-like content with narration should remain content.
  - Outro should start at real end-card / black screen / credits / logo / creator-name screen.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Segment-type constants
# ---------------------------------------------------------------------------

CONTENT = "content"
INTRO = "intro"
OUTRO = "outro"

AD = "ad"
SELF_PROMOTION = "self_promotion"
RECAP = "recap"
TRANSITION = "transition"
DEAD_AIR = "dead_air"

NON_CONTENT_TYPES = {
    INTRO,
    OUTRO,
    AD,
    SELF_PROMOTION,
    RECAP,
    TRANSITION,
    DEAD_AIR,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_segments(
    segments: list[dict],
    duration: float,
    global_profile: dict | None = None,
) -> list[dict]:
    """
    Intro/Outro-only classifier.

    Desired behavior:
      - complete silence + large same-color/simple background + text/logo screen
        CAN be intro/outro
      - music-only opening can still be intro
      - BGM lead-in alone should NOT start outro
      - speech should block intro/outro
      - intro can only appear at the beginning
      - outro can only appear at the end
      - main content layout/resolution/black-frame pattern should prevent
        false intro/outro extension
    """
    if not segments:
        return []

    gp = global_profile or {}

    # Estimate dominant main-content layout from the middle of the video.
    # You can also pass this in global_profile["main_layout"] if computed elsewhere.
    main_layout = gp.get("main_layout")
    if main_layout is None:
        main_layout = _estimate_main_layout_from_segments(segments, duration)

    ctx = _ClassifierContext(
        total_duration=duration,

        # Audio baselines
        avg_centroid=gp.get("avg_centroid", 1000.0),
        avg_bandwidth=gp.get("avg_bandwidth", 1000.0),
        avg_energy=gp.get("avg_energy", 0.05),
        avg_flatness=gp.get("avg_flatness", 0.1),
        avg_zcr=gp.get("avg_zcr", 0.05),

        # Visual baselines
        avg_motion=gp.get("avg_motion", 5.0),
        std_motion=gp.get("std_motion", 3.0),
        avg_edge=gp.get("avg_edge_density", 0.05),
        avg_color_variance=gp.get("avg_color_variance", 0.01),

        has_transcript=gp.get("has_transcript", False),

        # Dominant main-content layout baseline
        main_width=main_layout.get("width"),
        main_height=main_layout.get("height"),
        main_active_w_ratio=main_layout.get("active_width_ratio"),
        main_active_h_ratio=main_layout.get("active_height_ratio"),
        main_active_aspect=main_layout.get("active_aspect_ratio"),
        main_border_left=main_layout.get("border_left_ratio"),
        main_border_right=main_layout.get("border_right_ratio"),
        main_border_top=main_layout.get("border_top_ratio"),
        main_border_bottom=main_layout.get("border_bottom_ratio"),
        main_black_border_ratio=main_layout.get("black_border_ratio"),
    )

    classified = []

    for seg in segments:
        s = seg.copy()
        label, confidence = _classify_single(s, ctx)

        s["segment_type"] = label
        s["segment_label"] = label
        s["is_content"] = label == CONTENT
        s["confidence"] = confidence

        classified.append(s)

        # Capture intro audio profile for later outro matching.
        if label == INTRO and ctx.intro_audio is None:
            ctx.intro_audio = {
                "centroid": s.get("spectral_centroid", ctx.avg_centroid),
                "bandwidth": s.get("spectral_bandwidth", ctx.avg_bandwidth),
                "energy": s.get("audio_energy", ctx.avg_energy),
                "flatness": s.get("spectral_flatness", ctx.avg_flatness),
                "silence_ratio": s.get("silence_ratio", 0.0),
                "zcr": s.get("zcr", ctx.avg_zcr),
            }

    # -----------------------------------------------------------------------
    # Post-processing
    # -----------------------------------------------------------------------
    # Important:
    # Demote bad boundary chunks BEFORE the first merge.
    # Otherwise many small wrong OUTRO chunks can merge into one huge OUTRO,
    # and speech/content evidence from child chunks may be hidden.
    merged = _demote_speech_or_main_content_boundary_segments(classified, ctx)
    merged = merge_consecutive_segments(merged)

    merged = _cascade_intro(merged, ctx)
    merged = _cascade_outro(merged, ctx)

    merged = _demote_speech_or_main_content_boundary_segments(merged, ctx)
    merged = _merge_noncontent_gaps(merged, max_gap=4.0)
    merged = merge_consecutive_segments(merged)

    merged = _refine_short_noncontent(merged)
    merged = _bridge_boundary_screen_gaps(merged, ctx)
    merged = _demote_speech_or_main_content_boundary_segments(merged, ctx)
    merged = merge_consecutive_segments(merged)

    merged = _enforce_boundary_intro_outro_only(merged, ctx)
    merged = _demote_speech_or_main_content_boundary_segments(merged, ctx)
    merged = merge_consecutive_segments(merged)

    merged = _apply_boundary_guards(merged)
    merged = _drop_tiny_boundary_noncontent(merged, min_duration=0.6)
    merged = merge_consecutive_segments(merged)

    return merged


# ---------------------------------------------------------------------------
# Classifier context
# ---------------------------------------------------------------------------

class _ClassifierContext:
    __slots__ = (
        "total_duration",

        "avg_centroid",
        "avg_bandwidth",
        "avg_energy",
        "avg_motion",
        "std_motion",
        "avg_flatness",
        "avg_edge",
        "avg_color_variance",
        "avg_zcr",
        "has_transcript",

        "intro_audio",

        # Main-content visual layout baseline
        "main_width",
        "main_height",
        "main_active_w_ratio",
        "main_active_h_ratio",
        "main_active_aspect",
        "main_border_left",
        "main_border_right",
        "main_border_top",
        "main_border_bottom",
        "main_black_border_ratio",
    )

    def __init__(self, **kwargs):
        self.intro_audio = None
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _boundary_window_seconds(total_duration: float) -> float:
    """
    Candidate intro/outro search window.

    Earlier versions used up to 140s, which caused long end sections to become
    OUTRO. This narrower window is safer for documentary/animation videos.
    """
    return min(60.0, max(15.0, total_duration * 0.04))


def _first_float(d: dict, keys: tuple[str, ...], default=None):
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return default


def _median(vals: list[float]):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None

    n = len(vals)
    mid = n // 2

    if n % 2 == 1:
        return vals[mid]

    return 0.5 * (vals[mid - 1] + vals[mid])


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _estimate_main_layout_from_segments(segments: list[dict], duration: float) -> dict:
    """
    Estimate dominant/main-content visual layout from the middle of the video.

    This avoids intro/outro polluting the baseline.

    Expected optional segment fields:
      - frame_width / width / resolution_width / video_width
      - frame_height / height / resolution_height / video_height
      - active_width_ratio / content_width_ratio / inner_width_ratio
      - active_height_ratio / content_height_ratio / inner_height_ratio
      - active_aspect_ratio / content_aspect_ratio / inner_aspect_ratio
      - border_left_ratio / black_left_ratio / left_border_ratio
      - border_right_ratio / black_right_ratio / right_border_ratio
      - border_top_ratio / black_top_ratio / top_border_ratio
      - border_bottom_ratio / black_bottom_ratio / bottom_border_ratio
      - black_border_ratio / letterbox_ratio / border_black_ratio
    """
    if not segments:
        return {}

    lo = duration * 0.20
    hi = duration * 0.80

    candidates = []
    for s in segments:
        start = s.get("start_seconds", 0.0)
        end = s.get("end_seconds", start)
        mid = 0.5 * (start + end)

        if lo <= mid <= hi:
            candidates.append(s)

    if len(candidates) < 3:
        candidates = segments

    def med(keys: tuple[str, ...]):
        vals = [_first_float(s, keys) for s in candidates]
        return _median(vals)

    layout = {
        "width": med(("frame_width", "width", "resolution_width", "video_width")),
        "height": med(("frame_height", "height", "resolution_height", "video_height")),

        "active_width_ratio": med(
            ("active_width_ratio", "content_width_ratio", "inner_width_ratio")
        ),
        "active_height_ratio": med(
            ("active_height_ratio", "content_height_ratio", "inner_height_ratio")
        ),
        "active_aspect_ratio": med(
            ("active_aspect_ratio", "content_aspect_ratio", "inner_aspect_ratio")
        ),

        "border_left_ratio": med(
            ("border_left_ratio", "black_left_ratio", "left_border_ratio")
        ),
        "border_right_ratio": med(
            ("border_right_ratio", "black_right_ratio", "right_border_ratio")
        ),
        "border_top_ratio": med(
            ("border_top_ratio", "black_top_ratio", "top_border_ratio")
        ),
        "border_bottom_ratio": med(
            ("border_bottom_ratio", "black_bottom_ratio", "bottom_border_ratio")
        ),
        "black_border_ratio": med(
            ("black_border_ratio", "letterbox_ratio", "border_black_ratio")
        ),
    }

    # Derive active aspect if not directly available.
    if layout["active_aspect_ratio"] is None:
        aw = layout.get("active_width_ratio")
        ah = layout.get("active_height_ratio")

        if aw is not None and ah is not None and ah > 1e-6:
            layout["active_aspect_ratio"] = aw / ah

    return layout


def _has_layout_reference(ctx: _ClassifierContext) -> bool:
    return any(
        v is not None
        for v in (
            ctx.main_width,
            ctx.main_height,
            ctx.main_active_w_ratio,
            ctx.main_active_h_ratio,
            ctx.main_active_aspect,
            ctx.main_border_left,
            ctx.main_border_right,
            ctx.main_border_top,
            ctx.main_border_bottom,
            ctx.main_black_border_ratio,
        )
    )


def _layout_anomaly_score(seg: dict, ctx: _ClassifierContext) -> float:
    """
    Score [0, 1] for how different this segment's visual layout is from
    the dominant main-content layout.

    This helps when main content uses one stable active area / black-frame pattern,
    while intro/outro uses a different layout.
    """
    if not _has_layout_reference(ctx):
        return 0.0

    score = 0.0

    w = _first_float(seg, ("frame_width", "width", "resolution_width", "video_width"))
    h = _first_float(seg, ("frame_height", "height", "resolution_height", "video_height"))

    if ctx.main_width and ctx.main_height and w and h:
        w_diff = abs(w - ctx.main_width) / max(ctx.main_width, 1e-6)
        h_diff = abs(h - ctx.main_height) / max(ctx.main_height, 1e-6)

        if w_diff > 0.03 or h_diff > 0.03:
            score += 0.40

    aw = _first_float(
        seg,
        ("active_width_ratio", "content_width_ratio", "inner_width_ratio"),
    )
    ah = _first_float(
        seg,
        ("active_height_ratio", "content_height_ratio", "inner_height_ratio"),
    )
    ar = _first_float(
        seg,
        ("active_aspect_ratio", "content_aspect_ratio", "inner_aspect_ratio"),
    )

    if ar is None and aw is not None and ah is not None and ah > 1e-6:
        ar = aw / ah

    if ctx.main_active_aspect is not None and ar is not None:
        ar_diff = abs(ar - ctx.main_active_aspect) / max(ctx.main_active_aspect, 1e-6)

        if ar_diff > 0.08:
            score += 0.28
        elif ar_diff > 0.04:
            score += 0.15

    if ctx.main_active_w_ratio is not None and aw is not None:
        aw_diff = abs(aw - ctx.main_active_w_ratio)

        if aw_diff > 0.10:
            score += 0.18
        elif aw_diff > 0.05:
            score += 0.08

    if ctx.main_active_h_ratio is not None and ah is not None:
        ah_diff = abs(ah - ctx.main_active_h_ratio)

        if ah_diff > 0.10:
            score += 0.18
        elif ah_diff > 0.05:
            score += 0.08

    border_pairs = [
        (
            ctx.main_border_left,
            _first_float(seg, ("border_left_ratio", "black_left_ratio", "left_border_ratio")),
        ),
        (
            ctx.main_border_right,
            _first_float(seg, ("border_right_ratio", "black_right_ratio", "right_border_ratio")),
        ),
        (
            ctx.main_border_top,
            _first_float(seg, ("border_top_ratio", "black_top_ratio", "top_border_ratio")),
        ),
        (
            ctx.main_border_bottom,
            _first_float(seg, ("border_bottom_ratio", "black_bottom_ratio", "bottom_border_ratio")),
        ),
    ]

    border_diffs = [
        abs(cur - ref)
        for ref, cur in border_pairs
        if ref is not None and cur is not None
    ]

    if border_diffs:
        max_border_diff = max(border_diffs)

        if max_border_diff > 0.10:
            score += 0.25
        elif max_border_diff > 0.05:
            score += 0.12

    black_border = _first_float(
        seg,
        ("black_border_ratio", "letterbox_ratio", "border_black_ratio"),
    )

    if ctx.main_black_border_ratio is not None and black_border is not None:
        bb_diff = abs(black_border - ctx.main_black_border_ratio)

        if bb_diff > 0.15:
            score += 0.22
        elif bb_diff > 0.08:
            score += 0.10

    return max(0.0, min(1.0, score))


def _matches_main_layout(seg: dict, ctx: _ClassifierContext) -> bool:
    """
    True if this segment visually matches the main content layout.

    This helps stop intro/outro cascade before it eats quiet main content.
    """
    if not _has_layout_reference(ctx):
        return False

    has_seg_layout = any(
        k in seg
        for k in (
            "frame_width",
            "width",
            "resolution_width",
            "video_width",
            "frame_height",
            "height",
            "resolution_height",
            "video_height",
            "active_width_ratio",
            "content_width_ratio",
            "inner_width_ratio",
            "active_height_ratio",
            "content_height_ratio",
            "inner_height_ratio",
            "active_aspect_ratio",
            "content_aspect_ratio",
            "inner_aspect_ratio",
            "black_border_ratio",
            "letterbox_ratio",
            "border_black_ratio",
            "border_left_ratio",
            "black_left_ratio",
            "left_border_ratio",
            "border_right_ratio",
            "black_right_ratio",
            "right_border_ratio",
            "border_top_ratio",
            "black_top_ratio",
            "top_border_ratio",
            "border_bottom_ratio",
            "black_bottom_ratio",
            "bottom_border_ratio",
        )
    )

    if not has_seg_layout:
        return False

    return _layout_anomaly_score(seg, ctx) <= 0.12


# ---------------------------------------------------------------------------
# Speech/audio/content guards
# ---------------------------------------------------------------------------

def _has_speech(seg: dict, ctx: _ClassifierContext) -> bool:
    """
    Strict speech detector for intro/outro boundary logic.

    With 0.5s boundary chunks, words_per_second can be unreliable because
    transcript timestamps do not always overlap perfectly with the chunk.
    If transcript text or word_count exists, treat it as real speech and
    block intro/outro classification/extension.
    """
    text = str(seg.get("transcript_text", "") or "").strip()
    word_count = _to_int(seg.get("word_count", 0), 0)
    wps = _to_float(seg.get("words_per_second", 0.0), 0.0)

    if word_count > 0:
        return True

    if text:
        return True

    # Fallback when transcript is disabled or unavailable.
    return wps > 0.8


def _has_active_audio(seg: dict, ctx: _ClassifierContext) -> bool:
    """
    Detect active non-silent audio when transcript is missing/unreliable.

    This is NOT a speech recognizer. It is a safety cue: if the video is in the
    normal main-content layout and the audio is clearly active, do not call it
    outro just because the picture is visually stable.
    """
    silence = _to_float(seg.get("silence_ratio", 0.0), 0.0)
    energy = _to_float(seg.get("audio_energy", ctx.avg_energy), ctx.avg_energy)

    # Active if most frames are not silent. The energy floor avoids treating
    # numeric noise as meaningful audio.
    energy_floor = max(ctx.avg_energy * 0.20, 1e-5)
    return silence <= 0.55 and energy >= energy_floor


def _visual_boundary_card_score(
    seg: dict,
    ctx: _ClassifierContext,
    screen_score: float | None = None,
    layout_score: float | None = None,
) -> float:
    """
    Score [0, 1] for whether this segment visually looks like a real
    intro/outro card, end screen, title card, logo card, credits screen, or
    black screen.

    Important:
    BGM/audio changes alone should NOT make an outro.
    The classifier should wait until the visual boundary actually begins.
    """
    silence = _to_float(seg.get("silence_ratio", 0.0), 0.0)
    self_sim = _to_float(seg.get("frame_self_similarity", 0.5), 0.5)
    motion = _to_float(seg.get("motion_score", ctx.avg_motion), ctx.avg_motion)
    edge_density = _to_float(seg.get("edge_density", ctx.avg_edge), ctx.avg_edge)
    color_variance = _to_float(
        seg.get("color_variance", ctx.avg_color_variance),
        ctx.avg_color_variance,
    )
    wps = _to_float(seg.get("words_per_second", 0.0), 0.0)

    if screen_score is None:
        screen_score = _screen_like_score(
            self_sim,
            motion,
            ctx.avg_motion,
            edge_density,
            ctx.avg_edge,
            color_variance,
            ctx.avg_color_variance,
            silence,
            wps,
        )

    if layout_score is None:
        layout_score = _layout_anomaly_score(seg, ctx)

    ref_cv = max(ctx.avg_color_variance, 1e-6)
    cv_ratio = color_variance / ref_cv

    black_border = _first_float(
        seg,
        ("black_border_ratio", "letterbox_ratio", "border_black_ratio"),
        None,
    )
    active_w = _first_float(
        seg,
        ("active_width_ratio", "content_width_ratio", "inner_width_ratio"),
        None,
    )
    active_h = _first_float(
        seg,
        ("active_height_ratio", "content_height_ratio", "inner_height_ratio"),
        None,
    )

    score = 0.0

    # Static / held card behavior.
    if self_sim >= 0.94:
        score += 0.24
    elif self_sim >= 0.88:
        score += 0.12

    if motion <= ctx.avg_motion * 0.28:
        score += 0.22
    elif motion <= ctx.avg_motion * 0.45:
        score += 0.12

    # Simple / near-solid screens, including black/white title cards.
    if cv_ratio <= 0.45:
        score += 0.22
    elif cv_ratio <= 0.70:
        score += 0.12

    # Layout changes compared with main content.
    if layout_score >= 0.55:
        score += 0.22
    elif layout_score >= 0.35:
        score += 0.12

    # Strong black/end-screen or letterbox pattern change.
    if black_border is not None:
        main_bb = ctx.main_black_border_ratio if ctx.main_black_border_ratio is not None else 0.0
        if black_border >= max(0.45, main_bb + 0.18):
            score += 0.22
        elif black_border >= max(0.30, main_bb + 0.10):
            score += 0.12

    # Very small active visual area often means black card / end card.
    if active_w is not None and active_h is not None:
        if active_w <= 0.65 or active_h <= 0.65:
            score += 0.12

    # Text/logo/credits edges help only if the shot is already card-like.
    if (
        (self_sim >= 0.88 or motion <= ctx.avg_motion * 0.45 or cv_ratio <= 0.70)
        and edge_density > ctx.avg_edge * 1.12
    ):
        score += 0.08

    # Existing screen score is useful, but should not dominate by itself.
    if screen_score >= 0.72:
        score += 0.12
    elif screen_score >= 0.60:
        score += 0.06

    return max(0.0, min(1.0, score))


def _has_boundary_visual_evidence(
    seg: dict,
    ctx: _ClassifierContext,
    screen_score: float,
    layout_score: float,
    boundary_kw: float = 0.0,
) -> bool:
    """
    Decide whether a segment has enough visual evidence to be intro/outro.

    Policy:
      - BGM start alone is NOT outro.
      - Normal colorful/spoken animation scene is NOT outro.
      - Outro starts when the actual visual end card / black card / logo /
        credits-like screen begins, or when there is explicit outro wording.
    """
    if _has_speech(seg, ctx):
        return False

    if boundary_kw >= 0.18:
        return True

    silence = _to_float(seg.get("silence_ratio", 0.0), 0.0)
    active_audio = _has_active_audio(seg, ctx)
    card_score = _visual_boundary_card_score(seg, ctx, screen_score, layout_score)

    # Clear visual card evidence can override active music.
    if card_score >= 0.58:
        return True

    # Quiet/silent boundary screens can be accepted with weaker visual evidence.
    if silence >= 0.70 and (
        screen_score >= 0.32
        or layout_score >= 0.35
        or card_score >= 0.42
    ):
        return True

    # Layout-anomalous static screen, but not just because music started.
    if not active_audio and layout_score >= 0.50 and screen_score >= 0.24:
        return True

    return False


def _looks_like_main_content_with_audio(
    seg: dict,
    ctx: _ClassifierContext,
    screen_score: float,
    layout_score: float,
    boundary_kw: float = 0.0,
) -> bool:
    """
    Conservative guard against labeling normal video as intro/outro.

    Any transcript speech blocks boundary labels. If transcript is missing,
    normal main-content layout plus active audio also blocks boundary labels
    unless there is explicit boundary wording.
    """
    if _has_speech(seg, ctx):
        return True

    if boundary_kw >= 0.18:
        return False

    active_audio = _has_active_audio(seg, ctx)
    if not active_audio:
        return False

    main_layout_match = _matches_main_layout(seg, ctx) or layout_score <= 0.12

    if main_layout_match:
        return True

    # If we hear active audio but do not have strong visual evidence of a card,
    # keep it as content. This prevents BGM lead-ins and colorful spoken scenes
    # from becoming outro just because they are near the end.
    visual_card = _visual_boundary_card_score(seg, ctx, screen_score, layout_score)
    if visual_card < 0.58:
        return True

    return False


# ---------------------------------------------------------------------------
# Per-segment classifier
# ---------------------------------------------------------------------------

def _classify_single(seg: dict, ctx: _ClassifierContext) -> tuple[str, float]:
    """Return (label, confidence)."""

    start = seg["start_seconds"]
    end = seg["end_seconds"]
    dur = seg["duration_seconds"]

    td = ctx.total_duration

    # Audio features
    energy = seg.get("audio_energy", ctx.avg_energy)
    centroid = seg.get("spectral_centroid", ctx.avg_centroid)
    bandwidth = seg.get("spectral_bandwidth", ctx.avg_bandwidth)
    zcr = seg.get("zcr", ctx.avg_zcr)
    silence = seg.get("silence_ratio", 0.0)
    flatness = seg.get("spectral_flatness", ctx.avg_flatness)

    # Visual features
    motion = seg.get("motion_score", ctx.avg_motion)
    edge_density = seg.get("edge_density", ctx.avg_edge)
    color_variance = seg.get("color_variance", ctx.avg_color_variance)
    self_sim = seg.get("frame_self_similarity", 0.5)

    # Transcript/keyword features
    intro_kw = seg.get("intro_keyword_score", 0.0)
    outro_kw = seg.get("outro_keyword_score", 0.0)
    wps = seg.get("words_per_second", 0.0)

    rel_start = start / td if td > 0 else 0.0
    rel_end = end / td if td > 0 else 1.0

    freq_shift = abs(centroid - ctx.avg_centroid) / (ctx.avg_centroid + 1e-6)
    richness_diff = abs(bandwidth - ctx.avg_bandwidth) / (ctx.avg_bandwidth + 1e-6)
    energy_ratio = energy / (ctx.avg_energy + 1e-6)

    boundary_window = _boundary_window_seconds(td)
    intro_zone = boundary_window
    outro_zone = max(0.0, td - boundary_window)

    # Hard speech block.
    # If real speech starts, this should usually be main content.
    speech_block = _has_speech(seg, ctx)

    screen_score = _screen_like_score(
        self_sim,
        motion,
        ctx.avg_motion,
        edge_density,
        ctx.avg_edge,
        color_variance,
        ctx.avg_color_variance,
        silence,
        wps,
    )

    audio_score = _audio_boundary_score(
        energy_ratio,
        freq_shift,
        richness_diff,
        flatness,
        ctx.avg_flatness,
        zcr,
        ctx.avg_zcr,
    )

    layout_score = _layout_anomaly_score(seg, ctx)

    intro_content_guard = _looks_like_main_content_with_audio(
        seg,
        ctx,
        screen_score,
        layout_score,
        intro_kw,
    )
    outro_content_guard = _looks_like_main_content_with_audio(
        seg,
        ctx,
        screen_score,
        layout_score,
        outro_kw,
    )

    intro_visual_evidence = _has_boundary_visual_evidence(
        seg,
        ctx,
        screen_score,
        layout_score,
        intro_kw,
    )
    outro_visual_evidence = _has_boundary_visual_evidence(
        seg,
        ctx,
        screen_score,
        layout_score,
        outro_kw,
    )

    # Silence/simple screen should count positively.
    silence_bonus = 0.10 if silence >= 0.70 else (0.05 if silence >= 0.50 else 0.0)

    # -----------------------------------------------------------------------
    # INTRO
    # -----------------------------------------------------------------------
    if start < intro_zone and not speech_block and not intro_content_guard and intro_visual_evidence:
        intro_score = (
            0.45 * screen_score
            + 0.25 * audio_score
            + 0.22 * layout_score
            + 0.08 * min(1.0, intro_kw)
        )

        intro_score += silence_bonus

        if rel_start < 0.01:
            intro_score += 0.08

        has_support = (
            (screen_score >= 0.38 and audio_score >= 0.12)
            or (screen_score >= 0.34 and silence >= 0.70)
            or (
                screen_score >= 0.68
                and (silence >= 0.55 or audio_score >= 0.18 or layout_score >= 0.30 or intro_kw >= 0.18)
            )
            or (
                screen_score >= 0.30
                and edge_density > ctx.avg_edge * 1.12
                and silence >= 0.55
            )
            or (
                layout_score >= 0.45
                and (screen_score >= 0.22 or silence >= 0.45 or audio_score >= 0.10)
            )
            or (layout_score >= 0.65 and silence >= 0.30)
        )

        # Strong special case: silent/simple/title-like opening screen.
        if silence >= 0.88 and screen_score >= 0.30:
            intro_score += 0.10
            has_support = True

        if has_support and intro_score >= 0.32:
            max_intro_dur = min(60.0, max(10.0, td * 0.04))

            if dur <= max_intro_dur:
                return INTRO, min(0.95, 0.40 + intro_score * 0.50)

    # -----------------------------------------------------------------------
    # OUTRO
    # -----------------------------------------------------------------------
    if end > outro_zone and not speech_block and not outro_content_guard and outro_visual_evidence:
        outro_score = (
            0.45 * screen_score
            + 0.25 * audio_score
            + 0.22 * layout_score
            + 0.08 * min(1.0, outro_kw)
        )

        outro_score += silence_bonus

        if rel_end > 0.99:
            outro_score += 0.08

        if ctx.intro_audio is not None:
            outro_score += 0.14 * _audio_similarity_boost(
                centroid,
                bandwidth,
                energy,
                flatness,
                ctx.intro_audio,
            )

        # Important:
        # No standalone "(screen_score >= 0.56)" rule.
        # That caused colorful/static animation and stable documentary footage
        # to become OUTRO.
        has_support = (
            (screen_score >= 0.34 and audio_score >= 0.08 and silence >= 0.45)
            or (screen_score >= 0.34 and silence >= 0.60)
            or (
                screen_score >= 0.68
                and (
                    silence >= 0.55
                    or audio_score >= 0.18
                    or layout_score >= 0.30
                    or outro_kw >= 0.18
                )
            )
            or (
                screen_score >= 0.30
                and edge_density > ctx.avg_edge * 1.10
                and silence >= 0.45
            )
            or (
                layout_score >= 0.45
                and (screen_score >= 0.22 or silence >= 0.40 or audio_score >= 0.08)
            )
            or (layout_score >= 0.65 and silence >= 0.25)
        )

        if silence >= 0.88 and screen_score >= 0.28:
            outro_score += 0.10
            has_support = True

        if has_support and outro_score >= 0.30:
            max_outro_dur = min(60.0, max(10.0, td * 0.04))

            if dur <= max_outro_dur:
                return OUTRO, min(0.95, 0.40 + outro_score * 0.50)

    confidence = 1.0 - min(0.35, freq_shift * 0.35 + richness_diff * 0.20)
    return CONTENT, confidence


# ---------------------------------------------------------------------------
# Intro/Outro scoring helpers
# ---------------------------------------------------------------------------

def _screen_like_score(
    self_sim: float,
    motion: float,
    avg_motion: float,
    edge_density: float,
    avg_edge: float,
    color_variance: float,
    avg_color_variance: float,
    silence: float,
    wps: float,
) -> float:
    """
    Score how much a segment looks like a branded/title/end screen.

    Important cues:
      - visually static / highly self-similar
      - low motion
      - often large pure-color areas or simple background
      - often text/logo overlays
      - usually little speech

    Warning:
      This score alone should not decide OUTRO, because animation/main footage
      can also be stable/colorful.
    """
    score = 0.0

    # Static / repeated screen
    if self_sim > 0.93:
        score += 0.32
    elif self_sim > 0.87:
        score += 0.20

    # Low motion
    if motion < avg_motion * 0.35:
        score += 0.22
    elif motion < avg_motion * 0.55:
        score += 0.10

    # Pure / simple color background cue
    ref_cv = max(avg_color_variance, 1e-6)
    cv_ratio = color_variance / ref_cv

    if cv_ratio < 0.55:
        score += 0.22
    elif cv_ratio < 0.80:
        score += 0.12

    # Text/logo cue
    if edge_density > avg_edge * 1.30:
        score += 0.18
    elif edge_density > avg_edge * 1.10:
        score += 0.10

    # Quiet / low-speech behavior
    if silence > 0.50:
        score += 0.08

    if wps < 1.0:
        score += 0.10
    elif wps > 2.1:
        score -= 0.12

    return max(0.0, min(1.0, score))


def _screen_sequence_continuation(
    prev_seg: dict,
    seg: dict,
    ctx: _ClassifierContext,
) -> bool:
    """
    Allow multi-part intro/outro screen sequences even if the dominant color changes.

    Example:
      black screen -> mixed black/white card -> white title card
    """
    prev_screen = _screen_like_score(
        prev_seg.get("frame_self_similarity", 0.5),
        prev_seg.get("motion_score", ctx.avg_motion),
        ctx.avg_motion,
        prev_seg.get("edge_density", ctx.avg_edge),
        ctx.avg_edge,
        prev_seg.get("color_variance", ctx.avg_color_variance),
        ctx.avg_color_variance,
        prev_seg.get("silence_ratio", 0.0),
        prev_seg.get("words_per_second", 0.0),
    )

    cur_screen = _screen_like_score(
        seg.get("frame_self_similarity", 0.5),
        seg.get("motion_score", ctx.avg_motion),
        ctx.avg_motion,
        seg.get("edge_density", ctx.avg_edge),
        ctx.avg_edge,
        seg.get("color_variance", ctx.avg_color_variance),
        ctx.avg_color_variance,
        seg.get("silence_ratio", 0.0),
        seg.get("words_per_second", 0.0),
    )

    prev_layout = _layout_anomaly_score(prev_seg, ctx)
    cur_layout = _layout_anomaly_score(seg, ctx)

    return (
        not _has_speech(seg, ctx)
        and not _has_speech(prev_seg, ctx)
        and seg.get("duration_seconds", 0.0) < 40.0
        and (
            (prev_screen >= 0.42 and cur_screen >= 0.42)
            or (prev_layout >= 0.45 and cur_layout >= 0.35 and cur_screen >= 0.18)
        )
    )


def _audio_boundary_score(
    energy_ratio: float,
    freq_shift: float,
    richness_diff: float,
    flatness: float,
    avg_flatness: float,
    zcr: float,
    avg_zcr: float,
) -> float:
    """
    Score whether the audio feels different from the main body:
    music bumper, sting, silence bed, or branded sound.

    Important:
    This score helps, but BGM alone should never create OUTRO.
    """
    score = 0.0

    if freq_shift > 0.18:
        score += 0.18

    if richness_diff > 0.18:
        score += 0.16

    if flatness > avg_flatness * 1.20:
        score += 0.12

    if energy_ratio < 0.70 or energy_ratio > 1.35:
        score += 0.10

    if avg_zcr > 1e-6:
        zcr_ratio = zcr / avg_zcr

        if zcr_ratio < 0.70 or zcr_ratio > 1.35:
            score += 0.08

    return max(0.0, min(1.0, score))


def _audio_continuation_score(
    centroid: float,
    bandwidth: float,
    energy: float,
    flatness: float,
    silence_ratio: float,
    zcr: float,
    ref_audio: dict | None,
) -> float:
    """
    Score whether a neighboring segment continues the same intro/outro audio state.

    Supports:
      - dead silence across multiple title-card segments
      - same music / same bumper bed while visuals change

    Important:
    This should only extend boundary screens, not normal BGM lead-in footage.
    """
    if ref_audio is None:
        return 0.0

    ref_centroid = ref_audio.get("centroid", centroid)
    ref_bandwidth = ref_audio.get("bandwidth", bandwidth)
    ref_energy = ref_audio.get("energy", energy)
    ref_flatness = ref_audio.get("flatness", flatness)
    ref_silence = ref_audio.get("silence_ratio", silence_ratio)
    ref_zcr = ref_audio.get("zcr", zcr)

    c_diff = abs(centroid - ref_centroid) / (ref_centroid + 1e-6)
    b_diff = abs(bandwidth - ref_bandwidth) / (ref_bandwidth + 1e-6)
    e_diff = abs(energy - ref_energy) / (ref_energy + 1e-6)
    f_diff = abs(flatness - ref_flatness) / (ref_flatness + 1e-6)
    s_diff = abs(silence_ratio - ref_silence)
    z_diff = abs(zcr - ref_zcr) / (ref_zcr + 1e-6)

    # Case 1: same silence / near-silence state
    if silence_ratio >= 0.88 and ref_silence >= 0.88:
        silence_base = 0.48
    elif silence_ratio >= 0.70 and ref_silence >= 0.70:
        silence_base = 0.30
    else:
        silence_base = 0.0

    # Case 2: same music / same bumper-like audio state
    avg_diff = (c_diff + b_diff + e_diff + f_diff + z_diff) / 5.0

    if avg_diff < 0.16:
        music_sim = 0.52
    elif avg_diff < 0.28:
        music_sim = 0.34
    elif avg_diff < 0.42:
        music_sim = 0.18
    else:
        music_sim = 0.0

    if silence_ratio < 0.60 and ref_silence < 0.60 and e_diff < 0.30:
        music_sim += 0.06

    if s_diff < 0.18:
        music_sim += 0.04

    return max(0.0, min(1.0, max(silence_base, music_sim)))


def _audio_similarity_boost(
    centroid: float,
    bandwidth: float,
    energy: float,
    flatness: float,
    intro_audio: dict,
) -> float:
    """
    Bonus if this segment's audio profile is similar to intro audio.
    Intro and outro often share the same music.

    This is only a bonus, never enough by itself.
    """
    c_diff = abs(centroid - intro_audio["centroid"]) / (intro_audio["centroid"] + 1e-6)
    b_diff = abs(bandwidth - intro_audio["bandwidth"]) / (intro_audio["bandwidth"] + 1e-6)
    e_diff = abs(energy - intro_audio["energy"]) / (intro_audio["energy"] + 1e-6)
    f_diff = abs(flatness - intro_audio["flatness"]) / (intro_audio["flatness"] + 1e-6)

    avg_diff = (c_diff + b_diff + e_diff + f_diff) / 4.0

    if avg_diff < 0.15:
        return 0.25
    elif avg_diff < 0.30:
        return 0.15
    elif avg_diff < 0.50:
        return 0.05

    return 0.0


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def merge_consecutive_segments(segments: list[dict]) -> list[dict]:
    """
    Combine adjacent segments of the same type while preserving speech evidence.

    This is important because a merged OUTRO should not hide the fact that
    one or more child chunks had transcript speech.
    """
    if not segments:
        return []

    def absorb_features(dst: dict, src: dict):
        # Preserve transcript/speech evidence.
        dst_wc = _to_int(dst.get("word_count", 0), 0)
        src_wc = _to_int(src.get("word_count", 0), 0)
        dst["word_count"] = dst_wc + src_wc

        txt1 = str(dst.get("transcript_text", "") or "").strip()
        txt2 = str(src.get("transcript_text", "") or "").strip()
        if txt2:
            dst["transcript_text"] = (txt1 + " " + txt2).strip() if txt1 else txt2

        dur = max(_to_float(dst.get("duration_seconds", 0.0), 0.0), 1e-6)
        dst["words_per_second"] = float(dst.get("word_count", 0) or 0) / dur

        # Preserve keyword evidence.
        for key in (
            "intro_keyword_score",
            "outro_keyword_score",
            "sponsor_score",
            "self_promo_score",
            "recap_score",
        ):
            dst[key] = max(
                _to_float(dst.get(key, 0.0), 0.0),
                _to_float(src.get(key, 0.0), 0.0),
            )

        # Preserve active-audio evidence.
        # Lower silence ratio and higher energy means "some part had active audio".
        if "silence_ratio" in src:
            dst["silence_ratio"] = min(
                _to_float(dst.get("silence_ratio", 1.0), 1.0),
                _to_float(src.get("silence_ratio", 1.0), 1.0),
            )

        if "audio_energy" in src:
            dst["audio_energy"] = max(
                _to_float(dst.get("audio_energy", 0.0), 0.0),
                _to_float(src.get("audio_energy", 0.0), 0.0),
            )

        # Preserve whether all merged children were too short.
        dst["audio_too_short"] = bool(dst.get("audio_too_short", False)) and bool(
            src.get("audio_too_short", False)
        )

        # Preserve some visual/main-content evidence conservatively.
        # Higher motion makes it less card-like.
        if "motion_score" in src:
            dst["motion_score"] = max(
                _to_float(dst.get("motion_score", 0.0), 0.0),
                _to_float(src.get("motion_score", 0.0), 0.0),
            )

        # If any child has normal-looking black border ratio, keep the smaller
        # value; this helps avoid a whole merged segment becoming "black card"
        # just because the first child was black.
        for key in (
            "black_border_ratio",
            "letterbox_ratio",
            "border_black_ratio",
        ):
            if key in src:
                dst[key] = min(
                    _to_float(dst.get(key, 1.0), 1.0),
                    _to_float(src.get(key, 1.0), 1.0),
                )

    merged = []
    current = segments[0].copy()

    for i in range(1, len(segments)):
        nxt = segments[i]

        if nxt["segment_type"] == current["segment_type"]:
            current["end_seconds"] = nxt["end_seconds"]
            current["duration_seconds"] = current["end_seconds"] - current["start_seconds"]
            current["confidence"] = min(current["confidence"], nxt.get("confidence", 1.0))
            absorb_features(current, nxt)
        else:
            merged.append(current)
            current = nxt.copy()

    merged.append(current)
    return merged


def _cascade_intro(segments: list[dict], ctx: _ClassifierContext) -> list[dict]:
    """
    If intro exists, extend it forward until clear speech appears,
    as long as neighboring segments still look like opening screen/bumper material
    or continue the same intro audio/layout state.
    """
    if not segments or segments[0]["segment_type"] != INTRO:
        return segments

    intro_seg = segments[0].copy()
    result = [intro_seg]

    intro_end = intro_seg["end_seconds"]
    max_intro = min(60.0, max(10.0, ctx.total_duration * 0.04))

    stopped = False

    for seg in segments[1:]:
        if stopped or intro_end >= max_intro:
            result.append(seg)
            continue

        if seg["start_seconds"] > max_intro:
            result.append(seg)
            stopped = True
            continue

        wps = seg.get("words_per_second", 0.0)

        # Hard stop: once real speech appears, intro ends.
        if _has_speech(seg, ctx):
            result.append(seg)
            stopped = True
            continue

        silence = seg.get("silence_ratio", 0.0)
        self_sim = seg.get("frame_self_similarity", 0.5)
        motion = seg.get("motion_score", ctx.avg_motion)
        edge_density = seg.get("edge_density", ctx.avg_edge)
        color_variance = seg.get("color_variance", ctx.avg_color_variance)
        intro_kw = seg.get("intro_keyword_score", 0.0)

        screen_score = _screen_like_score(
            self_sim,
            motion,
            ctx.avg_motion,
            edge_density,
            ctx.avg_edge,
            color_variance,
            ctx.avg_color_variance,
            silence,
            wps,
        )

        layout_score = _layout_anomaly_score(seg, ctx)
        main_layout_match = _matches_main_layout(seg, ctx)

        audio_cont = _audio_continuation_score(
            seg.get("spectral_centroid", ctx.avg_centroid),
            seg.get("spectral_bandwidth", ctx.avg_bandwidth),
            seg.get("audio_energy", ctx.avg_energy),
            seg.get("spectral_flatness", ctx.avg_flatness),
            silence,
            seg.get("zcr", ctx.avg_zcr),
            ctx.intro_audio,
        )

        continuation_score = (
            0.42 * screen_score
            + 0.28 * audio_cont
            + 0.20 * layout_score
            + 0.10 * min(1.0, intro_kw)
        )

        # If visual layout has returned to dominant main-content layout,
        # stop intro even if audio is still quiet/music-like.
        if main_layout_match and screen_score < 0.56 and audio_cont < 0.40:
            result.append(seg)
            stopped = True
            continue

        visual_evidence = _has_boundary_visual_evidence(
            seg,
            ctx,
            screen_score,
            layout_score,
            intro_kw,
        )

        if not visual_evidence:
            result.append(seg)
            stopped = True
            continue

        if (
            (
                screen_score >= 0.26
                and (audio_cont >= 0.16 or silence >= 0.60)
            )
            or screen_score >= 0.58
            or audio_cont >= 0.40
            or (
                layout_score >= 0.45
                and (screen_score >= 0.20 or audio_cont >= 0.12 or silence >= 0.45)
            )
            or _screen_sequence_continuation(result[0], seg, ctx)
        ) and continuation_score >= 0.20:
            result[0]["end_seconds"] = seg["end_seconds"]
            result[0]["duration_seconds"] = (
                result[0]["end_seconds"] - result[0]["start_seconds"]
            )
            intro_end = seg["end_seconds"]
        else:
            result.append(seg)
            stopped = True

    return result


def _cascade_outro(segments: list[dict], ctx: _ClassifierContext) -> list[dict]:
    """
    Extend outro backward across neighboring ending segments only if they still look
    like end-card/credits/branded closing material.

    Important:
    BGM lead-in before black/end screen should remain content.
    """
    if not segments or segments[-1]["segment_type"] != OUTRO:
        return segments

    result = list(segments[:-1])
    outro_seg = segments[-1].copy()

    max_outro = min(45.0, max(10.0, ctx.total_duration * 0.035))
    ref_audio = ctx.intro_audio

    absorbed = []

    for seg in reversed(result):
        outro_dur = outro_seg["end_seconds"] - seg["start_seconds"]

        if outro_dur > max_outro:
            break

        if seg["end_seconds"] < ctx.total_duration - max_outro:
            break

        wps = seg.get("words_per_second", 0.0)

        if _has_speech(seg, ctx):
            break

        outro_kw = seg.get("outro_keyword_score", 0.0)
        sponsor_kw = seg.get("sponsor_score", 0.0)
        promo_kw = seg.get("self_promo_score", 0.0)

        silence = seg.get("silence_ratio", 0.0)
        self_sim = seg.get("frame_self_similarity", 0.5)
        motion = seg.get("motion_score", ctx.avg_motion)
        edge_density = seg.get("edge_density", ctx.avg_edge)
        color_variance = seg.get("color_variance", ctx.avg_color_variance)

        screen_score = _screen_like_score(
            self_sim,
            motion,
            ctx.avg_motion,
            edge_density,
            ctx.avg_edge,
            color_variance,
            ctx.avg_color_variance,
            silence,
            wps,
        )

        layout_score = _layout_anomaly_score(seg, ctx)
        main_layout_match = _matches_main_layout(seg, ctx)

        audio_match = _audio_continuation_score(
            seg.get("spectral_centroid", ctx.avg_centroid),
            seg.get("spectral_bandwidth", ctx.avg_bandwidth),
            seg.get("audio_energy", ctx.avg_energy),
            seg.get("spectral_flatness", ctx.avg_flatness),
            silence,
            seg.get("zcr", ctx.avg_zcr),
            ref_audio,
        )

        score = (
            0.46 * screen_score
            + 0.24 * audio_match
            + 0.20 * layout_score
            + 0.10 * min(1.0, outro_kw)
        )

        if sponsor_kw > 0.0 or promo_kw > 0.0:
            score -= 0.25

        visual_evidence = _has_boundary_visual_evidence(
            seg,
            ctx,
            screen_score,
            layout_score,
            outro_kw,
        )

        # If this still looks/sounds like normal main content, do not absorb it.
        if _looks_like_main_content_with_audio(seg, ctx, screen_score, layout_score, outro_kw):
            break

        # BGM/audio continuity alone is not enough.
        if not visual_evidence:
            break

        if main_layout_match and screen_score < 0.54 and audio_match < 0.36 and outro_kw < 0.18:
            break

        if (
            (
                screen_score >= 0.30
                and (audio_match >= 0.18 or outro_kw >= 0.18 or silence >= 0.60)
            )
            or (
                screen_score >= 0.68
                and (
                    silence >= 0.55
                    or audio_match >= 0.18
                    or layout_score >= 0.30
                    or outro_kw >= 0.18
                )
            )
            or (
                layout_score >= 0.45
                and (
                    screen_score >= 0.20
                    or audio_match >= 0.12
                    or silence >= 0.40
                    or outro_kw >= 0.10
                )
            )
        ) and score >= 0.26 and seg["duration_seconds"] < 55:
            absorbed.append(seg)
            outro_seg["start_seconds"] = seg["start_seconds"]
            outro_seg["duration_seconds"] = (
                outro_seg["end_seconds"] - outro_seg["start_seconds"]
            )
        else:
            break

    for a in absorbed:
        result.remove(a)

    result.append(outro_seg)
    return result


def _merge_noncontent_gaps(
    segments: list[dict],
    max_gap: float = 4.0,
) -> list[dict]:
    """
    Merge only:
      intro -> short content -> intro
      outro -> short content -> outro

    The gap is intentionally small. A large gap often means real content.
    """
    if len(segments) < 3:
        return segments

    result = [segments[0].copy()]
    i = 1

    while i < len(segments):
        seg = segments[i]

        if (
            i + 1 < len(segments)
            and seg["is_content"]
            and seg["duration_seconds"] <= max_gap
            and not result[-1]["is_content"]
            and not segments[i + 1]["is_content"]
            and result[-1]["segment_type"] == segments[i + 1]["segment_type"]
            and result[-1]["segment_type"] in (INTRO, OUTRO)
            and not _has_speech(seg, _DummyContextProxy())
        ):
            nxt = segments[i + 1]
            result[-1]["end_seconds"] = nxt["end_seconds"]
            result[-1]["duration_seconds"] = (
                result[-1]["end_seconds"] - result[-1]["start_seconds"]
            )
            result[-1]["confidence"] = min(
                result[-1]["confidence"],
                nxt.get("confidence", 1.0),
            )
            i += 2
        else:
            result.append(seg.copy())
            i += 1

    return result


class _DummyContextProxy:
    """
    Small helper so _has_speech can be used in gap merge without needing full ctx.
    """
    has_transcript = False


def _refine_short_noncontent(
    segments: list[dict],
    min_nc_duration: float = 3.0,
) -> list[dict]:
    """
    Absorb very short non-content segments back into neighboring content.

    For intro/outro-only mode, this mostly exists as a safety no-op because
    intro/outro are handled by boundary rules.
    """
    if len(segments) <= 1:
        return segments

    refined = []

    for seg in segments:
        if (
            not seg["is_content"]
            and seg["duration_seconds"] < min_nc_duration
            and seg["segment_type"] not in (INTRO, OUTRO)
        ):
            if refined and refined[-1]["is_content"]:
                refined[-1]["end_seconds"] = seg["end_seconds"]
                refined[-1]["duration_seconds"] = (
                    refined[-1]["end_seconds"] - refined[-1]["start_seconds"]
                )
                continue

        refined.append(seg)

    return refined


def _is_screenish_enough(seg: dict, ctx: _ClassifierContext) -> float:
    return _screen_like_score(
        seg.get("frame_self_similarity", 0.5),
        seg.get("motion_score", ctx.avg_motion),
        ctx.avg_motion,
        seg.get("edge_density", ctx.avg_edge),
        ctx.avg_edge,
        seg.get("color_variance", ctx.avg_color_variance),
        ctx.avg_color_variance,
        seg.get("silence_ratio", 0.0),
        seg.get("words_per_second", 0.0),
    )


def _bridge_boundary_screen_gaps(
    segments: list[dict],
    ctx: _ClassifierContext,
    max_gap_sec: float = 3.5,
) -> list[dict]:
    """
    Bridge short boundary gaps inside a multi-part intro/outro sequence.

    Example:
      INTRO -> CONTENT(short mixed black/white frame) -> INTRO
    becomes:
      INTRO
    """
    if len(segments) < 3:
        return segments

    fixed = [s.copy() for s in segments]

    for i in range(1, len(fixed) - 1):
        prev_seg = fixed[i - 1]
        cur_seg = fixed[i]
        next_seg = fixed[i + 1]

        prev_label = prev_seg.get("segment_type", CONTENT)
        cur_label = cur_seg.get("segment_type", CONTENT)
        next_label = next_seg.get("segment_type", CONTENT)

        if cur_seg["duration_seconds"] > max_gap_sec:
            continue

        if _has_speech(cur_seg, ctx):
            continue

        prev_screen = _is_screenish_enough(prev_seg, ctx)
        cur_screen = _is_screenish_enough(cur_seg, ctx)
        next_screen = _is_screenish_enough(next_seg, ctx)

        cur_layout = _layout_anomaly_score(cur_seg, ctx)
        cur_intro_kw = _to_float(cur_seg.get("intro_keyword_score", 0.0), 0.0)
        cur_outro_kw = _to_float(cur_seg.get("outro_keyword_score", 0.0), 0.0)

        cur_intro_visual = _has_boundary_visual_evidence(
            cur_seg,
            ctx,
            cur_screen,
            cur_layout,
            cur_intro_kw,
        )
        cur_outro_visual = _has_boundary_visual_evidence(
            cur_seg,
            ctx,
            cur_screen,
            cur_layout,
            cur_outro_kw,
        )

        boundary_window = _boundary_window_seconds(ctx.total_duration)
        intro_boundary = boundary_window
        outro_boundary = max(0.0, ctx.total_duration - boundary_window)

        # Intro-side bridging near the beginning
        if (
            prev_label == INTRO
            and next_label == INTRO
            and cur_label == CONTENT
            and next_seg["start_seconds"] <= intro_boundary
        ):
            if (
                cur_intro_visual
                and (
                    cur_screen >= 0.22
                    or cur_layout >= 0.35
                    or (prev_screen >= 0.42 and next_screen >= 0.42)
                )
            ):
                cur_seg["segment_type"] = INTRO
                cur_seg["segment_label"] = INTRO
                cur_seg["is_content"] = False
                cur_seg["confidence"] = max(0.55, cur_seg.get("confidence", 0.55))
                continue

        # Outro-side bridging near the end
        if (
            prev_label == OUTRO
            and next_label == OUTRO
            and cur_label == CONTENT
            and prev_seg["start_seconds"] >= outro_boundary
        ):
            if (
                cur_outro_visual
                and (
                    cur_screen >= 0.22
                    or cur_layout >= 0.35
                    or (prev_screen >= 0.40 and next_screen >= 0.40)
                )
            ):
                cur_seg["segment_type"] = OUTRO
                cur_seg["segment_label"] = OUTRO
                cur_seg["is_content"] = False
                cur_seg["confidence"] = max(0.55, cur_seg.get("confidence", 0.55))
                continue

    return fixed


def _enforce_boundary_intro_outro_only(
    segments: list[dict],
    ctx: _ClassifierContext,
) -> list[dict]:
    """
    Enforce structural rule:

      - intro may appear only as a prefix at the beginning
      - outro may appear only as a suffix at the end
      - if beginning has no intro, fine
      - if end has no outro, fine
    """
    if not segments:
        return segments

    fixed = [s.copy() for s in segments]

    boundary_window = _boundary_window_seconds(ctx.total_duration)
    intro_zone = boundary_window
    outro_zone_start = max(0.0, ctx.total_duration - boundary_window)

    # Pass 1: intro may only appear in initial prefix.
    left_content_started = False

    for i, seg in enumerate(fixed):
        label = seg.get("segment_type", CONTENT)

        if label == INTRO:
            if seg["start_seconds"] > intro_zone:
                seg["segment_type"] = CONTENT
                seg["segment_label"] = CONTENT
                seg["is_content"] = True
                continue

            if left_content_started:
                seg["segment_type"] = CONTENT
                seg["segment_label"] = CONTENT
                seg["is_content"] = True
                continue

        if seg.get("segment_type") == CONTENT:
            left_content_started = True
        elif i > 0 and fixed[i - 1].get("segment_type") == INTRO and label != INTRO:
            left_content_started = True

    # Pass 2: outro may only appear in final suffix.
    right_content_seen = False

    for i in range(len(fixed) - 1, -1, -1):
        seg = fixed[i]
        label = seg.get("segment_type", CONTENT)

        if label == OUTRO:
            if seg["end_seconds"] < outro_zone_start:
                seg["segment_type"] = CONTENT
                seg["segment_label"] = CONTENT
                seg["is_content"] = True
                continue

            if right_content_seen:
                seg["segment_type"] = CONTENT
                seg["segment_label"] = CONTENT
                seg["is_content"] = True
                continue

        if seg.get("segment_type") == CONTENT:
            right_content_seen = True

    return fixed


def _apply_boundary_guards(
    segments: list[dict],
    intro_trim_sec: float = 0.0,
    outro_trim_sec: float = 0.0,
) -> list[dict]:
    """
    Boundary guard disabled.

    Earlier versions intentionally moved:
      - intro end earlier
      - outro start later

    That is safer for avoiding speech clipping, but it causes the timing issue
    where intro ends too early and outro starts too late.

    If your output clips speech slightly, you can restore small values like:
      intro_trim_sec=0.3
      outro_trim_sec=0.3
    """
    return [seg.copy() for seg in segments]


def _demote_speech_or_main_content_boundary_segments(
    segments: list[dict],
    ctx: _ClassifierContext,
) -> list[dict]:
    """
    Final safety pass.

    If a boundary segment contains transcript speech or looks like normal
    main-content layout with active audio, demote it back to content.
    This prevents cases like a final normal spoken scene being marked as OUTRO.
    """
    fixed = []

    for seg in segments:
        label = seg.get("segment_type", CONTENT)

        if label not in (INTRO, OUTRO):
            fixed.append(seg.copy())
            continue

        silence = _to_float(seg.get("silence_ratio", 0.0), 0.0)
        wps = _to_float(seg.get("words_per_second", 0.0), 0.0)
        boundary_kw = _to_float(
            seg.get("intro_keyword_score", 0.0)
            if label == INTRO
            else seg.get("outro_keyword_score", 0.0),
            0.0,
        )

        screen_score = _screen_like_score(
            seg.get("frame_self_similarity", 0.5),
            seg.get("motion_score", ctx.avg_motion),
            ctx.avg_motion,
            seg.get("edge_density", ctx.avg_edge),
            ctx.avg_edge,
            seg.get("color_variance", ctx.avg_color_variance),
            ctx.avg_color_variance,
            silence,
            wps,
        )

        layout_score = _layout_anomaly_score(seg, ctx)
        visual_evidence = _has_boundary_visual_evidence(
            seg,
            ctx,
            screen_score,
            layout_score,
            boundary_kw,
        )

        should_demote = (
            _has_speech(seg, ctx)
            or _looks_like_main_content_with_audio(
                seg,
                ctx,
                screen_score,
                layout_score,
                boundary_kw,
            )
            or not visual_evidence
        )

        if should_demote:
            s = seg.copy()
            s["segment_type"] = CONTENT
            s["segment_label"] = CONTENT
            s["is_content"] = True
            s["confidence"] = min(s.get("confidence", 1.0), 0.50)
            fixed.append(s)
        else:
            fixed.append(seg.copy())

    return fixed


def _drop_tiny_boundary_noncontent(
    segments: list[dict],
    min_duration: float = 0.6,
) -> list[dict]:
    """
    Remove near-zero intro/outro artifacts.

    This is intentionally much smaller than a normal short-segment cleanup.
    It removes weird boundary fragments without deleting real short title cards.
    """
    if not segments:
        return segments

    fixed = []

    for seg in segments:
        label = seg.get("segment_type", CONTENT)
        dur = _to_float(seg.get("duration_seconds", 0.0), 0.0)

        if label in (INTRO, OUTRO) and dur < min_duration:
            s = seg.copy()
            s["segment_type"] = CONTENT
            s["segment_label"] = CONTENT
            s["is_content"] = True
            s["confidence"] = min(_to_float(s.get("confidence", 1.0), 1.0), 0.50)
            fixed.append(s)
        else:
            fixed.append(seg.copy())

    return fixed