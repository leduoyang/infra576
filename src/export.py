"""
export.py – Builds the JSON output schema from classified segments.

Produces output compatible with both:
  - Legacy schema (inserted_ads, ad_index, ad_filename, original_video_start/end)
  - New taxonomy schema (segment_label, is_content, confidence, non_content_summary)

The player and evaluator scripts work with either format.
"""

import os
from src.ingest import seconds_to_formatted


def build_output(
    video_path: str,
    video_metadata: dict,
    classified_segments: list[dict],
) -> dict:
    video_filename = os.path.basename(video_path)
    total_duration = video_metadata["duration_seconds"]
    resolution = video_metadata["resolution"]

    timeline_segments = _build_timeline(classified_segments)
    inserted_ads = _build_inserted_ads(classified_segments)
    nc_summary = _build_noncontent_summary(classified_segments)
    chapters = _build_chapters(classified_segments)

    total_nc_dur = sum(
        seg["duration_seconds"]
        for seg in classified_segments
        if not seg.get("is_content", True)
    )
    total_ad_dur = sum(ad["ad_duration_seconds"] for ad in inserted_ads)
    content_duration = total_duration - total_nc_dur

    return {
        # ── Legacy-compatible top-level fields ──
        "video_filename": video_filename,
        "original_video_duration_seconds": round(content_duration, 3),
        "original_video_duration_formatted": seconds_to_formatted(content_duration),
        "original_video_resolution": resolution,
        "output_filename": video_filename,
        "output_duration_seconds": round(total_duration, 3),
        "output_duration_formatted": seconds_to_formatted(total_duration),
        "total_ads_duration_seconds": round(total_ad_dur, 3),
        "num_ads_inserted": len(inserted_ads),

        # ── New taxonomy fields ──
        "content_duration_seconds": round(content_duration, 3),
        "total_noncontent_duration_seconds": round(total_nc_dur, 3),
        "num_segments": len(timeline_segments),
        "non_content_summary": nc_summary,

        # ── Segment arrays ──
        "inserted_ads": inserted_ads,
        "timeline_segments": timeline_segments,
        "chapters": chapters,
    }


# ---------------------------------------------------------------------------
# Timeline builder – includes both legacy and new fields per segment
# ---------------------------------------------------------------------------

def _build_timeline(classified_segments: list[dict]) -> list[dict]:
    timeline = []
    content_idx = 0
    nc_idx = 1
    acc_nc_dur = 0.0

    for seg in classified_segments:
        start = seg["start_seconds"]
        end = seg["end_seconds"]
        duration = seg["duration_seconds"]
        is_content = seg.get("is_content", True)
        label = seg.get("segment_label", seg.get("segment_type", "content"))

        orig_start = start - acc_nc_dur

        if is_content:
            orig_end = orig_start + duration
            entry = {
                # Legacy fields
                "type": "video_content",
                "segment_index": content_idx,
                "original_video_start_seconds": round(orig_start, 3),
                "original_video_end_seconds": round(orig_end, 3),
                # New taxonomy fields
                "segment_label": label,
                "is_content": True,
                "confidence": round(seg.get("confidence", 1.0), 3),
                # Timing (shared)
                "final_video_start_seconds": round(start, 3),
                "final_video_start_formatted": seconds_to_formatted(start),
                "final_video_end_seconds": round(end, 3),
                "final_video_end_formatted": seconds_to_formatted(end),
                "duration_seconds": round(duration, 3),
                # Skip suggestion
                "skip_suggestion": False,
            }
            content_idx += 1
        else:
            acc_nc_dur += duration
            entry = {
                # Legacy fields
                "type": "ad" if label in ("ad", "self_promotion") else label,
                "ad_index": nc_idx,
                "ad_filename": f"detected_{label}_{nc_idx:03d}.mp4",
                # New taxonomy fields
                "segment_label": label,
                "is_content": False,
                "confidence": round(seg.get("confidence", 1.0), 3),
                # Timing (shared)
                "final_video_start_seconds": round(start, 3),
                "final_video_start_formatted": seconds_to_formatted(start),
                "final_video_end_seconds": round(end, 3),
                "final_video_end_formatted": seconds_to_formatted(end),
                "duration_seconds": round(duration, 3),
                # Skip suggestion
                "skip_suggestion": True,
                "skip_reason": _skip_reason(label),
            }
            nc_idx += 1

        timeline.append(entry)

    return timeline


def _skip_reason(label: str) -> str:
    reasons = {
        "intro": "Opening sequence",
        "outro": "Closing / end card",
        "ad": "Advertisement / sponsorship",
        "self_promotion": "Channel self-promotion",
        "recap": "Repeated recap",
        "transition": "Transition screen",
        "dead_air": "Silence / inactivity",
    }
    return reasons.get(label, "Non-content segment")


# ---------------------------------------------------------------------------
# Inserted-ads list (legacy compat)
# ---------------------------------------------------------------------------

def _build_inserted_ads(classified_segments: list[dict]) -> list[dict]:
    ads = []
    nc_idx = 1
    acc_nc_dur = 0.0

    for seg in classified_segments:
        start = seg["start_seconds"]
        end = seg["end_seconds"]
        duration = seg["duration_seconds"]
        is_content = seg.get("is_content", True)
        label = seg.get("segment_label", "ad")

        if not is_content:
            orig_insert = start - acc_nc_dur
            ads.append({
                "ad_index": nc_idx,
                "ad_filename": f"detected_{label}_{nc_idx:03d}.mp4",
                "ad_duration_seconds": round(duration, 3),
                "ad_type": label,
                "original_video_insert_at_seconds": round(orig_insert, 3),
                "original_video_insert_at_formatted": seconds_to_formatted(orig_insert),
                "final_video_ad_start_seconds": round(start, 3),
                "final_video_ad_start_formatted": seconds_to_formatted(start),
                "final_video_ad_end_seconds": round(end, 3),
                "final_video_ad_end_formatted": seconds_to_formatted(end),
            })
            nc_idx += 1
            acc_nc_dur += duration

    return ads


# ---------------------------------------------------------------------------
# Non-content summary (new)
# ---------------------------------------------------------------------------

def _build_noncontent_summary(classified_segments: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for seg in classified_segments:
        if seg.get("is_content", True):
            continue
        label = seg.get("segment_label", "unknown")
        if label not in summary:
            summary[label] = {"count": 0, "total_duration_seconds": 0.0}
        summary[label]["count"] += 1
        summary[label]["total_duration_seconds"] += seg["duration_seconds"]
    for v in summary.values():
        v["total_duration_seconds"] = round(v["total_duration_seconds"], 3)
    return summary


# ---------------------------------------------------------------------------
# Chapter markers (new)
# ---------------------------------------------------------------------------

def _build_chapters(classified_segments: list[dict]) -> list[dict]:
    """Generate chapter markers from classified segments."""
    chapters = []
    chap_idx = 1
    for seg in classified_segments:
        start = seg["start_seconds"]
        label = seg.get("segment_label", seg.get("segment_type", "content"))
        is_content = seg.get("is_content", True)

        if is_content:
            title = f"Content part {chap_idx}"
            chap_idx += 1
        else:
            title = _chapter_title(label)

        chapters.append({
            "title": title,
            "start_seconds": round(start, 3),
            "start_formatted": seconds_to_formatted(start),
            "is_content": is_content,
            "segment_label": label,
        })
    return chapters


def _chapter_title(label: str) -> str:
    titles = {
        "intro": "Intro",
        "outro": "Outro",
        "ad": "Ad break",
        "self_promotion": "Channel promo",
        "recap": "Recap",
        "transition": "Transition",
        "dead_air": "Dead air",
    }
    return titles.get(label, "Non-content")
