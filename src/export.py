"""
export.py – Builds the JSON output schema from classified segments.

Produces output matching the sample schema exactly:
- video_filename, original_video_duration_seconds, original_video_resolution
- output_filename, output_duration_seconds, total_ads_duration_seconds, num_ads_inserted
- inserted_ads
- timeline_segments (type = ad | video_content)
"""

import os
from src.ingest import seconds_to_formatted

def build_output(
    video_path: str,
    video_metadata: dict,
    classified_segments: list[dict],
) -> dict:
    video_filename = os.path.basename(video_path)
    output_duration = video_metadata["duration_seconds"]
    resolution = video_metadata["resolution"]

    timeline_segments = _build_timeline(classified_segments)
    inserted_ads = _build_inserted_ads(classified_segments)

    total_ads_duration = sum(ad["ad_duration_seconds"] for ad in inserted_ads)
    original_duration = output_duration - total_ads_duration

    return {
        "video_filename": video_filename,
        "original_video_duration_seconds": round(original_duration, 3),
        "original_video_duration_formatted": seconds_to_formatted(original_duration),
        "original_video_resolution": resolution,
        "output_filename": video_filename,
        "output_duration_seconds": round(output_duration, 3),
        "output_duration_formatted": seconds_to_formatted(output_duration),
        "total_ads_duration_seconds": round(total_ads_duration, 3),
        "num_ads_inserted": len(inserted_ads),
        "inserted_ads": inserted_ads,
        "timeline_segments": timeline_segments,
    }


def _build_timeline(classified_segments: list[dict]) -> list[dict]:
    timeline = []
    content_idx = 0
    nc_idx = 1
    acc_ad_dur = 0.0

    for seg in classified_segments:
        start = seg["start_seconds"]
        end = seg["end_seconds"]
        duration = seg["duration_seconds"]

        orig_start = start - acc_ad_dur

        if seg["is_content"]:
            orig_end = orig_start + duration
            entry = {
                "type": "video_content",
                "segment_index": content_idx,
                "final_video_start_seconds": round(start, 3),
                "final_video_start_formatted": seconds_to_formatted(start),
                "final_video_end_seconds": round(end, 3),
                "final_video_end_formatted": seconds_to_formatted(end),
                "duration_seconds": round(duration, 3),
                "original_video_start_seconds": round(orig_start, 3),
                "original_video_end_seconds": round(orig_end, 3),
            }
            content_idx += 1
        else:
            acc_ad_dur += duration
            entry = {
                "type": "ad",
                "ad_index": nc_idx,
                "ad_filename": f"detected_ad_{nc_idx:03d}.mp4",
                "final_video_start_seconds": round(start, 3),
                "final_video_start_formatted": seconds_to_formatted(start),
                "final_video_end_seconds": round(end, 3),
                "final_video_end_formatted": seconds_to_formatted(end),
                "duration_seconds": round(duration, 3),
            }
            nc_idx += 1

        timeline.append(entry)

    return timeline


def _build_inserted_ads(classified_segments: list[dict]) -> list[dict]:
    ads = []
    nc_idx = 1
    acc_ad_dur = 0.0

    for seg in classified_segments:
        start = seg["start_seconds"]
        end = seg["end_seconds"]
        duration = seg["duration_seconds"]
        
        if not seg["is_content"]:
            orig_insert = start - acc_ad_dur
            ads.append({
                "ad_index": nc_idx,
                "ad_filename": f"detected_ad_{nc_idx:03d}.mp4",
                "ad_duration_seconds": round(duration, 3),
                "original_video_insert_at_seconds": round(orig_insert, 3),
                "original_video_insert_at_formatted": seconds_to_formatted(orig_insert),
                "final_video_ad_start_seconds": round(start, 3),
                "final_video_ad_start_formatted": seconds_to_formatted(start),
                "final_video_ad_end_seconds": round(end, 3),
                "final_video_ad_end_formatted": seconds_to_formatted(end),
            })
            nc_idx += 1
            acc_ad_dur += duration

    return ads
