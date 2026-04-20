#!/usr/bin/env python3
"""
main.py – Simplified Video Segmentation Pipeline
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import click

from src.ingest import get_video_metadata
from src.segmentation import run_segmentation_pipeline
from src.classification import classify_segments
from src.export import build_output


def run_pipeline(
    video_path: str,
    output_path: str = None,
    scene_threshold: float = 15.0,
    features_cache: str = None,
) -> dict:
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{stem}_segments.json")

    metadata = get_video_metadata(video_path)

    segments = None
    global_profile = None
    if features_cache and Path(features_cache).exists():
        data = json.loads(Path(features_cache).read_text())
        if data.get("video_path") == video_path:
            segments = data["segments"]
            global_profile = data["global_profile"]
            print(f"  [cache] loaded features from {features_cache}")

    if segments is None:
        transcript_cache = None
        if features_cache:
            transcript_cache = str(Path(features_cache).parent / f"{Path(video_path).stem}_transcript.json")
        segments, global_profile = run_segmentation_pipeline(
            video_path, metadata, scene_threshold, transcript_cache
        )
        if features_cache:
            Path(features_cache).parent.mkdir(parents=True, exist_ok=True)
            Path(features_cache).write_text(
                json.dumps(
                    {"video_path": video_path, "segments": segments, "global_profile": global_profile},
                    indent=2,
                )
            )

    classified_segments = classify_segments(segments, metadata["duration_seconds"], global_profile)

    result = build_output(video_path, metadata, classified_segments)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


@click.command()
@click.option("--input", "-i", "video_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", "output_path", default=None)
@click.option("--scene-threshold", default=15.0)
@click.option("--features-cache", default=None, help="Path to JSON cache of pre-computed features")
def main(video_path, output_path, scene_threshold, features_cache):
    print(f"=== Video Segmentation ===")
    print(f"Input: {video_path}")
    result = run_pipeline(video_path, output_path, scene_threshold, features_cache)
    print(f"✓ Saved results with {len(result['timeline_segments'])} segments.")


if __name__ == "__main__":
    main()
