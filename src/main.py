#!/usr/bin/env python3
"""
main.py – Simplified Video Segmentation Pipeline
"""

import json
import sys
import os
from pathlib import Path

# Ensure project root is in path BEFORE importing internal modules
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
) -> dict:
    """
    Executes the multimodal video segmentation pipeline from ingest to export.
    Returns the final result dictionary.
    """
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{stem}_segments.json")

    # 1. INGEST (Common Infra)
    metadata = get_video_metadata(video_path)

    # 2. SEGMENTATION & FEATURE EXTRACTION (The Orchestrator)
    # This now combines shot detection and feature mapping
    segments, global_profile = run_segmentation_pipeline(video_path, metadata, scene_threshold)

    # 3. CLASSIFICATION (The Decision Maker)
    classified_segments = classify_segments(segments, metadata["duration_seconds"], global_profile)


    # 4. EXPORT (Common Infra)
    result = build_output(video_path, metadata, classified_segments)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result

@click.command()
@click.option("--input", "-i", "video_path", required=True, type=click.Path(exists=True), help="Path to input video file")
@click.option("--output", "-o", "output_path", default=None, help="Path to output JSON file")
@click.option("--scene-threshold", default=15.0, help="PySceneDetect threshold")
def main(
    video_path: str,
    output_path: str,
    scene_threshold: float,
):
    """
    Multimodal Video Segmentation – baseline skeleton.
    """
    print(f"=== Video Segmentation ===")
    print(f"Input: {video_path}")

    # Orchestrate
    result = run_pipeline(video_path, output_path, scene_threshold)

    print(f"✓ Saved results with {len(result['timeline_segments'])} segments.")

if __name__ == "__main__":
    main()
