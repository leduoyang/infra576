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

    # 1. INGEST
    print(f"[1/4] Extracting video metadata...")
    metadata = get_video_metadata(video_path)

    # 2. SEGMENTATION & FEATURE EXTRACTION (Steps 1 & 2)
    print(f"[2/4] Running slice-first segmentation and per-segment feature extraction...")
    segments, all_cuts = run_segmentation_pipeline(video_path, metadata, scene_threshold)

    # 3. CLASSIFICATION (Steps 3, 4 & 5: K-Means, duration gate, sequence validation)
    print(f"[3/4] Running K-Means classification...")
    classified_segments = classify_segments(segments, metadata["duration_seconds"], all_cuts=all_cuts)

    # 4. EXPORT
    print(f"[4/4] Building and exporting results to {output_path}...")
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
