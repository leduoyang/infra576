#!/usr/bin/env python3
"""
main.py – Multimodal Video Segmentation Pipeline

Supports single-video and batch processing modes.
Generates JSON metadata with the full non-content taxonomy.

Usage:
    # Single video
    python src/main.py -i video.mp4

    # Batch (all videos in a directory)
    python src/main.py --input-dir ./videos/ --output-dir ./output/

    # Without transcript (faster, audio/visual only)
    python src/main.py -i video.mp4 --no-transcript

    # With a specific Whisper model
    python src/main.py -i video.mp4 --whisper-model medium
"""

import json
import sys
import os
from pathlib import Path
from glob import glob

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import click

from src.ingest import get_video_metadata
from src.segmentation import run_segmentation_pipeline
from src.classification import classify_segments
from src.export import build_output


# ---------------------------------------------------------------------------
# Single video pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    video_path: str,
    output_path: str = None,
    scene_threshold: float = 15.0,
    use_transcript: bool = True,
    whisper_model: str = "base",
) -> dict:
    """
    Executes the full multimodal video segmentation pipeline.
    Returns the result dictionary and saves it to output_path.
    """
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{stem}_segments.json")

    print(f"  [1/4] Ingesting metadata...")
    metadata = get_video_metadata(video_path)

    print(f"  [2/4] Segmenting & extracting features...")
    if use_transcript:
        print(f"         (transcript enabled, model={whisper_model})")
    segments, global_profile = run_segmentation_pipeline(
        video_path, metadata, scene_threshold,
        use_transcript=use_transcript,
        whisper_model=whisper_model,
    )
    print(f"         Found {len(segments)} raw segments")

    print(f"  [3/4] Classifying segments...")
    classified_segments = classify_segments(
        segments, metadata["duration_seconds"], global_profile
    )
    print(f"         Classified into {len(classified_segments)} segments")

    # Report classification summary
    from collections import Counter
    label_counts = Counter(s["segment_type"] for s in classified_segments)
    for label, count in sorted(label_counts.items()):
        total_dur = sum(
            s["duration_seconds"] for s in classified_segments
            if s["segment_type"] == label
        )
        print(f"           {label}: {count} segment(s), {total_dur:.1f}s")

    print(f"  [4/4] Exporting results...")
    result = build_output(video_path, metadata, classified_segments)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {output_path}")
    return result


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}


def run_batch(
    input_dir: str,
    output_dir: str,
    scene_threshold: float = 15.0,
    use_transcript: bool = True,
    whisper_model: str = "base",
) -> list[dict]:
    """Process all video files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        f for f in input_path.iterdir()
        if f.suffix.lower() in VIDEO_EXTENSIONS and f.is_file()
    )

    if not video_files:
        print(f"No video files found in {input_dir}")
        return []

    print(f"Found {len(video_files)} video(s) in {input_dir}\n")

    results = []
    for i, vf in enumerate(video_files, 1):
        print(f"{'='*60}")
        print(f"[{i}/{len(video_files)}] Processing: {vf.name}")
        print(f"{'='*60}")

        out_json = str(output_path / f"{vf.stem}_segments.json")
        try:
            result = run_pipeline(
                str(vf), out_json, scene_threshold,
                use_transcript, whisper_model,
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        print()

    # Cross-video repeat report
    if len(results) > 1:
        _generate_cross_video_report(results, output_path)

    print(f"\nBatch complete: {len(results)}/{len(video_files)} videos processed.")
    return results


def _generate_cross_video_report(results: list[dict], output_dir: Path):
    """Generate a summary report across all processed videos."""
    report = {
        "num_videos": len(results),
        "videos": [],
        "aggregate": {},
    }

    total_dur = 0.0
    total_nc_dur = 0.0
    total_segs = 0
    label_totals: dict[str, float] = {}

    for r in results:
        vid_dur = r.get("output_duration_seconds", 0)
        nc_dur = r.get("total_noncontent_duration_seconds", 0)
        n_segs = r.get("num_segments", 0)
        total_dur += vid_dur
        total_nc_dur += nc_dur
        total_segs += n_segs

        for seg in r.get("timeline_segments", []):
            label = seg.get("segment_label", "unknown")
            label_totals[label] = label_totals.get(label, 0) + seg.get("duration_seconds", 0)

        report["videos"].append({
            "filename": r.get("video_filename"),
            "duration": vid_dur,
            "noncontent_duration": nc_dur,
            "num_segments": n_segs,
        })

    report["aggregate"] = {
        "total_duration_seconds": round(total_dur, 3),
        "total_noncontent_seconds": round(total_nc_dur, 3),
        "noncontent_percentage": round(total_nc_dur / total_dur * 100, 1) if total_dur > 0 else 0,
        "total_segments": total_segs,
        "duration_by_label": {k: round(v, 3) for k, v in sorted(label_totals.items())},
    }

    report_path = output_dir / "batch_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Batch report saved: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--input", "-i", "video_path", default=None,
              type=click.Path(exists=True), help="Path to input video file")
@click.option("--output", "-o", "output_path", default=None,
              help="Path to output JSON file")
@click.option("--input-dir", default=None, type=click.Path(exists=True),
              help="Directory of videos for batch processing")
@click.option("--output-dir", default=None,
              help="Output directory for batch results")
@click.option("--scene-threshold", default=15.0,
              help="PySceneDetect content threshold")
@click.option("--no-transcript", is_flag=True, default=False,
              help="Disable Whisper transcription (faster)")
@click.option("--whisper-model", default="base",
              type=click.Choice(["tiny", "base", "small", "medium", "large"]),
              help="Whisper model size")
def main(
    video_path, output_path, input_dir, output_dir,
    scene_threshold, no_transcript, whisper_model,
):
    """
    Multimodal Video Segmentation – content vs non-content classifier.
    """
    use_transcript = not no_transcript

    if input_dir:
        # Batch mode
        if not output_dir:
            output_dir = os.path.join(input_dir, "output")
        run_batch(input_dir, output_dir, scene_threshold, use_transcript, whisper_model)

    elif video_path:
        # Single video mode
        print(f"=== Video Segmentation ===")
        print(f"Input: {video_path}")
        result = run_pipeline(
            video_path, output_path, scene_threshold,
            use_transcript, whisper_model,
        )
        n_segs = len(result.get("timeline_segments", []))
        print(f"\nDone: {n_segs} segments found.")

    else:
        print("Provide --input (-i) for single video or --input-dir for batch mode.")
        sys.exit(1)


if __name__ == "__main__":
    main()
