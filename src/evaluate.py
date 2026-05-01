#!/usr/bin/env python3
"""
evaluate.py – Accuracy evaluator for the segmentation pipeline.

Compares pipeline output against ground-truth JSON files.
Computes per-segment and overall metrics:
  - Precision, Recall, F1 for non-content detection
  - Temporal IoU (Intersection over Union) for detected boundaries
  - Boundary accuracy (how close detected boundaries are to ground truth)

Usage:
    python src/evaluate.py --ground-truth gt.json --predicted pred.json
    python src/evaluate.py --ground-truth-dir tests/ground_truth/ --predicted-dir output/
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def temporal_iou(seg_a: dict, seg_b: dict) -> float:
    """Compute Intersection-over-Union between two time intervals."""
    a_start = seg_a["final_video_start_seconds"]
    a_end = seg_a["final_video_end_seconds"]
    b_start = seg_b["final_video_start_seconds"]
    b_end = seg_b["final_video_end_seconds"]

    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0


def boundary_error(gt_seg: dict, pred_seg: dict) -> dict:
    """Compute boundary errors in seconds between matched segments."""
    start_err = abs(
        gt_seg["final_video_start_seconds"] - pred_seg["final_video_start_seconds"]
    )
    end_err = abs(
        gt_seg["final_video_end_seconds"] - pred_seg["final_video_end_seconds"]
    )
    return {"start_error_sec": round(start_err, 3), "end_error_sec": round(end_err, 3)}


def is_noncontent(seg: dict) -> bool:
    """Check if a segment is non-content (works with both old and new schema)."""
    if "is_content" in seg:
        return not seg["is_content"]
    return seg.get("type") not in ("video_content", "content")


# ---------------------------------------------------------------------------
# Matching & evaluation
# ---------------------------------------------------------------------------

def match_segments(gt_ncs: list[dict], pred_ncs: list[dict], iou_threshold: float = 0.3):
    """
    Greedily match predicted non-content segments to ground-truth ones
    based on temporal IoU. Returns (matches, unmatched_gt, unmatched_pred).
    """
    used_pred = set()
    matches = []

    for gt in gt_ncs:
        best_iou = 0.0
        best_idx = -1
        for j, pred in enumerate(pred_ncs):
            if j in used_pred:
                continue
            iou = temporal_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_idx >= 0 and best_iou >= iou_threshold:
            matches.append((gt, pred_ncs[best_idx], best_iou))
            used_pred.add(best_idx)

    unmatched_gt = [
        gt for i, gt in enumerate(gt_ncs) if not any(m[0] is gt for m in matches)
    ]
    unmatched_pred = [
        pred for j, pred in enumerate(pred_ncs) if j not in used_pred
    ]
    return matches, unmatched_gt, unmatched_pred


def evaluate_single(gt_data: dict, pred_data: dict, iou_threshold: float = 0.3) -> dict:
    """Evaluate a single video's predictions against ground truth."""
    gt_segs = gt_data.get("timeline_segments", [])
    pred_segs = pred_data.get("timeline_segments", [])

    gt_nc = [s for s in gt_segs if is_noncontent(s)]
    pred_nc = [s for s in pred_segs if is_noncontent(s)]
    gt_content = [s for s in gt_segs if not is_noncontent(s)]
    pred_content = [s for s in pred_segs if not is_noncontent(s)]

    matches, unmatched_gt, unmatched_pred = match_segments(
        gt_nc, pred_nc, iou_threshold
    )

    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_iou = sum(m[2] for m in matches) / len(matches) if matches else 0.0

    boundary_errors = [boundary_error(m[0], m[1]) for m in matches]
    avg_start_err = (
        sum(e["start_error_sec"] for e in boundary_errors) / len(boundary_errors)
        if boundary_errors else 0.0
    )
    avg_end_err = (
        sum(e["end_error_sec"] for e in boundary_errors) / len(boundary_errors)
        if boundary_errors else 0.0
    )

    # Time-weighted accuracy: what fraction of total time is correctly classified?
    total_dur = gt_data.get("output_duration_seconds", 0)
    correct_time = 0.0
    for m in matches:
        gt_s, pred_s, _ = m
        overlap_start = max(
            gt_s["final_video_start_seconds"], pred_s["final_video_start_seconds"]
        )
        overlap_end = min(
            gt_s["final_video_end_seconds"], pred_s["final_video_end_seconds"]
        )
        correct_time += max(0, overlap_end - overlap_start)

    # Content correctly identified (approximate: total - nc errors)
    gt_nc_dur = sum(s["duration_seconds"] for s in gt_nc)
    gt_content_dur = total_dur - gt_nc_dur
    correct_time += gt_content_dur * 0.9  # rough estimate
    time_accuracy = min(1.0, correct_time / total_dur) if total_dur > 0 else 0.0

    return {
        "video": gt_data.get("video_filename", "unknown"),
        "gt_noncontent_count": len(gt_nc),
        "pred_noncontent_count": len(pred_nc),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_iou": round(avg_iou, 4),
        "avg_start_error_sec": round(avg_start_err, 3),
        "avg_end_error_sec": round(avg_end_err, 3),
        "time_accuracy_approx": round(time_accuracy, 4),
        "matches": [
            {
                "gt_start": m[0]["final_video_start_seconds"],
                "gt_end": m[0]["final_video_end_seconds"],
                "pred_start": m[1]["final_video_start_seconds"],
                "pred_end": m[1]["final_video_end_seconds"],
                "iou": round(m[2], 4),
            }
            for m in matches
        ],
        "missed_gt": [
            {
                "start": s["final_video_start_seconds"],
                "end": s["final_video_end_seconds"],
                "duration": s["duration_seconds"],
            }
            for s in unmatched_gt
        ],
        "false_alarms": [
            {
                "start": s["final_video_start_seconds"],
                "end": s["final_video_end_seconds"],
                "duration": s["duration_seconds"],
                "label": s.get("segment_label", s.get("type", "unknown")),
            }
            for s in unmatched_pred
        ],
    }


def evaluate_batch(gt_dir: str, pred_dir: str, iou_threshold: float = 0.3) -> dict:
    """Evaluate all matching JSON files in two directories."""
    gt_files = {f: os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith(".json")}
    pred_files = {f: os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.endswith(".json")}

    common = sorted(set(gt_files.keys()) & set(pred_files.keys()))
    if not common:
        print("No matching JSON files found between directories.")
        return {"results": [], "aggregate": {}}

    results = []
    for fname in common:
        with open(gt_files[fname]) as f:
            gt = json.load(f)
        with open(pred_files[fname]) as f:
            pred = json.load(f)
        results.append(evaluate_single(gt, pred, iou_threshold))

    # Aggregate
    n = len(results)
    agg = {
        "num_videos": n,
        "avg_precision": round(sum(r["precision"] for r in results) / n, 4),
        "avg_recall": round(sum(r["recall"] for r in results) / n, 4),
        "avg_f1": round(sum(r["f1"] for r in results) / n, 4),
        "avg_iou": round(sum(r["avg_iou"] for r in results) / n, 4),
        "avg_start_error_sec": round(
            sum(r["avg_start_error_sec"] for r in results) / n, 3
        ),
        "avg_end_error_sec": round(
            sum(r["avg_end_error_sec"] for r in results) / n, 3
        ),
    }
    return {"results": results, "aggregate": agg}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--ground-truth", "-g", default=None, help="Path to ground-truth JSON")
@click.option("--predicted", "-p", default=None, help="Path to predicted JSON")
@click.option("--ground-truth-dir", "-gd", default=None, help="Dir of ground-truth JSONs")
@click.option("--predicted-dir", "-pd", default=None, help="Dir of predicted JSONs")
@click.option("--iou-threshold", default=0.3, help="IoU threshold for matching")
@click.option("--output", "-o", default=None, help="Save results to JSON file")
def main(ground_truth, predicted, ground_truth_dir, predicted_dir, iou_threshold, output):
    """Evaluate segmentation accuracy against ground truth."""

    if ground_truth and predicted:
        with open(ground_truth) as f:
            gt = json.load(f)
        with open(predicted) as f:
            pred = json.load(f)
        result = evaluate_single(gt, pred, iou_threshold)
        _print_single(result)
        if output:
            with open(output, "w") as f:
                json.dump(result, f, indent=2)

    elif ground_truth_dir and predicted_dir:
        batch = evaluate_batch(ground_truth_dir, predicted_dir, iou_threshold)
        _print_batch(batch)
        if output:
            with open(output, "w") as f:
                json.dump(batch, f, indent=2)
    else:
        print("Provide --ground-truth + --predicted  OR  --ground-truth-dir + --predicted-dir")
        sys.exit(1)


def _print_single(r: dict):
    print(f"\n{'='*50}")
    print(f"  Video: {r['video']}")
    print(f"  GT non-content: {r['gt_noncontent_count']}  |  Predicted: {r['pred_noncontent_count']}")
    print(f"  TP: {r['true_positives']}  FP: {r['false_positives']}  FN: {r['false_negatives']}")
    print(f"  Precision: {r['precision']:.2%}  Recall: {r['recall']:.2%}  F1: {r['f1']:.2%}")
    print(f"  Avg IoU: {r['avg_iou']:.2%}")
    print(f"  Avg boundary error: start={r['avg_start_error_sec']:.1f}s  end={r['avg_end_error_sec']:.1f}s")
    if r["missed_gt"]:
        print(f"  Missed GT segments:")
        for m in r["missed_gt"]:
            print(f"    {m['start']:.1f}s → {m['end']:.1f}s ({m['duration']:.1f}s)")
    if r["false_alarms"]:
        print(f"  False alarms:")
        for fa in r["false_alarms"]:
            print(f"    {fa['start']:.1f}s → {fa['end']:.1f}s ({fa['duration']:.1f}s) [{fa['label']}]")
    print(f"{'='*50}\n")


def _print_batch(batch: dict):
    for r in batch["results"]:
        _print_single(r)
    agg = batch["aggregate"]
    print(f"\n{'='*50}")
    print(f"  AGGREGATE ({agg['num_videos']} videos)")
    print(f"  Avg Precision: {agg['avg_precision']:.2%}")
    print(f"  Avg Recall:    {agg['avg_recall']:.2%}")
    print(f"  Avg F1:        {agg['avg_f1']:.2%}")
    print(f"  Avg IoU:       {agg['avg_iou']:.2%}")
    print(f"  Avg boundary:  start={agg['avg_start_error_sec']:.1f}s  end={agg['avg_end_error_sec']:.1f}s")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
