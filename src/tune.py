#!/usr/bin/env python3
"""Iterate classifier on cached features — fast tuning without re-extracting audio/video."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classification import classify_segments
from src.export import build_output
from src.evaluate import eval_pair


def run_one(cache_path: Path, gt_path: Path, pred_out: Path) -> dict:
    data = json.loads(cache_path.read_text())
    video_path = data["video_path"]
    segments = data["segments"]
    global_profile = data["global_profile"]

    gt = json.loads(gt_path.read_text())
    metadata = {
        "duration_seconds": gt["output_duration_seconds"],
        "resolution": gt["original_video_resolution"],
    }

    classified = classify_segments(segments, metadata["duration_seconds"], global_profile)
    result = build_output(video_path, metadata, classified)
    pred_out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return eval_pair(pred_out, gt_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="features_cache")
    ap.add_argument("--gt-dir", default="/Users/chenghaotien/Downloads/csci576/dataset/video_info")
    ap.add_argument("--pred-dir", default="predictions")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(exist_ok=True)

    results = []
    for c in sorted(cache.glob("test_*.json")):
        gt = gt_dir / c.name
        if not gt.exists():
            continue
        r = run_one(c, gt, pred_dir / c.name)
        r["name"] = c.stem
        results.append(r)

    header = f"{'video':<10} {'#gt':>4} {'#pred':>5} {'meanIoU':>8} {'F1@.5':>7} {'F1@.3':>7} {'frameF1':>8} {'frameIoU':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<10} {r['num_gt']:>4} {r['num_pred']:>5} "
            f"{r['mean_gt_iou']:>8.3f} {r['interval_f1_iou50']['f1']:>7.3f} "
            f"{r['interval_f1_iou30']['f1']:>7.3f} {r['frame_metrics']['f1']:>8.3f} "
            f"{r['frame_metrics']['iou']:>9.3f}"
        )

    def mean(path):
        vs = []
        for r in results:
            v = r
            for k in path:
                v = v[k]
            vs.append(v)
        return sum(vs) / len(vs) if vs else 0.0

    print(
        f"{'MEAN':<10} {'':>4} {'':>5} "
        f"{mean(['mean_gt_iou']):>8.3f} {mean(['interval_f1_iou50','f1']):>7.3f} "
        f"{mean(['interval_f1_iou30','f1']):>7.3f} {mean(['frame_metrics','f1']):>8.3f} "
        f"{mean(['frame_metrics','iou']):>9.3f}"
    )


if __name__ == "__main__":
    main()
