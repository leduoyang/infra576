#!/usr/bin/env python3
"""
evaluate.py - Compare predicted ad intervals vs ground-truth.

Metrics:
  - Per-ad best-match IoU
  - Interval-level precision / recall / F1 at IoU >= threshold (default 0.5)
  - Frame-level (per-second) precision / recall / F1 on ad mask
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def load_ads(json_path: Path) -> List[Tuple[float, float]]:
    data = json.loads(json_path.read_text())
    segs = data.get("timeline_segments") or data.get("segments") or []
    ads = []
    for s in segs:
        t = s.get("type") or s.get("label")
        if t == "ad" or t == "non_content" or t == "non-content":
            start = s.get("final_video_start_seconds", s.get("start_seconds", s.get("start")))
            end = s.get("final_video_end_seconds", s.get("end_seconds", s.get("end")))
            if start is None or end is None:
                continue
            ads.append((float(start), float(end)))
    # sort
    ads.sort()
    return ads


def iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def interval_prf(pred: List[Tuple[float, float]], gt: List[Tuple[float, float]], iou_thr: float = 0.5):
    matched_gt = set()
    matched_pred = set()
    pairs = []
    for i, p in enumerate(pred):
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gt):
            if j in matched_gt:
                continue
            v = iou(p, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thr:
            matched_gt.add(best_j)
            matched_pred.add(i)
            pairs.append((i, best_j, best_iou))
    tp = len(matched_pred)
    fp = len(pred) - tp
    fn = len(gt) - len(matched_gt)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1, pairs=pairs)


def frame_prf(pred: List[Tuple[float, float]], gt: List[Tuple[float, float]], duration: float, step: float = 0.1):
    n = int(duration / step) + 1

    def mask(intervals):
        m = bytearray(n)
        for s, e in intervals:
            i0 = max(0, int(s / step))
            i1 = min(n, int(e / step))
            for k in range(i0, i1):
                m[k] = 1
        return m

    mp, mg = mask(pred), mask(gt)
    tp = sum(1 for a, b in zip(mp, mg) if a and b)
    fp = sum(1 for a, b in zip(mp, mg) if a and not b)
    fn = sum(1 for a, b in zip(mp, mg) if b and not a)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    iou_all = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return dict(precision=prec, recall=rec, f1=f1, iou=iou_all)


def eval_pair(pred_path: Path, gt_path: Path) -> dict:
    pred = load_ads(pred_path)
    gt = load_ads(gt_path)
    gt_data = json.loads(gt_path.read_text())
    duration = float(gt_data.get("output_duration_seconds") or gt_data.get("original_video_duration_seconds") or 0)

    per_gt_iou = []
    for g in gt:
        best = max((iou(p, g) for p in pred), default=0.0)
        per_gt_iou.append(best)

    return {
        "num_pred": len(pred),
        "num_gt": len(gt),
        "pred_intervals": pred,
        "gt_intervals": gt,
        "per_gt_iou": per_gt_iou,
        "mean_gt_iou": sum(per_gt_iou) / len(per_gt_iou) if per_gt_iou else 0.0,
        "interval_f1_iou50": interval_prf(pred, gt, 0.5),
        "interval_f1_iou30": interval_prf(pred, gt, 0.3),
        "frame_metrics": frame_prf(pred, gt, duration),
        "duration": duration,
    }


def fmt_intervals(xs):
    return ", ".join(f"[{a:.1f}-{b:.1f}]" for a, b in xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="dir with predicted JSONs (name matches gt)")
    ap.add_argument("--gt-dir", required=True, help="ground-truth JSON dir")
    ap.add_argument("--pattern", default="test_*.json")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    results = []
    for gt in sorted(gt_dir.glob(args.pattern)):
        pred = pred_dir / gt.name
        if not pred.exists():
            print(f"[skip] {gt.name} no prediction at {pred}")
            continue
        r = eval_pair(pred, gt)
        r["name"] = gt.stem
        results.append(r)

    if not results:
        print("no results")
        return

    print("\n=== Per-video ===")
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

    def mean(key_path):
        vals = []
        for r in results:
            v = r
            for k in key_path:
                v = v[k]
            vals.append(v)
        return sum(vals) / len(vals)

    print("\n=== Mean ===")
    print(f"mean per-gt IoU : {mean(['mean_gt_iou']):.3f}")
    print(f"interval F1@0.5 : {mean(['interval_f1_iou50','f1']):.3f}")
    print(f"interval F1@0.3 : {mean(['interval_f1_iou30','f1']):.3f}")
    print(f"frame F1        : {mean(['frame_metrics','f1']):.3f}")
    print(f"frame IoU       : {mean(['frame_metrics','iou']):.3f}")

    print("\n=== Detail ===")
    for r in results:
        print(f"\n[{r['name']}] dur={r['duration']:.1f}s")
        print(f"  GT  ({r['num_gt']}): {fmt_intervals(r['gt_intervals'])}")
        print(f"  PRED({r['num_pred']}): {fmt_intervals(r['pred_intervals'])}")
        print(f"  per-GT IoU: " + ", ".join(f"{v:.2f}" for v in r["per_gt_iou"]))
        m50 = r["interval_f1_iou50"]
        print(f"  @IoU>=0.5: TP={m50['tp']} FP={m50['fp']} FN={m50['fn']}")


if __name__ == "__main__":
    main()
