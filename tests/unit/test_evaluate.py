"""Unit tests for src/evaluate.py – accuracy evaluator."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.evaluate import temporal_iou, boundary_error, evaluate_single, match_segments


def _seg(start, end, seg_type="ad", is_content=False):
    return {
        "final_video_start_seconds": start,
        "final_video_end_seconds": end,
        "duration_seconds": end - start,
        "type": seg_type,
        "segment_label": seg_type,
        "is_content": is_content,
    }


class TestTemporalIoU:
    def test_perfect_overlap(self):
        a = _seg(10, 20)
        assert temporal_iou(a, a) == 1.0

    def test_no_overlap(self):
        a = _seg(0, 10)
        b = _seg(20, 30)
        assert temporal_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = _seg(0, 10)
        b = _seg(5, 15)
        # intersection = 5, union = 15
        assert abs(temporal_iou(a, b) - 5.0 / 15.0) < 0.01

    def test_contained(self):
        a = _seg(0, 20)
        b = _seg(5, 15)
        # intersection = 10, union = 20
        assert abs(temporal_iou(a, b) - 0.5) < 0.01


class TestBoundaryError:
    def test_exact_match(self):
        a = _seg(10, 20)
        err = boundary_error(a, a)
        assert err["start_error_sec"] == 0.0
        assert err["end_error_sec"] == 0.0

    def test_offset(self):
        a = _seg(10, 20)
        b = _seg(12, 22)
        err = boundary_error(a, b)
        assert err["start_error_sec"] == 2.0
        assert err["end_error_sec"] == 2.0


class TestMatchSegments:
    def test_perfect_match(self):
        gt = [_seg(10, 20), _seg(50, 70)]
        pred = [_seg(10, 20), _seg(50, 70)]
        matches, missed, false_alarms = match_segments(gt, pred)
        assert len(matches) == 2
        assert len(missed) == 0
        assert len(false_alarms) == 0

    def test_missed_detection(self):
        gt = [_seg(10, 20), _seg(50, 70)]
        pred = [_seg(10, 20)]
        matches, missed, false_alarms = match_segments(gt, pred)
        assert len(matches) == 1
        assert len(missed) == 1
        assert len(false_alarms) == 0

    def test_false_alarm(self):
        gt = [_seg(10, 20)]
        pred = [_seg(10, 20), _seg(50, 70)]
        matches, missed, false_alarms = match_segments(gt, pred)
        assert len(matches) == 1
        assert len(missed) == 0
        assert len(false_alarms) == 1


class TestEvaluateSingle:
    def test_perfect_detection(self):
        gt = {
            "video_filename": "test.mp4",
            "output_duration_seconds": 300,
            "timeline_segments": [
                _seg(0, 100, "video_content", True),
                _seg(100, 130, "ad", False),
                _seg(130, 300, "video_content", True),
            ],
        }
        pred = {
            "timeline_segments": [
                _seg(0, 100, "video_content", True),
                _seg(100, 130, "ad", False),
                _seg(130, 300, "video_content", True),
            ],
        }
        result = evaluate_single(gt, pred)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["avg_iou"] == 1.0

    def test_no_detections(self):
        gt = {
            "video_filename": "test.mp4",
            "output_duration_seconds": 300,
            "timeline_segments": [
                _seg(0, 100, "video_content", True),
                _seg(100, 130, "ad", False),
                _seg(130, 300, "video_content", True),
            ],
        }
        pred = {
            "timeline_segments": [
                _seg(0, 300, "video_content", True),
            ],
        }
        result = evaluate_single(gt, pred)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["false_negatives"] == 1
