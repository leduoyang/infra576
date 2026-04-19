import numpy as np
from src.classification import classify_segments, _is_commercial_duration

def test_is_commercial_duration():
    assert _is_commercial_duration(30.0) is True
    assert _is_commercial_duration(15.0) is True
    assert _is_commercial_duration(60.0) is True
    assert _is_commercial_duration(45.0) is True
    assert _is_commercial_duration(400.0) is False
    assert _is_commercial_duration(150.0) is False

def test_classify_segments_empty():
    assert classify_segments([], 100.0) == []

def test_classify_segments_merging_and_smoothing():
    windows = []
    
    # 0 to 110s -> 22 windows (stride is 5)
    for i in range(22):
        s = i * 5.0
        windows.append({"start_seconds": s, "end_seconds": s + 15.0, "duration_seconds": 15.0, "pacing_score": 0.0})

    # Ad block: 110s to 170s. Step mapping creates exactly a 60-second block out of 12 contiguous windows
    for i in range(22, 34):
        s = i * 5.0
        windows.append({"start_seconds": s, "end_seconds": s + 15.0, "duration_seconds": 15.0, "pacing_score": 99.0})
        
    # Content block: 170s onwards -> 20 windows
    for i in range(34, 54):
        s = i * 5.0
        windows.append({"start_seconds": s, "end_seconds": s + 15.0, "duration_seconds": 15.0, "pacing_score": 0.0})
        
    final = classify_segments(windows, 300.0)
    
    assert final[0]["is_content"] is True
    assert final[1]["is_content"] is False
    assert final[1]["duration_seconds"] == 60.0
    assert final[1]["start_seconds"] == 115.0
    assert final[1]["end_seconds"] == 175.0
