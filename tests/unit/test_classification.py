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
    
    # 0 to 100s -> 20 windows (stride is 5)
    for i in range(20):
        s = i * 5.0
        windows.append({"start_seconds": s, "end_seconds": s + 15.0, "duration_seconds": 15.0, "pacing_score": 0.0})

    # Ad block: 100s to 160s. Step mapping creates exactly a 60-second block out of 12 contiguous windows
    # i=20 (starts 100, mapped 100-105) through i=31 (starts 155, mapped 155-160)
    for i in range(20, 32):
        s = i * 5.0
        windows.append({"start_seconds": s, "end_seconds": s + 15.0, "duration_seconds": 15.0, "pacing_score": 99.0})
        
    # Content block: 160s onwards -> 20 windows
    for i in range(32, 52):
        s = i * 5.0
        windows.append({"start_seconds": s, "end_seconds": s + 15.0, "duration_seconds": 15.0, "pacing_score": 0.0})
        
    final = classify_segments(windows, 300.0)
    
    assert final[0]["is_content"] is True
    assert final[1]["is_content"] is False
    assert final[1]["duration_seconds"] == 60.0
    assert final[1]["start_seconds"] == 105.0
    assert final[1]["end_seconds"] == 165.0
