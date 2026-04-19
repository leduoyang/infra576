import numpy as np
from src.classification import (
    classify_segments, 
    _is_commercial_duration,
    _apply_heuristic_post_processing
)

def test_is_commercial_duration():
    assert _is_commercial_duration(30.0) is True
    assert _is_commercial_duration(15.0) is True
    assert _is_commercial_duration(60.0) is True
    assert _is_commercial_duration(45.0) is True
    assert _is_commercial_duration(11.0) is False
    assert _is_commercial_duration(110.0) is False

def test_apply_heuristic_post_processing():
    segments = [
        {"start_seconds": 0.0, "end_seconds": 50.0, "duration_seconds": 50.0, "is_content": True},
        # Short ad -> Should become content
        {"start_seconds": 50.0, "end_seconds": 65.0, "duration_seconds": 15.0, "is_content": False},
        {"start_seconds": 65.0, "end_seconds": 80.0, "duration_seconds": 15.0, "is_content": True},
        
        # Valid Ad 1
        {"start_seconds": 80.0, "end_seconds": 120.0, "duration_seconds": 40.0, "is_content": False},
        # Sandwich Content (length = 20, combined ads = 40+40 = 80 -> Swallow it)
        {"start_seconds": 120.0, "end_seconds": 140.0, "duration_seconds": 20.0, "is_content": True},
        # Valid Ad 2
        {"start_seconds": 140.0, "end_seconds": 180.0, "duration_seconds": 40.0, "is_content": False},
        
        # Long content (ends the sandwich properly)
        {"start_seconds": 180.0, "end_seconds": 300.0, "duration_seconds": 120.0, "is_content": True},
    ]
    
    # Process
    final = _apply_heuristic_post_processing(segments)
    
    # Check outputs:
    # Segments should structurally collapse down to just:
    # 1. Content (0.0 -> 80.0): 50s solid content + (15s short ad overridden) + 15s content = 80s
    # 2. Ad (80.0 -> 180.0): 40s ad + (20s content swallowed) + 40s ad = 100s
    # 3. Content (180.0 -> 300.0): 120s
    
    assert len(final) == 3
    
    assert final[0]["is_content"] is True
    assert final[0]["duration_seconds"] == 80.0
    
    assert final[1]["is_content"] is False
    assert final[1]["duration_seconds"] == 100.0
    
    assert final[2]["is_content"] is True
    assert final[2]["duration_seconds"] == 120.0
