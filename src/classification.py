"""
classification.py – The core algorithmic endpoint for video segmentation.

Teammates should implement their algorithm here.
Inputs: Video segments (found by PySceneDetect) and optional feature metrics.
Outputs: The segmented list marked with "is_content", "segment_type".
"""

def classify_segments(segments: list[dict], duration: float, global_profile: dict = None) -> list[dict]:
    """
    Classifies raw segments using Universal Profile comparison.
    """
    classified = []
    gp = global_profile or {}
    
    # Baseline averages
    avg_centroid = gp.get("avg_centroid", 1000.0)
    avg_bandwidth = gp.get("avg_bandwidth", 1000.0)
    avg_motion = gp.get("avg_motion", 5.0)
    
    for segment in segments:
        s = segment.copy()
        
        # Segment local features
        seg_dur = segment["duration_seconds"]
        seg_centroid = segment.get("spectral_centroid", avg_centroid)
        seg_bandwidth = segment.get("spectral_bandwidth", avg_bandwidth)
        seg_motion = segment.get("motion_score", avg_motion)
        
        # COMPARISON LOGIC:
        # 1. Frequency Distribution Check
        # Significant deviation in centroid often indicates a shift from speech (main) to music/noise (ad)
        freq_shift = abs(seg_centroid - avg_centroid) / (avg_centroid + 1e-6)
        
        # 2. Richness Check
        # Bandwidth represents the "fullness" of sound. 
        richness_diff = abs(seg_bandwidth - avg_bandwidth) / (avg_bandwidth + 1e-6)
        
        # 3. Motion Deviation
        # Ads are often much more dynamic than the main content average
        motion_ratio = seg_motion / (avg_motion + 1e-6)
        
        # Decision (Example: Content is the "norm")
        # If it deviates > 30% in frequency or has 2x higher motion, it might be an ad.
        is_outlier = (freq_shift > 0.3) or (motion_ratio > 2.5) or (seg_dur < 5.0)
        
        # Specific "Music" marker: High bandwidth + higher than average centroid
        is_music = (seg_bandwidth > avg_bandwidth * 1.2) and (seg_centroid > avg_centroid)
        
        is_ad = is_outlier or (is_music and seg_dur < 15.0)
        
        s["is_content"] = not is_ad
        s["segment_type"] = "ad" if is_ad else "video_content"
        s["confidence"] = 1.0 - min(0.5, freq_shift) # Simpler confidence proxy
        
        classified.append(s)
        
    return merge_consecutive_segments(classified)

def merge_consecutive_segments(segments: list[dict]) -> list[dict]:
    """
    Combines adjacent segments of the same type ('ad' or 'video_content').
    """
    if not segments:
        return []
    
    merged = []
    current = segments[0].copy()
    
    for i in range(1, len(segments)):
        nxt = segments[i]
        # If types match, merge them
        if nxt["segment_type"] == current["segment_type"]:
            current["end_seconds"] = nxt["end_seconds"]
            current["duration_seconds"] = current["end_seconds"] - current["start_seconds"]
            # Conservatively take the minimum confidence
            current["confidence"] = min(current["confidence"], nxt.get("confidence", 1.0))
        else:
            merged.append(current)
            current = nxt.copy()
            
    merged.append(current)
    return merged


