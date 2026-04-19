"""
classification.py – 5-Step K-Means Segmentation Engine

Step 1: Fast Slicing (bookend markers) — handled in segmentation.py
Step 2: Per-Segment Feature Extraction — features attached in segmentation.py
Step 3: Statistical Grouping — K-Means (K=2) on feature matrix
Step 4: Deterministic Labeling — D mod 15 ≈ 0 duration gate
Step 5: Sequence Validation — override isolated false positives
"""
import numpy as np


# ---------------------------------------------------------------------------
# Step 3: K-Means clustering (pure numpy, Lloyd's algorithm)
# ---------------------------------------------------------------------------

def _build_feature_matrix(segments: list[dict]) -> np.ndarray:
    """
    Build an N×8 feature matrix from per-segment data.

    Features (normalised internally via z-score before clustering):
      0 – pacing_score          : shot boundary frequency (cuts/min proxy)
      1 – audio_rms_variance    : variance of per-frame RMS  → dynamic range
                                  (ads are compressed → LOW variance)
      2 – audio_energy          : mean RMS loudness
                                  (ads are uniformly LOUD)
      3 – zcr_mean              : Zero-Crossing Rate → noisiness
                                  (excited speech / music in ads = HIGH)
      4 – spatial_edge_density  : fraction of Canny edge pixels across segment
                                  (on-screen text / motion graphics in ads = HIGH)
      5 – motion_energy_variance: variance of per-frame SAD scores → freneticism
                                  (fast-cutting / shaky ads = HIGH)
      6 – median_sat_lum        : (median_S + median_V) / 2 → HSV Color Volume
                                  (ads are graded to 'pop' → HIGH)
      7 – duration_seconds      : absolute segment length
                                  (commercials are short → LOW)
    """
    rows = []
    for seg in segments:
        sbf    = float(seg.get("pacing_score",         0.0))
        armsv  = float(seg.get("audio_rms_variance",   0.0))
        ae     = float(seg.get("audio_energy",         0.0))
        zcr    = float(seg.get("zcr_mean",             0.0))
        sed    = float(seg.get("spatial_edge_density", 0.0))
        mev    = float(seg.get("motion_energy_variance", 0.0))
        sl     = (float(seg.get("avg_saturation", 0.0)) +
                  float(seg.get("avg_luminance",  0.0))) / 2.0
        dur    = float(seg.get("duration_seconds", 0.0))
        rows.append([sbf, armsv, ae, zcr, sed, mev, sl, dur])
    return np.array(rows, dtype=np.float64)


def _normalise(X: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalisation column-wise. Safe against zero-std cols."""
    mu  = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    return (X - mu) / std


def _kmeans(X: np.ndarray, k: int = 2, max_iter: int = 100, n_init: int = 10,
            seed: int = 42) -> np.ndarray:
    """
    Lloyd's K-Means with multiple random restarts.
    Returns label array of shape (N,).
    """
    rng = np.random.default_rng(seed)
    n   = len(X)
    if n <= k:
        return np.arange(n, dtype=int)

    best_labels = None
    best_inertia = np.inf

    for _ in range(n_init):
        # K-Means++ style init: pick first centroid at random, then farthest
        idx0 = rng.integers(0, n)
        centroids = [X[idx0]]
        for _c in range(1, k):
            dists = np.array([min(np.sum((x - c) ** 2) for c in centroids) for x in X])
            probs = dists / (dists.sum() + 1e-12)
            centroids.append(X[rng.choice(n, p=probs)])
        centroids = np.array(centroids)

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            # Assignment step
            dists = np.stack([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=1)
            new_labels = np.argmin(dists, axis=1)
            if np.all(new_labels == labels):
                break
            labels = new_labels
            # Update step
            for j in range(k):
                members = X[labels == j]
                if len(members):
                    centroids[j] = members.mean(axis=0)

        inertia = sum(
            np.sum((X[labels == j] - centroids[j]) ** 2)
            for j in range(k) if np.any(labels == j)
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels  = labels.copy()

    return best_labels


# ---------------------------------------------------------------------------
# Step 4: Deterministic labeling — D mod 15 ≈ 0
# ---------------------------------------------------------------------------
_COMMERCIAL_LENGTHS = [15.0, 30.0, 45.0, 60.0]  # standard broadcast ad durations
_DURATION_TOLERANCE = 2.0                  # ±2s for frame-rate rounding
_MAX_COMMERCIAL_DURATION = 120.0           # no single ad block > 2 min


def _is_commercial_duration(dur: float) -> bool:
    """Returns True if `dur` is within tolerance of a standard ad length.
    Segments longer than MAX_COMMERCIAL_DURATION are never commercials."""
    if dur > _MAX_COMMERCIAL_DURATION:
        return False
    return any(abs(dur % cl) <= _DURATION_TOLERANCE or
               abs(dur % cl - cl) <= _DURATION_TOLERANCE
               for cl in _COMMERCIAL_LENGTHS)


# ---------------------------------------------------------------------------
# Step 5: Sequence Validation
# ---------------------------------------------------------------------------
_ISOLATION_CONTENT_THRESHOLD = 600.0   # 10 min of content on each side


def _sequence_validate(segments: list[dict]) -> list[dict]:
    """
    Override isolated ad blocks that are completely surrounded by long content
    blocks. This handles single stray false-positive segments deep inside
    a long program.
    Modifies segments in-place.
    """
    n = len(segments)

    for i in range(n):
        if segments[i]["is_content"]:
            continue

        # Measure content duration to the left (before this ad)
        left_content = 0.0
        j = i - 1
        while j >= 0 and segments[j]["is_content"]:
            left_content += segments[j]["duration_seconds"]
            j -= 1

        # Measure content duration to the right (after this ad)
        right_content = 0.0
        j = i + 1
        while j < n and segments[j]["is_content"]:
            right_content += segments[j]["duration_seconds"]
            j += 1

        if (left_content  >= _ISOLATION_CONTENT_THRESHOLD and
                right_content >= _ISOLATION_CONTENT_THRESHOLD):
            segments[i]["is_content"] = True   # override: false positive

    return segments


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify_segments(windows: list[dict], duration: float,
                      global_profile: dict = None) -> list[dict]:
    """
    5-step classification pipeline for overlapping windows.
    """
    if not windows:
        return []

    # ---- 1. Build & normalise feature matrix -------------------------
    X_raw = _build_feature_matrix(windows)
    X     = _normalise(X_raw)

    # ---- 2. K-Means (K=2) on individual windows -----------------------
    raw_labels = _kmeans(X, k=2)

    # ---- 3. Label Smoothing (Median Filter) ----------------------------
    # Smooths isolated false positive windows (e.g., [1, 0, 1] -> [1, 1, 1])
    import scipy.signal
    smoothed_labels = scipy.signal.medfilt(raw_labels, kernel_size=3).astype(int)

    # ---- 4. Collapse adjacent identical windows into macro blocks ------
    # Shift output bounds by 5.0s to align the window center with the timeline
    step_size = 5.0
    macro_blocks = []
    current_label = smoothed_labels[0]
    
    current_block = {
        "start": 0.0,  # stretch the first window to 0.0 to fill the start gap
        "end": min(windows[0]["start_seconds"] + step_size + 5.0, duration),
        "label": current_label,
    }

    for i in range(1, len(windows)):
        label = smoothed_labels[i]
        step_end = min(windows[i]["start_seconds"] + step_size + 5.0, duration)

        if label == current_label:
            current_block["end"] = step_end
        else:
            current_block["duration"] = current_block["end"] - current_block["start"]
            macro_blocks.append(current_block)
            current_label = label
            current_block = {
                "start": windows[i]["start_seconds"] + 5.0,
                "end": step_end,
                "label": current_label,
            }
    current_block["duration"] = current_block["end"] - current_block["start"]
    macro_blocks.append(current_block)

    # ---- 5. Duration Gate: identify Ad cluster from Macro Blocks -------
    ad_cluster = -1
    k_num = 2
    scores = []
    for j in range(k_num):
        blocks_j = [b for b in macro_blocks if b["label"] == j]
        if not blocks_j:
            scores.append(0.0)
            continue
        hits = sum(_is_commercial_duration(b["duration"]) for b in blocks_j)
        scores.append(hits / len(blocks_j))

    best = int(np.argmax(scores))
    if scores[best] >= 0.30:  # Relaxed from 0.50 to catch more valid clusters
        ad_cluster = best

    # ---- 6. Assemble output -------
    classified = []
    for b in macro_blocks:
        is_ad = (b["label"] == ad_cluster and ad_cluster != -1)
        classified.append({
            "start_seconds":   b["start"],
            "end_seconds":     b["end"],
            "duration_seconds": b["duration"],
            "is_content":      not is_ad,
            "confidence":      0.75 if is_ad else 0.85,
        })
        
    classified = _sequence_validate(classified)

    # Assign final intro/outro types
    for i, seg in enumerate(classified):
        if seg["is_content"]:
            seg["segment_type"] = "video_content"
        else:
            if i == 0: seg["segment_type"] = "intro"
            elif i == len(classified) - 1: seg["segment_type"] = "outro"
            else: seg["segment_type"] = "ad"
            
    return classified
