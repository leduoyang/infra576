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

_COL_DELTA_AE  = 0
_COL_ZCR       = 1
_COL_DELTA_MEV = 2
_COL_EDGES     = 3
_COL_PACING    = 4
_COL_DELTA_RMS = 5

def _build_feature_matrix(windows: list[dict]) -> np.ndarray:
    import scipy.ndimage
    
    ae_raw  = np.array([float(w.get("audio_energy", 0.0)) for w in windows])
    zcr_raw = np.array([float(w.get("zcr_mean", 0.0)) for w in windows])
    mev_raw = np.array([float(w.get("motion_energy_variance", 0.0)) for w in windows])
    edg_raw = np.array([float(w.get("spatial_edge_density", 0.0)) for w in windows])
    pac_raw = np.array([float(w.get("pacing_score", 0.0)) for w in windows])
    rms_raw = np.array([float(w.get("audio_rms_variance", 0.0)) for w in windows])

    # 5-minute rolling median (assuming 5s steps => 60 windows)
    k_size = 60
    ae_med  = scipy.ndimage.median_filter(ae_raw,  size=k_size)
    mev_med = scipy.ndimage.median_filter(mev_raw, size=k_size)
    rms_med = scipy.ndimage.median_filter(rms_raw, size=k_size)

    # Compute deltas
    delta_ae  = ae_raw - ae_med
    delta_mev = mev_raw - mev_med
    delta_rms = rms_raw - rms_med

    X = np.stack([
        delta_ae,
        zcr_raw,
        delta_mev,
        edg_raw,
        pac_raw,
        delta_rms
    ], axis=1)

    return X


def _normalise(X: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalisation column-wise. Safe against zero-std cols."""
    mu  = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    return (X - mu) / std


def _kmeans(X: np.ndarray, k: int = 6, max_iter: int = 100, n_init: int = 10,
            seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Lloyd's K-Means with multiple random restarts.
    Returns label array of shape (N,) and centroids of shape (K, D).
    """
    rng = np.random.default_rng(seed)
    n   = len(X)
    if n <= k:
        return np.arange(n, dtype=int), X

    best_labels = None
    best_centroids = None
    best_inertia = np.inf

    for _ in range(n_init):
        # K-Means++ style init: pick first centroid at random, then farthest
        idx0 = rng.integers(0, n)
        centroids = [X[idx0]]
        for _c in range(1, k):
            dists = np.array([min(np.sum((x - c) ** 2) for c in centroids) for x in X])
            s = dists.sum()
            if s == 0:
                probs = np.ones(n) / n
            else:
                probs = dists / s
            probs = probs / probs.sum()  # ensure perfect sum to 1.0
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
            best_centroids = np.array(centroids)

    return best_labels, best_centroids


def _cluster_contrast_score(centroid: np.ndarray) -> float:
    """
    Compute a contrast-based ad-likelihood score from a Z-score centroid.
    contrast_score = abs(ΔAE) + abs(ΔMEV) + Edges – ΔRMS_Var
    """
    return (abs(centroid[_COL_DELTA_AE])
            + abs(centroid[_COL_DELTA_MEV])
            + centroid[_COL_EDGES]
            - centroid[_COL_DELTA_RMS])

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
        left_content = 0.0
        j = i - 1
        while j >= 0 and segments[j]["is_content"]:
            left_content += segments[j]["duration_seconds"]
            j -= 1
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
                      global_profile: dict = None,
                      all_cuts: list[float] = None) -> list[dict]:
    """
    5-step classification pipeline for overlapping windows.
    """
    if not windows:
        return []

    # ---- 1. Build & normalise feature matrix -------------------------
    X_raw = _build_feature_matrix(windows)
    X     = _normalise(X_raw)

    # ---- 2. K-Means (K=6) on individual windows -----------------------
    K_CLUSTERS = 6
    raw_labels, centroids = _kmeans(X, k=K_CLUSTERS)

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
    scores = []
    
    for j in range(K_CLUSTERS):
        blocks_j = [b for b in macro_blocks if b["label"] == j]
        if not blocks_j:
            scores.append(0.0)
            continue
        hits = sum(_is_commercial_duration(b["duration"]) for b in blocks_j)
        scores.append(hits / len(blocks_j))

    if scores:
        best = int(np.argmax(scores))
        if scores[best] >= 0.30:  # Modulo-15 base gate
            ad_cluster = best

    # ---- 6. Assemble output -------
    classified = []
    
    def snap(target: float, points: list[float], threshold=5.0):
        if not points: return target
        closest = min(points, key=lambda x: abs(x - target))
        return closest if abs(closest - target) <= threshold else target

    for b in macro_blocks:
        is_ad = (b["label"] == ad_cluster and ad_cluster != -1)
        
        start_t = b["start"]
        end_t   = b["end"]
        
        if all_cuts is not None:
            # Prevent start boundary snapping back behind 0 or exceeding original bounds improperly
            start_t = snap(start_t, all_cuts, threshold=5.0)
            end_t = snap(end_t, all_cuts, threshold=5.0)

        # Ensure we don't snap the bounds past each other
        if end_t <= start_t:
            end_t = start_t + 0.1
            
        classified.append({
            "start_seconds":   start_t,
            "end_seconds":     end_t,
            "duration_seconds": end_t - start_t,
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
