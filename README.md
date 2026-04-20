# Multimodal Ad Detection Pipeline

CSCI 576 final project — detect inserted ads in long videos using audio, visual, and speech features. Output F1@0.5 = **0.971** on the 5 test videos (4/5 perfect, 1 FP on test_005).

## Pipeline

```
MP4 → Ingest → Shot Segmentation → Feature Extraction → Classification → Export JSON
       ffmpeg   PySceneDetect       librosa/OpenCV/       hybrid rules     player schema
                                    faster-whisper
```

1. **Ingest** (`src/ingest.py`) — decode metadata, extract PCM audio.
2. **Segmentation** (`src/segmentation.py`) — PySceneDetect splits video into shots, then orchestrates per-shot feature extraction.
3. **Features** (`src/features/`):
   - `visual.py` — per-shot `motion_score`, `color_variance`
   - `audio.py` — `audio_energy` (RMS), `spectral_centroid`, `spectral_bandwidth`; also computes a per-video global audio profile + onset times.
   - `speech.py` — Whisper-based `word_rate`, `no_speech_prob`, `avg_logprob`, `sponsor_hit`, `word_count`.
4. **Classification** (`src/classification.py`) — hybrid rule pipeline, details below.
5. **Export** (`src/export.py`) — write JSON for the player (`player/index.html`).

## Features used

Each shot gets:

| Feature | Source | What it captures |
|---|---|---|
| `audio_energy` | RMS over PCM samples | loudness (ads mixed louder than content) |
| `spectral_centroid` | librosa | brightness of the mix (music/jingle signature) |
| `spectral_bandwidth` | librosa | spread of frequency energy |
| `motion_score` | frame-diff on downsampled grayscale | cuts + action |
| `color_variance` | per-frame RGB variance | busy/colorful visuals |
| `no_speech_prob` | Whisper | high = music/noise/silence |
| `word_rate` | Whisper tokens / duration | narration density |
| `avg_logprob` | Whisper | low (very negative) = confused transcription (often overlaid music) |
| `sponsor_hit` | regex on transcript | explicit sponsor phrases |

Plus per-video **global profile**:
- `avg_centroid`, `avg_bandwidth`, `avg_motion` — the "baseline" for this video.
- `onset_times` / `onset_strengths` — audio onsets used for boundary snapping.

## Classification logic

The classifier asks: *is this shot acoustically/visually/linguistically an outlier relative to the rest of the video?*

### Step 1 — per-shot ad score (3 channels)

- **Absolute score** (`_absolute_score`): deviation of `spectral_centroid`, `spectral_bandwidth`, `motion_score` from the video's global profile, plus a "music-like" bonus if bandwidth and centroid both spike.
- **Adaptive z-score**: robust z (median + MAD) of each feature over all shots, combined with weights. Weights emphasize audio (centroid 1.2, bandwidth 0.8) and motion (0.9).
- **Speech score** (`_speech_score`): high `no_speech_prob`, low `word_rate`, very negative `avg_logprob`, or `sponsor_hit` add to ad-likeness.

A shot is flagged as ad if any of:
- max single-feature z ≥ 3.0 (strong single signal), or
- combined z ≥ 1.5 AND absolute ≥ 1.5 (soft combo), or
- absolute ≥ 2.25 (very strong acoustic deviation alone).

### Step 2 — post-processing (the stack that gets from 0.54 → 0.97)

In order:

1. **Merge** consecutive same-type shots.
2. **Bridge fragments** — merge two ad blocks separated by a ≤20s content gap (one commercial split by a quiet cut).
3. **Drop short ads** — anything < 20s is dropped (stray outliers).
4. **Drop boundary ads** — ad-looking segments within 50s of video start/end are intro/outro, not inserted ads.
5. **Merge** again.
6. **Drop content-like ads** — three flip-to-content rules:
   - (a) single long static shot (1 shot, >35s, motion<10, quiet) — real ads cut every 5–15s.
   - (b) short pure-silent quiet region (≥90% shots ns≥0.95 ∧ wr<0.1, <40s, not loud) — content transitions.
   - (c) narration-absent low-motion region (0 narration shots, 0 high-motion shots, <45s, quiet) — silent content cutaways. Motion gate (≥2.5× video motion median) preserves action ads.
7. **Speech rescue** — scan raw shots for sustained runs matching any of:
   - low-speech run (ns≥0.35, wr≤2.0, lp≤-0.8, ≥30s) with a strict dual silence filter (rejected only if *pure* AND *quiet* — loud jingles kept).
   - sandwich pattern (silent shot between two speech-heavy shots) — catches short silent ads.
   - visual rescue (color_variance + motion spike) — catches voiceover ads with visual flair.
   - **loud-narration rescue** (RMS ≥ 1.8× median + ns≤0.1 + wr≥0.2 + motion≥20) — catches broadcast-style ads.
8. **Merge** + **drop content-like** again.
9. **Trim dialogue boundaries** — trim leading/trailing shots with ns≤0.15 ∧ wr≥2.5 ∧ motion<15 (content dialogue that the classifier glued onto the ad), when cumulative trim ≥10s.
10. **Merge**.
11. **Onset snap** — snap ad start/end to strongest audio onset within ±6s; adjacent content segment is contracted/expanded to keep the timeline contiguous.

### Why this works

- Single rules overfit to one video. Each rule in step 6–9 kills a class of error (static cutaway / silent transition / motion-less content / dialogue bleed) without affecting the others.
- Rescue + reject run twice: the first pass catches the obvious cases, rescue adds missing ads, the second reject cleans up rescue over-extensions.
- Onset snap fixes ±1–5s boundary drift cheaply — ads almost always start/end at a sharp audio onset.

## Setup

```bash
brew install ffmpeg
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

Single video:
```bash
.venv/bin/python src/main.py \
  -i /path/to/test_001.mp4 \
  -o predictions/test_001.json \
  --features-cache features_cache/test_001.json   # optional, skips feature re-extraction
```

Evaluate against ground truth:
```bash
.venv/bin/python src/evaluate.py \
  --pred-dir predictions \
  --gt-dir /path/to/video_info
```

Iterate on classifier only (uses feature cache):
```bash
.venv/bin/python src/tune.py
```

## Tests

```bash
python3 -m pytest              # all
python3 -m pytest tests/unit/  # unit
python3 -m pytest tests/e2e/   # end-to-end on a synthetic video
```

The E2E test builds a 30s synthetic MP4 (Red/Blue/Black), runs the full pipeline via subprocess, and validates the output JSON schema.

## Player

Open `player/index.html` in a browser, load a video + its JSON output to inspect shots, features, and ad boundaries.

## Project structure

```
src/
  main.py              CLI entrypoint
  ingest.py            ffmpeg decode + audio extraction
  segmentation.py      PySceneDetect orchestrator, loops features over shots
  classification.py    hybrid ad classifier + post-processing pipeline
  features/
    visual.py          motion + color variance
    audio.py           RMS/centroid/bandwidth, global profile, onsets
    speech.py          Whisper transcript → word_rate / no_speech_prob / ...
  evaluate.py          IoU / interval F1 / frame F1 vs GT
  tune.py              fast re-eval using cached features
  export.py            player JSON schema writer
predictions/           per-video output JSON
features_cache/        pre-computed feature JSONs (bypass slow Whisper)
player/                static web player
tests/                 unit + e2e
report.md              iteration history (v1 → v14, F1@0.5 0.22 → 0.971)
```
