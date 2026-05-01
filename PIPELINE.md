# Multimodal Video Ad-Detection Pipeline

## Description

Given a long-form video that may contain inserted advertisements (intros, sponsorships, mid-roll ads, brand placements), this pipeline produces a JSON timeline that marks each ad segment. The output drives a web player that lets a viewer skip non-content automatically (`Play Content Only` mode) or jump straight between ads (`Play Non-Content Only` mode).

The detection is **audio-only**. It runs three audio models on a single 16 kHz mono WAV — Silero VAD for speech presence, Whisper for word-level transcripts inside speech regions, and YAMNet for framewise audio-event labels over the entire audio — fuses them into a numbered transcript, and asks an LLM to flag the ad-like passages. Visual features are deferred; they would serve as a downstream filter for pure-music false positives but are not part of the current pipeline.

## High-level pipeline

```
            ┌─────────────────────────────────────────────────────────────────┐
input.mp4 → │ 1. ingest        ffprobe metadata                              │
            │ 2. audio extract ffmpeg → 16 kHz mono WAV                       │           │
            │ 4. Whisper ASR   word-level transcript                          │
            │                  (vad_filter=True; runs Silero again as a gate) │
            │ 5. YAMNet        framewise audio-event tags over the entire WAV │
            │ 6. block builder speech blocks bounded by Silero intervals,     │
            │                  text from Whisper words, events from YAMNet;   │
            │                  YAMNet-grouped blocks fill the non-speech gaps │
            │ 7. llm_input     numbered "sentence stream" + sentence index    │
            │ 8. llm_classify  OpenAI call → list of ad sentence numbers      │
            │ 9. llm_merge     map sentences → time → 5 s grid → player JSON  │
            └─────────────────────────────────────────────────────────────────┘
                                                     ↓
                                              <stem>_segments.json
                                                     ↓
                                              player/index.html
```

Stages 1–6 are deterministic feature extraction. Stage 8 is the only step that calls an external service; everything else is local. Every invocation re-runs every stage from scratch — no caching, no skip-if-exists. Silero runs twice — once standalone in step 3, once again inside Whisper in step 4 (see below).

## Method, stage by stage

### 1. Ingest
`ffprobe` reads container metadata: duration, fps, resolution, audio/video stream offsets. The audio offset (`audio_start_time`) is needed because some videos have audio that doesn't start at video t=0.

### 2. Audio extraction
A single `ffmpeg` pass to **16 kHz mono WAV**. All three audio models downstream (Silero, Whisper, YAMNet) want this exact format, so one extraction serves all three. Output: `work/audio/<stem>.wav`.

### 3. Silero VAD — speech presence (called twice)

Silero VAD is a small CNN+LSTM (≈1.8 MB ONNX) trained for binary "speech / not speech" classification at ~30 ms granularity. The same model is invoked in two places in the pipeline:

**a) Standalone — `src.features.transcript.run_silero()`**
Reads `work/audio/<stem>.wav`, calls `faster_whisper.vad.get_speech_timestamps` with:

```python
VadOptions(min_silence_duration_ms=500, max_speech_duration_s=30.0, speech_pad_ms=200)
```

Returns `[{start, end}, ...]` in seconds. **These intervals are what defines the speech-block boundaries** in step 6. The 30 s cap forces a split inside any continuous-music-with-sparse-speech region so we don't get one mega-interval covering an entire ad.

**b) Inside Whisper — `vad_filter=True` in step 4**
faster-whisper runs Silero internally before transcription as a *gate*: anything Silero rejects is not sent to the decoder, so Whisper can't hallucinate fluent English over silence/music/jingles. The boundaries Whisper reports back at the segment level can incorrectly span large gaps between two separated speech windows, so we **discard those segment timestamps** and consume only the per-word `(start, end)` data from this call.

**Cost of running Silero twice:** ~3.5 s on a 30-min file, negligible next to Whisper's 134 s. We could remove (a) by clustering Whisper words into pseudo-intervals after the fact, or remove (b) by pre-chunking the audio with our (a) intervals and disabling `vad_filter`. Neither was worth the code change.

### 4. Whisper ASR — word-level transcript

We use `faster-whisper` (CTranslate2 inference engine running OpenAI Whisper weights — no PyTorch required at runtime). Default model: `small.en` (244 M params). Inference runs at int8 on CPU; ~14× real-time on an M-series Mac.

The full call:

```python
model.transcribe(
    audio_path,
    language="en",
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 500, "max_speech_duration_s": 30.0},
    word_timestamps=True,
)
```

`vad_filter=True` runs Silero as a gate (see step 3b). `word_timestamps=True` adds a post-decoding DTW alignment pass over cross-attention to give per-word `(start, end)`.

We **iterate the segments returned by Whisper to pull out their word lists, then throw the segments themselves away**. The block builder uses only the word stream.

**Output**: `[{start, end, text}, ...]` — flat word list, time-ordered, written to `work/whisper/<stem>.json`.

### 5. YAMNet — audio-event tagging
YAMNet (Google, 2020) is a MobileNetV1 backbone trained on AudioSet (~2 M YouTube clips, 521 sound classes including Speech, Music, Vehicle, Engine, Applause, Cat, Glass, etc.). It processes **0.96 s windows hopped by 0.48 s**, producing a 521-class score vector per window.

We take top-K classes per frame (K=5), then median-filter the top-1 over an 11-frame (~5 s) window to suppress label flicker.

**Output**: per-frame `{start, end, top: [(label, score), ...], smoothed_top}` for the entire audio.

### 6. Block builder
Combines Whisper words and YAMNet frames into a unified timeline:

- **Speech blocks** — bounded by Silero intervals. Text comes from Whisper words whose start falls inside the interval. Top-3 YAMNet events are attached by overlap-weighted scoring (with a 15 % minimum overlap fraction so noise events are dropped).
- **Non-speech blocks** — fill the gaps between Silero intervals, grouped by consecutive YAMNet smoothed-top label. Long non-speech runs (>15 s) are split along YAMNet label transitions so a 90 s ad can be analyzed in pieces rather than as one opaque chunk.

Adjacent non-speech blocks sharing the same top-1 event are merged. Block boundaries use the YAMNet hop (0.48 s) rather than the full 0.96 s window so adjacent groups never overlap.

**Output**: `work/transcript/<stem>.json` (structured) and `work/transcript/<stem>.txt` (human/LLM-readable).

### 7. LLM input — numbered sentence stream
The block list is rendered as a flowing, numbered transcript that an LLM can read top-to-bottom:

```
S1: "Give it up!"
S2: [non-speech: Speech, Screaming, Shout, 0.6s]
S3: "Thank you, okay. Now"
...
S332: [non-speech: Music, Speech, Television, 1.2s]
S333: "to make friends. You"
S334: [non-speech: Speech, Music, Child speech, kid speaking, 3.9s]
S335: "want a Dorito? Hey,"
S337: "everyone! The new containing out Doritos! Doritos?"
...
```

Two artifacts are written:
- `work/llm_input/<stem>.prompt.txt` — full chain-of-thought prompt + the numbered transcript. ~25 KB for a 30-min video.
- `work/llm_input/<stem>.index.json` — back-pointer table mapping every sentence number to a time range, consumed by the merge step (the LLM never has to emit timestamps).

### 8. LLM classification
The prompt is sent to OpenAI Chat Completions. The default model is configurable; the prompt is carefully designed:

1. **Step 1 — Summarize**: read the full transcript, identify genre / topic / narrative arc.
2. **Step 2 — Trace narrative thread**: state the storyline / topic flow as a baseline.
3. **Step 3 — Classify each region**:
    - **Speech sentences** default to **content**; only flagged as ad on a clear commercial signal (brand mention, CTA, characteristic ad copy).
    - **Non-speech sentences** default to **ad**, classified as content only if they pass three flow tests: (a) does surrounding speech describe/imply the sounds; (b) do the events match the scene the characters are in; (c) does the block sit smoothly between two speech sentences continuing the same topic. If all three fail → ad.
    - In-universe ads (a character watching an ad on TV) are part of the narrative → content.
4. **Step 4 — Output**: a single JSON line `{"ad_sentences": [195, 196, 197, 199]}`.

The bias is **deliberately aggressive on non-speech** because audio-only signals are ambiguous (orchestral score vs ad jingle), and a downstream visual filter is the planned fix for false positives. Erring toward ad keeps recall high.

**Output**: `work/llm_reply/<stem>.txt` — the model's full chain-of-thought reasoning followed by the JSON line.

### 9. Merge into player schema
The merge step:
1. Parses `{"ad_sentences": [...]}` from the reply (tolerant of surrounding prose, code fences, multiple matches — takes the last one).
2. Looks up each ad sentence's `(start, end)` from the index.
3. Merges adjacent ad sentences whose gap is < 5 s into continuous **ad runs**.
4. Slices the timeline into uniform **5 s blocks**. Each block is "ad" if it overlaps any ad run by ≥ 50 %.
5. Merges consecutive same-label 5 s blocks into final timeline runs.
6. Builds the player JSON via `src/export.py` (existing schema with `timeline_segments`, `inserted_ads`, `original_video_*` / `final_video_*` time fields).

**Output**: `<stem>_segments.json` in the repo root (consumed by `player/index.html`).

## Outputs

```
infra576/
  <stem>_segments.json            ← FINAL — load in player/index.html
  work/
    audio/<stem>.wav              ← 16 kHz mono PCM
    silero/<stem>.json            ← speech intervals
    whisper/<stem>.json           ← word-level transcript
    yamnet/<stem>.json            ← framewise audio events
    transcript/<stem>.{json,txt}  ← block-builder output (debug / human view)
    llm_input/<stem>.prompt.txt   ← what gets sent to OpenAI
    llm_input/<stem>.index.json   ← sentence → time map
    llm_reply/<stem>.txt          ← model's reasoning + JSON answer
```

Final output stays at the top level so `player/index.html` finds it; everything else is intermediate and gitignored.

## Why this design

- **Audio-only, visual deferred.** The prior version of this project (in `src/segmentation.py` / `src/classification.py`) classified ads from frame motion + color variance + audio spectral centroid using thresholds. On `test_003.mp4` it reported 72 ads against a ground truth of 3. The thresholds were tripped by every shot cut and music swell in the movie. We switched to audio-only because the speech and event semantics are more separable; visual could come back as a downstream filter.
- **LLM in the loop.** Hand-tuned heuristics over deviation-from-baseline can't tell "30 s of orchestral music in the score" from "30 s of orchestral music in an ad" — they look identical to a feature extractor. An LLM reading the surrounding transcript can use that context to decide.
- **Sentence stream, not per-block JSON.** First attempt asked the LLM to return one label per block (362 entries). The model returned 336 labels — 26 short — and missed every ground-truth ad. Switching to a numbered transcript with a small `{"ad_sentences": [int, ...]}` output got Ad 3 (with a brand mention) on the next attempt and made the response trivially parseable.
- **No caching.** Each run re-extracts audio, re-runs Silero / Whisper / YAMNet, and rebuilds the prompt. Intermediate JSONs under `work/` are stage-to-stage handoffs (overwritten every run), not skip-if-exists caches. The end-to-end runtime is therefore ~2:30 plus the API call on every invocation.
- **Time mapping is local, not LLM-driven.** The LLM never emits timestamps. It returns sentence numbers; we resolve them via the index file we wrote alongside the prompt.

## Performance

On a 30-minute video, M-series Mac CPU at int8 — every run re-executes every stage:

| Stage | Time |
|---|---|
| Audio extract | 1.1 s |
| Silero VAD (standalone) | 3.5 s |
| Whisper (`small.en`, word ts; includes Silero gate) | 134 s |
| YAMNet (load + infer) | 12 s |
| Block builder | 0.3 s |
| LLM input | 1 s |
| OpenAI call | 5–25 s (model-dependent) |
| Merge | 0.5 s |
| **End-to-end** | **~2:30 + LLM call** |

Whisper inference dominates total runtime.

## How to run

```bash
# One-shot end-to-end
./run.sh videos_with_ads/test_003.mp4

# Override model (default is set in src/features/llm_classify.py)
./run.sh videos_with_ads/test_003.mp4 --model gpt-4o
```

API key is read from `src/.env` (either `OPENAI_API_KEY=...` or `API_KEY=...`).
