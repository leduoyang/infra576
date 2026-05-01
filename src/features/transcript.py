"""
src/features/transcript.py

Whisper-anchored transcript with YAMNet audio-event labels per block.

Pipeline:
  1. Audio extracted to 16 kHz mono WAV (reuses src.ingest.extract_audio).
  2. faster-whisper (small.en) with built-in Silero VAD transcribes speech.
  3. YAMNet labels every 0.48 s frame of the whole audio.
  4. Block builder anchors to Whisper segments and uses YAMNet to fill non-speech gaps.
  5. Each block gets top-K YAMNet events ranked by overlap-weighted score,
     gated by a minimum overlap fraction.

Outputs:
  - <stem>_transcript.json   structured, for the player and code
  - <stem>_transcript.txt    flat, human/LLM-readable

CLI:
  python3 -m src.features.transcript -i videos_with_ads/test_003.mp4
"""

import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import click
import numpy as np

# Allow running as a module from anywhere in the repo
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ingest import extract_audio, seconds_to_formatted
from src.features import paths as P

YAMNET_HUB_URL = "https://tfhub.dev/google/yamnet/1"
DEFAULT_WHISPER_MODEL = "small.en"

# YAMNet runs 0.96 s windows hopped by 0.48 s.  We use the hop for
# *boundary* arithmetic (so adjacent groups don't overlap) but keep the
# full window for overlap-weighted event scoring.
YAMNET_HOP_S = 0.48


# ── Whisper ─────────────────────────────────────────────────────────────────

def run_silero(audio_path: str, timings: Optional[dict] = None) -> list[dict]:
    """
    Run Silero VAD directly to get authoritative speech intervals.
    Returns: [{"start": float, "end": float}, ...] in seconds.

    Why we run it ourselves: faster-whisper's `vad_filter=True` consumes Silero
    internally but reports Whisper segment timestamps in original-audio
    coordinates — a single "segment" can span a 5-min gap between two short
    speech bursts. Using Silero's intervals directly avoids that.
    """
    import soundfile as sf
    from faster_whisper.vad import get_speech_timestamps, VadOptions

    t0 = time.perf_counter()
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    opts = VadOptions(
        min_silence_duration_ms=500,
        max_speech_duration_s=30.0,
        speech_pad_ms=200,
    )
    ts = get_speech_timestamps(audio, vad_options=opts)
    intervals = [{"start": float(t["start"]) / sr, "end": float(t["end"]) / sr} for t in ts]

    if timings is not None:
        timings["silero_s"] = time.perf_counter() - t0
    return intervals


def run_whisper(
    audio_path: str,
    model_size: str = DEFAULT_WHISPER_MODEL,
    timings: Optional[dict] = None,
) -> list[dict]:
    """
    Transcribe with faster-whisper. Returns a flat list of *words* with timestamps:
      [{"start": float, "end": float, "text": str}, ...]
    `vad_filter=True` still runs Silero internally so Whisper doesn't hallucinate
    over silence/music; we ignore the segment boundaries and use only the words.
    """
    from faster_whisper import WhisperModel

    t0 = time.perf_counter()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    if timings is not None:
        timings["whisper_load_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    segments, _info = model.transcribe(
        audio_path,
        language="en",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500, "max_speech_duration_s": 30.0},
        word_timestamps=True,
    )

    words: list[dict] = []
    for seg in segments:
        for w in (seg.words or []):
            text = w.word.strip()
            if not text:
                continue
            words.append({"start": float(w.start), "end": float(w.end), "text": text})
    if timings is not None:
        timings["whisper_infer_s"] = time.perf_counter() - t0
    return words


# ── YAMNet ──────────────────────────────────────────────────────────────────

def _load_yamnet():
    """Load YAMNet from TF-Hub and read its class map CSV."""
    import tensorflow as tf
    import tensorflow_hub as hub

    model = hub.load(YAMNET_HUB_URL)
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    class_names = []
    with tf.io.gfile.GFile(class_map_path) as f:
        for row in csv.DictReader(f):
            class_names.append(row["display_name"])
    return model, class_names


def run_yamnet(
    audio_path: str,
    top_k: int = 5,
    timings: Optional[dict] = None,
) -> list[dict]:
    """
    Run YAMNet on a 16 kHz mono WAV. Returns one entry per 0.48 s frame:
      [{"start": float, "end": float,
        "top": [(label, score), ...],
        "smoothed_top": label}, ...]
    If `timings` is provided, populates `yamnet_load_s` and `yamnet_infer_s`.
    """
    import soundfile as sf

    t0 = time.perf_counter()
    model, class_names = _load_yamnet()
    if timings is not None:
        timings["yamnet_load_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    waveform, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)

    scores, _embeddings, _log_mel = model(waveform)
    scores = scores.numpy()  # (num_frames, 521)

    frame_starts = np.arange(scores.shape[0]) * 0.48
    frame_ends = frame_starts + 0.96

    top_idx_per_frame = np.argmax(scores, axis=1)
    # Wider filter (11 frames ≈ 5.3 s) cuts label flicker, which keeps
    # consecutive groups from churning into many tiny blocks.
    smoothed_top_idx = _mode_filter(top_idx_per_frame, k=11)

    frames = []
    for i in range(scores.shape[0]):
        top_k_idx = np.argsort(scores[i])[::-1][:top_k]
        frames.append({
            "start": float(frame_starts[i]),
            "end": float(frame_ends[i]),
            "top": [(class_names[j], float(scores[i][j])) for j in top_k_idx],
            "smoothed_top": class_names[int(smoothed_top_idx[i])],
        })
    if timings is not None:
        timings["yamnet_infer_s"] = time.perf_counter() - t0
    return frames


def _mode_filter(arr: np.ndarray, k: int = 5) -> np.ndarray:
    """1-D mode (most-frequent value) filter over a window of size k."""
    n = len(arr)
    half = k // 2
    out = arr.copy()
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        vals, counts = np.unique(arr[a:b], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


# ── Block builder ───────────────────────────────────────────────────────────

def build_blocks(
    silero_intervals: list[dict],
    whisper_words: list[dict],
    yamnet_frames: list[dict],
    duration: float,
    top_k: int = 3,
    min_event_overlap_frac: float = 0.15,
    min_gap_duration: float = 1.0,
    snap_tolerance: float = 0.1,
    word_attach_tolerance: float = 0.5,
) -> list[dict]:
    """
    Build a single timeline:
      - Speech blocks bounded by Silero VAD intervals.
      - Each block's text is the concatenation of Whisper words whose start time
        falls within [interval.start - tol, interval.end + tol].
      - Non-speech gaps (between Silero intervals) become YAMNet-grouped blocks.
    Every block carries top-K YAMNet events.
    """
    blocks: list[dict] = []

    for iv in silero_intervals:
        words_in = [
            w for w in whisper_words
            if w["start"] >= iv["start"] - word_attach_tolerance
            and w["start"] <  iv["end"]   + word_attach_tolerance
        ]
        text = " ".join(w["text"] for w in words_in).strip()
        blocks.append({
            "start": iv["start"],
            "end": iv["end"],
            "kind": "speech",
            "text": text,
            "events": _events_in_range(
                yamnet_frames, iv["start"], iv["end"], top_k, min_event_overlap_frac
            ),
        })

    boundaries = (
        [(0.0, 0.0)]
        + [(iv["start"], iv["end"]) for iv in silero_intervals]
        + [(duration, duration)]
    )
    boundaries.sort()
    for i in range(len(boundaries) - 1):
        gap_start = boundaries[i][1]
        gap_end = boundaries[i + 1][0]
        if gap_end - gap_start < min_gap_duration:
            continue
        for g_start, g_end, _g_top in _group_yamnet_in_range(yamnet_frames, gap_start, gap_end):
            blocks.append({
                "start": g_start,
                "end": g_end,
                "kind": "non-speech",
                "text": "",
                "events": _events_in_range(
                    yamnet_frames, g_start, g_end, top_k, min_event_overlap_frac
                ),
            })

    blocks.sort(key=lambda b: b["start"])
    for i in range(len(blocks) - 1):
        if abs(blocks[i]["end"] - blocks[i + 1]["start"]) < snap_tolerance:
            mid = (blocks[i]["end"] + blocks[i + 1]["start"]) / 2
            blocks[i]["end"] = mid
            blocks[i + 1]["start"] = mid

    blocks = _merge_adjacent_nonspeech(
        blocks, yamnet_frames, top_k, min_event_overlap_frac
    )
    return blocks


def _merge_adjacent_nonspeech(blocks, yamnet_frames, top_k, min_event_overlap_frac):
    """Collapse consecutive non-speech blocks that share the same top-1 event.

    Re-computes events on the merged span so the weights reflect the union.
    """
    if not blocks:
        return blocks

    def _top1(b):
        return b["events"][0]["label"] if b["events"] else None

    out = [blocks[0]]
    for b in blocks[1:]:
        prev = out[-1]
        if (
            prev["kind"] == "non-speech"
            and b["kind"] == "non-speech"
            and _top1(prev) is not None
            and _top1(prev) == _top1(b)
            and abs(b["start"] - prev["end"]) < 0.6  # contiguous-ish
        ):
            prev["end"] = b["end"]
            prev["events"] = _events_in_range(
                yamnet_frames, prev["start"], prev["end"], top_k, min_event_overlap_frac
            )
        else:
            out.append(b)
    return out


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _events_in_range(frames, start, end, top_k, min_frac):
    """Top-K labels in [start, end] by overlap-weighted score, gated by min_frac coverage."""
    if end <= start:
        return []
    span = end - start
    weighted: dict[str, float] = {}
    overlap_sec: dict[str, float] = {}
    for fr in frames:
        ov = _overlap(start, end, fr["start"], fr["end"])
        if ov <= 0:
            continue
        for label, score in fr["top"]:
            weighted[label] = weighted.get(label, 0.0) + score * ov
            overlap_sec[label] = overlap_sec.get(label, 0.0) + ov

    candidates = [
        (label, w) for label, w in weighted.items()
        if (overlap_sec.get(label, 0.0) / span) >= min_frac
    ]
    candidates.sort(key=lambda x: -x[1])
    return [
        {"label": label, "weight": round(min(1.0, w / span), 3)}
        for label, w in candidates[:top_k]
    ]


def _group_yamnet_in_range(frames, start, end):
    """Group consecutive frames sharing smoothed_top within [start, end].

    Boundaries use the YAMNet hop (0.48 s), not the 0.96 s window, so adjacent
    groups don't overlap.  Frame i "owns" the interval [i*hop, (i+1)*hop].
    """
    indexed = []
    for i, fr in enumerate(frames):
        f_lo = i * YAMNET_HOP_S
        f_hi = (i + 1) * YAMNET_HOP_S
        if f_hi > start and f_lo < end:
            indexed.append((i, fr["smoothed_top"]))
    if not indexed:
        return []

    def _bounds(a_idx: int, b_idx: int) -> tuple[float, float]:
        return (
            max(start, a_idx * YAMNET_HOP_S),
            min(end, (b_idx + 1) * YAMNET_HOP_S),
        )

    groups = []
    cur_a, cur_label = indexed[0]
    cur_b = cur_a
    for idx, label in indexed[1:]:
        if label == cur_label:
            cur_b = idx
        else:
            g_lo, g_hi = _bounds(cur_a, cur_b)
            groups.append((g_lo, g_hi, cur_label))
            cur_a, cur_b, cur_label = idx, idx, label
    g_lo, g_hi = _bounds(cur_a, cur_b)
    groups.append((g_lo, g_hi, cur_label))
    return groups


# ── Render ──────────────────────────────────────────────────────────────────

def render_text(blocks: list[dict]) -> str:
    """Flat human/LLM-readable transcript."""
    lines = []
    for b in blocks:
        ts = f"[{seconds_to_formatted(b['start'])} - {seconds_to_formatted(b['end'])}]"
        events = ", ".join(e["label"] for e in b["events"]) or "—"
        if b["kind"] == "speech":
            lines.append(f'{ts} ({events}) "{b["text"]}"')
        else:
            lines.append(f"{ts} ({events})")
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--input", "-i", "video_path", required=True, type=click.Path(exists=True))
@click.option("--json", "-j", "json_out", default=None, help="Output JSON path")
@click.option("--txt", "-t", "txt_out", default=None, help="Output flat-text path")
@click.option("--whisper-model", default=DEFAULT_WHISPER_MODEL)
def main(
    video_path: str,
    json_out: Optional[str],
    txt_out: Optional[str],
    whisper_model: str,
):
    stem = Path(video_path).stem
    json_out = Path(json_out) if json_out else P.transcript_json(stem)
    txt_out = Path(txt_out) if txt_out else P.transcript_txt(stem)
    out_silero = P.silero(stem)
    out_whisper = P.whisper_words(stem)
    out_yamnet = P.yamnet_frames(stem)

    print("=== Transcript builder ===")
    print(f"Input: {video_path}")

    timings: dict = {}
    t_total = time.perf_counter()

    audio_wav = str(P.audio(stem))
    print(f"[1/5] Extracting audio → {audio_wav}")
    t0 = time.perf_counter()
    extract_audio(video_path, audio_wav, sample_rate=16000)
    timings["audio_extract_s"] = time.perf_counter() - t0
    print(f"      done in {timings['audio_extract_s']:.2f}s")

    print("[2/5] Silero VAD ...")
    silero_intervals = run_silero(audio_wav, timings=timings)
    out_silero.write_text(json.dumps(silero_intervals))
    print(f"      → {len(silero_intervals)} speech intervals")

    print(f"[3/5] Whisper ({whisper_model}) ...")
    whisper_words = run_whisper(audio_wav, model_size=whisper_model, timings=timings)
    out_whisper.write_text(json.dumps(whisper_words))
    print(f"      → {len(whisper_words)} words")

    print("[4/5] YAMNet ...")
    yamnet_frames = run_yamnet(audio_wav, timings=timings)
    out_yamnet.write_text(json.dumps(yamnet_frames))
    print(f"      → {len(yamnet_frames)} frames")

    duration = max(
        [y["end"] for y in yamnet_frames]
        + [iv["end"] for iv in silero_intervals]
        + [w["end"] for w in whisper_words]
        + [0.0]
    )

    print("[5/5] Building blocks ...")
    t0 = time.perf_counter()
    blocks = build_blocks(silero_intervals, whisper_words, yamnet_frames, duration)
    timings["block_build_s"] = time.perf_counter() - t0
    print(f"      → {len(blocks)} blocks  ({timings['block_build_s']:.2f}s)")

    timings["total_s"] = time.perf_counter() - t_total

    print()
    print("=== Timing ===")
    for k in (
        "audio_extract_s",
        "silero_s",
        "whisper_load_s",
        "whisper_infer_s",
        "yamnet_load_s",
        "yamnet_infer_s",
        "block_build_s",
        "total_s",
    ):
        if k in timings:
            print(f"  {k:20s} {timings[k]:7.2f}s")

    payload = {
        "video_filename": Path(video_path).name,
        "duration_seconds": duration,
        "timings_seconds": {k: round(v, 3) for k, v in timings.items()},
        "blocks": blocks,
    }
    json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    txt_out.write_text(render_text(blocks))
    print(f"✓ JSON: {json_out}")
    print(f"✓ TXT:  {txt_out}")


if __name__ == "__main__":
    main()
