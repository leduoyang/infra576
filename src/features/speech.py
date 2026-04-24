"""
speech.py – Whisper-based speech features.

Transcribes a video once (cached to JSON), then derives per-segment features:
  - word_rate: words per second in the shot
  - no_speech_prob: average "no speech" probability across overlapping whisper segs
  - sponsor_hit: 1.0 if common sponsor/ad keywords land inside the shot, else 0.0
  - avg_logprob: confidence (lower = more uncertain, often music/noise)
"""

import json
import re
from pathlib import Path
from typing import List

import whisper

_MODEL = None
_MODEL_NAME = "tiny.en"

SPONSOR_PATTERNS = re.compile(
    r"\b(sponsor|sponsored by|brought to you by|promo code|use code|"
    r"discount|subscribe|follow us|check out|link in|our partner|"
    r"today's video is|this video is|special offer|limited time|"
    r"buy now|visit|click the link)\b",
    re.IGNORECASE,
)


def _load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = whisper.load_model(_MODEL_NAME)
    return _MODEL


def transcribe(audio_path: str, cache_path: str = None) -> dict:
    """Transcribe the audio; cache to JSON. Returns whisper result dict."""
    if cache_path and Path(cache_path).exists():
        return json.loads(Path(cache_path).read_text())

    model = _load_model()
    result = model.transcribe(audio_path, verbose=False, word_timestamps=False, fp16=False)

    trimmed = {
        "language": result.get("language"),
        "text": result.get("text", ""),
        "segments": [
            {
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
                "avg_logprob": s.get("avg_logprob", 0.0),
                "no_speech_prob": s.get("no_speech_prob", 0.0),
            }
            for s in result.get("segments", [])
        ],
    }
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(trimmed, indent=2, ensure_ascii=False))
    return trimmed


def speech_features_for_segment(transcript: dict, start: float, end: float) -> dict:
    """Derive per-shot features from an overlapping slice of the transcript."""
    overlapping = [
        s for s in transcript.get("segments", [])
        if s["end"] > start and s["start"] < end
    ]
    if not overlapping:
        return {
            "word_rate": 0.0,
            "no_speech_prob": 1.0,
            "avg_logprob": -2.0,
            "sponsor_hit": 0.0,
            "word_count": 0,
        }

    word_count = 0
    sponsor_hit = 0.0
    no_speech = []
    logprobs = []
    for s in overlapping:
        text = s.get("text", "")
        words = text.strip().split()
        word_count += len(words)
        if SPONSOR_PATTERNS.search(text):
            sponsor_hit = 1.0
        no_speech.append(s.get("no_speech_prob", 0.0))
        logprobs.append(s.get("avg_logprob", 0.0))

    dur = max(1e-3, end - start)
    return {
        "word_rate": word_count / dur,
        "no_speech_prob": sum(no_speech) / len(no_speech),
        "avg_logprob": sum(logprobs) / len(logprobs),
        "sponsor_hit": sponsor_hit,
        "word_count": word_count,
    }
