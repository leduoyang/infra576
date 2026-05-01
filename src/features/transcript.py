"""
transcript.py – Speech-to-text feature extraction using OpenAI Whisper.

Provides:
  - Full video transcription with word-level timestamps
  - Per-segment transcript text extraction
  - Keyword-based signals for sponsorship, self-promo, recap detection
"""

from __future__ import annotations
import re
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Keyword dictionaries for non-content detection
# ---------------------------------------------------------------------------

SPONSOR_KEYWORDS = [
    "sponsor", "sponsored", "sponsorship",
    "brought to you by", "thanks to",
    "use code", "use my code", "promo code", "discount code",
    "link in the description", "link below", "check out",
    "sign up", "free trial",
    "today's sponsor", "this episode is sponsored",
    "this video is sponsored", "this segment is sponsored",
    "go to", "visit", "head over to",
]

SELF_PROMO_KEYWORDS = [
    "subscribe", "like and subscribe", "hit the bell",
    "notification bell", "smash that like",
    "follow me on", "follow us on",
    "join the channel", "become a member",
    "check out my", "check out our",
    "merch", "merchandise", "store",
    "patreon", "buy me a coffee", "ko-fi",
    "my other channel", "second channel",
    "don't forget to like", "leave a comment",
    "share this video",
]

RECAP_KEYWORDS = [
    "previously on", "last time",
    "in the last episode", "last episode",
    "as we discussed", "as i mentioned",
    "to recap", "quick recap", "let's recap",
    "recap of", "summary of",
    "if you missed", "in case you missed",
]

INTRO_KEYWORDS = [
    "welcome to", "welcome back",
    "hey guys", "hey everyone", "hello everyone",
    "what's up", "what is up",
    "today we", "today i", "today's video",
    "in this video", "in today's",
    "my name is", "i'm your host",
]

OUTRO_KEYWORDS = [
    "thanks for watching", "thank you for watching",
    "see you next time", "see you in the next",
    "until next time", "catch you later",
    "goodbye", "bye bye", "take care",
    "that's it for today", "that's all for today",
    "this has been", "peace out",
    "don't forget to subscribe",
]


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    language: str = "en",
) -> list[dict]:
    """
    Transcribe audio using OpenAI Whisper.

    Returns a list of word-level segments:
        [{"start": 0.0, "end": 1.5, "text": "Hello world"}, ...]

    Falls back gracefully if Whisper is not installed.
    """
    try:
        import whisper
    except ImportError:
        print("[transcript] Whisper not installed. Run: pip install openai-whisper")
        print("[transcript] Falling back to empty transcript.")
        return []

    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            verbose=False,
        )

        # Extract segments with timestamps
        transcript_segments = []
        for segment in result.get("segments", []):
            transcript_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
            })
        return transcript_segments

    except Exception as e:
        print(f"[transcript] Whisper transcription failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Per-segment transcript extraction
# ---------------------------------------------------------------------------

def get_segment_transcript(
    transcript: list[dict],
    start_sec: float,
    end_sec: float,
) -> str:
    """Extract concatenated transcript text for a time range."""
    parts = []
    for seg in transcript:
        # Include if there's any overlap
        if seg["end"] > start_sec and seg["start"] < end_sec:
            parts.append(seg["text"])
    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Keyword signal scoring
# ---------------------------------------------------------------------------

def analyze_transcript_features(
    transcript: list[dict],
    start_sec: float,
    end_sec: float,
) -> dict:
    """
    Analyze transcript text for a segment and return keyword signal scores.

    Returns dict with:
        transcript_text, sponsor_score, self_promo_score,
        recap_score, intro_score, outro_score, word_count,
        words_per_second
    """
    text = get_segment_transcript(transcript, start_sec, end_sec)
    text_lower = text.lower()
    duration = end_sec - start_sec

    sponsor_score = _keyword_score(text_lower, SPONSOR_KEYWORDS)
    self_promo_score = _keyword_score(text_lower, SELF_PROMO_KEYWORDS)
    recap_score = _keyword_score(text_lower, RECAP_KEYWORDS)
    intro_score = _keyword_score(text_lower, INTRO_KEYWORDS)
    outro_score = _keyword_score(text_lower, OUTRO_KEYWORDS)

    words = text.split()
    word_count = len(words)
    wps = word_count / duration if duration > 0 else 0.0

    return {
        "transcript_text": text,
        "sponsor_score": sponsor_score,
        "self_promo_score": self_promo_score,
        "recap_score": recap_score,
        "intro_keyword_score": intro_score,
        "outro_keyword_score": outro_score,
        "word_count": word_count,
        "words_per_second": round(wps, 2),
    }


def _keyword_score(text: str, keywords: list[str]) -> float:
    """
    Count how many keywords/phrases appear in the text.
    Returns a normalized score in [0, 1].
    """
    if not text:
        return 0.0
    hits = sum(1 for kw in keywords if kw in text)
    # Normalize: 1 hit = 0.3, 2 hits = 0.5, 3+ = 0.7+, capped at 1.0
    return min(1.0, hits * 0.25) if hits > 0 else 0.0


# ---------------------------------------------------------------------------
# Global transcript profile
# ---------------------------------------------------------------------------

def compute_global_transcript_profile(transcript: list[dict], duration: float) -> dict:
    """
    Compute global transcript statistics for the whole video.
    """
    if not transcript:
        return {
            "total_word_count": 0,
            "avg_words_per_second": 0.0,
            "has_transcript": False,
        }

    all_text = " ".join(seg["text"] for seg in transcript)
    word_count = len(all_text.split())
    wps = word_count / duration if duration > 0 else 0.0

    return {
        "total_word_count": word_count,
        "avg_words_per_second": round(wps, 2),
        "has_transcript": True,
    }
