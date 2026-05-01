"""Unit tests for src/features/transcript.py – keyword scoring & extraction."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.features.transcript import (
    get_segment_transcript,
    analyze_transcript_features,
    _keyword_score,
    compute_global_transcript_profile,
    SPONSOR_KEYWORDS,
    SELF_PROMO_KEYWORDS,
    RECAP_KEYWORDS,
    INTRO_KEYWORDS,
    OUTRO_KEYWORDS,
)


SAMPLE_TRANSCRIPT = [
    {"start": 0.0, "end": 5.0, "text": "Welcome to the show everyone"},
    {"start": 5.0, "end": 10.0, "text": "today we are going to talk about AI"},
    {"start": 10.0, "end": 20.0, "text": "but first this video is sponsored by NordVPN"},
    {"start": 20.0, "end": 30.0, "text": "use code MYCODE for a free trial"},
    {"start": 30.0, "end": 100.0, "text": "So let us get into the main content about neural networks"},
    {"start": 100.0, "end": 110.0, "text": "thanks for watching see you next time"},
    {"start": 110.0, "end": 115.0, "text": "don't forget to subscribe and hit the bell"},
]


class TestGetSegmentTranscript:
    def test_exact_range(self):
        text = get_segment_transcript(SAMPLE_TRANSCRIPT, 0.0, 10.0)
        assert "Welcome" in text
        assert "today" in text

    def test_partial_overlap(self):
        text = get_segment_transcript(SAMPLE_TRANSCRIPT, 8.0, 25.0)
        assert "sponsored" in text
        assert "NordVPN" in text

    def test_no_overlap(self):
        text = get_segment_transcript(SAMPLE_TRANSCRIPT, 200.0, 300.0)
        assert text == ""

    def test_empty_transcript(self):
        assert get_segment_transcript([], 0.0, 100.0) == ""


class TestKeywordScore:
    def test_no_match(self):
        assert _keyword_score("hello world", SPONSOR_KEYWORDS) == 0.0

    def test_single_match(self):
        score = _keyword_score("this video is sponsored by acme", SPONSOR_KEYWORDS)
        assert score > 0.0

    def test_multiple_matches(self):
        text = "sponsored by acme, use code MYCODE, link in the description"
        score = _keyword_score(text, SPONSOR_KEYWORDS)
        assert score > 0.25

    def test_promo_keywords(self):
        text = "don't forget to subscribe and hit the bell"
        score = _keyword_score(text, SELF_PROMO_KEYWORDS)
        assert score > 0.0

    def test_recap_keywords(self):
        text = "previously on the show, let's recap what happened"
        score = _keyword_score(text, RECAP_KEYWORDS)
        assert score > 0.0

    def test_intro_keywords(self):
        text = "welcome to the show, today we are going to discuss"
        score = _keyword_score(text, INTRO_KEYWORDS)
        assert score > 0.0

    def test_outro_keywords(self):
        text = "thanks for watching, see you next time"
        score = _keyword_score(text, OUTRO_KEYWORDS)
        assert score > 0.0


class TestAnalyzeTranscriptFeatures:
    def test_sponsor_segment(self):
        feats = analyze_transcript_features(SAMPLE_TRANSCRIPT, 10.0, 30.0)
        assert feats["sponsor_score"] > 0.0
        assert feats["word_count"] > 0
        assert "sponsored" in feats["transcript_text"].lower()

    def test_intro_segment(self):
        feats = analyze_transcript_features(SAMPLE_TRANSCRIPT, 0.0, 10.0)
        assert feats["intro_keyword_score"] > 0.0
        assert "welcome" in feats["transcript_text"].lower()

    def test_outro_segment(self):
        feats = analyze_transcript_features(SAMPLE_TRANSCRIPT, 100.0, 115.0)
        assert feats["outro_keyword_score"] > 0.0

    def test_content_segment(self):
        feats = analyze_transcript_features(SAMPLE_TRANSCRIPT, 30.0, 100.0)
        assert feats["sponsor_score"] == 0.0
        assert feats["self_promo_score"] == 0.0

    def test_words_per_second(self):
        feats = analyze_transcript_features(SAMPLE_TRANSCRIPT, 0.0, 10.0)
        assert feats["words_per_second"] > 0


class TestGlobalTranscriptProfile:
    def test_with_transcript(self):
        profile = compute_global_transcript_profile(SAMPLE_TRANSCRIPT, 115.0)
        assert profile["has_transcript"] is True
        assert profile["total_word_count"] > 0
        assert profile["avg_words_per_second"] > 0

    def test_empty_transcript(self):
        profile = compute_global_transcript_profile([], 100.0)
        assert profile["has_transcript"] is False
        assert profile["total_word_count"] == 0
