"""
src/features/llm_input.py

Builds an LLM-friendly numbered transcript from the cached Whisper / YAMNet /
Silero data produced by `src.features.transcript`.

Each numbered "sentence" is one of:
  - speech     : a Silero speech interval, text = concatenated Whisper words
  - non-speech : a gap between Silero intervals (>= min_gap_s), shown as
                 [non-speech: <top events>, <duration>s]

Outputs:
  <stem>_llm_prompt.txt        ─ paste this into ChatGPT
  <stem>_sentence_index.json   ─ time-mapping table consumed by llm_merge.py

CLI:
  python3 -m src.features.llm_input -i videos_with_ads/test_003.mp4
"""

import json
import sys
from pathlib import Path

import click

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ingest import seconds_to_formatted  # noqa: F401
from src.features import paths as P

# Long non-speech runs (e.g. an ad with no transcribable speech) get split
# along YAMNet label-change boundaries so the LLM can localize ad sub-regions.
MAX_NON_SPEECH_S = 15.0
YAMNET_HOP_S = 0.48


PROMPT_PREAMBLE = """\
You will read a flowing transcript of a long-form video and identify which
numbered sentences are part of an advertisement (commercial ad, sponsorship,
brand insert, or product placement).

The transcript was machine-generated.  Speech sentences are transcribed by
Whisper (minor errors are OK — classify by meaning, not exact wording, e.g.
"the new containing out Doritos" is a Doritos brand mention).  Non-speech
sentences are summarized as "[non-speech: <top audio events>, <duration>s]"
using YAMNet's AudioSet ontology.

THINK STEP BY STEP.  Walk through these four steps explicitly in your reply
before outputting the final JSON.

Step 1 — Summarize the video.
  Read the ENTIRE transcript first.  Tell me what kind of video this is:
  genre, topic, tone, who's speaking, the overall narrative arc.  One paragraph.

Step 2 — Trace the narrative thread.
  Identify the main storyline / topic flow across the transcript: which
  scenes or topics follow each other, what characters are doing, what is
  being explained.  This is your reference for "what fits" the video.
  Two to four sentences.

Step 3 — Classify each region.

  For SPEECH sentences:
    Default = content.  Mark as ad only if there's a clear commercial
    signal — brand name in the text ("Doritos", "use code XYZ", "save
    20% at..."), call-to-action ("sponsored by", "subscribe", "click the
    link"), or characteristic ad copy.  Whisper may garble brand names;
    classify by meaning.

  For NON-SPEECH sentences (the [non-speech: ...] blocks):
    *** Default = ad. ***  A non-speech block is treated as ad UNLESS it
    flows naturally from the surrounding speech / narrative.  Concretely,
    for each non-speech run ask:

      (a) Does the preceding or following speech describe / imply the
          sounds in this block?  E.g., dialogue says "the chase is on!"
          and the next block is [non-speech: Vehicle, Tire squeal, Engine]
          — that fits the narrative; content.
      (b) Are the audio events in this block consistent with the scene
          the characters are clearly in?  E.g., a cooking scene with
          [non-speech: Pots and pans, Frying] — fits; content.
      (c) Does it sit smoothly between two speech sentences that continue
          the same topic — i.e., it doesn't interrupt the conversation?
          If yes, content.

    If the answer to all of (a), (b), (c) is no — the block does NOT fit
    the surrounding narrative, the events are out of place, the dialogue
    on either side is unrelated to the sounds — classify as ad.

    When uncertain about a non-speech block, default to ad.  We will
    filter false positives downstream using visual features, so erring
    toward ad here is the desired behaviour.

  In-universe ads (a character explicitly watching an ad on TV inside
  the narrative) are part of the story — label them content.

  In your reasoning, walk through each candidate ad region and explain:
  - which sentence numbers are part of it (e.g. S195–S201);
  - which test ((a) speech anchor, (b) scene consistency, (c) topical
    continuity) it failed, OR which commercial signal it matched.

Step 4 — Output.
  After your reasoning, output a JSON object on its own line at the very end
  of your reply.  No code fence, no markdown around it.  Format:

    {"ad_sentences": [195, 196, 197, 199]}

  - Each integer = sentence number (without the "S" prefix) belonging to
    an ad.
  - Sorted ascending, no duplicates.
  - If no ads, output {"ad_sentences": []}.

Begin.

================================================================================
TRANSCRIPT
================================================================================

"""


def _yamnet_top_in_range(
    yamnet_frames: list[dict],
    start: float,
    end: float,
    top_k: int = 3,
    min_overlap_frac: float = 0.15,
) -> list[str]:
    """Top-K YAMNet labels within [start, end] by overlap-weighted score."""
    if end <= start or not yamnet_frames:
        return []
    span = end - start
    weighted: dict[str, float] = {}
    overlap_sec: dict[str, float] = {}
    for fr in yamnet_frames:
        f_lo, f_hi = fr["start"], fr["end"]
        ov = max(0.0, min(end, f_hi) - max(start, f_lo))
        if ov <= 0:
            continue
        for label, score in fr["top"]:
            weighted[label] = weighted.get(label, 0.0) + score * ov
            overlap_sec[label] = overlap_sec.get(label, 0.0) + ov
    cands = [
        (lab, w) for lab, w in weighted.items()
        if (overlap_sec.get(lab, 0.0) / span) >= min_overlap_frac
    ]
    cands.sort(key=lambda x: -x[1])
    return [lab for lab, _ in cands[:top_k]]


def _group_yamnet_in_range(yamnet_frames, start, end):
    """Group consecutive YAMNet frames sharing smoothed_top within [start, end]
    using the hop (0.48 s) for boundaries so adjacent groups don't overlap."""
    indexed = []
    for i, fr in enumerate(yamnet_frames):
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
            lo, hi = _bounds(cur_a, cur_b)
            groups.append((lo, hi))
            cur_a, cur_b, cur_label = idx, idx, label
    lo, hi = _bounds(cur_a, cur_b)
    groups.append((lo, hi))
    return groups


def _split_long_nonspeech(raw, yamnet_frames, max_s):
    """Replace any non-speech run > max_s with YAMNet-label-grouped sub-runs."""
    out = []
    for s in raw:
        if s["kind"] == "speech" or (s["end"] - s["start"]) <= max_s:
            out.append(s)
            continue
        sub = _group_yamnet_in_range(yamnet_frames, s["start"], s["end"])
        if not sub:
            out.append(s)
            continue
        # Further split any still-too-long sub-run into max_s slices
        for lo, hi in sub:
            if hi - lo <= max_s:
                out.append({"kind": "non-speech", "start": lo, "end": hi})
            else:
                t = lo
                while t < hi:
                    end = min(t + max_s, hi)
                    out.append({"kind": "non-speech", "start": t, "end": end})
                    t = end
    return out


def build_sentences(
    silero_intervals: list[dict],
    whisper_words: list[dict],
    yamnet_frames: list[dict],
    duration: float,
    min_gap_s: float = 0.6,
    word_tol_s: float = 0.5,
    max_non_speech_s: float = MAX_NON_SPEECH_S,
) -> list[dict]:
    """
    Interleave speech intervals and non-speech gaps in time order; split any
    non-speech run longer than `max_non_speech_s` along YAMNet label changes;
    number them 1..N; attach Whisper words and YAMNet top-3 events.
    """
    intervals = sorted(silero_intervals, key=lambda iv: iv["start"])
    raw: list[dict] = []
    cursor = 0.0
    for iv in intervals:
        if iv["start"] - cursor >= min_gap_s:
            raw.append({"kind": "non-speech", "start": cursor, "end": iv["start"]})
        raw.append({"kind": "speech", "start": iv["start"], "end": iv["end"]})
        cursor = iv["end"]
    if duration - cursor >= min_gap_s:
        raw.append({"kind": "non-speech", "start": cursor, "end": duration})

    raw = _split_long_nonspeech(raw, yamnet_frames, max_non_speech_s)

    final = []
    for idx, s in enumerate(raw, start=1):
        events = _yamnet_top_in_range(yamnet_frames, s["start"], s["end"])
        text = ""
        if s["kind"] == "speech":
            words_in = [
                w for w in whisper_words
                if w["start"] >= s["start"] - word_tol_s
                and w["start"] <  s["end"]   + word_tol_s
            ]
            text = " ".join(w["text"] for w in words_in).strip()
            if not text:
                text = "(unintelligible)"
        final.append({
            "i": idx,
            "kind": s["kind"],
            "start": round(s["start"], 3),
            "end":   round(s["end"],   3),
            "text":  text,
            "events": events,
        })
    return final


def render_prompt(sentences: list[dict]) -> str:
    """Render numbered sentences into the prompt body."""
    lines = []
    for s in sentences:
        if s["kind"] == "speech":
            lines.append(f'S{s["i"]}: "{s["text"]}"')
        else:
            ev = ", ".join(s["events"]) if s["events"] else "—"
            lines.append(f'S{s["i"]}: [non-speech: {ev}, {s["end"] - s["start"]:.1f}s]')
    return "\n".join(lines)


@click.command()
@click.option("--input", "-i", "video_path", required=True, type=click.Path(),
              help="Video path (file does not need to exist if caches are present).")
@click.option("--prompt-out", default=None)
@click.option("--index-out", default=None)
def main(video_path: str, prompt_out, index_out):
    stem = Path(video_path).stem
    cache_silero  = P.silero(stem)
    cache_whisper = P.whisper_words(stem)
    cache_yamnet  = P.yamnet_frames(stem)

    missing = [p.name for p in (cache_silero, cache_whisper, cache_yamnet) if not p.exists()]
    if missing:
        raise click.ClickException(
            f"Missing cache file(s): {missing}. "
            f"Run `python3 -m src.features.transcript -i {video_path}` first."
        )

    silero  = json.loads(cache_silero.read_text())
    whisper = json.loads(cache_whisper.read_text())
    yamnet  = json.loads(cache_yamnet.read_text())

    duration = max(
        [w["end"] for w in whisper]
        + [iv["end"] for iv in silero]
        + [y["end"] for y in yamnet]
        + [0.0]
    )

    sentences = build_sentences(silero, whisper, yamnet, duration)

    prompt_path = Path(prompt_out) if prompt_out else P.llm_prompt(stem)
    index_path  = Path(index_out)  if index_out  else P.sentence_index(stem)

    prompt_path.write_text(PROMPT_PREAMBLE + render_prompt(sentences))
    index_path.write_text(json.dumps({
        "video_filename": Path(video_path).name,
        "duration_seconds": duration,
        "sentences": sentences,
    }, indent=2, ensure_ascii=False))

    speech    = sum(1 for s in sentences if s["kind"] == "speech")
    nonspeech = len(sentences) - speech

    print("=== llm_input ===")
    print(f"sentences: {len(sentences)}  (speech={speech}, non-speech={nonspeech})")
    print(f"prompt:    {prompt_path}  ({prompt_path.stat().st_size / 1024:.1f} KB)")
    print(f"index:     {index_path}")


if __name__ == "__main__":
    main()
