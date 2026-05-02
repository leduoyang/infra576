"""
src/features/llm_merge.py

Reads the LLM's reply (which may include chain-of-thought prose followed by a
final JSON line) plus the sentence index produced by `llm_input.py`, and
emits a player-schema *_segments.json on a 5 s grid.

Pipeline:
  1. Parse {"ad_sentences": [...]} from the reply (tolerant of surrounding text).
  2. Look up each ad-sentence's time range from the sentence index.
  3. Merge adjacent ad sentences (gap < merge_gap_s) into ad runs.
  4. Slice the timeline into 5 s blocks.  Each block is "ad" if it overlaps
     any ad run by >= 50%.
  5. Merge consecutive same-label blocks into runs.
  6. Build the player JSON via src.export.build_output (existing schema).

CLI:
  python3 -m src.features.llm_merge \\
      -v videos_with_ads/test_003.mp4 \\
      -l test_003_llm_reply.txt
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

import click

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ingest import seconds_to_formatted, get_video_metadata
from src.export import build_output
from src.features import paths as P

GRID_S = 5.0
MERGE_GAP_S = 5.0
OVERLAP_THRESHOLD = 0.5


_AD_JSON_RE = re.compile(
    r'\{\s*"ad_sentences"\s*:\s*\[(?P<ids>[^\]]*)\]\s*\}',
    re.DOTALL,
)


def parse_ad_sentences(reply_text: str) -> list[int]:
    """
    Extract ad-sentence indices from the LLM reply.  Tolerates:
      - surrounding chain-of-thought prose,
      - markdown code fences,
      - multiple JSON candidates (we take the LAST match — the final answer).
    """
    cleaned = reply_text.strip()
    # Strip outer code fences if the whole reply is fenced
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[-1].startswith("```"):
            cleaned = "\n".join(lines[1:-1])
        else:
            cleaned = "\n".join(lines[1:])

    matches = list(_AD_JSON_RE.finditer(cleaned))
    if not matches:
        # Last-ditch: maybe the LLM returned a bare JSON
        try:
            data = json.loads(cleaned)
            return [int(x) for x in data.get("ad_sentences", [])]
        except Exception as e:
            raise click.ClickException(
                "Could not find {\"ad_sentences\": [...]} in the LLM reply. "
                f"First 200 chars: {cleaned[:200]!r}"
            ) from e

    ids_text = matches[-1].group("ids")
    ids = [int(x.strip()) for x in ids_text.split(",") if x.strip()]
    # dedupe + sort
    return sorted(set(ids))


def ad_runs_from_sentences(sentences: list[dict], ad_ids: set[int]) -> list[tuple[float, float]]:
    """Convert ad-sentence ids into time runs, merging when their gap is short."""
    spans = sorted(
        (s["start"], s["end"]) for s in sentences if s["i"] in ad_ids
    )
    if not spans:
        return []
    merged: list[list[float]] = [list(spans[0])]
    for start, end in spans[1:]:
        if start - merged[-1][1] <= MERGE_GAP_S:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def grid_blocks(
    duration: float,
    ad_runs: list[tuple[float, float]],
    grid_s: float = GRID_S,
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> list[dict]:
    """Slice the timeline into uniform `grid_s` blocks; mark ad if overlap >= threshold."""
    blocks = []
    t = 0.0
    while t < duration:
        end = min(t + grid_s, duration)
        ov = 0.0
        for r_start, r_end in ad_runs:
            ov += max(0.0, min(end, r_end) - max(t, r_start))
        is_ad = (ov / max(end - t, 1e-6)) >= overlap_threshold
        blocks.append({
            "start_seconds": t,
            "end_seconds": end,
            "duration_seconds": end - t,
            "is_content": not is_ad,
            "segment_type": "ad" if is_ad else "video_content",
            "confidence": 1.0,
        })
        t = end
    return blocks


def merge_consecutive(classified: list[dict]) -> list[dict]:
    """Merge consecutive same-label blocks into one entry each."""
    if not classified:
        return []
    merged = [dict(classified[0])]
    for c in classified[1:]:
        prev = merged[-1]
        if c["segment_type"] == prev["segment_type"]:
            prev["end_seconds"] = c["end_seconds"]
            prev["duration_seconds"] = prev["end_seconds"] - prev["start_seconds"]
        else:
            merged.append(dict(c))
    return merged


def run_merge(
    video_path: str,
    labels_path: str,
    index_path: Optional[str] = None,
    output_path: Optional[str] = None,
    quiet: bool = False,
) -> dict:
    """Programmatic entry point.  Returns dict with ad_runs, output_path, and player payload."""
    stem = Path(video_path).stem
    index_path = index_path or str(P.sentence_index(stem))
    output_path = output_path or str(P.segments(stem))

    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"Missing sentence index: {index_path}. "
            f"Run `python3 -m src.features.llm_input -i {video_path}` first."
        )

    index = json.loads(Path(index_path).read_text())
    sentences = index["sentences"]
    duration_audio = index["duration_seconds"]

    if Path(video_path).exists():
        metadata = get_video_metadata(video_path)
        duration = max(metadata["duration_seconds"], duration_audio)
    else:
        # Video file not present locally; fall back to the duration recorded
        # in the sentence index (audio extent from the cached intermediates).
        metadata = {
            "duration_seconds": duration_audio,
            "resolution": "unknown",
            "width": 0, "height": 0, "fps": 0.0,
            "has_audio": True, "video_codec": "unknown", "audio_codec": "unknown",
        }
        duration = duration_audio

    reply_text = Path(labels_path).read_text()
    ad_ids_list = parse_ad_sentences(reply_text)
    ad_ids = set(ad_ids_list)

    runs = ad_runs_from_sentences(sentences, ad_ids)
    grid = grid_blocks(duration, runs)
    classified = merge_consecutive(grid)

    result = build_output(video_path, metadata, classified)
    Path(output_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))

    if not quiet:
        print("=== llm_merge ===")
        print(f"video duration:      {seconds_to_formatted(duration)}")
        print(f"ad sentences (LLM):  {len(ad_ids)}  → {sorted(ad_ids)}")
        print(f"ad runs (merged):    {len(runs)}")
        for s, e in runs:
            print(f"  [{seconds_to_formatted(s)} - {seconds_to_formatted(e)}]  ({e - s:.1f}s)")
        print(f"timeline segments:   {len(result['timeline_segments'])} "
              f"({result['num_ads_inserted']} ad / "
              f"{len(result['timeline_segments']) - result['num_ads_inserted']} content)")
        print(f"output: {output_path}")

    return {
        "ad_ids": sorted(ad_ids),
        "ad_runs": runs,
        "output_path": output_path,
        "result": result,
    }


@click.command()
@click.option("--video", "-v", "video_path", required=True, type=click.Path(),
              help="Video path (file does not need to exist; falls back to cached duration).")
@click.option("--labels", "-l", "labels_path", required=True, type=click.Path(exists=True),
              help="LLM reply file (.txt or .json with chain-of-thought + final JSON).")
@click.option("--index", "-x", "index_path", default=None, type=click.Path(),
              help="Sentence index; defaults to <stem>_sentence_index.json")
@click.option("--output", "-o", "output_path", default=None,
              help="Output player JSON; defaults to <stem>_segments.json")
def main(video_path, labels_path, index_path, output_path):
    try:
        run_merge(video_path, labels_path, index_path, output_path)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))


if __name__ == "__main__":
    main()
