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


_LIST_RE = re.compile(
    r'"(?P<key>ad_sentences|intro_sentences|outro_sentences)"\s*:\s*\[(?P<ids>[^\]]*)\]',
    re.DOTALL,
)


def _parse_id_list(ids_text: str) -> list[int]:
    return sorted({int(x.strip()) for x in ids_text.split(",") if x.strip()})


def parse_classification(reply_text: str) -> dict:
    """
    Extract ad / intro / outro sentence-id lists from the LLM reply.  Tolerates:
      - chain-of-thought prose around the JSON,
      - markdown code fences,
      - a missing intro_sentences / outro_sentences key (treated as empty),
      - multiple JSON candidates (last one wins; keys are matched independently).
    Returns: {"ad": [...], "intro": [...], "outro": [...]}.
    """
    cleaned = reply_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[-1].startswith("```"):
            cleaned = "\n".join(lines[1:-1])
        else:
            cleaned = "\n".join(lines[1:])

    by_key: dict[str, list[int]] = {}
    for m in _LIST_RE.finditer(cleaned):
        # Last occurrence wins (Step-4 line at the end of the reply).
        by_key[m.group("key")] = _parse_id_list(m.group("ids"))

    if not by_key:
        # Fallback: maybe the model returned a bare JSON object.
        try:
            data = json.loads(cleaned)
            for key in ("ad_sentences", "intro_sentences", "outro_sentences"):
                if key in data:
                    by_key[key] = sorted({int(x) for x in data[key]})
        except Exception as e:
            raise click.ClickException(
                'Could not find {"ad_sentences": [...], "intro_sentences": ...} '
                f"in the LLM reply. First 200 chars: {cleaned[:200]!r}"
            ) from e

    if "ad_sentences" not in by_key:
        raise click.ClickException(
            'LLM reply lacks "ad_sentences" key. '
            f"First 200 chars: {cleaned[:200]!r}"
        )

    # Ad takes priority — strip ad ids from intro/outro to keep categories disjoint.
    ad_ids = set(by_key.get("ad_sentences", []))
    intro_ids = set(by_key.get("intro_sentences", [])) - ad_ids
    outro_ids = set(by_key.get("outro_sentences", [])) - ad_ids
    return {
        "ad": sorted(ad_ids),
        "intro": sorted(intro_ids),
        "outro": sorted(outro_ids),
    }


# Backwards-compat shim for any caller that still imports the old name.
def parse_ad_sentences(reply_text: str) -> list[int]:
    return parse_classification(reply_text)["ad"]


def runs_from_sentences(
    sentences: list[dict],
    ids: set[int],
    kind: str,
) -> list[tuple[float, float, str]]:
    """Convert sentence ids into typed time runs, merging across short gaps."""
    spans = sorted((s["start"], s["end"]) for s in sentences if s["i"] in ids)
    if not spans:
        return []
    merged: list[list[float]] = [list(spans[0])]
    for start, end in spans[1:]:
        if start - merged[-1][1] <= MERGE_GAP_S:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e, kind) for s, e in merged]


# Higher number = higher priority when grid cells overlap multiple kinds.
_KIND_PRIORITY = {"ad": 3, "intro": 2, "outro": 2, "video_content": 0}


def grid_blocks(
    duration: float,
    typed_runs: list[tuple[float, float, str]],
    grid_s: float = GRID_S,
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> list[dict]:
    """Slice the timeline into uniform `grid_s` blocks.

    For each cell, accumulate per-kind overlap.  The cell's label is the kind
    with highest overlap above `overlap_threshold`; ties broken by priority
    (ad > intro/outro > content).  Cells with no qualifying overlap are
    `video_content`.
    """
    blocks = []
    t = 0.0
    while t < duration:
        end = min(t + grid_s, duration)
        cell_size = max(end - t, 1e-6)
        per_kind: dict[str, float] = {}
        for r_start, r_end, r_kind in typed_runs:
            ov = max(0.0, min(end, r_end) - max(t, r_start))
            if ov > 0:
                per_kind[r_kind] = per_kind.get(r_kind, 0.0) + ov

        best_kind = "video_content"
        best_score = -1.0
        for kind, ov in per_kind.items():
            frac = ov / cell_size
            if frac < overlap_threshold:
                continue
            score = (ov, _KIND_PRIORITY.get(kind, 1))
            if score > (best_score, _KIND_PRIORITY.get(best_kind, -1)):
                best_kind = kind
                best_score = ov

        blocks.append({
            "start_seconds": t,
            "end_seconds": end,
            "duration_seconds": end - t,
            "is_content": best_kind == "video_content",
            "segment_type": best_kind,
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
    classification = parse_classification(reply_text)
    ad_ids    = set(classification["ad"])
    intro_ids = set(classification["intro"])
    outro_ids = set(classification["outro"])

    typed_runs = (
        runs_from_sentences(sentences, ad_ids,    "ad")
        + runs_from_sentences(sentences, intro_ids, "intro")
        + runs_from_sentences(sentences, outro_ids, "outro")
    )
    grid = grid_blocks(duration, typed_runs)
    classified = merge_consecutive(grid)

    result = build_output(video_path, metadata, classified)
    Path(output_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))

    if not quiet:
        print("=== llm_merge ===")
        print(f"video duration:        {seconds_to_formatted(duration)}")
        print(f"ad / intro / outro:    {len(ad_ids)} / {len(intro_ids)} / {len(outro_ids)} sentences")
        for kind, ids in (("ad", ad_ids), ("intro", intro_ids), ("outro", outro_ids)):
            runs = [r for r in typed_runs if r[2] == kind]
            if not runs:
                continue
            print(f"  {kind} runs ({len(runs)}):")
            for s, e, _k in runs:
                print(f"    [{seconds_to_formatted(s)} - {seconds_to_formatted(e)}]  ({e - s:.1f}s)")
        counts = {k: 0 for k in ("video_content", "ad", "intro", "outro")}
        for seg in result["timeline_segments"]:
            counts[seg["type"]] = counts.get(seg["type"], 0) + 1
        summary = ", ".join(f"{counts[k]} {k}" for k in ("video_content", "ad", "intro", "outro") if counts.get(k))
        print(f"timeline segments:     {len(result['timeline_segments'])}  ({summary})")
        print(f"output: {output_path}")

    return {
        "classification": classification,
        "typed_runs": typed_runs,
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
