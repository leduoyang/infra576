"""
src/features/paths.py

One place that knows where every intermediate artifact lives.  The repo root
holds only final outputs (the `<stem>_segments.json` consumed by the player);
all per-process intermediates live under `work/<process>/<stem>.*`.

Layout:
  infra576/
    <stem>_segments.json        ← FINAL, in repo root
    work/
      audio/<stem>.wav
      silero/<stem>.json
      whisper/<stem>.json
      yamnet/<stem>.json
      transcript/<stem>.json    ← block-builder output (legacy / debug)
      transcript/<stem>.txt
      llm_input/<stem>.prompt.txt
      llm_input/<stem>.index.json
      llm_reply/<stem>.txt
"""

from pathlib import Path

WORK_DIR = Path("work")


def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def audio(stem: str) -> Path:
    return _ensure(WORK_DIR / "audio") / f"{stem}.wav"


def silero(stem: str) -> Path:
    return _ensure(WORK_DIR / "silero") / f"{stem}.json"


def whisper_words(stem: str) -> Path:
    return _ensure(WORK_DIR / "whisper") / f"{stem}.json"


def yamnet_frames(stem: str) -> Path:
    return _ensure(WORK_DIR / "yamnet") / f"{stem}.json"


def transcript_json(stem: str) -> Path:
    return _ensure(WORK_DIR / "transcript") / f"{stem}.json"


def transcript_txt(stem: str) -> Path:
    return _ensure(WORK_DIR / "transcript") / f"{stem}.txt"


def llm_prompt(stem: str) -> Path:
    return _ensure(WORK_DIR / "llm_input") / f"{stem}.prompt.txt"


def sentence_index(stem: str) -> Path:
    return _ensure(WORK_DIR / "llm_input") / f"{stem}.index.json"


def llm_reply(stem: str) -> Path:
    return _ensure(WORK_DIR / "llm_reply") / f"{stem}.txt"


def segments(stem: str) -> Path:
    """Final player JSON — stays in repo root."""
    return Path(f"{stem}_segments.json")
