#!/usr/bin/env bash
# run.sh — end-to-end pipeline: video → <stem>_segments.json
#
# Stages (cached; safe to re-run):
#   1. transcript    : audio extract + Silero VAD + Whisper words + YAMNet frames
#   2. llm_input     : numbered transcript + sentence index for the LLM
#   3. llm_classify  : call OpenAI, parse reply, merge into player schema
#
# Reads API key from src/.env (OPENAI_API_KEY or API_KEY).
# Final output:  <stem>_segments.json  (in repo root, consumed by player/index.html)
# Intermediates: work/<process>/<stem>.*
#
# Usage:
#   ./run.sh <video_path> [extra flags forwarded to llm_classify]
#
# Examples:
#   ./run.sh videos_with_ads/test_003.mp4
#   ./run.sh videos_with_ads/test_003.mp4 --model gpt-4o
#   ./run.sh videos_with_ads/test_003.mp4 --temperature 0.2

set -euo pipefail

if [ $# -lt 1 ]; then
    sed -n '2,/^$/p' "$0" >&2
    exit 1
fi

VIDEO="$1"
shift

# Prefer the project virtualenv, then any configured Python, then system python3.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if [ -x .venv/bin/python ]; then
        PY=.venv/bin/python
    elif [ -x /opt/anaconda3/bin/python ]; then
        PY=/opt/anaconda3/bin/python
    else
        PY=python3
    fi
fi

# Some native ML dependencies used by the transcript stage leave a semaphore
# registered at interpreter shutdown on macOS/Python 3.9. The pipeline output is
# already written by then, so keep the warning from being mistaken for a failure.
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}ignore:resource_tracker:UserWarning"

fmt_dur() {
    local s=$1
    printf '%dm%02ds' $((s / 60)) $((s % 60))
}

T0=$SECONDS

echo "=== [1/3] transcript ==="
"$PY" -m src.features.transcript -i "$VIDEO"
T1=$SECONDS

echo
echo "=== [2/3] llm_input ==="
"$PY" -m src.features.llm_input -i "$VIDEO"
T2=$SECONDS

echo
echo "=== [3/3] llm_classify + merge ==="
"$PY" -m src.features.llm_classify -i "$VIDEO" --merge "$@"
T3=$SECONDS

echo
echo "=== run.sh totals ==="
printf '  transcript            %s\n'  "$(fmt_dur $((T1 - T0)))"

# Per-model breakdown from work/transcript/<stem>.json (written by transcript.py).
"$PY" - "$VIDEO" <<'PY' || true
import json, sys
from pathlib import Path
stem = Path(sys.argv[1]).stem
p = Path(f"work/transcript/{stem}.json")
if not p.exists():
    sys.exit(0)
ts = json.loads(p.read_text()).get("timings_seconds", {})
labels = [
    ("audio_extract_s",  "audio_extract"),
    ("silero_s",         "silero"),
    ("whisper_load_s",   "whisper (load)"),
    ("whisper_infer_s",  "whisper (infer)"),
    ("yamnet_load_s",    "yamnet (load)"),
    ("yamnet_infer_s",   "yamnet (infer)"),
    ("block_build_s",    "block_build"),
]
for key, label in labels:
    if key in ts:
        print(f"      {label:18s} {ts[key]:7.2f}s")
PY

printf '  llm_input             %s\n'  "$(fmt_dur $((T2 - T1)))"
printf '  llm_classify + merge  %s\n'  "$(fmt_dur $((T3 - T2)))"
printf '  TOTAL                 %s\n'  "$(fmt_dur $((T3 - T0)))"

# ./run.sh videos_with_ads/test_003.mp4
