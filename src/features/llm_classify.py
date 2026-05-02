"""
src/features/llm_classify.py

Calls the OpenAI Chat Completions API with the transcript prompt produced by
`src.features.llm_input`, saves the model's reply, and (optionally) merges the
result into a player-schema *_segments.json.

Required:
  - OPENAI_API_KEY environment variable, or `--api-key`.
  - <stem>_llm_prompt.txt must exist (run `python3 -m src.features.llm_input -i <video>` first).

CLI:
  python3 -m src.features.llm_classify -i videos_with_ads/test_003.mp4 \\
      --model gpt-4o --merge

Notes on `--model`:
  Pass any model id the API accepts.  Defaults to "gpt-4o" because it's
  reliably available; users can override (e.g. "gpt-4.1", "gpt-5", "o3", etc.).
  If the API rejects the name we surface the error directly.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import click

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.features import paths as P  # noqa: E402

DEFAULT_MODEL = "gpt-5.5"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4000


def _load_dotenv() -> None:
    """
    Tiny in-process .env loader (no extra dependency).  Checks `src/.env`
    first (project convention), then `.env` in repo root.  First file wins
    per key (uses os.environ.setdefault).
    """
    candidates = [
        project_root / "src" / ".env",
        project_root / ".env",
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def call_openai(
    prompt_content,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """
    Send the prompt as a single user message; return dict with
    text, prompt_tokens, completion_tokens, elapsed_s.
    """
    from openai import OpenAI

    # Accept either OPENAI_API_KEY (canonical) or API_KEY (project convention).
    api_key = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "No API key found. Put OPENAI_API_KEY=... (or API_KEY=...) in "
            "src/.env, set it as an env var, or pass --api-key."
        )

    client = OpenAI(api_key=api_key)

    t0 = time.perf_counter()
    # Some newer reasoning models (o-series) reject `temperature` and
    # `max_tokens`; fall back gracefully.
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_content}],
    }
    try:
        kwargs_full = {**kwargs, "temperature": temperature, "max_tokens": max_tokens}
        response = client.chat.completions.create(**kwargs_full)
    except Exception as e:
        msg = str(e)
        if "temperature" in msg or "max_tokens" in msg or "max_completion_tokens" in msg:
            response = client.chat.completions.create(**kwargs)
        else:
            raise
    elapsed = time.perf_counter() - t0

    text = response.choices[0].message.content or ""
    usage = response.usage
    return {
        "text": text,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "elapsed_s": elapsed,
        "model": response.model,
    }


@click.command()
@click.option("--input", "-i", "video_path", required=True, type=click.Path(),
              help="Video path (file does not need to exist if a prompt cache is present).")
@click.option("--model", default=DEFAULT_MODEL, help=f"OpenAI model id (default: {DEFAULT_MODEL})")
@click.option("--api-key", default=None, help="Overrides OPENAI_API_KEY env var")
@click.option("--temperature", default=DEFAULT_TEMPERATURE, type=float)
@click.option("--max-tokens", default=DEFAULT_MAX_TOKENS, type=int)
@click.option("--prompt", "prompt_path", default=None,
              help="Override prompt path; defaults to <stem>_llm_prompt.txt")
@click.option("--reply-out", default=None,
              help="Override reply output; defaults to <stem>_llm_reply.txt")
@click.option("--merge", is_flag=True,
              help="After getting the reply, run llm_merge to produce *_segments.json")
def main(
    video_path: str,
    model: str,
    api_key: Optional[str],
    temperature: float,
    max_tokens: int,
    prompt_path: Optional[str],
    reply_out: Optional[str],
    merge: bool,
):
    _load_dotenv()

    stem = Path(video_path).stem
    prompt_path = Path(prompt_path) if prompt_path else P.llm_prompt(stem)
    reply_path = Path(reply_out) if reply_out else P.llm_reply(stem)

    if not prompt_path.exists():
        raise click.ClickException(
            f"Missing prompt file: {prompt_path}. "
            f"Run `python3 -m src.features.llm_input -i {video_path}` first."
        )

    prompt_text = prompt_path.read_text()

    parts = prompt_text.split("================================================================================\nTRANSCRIPT\n================================================================================")
    if len(parts) == 2:
        preamble = parts[0] + "================================================================================\nTRANSCRIPT\n================================================================================\n\n"
        
        print("Extracting screenshots for multimodal prompt...")
        import cv2
        import base64
        import numpy as np
        import tqdm
        
        index_path = P.sentence_index(stem)
        index_data = json.loads(index_path.read_text())
        
        prompt_content = [{"type": "text", "text": preamble}]
        
        cap = cv2.VideoCapture(video_path)
        sentences = index_data["sentences"]
        
        for s in tqdm.tqdm(sentences, desc="Screenshots"):
            ts = s["start"]
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((288, 512, 3), dtype=np.uint8)
                
            frame = cv2.resize(frame, (512, 288))
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            b64 = base64.b64encode(buffer).decode('utf-8')
            
            if s["kind"] == "speech":
                txt = f'S{s["i"]}: "{s["text"]}"\n'
            else:
                ev = ", ".join(s["events"]) if s["events"] else "—"
                txt = f'S{s["i"]}: [non-speech: {ev}, {s["end"] - s["start"]:.1f}s]\n'
                
            prompt_content.append({"type": "text", "text": f"Screenshot at {ts:.1f}s for S{s['i']}:"})
            prompt_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})
            prompt_content.append({"type": "text", "text": txt})
            
        cap.release()
    else:
        prompt_content = prompt_text

    print("=== llm_classify ===")
    print(f"prompt:      {prompt_path} ({prompt_path.stat().st_size / 1024:.1f} KB)")
    print(f"model:       {model}")
    print(f"temperature: {temperature}")
    print(f"max_tokens:  {max_tokens}")
    if isinstance(prompt_content, list):
        print(f"images:      {len(index_data['sentences'])} (low detail)")
    print()
    print("calling OpenAI API ...")

    try:
        result = call_openai(
            prompt_content,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        raise click.ClickException(f"API call failed: {e}")

    reply_path.write_text(result["text"])

    print(f"  → resolved model: {result['model']}")
    print(f"  → prompt tokens:  {result['prompt_tokens']}")
    print(f"  → completion tokens: {result['completion_tokens']}")
    print(f"  → elapsed: {result['elapsed_s']:.1f}s")
    print(f"reply saved: {reply_path}")

    if merge:
        from src.features.llm_merge import run_merge
        print()
        run_merge(video_path, str(reply_path))


if __name__ == "__main__":
    main()
