#!/usr/bin/env python3
"""
transcribe_audio.py - Transcribe a single episode audio file with Whisper.

Given a `channel_name` and `episode_id`, this script looks under
`data/channels/<channel_name>/<episode_id>/audio.mp3`, sends it through the
selected Whisper model, and stores the raw JSON transcript alongside the
episode directory (defaults to `transcript.json`).

Requirements:
    pip install git+https://github.com/openai/whisper.git
    ffmpeg (installed and available on PATH)

Usage:
    python scripts/clip_pipeline/transcribe_audio.py CHANNEL EPISODE
    python scripts/clip_pipeline/transcribe_audio.py CHANNEL EPISODE --model-size medium
    python scripts/clip_pipeline/transcribe_audio.py CHANNEL EPISODE \\
        --channels-dir data/channels \\
        --audio-filename audio.mp3 \\
        --output-filename transcript.json \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
try:
    import whisper  # type: ignore[attr-defined]
except ImportError:
    print("❌ ERROR: whisper is not installed. Install with:")
    print("    pip install git+https://github.com/openai/whisper.git")
    sys.exit(1)


def transcribe_file(
    model: "whisper.Whisper",
    input_path: Path,
    output_path: Path,
    overwrite: bool,
    language: str | None,
) -> bool:
    """Transcribe a single audio file and dump the JSON output."""
    if output_path.exists() and not overwrite:
        print(f"⏭️  Skipping {input_path.name} (already transcribed)")
        return True

    print(f"🎤 Transcribing {input_path.name}...")

    try:
        result = model.transcribe(
            str(input_path),
            language=language,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"   ❌ Error while transcribing {input_path.name}: {exc}")
        return False

    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"   ❌ Failed to write {output_path.name}: {exc}")
        return False

    segments = result.get("segments") or []
    duration = segments[-1]["end"] if segments else 0.0
    print(
        f"   ✅ Saved {output_path.name} "
        f"(segments: {len(segments)}, duration: {duration:.1f}s)"
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a single channel episode audio file with OpenAI Whisper.",
    )
    parser.add_argument(
        "channel_name",
        type=str,
        help="Name of the channel directory inside data/channels.",
    )
    parser.add_argument(
        "episode_id",
        type=str,
        help="Episode folder name inside the channel directory.",
    )
    parser.add_argument(
        "--channels-dir",
        type=Path,
        default=Path("data/channels"),
        help="Root directory that stores channel data (default: data/channels).",
    )
    parser.add_argument(
        "--audio-filename",
        type=str,
        default="audio.mp3",
        help="Audio filename to read within the episode directory (default: audio.mp3).",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="transcript.json",
        help="Filename for the transcript output within the episode directory "
        "(default: transcript.json).",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size to load (default: small).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override for Whisper, e.g. cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional language hint passed to Whisper.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing transcript file instead of skipping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    channels_dir = args.channels_dir.expanduser()
    channel_dir = channels_dir / args.channel_name
    episode_dir = channel_dir / args.episode_id
    input_path = episode_dir / args.audio_filename
    output_path = episode_dir / args.output_filename

    if not input_path.exists():
        print(f"❌ Audio file does not exist: {input_path}")
        sys.exit(1)

    print("=" * 60)
    print("SINGLE EPISODE TRANSCRIPTION")
    print("=" * 60)
    print(f"📺 Channel:          {args.channel_name}")
    print(f"🎬 Episode:          {args.episode_id}")
    print(f"📁 Audio path:       {input_path.resolve()}")
    print(f"📁 Output path:      {output_path.resolve()}")
    print(f"🎚  Model size:      {args.model_size}")
    print()

    try:
        model = whisper.load_model(args.model_size, device=args.device)
    except Exception as exc:
        print(f"❌ Failed to load Whisper model '{args.model_size}': {exc}")
        sys.exit(1)

    outcome = transcribe_file(
        model=model,
        input_path=input_path,
        output_path=output_path,
        overwrite=args.overwrite,
        language=args.language,
    )

    print("=" * 60)
    if outcome:
        print("✅ Completed transcription.")
        print(f"📁 Transcript stored in: {output_path.resolve()}")
    else:
        print("❌ Transcription failed.")


if __name__ == "__main__":
    main()
