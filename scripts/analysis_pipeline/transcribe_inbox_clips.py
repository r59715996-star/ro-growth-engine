#!/usr/bin/env python3
"""
Transcribe inbox clips (niche tagging layout) with Groq Whisper v3.

Given `channel` and `niche`, this script:
1. Locates every clip inside data/tagging/<niche>/<channel>/inbox (or one via --clip-id).
2. Uses ffmpeg to extract mono PCM audio at 16 kHz into a temporary WAV file.
3. Sends the audio to Groq's Whisper-large-v3 model with word-level timestamps.
4. Stores the verbose JSON response under transcripts/<clip>_tr.json.

Requirements:
    pip install groq
    ffmpeg (installed and available on PATH)
    export GROQ_API_KEY="your-key-here"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, List

try:
    from groq import Groq  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - dependency missing at runtime
    Groq = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe niche inbox clips with Groq Whisper large v3.",
    )
    parser.add_argument("channel", type=str, help="Channel folder under data/tagging/{niche}.")
    parser.add_argument(
        "--niche",
        type=str,
        default="entrepreneurship",
        help="Niche folder under data/tagging (default: entrepreneurship).",
    )
    parser.add_argument(
        "--channels-dir",
        type=Path,
        default=Path("data/tagging"),
        help="Root path for all niche data (default: data/tagging).",
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        default=None,
        help="Optional clip identifier (e.g. clip01). When omitted all clips with the extension are processed.",
    )
    parser.add_argument(
        "--clip-extension",
        type=str,
        default=".mp4",
        help="Clip file extension appended when clip_id has no suffix (default: .mp4).",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=str,
        default="ffmpeg",
        help="Path or name of the ffmpeg binary (default: ffmpeg on PATH).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate used during extraction (default: 16000 Hz).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="whisper-large-v3",
        help="Groq Whisper model identifier (default: whisper-large-v3).",
    )
    parser.add_argument(
        "--response-format",
        type=str,
        default="verbose_json",
        help="Groq Whisper response format (default: verbose_json).",
    )
    parser.add_argument(
        "--timestamp-granularity",
        type=str,
        default="word",
        help="Timestamp granularity to request (default: word).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Whisper (default: 0).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional Groq API key (overrides GROQ_API_KEY env var).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing transcript JSON instead of skipping.",
    )
    return parser.parse_args()


def ensure_clip_path(clips_dir: Path, clip_id: str, extension: str) -> Path:
    """Return the resolved clip path for a clip id."""
    candidate = clips_dir / clip_id
    if candidate.suffix.lower() == extension.lower():
        return candidate
    return clips_dir / f"{clip_id}{extension}"


def list_clip_paths(clips_dir: Path, clip_id: str | None, extension: str) -> List[Path]:
    """Return a sorted list of clip paths based on optional clip ID."""
    extension = extension if extension.startswith(".") else f".{extension}"
    extension_lower = extension.lower()

    if clip_id:
        clip_path = ensure_clip_path(clips_dir, clip_id, extension_lower)
        if not clip_path.exists():
            raise FileNotFoundError(f"Clip not found: {clip_path}")
        return [clip_path]

    clips = [
        path
        for path in clips_dir.glob("*")
        if path.is_file() and path.suffix.lower() == extension_lower
    ]
    clips.sort()
    return clips


def extract_audio(
    ffmpeg_bin: str,
    clip_path: Path,
    audio_path: Path,
    sample_rate: int,
) -> None:
    """Use ffmpeg to convert clip video into mono WAV audio."""
    cmd: List[str] = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(clip_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(audio_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - runtime environment issue
        raise RuntimeError(f"ffmpeg binary not found: {ffmpeg_bin}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed while extracting audio: {stderr}") from exc
    else:
        if result.stderr:
            print(result.stderr.strip())


def transcribe_audio(
    client: Any,
    audio_path: Path,
    model_name: str,
    response_format: str,
    temperature: float,
    timestamp_granularity: str,
) -> dict:
    """Send the WAV audio to Groq Whisper and return the parsed JSON."""
    with audio_path.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=model_name,
            file=audio_file,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=[timestamp_granularity],
        )

    if hasattr(transcription, "model_dump"):
        return transcription.model_dump()
    if hasattr(transcription, "to_dict"):  # pragma: no cover - fallback for older SDKs
        return transcription.to_dict()
    return json.loads(transcription)  # type: ignore[arg-type]


def main() -> None:
    args = parse_args()
    if Groq is None:
        print("❌ ERROR: Missing groq package. Install with `pip install groq`.", file=sys.stderr)
        sys.exit(1)

    base_dir = args.channels_dir.expanduser() / args.niche / args.channel
    inbox_dir = base_dir / "inbox"
    transcripts_dir = base_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    if not inbox_dir.exists():
        print(f"❌ Inbox directory not found: {inbox_dir}")
        sys.exit(1)

    try:
        clip_paths = list_clip_paths(inbox_dir, args.clip_id, args.clip_extension)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    if not clip_paths:
        print(f"❌ No clips ending with {args.clip_extension} found in {inbox_dir}")
        sys.exit(1)

    print("=" * 60)
    print("INBOX CLIP TRANSCRIPTION (Groq Whisper v3)")
    print("=" * 60)
    print(f"📺 Channel:     {args.channel}")
    print(f"🏷️  Niche:       {args.niche}")
    print(f"✂️  Clips:       {len(clip_paths)} target(s)")
    print(f"📁 Inbox dir:   {inbox_dir.resolve()}")
    print(f"📁 Output dir:  {transcripts_dir.resolve()}")
    print(f"🧠 Model:       {args.model_name}")
    print()

    try:
        client = Groq(api_key=args.api_key)
    except Exception as exc:
        print(f"❌ Failed to initialize Groq client: {exc}")
        sys.exit(1)

    successes = 0
    for clip_path in clip_paths:
        clip_stem = clip_path.stem
        output_path = transcripts_dir / f"{clip_stem}_tr.json"

        print("─" * 60)
        print(f"✂️  Processing clip: {clip_path.name}")
        print(f"   Source:   {clip_path.resolve()}")
        print(f"   Output:   {output_path.resolve()}")

        if output_path.exists() and not args.overwrite:
            print("   ⏭️  Transcript already exists, skipping.")
            continue

        with tempfile.TemporaryDirectory(prefix=f"{clip_stem}_audio_") as tmp_dir:
            tmp_audio_path = Path(tmp_dir) / f"{clip_stem}.wav"
            print("   🎧 Extracting audio with ffmpeg...")
            try:
                extract_audio(
                    ffmpeg_bin=args.ffmpeg_bin,
                    clip_path=clip_path,
                    audio_path=tmp_audio_path,
                    sample_rate=args.sample_rate,
                )
            except RuntimeError as exc:
                print(f"   ❌ {exc}")
                continue

            print("   🤖 Sending audio to Groq Whisper...")
            try:
                transcript_payload = transcribe_audio(
                    client=client,
                    audio_path=tmp_audio_path,
                    model_name=args.model_name,
                    response_format=args.response_format,
                    temperature=args.temperature,
                    timestamp_granularity=args.timestamp_granularity,
                )
            except Exception as exc:
                print(f"   ❌ Groq transcription failed: {exc}")
                continue

        try:
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(transcript_payload, handle, indent=2, ensure_ascii=False)
        except OSError as exc:
            print(f"   ❌ Failed to write transcript: {exc}")
            continue

        num_words = (
            len(transcript_payload.get("words", []))
            if isinstance(transcript_payload, dict)
            else "?"
        )
        print("   ✅ Clip transcription complete.")
        print(f"      🔤 Word-level entries: {num_words}")
        successes += 1

    print("=" * 60)
    print(f"Completed. {successes}/{len(clip_paths)} clip(s) transcribed.")


if __name__ == "__main__":
    main()
