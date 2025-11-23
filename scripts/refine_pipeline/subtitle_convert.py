#!/usr/bin/env python3
r"""
subtitle_convert.py - Convert WhisperX word-level transcripts into ASS karaoke subtitles.

Reads a verbose Whisper-style JSON transcript containing a top-level "words" array
with start/end timestamps and emits an Advanced SubStation Alpha (.ass) subtitle file
that uses karaoke timing (\\k tags). Words are grouped sequentially with a maximum of
six words per subtitle line, aligned bottom-center for 1080x1920 canvases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, List, Sequence


DEFAULT_STYLE_NAME = "Karaoke"
DEFAULT_FONT = "Arial"
DEFAULT_FONT_SIZE = 25
DEFAULT_PRIMARY = "&H00FFFFFF"  # white text
DEFAULT_SECONDARY = "&H0000FFFF"  # yellow highlight
DEFAULT_OUTLINE = "&H00000000"  # black outline
DEFAULT_BACK = "&H64000000"  # translucent shadow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert WhisperX word-level transcript JSONs into ASS subtitles for refined clips.",
    )
    parser.add_argument("channel_name", type=str, help="Channel folder under data/channels.")
    parser.add_argument("episode_id", type=str, help="Episode folder inside the channel.")
    parser.add_argument(
        "--channels-dir",
        type=Path,
        default=Path("data/channels"),
        help="Root path for all channel data (default: data/channels).",
    )
    parser.add_argument(
        "--max-words-per-line",
        type=int,
        default=6,
        help="Maximum number of words per subtitle line (default: 6).",
    )
    parser.add_argument(
        "--transcripts-subdir",
        type=str,
        default="refined_clips/transcripts",
        help="Relative path under the episode where transcripts are stored.",
    )
    parser.add_argument(
        "--subtitles-subdir",
        type=str,
        default="refined_clips/subtitles",
        help="Relative path under the episode where ASS files will be written.",
    )
    parser.add_argument(
        "--transcript-suffix",
        type=str,
        default="_whisper_v3.json",
        help="Suffix identifying transcript files (default: _whisper_v3.json).",
    )
    parser.add_argument(
        "--subtitle-suffix",
        type=str,
        default="_subtitles.ass",
        help="Suffix appended to the clip base name for the output ASS file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate ASS files even if they already exist.",
    )
    return parser.parse_args()


def load_words(json_path: Path) -> List[dict[str, Any]]:
    """Load the transcript and return the word entries."""
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    words = payload.get("words")
    if not isinstance(words, list) or not words:
        raise ValueError(f"No 'words' array found in {json_path}")
    return words


def format_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp (H:MM:SS.cc)."""
    centiseconds = int(round(seconds * 100))
    total_seconds, cs = divmod(centiseconds, 100)
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{cs:02d}"


def chunk_words(words: Sequence[dict[str, Any]], max_words: int) -> Iterable[Sequence[dict[str, Any]]]:
    """Yield consecutive groups of up to max_words entries."""
    if max_words <= 0:
        raise ValueError("max_words must be positive.")
    chunk: List[dict[str, Any]] = []
    for word in words:
        chunk.append(word)
        if len(chunk) >= max_words:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def format_karaoke_line(word_group: Sequence[dict[str, Any]]) -> tuple[str, str, str]:
    """Return (start_time, end_time, dialogue_text) for a group of words."""
    start = float(word_group[0]["start"])
    end = float(word_group[-1]["end"])
    start_ts = format_time(start)
    end_ts = format_time(end)

    fragments: List[str] = []
    for index, entry in enumerate(word_group):
        word = str(entry.get("word", "")).strip()
        if not word:
            continue
        duration = max(1, int(round((float(entry["end"]) - float(entry["start"])) * 100)))
        fragments.append(rf"{{\k{duration}}}{word}")
        if index != len(word_group) - 1:
            fragments.append(" ")

    dialogue_text = "".join(fragments)
    return start_ts, end_ts, dialogue_text


def build_ass_header(style_name: str = DEFAULT_STYLE_NAME) -> str:
    """Return the ASS header with script info, styles, and events header."""
    header_lines = [
        "[Script Info]",
        "Title: WhisperX Karaoke",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour,"
        " Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline,"
        " Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        (
            f"Style: {style_name},{DEFAULT_FONT},{DEFAULT_FONT_SIZE},"
            f"{DEFAULT_PRIMARY},{DEFAULT_SECONDARY},{DEFAULT_OUTLINE},{DEFAULT_BACK},"
            "0,0,0,0,100,100,0,0,1,3,0,2,20,20,60,1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    return "\n".join(header_lines)


def convert_to_ass(words: List[dict[str, Any]], max_words: int) -> List[str]:
    """Create ASS dialogue lines for all word groups."""
    events: List[str] = []
    for word_group in chunk_words(words, max_words):
        if not word_group:
            continue
        start_ts, end_ts, dialogue_text = format_karaoke_line(word_group)
        events.append(
            f"Dialogue: 0,{start_ts},{end_ts},{DEFAULT_STYLE_NAME},,0,0,0,,{dialogue_text}"
        )
    return events


def find_transcripts(directory: Path, suffix: str) -> List[Path]:
    """Return all transcript paths matching the suffix."""
    suffix = suffix if suffix.startswith("_") else suffix
    return sorted(path for path in directory.glob(f"*{suffix}") if path.is_file())


def generate_ass_for_clip(
    transcript_path: Path,
    output_path: Path,
    max_words: int,
) -> bool:
    """Convert a single transcript JSON into an ASS file."""
    try:
        words = load_words(transcript_path)
    except (OSError, ValueError) as exc:
        print(f"   ❌ Failed to load {transcript_path.name}: {exc}")
        return False

    events = convert_to_ass(words, max_words)
    header = build_ass_header()
    ass_content = header + "\n" + "\n".join(events) + "\n"

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write(ass_content)
    except OSError as exc:
        print(f"   ❌ Failed to write {output_path.name}: {exc}")
        return False

    print(f"   ✅ Wrote subtitles: {output_path.name}")
    return True


def main() -> None:
    args = parse_args()

    base_dir = args.channels_dir.expanduser() / args.channel_name / args.episode_id
    transcripts_dir = base_dir / args.transcripts_subdir
    subtitles_dir = base_dir / args.subtitles_subdir

    if not transcripts_dir.exists():
        print(f"❌ Transcript directory not found: {transcripts_dir}")
        sys.exit(1)

    transcript_paths = find_transcripts(transcripts_dir, args.transcript_suffix)
    if not transcript_paths:
        print(f"❌ No transcripts ending with '{args.transcript_suffix}' in {transcripts_dir}")
        sys.exit(1)

    print("=" * 60)
    print("CONVERTING WORD TRANSCRIPTS TO ASS")
    print("=" * 60)
    print(f"📺 Channel:   {args.channel_name}")
    print(f"🎬 Episode:   {args.episode_id}")
    print(f"📂 Source:    {transcripts_dir}")
    print(f"📁 Output:    {subtitles_dir}")
    print(f"🔤 Max words: {args.max_words_per_line} per line")
    print()

    converted = 0
    for transcript_path in transcript_paths:
        stem = transcript_path.name
        if stem.endswith(args.transcript_suffix):
            clip_base = stem[: -len(args.transcript_suffix)]
        else:
            clip_base = transcript_path.stem

        output_name = f"{clip_base}{args.subtitle_suffix}"
        output_path = subtitles_dir / output_name

        print(f"→ {transcript_path.name}")
        if output_path.exists() and not args.overwrite:
            print(f"   ⏭️  Skipping (exists): {output_name}")
            continue

        if generate_ass_for_clip(
            transcript_path=transcript_path,
            output_path=output_path,
            max_words=args.max_words_per_line,
        ):
            converted += 1

    print("=" * 60)
    print(f"Completed conversion: {converted}/{len(transcript_paths)} ASS files generated.")


if __name__ == "__main__":
    main()
