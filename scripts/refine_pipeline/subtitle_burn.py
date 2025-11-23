#!/usr/bin/env python3
"""
subtitle_burn.py - Burn ASS subtitles into MP4 videos using ffmpeg.

Given a channel and episode, this script looks for refined clip MP4s and their
corresponding ASS subtitles (generated via subtitle_convert.py) and burns the
subtitles into MP4 outputs using ffmpeg's subtitles filter.

Requirements:
    ffmpeg installed and available on PATH
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Burn ASS subtitles into refined clips for a given channel/episode.",
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
        "--clips-subdir",
        type=str,
        default="refined_clips",
        help="Relative directory under the episode that contains MP4 clips.",
    )
    parser.add_argument(
        "--subtitles-subdir",
        type=str,
        default="refined_clips/subtitles",
        help="Relative directory under the episode that contains ASS subtitles.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="refined_clips/subtitled",
        help="Relative directory where burned videos will be stored.",
    )
    parser.add_argument(
        "--subtitle-suffix",
        type=str,
        default="_subtitles.ass",
        help="Suffix appended to clip base names for the ASS files.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=str,
        default="ffmpeg",
        help="Path or name of the ffmpeg binary (default: ffmpeg on PATH).",
    )
    parser.add_argument(
        "--video-codec",
        type=str,
        default="libx264",
        help="Codec used for video re-encoding (default: libx264).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="Constant Rate Factor for video quality (default: 18).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        help="libx264 preset for ffmpeg (default: medium).",
    )
    parser.add_argument(
        "--audio-codec",
        type=str,
        default="copy",
        help="Audio codec setting (default: copy to avoid re-encoding).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing subtitled videos instead of skipping.",
    )
    return parser.parse_args()


def run_ffmpeg(cmd: Sequence[str]) -> None:
    """Execute the ffmpeg command and raise informative errors on failure."""
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg binary not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed with exit code {exc.returncode}") from exc


def build_command(
    ffmpeg_bin: str,
    input_video: Path,
    input_ass: Path,
    output_video: Path,
    video_codec: str,
    crf: int,
    preset: str,
    audio_codec: str,
) -> list[str]:
    """Construct the ffmpeg command for burning subtitles."""
    return [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_video),
        "-vf",
        f"subtitles={str(input_ass)}",
        "-c:v",
        video_codec,
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-c:a",
        audio_codec,
        str(output_video),
    ]


def main() -> None:
    args = parse_args()

    base_dir = args.channels_dir.expanduser() / args.channel_name / args.episode_id
    clips_dir = base_dir / args.clips_subdir
    subtitles_dir = base_dir / args.subtitles_subdir
    output_dir = base_dir / args.output_subdir

    if not clips_dir.exists():
        print(f"❌ Clips directory not found: {clips_dir}")
        sys.exit(1)
    if not subtitles_dir.exists():
        print(f"❌ Subtitles directory not found: {subtitles_dir}")
        sys.exit(1)

    mp4_files = sorted(p for p in clips_dir.glob("*.mp4") if p.is_file())
    if not mp4_files:
        print(f"❌ No MP4 files found in {clips_dir}")
        sys.exit(1)

    subtitle_map: Dict[str, Path] = {}
    for subtitle in subtitles_dir.glob("*.ass"):
        subtitle_map[subtitle.name] = subtitle

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BURNING SUBTITLES INTO REFINED CLIPS")
    print("=" * 60)
    print(f"📺 Channel:  {args.channel_name}")
    print(f"🎬 Episode:  {args.episode_id}")
    print(f"🎞  Clips:    {clips_dir}")
    print(f"📝 Subs:     {subtitles_dir}")
    print(f"📁 Output:   {output_dir}")
    print()

    processed = 0
    for video_path in mp4_files:
        clip_base = video_path.stem
        subtitle_name = f"{clip_base}{args.subtitle_suffix}"
        subtitle_path = subtitle_map.get(subtitle_name)

        print(f"→ {video_path.name}")
        if subtitle_path is None or not subtitle_path.exists():
            print(f"   ⚠️  Missing subtitles: {subtitle_name}")
            continue

        output_path = output_dir / video_path.name
        if output_path.exists() and not args.overwrite:
            print("   ⏭️  Output exists, skipping.")
            continue

        cmd = build_command(
            ffmpeg_bin=args.ffmpeg_bin,
            input_video=video_path,
            input_ass=subtitle_path,
            output_video=output_path,
            video_codec=args.video_codec,
            crf=args.crf,
            preset=args.preset,
            audio_codec=args.audio_codec,
        )

        try:
            run_ffmpeg(cmd)
        except RuntimeError as exc:
            print(f"   ❌ Failed to burn subtitles: {exc}")
            continue

        print("   ✅ Subtitles burned.")
        processed += 1

    print("=" * 60)
    print(f"Completed: {processed}/{len(mp4_files)} clip(s) subtitled.")


if __name__ == "__main__":
    main()
