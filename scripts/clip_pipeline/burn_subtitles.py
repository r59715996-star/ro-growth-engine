#!/usr/bin/env python3
"""Burn subtitle files into rendered clips using FFmpeg."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Burn subtitles into rendered clips using FFmpeg."
    )
    parser.add_argument("--channel-name", required=True, help="Channel directory name")
    parser.add_argument("--episode-id", required=True, help="Episode identifier")
    return parser.parse_args()


def run_ffmpeg(input_video: Path, subtitle_file: Path, output_video: Path) -> None:
    """Invoke FFmpeg to burn subtitles into the input video."""
    filter_str = (
        f"subtitles='{subtitle_file.as_posix()}':"
        "force_style='FontName=Arial,FontSize=14,Bold=1,"
        "PrimaryColour=&H00FFFFFF&,OutlineColour=&H00000000&,BorderStyle=1,"
        "Outline=4,Shadow=2,Alignment=2,MarginV=220,MarginL=80,MarginR=80,"
        "WrapStyle=2,LineSpacing=4'"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_video),
        "-vf",
        filter_str,
        "-c:a",
        "copy",
        str(output_video),
    ]
    subprocess.run(cmd, check=True)


def burn_subtitles(channel_name: str, episode_id: str) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    clips_dir = (
        repo_root / "data" / "channels" / channel_name / episode_id / "clips"
    )
    subtitles_dir = clips_dir / "subtitles"

    if not clips_dir.exists():
        print(f"[ERROR] Clips directory not found: {clips_dir}")
        return 1
    if not clips_dir.is_dir():
        print(f"[ERROR] Clips path is not a directory: {clips_dir}")
        return 1
    if not subtitles_dir.exists() or not subtitles_dir.is_dir():
        print(f"[ERROR] Subtitles directory not found: {subtitles_dir}")
        return 1

    clip_files = sorted(clips_dir.glob("*.mp4"))
    if not clip_files:
        print(f"[INFO] No clips found in {clips_dir}")
        return 0

    processed = 0
    missing_subtitles = 0
    failed = 0

    for clip_path in clip_files:
        subtitle_path = subtitles_dir / f"{clip_path.stem}.srt"
        output_path = clip_path.with_name(f"{clip_path.stem}_subtitled.mp4")

        if not subtitle_path.exists():
            print(f"[WARN] Missing subtitles for {clip_path.name}, skipping.")
            missing_subtitles += 1
            continue

        if output_path.exists():
            print(
                f"[INFO] Output already exists for {clip_path.name}, skipping "
                f"({output_path.name})."
            )
            continue

        print(f"[INFO] Burning subtitles into {clip_path.name}...")
        try:
            run_ffmpeg(clip_path, subtitle_path, output_path)
            processed += 1
            print(f"[INFO] Wrote {output_path.name}")
        except subprocess.CalledProcessError as exc:
            failed += 1
            print(f"[ERROR] FFmpeg failed for {clip_path.name}: {exc}")

    print(
        "[SUMMARY] Processed: "
        f"{processed}, Missing subtitles: {missing_subtitles}, Failed: {failed}, "
        f"Total clips: {len(clip_files)}"
    )
    return 1 if failed else 0


def main() -> int:
    args = parse_args()
    return burn_subtitles(args.channel_name, args.episode_id)


if __name__ == "__main__":
    sys.exit(main())
