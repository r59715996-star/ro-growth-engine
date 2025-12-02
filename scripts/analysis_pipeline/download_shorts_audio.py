#!/usr/bin/env python3
"""
download_shorts_audio.py - Download Shorts audio as MP3 via yt-dlp.

Reads:
  data/tagging/{NICHE}/{CHANNEL_NAME}/performance/shorts_performance.json

For each entry with a video_id:
  - Downloads from YouTube using yt-dlp
  - Extracts audio as MP3
  - Saves into data/tagging/{NICHE}/{CHANNEL_NAME}/inbox with the original title as filename

Requirements:
  pip install yt-dlp
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# Set target niche + channel folder
NICHE_DEFAULT = "entrepreneurship"
CHANNEL_DEFAULT = "millennial_masters"


def load_shorts(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Performance file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")
    return data


def build_download_command(video_id: str, output_dir: Path) -> List[str]:
    """
    Build yt-dlp command to download audio as MP3 with title-based filename.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = str(output_dir / f"{video_id}.%(ext)s")
    return [
        "yt-dlp",
        "-x",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "-o",
        output_template,
        url,
    ]


def download_audio(entries: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        video_id = entry.get("video_id")
        if not video_id:
            print("⚠️  Skipping entry without video_id")
            continue
        cmd = build_download_command(str(video_id), output_dir)
        print(f"→ Downloading {video_id} to {output_dir}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"❌ Failed to download {video_id}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Shorts audio as MP3 into channel inbox."
    )
    parser.add_argument(
        "--niche",
        default=NICHE_DEFAULT,
        help=f"Niche folder (default: {NICHE_DEFAULT}).",
    )
    parser.add_argument(
        "--channel-name",
        "--channel_name",
        dest="channel_name",
        default=CHANNEL_DEFAULT,
        help=f"Channel folder name/slug (default: {CHANNEL_DEFAULT}).",
    )
    args = parser.parse_args()

    niche_root = Path("data/tagging") / args.niche / args.channel_name
    performance_file = niche_root / "performance" / "shorts_performance.json"
    inbox_dir = niche_root / "inbox"

    try:
        entries = load_shorts(performance_file)
    except Exception as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    if not entries:
        print("⚠️  No entries found to download.")
        sys.exit(0)

    download_audio(entries, inbox_dir)
    print("✅ Downloads complete.")


if __name__ == "__main__":
    main()
