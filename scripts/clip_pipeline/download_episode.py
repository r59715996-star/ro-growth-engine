#!/usr/bin/env python3
"""
download_episode.py - Download a YouTube episode by ID using yt-dlp.

This helper wraps yt-dlp so the clipping pipeline can fetch source videos
directly from YouTube given an episode identifier.

Requirements:
    pip install yt-dlp
    ffmpeg (for yt-dlp remux/merge operations)

Usage:
    python scripts/clip_pipeline/download_episode.py EPISODE_ID
    python scripts/clip_pipeline/download_episode.py EPISODE_ID \\
        --output-dir data/channels/<channel>/<episode> \\
        --filename source.mp4 \\
        --force
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a YouTube episode by ID using yt-dlp."
    )
    parser.add_argument(
        "episode_id",
        type=str,
        help="YouTube video ID or full URL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to place the downloaded video (default: current directory).",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="source.mp4",
        help="Filename for the downloaded video (default: source.mp4).",
    )
    parser.add_argument(
        "--url-template",
        type=str,
        default="https://www.youtube.com/watch?v={episode_id}",
        help="Template used to build the YouTube URL when only an ID is provided.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="bv*+ba/b",
        help="yt-dlp format selector (default: best video+audio fallbacks).",
    )
    parser.add_argument(
        "--merge-format",
        type=str,
        default="mp4",
        help="Container to remux/merge into (default: mp4).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if the target file already exists.",
    )
    return parser.parse_args()


def build_url(episode_id: str, template: str) -> str:
    if episode_id.startswith("http://") or episode_id.startswith("https://"):
        return episode_id
    return template.format(episode_id=episode_id)


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / args.filename

    if target_path.exists():
        if args.force:
            target_path.unlink()
        else:
            print(f"✅ Video already downloaded: {target_path}")
            return

    url = build_url(args.episode_id, args.url_template)

    cmd = [
        "yt-dlp",
        url,
        "--no-playlist",
        "--merge-output-format",
        args.merge_format,
        "-f",
        args.format,
        "-o",
        str(target_path),
    ]

    print("=" * 60)
    print("DOWNLOADING EPISODE")
    print("=" * 60)
    print(f"📺 Episode ID:  {args.episode_id}")
    print(f"🔗 URL:         {url}")
    print(f"📁 Output:      {target_path}")
    print()

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("❌ yt-dlp is not installed or not on PATH.")
        print("   Install with: pip install yt-dlp")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"❌ yt-dlp failed with exit code {exc.returncode}")
        sys.exit(exc.returncode)

    print("✅ Download complete.")


if __name__ == "__main__":
    main()
