#!/usr/bin/env python3
"""Run the full Shorts analysis pipeline end-to-end with validations."""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def log_header(text: str) -> None:
    print("\n" + "─" * 60)
    print(text)
    print("─" * 60)


def run_command(cmd: List[str], description: str) -> bool:
    """Run a subprocess command with logging."""
    log_header(description)
    print("Command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print("✓ Success")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"❌ Step failed: {exc}")
        return False


def validate_file(path: Path, description: str) -> bool:
    if not path.exists():
        print(f"❌ Validation failed: {description} missing at {path}")
        return False
    return True


def validate_glob(paths: Iterable[Path], description: str) -> bool:
    items = list(paths)
    if not items:
        print(f"❌ Validation failed: {description}")
        return False
    return True


def validate_db_rows(db_path: Path, channel_id: str) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM videos WHERE channel_id = ?", (channel_id,)
        )
        count = cursor.fetchone()[0]
        conn.close()
        if count <= 0:
            print(
                f"❌ Validation failed: No rows found in videos for channel_id={channel_id}"
            )
            return False
        print(f"✓ Validation: {count} video(s) found for channel_id={channel_id}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Validation failed querying DB: {exc}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate the full Shorts analysis pipeline."
    )
    parser.add_argument("--niche", required=True, help="Niche folder name.")
    parser.add_argument("--channel-name", required=True, help="Channel folder name.")
    parser.add_argument("--channel-id", required=True, help="YouTube channel ID.")
    parser.add_argument(
        "--prompt-path",
        required=True,
        type=Path,
        help="Path to qualitative tagging prompt file.",
    )
    parser.add_argument(
        "--api-key",
        help="YouTube API key for the extractor (optional if set in env).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    niche = args.niche
    channel = args.channel_name
    channel_id = args.channel_id

    base_dir = Path("data/tagging") / niche / channel
    performance_json = base_dir / "performance" / "shorts_performance.json"
    performance_enhanced = base_dir / "performance" / "shorts_performance_enhanced.json"
    inbox_dir = base_dir / "inbox"
    clip_tags_dir = base_dir / "clip_tags"
    db_dir = Path("data/tagging") / niche
    db_path = db_dir / "videos.db"
    infra_dir = db_dir / "_infra"

    steps = [
        {
            "description": "Step 1/6: Extracting performance data",
            "command": [
                sys.executable,
                str(SCRIPT_DIR / "youtube_performance_extractor.py"),
                "--channel-id",
                channel_id,
                "--niche",
                niche,
                "--channel-output",
                channel,
            ]
            + (["--api-key", args.api_key] if args.api_key else []),
            "validate": lambda: validate_file(
                performance_json, "shorts_performance.json"
            ),
        },
        {
            "description": "Step 2/6: Enhancing performance metrics",
            "command": [
                sys.executable,
                str(SCRIPT_DIR / "enhance_shorts_performance.py"),
                "--niche",
                niche,
                "--channel-name",
                channel,
            ],
            "validate": lambda: validate_file(
                performance_enhanced, "shorts_performance_enhanced.json"
            ),
        },
        {
            "description": "Step 3/6: Downloading audio",
            "command": [
                sys.executable,
                str(SCRIPT_DIR / "download_shorts_audio.py"),
                "--niche",
                niche,
                "--channel_name",
                channel,
            ],
            "validate": lambda: validate_glob(
                inbox_dir.glob("*.mp3"), "MP3 files in inbox/"
            ),
        },
        {
            "description": "Step 4/6: Transcribe + Quant + Qual analysis",
            "command": [
                sys.executable,
                str(SCRIPT_DIR / "pipeline_mp3.py"),
                "--channel",
                channel,
                "--niche",
                niche,
                "--prompt-path",
                str(args.prompt_path),
                "--quant-output-dir",
                str(clip_tags_dir),
            ],
            "validate": lambda: validate_glob(
                clip_tags_dir.glob("*_ta.json"), "Quantitative tag files in clip_tags/"
            ),
        },
        {
            "description": "Step 5/6: Ensuring database exists",
            "command": [
                sys.executable,
                str(infra_dir / "create_db.py"),
                "--niche",
                niche,
                "--channel-name",
                channel,
            ],
            "validate": lambda: validate_file(db_path, "videos.db"),
        },
        {
            "description": "Step 6/6: Inserting clips into database",
            "command": [
                sys.executable,
                str(infra_dir / "insert_clips.py"),
                "--channel-name",
                channel,
                "--channel-id",
                channel_id,
            ],
            "validate": lambda: validate_db_rows(db_path, channel_id),
        },
    ]

    for step in steps:
        if not run_command(step["command"], step["description"]):
            sys.exit(1)
        if not step["validate"]():
            sys.exit(1)

    log_header("✅ Analysis complete")
    print(f"Channel: {channel} ({channel_id})")
    print(f"Database: {db_path}")


if __name__ == "__main__":
    main()
