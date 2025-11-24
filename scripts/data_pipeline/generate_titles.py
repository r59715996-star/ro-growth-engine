"""
CLI script to generate YouTube Shorts titles with Groq.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Optional

from groq_title_client import _transcript_to_text, generate_title_from_transcript


def load_system_prompt(path: Path) -> str:
    """
    Read the system prompt from disk.
    """
    if not path.exists():
        raise SystemExit(f"ERROR: System prompt file not found at {path}")
    return path.read_text(encoding="utf-8").strip()


def get_clips_needing_titles(db_path: Path) -> List[str]:
    """
    Return clip_ids that do not yet have titles.
    """
    query = """
        SELECT c.clip_id
        FROM clips c
        LEFT JOIN short_meta sm ON c.clip_id = sm.clip_id
        WHERE sm.title IS NULL OR sm.clip_id IS NULL
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(query)
            return [row[0] for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        raise SystemExit(f"ERROR: Failed to query database: {exc}") from exc


def _title_exists(db_path: Path, clip_id: str) -> bool:
    """
    Check if a title already exists for a clip.
    """
    query = "SELECT 1 FROM short_meta WHERE clip_id = ? AND title IS NOT NULL LIMIT 1"
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(query, (clip_id,))
            return cursor.fetchone() is not None
    except sqlite3.Error:
        return False


def _load_transcript(
    channel_root: Path, clip_id: str
) -> Optional[tuple[dict, str]]:
    """
    Load transcript JSON and extracted text.
    """
    transcript_path = channel_root / "tagging" / "transcripts" / f"{clip_id}_tr.json"
    if not transcript_path.exists():
        print(f"Warning: Transcript file not found for {clip_id}: {transcript_path}")
        return None

    try:
        transcript_json = json.loads(transcript_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error: Malformed transcript JSON for {clip_id}: {exc}")
        return None

    try:
        transcript_text = _transcript_to_text(transcript_json)
    except ValueError as exc:
        print(f"Error: {exc} ({clip_id})")
        return None

    return transcript_json, transcript_text


def _persist_title(
    db_path: Path, clip_id: str, title: str, contains_number: bool
) -> bool:
    """
    Insert or replace title metadata in the database.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(
                """
                INSERT OR REPLACE INTO short_meta
                (clip_id, title, contains_number, title_version, title_generated_at)
                VALUES (?, ?, ?, 'v1', CURRENT_TIMESTAMP)
                """,
                (clip_id, title, contains_number),
            )
            conn.commit()
            return True
    except sqlite3.IntegrityError as exc:
        print(
            f"Error: Insert failed for {clip_id} (clip may be missing in clips table): {exc}"
        )
    except sqlite3.Error as exc:
        print(f"Error: Database write failed for {clip_id}: {exc}")
    return False


def _print_verbose_request(system_prompt: str, transcript_text: str) -> None:
    """
    Print detailed LLM request data.
    """
    user_prompt = (
        "Transcript of a trading clip:\n\n"
        f"{transcript_text}\n\n"
        "Generate a YouTube Shorts title (30-80 characters).\n\n"
        "Output ONLY the title text, nothing else. No quotes, no explanation, just the title."
    )
    print("\nLLM Request:")
    print(f"System prompt length: {len(system_prompt)} chars")
    print(f"User prompt preview ({len(user_prompt)} chars):")
    print(user_prompt)


def _print_verbose_response(title: str, contains_number: bool) -> None:
    """
    Print LLM response details.
    """
    print("LLM Response:")
    print(title)
    print(f"Validation: length={len(title)} contains_number={contains_number}")


def generate_title_for_clip(
    clip_id: str,
    channel_root: Path,
    system_prompt: str,
    db_path: Path,
    dry_run: bool,
    verbose: bool,
) -> Optional[str]:
    """
    Generate and optionally save a title for one clip.
    """
    if _title_exists(db_path, clip_id):
        print(f"Skipping {clip_id}: title already exists.")
        return None

    loaded = _load_transcript(channel_root, clip_id)
    if loaded is None:
        return None
    transcript_json, transcript_text = loaded

    if verbose:
        _print_verbose_request(system_prompt, transcript_text)

    try:
        title = generate_title_from_transcript(transcript_json, system_prompt)
    except (ValueError, RuntimeError) as exc:
        print(f"Error: LLM generation failed for {clip_id}: {exc}")
        return None
    except Exception as exc:  # pragma: no cover - defensive against unexpected errors
        print(f"Error: Unexpected failure for {clip_id}: {exc}")
        return None

    contains_number = bool(re.search(r"\d", title))

    if verbose:
        _print_verbose_response(title, contains_number)

    if dry_run:
        print(f"[DRY RUN] {clip_id}: {title!r} ({len(title)} chars)")
        return title

    if not _persist_title(db_path, clip_id, title, contains_number):
        return None

    print(f"Saved {clip_id}: {title!r} ({len(title)} chars)")
    return title


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Generate YouTube titles for clips")
    parser.add_argument(
        "--channel", required=True, help="Channel root directory path"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate without inserting to database"
    )
    parser.add_argument(
        "--clip-id", help="Generate for single clip only (e.g., clip003)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed LLM call info"
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """
    Entry point for CLI execution.
    """
    args = parse_args(argv)
    channel_root = Path(args.channel)
    if not channel_root.exists():
        raise SystemExit(f"ERROR: Channel path not found: {channel_root}")

    db_path = channel_root / "tagging" / "clips.db"
    system_prompt_path = channel_root / "tagging" / "config" / "title_system_v1.txt"
    if not db_path.exists():
        raise SystemExit(f"ERROR: Database not found at {db_path}")

    system_prompt = load_system_prompt(system_prompt_path)
    print(f"System prompt: {system_prompt_path}")
    print(f"Database: {db_path}")

    if args.clip_id:
        clip_ids = [args.clip_id]
    else:
        clip_ids = get_clips_needing_titles(db_path)

    if not clip_ids:
        print("No clips need titles.")
        return

    total = len(clip_ids)
    print(f"Generating titles for {total} clip(s)...")
    print("-" * 50)

    successes = 0
    failures = 0
    skipped = 0

    for idx, clip_id in enumerate(clip_ids):
        if _title_exists(db_path, clip_id):
            print(f"Skipping {clip_id}: title already exists.")
            skipped += 1
        else:
            title = generate_title_for_clip(
                clip_id=clip_id,
                channel_root=channel_root,
                system_prompt=system_prompt,
                db_path=db_path,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )
            if title is None:
                failures += 1
            else:
                successes += 1

        if idx < total - 1:
            time.sleep(2)

    print("-" * 50)
    print(f"Generated {successes}/{total} titles successfully.")
    if skipped:
        print(f"Skipped {skipped} clip(s) with existing titles.")
    if failures:
        print(f"{failures} clip(s) failed. See messages above.")
    if args.dry_run:
        print("[DRY RUN] Titles were not written to the database.")


if __name__ == "__main__":
    main(sys.argv[1:])
