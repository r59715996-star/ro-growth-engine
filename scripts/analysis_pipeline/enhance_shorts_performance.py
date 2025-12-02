#!/usr/bin/env python3
"""
enhance_shorts_performance.py - Add engagement metrics to Shorts performance JSON.

Reads:
  data/tagging/{NICHE}/{CHANNEL_NAME}/performance/shorts_performance.json

Computes per-video:
  engagement_rate = (like_count + (comment_count * 3)) / view_count
  (0 if view_count is 0 or missing)

Writes:
  data/tagging/{NICHE}/{CHANNEL_NAME}/performance/shorts_performance_enhanced.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Set this to target a specific channel folder.
NICHE = "entrepreneurship"
CHANNEL_NAME = "millennial_masters"
NICHE_ROOT = Path("data/tagging") / NICHE / CHANNEL_NAME


def load_performance_json(path: Path) -> List[Dict[str, Any]]:
    """Load the base performance JSON array."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")
    return data


def compute_engagement_rate(entry: Dict[str, Any]) -> float:
    """Compute engagement rate safely."""
    likes = entry.get("like_count") or 0
    comments = entry.get("comment_count") or 0
    views = entry.get("view_count") or 0
    try:
        likes_val = float(likes)
        comments_val = float(comments)
        views_val = float(views)
    except (TypeError, ValueError):
        return 0.0
    if views_val <= 0:
        return 0.0
    return (likes_val + (comments_val * 3.0)) / views_val


def compute_time_features(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Annotate entries with publish day-of-week and hour."""
    updated: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        published_at = entry.get("published_at")
        day_number = None
        hour_number = None
        if isinstance(published_at, str):
            try:
                dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                day_number = dt.weekday()
                hour_number = dt.hour
            except ValueError:
                pass
        new_entry = dict(entry)
        new_entry["day_published_number"] = day_number
        new_entry["hour_published_number"] = hour_number
        updated.append(new_entry)
    return updated


def add_topic_slugs(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add normalized topic category slugs derived from topic_categories_raw URLs."""
    for entry in entries:
        urls = entry.get("topic_categories_raw") or []
        slugs = []
        for url in urls:
            slug = url.rsplit("/", 1)[-1].lower()
            slugs.append(slug)
        entry["topic_categories_slugs"] = slugs
    return entries


def enhance_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add engagement_rate to each entry."""
    entries = compute_time_features(entries)
    enhanced: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        new_entry = dict(entry)
        new_entry["engagement_rate"] = compute_engagement_rate(entry)
        enhanced.append(new_entry)
    enhanced = add_topic_slugs(enhanced)
    return enhanced


def write_output(path: Path, data: List[Dict[str, Any]]) -> None:
    """Write enhanced JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhance Shorts performance JSON with additional metrics."
    )
    parser.add_argument(
        "--niche",
        default=NICHE,
        help=f"Niche folder (default: {NICHE}).",
    )
    parser.add_argument(
        "--channel-name",
        "--channel_name",
        dest="channel_name",
        default=CHANNEL_NAME,
        help=f"Channel folder name/slug (default: {CHANNEL_NAME}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path("data/tagging") / args.niche / args.channel_name / "performance"
    input_path = base_dir / "shorts_performance.json"
    output_path = base_dir / "shorts_performance_enhanced.json"

    entries = load_performance_json(input_path)
    enhanced = enhance_entries(entries)
    write_output(output_path, enhanced)

    print(f"✅ Enhanced {len(enhanced)} entries")
    print(f"→ {output_path}")


if __name__ == "__main__":
    main()
