#!/usr/bin/env python3
"""
youtube_performance_extractor.py - Fetch recent Shorts stats into JSON.

Pulls the latest Shorts from a channel and records:
- video_id
- view_count
- like_count
- comment_count
- days_since_publish

Output: data/tagging/{niche}/{channel_name}/performance/shorts_performance.json

Usage:
    python youtube_performance_extractor.py --channel-name "Millennial Masters" --api-key YOUR_KEY
    python youtube_performance_extractor.py --channel-id UCxxxxxxxxxxxx --api-key YOUR_KEY
    python youtube_performance_extractor.py --channel-id UCxxxxxxxxxxxx --niche entrepreneurship --channel-output millennial_masters

Environment:
    export YOUTUBE_API_KEY="your-key-here"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_NICHE = "entrepreneurship"
# Hard-coded channel name (slug) for output directory resolution (overridable via CLI).
DEFAULT_CHANNEL_OUTPUT = "millennial_masters"

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError as exc:  # pragma: no cover - dependency missing at runtime
    print("❌ google-api-python-client is required. Install with:")
    print("   pip install google-api-python-client")
    raise SystemExit(1) from exc


DEFAULT_MAX_SHORTS = 50
OUTPUT_FILENAME = "shorts_performance.json"


def build_youtube_client(api_key: str):
    """Instantiate a YouTube Data API client."""
    return build("youtube", "v3", developerKey=api_key)


def find_channel_id(youtube, channel_name: str) -> Optional[str]:
    """Find a channel ID by name (first search hit)."""
    try:
        resp = (
            youtube.search()
            .list(part="snippet", q=channel_name, type="channel", maxResults=5)
            .execute()
        )
    except HttpError as exc:
        print(f"❌ Channel search failed: {exc}")
        return None

    items = resp.get("items") or []
    if not items:
        print(f"❌ No channels found for '{channel_name}'")
        return None

    best = items[0]
    channel_id = best["id"]["channelId"]
    title = best["snippet"]["title"]
    print(f"✓ Selected channel: {title} ({channel_id})")
    return channel_id


def get_uploads_playlist_id(youtube, channel_id: str) -> Optional[str]:
    """Return the uploads playlist for a channel."""
    try:
        resp = (
            youtube.channels()
            .list(part="contentDetails", id=channel_id)
            .execute()
        )
    except HttpError as exc:
        print(f"❌ Failed to fetch uploads playlist: {exc}")
        return None

    items = resp.get("items") or []
    if not items:
        print("❌ No channel items returned")
        return None
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def list_recent_videos(youtube, playlist_id: str, max_results: int = 100) -> List[Dict]:
    """List recent uploads (basic info) from a playlist."""
    videos: List[Dict] = []
    next_page_token: Optional[str] = None

    try:
        while len(videos) < max_results:
            resp = (
                youtube.playlistItems()
                .list(
                    part="snippet,contentDetails",
                    playlistId=playlist_id,
                    maxResults=min(50, max_results - len(videos)),
                    pageToken=next_page_token,
                )
                .execute()
            )
            for item in resp.get("items", []):
                content_details = item.get("contentDetails", {}) or {}
                snippet = item.get("snippet", {}) or {}
                published_at = (
                    content_details.get("videoPublishedAt")
                    or snippet.get("publishedAt")
                    or ""
                )
                videos.append(
                    {
                        "video_id": content_details.get("videoId"),
                        "published_at": published_at,
                        "title": snippet.get("title", ""),
                    }
                )
            next_page_token = resp.get("nextPageToken")
            if not next_page_token:
                break
    except HttpError as exc:
        print(f"❌ Error listing playlist items: {exc}")
    return videos


def fetch_video_details(youtube, video_ids: List[str]) -> List[Dict]:
    """Fetch detailed metadata for a list of video IDs."""
    details: List[Dict] = []
    batch_size = 50
    try:
        for start in range(0, len(video_ids), batch_size):
            batch = video_ids[start : start + batch_size]
            resp = (
                youtube.videos()
                .list(
                    part="snippet,contentDetails,statistics,topicDetails",
                    id=",".join(batch),
                )
                .execute()
            )
            for item in resp.get("items", []):
                snippet = item.get("snippet", {}) or {}
                item["channel_name"] = snippet.get("channelTitle", "")
                details.append(item)
            time.sleep(0.1)  # polite spacing
    except HttpError as exc:
        print(f"❌ Error fetching video details: {exc}")
    return details


def is_short(duration_iso8601: str) -> bool:
    """Determine if a video duration corresponds to a Short (<= 180s)."""
    import re
    
    # Reject anything with hours
    if "H" in (duration_iso8601 or ""):
        return False
    
    match = re.match(r"^PT(?:(\d+)M)?(?:(\d+)S)?$", duration_iso8601 or "")
    if not match:
        return False
    minutes = int(match.group(1) or 0)
    seconds = int(match.group(2) or 0)
    return (minutes * 60 + seconds) <= 180


def compute_days_since(published_at: str) -> int:
    """Return integer days since published timestamp."""
    try:
        published_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return 0
    now = datetime.now(published_dt.tzinfo)
    return (now - published_dt).days


def extract_shorts_performance(
    youtube,
    channel_identifier: str,
    *,
    is_channel_id: bool = False,
    max_shorts: int = DEFAULT_MAX_SHORTS,
) -> tuple[List[Dict], Optional[str], Optional[str]]:
    """Collect recent Shorts performance stats for a channel."""
    if is_channel_id:
        channel_id = channel_identifier
    else:
        channel_id = find_channel_id(youtube, channel_identifier)
        if not channel_id:
            return [], None, None

    uploads_playlist = get_uploads_playlist_id(youtube, channel_id)
    if not uploads_playlist:
        return [], channel_id, None

    print("📹 Fetching recent uploads...")
    recent_videos = list_recent_videos(youtube, uploads_playlist, max_results=500)
    print(f"  Found {len(recent_videos)} recent uploads")

    if not recent_videos:
        return [], channel_id, None

    print("🔍 Fetching details and filtering for Shorts...")
    details = fetch_video_details(youtube, [v["video_id"] for v in recent_videos])
    shorts: List[Dict] = []
    channel_name: Optional[str] = None
    for detail in details:
        duration = detail.get("contentDetails", {}).get("duration", "")
        if not is_short(duration):
            continue

        stats = detail.get("statistics", {})
        snippet = detail.get("snippet", {})
        topic_details = detail.get("topicDetails", {}) or {}
        published_at = snippet.get("publishedAt", "")
        channel_title = detail.get("channel_name") or snippet.get("channelTitle", "")
        if channel_title:
            channel_name = channel_title

        shorts.append(
            {
                "video_id": detail.get("id"),
                "view_count": int(stats.get("viewCount", 0) or 0),
                "like_count": int(stats.get("likeCount", 0) or 0),
                "comment_count": int(stats.get("commentCount", 0) or 0),
                "days_since_publish": compute_days_since(published_at),
                "published_at": published_at,
                "title": snippet.get("title", ""),
                "channel_name": channel_title,
                "topic_categories_raw": topic_details.get("topicCategories", []),
            }
        )
        if len(shorts) >= max_shorts:
            break

    shorts.sort(key=lambda item: item.get("published_at", ""), reverse=True)
    return shorts[:max_shorts], channel_id, channel_name


def update_manifest(
    niche: str, channel_id: str, channel_name: str, videos: List[Dict]
) -> Path:
    """Create or update a niche-level manifest with channel metadata."""
    manifest_dir = Path(f"data/tagging/{niche}")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"

    existing: List[Dict] = []
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle) or []
        except (json.JSONDecodeError, OSError):
            existing = []

    manifest_map = {item.get("channel_id"): item for item in existing if item.get("channel_id")}

    new_videos = [{"video_id": v.get("video_id")} for v in videos if v.get("video_id")]

    if channel_id in manifest_map:
        entry = manifest_map[channel_id]
        existing_videos = entry.get("videos") or []
        seen: set[str] = set()
        merged_videos = []
        for vid in existing_videos + new_videos:
            vid_id = vid.get("video_id")
            if not vid_id or vid_id in seen:
                continue
            seen.add(vid_id)
            merged_videos.append({"video_id": vid_id})
        entry["videos"] = merged_videos
        if channel_name:
            entry["channel_name"] = channel_name
    else:
        seen_new: set[str] = set()
        deduped_new = []
        for vid in new_videos:
            vid_id = vid.get("video_id")
            if not vid_id or vid_id in seen_new:
                continue
            seen_new.add(vid_id)
            deduped_new.append({"video_id": vid_id})
        existing.append(
            {
                "channel_id": channel_id,
                "channel_name": channel_name,
                "videos": deduped_new,
            }
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, ensure_ascii=False)

    return manifest_path


def write_output(data: List[Dict], output_dir: Path) -> Path:
    """Persist extracted data to the channel performance directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Extract recent YouTube Shorts performance stats into JSON."
    )
    parser.add_argument(
        "--channel-name",
        help="Channel name to search for (required unless --channel-id is provided).",
    )
    parser.add_argument(
        "--channel-id",
        help="YouTube channel ID to query directly (skips search).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("YOUTUBE_API_KEY"),
        help="YouTube Data API v3 key (or set YOUTUBE_API_KEY).",
    )
    parser.add_argument(
        "--max-shorts",
        type=int,
        default=DEFAULT_MAX_SHORTS,
        help=f"Maximum number of shorts to fetch (default: {DEFAULT_MAX_SHORTS}).",
    )
    parser.add_argument(
        "--niche",
        help=f"Niche output directory (default: {DEFAULT_NICHE}).",
    )
    parser.add_argument(
        "--channel-output",
        "--channel_output",
        dest="channel_output",
        help=f"Channel output directory name/slug (default: {DEFAULT_CHANNEL_OUTPUT}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    niche = args.niche or DEFAULT_NICHE
    channel_output = args.channel_output or DEFAULT_CHANNEL_OUTPUT

    if not args.channel_name and not args.channel_id:
        print("❌ Provide --channel-name or --channel-id.")
        raise SystemExit(1)
    if not args.api_key:
        print("❌ Missing YouTube API key. Use --api-key or set YOUTUBE_API_KEY.")
        raise SystemExit(1)

    try:
        youtube = build_youtube_client(args.api_key)
    except Exception as exc:  # pragma: no cover - runtime failure
        print(f"❌ Failed to initialize YouTube client: {exc}")
        raise SystemExit(1) from exc

    channel_identifier = args.channel_id or args.channel_name
    output_dir = Path(f"data/tagging/{niche}/{channel_output}/performance")
    shorts, channel_id, channel_name = extract_shorts_performance(
        youtube,
        channel_identifier=channel_identifier,
        is_channel_id=bool(args.channel_id),
        max_shorts=args.max_shorts,
    )

    if not shorts:
        print("⚠️  No shorts data extracted.")
        raise SystemExit(1)
    if not channel_id:
        print("⚠️  Channel ID missing; skipping manifest update.")
    manifest_path: Optional[Path] = None

    # Drop helper fields before writing
    output_data = [
        {
            "video_id": item["video_id"],
            "view_count": item["view_count"],
            "like_count": item["like_count"],
            "comment_count": item["comment_count"],
            "days_since_publish": item["days_since_publish"],
            "published_at": item.get("published_at", ""),
            "topic_categories_raw": item.get("topic_categories_raw", []),
        }
        for item in shorts
    ]
    output_path = write_output(output_data, output_dir)

    if channel_id:
        manifest_path = update_manifest(
            niche=niche,
            channel_id=channel_id,
            channel_name=channel_name or "",
            videos=shorts,
        )

    print(f"✅ Wrote {len(output_data)} shorts to {output_path}")
    if manifest_path:
        print(f"✅ Updated manifest at {manifest_path}")


if __name__ == "__main__":
    main()
