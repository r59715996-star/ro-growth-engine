#!/usr/bin/env python3
"""
youtube_longform_extractor.py - Analyze long-form YouTube videos (>30 minutes)

Pulls recent uploads from a channel, filters videos longer than a configurable
threshold (default 30 minutes), and reports:
- Views per qualifying video
- Engagement (likes + comments) per qualifying video
- Standout videos ranked by engagement rate

Usage:
    python youtube_longform_extractor.py --channel-names "Channel A" "Channel B" "Channel C"
    python youtube_longform_extractor.py --channel-name "Channel Name"
    python youtube_longform_extractor.py --channel-id UCxxxxxxxx --max-videos 150
    python youtube_longform_extractor.py --channel-name "Channel Name" --min-duration-minutes 45

Environment:
    export YOUTUBE_API_KEY="your-key-here"

Output:
    results/{channel_name}.json
    results/{channel_name}.csv
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    print("ERROR: Missing google-api-python-client. Install with:")
    print("  pip install google-api-python-client")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("WARNING: pandas not installed. CSV export will be disabled.")
    print("  Install with: pip install pandas")
    pd = None


# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Helpers
# ============================================================================

def parse_iso_duration(duration_str: str) -> int:
    """Convert ISO8601 duration (PT#H#M#S) into total seconds."""

    pattern = re.compile(
        r"PT"
        r"(?:(?P<hours>\d+)H)?"
        r"(?:(?P<minutes>\d+)M)?"
        r"(?:(?P<seconds>\d+)S)?"
    )
    match = pattern.fullmatch(duration_str)
    if not match:
        return 0

    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return hours * 3600 + minutes * 60 + seconds


def slugify_channel(identifier: str) -> str:
    """Produce a filesystem-friendly name from a channel identifier."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", identifier.strip())
    safe = safe.strip("_").lower()
    return safe or "channel"


# ============================================================================
# YouTube Client
# ============================================================================

class YouTubeLongformExtractor:
    """Extract long-form video metrics using YouTube Data API v3."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=api_key)
        print("✓ YouTube API client initialized")

    def find_channel_id(self, channel_name: str) -> Optional[str]:
        """Resolve a channel name to its ID."""
        try:
            print(f"\n🔍 Searching for channel: '{channel_name}'")
            request = self.youtube.search().list(
                part="snippet",
                q=channel_name,
                type="channel",
                maxResults=5,
            )
            response = request.execute()

            items = response.get("items", [])
            if not items:
                print(f"  ❌ No channels found for '{channel_name}'")
                return None

            print(f"\n  Found {len(items)} channel candidates:")
            for idx, item in enumerate(items, 1):
                cid = item["id"]["channelId"]
                title = item["snippet"]["title"]
                description = item["snippet"]["description"][:120]
                print(f"\n  {idx}. {title}")
                print(f"     ID: {cid}")
                print(f"     Description: {description}...")

            selected = items[0]
            selected_id = selected["id"]["channelId"]
            print(f"\n  ✓ Selected: {selected['snippet']['title']} ({selected_id})")
            return selected_id
        except HttpError as exc:
            print(f"  ❌ API error: {exc}")
            return None

    def get_channel_uploads_playlist(self, channel_id: str) -> Optional[str]:
        """Return the uploads playlist ID for a channel."""
        try:
            request = self.youtube.channels().list(
                part="contentDetails",
                id=channel_id,
            )
            response = request.execute()
            items = response.get("items", [])
            if not items:
                return None
            return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        except HttpError as exc:
            print(f"  ❌ Error getting uploads playlist: {exc}")
            return None

    def get_recent_videos(self, playlist_id: str, max_results: int) -> List[Dict]:
        """Fetch recent uploads from the playlist."""
        videos: List[Dict] = []
        next_page_token = None

        try:
            while len(videos) < max_results:
                request = self.youtube.playlistItems().list(
                    part="snippet,contentDetails",
                    playlistId=playlist_id,
                    maxResults=min(50, max_results - len(videos)),
                    pageToken=next_page_token,
                )
                response = request.execute()

                for item in response.get("items", []):
                    videos.append(
                        {
                            "video_id": item["contentDetails"]["videoId"],
                            "title": item["snippet"]["title"],
                            "published_at": item["snippet"]["publishedAt"],
                        }
                    )

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
            return videos
        except HttpError as exc:
            print(f"  ❌ Error fetching playlist items: {exc}")
            return videos

    def get_video_details(self, video_ids: List[str]) -> List[Dict]:
        """Retrieve statistics for a list of video IDs."""
        details: List[Dict] = []
        batch_size = 50

        try:
            for idx in range(0, len(video_ids), batch_size):
                batch = video_ids[idx : idx + batch_size]
                request = self.youtube.videos().list(
                    part="snippet,contentDetails,statistics",
                    id=",".join(batch),
                )
                response = request.execute()
                details.extend(response.get("items", []))
                time.sleep(0.1)  # Be polite with the API
            return details
        except HttpError as exc:
            print(f"  ❌ Error fetching video details: {exc}")
            return details

    def extract_longform_data(
        self,
        channel_identifier: str,
        *,
        is_channel_id: bool,
        max_videos: int,
        min_duration_seconds: int,
    ) -> List[Dict]:
        """Collect long-form video metrics for a channel."""
        print("\n" + "=" * 60)
        print("YOUTUBE LONG-FORM ANALYSIS")
        print("=" * 60)

        # Resolve channel
        if is_channel_id:
            channel_id = channel_identifier
            print(f"\nUsing channel ID: {channel_id}")
        else:
            channel_id = self.find_channel_id(channel_identifier)
            if not channel_id:
                print("\n❌ Could not find channel")
                return []

        # Playlist
        print("\n📋 Fetching uploads playlist...")
        uploads_playlist = self.get_channel_uploads_playlist(channel_id)
        if not uploads_playlist:
            print("  ❌ Could not find uploads playlist")
            return []
        print(f"  ✓ Uploads playlist: {uploads_playlist}")

        # Videos
        print(f"\n📹 Fetching up to {max_videos} recent uploads...")
        recent_videos = self.get_recent_videos(uploads_playlist, max_results=max_videos)
        print(f"  ✓ Retrieved {len(recent_videos)} uploads")

        if not recent_videos:
            return []

        # Details
        video_ids = [v["video_id"] for v in recent_videos]
        details = self.get_video_details(video_ids)

        # Filter and parse
        qualifying: List[Dict] = []
        for detail in details:
            duration_seconds = parse_iso_duration(detail["contentDetails"]["duration"])
            if duration_seconds < min_duration_seconds:
                continue

            data = self.parse_longform_video(detail, duration_seconds)
            if data["duration_seconds"] <= 0:
                continue
            qualifying.append(data)

        print(f"\n  ✓ {len(qualifying)} videos meet the duration threshold")
        qualifying.sort(key=lambda entry: entry["published_at"], reverse=True)
        return qualifying

    @staticmethod
    def parse_longform_video(video_detail: Dict, duration_seconds: int) -> Dict:
        """Convert API payload into a structured metrics record."""
        stats = video_detail.get("statistics", {})
        snippet = video_detail.get("snippet", {})
        video_id = video_detail["id"]

        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0))
        comments = int(stats.get("commentCount", 0))

        engagement_rate = ((likes + comments) / views * 100) if views else 0
        published_dt = datetime.fromisoformat(
            snippet.get("publishedAt", "1970-01-01T00:00:00Z").replace("Z", "+00:00")
        )
        days_live = max((datetime.now(published_dt.tzinfo) - published_dt).days, 1)

        return {
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "title": snippet.get("title", "Untitled"),
            "description": snippet.get("description", "")[:300],
            "published_at": snippet.get("publishedAt"),
            "duration_seconds": duration_seconds,
            "duration_minutes": round(duration_seconds / 60, 2),
            "views": views,
            "likes": likes,
            "comments": comments,
            "engagement_rate": round(engagement_rate, 2),
            "views_per_day": round(views / days_live, 2),
            "likes_per_day": round(likes / days_live, 2),
        }


# ============================================================================
# Export and Reporting
# ============================================================================

def export_to_json(data: List[Dict], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON saved to: {output_path}")


def export_to_csv(data: List[Dict], output_path: Path):
    if pd is None:
        print("  ⚠️  Skipping CSV export (pandas not installed)")
        return
    pd.DataFrame(data).to_csv(output_path, index=False)
    print(f"  ✓ CSV saved to: {output_path}")


def print_summary(data: List[Dict]):
    if not data:
        print("\n❌ No qualifying long-form videos found.")
        return

    print("\n" + "=" * 60)
    print("LONG-FORM PERFORMANCE SNAPSHOT")
    print("=" * 60)

    total_views = sum(item["views"] for item in data)
    total_likes = sum(item["likes"] for item in data)
    total_comments = sum(item["comments"] for item in data)

    avg_views = total_views / len(data)
    avg_engagement = sum(item["engagement_rate"] for item in data) / len(data)

    print(f"\n  Videos analyzed (> threshold): {len(data)}")
    print(f"  Total views:                  {total_views:,}")
    print(f"  Total likes:                  {total_likes:,}")
    print(f"  Total comments:               {total_comments:,}")
    print(f"  Avg views/video (>30m):       {avg_views:,.0f}")
    print(f"  Avg engagement rate:          {avg_engagement:.2f}%")

    # Standout videos by engagement rate
    standout = sorted(data, key=lambda item: item["engagement_rate"], reverse=True)[:5]
    print(f"\n  Standout videos by engagement:")
    for idx, video in enumerate(standout, 1):
        print(f"\n    {idx}. {video['title'][:70]}...")
        print(
            f"       Views: {video['views']:,} | Engagement: {video['engagement_rate']:.2f}%"
        )
        print(f"       Likes: {video['likes']:,} | Comments: {video['comments']:,}")
        print(f"       Duration: {video['duration_minutes']:.1f} minutes")
        print(f"       URL: {video['url']}")


def analyze_channel(
    channel_identifier: str,
    *,
    is_channel_id: bool,
    api_key: str,
    max_videos: int,
    min_duration_seconds: int,
    output_dir: Path,
) -> Optional[Path]:
    """Run extraction + export for a single channel, returning the JSON path."""
    extractor = YouTubeLongformExtractor(api_key)
    longform_data = extractor.extract_longform_data(
        channel_identifier,
        is_channel_id=is_channel_id,
        max_videos=max_videos,
        min_duration_seconds=min_duration_seconds,
    )

    if not longform_data:
        print(f"\n❌ No long-form videos found for '{channel_identifier}'.")
        return None

    slug = slugify_channel(channel_identifier)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{slug}.json"
    csv_path = output_dir / f"{slug}.csv"

    print("\n" + "─" * 60)
    print(f"EXPORTING DATA FOR {channel_identifier}")
    print("─" * 60)
    export_to_json(longform_data, json_path)
    export_to_csv(longform_data, csv_path)

    print_summary(longform_data)
    print("\n" + "=" * 60)
    print(f"✅ COMPLETED {channel_identifier}")
    print("=" * 60)
    return json_path


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract metrics for long-form YouTube videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --channel-names "Channel One" "Channel Two" "Channel Three"
  %(prog)s --channel-name "Millennium Masters"
  %(prog)s --channel-id UCxxxxxxxxxxxx --max-videos 200
  %(prog)s --channel-name "Millennium Masters" --min-duration-minutes 45

Environment Variables:
  YOUTUBE_API_KEY - Your YouTube Data API v3 key
        """,
    )

    parser.add_argument(
        "--channel-name",
        action="append",
        dest="channel_name_inputs",
        help="Channel name to search for (can be provided multiple times)",
    )
    parser.add_argument(
        "--channel-names",
        nargs="+",
        dest="channel_names_batch",
        help="Space-separated list of channel names to process together",
    )
    parser.add_argument(
        "--channel-id",
        action="append",
        dest="channel_ids",
        help="YouTube channel ID(s); can be provided multiple times",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("YOUTUBE_API_KEY"),
        help="YouTube Data API v3 key (or set YOUTUBE_API_KEY env var)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=200,
        help="How many recent uploads to inspect (default: 200)",
    )
    parser.add_argument(
        "--min-duration-minutes",
        type=int,
        default=30,
        help="Minimum duration (in minutes) for a video to qualify",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR),
        help="Directory to store channel-level JSON/CSV exports (default: results)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="How many channels to analyze in parallel (default: 3)",
    )

    args = parser.parse_args()

    channel_names: List[str] = []
    if args.channel_name_inputs:
        channel_names.extend(args.channel_name_inputs)
    if args.channel_names_batch:
        channel_names.extend(args.channel_names_batch)
    channel_ids: List[str] = args.channel_ids or []

    if not channel_names and not channel_ids:
        parser.error("Must provide at least one --channel-name/--channel-names or --channel-id value")

    if not args.api_key:
        print("❌ ERROR: YouTube API key not provided.")
        print("  Set export YOUTUBE_API_KEY='your-key' or pass --api-key KEY")
        sys.exit(1)

    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")

    min_duration_seconds = args.min_duration_minutes * 60
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[Tuple[str, bool]] = []
    jobs.extend((name, False) for name in channel_names)
    jobs.extend((cid, True) for cid in channel_ids)

    max_workers = min(args.max_workers, len(jobs))
    print(
        f"\nQueued {len(jobs)} channel(s). "
        f"Processing with up to {max_workers} parallel worker(s)..."
    )

    successes = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                analyze_channel,
                identifier,
                is_channel_id=is_channel_id,
                api_key=args.api_key,
                max_videos=args.max_videos,
                min_duration_seconds=min_duration_seconds,
                output_dir=output_dir,
            ): (identifier, is_channel_id)
            for identifier, is_channel_id in jobs
        }

        for future in as_completed(future_map):
            identifier, _ = future_map[future]
            try:
                result_path = future.result()
                if result_path:
                    successes += 1
                    print(f"\n📁 Results for '{identifier}' saved to {result_path}")
                else:
                    failures += 1
            except Exception as exc:  # pylint: disable=broad-except
                failures += 1
                print(f"\n❌ Error processing '{identifier}': {exc}")

    if successes == 0:
        print("\n❌ No channels completed successfully.")
        sys.exit(1)

    print(
        f"\n✅ Long-form analysis complete for {successes} channel(s). "
        f"Results saved under: {output_dir}"
    )
    if failures:
        print(f"  ⚠️  {failures} channel(s) failed or had no qualifying videos.")
    print("  Review exported JSON (and CSV if available) per channel.")


if __name__ == "__main__":
    main()
