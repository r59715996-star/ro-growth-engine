#!/usr/bin/env python3
"""
Upload clips to YouTube Shorts using database-backed metadata and logging.

Usage: python scripts/upload_pipeline/upload_clips.py \
  --channel "data/channels/Odds On Open" \
  --token-path credentials/test_youtube_token.pickle \
  --check-schedule
"""

import argparse
import os
import pickle
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = [
    'https://www.googleapis.com/auth/youtube.upload',
    'https://www.googleapis.com/auth/yt-analytics.readonly',
]
CLIENT_SECRETS = 'credentials/client_secrets.json'


def authenticate(token_path: str, client_secrets_path: str):
    """Authenticate with YouTube and return service."""
    credentials = None

    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            credentials = pickle.load(token)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            print("  Refreshing token...")
            credentials.refresh(Request())
        else:
            print("  Starting OAuth flow...")
            print("  Browser will open for authentication")
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_path, SCOPES)
            credentials = flow.run_local_server(port=8080)

        Path(token_path).parent.mkdir(parents=True, exist_ok=True)
        with open(token_path, 'wb') as token:
            pickle.dump(credentials, token)
        print("  ✓ Authenticated")

    return build('youtube', 'v3', credentials=credentials)


def upload_video(
    youtube,
    video_path: str,
    title: str,
    description: str,
    tags: list,
    category_id: str = "22",
    privacy: str = "public"
):
    """Upload video to YouTube."""
    body = {
        'snippet': {
            'title': title[:100],
            'description': description,
            'tags': tags[:15],
            'categoryId': category_id
        },
        'status': {
            'privacyStatus': privacy,
            'selfDeclaredMadeForKids': False
        }
    }

    media = MediaFileUpload(
        video_path,
        mimetype='video/mp4',
        resumable=True,
        chunksize=1024 * 1024
    )

    request = youtube.videos().insert(
        part='snippet,status',
        body=body,
        media_body=media
    )

    response = None
    print(f"    Uploading... ", end='', flush=True)

    while response is None:
        status, response = request.next_chunk()
        if status:
            progress = int(status.progress() * 100)
            print(f"\r    {progress}% ", end='', flush=True)

    print(f"\r    ✓ Upload complete")

    video_id = response['id']
    return video_id


def get_clips_to_upload(db_path: Path, clip_id: str | None, check_schedule: bool):
    """Return list of clips ready for upload."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if clip_id:
        query = """
            SELECT c.clip_id, sm.title
            FROM clips c
            JOIN short_meta sm ON c.clip_id = sm.clip_id
            WHERE c.clip_id = ?
              AND sm.title IS NOT NULL
              AND c.clip_id NOT IN (SELECT clip_id FROM clip_uploads)
        """
        cursor.execute(query, (clip_id,))
    elif check_schedule:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        query = """
            SELECT c.clip_id, sm.title
            FROM clips c
            INNER JOIN short_meta sm ON c.clip_id = sm.clip_id
            INNER JOIN clip_schedule cs ON c.clip_id = cs.clip_id
            WHERE sm.title IS NOT NULL
              AND cs.status = 'pending'
              AND cs.scheduled_for <= ?
              AND c.clip_id NOT IN (SELECT clip_id FROM clip_uploads)
            ORDER BY cs.scheduled_for ASC
        """
        cursor.execute(query, (now,))
    else:
        query = """
            SELECT c.clip_id, sm.title
            FROM clips c
            JOIN short_meta sm ON c.clip_id = sm.clip_id
            WHERE sm.title IS NOT NULL
              AND c.clip_id NOT IN (SELECT clip_id FROM clip_uploads)
        """
        cursor.execute(query)

    clips = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return clips


def store_upload_record(db_path: Path, clip_id: str, youtube_video_id: str):
    """Write upload info to clip_uploads and update schedule status."""
    now = datetime.utcnow()
    uploaded_at = now.isoformat()
    upload_day = now.strftime('%Y-%m-%d')
    upload_hour = now.hour

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO clip_uploads
        (clip_id, youtube_video_id, uploaded_at, upload_day, upload_hour)
        VALUES (?, ?, ?, ?, ?)
        """,
        (clip_id, youtube_video_id, uploaded_at, upload_day, upload_hour),
    )

    cursor.execute(
        """
        UPDATE clip_schedule
        SET status = 'uploaded'
        WHERE clip_id = ?
        """,
        (clip_id,),
    )

    conn.commit()
    conn.close()


def get_video_path(channel_root: str, clip_id: str) -> Path:
    """Return path to video file for the given clip."""
    channel_name = Path(channel_root).name
    return Path("data/channels") / channel_name / "tagging" / "inbox" / f"{clip_id}.mp4"


def main():
    parser = argparse.ArgumentParser(description='Upload clips to YouTube Shorts')
    parser.add_argument('--channel', required=True, help='Channel root directory (e.g., "data/channels/Odds On Open")')
    parser.add_argument('--token-path', required=True, help='Path to OAuth token pickle')
    parser.add_argument('--clip-id', help='Specific clip to upload (optional)')
    parser.add_argument('--privacy', default='public', choices=['public', 'unlisted', 'private'])
    parser.add_argument('--check-schedule', action='store_true', help='Respect scheduled_for times in clip_schedule')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be uploaded without uploading')
    args = parser.parse_args()

    print("=" * 60)
    print("YOUTUBE SHORTS UPLOADER")
    print("=" * 60)
    print()

    channel_name = Path(args.channel).name
    db_path = Path("data/channels") / channel_name / "tagging" / "clips.db"
    client_secrets_path = Path("data/channels") / channel_name / "tagging" / "credentials" / "client_secrets.json"

    if not db_path.exists():
        print(f"  ❌ Database not found at {db_path}")
        sys.exit(1)

    if not client_secrets_path.exists():
        print(f"  ❌ Client secrets not found at {client_secrets_path}")
        sys.exit(1)

    print("Authenticating with YouTube...")
    youtube = authenticate(args.token_path, str(client_secrets_path))
    print()

    clips = get_clips_to_upload(db_path, args.clip_id, args.check_schedule)

    if not clips:
        print("No clips ready for upload.")
        return

    if args.dry_run:
        print("\n[DRY RUN] Would upload:")
        for clip in clips:
            print(f"  - {clip['clip_id']}: \"{clip['title']}\"")
        return

    print(f"Uploading {len(clips)} clips...")
    print(f"Privacy: {args.privacy}")
    print("-" * 60)

    uploaded = 0
    failed = 0

    for clip in clips:
        clip_id = clip['clip_id']
        title = clip['title']

        print(f"\n{clip_id}:")
        print(f"  Title: {title}")

        video_path = get_video_path(args.channel, clip_id)
        if not video_path.exists():
            print(f"  ❌ Video not found: {video_path}")
            failed += 1
            continue

        try:
            video_id = upload_video(
                youtube,
                str(video_path),
                title,
                "",
                [],
                "22",
                args.privacy,
            )

            store_upload_record(db_path, clip_id, video_id)

            video_url = f"https://youtube.com/shorts/{video_id}"
            print(f"    Video ID: {video_id}")
            print(f"    URL: {video_url}")
            uploaded += 1
        except FileNotFoundError as e:
            print(f"  ⚠ Video file not found: {e}")
            failed += 1
        except Exception as e:
            print(f"    ❌ Upload failed: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"✅ Uploaded {uploaded} clip(s)")
    if failed:
        print(f"✗ Failed {failed} clip(s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
