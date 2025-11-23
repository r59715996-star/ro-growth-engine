#!/usr/bin/env python3
"""
insert_clips.py - Insert clip data from JSON files into SQLite
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
TAGGING_DIR = SCRIPT_DIR.parent  # tagging root (handles channel names with spaces)
DB_PATH = TAGGING_DIR / "clips.db"
MANIFEST_PATH = TAGGING_DIR / "inbox/manifest.json"

def load_manifest() -> Dict[str, str]:
    """Load clip_id -> episode_id mapping from manifest."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    
    with open(MANIFEST_PATH, 'r') as f:
        return json.load(f)

def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def insert_clip(conn: sqlite3.Connection, clip_id: str, episode_id: str, filename: str):
    """Insert into clips table."""
    conn.execute(
        """
        INSERT OR REPLACE INTO clips (clip_id, episode_id, filename)
        VALUES (?, ?, ?)
        """,
        (clip_id, episode_id, filename)
    )

def insert_meta(conn: sqlite3.Connection, clip_id: str, quant_ver: str = 'v1', qual_ver: str = 'v1'):
    """Insert into clip_meta table."""
    conn.execute(
        """
        INSERT OR REPLACE INTO clip_meta (clip_id, quant_version, qual_version, tagged_at)
        VALUES (?, ?, ?, ?)
        """,
        (clip_id, quant_ver, qual_ver, datetime.now().isoformat())
    )

def insert_quant(conn: sqlite3.Connection, clip_id: str, data: Dict[str, Any]):
    """Insert into clip_quant table."""
    conn.execute(
        """
        INSERT OR REPLACE INTO clip_quant (
            clip_id, duration_s, word_count, wpm, hook_word_count, hook_wpm,
            num_sentences, question_start, reading_level, filler_count,
            filler_density, first_person_ratio, second_person_ratio
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            clip_id,
            data['duration_s'],
            data['word_count'],
            data['wpm'],
            data['hook_word_count'],
            data['hook_wpm'],
            data['num_sentences'],
            1 if data['question_start'] else 0,
            data['reading_level'],
            data['filler_count'],
            data['filler_density'],
            data['first_person_ratio'],
            data['second_person_ratio']
        )
    )

def insert_qual(conn: sqlite3.Connection, clip_id: str, data: Dict[str, Any]):
    """Insert into clip_qual table."""
    conn.execute(
        """
        INSERT OR REPLACE INTO clip_qual (
            clip_id, hook_type, hook_emotion, topic_primary,
            has_examples, has_payoff, has_numbers, insider_language
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            clip_id,
            data['hook_type'],
            data['hook_emotion'],
            data['topic_primary'],
            1 if data['has_examples'] else 0,
            1 if data['has_payoff'] else 0,
            1 if data['has_numbers'] else 0,
            1 if data['insider_language'] else 0
        )
    )

def main():
    """Main insertion pipeline."""
    
    print("="*60)
    print("CLIP DATA INSERTION")
    print("="*60)
    
    if not DB_PATH.exists():
        print(f"\n❌ Database not found: {DB_PATH}")
        print("   Run: python db/create_db.py")
        return
    
    # Load manifest
    print(f"\nLoading manifest: {MANIFEST_PATH}")
    try:
        manifest = load_manifest()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    print(f"  Found {len(manifest)} clips in manifest")
    
    conn = sqlite3.connect(DB_PATH)
    inserted = 0
    skipped = 0
    errors = []
    
    print(f"\nProcessing clips...")
    print("-"*60)
    
    for clip_id, episode_id in manifest.items():
        
        
        quant_file = TAGGING_DIR / f"clip_tags/{clip_id}_ta.json"
        qual_file = TAGGING_DIR / f"clip_tags/{clip_id}_qa.json"
        clip_file = TAGGING_DIR / f"inbox/{clip_id}.mp4"
        
        if not quant_file.exists():
            print(f"  ⚠️  Skipping {clip_id} (missing {quant_file.name})")
            skipped += 1
            continue
        
        if not qual_file.exists():
            print(f"  ⚠️  Skipping {clip_id} (missing {qual_file.name})")
            skipped += 1
            continue
        
        try:
            quant_data = load_json(quant_file)
            qual_data = load_json(qual_file)
            
            insert_clip(conn, clip_id, episode_id, clip_file.name if clip_file.exists() else f"{clip_id}.mp4")
            insert_meta(conn, clip_id)
            insert_quant(conn, clip_id, quant_data)
            insert_qual(conn, clip_id, qual_data)
            
            print(f"  ✓ {clip_id} → episode {episode_id}")
            inserted += 1
            
        except KeyError as e:
            error_msg = f"Missing field {e} in {clip_id}"
            print(f"  ❌ {error_msg}")
            errors.append(error_msg)
            skipped += 1
            
        except Exception as e:
            error_msg = f"Failed {clip_id}: {e}"
            print(f"  ❌ {error_msg}")
            errors.append(error_msg)
            skipped += 1
    
    conn.commit()
    conn.close()
    
    print(f"\n{'='*60}")
    print(f"✅ Successfully inserted: {inserted} clips")
    if skipped:
        print(f"⚠️  Skipped: {skipped} clips")
    if errors:
        print(f"\n❌ Errors:")
        for err in errors[:5]:
            print(f"   - {err}")
        if len(errors) > 5:
            print(f"   ... and {len(errors)-5} more")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
