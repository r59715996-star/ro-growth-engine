#!/usr/bin/env python3
"""
query_clips.py - Query and verify database contents
"""

import sqlite3
from pathlib import Path
from typing import List, Dict

CHANNEL_NAME = "Odds On Open"
DB_PATH = Path("data/channels") / CHANNEL_NAME / "tagging" / "clips.db"


def dict_factory(cursor, row):
    """Convert sqlite3 rows to dictionaries."""
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}


def run_query(conn: sqlite3.Connection, query: str, params: tuple = ()) -> List[Dict]:
    """Execute query and return results as list of dicts."""
    cursor = conn.execute(query, params)
    return cursor.fetchall()


def main():
    """Run sample queries to verify data."""
    
    if not DB_PATH.exists():
        print(f"❌ Database not found: {DB_PATH}")
        return
    
    print("="*60)
    print("DATABASE QUERY TOOL")
    print("="*60)
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = dict_factory
    
    # 1. Count total clips
    print("\n1. TOTAL CLIPS")
    print("-"*60)
    result = run_query(conn, "SELECT COUNT(*) as count FROM clips")
    print(f"Total clips in database: {result[0]['count']}")
    
    # 2. Clips by episode
    print("\n2. CLIPS BY EPISODE")
    print("-"*60)
    results = run_query(conn, """
        SELECT episode_id, COUNT(*) as count
        FROM clips
        GROUP BY episode_id
        ORDER BY count DESC
    """)
    for row in results:
        print(f"  {row['episode_id']}: {row['count']} clips")
    
    # 3. Sample joined data
    print("\n3. SAMPLE CLIPS (Full Feature Set)")
    print("-"*60)
    results = run_query(conn, """
        SELECT 
            c.clip_id,
            c.episode_id,
            m.quant_version,
            m.qual_version,
            qn.duration_s,
            qn.wpm,
            qn.hook_wpm,
            qn.reading_level,
            ql.hook_type,
            ql.hook_emotion,
            ql.topic_primary,
            ql.has_payoff
        FROM clips c
        JOIN clip_meta m ON c.clip_id = m.clip_id
        JOIN clip_quant qn ON c.clip_id = qn.clip_id
        JOIN clip_qual ql ON c.clip_id = ql.clip_id
        LIMIT 5
    """)
    for row in results:
        print(f"\n  {row['clip_id']}:")
        print(f"    Episode: {row['episode_id']}")
        print(f"    Versions: quant={row['quant_version']}, qual={row['qual_version']}")
        print(f"    Duration: {row['duration_s']:.1f}s, WPM: {row['wpm']:.0f}, Hook WPM: {row['hook_wpm']:.0f}")
        print(f"    Reading Level: {row['reading_level']:.1f}")
        print(f"    Hook: {row['hook_type']} ({row['hook_emotion']})")
        print(f"    Topic: {row['topic_primary']}, Has Payoff: {bool(row['has_payoff'])}")
    
    # 4. Hook type distribution
    print("\n4. HOOK TYPE DISTRIBUTION")
    print("-"*60)
    results = run_query(conn, """
        SELECT hook_type, COUNT(*) as count
        FROM clip_qual
        GROUP BY hook_type
        ORDER BY count DESC
    """)
    for row in results:
        print(f"  {row['hook_type']}: {row['count']}")
    
    # 5. Topic distribution
    print("\n5. TOPIC DISTRIBUTION")
    print("-"*60)
    results = run_query(conn, """
        SELECT topic_primary, COUNT(*) as count
        FROM clip_qual
        GROUP BY topic_primary
        ORDER BY count DESC
    """)
    for row in results:
        print(f"  {row['topic_primary']}: {row['count']}")
    
    # 6. clip_meta contents
    print("\n6. CLIP META (SELECT *)")
    print("-"*60)
    results = run_query(conn, "SELECT * FROM clip_meta")
    if not results:
        print("  (no rows)")
    else:
        for row in results:
            print(f"  {row}")
    
    # 7. Payoff analysis
    print("\n7. PAYOFF ANALYSIS")
    print("-"*60)
    results = run_query(conn, """
        SELECT 
            CASE has_payoff WHEN 1 THEN 'Has Payoff' ELSE 'No Payoff' END as payoff_status,
            COUNT(*) as count,
            AVG(qn.reading_level) as avg_reading_level,
            AVG(qn.duration_s) as avg_duration
        FROM clip_qual ql
        JOIN clip_quant qn ON ql.clip_id = qn.clip_id
        GROUP BY has_payoff
    """)
    for row in results:
        print(f"  {row['payoff_status']}:")
        print(f"    Count: {row['count']}")
        print(f"    Avg Reading Level: {row['avg_reading_level']:.2f}")
        print(f"    Avg Duration: {row['avg_duration']:.1f}s")
    
    # 8. Hook speed analysis
    print("\n8. HOOK SPEED ANALYSIS")
    print("-"*60)
    results = run_query(conn, """
        SELECT 
            clip_id,
            hook_wpm,
            wpm,
            ROUND((hook_wpm - wpm) / wpm * 100, 1) as speed_diff_pct
        FROM clip_quant
        ORDER BY speed_diff_pct DESC
        LIMIT 5
    """)
    print("  Top 5 clips with fastest hooks relative to overall pace:")
    for row in results:
        print(f"    {row['clip_id']}: Hook {row['hook_wpm']:.0f} vs Overall {row['wpm']:.0f} ({row['speed_diff_pct']:+.1f}%)")
    
    conn.close()
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
