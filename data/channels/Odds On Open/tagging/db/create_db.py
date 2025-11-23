#!/usr/bin/env python3
"""
create_db.py - Initialize SQLite database with schema
"""

import sqlite3
from pathlib import Path

# Paths
CHANNEL_NAME = "Odds On Open"
SCHEMA_FILE = Path(__file__).parent / "schema.sql"
DB_PATH = Path("data/channels") / CHANNEL_NAME / "tagging" / "clips.db"

def create_database():
    """Create database and apply schema."""
    
    print("="*60)
    print("CREATING DATABASE")
    print("="*60)
    
    # Ensure directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if database already exists
    if DB_PATH.exists():
        print(f"\n⚠️  Database already exists: {DB_PATH}")
        response = input("Recreate? This will DELETE all data (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        DB_PATH.unlink()
    
    # Read schema
    print(f"\nReading schema: {SCHEMA_FILE}")
    with open(SCHEMA_FILE, 'r') as f:
        schema_sql = f.read()
    
    # Create database and execute schema
    print(f"Creating database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(schema_sql)
    conn.commit()
    
    # Verify tables
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]
    
    # Verify indexes
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
    )
    indexes = [row[0] for row in cursor.fetchall()]
    
    print(f"\n{'='*60}")
    print(f"✅ Database created successfully")
    print(f"{'='*60}")
    
    print(f"\nTables ({len(tables)}):")
    for table in tables:
        print(f"  - {table}")
    
    print(f"\nIndexes ({len(indexes)}):")
    for idx in indexes:
        if not idx.startswith('sqlite_'):  # Skip auto-generated indexes
            print(f"  - {idx}")
    
    conn.close()

if __name__ == "__main__":
    create_database()
