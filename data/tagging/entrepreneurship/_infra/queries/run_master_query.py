#!/usr/bin/env python3
"""Execute master_query.sql against the niche database and export results to CSV."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


HERE = Path(__file__).resolve().parent
DB_PATH = HERE.parent.parent / "videos.db"
QUERY_FILE = HERE / "master_query.sql"
DEFAULT_OUTPUT = HERE / "master_query_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run master_query.sql against the niche database and export to CSV."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help=f"Path to videos.db (default: {DB_PATH}).",
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        default=QUERY_FILE,
        help=f"Path to SQL file (default: {QUERY_FILE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.db_path.exists():
        raise SystemExit(f"Database not found: {args.db_path}")
    if not args.query_file.exists():
        raise SystemExit(f"SQL file not found: {args.query_file}")

    sql = args.query_file.read_text(encoding="utf-8")

    conn = sqlite3.connect(args.db_path)
    cursor = conn.execute(sql)
    rows = cursor.fetchall()
    headers = [col[0] for col in cursor.description] if cursor.description else []
    conn.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if headers:
            writer.writerow(headers)
        writer.writerows(rows)

    print(f"✅ Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
