#!/usr/bin/env python3
"""
generate_batch.py - Convert resolved clips to FFmpeg batch format

Reads VIDEO_INPUT from environment variable set by orchestrate.py
"""

import json
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: Missing pyyaml. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

CLIP_PLAN_RESOLVED = Path(os.environ.get("CLIP_RESOLVED_PATH", "plans/clip_plan_resolved.yaml"))
OUTPUT_BATCH = Path(os.environ.get("BATCH_PATH", "data/batch.json"))

# Get video input from environment (set by orchestrate.py)
VIDEO_INPUT = os.environ.get("VIDEO_INPUT", "assets/wellness_pod.webm")

# ============================================================================
# Utilities
# ============================================================================

def seconds_to_hms(sec: float) -> str:
    """Convert seconds to HH:MM:SS string for FFmpeg."""
    s = int(round(sec))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    print("="*60)
    print("BATCH GENERATOR (SECONDS → HH:MM:SS)")
    print("="*60)
    
    print("\nConfiguration:")
    print(f"  Input Plan:    {CLIP_PLAN_RESOLVED}")
    print(f"  Video Input:   {VIDEO_INPUT}")
    print(f"  Output Batch:  {OUTPUT_BATCH}")
    print()
    
    # Validate inputs
    if not Path(CLIP_PLAN_RESOLVED).exists():
        print(f"ERROR: {CLIP_PLAN_RESOLVED} not found")
        print("Run resolve_times_v2.py first to generate this file")
        sys.exit(1)
    
    if not Path(VIDEO_INPUT).exists():
        print(f"ERROR: Video file not found: {VIDEO_INPUT}")
        sys.exit(1)
    
    # Load resolved clips
    print(f"Loading {CLIP_PLAN_RESOLVED}...")
    try:
        with open(CLIP_PLAN_RESOLVED, "r", encoding="utf-8") as f:
            clips = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR loading clip plan: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not isinstance(clips, list):
        print("ERROR: clip_plan_resolved.yaml must be a YAML list", file=sys.stderr)
        sys.exit(1)
    
    print(f"  Found {len(clips)} clips")
    
    # Build batch manifest
    print("\nGenerating batch.json...")
    print("-"*60)
    
    batch_clips = []
    processed = 0
    
    for clip in clips:
        clip_id = clip.get("clip_id", "unknown")
        
        # Skip clips that failed resolution
        if "adjustment_reason" in clip and any(
            reason.startswith("skipped") or reason.endswith("error") 
            for reason in clip["adjustment_reason"]
        ):
            print(f"  ⚠️  Skipping {clip_id} (resolution failed)")
            continue
        
        # Validate required fields
        if "start_s" not in clip or "end_s" not in clip:
            print(f"  ⚠️  Skipping {clip_id} (missing start_s/end_s)")
            continue
        
        start_s = float(clip["start_s"])
        end_s = float(clip["end_s"])
        
        # Convert to HH:MM:SS
        start_hms = seconds_to_hms(start_s)
        end_hms = seconds_to_hms(end_s)
        
        # Build batch entry
        batch_clip = {
            "start": start_hms,
            "end": end_hms,
            "output": f"{clip_id}.mp4",
            "segments": []  # TODO: Add view switching logic here if needed
        }
        
        batch_clips.append(batch_clip)
        processed += 1
        
        duration = round(end_s - start_s, 1)
        print(f"  ✓ {clip_id}: {start_hms} → {end_hms} ({duration}s)")
    
    # Build final batch document
    batch_doc = {
        "input": VIDEO_INPUT,
        "clips": batch_clips
    }
    
    # Write output
    print(f"\n{'-'*60}")
    try:
        with open(OUTPUT_BATCH, "w", encoding="utf-8") as f:
            json.dump(batch_doc, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Wrote {OUTPUT_BATCH}")
    except Exception as e:
        print(f"  ❌ Failed to write {OUTPUT_BATCH}: {e}")
        sys.exit(1)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Generated batch manifest with {processed} clips")
    print(f"\nNext step: Run ./make_final_batch.sh to render clips")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
