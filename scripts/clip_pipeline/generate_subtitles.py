#!/usr/bin/env python3
"""
generate_subtitles.py

Generate .srt subtitle files from transcript for all rendered clips.
"""

import json
import os
import yaml
from pathlib import Path
from typing import List, Dict

# ============================================================================
# Configuration
# ============================================================================

TRANSCRIPT = Path(os.environ.get("TRANSCRIPT_PATH", "data/transcripts/transcript2.json"))
CLIP_PLAN_RESOLVED = Path(os.environ.get("CLIP_RESOLVED_PATH", "plans/clip_plan_resolved.yaml"))
CLIPS_DIR = Path(os.environ.get("CLIPS_DIR", "assets/clips"))
SUBTITLES_DIR = Path(os.environ.get("SUBTITLES_DIR", "assets/clips/subtitles"))

# ============================================================================
# SRT Generation
# ============================================================================

def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(segments: List[Dict], clip_start_s: float, clip_end_s: float) -> str:
    """Generate SRT content from transcript segments."""
    srt_lines = []
    subtitle_num = 1
    clip_duration = clip_end_s - clip_start_s
    
    for seg in segments:
        abs_start = float(seg['start'])
        abs_end = float(seg['end'])
        text = seg['text'].strip()
        
        if not text:
            continue
        
        # Convert to clip-relative time
        rel_start = abs_start - clip_start_s
        rel_end = abs_end - clip_start_s
        
        # Skip if segment ends before clip starts
        if rel_end <= 0:
            continue
        
        # Skip if segment starts after clip ends
        if rel_start >= clip_duration:
            continue
        
        # Clamp start to clip beginning
        if rel_start < 0:
            rel_start = 0.0
        
        # Clamp end to clip end
        if rel_end > clip_duration:
            rel_end = clip_duration
        
        # Safety: Skip if invalid duration
        if rel_start >= rel_end:
            continue
        
        # Format as SRT
        start_time = seconds_to_srt_time(rel_start)
        end_time = seconds_to_srt_time(rel_end)
        
        srt_lines.append(str(subtitle_num))
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")
        
        subtitle_num += 1
    
    return "\n".join(srt_lines)


def extract_clip_segments(
    transcript: List[Dict],
    start_s: float,
    end_s: float
) -> List[Dict]:
    """Extract segments overlapping with clip."""
    return [
        seg for seg in transcript
        if float(seg['end']) >= start_s and float(seg['start']) <= end_s
    ]


# ============================================================================
# Data Loading
# ============================================================================

def load_transcript(path: str) -> List[Dict]:
    """Load transcript segments."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Unexpected transcript format")


def load_clip_plan(path: str) -> List[Dict]:
    """Load resolved clip plan."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*60)
    print("SUBTITLE GENERATOR")
    print("="*60)
    print()
    
    # Ensure output directory exists
    SUBTITLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading transcript: {TRANSCRIPT}")
    transcript = load_transcript(TRANSCRIPT)
    print(f"  ✓ {len(transcript)} segments")
    
    print(f"Loading clip plan: {CLIP_PLAN_RESOLVED}")
    clips = load_clip_plan(CLIP_PLAN_RESOLVED)
    print(f"  ✓ {len(clips)} clips")
    print()
    
    # Process clips
    print("Generating subtitles...")
    print(f"Output directory: {SUBTITLES_DIR}/")
    print("-"*60)
    
    generated = 0
    skipped = 0
    
    for clip in clips:
        clip_id = clip.get('clip_id')
        
        if not clip_id:
            skipped += 1
            continue
        
        # Check for precise timestamps
        if 'start_s' not in clip or 'end_s' not in clip:
            print(f"  ⏭️  {clip_id}: No timestamps")
            skipped += 1
            continue
        
        start_s = float(clip['start_s'])
        end_s = float(clip['end_s'])
        
        # Check if video exists
        clip_path = CLIPS_DIR / f"{clip_id}.mp4"
        srt_path = SUBTITLES_DIR / f"{clip_id}.srt"
        
        if not clip_path.exists():
            print(f"  ⏭️  {clip_id}: Video not rendered")
            skipped += 1
            continue
        
        if srt_path.exists():
            print(f"  ⏭️  {clip_id}: Subtitle exists")
            skipped += 1
            continue
        
        # Extract segments
        clip_segments = extract_clip_segments(transcript, start_s, end_s)
        
        if not clip_segments:
            print(f"  ❌ {clip_id}: No segments found")
            skipped += 1
            continue
        
        # Generate SRT
        srt_content = generate_srt(clip_segments, start_s, end_s)
        
        if not srt_content.strip():
            print(f"  ❌ {clip_id}: Empty content")
            skipped += 1
            continue
        
        # Write
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        duration = end_s - start_s
        print(f"  ✓ {clip_id}: {len(clip_segments)} segments, {duration:.2f}s")
        generated += 1
    
    # Summary
    print()
    print("="*60)
    print(f"✅ Generated {generated} subtitle files")
    if skipped:
        print(f"⏭️  Skipped {skipped} clips")
    print(f"📁 Subtitles: {SUBTITLES_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
