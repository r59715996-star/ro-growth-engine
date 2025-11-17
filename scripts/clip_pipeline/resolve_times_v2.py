#!/usr/bin/env python3
"""
resolve_times_v2.py - Timestamp resolver (pure seconds architecture)

Matches opening/closing sentences to transcript segments and outputs precise timestamps.
Works entirely in seconds. No HH:MM:SS conversion until final output.

Usage:
    python resolve_times_v2.py

Configure inputs via CONFIG section below.
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

try:
    import yaml
except ImportError:
    print("ERROR: Missing pyyaml. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

CLIP_PLAN = Path(os.environ.get("CLIP_SELECT_PATH", "plans/clip_select.yaml"))
TRANSCRIPT = Path(os.environ.get("TRANSCRIPT_PATH", "data/transcripts/transcript2.json"))
OUTPUT_RESOLVED = Path(os.environ.get("CLIP_RESOLVED_PATH", "plans/clip_plan_resolved.yaml"))

# Matching parameters
MATCH_WORDS = 5     # Number of words to match (first N for opening, last N for closing)
PAD_PRE = 0.0       # Seconds to pad before opening segment
PAD_POST = 0.0      # Seconds to pad after closing segment


# ============================================================================
# Text Utilities
# ============================================================================

def norm_text(s: str) -> str:
    """Normalize text: lowercase, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_first_words(text: str, n: int) -> str:
    """Get first N words from text."""
    words = text.split()
    return " ".join(words[:min(n, len(words))])


def get_last_words(text: str, n: int) -> str:
    """Get last N words from text."""
    words = text.split()
    return " ".join(words[-min(n, len(words)):])


def seconds_to_hms(sec: float) -> str:
    """Convert seconds to HH:MM:SS string (for human display only)."""
    s = int(round(sec))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Segment:
    """Transcript segment with normalized text."""
    idx: int
    seg_id: int
    start: float
    end: float
    text: str
    norm: str


# ============================================================================
# Transcript Loading
# ============================================================================

def load_segments(transcript_path: str) -> List[Segment]:
    """Load transcript segments from JSON file."""
    path = Path(transcript_path)
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle both formats: direct list or wrapped in "segments" key
    if isinstance(data, dict) and "segments" in data:
        seg_list = data["segments"]
    elif isinstance(data, list):
        seg_list = data
    else:
        raise ValueError(f"Unexpected transcript format in {transcript_path}")
    
    segments = []
    for i, s in enumerate(seg_list):
        if not all(k in s for k in ["start", "end"]):
            raise ValueError(f"Segment {i} missing required fields (start/end): {s}")
        
        text = s.get("text", "") or ""
        segments.append(
            Segment(
                idx=i,
                seg_id=int(s.get("id", i)),
                start=float(s["start"]),
                end=float(s["end"]),
                text=text,
                norm=norm_text(text),
            )
        )
    
    if not segments:
        raise ValueError(f"No segments found in {transcript_path}")
    
    return segments


# ============================================================================
# Matching Functions
# ============================================================================

def find_segment_by_first_words(
    sentence: str,
    segments: List[Segment],
    num_words: int,
    start_from_idx: int = 0,
) -> Tuple[Optional[int], Optional[int], str]:
    """
    Find segment where the FIRST N words of sentence match.
    
    Returns:
        (segment_index, segment_id, matched_text) or (None, None, error_msg)
    """
    first_words = get_first_words(sentence, num_words)
    pattern = norm_text(first_words)
    
    if not pattern:
        return None, None, "empty_pattern"
    
    # Search from start_from_idx forward
    for seg in segments[start_from_idx:]:
        if pattern in seg.norm:
            return seg.idx, seg.seg_id, seg.text.strip()
    
    # Fallback: search from beginning
    for seg in segments[:start_from_idx]:
        if pattern in seg.norm:
            return seg.idx, seg.seg_id, seg.text.strip()
    
    return None, None, f"no_match_for_first_{num_words}_words: '{first_words}'"


def find_segment_by_last_words(
    sentence: str,
    segments: List[Segment],
    num_words: int,
    start_from_idx: int = 0,
) -> Tuple[Optional[int], Optional[int], str]:
    """
    Find segment where the LAST N words of sentence match.
    
    Returns:
        (segment_index, segment_id, matched_text) or (None, None, error_msg)
    """
    last_words = get_last_words(sentence, num_words)
    pattern = norm_text(last_words)
    
    if not pattern:
        return None, None, "empty_pattern"
    
    # Search from start_from_idx forward
    for seg in segments[start_from_idx:]:
        if pattern in seg.norm:
            return seg.idx, seg.seg_id, seg.text.strip()
    
    # Fallback: search from beginning
    for seg in segments[:start_from_idx]:
        if pattern in seg.norm:
            return seg.idx, seg.seg_id, seg.text.strip()
    
    return None, None, f"no_match_for_last_{num_words}_words: '{last_words}'"


# ============================================================================
# Time Resolution
# ============================================================================

def resolve_clip_times(
    opening_sentence: str,
    closing_sentence: str,
    segments: List[Segment],
    pad_pre: float,
    pad_post: float,
    num_words: int,
) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    """
    Resolve clip times by matching opening (first N words) and closing (last N words).
    
    Returns:
        (start_seconds, end_seconds, metadata) or (None, None, error_metadata)
    """
    meta = {
        "adjustment_reason": [],
        "opening_segment_id": None,
        "closing_segment_id": None,
        "opening_matched_text": None,
        "closing_matched_text": None,
    }
    
    # Find opening segment (by FIRST words)
    start_idx, start_seg_id, start_text = find_segment_by_first_words(
        opening_sentence,
        segments,
        num_words,
        start_from_idx=0
    )
    
    if start_idx is None:
        meta["adjustment_reason"].append(start_text)
        return None, None, meta
    
    meta["opening_segment_id"] = start_seg_id
    meta["opening_matched_text"] = start_text
    
    # Find closing segment (by LAST words, search from opening forward)
    end_idx, end_seg_id, end_text = find_segment_by_last_words(
        closing_sentence,
        segments,
        num_words,
        start_from_idx=start_idx
    )
    
    if end_idx is None:
        meta["adjustment_reason"].append(end_text)
        return None, None, meta
    
    meta["closing_segment_id"] = end_seg_id
    meta["closing_matched_text"] = end_text
    
    # Use exact segment times
    start_s = segments[start_idx].start - pad_pre
    end_s = segments[end_idx].end + pad_post
    
    # Sanity check
    if end_s <= start_s:
        meta["adjustment_reason"].append("end_before_start_error")
        return None, None, meta
    
    # Round to centiseconds
    start_s = round(start_s, 2)
    end_s = round(end_s, 2)
    
    if not meta["adjustment_reason"]:
        meta["adjustment_reason"].append("exact_match")
    
    return start_s, end_s, meta


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    print("="*60)
    print("CLIP TIMESTAMP RESOLVER V2 (PURE SECONDS)")
    print("="*60)
    
    print("\nConfiguration:")
    print(f"  Clip Plan:     {CLIP_PLAN}")
    print(f"  Transcript:    {TRANSCRIPT}")
    print(f"  Match Words:   {MATCH_WORDS} (first for opening, last for closing)")
    print(f"  Padding:       {PAD_PRE}s pre / {PAD_POST}s post")
    print()
    
    # Validate inputs
    missing = []
    for name, path in [("Clip plan", CLIP_PLAN), ("Transcript", TRANSCRIPT)]:
        if not Path(path).exists():
            missing.append(f"  ❌ {name}: {path}")
    
    if missing:
        print("ERROR: Missing input files:")
        print("\n".join(missing))
        sys.exit(1)
    
    # Load clip plan
    print(f"Loading clip plan: {CLIP_PLAN}")
    try:
        with open(CLIP_PLAN, "r", encoding="utf-8") as f:
            clips = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR loading clip plan: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not isinstance(clips, list) or not clips:
        print("ERROR: clip_select.yaml must be a non-empty YAML list", file=sys.stderr)
        sys.exit(1)
    
    print(f"  Found {len(clips)} clips")
    
    # Load transcript
    print(f"Loading transcript: {TRANSCRIPT}")
    try:
        segments = load_segments(TRANSCRIPT)
        print(f"  Loaded {len(segments)} segments")
    except Exception as e:
        print(f"ERROR loading transcript: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Process clips
    print(f"\nResolving timestamps...")
    print("-"*60)
    resolved_clips = []
    skipped = 0
    
    for idx, clip in enumerate(clips, 1):
        clip_id = f"clip{idx:02d}"
        
        opening = clip.get("opening_sentence") or ""
        closing = clip.get("closing_sentence") or ""
        
        if not opening or not closing:
            print(f"  ⚠️  {clip_id}: Missing sentences, skipping")
            clip["clip_id"] = clip_id
            clip["adjustment_reason"] = ["skipped_missing_sentences"]
            resolved_clips.append(clip)
            skipped += 1
            continue
        
        # Resolve times
        start_s, end_s, meta = resolve_clip_times(
            opening_sentence=opening,
            closing_sentence=closing,
            segments=segments,
            pad_pre=PAD_PRE,
            pad_post=PAD_POST,
            num_words=MATCH_WORDS,
        )
        
        if start_s is None or end_s is None:
            print(f"  ❌ {clip_id}: FAILED - {', '.join(meta['adjustment_reason'])}")
            clip["clip_id"] = clip_id
            clip["adjustment_reason"] = meta["adjustment_reason"]
            resolved_clips.append(clip)
            skipped += 1
            continue
        
        # Build resolved clip (SECONDS ONLY)
        resolved = dict(clip)
        resolved["clip_id"] = clip_id
        resolved["start_s"] = start_s
        resolved["end_s"] = end_s
        resolved["duration_s"] = round(end_s - start_s, 2)
        
        # Add HH:MM:SS for human display only
        resolved["start"] = seconds_to_hms(start_s)
        resolved["end"] = seconds_to_hms(end_s)
        
        # Add segment metadata
        resolved["opening_segment_id"] = meta["opening_segment_id"]
        resolved["closing_segment_id"] = meta["closing_segment_id"]
        resolved["opening_matched_text"] = meta["opening_matched_text"]
        resolved["closing_matched_text"] = meta["closing_matched_text"]
        
        if meta["adjustment_reason"]:
            resolved["adjustment_reason"] = meta["adjustment_reason"]
        
        resolved_clips.append(resolved)
        
        print(f"  ✓ {clip_id}: {resolved['duration_s']}s ({resolved['start']} → {resolved['end']})")
        print(f"      Opening seg#{meta['opening_segment_id']}: {meta['opening_matched_text'][:60]}...")
        print(f"      Closing seg#{meta['closing_segment_id']}: {meta['closing_matched_text'][:60]}...")
    
    # Write output
    print(f"\n{'-'*60}")
    
    try:
        with open(OUTPUT_RESOLVED, "w", encoding="utf-8") as f:
            yaml.safe_dump(resolved_clips, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
        print(f"  ✓ Wrote {OUTPUT_RESOLVED}")
    except Exception as e:
        print(f"  ❌ Failed to write {OUTPUT_RESOLVED}: {e}")
        sys.exit(1)
    
    # Summary
    successful = len(clips) - skipped
    print(f"\n{'='*60}")
    print(f"✅ Complete: {successful}/{len(clips)} clips resolved")
    if skipped:
        print(f"⚠️  Skipped: {skipped} clips (see errors above)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
