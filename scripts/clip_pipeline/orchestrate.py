#!/usr/bin/env python3
"""
orchestrate.py - End-to-end video clipping pipeline (Production Version)

Complete automated pipeline from video ID to rendered clips.
Uses hook-first architecture with Python filtering + LLM refinement.

Steps:
1a. Download source video (yt-dlp) when no local input provided
1.  Extract audio from video (FFmpeg)
2.  Transcribe audio (Whisper)
3.  Discover hooks (Python scoring)
4.  Generate context (Gemini 2.5 Flash)
5.  Select clips (Gemini 2.5 Flash with candidates)
6.  Resolve timestamps
7.  Generate batch manifest
8.  Render clips
9.  Generate subtitles

Usage:
    python orchestrate.py --channel-name divot --episode-id ep001
    python orchestrate.py --channel-name divot --episode-id ep001 --input assets/wellness_pod.webm
    python orchestrate.py --channel-name divot --episode-id ep001 --input assets/wellness_pod.webm --skip-render
    python orchestrate.py --channel-name divot --episode-id ep001 --input assets/wellness_pod.webm --force

Requirements:
    - FFmpeg installed
    - google-generativeai Python package
    - GOOGLE_API_KEY environment variable set
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path

try:
    from llm_client import generate_context, select_clips
except ImportError:
    generate_context = None
    select_clips = None


# ============================================================================
# Configuration
# ============================================================================

SCRIPTS_DIR = Path("scripts/clip_pipeline")


# ============================================================================
# Utilities
# ============================================================================

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(title)
    print("="*60)


def run_command(cmd: list, description: str, capture: bool = True) -> bool:
    """Run a shell command and handle errors."""
    print(f"  Running: {description}")
    try:
        if capture:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        else:
            subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed: {e}")
        if capture and e.stderr:
            print(f"     Error: {e.stderr[:200]}")
        return False
    except FileNotFoundError:
        print(f"  ❌ Command not found: {cmd[0]}")
        print(f"     Make sure it's installed and in PATH")
        return False


# ============================================================================
# Pipeline Steps
# ============================================================================

def step_download_video(episode_id: str, video_path: Path, force: bool) -> bool:
    """Download the source video using yt-dlp."""
    print_header("STEP 1A: DOWNLOAD VIDEO")

    if video_path.exists() and not force:
        print(f"  ✓ Source video exists: {video_path.name}")
        return True

    download_script = SCRIPTS_DIR / "download_episode.py"

    if not download_script.exists():
        print(f"  ❌ Script not found: {download_script}")
        return False

    cmd = [
        "python",
        str(download_script),
        episode_id,
        "--output-dir",
        str(video_path.parent),
        "--filename",
        video_path.name,
    ]
    if force:
        cmd.append("--force")

    return run_command(
        cmd,
        "YouTube download",
        capture=False,
    )

def step_extract_audio(video_path: Path, audio_path: Path, force: bool) -> bool:
    """Extract audio from video using FFmpeg."""
    print_header("STEP 1: EXTRACT AUDIO")
    
    if audio_path.exists() and not force:
        print(f"  ✓ Audio exists: {audio_path.name}")
        return True
    
    print(f"  Video: {video_path.name}")
    print(f"  Output: {audio_path.name}")
    
    return run_command(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "libmp3lame",
            "-b:a", "64k",
            str(audio_path)
        ],
        "FFmpeg audio extraction"
    )


def step_transcribe(channel_name: str, episode_id: str, transcript_path: Path, force: bool) -> bool:
    """Transcribe audio using the local Whisper helper."""
    print_header("STEP 2: TRANSCRIBE AUDIO")
    
    if transcript_path.exists() and not force:
        print(f"  ✓ Transcript exists: {transcript_path.name}")
        return True
    
    transcribe_script = SCRIPTS_DIR / "transcribe_audio.py"
    
    if not transcribe_script.exists():
        print(f"  ❌ Script not found: {transcribe_script}")
        return False
    
    cmd = [
        "python",
        str(transcribe_script),
        channel_name,
        episode_id,
    ]
    if force:
        cmd.append("--overwrite")
    
    return run_command(
        cmd,
        "Automatic transcription",
        capture=False,
    )


def step_discover_hooks(transcript_path: Path, candidates_path: Path, force: bool) -> bool:
    """Run Python hook discovery."""
    print_header("STEP 3: DISCOVER HOOKS (Python Scoring)")
    
    if candidates_path.exists() and not force:
        print(f"  ✓ Candidates exist: {candidates_path.name}")
        return True
    
    discover_script = SCRIPTS_DIR / "discover_hooks.py"
    
    if not discover_script.exists():
        print(f"  ❌ Script not found: {discover_script}")
        return False
    
    return run_command(
        ["python", str(discover_script)],
        "Hook discovery",
        capture=False
    )


def step_generate_context(transcript_path: Path, context_path: Path, force: bool) -> bool:
    """Generate context using Gemini 2.5 Flash."""
    print_header("STEP 4: GENERATE CONTEXT (Gemini 2.5 Flash)")
    
    if context_path.exists() and not force:
        print(f"  ✓ Context exists: {context_path.name}")
        return True
    
    if generate_context is None:
        print("  ❌ Cannot import llm_client.generate_context")
        print("     Ensure scripts/llm_client.py exists")
        return False
    
    try:
        return generate_context(str(transcript_path), str(context_path))
    except Exception as e:
        print(f"  ❌ Context generation failed: {e}")
        return False


def step_select_clips(
    transcript_path: Path,
    context_path: Path,
    hook_candidates_path: Path,
    clip_path: Path,
    force: bool
) -> bool:
    """Select clips using LLM refinement of candidates."""
    print_header("STEP 5: SELECT CLIPS (Gemini 2.5 Flash)")
    
    if clip_path.exists() and not force:
        print(f"  ✓ Clips exist: {clip_path.name}")
        return True
    
    if select_clips is None:
        print("  ❌ Cannot import llm_client.select_clips")
        print("     Ensure scripts/llm_client.py exists")
        return False
    
    try:
        return select_clips(
            str(transcript_path),
            str(context_path),
            str(hook_candidates_path),
            str(clip_path)
        )
    except Exception as e:
        print(f"  ❌ Clip selection failed: {e}")
        return False


def step_resolve_timestamps(clip_path: Path, resolved_path: Path, force: bool) -> bool:
    """Resolve timestamps using resolve_times_v2.py."""
    print_header("STEP 6: RESOLVE TIMESTAMPS")
    
    if resolved_path.exists() and not force:
        print(f"  ✓ Resolved plan exists: {resolved_path.name}")
        return True
    
    resolve_script = SCRIPTS_DIR / "resolve_times_v2.py"
    
    if not resolve_script.exists():
        print(f"  ❌ Script not found: {resolve_script}")
        return False
    
    return run_command(
        ["python", str(resolve_script)],
        "Timestamp resolution",
        capture=False
    )


def step_generate_batch(resolved_path: Path, batch_path: Path, force: bool) -> bool:
    """Generate batch manifest using generate_batch.py."""
    print_header("STEP 7: GENERATE BATCH MANIFEST")
    
    if batch_path.exists() and not force:
        print(f"  ✓ Batch exists: {batch_path.name}")
        return True
    
    batch_script = SCRIPTS_DIR / "generate_batch.py"
    
    if not batch_script.exists():
        print(f"  ❌ Script not found: {batch_script}")
        return False
    
    print(f"  Running: Batch manifest generation")
    try:
        subprocess.run(
            ["python", str(batch_script)],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed: {e}")
        return False


def step_render_clips(batch_path: Path) -> bool:
    """Render clips using render_clips.sh."""
    print_header("STEP 8: RENDER CLIPS")
    
    render_script = SCRIPTS_DIR / "render_clips.sh"
    
    if not render_script.exists():
        print(f"  ❌ Script not found: {render_script}")
        return False
    
    return run_command(
        ["bash", str(render_script)],
        "FFmpeg clip rendering",
        capture=False
    )


def step_generate_subtitles(clips_dir: Path, clip_plan_resolved: Path) -> bool:
    """Generate .srt subtitle files from transcript."""
    print_header("STEP 9: GENERATE SUBTITLES")
    
    subtitle_script = SCRIPTS_DIR / "generate_subtitles.py"
    
    if not subtitle_script.exists():
        print(f"  ❌ Script not found: {subtitle_script}")
        return False
    
    print(f"  Generating subtitles for rendered clips...")
    
    return run_command(
        ["python", str(subtitle_script)],
        "Subtitle generation",
        capture=False
    )


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end video clipping pipeline with hook-first architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --channel-name divot --episode-id ep001 --input assets/wellness_pod.webm
  %(prog)s --channel-name divot --episode-id ep001 --input assets/wellness_pod.webm --skip-render
  %(prog)s --channel-name divot --episode-id ep001 --input assets/wellness_pod.webm --force

Architecture: Python hook scoring → LLM refinement → Clip rendering
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Optional path to an existing video file. If omitted, the episode "
        "ID is downloaded from YouTube.",
    )
    parser.add_argument(
        "--channel-name",
        type=str,
        required=True,
        help="Channel identifier (e.g., divot, the_j_curve_podcast)"
    )
    parser.add_argument(
        "--episode-id",
        type=str,
        required=True,
        help="Episode identifier (e.g., ep001, 2024_11_12_fundraising)"
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Skip rendering (useful for testing pipeline)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate all intermediate files"
    )
    
    args = parser.parse_args()
    
    provided_input = None
    if args.input:
        provided_input = Path(args.input).expanduser()
        if not provided_input.exists():
            print(f"❌ ERROR: Video not found: {provided_input}")
            sys.exit(1)
    
    # Build channel-specific directory structure
    channel_name = args.channel_name
    episode_id = args.episode_id
    base_dir = Path(f"data/channels/{channel_name}/{episode_id}")
    plans_dir = base_dir / "plans"
    clips_dir = base_dir / "clips"
    subtitles_dir = clips_dir / "subtitles"
    
    for directory in [base_dir, plans_dir, clips_dir, subtitles_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    video_path = provided_input if provided_input else base_dir / "source.mp4"
    audio_path = base_dir / "audio.mp3"
    transcript_path = base_dir / "transcript.json"
    hook_candidates_path = plans_dir / "hook_candidates.json"
    context_path = plans_dir / "context_gen.yaml"
    clip_select_path = plans_dir / "clip_select.yaml"
    clip_resolved_path = plans_dir / "clip_plan_resolved.yaml"
    batch_path = base_dir / "batch.json"
    
    # Set environment variables for child scripts to consume
    os.environ["TRANSCRIPT_PATH"] = str(transcript_path)
    os.environ["HOOK_CANDIDATES_PATH"] = str(hook_candidates_path)
    os.environ["CONTEXT_PATH"] = str(context_path)
    os.environ["CLIP_SELECT_PATH"] = str(clip_select_path)
    os.environ["CLIP_RESOLVED_PATH"] = str(clip_resolved_path)
    os.environ["BATCH_PATH"] = str(batch_path)
    os.environ["CLIPS_DIR"] = str(clips_dir)
    os.environ["SUBTITLES_DIR"] = str(subtitles_dir)
    os.environ["VIDEO_INPUT"] = str(video_path)
    
    # Print pipeline info
    print("="*60)
    print("VIDEO CLIPPING PIPELINE (Hook-First Architecture)")
    print("="*60)
    print(f"\n📹 Input:  {video_path}")
    print(f"🎯 Strategy:  Python Scoring → LLM Refinement")
    print(f"💰 Model:  Gemini 2.5 Flash (1M context)")
    
    if args.force:
        print(f"🔄 Mode:   FORCE (regenerating all files)")
    if args.skip_render:
        print(f"⏭️  Mode:   SKIP RENDER (testing only)")
    
    # Run pipeline
    pipeline_success = True

    # Step 1A: Download source video if needed
    if pipeline_success and provided_input is None:
        if not step_download_video(episode_id, video_path, args.force):
            pipeline_success = False
    
    # Step 1: Extract audio
    if pipeline_success:
        if not step_extract_audio(video_path, audio_path, args.force):
            pipeline_success = False
    
    # Step 2: Transcribe
    if pipeline_success:
        if not step_transcribe(channel_name, episode_id, transcript_path, args.force):
            pipeline_success = False
    
    # Step 3: Discover hooks (NEW)
    if pipeline_success:
        if not step_discover_hooks(transcript_path, hook_candidates_path, args.force):
            pipeline_success = False
    
    # Step 4: Generate context
    if pipeline_success:
        if not step_generate_context(transcript_path, context_path, args.force):
            pipeline_success = False
    
    # Step 4.5: Safety filter detection
    if pipeline_success and clip_select_path.exists():
        try:
            clip_content = clip_select_path.read_text(encoding="utf-8")
            if "BLOCKED_BY_SAFETY_FILTER" in clip_content:
                print("\n❌ Gemini safety filters blocked clip selection previously.")
                print("   Review the transcript or try a different episode before rerunning.")
                pipeline_success = False
        except Exception as e:
            print(f"  ⚠️  Unable to inspect clip_select.yaml: {e}")

    # Step 5: Select clips (UPDATED - uses candidates)
    if pipeline_success:
        if not step_select_clips(
            transcript_path,
            context_path,
            hook_candidates_path,
            clip_select_path,
            args.force
        ):
            pipeline_success = False
    
    # Step 6: Resolve timestamps
    if pipeline_success:
        if not step_resolve_timestamps(clip_select_path, clip_resolved_path, args.force):
            pipeline_success = False
    
    # Step 7: Generate batch
    if pipeline_success:
        if not step_generate_batch(clip_resolved_path, batch_path, args.force):
            pipeline_success = False
    
    # Step 8: Render clips
    if pipeline_success and not args.skip_render:
        if not step_render_clips(batch_path):
            pipeline_success = False
    elif args.skip_render:
        print_header("STEP 8: RENDER CLIPS")
        print("  ⏭️  Skipped (--skip-render flag)")
        print(f"\n  To render manually:")
        print(f"     bash scripts/render_clips.sh")
    
    # Step 9: Generate subtitles
    if pipeline_success and not args.skip_render:
        if not step_generate_subtitles(clips_dir, clip_resolved_path):
            print("  ⚠️  Subtitle generation failed (non-critical)")
    
    # Final summary
    print("\n" + "="*60)
    if pipeline_success:
        print("✅ PIPELINE COMPLETE!")
        if not args.skip_render:
            print(f"\n   📁 Clips: {clips_dir}/")
            print(f"   📝 Subtitles: {subtitles_dir}/")
        else:
            print(f"\n   📋 Ready to render")
            print(f"   Run: bash scripts/render_clips.sh")
    else:
        print("❌ PIPELINE FAILED")
        print("\n   Review errors above and fix before re-running")
        sys.exit(1)
    
    print("="*60)


if __name__ == "__main__":
    main()
