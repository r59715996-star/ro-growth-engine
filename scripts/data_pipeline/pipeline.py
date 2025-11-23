"""
High-level pipeline runner for tagging workflows.

Capabilities:
1. Full pipeline (transcribe inbox clips -> quantitative features).
2. Qualitative tagging via Groq llama-3.3-70b-versatile for single clips or entire channels.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from scripts.data_pipeline import groq_qual_client


SCRIPT_DIR = Path(__file__).resolve().parent


def build_cli_parser() -> argparse.ArgumentParser:
    """
    Construct the CLI parser supporting full and qualitative modes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Tagging pipeline runner. "
            "Without --transcript/--channel-root it runs transcription + quant. "
            "Otherwise it executes qualitative tagging using a provided system prompt."
        )
    )
    parser.add_argument(
        "channel_name",
        nargs="?",
        help="Channel folder under data/channels for the full pipeline (e.g. 'Odds On Open').",
    )
    parser.add_argument(
        "--channels-dir",
        type=Path,
        default=Path("data/channels"),
        help="Root directory containing channel folders (default: data/channels).",
    )
    parser.add_argument(
        "--clip-id",
        help="Optional clip identifier to process a single clip (default: process all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing transcripts instead of skipping.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="Path or name of the ffmpeg binary (default: ffmpeg).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate for ffmpeg extraction (default: 16000 Hz).",
    )
    parser.add_argument(
        "--api-key",
        help="Groq API key; if omitted Groq client reads GROQ_API_KEY env var.",
    )
    parser.add_argument(
        "--quant-output-dir",
        type=Path,
        help="Optional directory to store *_ta.json files (default: tagging/clip_tags).",
    )
    parser.add_argument(
        "--transcript",
        help="Path to a single transcript JSON for qualitative tagging.",
    )
    parser.add_argument(
        "--channel-root",
        help="Channel root directory for qualitative tagging (e.g. data/channels/Odds On Open).",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        help=(
            "Path to the qualitative system prompt text file. Required when "
            "running qualitative tagging (explicitly or after the full pipeline)."
        ),
    )
    parser.add_argument(
        "--qual-output",
        type=Path,
        help="Optional explicit output path for a single qualitative tagging run.",
    )
    return parser


def run_subprocess(command: Sequence[str]) -> None:
    """Invoke a subprocess command, raising a friendly error if it fails."""
    print("→ Running:", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime error
        raise SystemExit(
            f"Command failed with exit code {exc.returncode}: {' '.join(command)}"
        ) from exc


def run_transcription(args: argparse.Namespace) -> None:
    """Execute the transcription step."""
    channel_root = args.channels_dir / args.channel_name / "tagging"
    inbox_dir = channel_root / "inbox"
    transcripts_dir = channel_root / "transcripts"

    # Skip if transcripts already exist and overwrite not requested.
    if inbox_dir.exists() and transcripts_dir.exists() and not args.overwrite:
        clip_ext = ".mp4"
        if args.clip_id:
            clip_path = inbox_dir / args.clip_id
            if clip_path.suffix.lower() != clip_ext:
                clip_path = clip_path.with_suffix(clip_ext)
            candidate_clips = [clip_path]
        else:
            candidate_clips = sorted(
                path for path in inbox_dir.glob(f"*{clip_ext}") if path.is_file()
            )

        if candidate_clips:
            all_transcripts_exist = all(
                (transcripts_dir / f"{clip.stem}_tr.json").exists()
                for clip in candidate_clips
            )
            if all_transcripts_exist:
                print(f"Skipping transcription (existing transcripts in {transcripts_dir}).")
                return

    script_path = SCRIPT_DIR / "transcribe_inbox_clips.py"
    command = [
        sys.executable,
        str(script_path),
        args.channel_name,
        "--channels-dir",
        str(args.channels_dir),
        "--ffmpeg-bin",
        args.ffmpeg_bin,
        "--sample-rate",
        str(args.sample_rate),
    ]
    if args.clip_id:
        command.extend(["--clip-id", args.clip_id])
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    if args.overwrite:
        command.append("--overwrite")
    run_subprocess(command)


def run_quant(args: argparse.Namespace) -> None:
    """Execute the quantitative feature extraction step."""
    transcripts_dir = (
        args.channels_dir / args.channel_name / "tagging" / "transcripts"
    )
    if not transcripts_dir.exists():
        raise SystemExit(f"Transcript directory not found: {transcripts_dir}")

    script_path = SCRIPT_DIR / "quant_v1.py"
    command = [
        sys.executable,
        str(script_path),
        str(transcripts_dir),
    ]
    if args.quant_output_dir:
        command.extend(["--output", str(args.quant_output_dir)])
    run_subprocess(command)


def _load_prompt_text(prompt_path: Path) -> str:
    """
    Load the qualitative system prompt.
    """
    if not prompt_path:
        raise ValueError("prompt_path must be provided for qualitative tagging.")
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except OSError as exc:  # pragma: no cover - depends on filesystem state
        raise SystemExit(f"Failed to read system prompt: {prompt_path}: {exc}") from exc


def run_qual_v1_for_transcript(
    transcript_path: str,
    system_prompt: str,
    output_path: Optional[str] = None,
) -> Path:
    """
    Run qualitative v1 tagging for a single transcript JSON.
    """
    transcript_file = Path(transcript_path)
    if not transcript_file.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_file}")

    destination = (
        Path(output_path)
        if output_path is not None
        else _derive_qual_output_path(transcript_file)
    )
    if destination.exists():
        print(f"Skipping qualitative tagging (exists): {destination}")
        return destination

    with transcript_file.open("r", encoding="utf-8") as handle:
        transcript_json = json.load(handle)

    tags = groq_qual_client.compute_qual_v1_from_transcript(
        transcript_json, system_prompt
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as outfile:
        json.dump(tags, outfile, indent=2, sort_keys=True)
    print(f"Qualitative tags saved: {transcript_file} -> {destination}")
    return destination


def _derive_qual_output_path(transcript_file: Path) -> Path:
    """
    Determine default qualitative output path.
    """
    filename = transcript_file.name
    if filename.endswith("_tr.json"):
        output_name = f"{filename[:-8]}_qa.json"
    else:
        output_name = f"{transcript_file.stem}_qa.json"

    parts = list(transcript_file.parts)
    if "transcripts" in parts:
        idx = parts.index("transcripts")
        before = Path(*parts[:idx]) if idx > 0 else Path(".")
        after_parts = parts[idx + 1 : -1]
        destination_dir = before / "clip_tags"
        if after_parts:
            destination_dir = destination_dir / Path(*after_parts)
    else:
        destination_dir = transcript_file.parent
    return destination_dir / output_name


def run_qual_v1_for_channel(
    channel_root: str,
    system_prompt: str,
) -> None:
    """
    Run qualitative tagging across an entire channel.
    """
    root = Path(channel_root)
    transcripts_dir = root / "tagging" / "transcripts"
    if not transcripts_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {transcripts_dir}")

    transcripts = sorted(transcripts_dir.rglob("*_tr.json"))
    if not transcripts:
        print(f"No transcripts found under {transcripts_dir}")
        return

    print(f"Running qualitative tagging for {len(transcripts)} transcript(s).")
    for transcript in transcripts:
        run_qual_v1_for_transcript(transcript, system_prompt)


def main() -> None:
    parser = build_cli_parser()
    args = parser.parse_args()

    qual_mode = bool(args.transcript or args.channel_root)
    if qual_mode:
        if args.transcript and args.channel_root:
            parser.error("Use either --transcript or --channel-root (not both).")
        if not args.prompt_path:
            parser.error("--prompt-path is required for qualitative tagging.")
        system_prompt = _load_prompt_text(args.prompt_path)
        if args.transcript:
            run_qual_v1_for_transcript(
                args.transcript, system_prompt, args.qual_output
            )
        else:
            run_qual_v1_for_channel(args.channel_root, system_prompt)
        return

    if not args.channel_name:
        parser.error("channel_name is required for the transcription + quant pipeline.")

    run_transcription(args)
    run_quant(args)

    if args.prompt_path:
        system_prompt = _load_prompt_text(args.prompt_path)
        channel_root = args.channels_dir / args.channel_name
        run_qual_v1_for_channel(channel_root, system_prompt)
        print("✅ Pipeline completed (transcribe + quant + qual).")
    else:
        print("✅ Pipeline completed successfully (transcribe + quant).")


if __name__ == "__main__":
    main()
