"""
Pipeline variant that prefers MP3 clips in tagging inboxes.

This mirrors scripts/data_pipeline/pipeline.py but defaults to reading .mp3
files produced by download_shorts_audio.py (via yt-dlp). It forwards
--clip-extension to the transcription step so you can override if needed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# Ensure repo root is on sys.path so `scripts` imports work when run directly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_pipeline import groq_qual_client


DATA_PIPELINE_DIR = SCRIPT_DIR.parent / "data_pipeline"
TRANSCRIBE_SCRIPT = SCRIPT_DIR / "transcribe_inbox_clips.py"
QUANT_SCRIPT = DATA_PIPELINE_DIR / "quant_v1.py"


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tagging pipeline runner (MP3-friendly). "
            "Without --transcript/--channel-root it runs transcription + quant. "
            "Otherwise it executes qualitative tagging using a provided system prompt."
        )
    )
    parser.add_argument(
        "--channel",
        required=True,
        help="Channel folder under data/tagging/{niche} (e.g. 'millennial_masters').",
    )
    parser.add_argument(
        "--niche",
        default="entrepreneurship",
        help="Niche folder under data/tagging (default: entrepreneurship).",
    )
    parser.add_argument(
        "--clip-id",
        help="Optional clip identifier to process a single clip (default: process all).",
    )
    parser.add_argument(
        "--clip-extension",
        default=".mp3",
        help="Clip extension to target (default: .mp3).",
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
        help="Channel root directory for qualitative tagging (e.g. data/tagging/entrepreneurship/millennial_masters).",
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
    print("→ Running:", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime error
        raise SystemExit(
            f"Command failed with exit code {exc.returncode}: {' '.join(command)}"
        ) from exc


def run_transcription(args: argparse.Namespace) -> None:
    base_path = Path("data/tagging") / args.niche / args.channel
    inbox_dir = base_path / "inbox"
    transcripts_dir = base_path / "transcripts"

    if inbox_dir.exists() and transcripts_dir.exists() and not args.overwrite:
        clip_ext = args.clip_extension if args.clip_extension.startswith(".") else f".{args.clip_extension}"
        if args.clip_id:
            clip_path = inbox_dir / args.clip_id
            if clip_path.suffix.lower() != clip_ext.lower():
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

    command = [
        sys.executable,
        str(TRANSCRIBE_SCRIPT),
        args.channel,
        "--niche",
        args.niche,
        "--channels-dir",
        "data/tagging",
        "--ffmpeg-bin",
        args.ffmpeg_bin,
        "--sample-rate",
        str(args.sample_rate),
        "--clip-extension",
        args.clip_extension,
    ]
    if args.clip_id:
        command.extend(["--clip-id", args.clip_id])
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    if args.overwrite:
        command.append("--overwrite")
    run_subprocess(command)


def run_quant(args: argparse.Namespace) -> None:
    transcripts_dir = Path("data/tagging") / args.niche / args.channel / "transcripts"
    if not transcripts_dir.exists():
        raise SystemExit(f"Transcript directory not found: {transcripts_dir}")

    command = [
        sys.executable,
        str(QUANT_SCRIPT),
        str(transcripts_dir),
    ]
    if args.quant_output_dir:
        command.extend(["--output", str(args.quant_output_dir)])
    run_subprocess(command)


def _load_prompt_text(prompt_path: Path) -> str:
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

    try:
        tags = groq_qual_client.compute_qual_v1_from_transcript(
            transcript_json, system_prompt
        )
    except ValueError as exc:
        # Gracefully handle transcripts with no textual content.
        message = str(exc)
        if "does not contain textual content" in message:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8") as outfile:
                json.dump(
                    {
                        "skipped": True,
                        "reason": "Transcript contained no words.",
                        "source": str(transcript_file),
                    },
                    outfile,
                    indent=2,
                    sort_keys=True,
                )
            print(f"Skipping qualitative tagging (empty transcript): {transcript_file}")
            return destination
        raise


    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as outfile:
        json.dump(tags, outfile, indent=2, sort_keys=True)
    print(f"Qualitative tags saved: {transcript_file} -> {destination}")
    return destination


def _derive_qual_output_path(transcript_file: Path) -> Path:
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
    root = Path(channel_root)
    transcripts_dir = root / "transcripts"
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

    if not args.channel:
        parser.error("--channel is required for the transcription + quant pipeline.")

    run_transcription(args)
    run_quant(args)

    if args.prompt_path:
        system_prompt = _load_prompt_text(args.prompt_path)
        channel_root = Path("data/tagging") / args.niche / args.channel
        run_qual_v1_for_channel(channel_root, system_prompt)
        print("✅ Pipeline completed (transcribe + quant + qual).")
    else:
        print("✅ Pipeline completed successfully (transcribe + quant).")


if __name__ == "__main__":
    main()
