#!/usr/bin/env python3
"""
render_refined_clips.py

Render vertical clips with optional view switching instructions from
plans/directors_cut.yaml. The script reuses the same encode settings as
render_clips.sh while sourcing the precise start/end timestamps from
plans/clip_plan_resolved.yaml.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import yaml


DEFAULT_VERTICAL_FILTER = "crop=(ih*9/16):ih:(iw-(ih*9/16))/2:0"


@dataclass
class ClipPlanEntry:
    clip_id: str
    start_s: float
    end_s: float


@dataclass
class SegmentInstruction:
    start_offset: float
    end_offset: float
    view: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render refined clips using directors_cut.yaml instructions.",
    )
    parser.add_argument("channel_name", type=str, help="Channel folder under data/channels.")
    parser.add_argument("episode_id", type=str, help="Episode folder inside the channel.")
    parser.add_argument(
        "--clip-id",
        action="append",
        dest="clip_ids",
        help="Optional clip_id to render (can be provided multiple times).",
    )
    parser.add_argument(
        "--channels-dir",
        type=Path,
        default=Path("data/channels"),
        help="Root directory for channel data (default: data/channels).",
    )
    parser.add_argument(
        "--clip-plan",
        type=Path,
        default=None,
        help="Override path to clip_plan_resolved.yaml.",
    )
    parser.add_argument(
        "--directors-cut",
        type=Path,
        default=None,
        help="Override path to directors_cut.yaml.",
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        default=None,
        help="Override source video file (default: source.mp4 inside the episode).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: refined_clips inside the episode).",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=str,
        default="ffmpeg",
        help="Path or name of ffmpeg binary (default: ffmpeg on PATH).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="CRF value for libx264 encoding (default: 18).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="veryfast",
        help="ffmpeg preset for libx264 encoding (default: veryfast).",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict) and not isinstance(data, list):
        raise ValueError(f"Unexpected YAML format in {path}")
    return data


def load_clip_plan(path: Path) -> Dict[str, ClipPlanEntry]:
    raw = load_yaml(path)
    if not isinstance(raw, list):
        raise ValueError(f"clip_plan_resolved must be a list: {path}")
    entries: Dict[str, ClipPlanEntry] = {}
    for item in raw:
        clip_id = item.get("clip_id")
        start_s = item.get("start_s")
        end_s = item.get("end_s")
        if clip_id is None or start_s is None or end_s is None:
            continue
        try:
            entries[clip_id] = ClipPlanEntry(
                clip_id=str(clip_id),
                start_s=float(start_s),
                end_s=float(end_s),
            )
        except (TypeError, ValueError):
            raise ValueError(f"Invalid numeric values for clip {clip_id} in clip plan.")
    return entries


def load_directors_cut(path: Path) -> tuple[Dict[str, dict], Dict[str, dict]]:
    raw = load_yaml(path)
    camera_setup = raw.get("camera_setup", {}) if isinstance(raw, dict) else {}
    clips_list = raw.get("clips", []) if isinstance(raw, dict) else []

    if not isinstance(camera_setup, dict):
        raise ValueError(f"camera_setup must be a dict in {path}")
    if not isinstance(clips_list, list):
        raise ValueError(f"clips must be a list in {path}")

    clips: Dict[str, dict] = {}
    for entry in clips_list:
        clip_id = entry.get("clip_id")
        if not clip_id:
            continue
        clips[clip_id] = entry
    return camera_setup, clips


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def build_crop_filters(camera_setup: Dict[str, dict]) -> Dict[str, str]:
    filters: Dict[str, str] = {}
    width_expr = "(ih*9/16)"
    offset_expr = f"(iw-({width_expr}))"

    for view, config in camera_setup.items():
        position = config.get("crop_position")
        if position is None:
            continue
        try:
            pos_value = clamp(float(position))
        except (TypeError, ValueError):
            raise ValueError(f"crop_position must be numeric for view '{view}'.")
        filters[view] = f"crop={width_expr}:ih:({offset_expr}*{pos_value}):0"

    filters.setdefault("center", DEFAULT_VERTICAL_FILTER)
    return filters


def format_seconds(value: float) -> str:
    return f"{value:.3f}"


def run_ffmpeg(cmd: Sequence[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg binary not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed with exit code {exc.returncode}") from exc


def render_simple_clip(
    ffmpeg_bin: str,
    source: Path,
    output_path: Path,
    start_s: float,
    end_s: float,
    crop_filter: str,
    crf: int,
    preset: str,
) -> None:
    duration = end_s - start_s
    if duration <= 0:
        raise ValueError("Clip duration must be positive for simple rendering.")

    vf = f"{crop_filter},scale=1080:1920"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-stats",
        "-ss",
        format_seconds(start_s),
        "-t",
        format_seconds(duration),
        "-i",
        str(source),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-c:a",
        "aac",
        str(output_path),
    ]
    run_ffmpeg(cmd)


def render_segmented_clip(
    ffmpeg_bin: str,
    source: Path,
    output_path: Path,
    clip_start: float,
    segments: List[SegmentInstruction],
    view_filters: Dict[str, str],
    crf: int,
    preset: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        part_paths: List[Path] = []
        for index, segment in enumerate(segments):
            abs_start = clip_start + segment.start_offset
            abs_end = clip_start + segment.end_offset
            duration = abs_end - abs_start
            if duration <= 0:
                raise ValueError(
                    f"Segment {index} has non-positive duration "
                    f"({segment.start_offset} -> {segment.end_offset})."
                )
            filter_key = segment.view or "center"
            crop_filter = view_filters.get(filter_key, DEFAULT_VERTICAL_FILTER)
            vf = f"{crop_filter},scale=1080:1920"
            part_path = Path(tmpdir) / f"segment_{index:02d}.mp4"
            part_paths.append(part_path)

            cmd = [
                ffmpeg_bin,
                "-y",
                "-loglevel",
                "error",
                "-stats",
                "-ss",
                format_seconds(abs_start),
                "-t",
                format_seconds(duration),
                "-i",
                str(source),
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-c:a",
                "aac",
                str(part_path),
            ]
            run_ffmpeg(cmd)

        list_path = Path(tmpdir) / "concat_list.txt"
        with list_path.open("w", encoding="utf-8") as handle:
            for part in part_paths:
                handle.write(f"file '{part}'\n")

        concat_cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-stats",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(output_path),
        ]
        run_ffmpeg(concat_cmd)


def collect_segments(entry: dict) -> List[SegmentInstruction]:
    segments_raw = entry.get("segments") or []
    segments: List[SegmentInstruction] = []
    for segment in segments_raw:
        start = segment.get("from")
        end = segment.get("to")
        if start is None or end is None:
            continue
        try:
            start_float = float(start)
            end_float = float(end)
        except (TypeError, ValueError):
            raise ValueError("Segment from/to must be numeric.")
        view = segment.get("view")
        segments.append(
            SegmentInstruction(
                start_offset=start_float,
                end_offset=end_float,
                view=view,
            )
        )
    return segments


def determine_clip_ids(
    selected_ids: Iterable[str] | None,
    available: Dict[str, dict],
) -> List[str]:
    if selected_ids:
        clip_ids = []
        for clip_id in selected_ids:
            clip_ids.append(clip_id)
        return clip_ids
    return list(available.keys())


def main() -> int:
    args = parse_args()

    episode_dir = args.channels_dir / args.channel_name / args.episode_id
    clip_plan_path = args.clip_plan or (episode_dir / "plans" / "clip_plan_resolved.yaml")
    directors_cut_path = args.directors_cut or (episode_dir / "plans" / "directors_cut.yaml")
    source_video = args.input_video or (episode_dir / "source.mp4")
    output_dir = args.output_dir or (episode_dir / "refined_clips")

    if not episode_dir.exists():
        print(f"Episode directory not found: {episode_dir}")
        return 1
    if not source_video.exists():
        print(f"Source video not found: {source_video}")
        return 1

    try:
        clip_plan = load_clip_plan(clip_plan_path)
    except Exception as exc:  # pragma: no cover - runtime validation
        print(f"Failed to load clip plan: {exc}")
        return 1

    try:
        camera_setup, director_clips = load_directors_cut(directors_cut_path)
    except FileNotFoundError:
        print(f"Directors cut file not found: {directors_cut_path}")
        return 1
    except Exception as exc:  # pragma: no cover - runtime validation
        print(f"Failed to load directors cut: {exc}")
        return 1

    if not director_clips:
        print("No clips defined in directors_cut.yaml. Nothing to render.")
        return 0

    clip_ids = determine_clip_ids(args.clip_ids, director_clips)
    view_filters = build_crop_filters(camera_setup)

    output_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    skipped: List[str] = []

    for clip_id in clip_ids:
        director_entry = director_clips.get(clip_id)
        if director_entry is None:
            print(f"[skip] clip_id '{clip_id}' not present in directors_cut.yaml.")
            skipped.append(clip_id)
            continue

        if director_entry.get("publish", True) is False:
            print(f"[skip] clip_id '{clip_id}' marked publish=false.")
            skipped.append(clip_id)
            continue

        plan_entry = clip_plan.get(clip_id)
        if plan_entry is None:
            print(f"[skip] clip_id '{clip_id}' missing from clip_plan_resolved.yaml.")
            skipped.append(clip_id)
            continue

        segments = collect_segments(director_entry)
        output_path = output_dir / f"{clip_id}.mp4"

        try:
            if not segments:
                print(f"[render] {clip_id}: simple center crop.")
                render_simple_clip(
                    ffmpeg_bin=args.ffmpeg_bin,
                    source=source_video,
                    output_path=output_path,
                    start_s=plan_entry.start_s,
                    end_s=plan_entry.end_s,
                    crop_filter=view_filters.get("center", DEFAULT_VERTICAL_FILTER),
                    crf=args.crf,
                    preset=args.preset,
                )
            else:
                print(f"[render] {clip_id}: {len(segments)} segments with view switching.")
                render_segmented_clip(
                    ffmpeg_bin=args.ffmpeg_bin,
                    source=source_video,
                    output_path=output_path,
                    clip_start=plan_entry.start_s,
                    segments=segments,
                    view_filters=view_filters,
                    crf=args.crf,
                    preset=args.preset,
                )
        except Exception as exc:
            print(f"[error] Failed to render {clip_id}: {exc}")
            skipped.append(clip_id)
            continue

        print(f"[done] {clip_id} → {output_path}")
        rendered += 1

    print(f"\nRendered {rendered} clip(s). Skipped {len(skipped)}.")
    if skipped:
        print("Skipped clip_ids:", ", ".join(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())

