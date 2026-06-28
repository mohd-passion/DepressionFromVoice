"""Create clean DAIC-WOZ Phase 1 manifests with true speech-window mappings.

This script does not extract WavLM embeddings and does not train any model.

It reads the original Phase 1 participant manifest, applies documented
participant exclusions, then creates clean participant/window manifests and a
window-to-source-audio mapping. Transcript use is limited to speaker labels and
timestamps; transcript text is never read from rows or copied to outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXCLUSIONS = {
    "409": "PHQ8_Score=10 conflicts with PHQ8_Binary=0 under DAIC-WOZ depression threshold >= 10.",
    "422": "Local audio/transcript files are missing and zero windows were generated.",
}


SOURCE_TIME_MODE = "participant_speech_concatenated_with_source_mapping"


CLEAN_PARTICIPANT_FIELDS = [
    "dataset",
    "participant_id",
    "participant_dir",
    "audio_path",
    "transcript_path",
    "label_binary",
    "phq8_score",
    "official_split",
    "gender",
    "audio_duration_sec",
    "participant_speech_duration_sec",
    "sample_rate",
    "num_channels",
    "window_sec",
    "overlap",
    "hop_sec",
    "num_windows",
    "has_audio",
    "has_transcript",
    "has_label",
    "source_time_mode",
]


CLEAN_WINDOW_FIELDS = [
    "participant_id",
    "window_id",
    "concat_start_sec",
    "concat_end_sec",
    "duration_sec",
    "label_binary",
    "official_split",
    "num_source_intervals",
    "source_time_mode",
]


WINDOW_MAPPING_FIELDS = [
    "participant_id",
    "window_id",
    "concat_start_sec",
    "concat_end_sec",
    "source_interval_index",
    "source_start_sec",
    "source_end_sec",
    "source_duration_sec",
    "official_split",
    "label_binary",
]


CLEAN_SPLIT_SUMMARY_FIELDS = [
    "split",
    "participants",
    "non_depressed",
    "depressed",
    "windows",
    "avg_windows_per_participant",
]


CLEANING_REPORT_FIELDS = [
    "participant_id",
    "official_split",
    "label_binary",
    "phq8_score",
    "decision",
    "reason",
]


MAPPING_QUALITY_FIELDS = [
    "participant_id",
    "official_split",
    "participant_speech_duration_sec",
    "total_mapped_window_source_duration_sec",
    "num_windows",
    "num_source_intervals",
    "has_audio",
    "has_transcript",
    "issue_type",
    "details",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/phase1_daic.json",
        help="Path to the Phase 1 JSON config.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def participant_sort_key(row_or_id: dict[str, Any] | str) -> tuple[int, str]:
    participant_id = row_or_id["participant_id"] if isinstance(row_or_id, dict) else row_or_id
    try:
        return (int(participant_id), participant_id)
    except ValueError:
        return (10**9, participant_id)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_participant_intervals(transcript_path: Path, speaker: str) -> tuple[list[dict[str, float]], list[str]]:
    """Read participant timing intervals without using transcript text."""
    intervals: list[dict[str, float]] = []
    issues: list[str] = []

    if not transcript_path.exists():
        return intervals, ["missing_transcript"]

    with transcript_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"start_time", "stop_time", "speaker"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            return intervals, ["invalid_transcript_header"]

        concat_cursor = 0.0
        source_interval_index = 0
        for row_number, row in enumerate(reader, start=2):
            if clean_value(row.get("speaker")) != speaker:
                continue

            try:
                source_start = float(clean_value(row.get("start_time")))
                source_end = float(clean_value(row.get("stop_time")))
            except ValueError:
                issues.append(f"invalid_time_row_{row_number}")
                continue

            if source_end <= source_start:
                issues.append(f"nonpositive_interval_row_{row_number}")
                continue

            duration = source_end - source_start
            intervals.append(
                {
                    "source_interval_index": float(source_interval_index),
                    "source_start_sec": source_start,
                    "source_end_sec": source_end,
                    "source_duration_sec": duration,
                    "concat_start_sec": concat_cursor,
                    "concat_end_sec": concat_cursor + duration,
                }
            )
            concat_cursor += duration
            source_interval_index += 1

    return intervals, issues


def generate_window_starts(total_duration: float, window_sec: float, hop_sec: float) -> list[float]:
    starts: list[float] = []
    cursor = 0.0
    epsilon = 1e-9
    while cursor + window_sec <= total_duration + epsilon:
        starts.append(cursor)
        cursor += hop_sec
    return starts


def map_window_to_sources(
    intervals: list[dict[str, float]],
    concat_start: float,
    concat_end: float,
) -> list[dict[str, float]]:
    mappings: list[dict[str, float]] = []
    epsilon = 1e-9

    for interval in intervals:
        interval_start = interval["concat_start_sec"]
        interval_end = interval["concat_end_sec"]
        overlap_start = max(concat_start, interval_start)
        overlap_end = min(concat_end, interval_end)
        if overlap_end <= overlap_start + epsilon:
            continue

        source_start = interval["source_start_sec"] + (overlap_start - interval_start)
        source_end = interval["source_start_sec"] + (overlap_end - interval_start)
        mappings.append(
            {
                "source_interval_index": int(interval["source_interval_index"]),
                "source_start_sec": source_start,
                "source_end_sec": source_end,
                "source_duration_sec": source_end - source_start,
            }
        )

    return mappings


def build_clean_outputs(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    output_dir = Path(config["output_dir"])
    participant_manifest_path = output_dir / "daic_phase1_participant_manifest.csv"
    participant_rows = sorted(read_csv(participant_manifest_path), key=participant_sort_key)

    window_sec = float(config["window_sec"])
    hop_sec = window_sec * (1.0 - float(config["overlap"]))
    speaker = str(config["transcript_speaker"])

    clean_participants: list[dict[str, Any]] = []
    clean_windows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    cleaning_report: list[dict[str, Any]] = []
    quality_report: list[dict[str, Any]] = []

    for row in participant_rows:
        participant_id = clean_value(row.get("participant_id"))
        if participant_id in EXCLUSIONS:
            cleaning_report.append(
                {
                    "participant_id": participant_id,
                    "official_split": clean_value(row.get("official_split")),
                    "label_binary": clean_value(row.get("label_binary")),
                    "phq8_score": clean_value(row.get("phq8_score")),
                    "decision": "excluded",
                    "reason": EXCLUSIONS[participant_id],
                }
            )
            continue

        audio_path = Path(clean_value(row.get("audio_path")))
        transcript_path = Path(clean_value(row.get("transcript_path")))
        has_audio = audio_path.exists()
        has_transcript = transcript_path.exists()
        intervals, interval_issues = read_participant_intervals(transcript_path, speaker)
        participant_speech_duration = sum(interval["source_duration_sec"] for interval in intervals)
        window_starts = generate_window_starts(participant_speech_duration, window_sec, hop_sec)
        participant_source_interval_count = len(intervals)
        mapped_duration_total = 0.0
        participant_has_mapping_issue = False

        for window_index, concat_start in enumerate(window_starts):
            concat_end = concat_start + window_sec
            window_id = f"{participant_id}_w{window_index:04d}"
            source_mappings = map_window_to_sources(intervals, concat_start, concat_end)
            if not source_mappings:
                participant_has_mapping_issue = True
                quality_report.append(
                    {
                        "participant_id": participant_id,
                        "official_split": clean_value(row.get("official_split")),
                        "participant_speech_duration_sec": format_float(participant_speech_duration),
                        "total_mapped_window_source_duration_sec": format_float(mapped_duration_total),
                        "num_windows": len(window_starts),
                        "num_source_intervals": participant_source_interval_count,
                        "has_audio": int(has_audio),
                        "has_transcript": int(has_transcript),
                        "issue_type": "window_without_source_interval",
                        "details": f"Window {window_id} has no mapped participant source interval.",
                    }
                )

            clean_windows.append(
                {
                    "participant_id": participant_id,
                    "window_id": window_id,
                    "concat_start_sec": format_float(concat_start),
                    "concat_end_sec": format_float(concat_end),
                    "duration_sec": format_float(window_sec),
                    "label_binary": clean_value(row.get("label_binary")),
                    "official_split": clean_value(row.get("official_split")),
                    "num_source_intervals": len(source_mappings),
                    "source_time_mode": SOURCE_TIME_MODE,
                }
            )

            for source_mapping in source_mappings:
                mapped_duration_total += source_mapping["source_duration_sec"]
                mapping_rows.append(
                    {
                        "participant_id": participant_id,
                        "window_id": window_id,
                        "concat_start_sec": format_float(concat_start),
                        "concat_end_sec": format_float(concat_end),
                        "source_interval_index": source_mapping["source_interval_index"],
                        "source_start_sec": format_float(source_mapping["source_start_sec"]),
                        "source_end_sec": format_float(source_mapping["source_end_sec"]),
                        "source_duration_sec": format_float(source_mapping["source_duration_sec"]),
                        "official_split": clean_value(row.get("official_split")),
                        "label_binary": clean_value(row.get("label_binary")),
                    }
                )

        clean_row = dict(row)
        clean_row["participant_speech_duration_sec"] = format_float(participant_speech_duration)
        clean_row["num_windows"] = len(window_starts)
        clean_row["has_audio"] = int(has_audio)
        clean_row["has_transcript"] = int(has_transcript)
        clean_row["source_time_mode"] = SOURCE_TIME_MODE
        clean_participants.append(clean_row)

        participant_issues: list[str] = []
        if not has_audio:
            participant_issues.append("missing_audio")
        if not has_transcript:
            participant_issues.append("missing_transcript")
        if not window_starts:
            participant_issues.append("zero_windows")
        if interval_issues:
            participant_issues.extend(interval_issues)
        if participant_has_mapping_issue:
            participant_issues.append("window_mapping_issue")

        if participant_issues:
            for issue in participant_issues:
                quality_report.append(
                    {
                        "participant_id": participant_id,
                        "official_split": clean_value(row.get("official_split")),
                        "participant_speech_duration_sec": format_float(participant_speech_duration),
                        "total_mapped_window_source_duration_sec": format_float(mapped_duration_total),
                        "num_windows": len(window_starts),
                        "num_source_intervals": participant_source_interval_count,
                        "has_audio": int(has_audio),
                        "has_transcript": int(has_transcript),
                        "issue_type": issue,
                        "details": "Participant failed one or more clean mapping quality checks.",
                    }
                )
        else:
            quality_report.append(
                {
                    "participant_id": participant_id,
                    "official_split": clean_value(row.get("official_split")),
                    "participant_speech_duration_sec": format_float(participant_speech_duration),
                    "total_mapped_window_source_duration_sec": format_float(mapped_duration_total),
                    "num_windows": len(window_starts),
                    "num_source_intervals": participant_source_interval_count,
                    "has_audio": int(has_audio),
                    "has_transcript": int(has_transcript),
                    "issue_type": "ok",
                    "details": "All clean participant-speech windows have at least one mapped participant source interval.",
                }
            )

    split_summary = build_split_summary(clean_participants)
    return {
        "clean_participants": clean_participants,
        "clean_windows": clean_windows,
        "mapping_rows": mapping_rows,
        "cleaning_report": cleaning_report,
        "quality_report": quality_report,
        "split_summary": split_summary,
    }


def build_split_summary(clean_participants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean_participants:
        rows_by_split[clean_value(row.get("official_split"))].append(row)

    summary: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        rows = rows_by_split.get(split, [])
        labels = Counter(clean_value(row.get("label_binary")) for row in rows)
        windows = sum(int(row.get("num_windows") or 0) for row in rows)
        participants = len(rows)
        avg_windows = windows / participants if participants else 0.0
        summary.append(
            {
                "split": split,
                "participants": participants,
                "non_depressed": labels.get("0", 0),
                "depressed": labels.get("1", 0),
                "windows": windows,
                "avg_windows_per_participant": format_float(avg_windows, 3),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    output_dir = Path(config["output_dir"])
    outputs = build_clean_outputs(config)

    output_paths = {
        "participant_manifest_clean": output_dir / "daic_phase1_participant_manifest_clean.csv",
        "window_manifest_clean": output_dir / "daic_phase1_window_manifest_clean.csv",
        "split_summary_clean": output_dir / "daic_phase1_split_summary_clean.csv",
        "manifest_cleaning_report": output_dir / "manifest_cleaning_report.csv",
        "participant_window_mapping": output_dir / "participant_window_mapping.csv",
        "participant_speech_mapping_quality_report": output_dir / "participant_speech_mapping_quality_report.csv",
    }

    write_csv(output_paths["participant_manifest_clean"], outputs["clean_participants"], CLEAN_PARTICIPANT_FIELDS)
    write_csv(output_paths["window_manifest_clean"], outputs["clean_windows"], CLEAN_WINDOW_FIELDS)
    write_csv(output_paths["split_summary_clean"], outputs["split_summary"], CLEAN_SPLIT_SUMMARY_FIELDS)
    write_csv(output_paths["manifest_cleaning_report"], outputs["cleaning_report"], CLEANING_REPORT_FIELDS)
    write_csv(output_paths["participant_window_mapping"], outputs["mapping_rows"], WINDOW_MAPPING_FIELDS)
    write_csv(
        output_paths["participant_speech_mapping_quality_report"],
        outputs["quality_report"],
        MAPPING_QUALITY_FIELDS,
    )

    print("Clean Phase 1 manifests created. No WavLM extraction or model training was run.")
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
