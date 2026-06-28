"""Phase 2A Androids Corpus preparation and compatibility audit.

This script performs dataset inventory, manifest generation, fold/timing audits,
and DAIC-to-Android compatibility analysis only. It does not train models,
extract WavLM embeddings, apply frozen DAIC models, tune thresholds, or generate
depression predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_DATASET = "Androids-Corpus"
RANDOM_NOTE = "No modeling or prediction was performed in Phase 2A."
CONDITION_TO_LABEL = {"HC": 0, "PT": 1}
CONDITION_CHAR_TO_CONDITION = {"C": "HC", "P": "PT"}
CONDITION_SOURCE = {"HC": "HC_control", "PT": "PT_depression_patient"}
TASKS = {"reading": ("Reading-Task", "audio"), "interview": ("Interview-Task", "audio")}
PHASE1_TIMING_FEATURES: list[tuple[str, str]] = [
    ("interview_duration_sec", "maximum valid stop_time - minimum valid start_time"),
    ("participant_speech_duration_sec", "sum duration of valid Participant intervals"),
    ("participant_speech_ratio", "participant_speech_duration_sec / interview_duration_sec"),
    ("participant_turn_count", "count of valid Participant intervals"),
    ("participant_turns_per_minute", "participant_turn_count / interview_duration_minutes"),
    ("turn_duration_mean", "mean duration of Participant turns"),
    ("turn_duration_median", "median duration of Participant turns"),
    ("turn_duration_std", "population standard deviation of Participant turn durations"),
    ("turn_duration_iqr", "IQR of Participant turn durations"),
    ("turn_duration_p10", "10th percentile of Participant turn durations"),
    ("turn_duration_p90", "90th percentile of Participant turn durations"),
    ("turn_duration_min", "minimum Participant turn duration"),
    ("turn_duration_max", "maximum Participant turn duration"),
    ("response_latency_count", "count of non-negative latencies from interviewer stop to following Participant start"),
    ("response_latency_mean", "mean non-negative interviewer-to-participant latency"),
    ("response_latency_median", "median non-negative interviewer-to-participant latency"),
    ("response_latency_std", "population standard deviation of non-negative interviewer-to-participant latencies"),
    ("response_latency_iqr", "IQR of non-negative interviewer-to-participant latencies"),
    ("response_latency_p10", "10th percentile of non-negative interviewer-to-participant latencies"),
    ("response_latency_p90", "90th percentile of non-negative interviewer-to-participant latencies"),
    ("response_latency_max", "maximum non-negative interviewer-to-participant latency"),
    ("participant_gap_count", "count of positive gaps between consecutive Participant intervals"),
    ("participant_gap_mean", "mean positive gap between consecutive Participant intervals"),
    ("participant_gap_median", "median positive gap between consecutive Participant intervals"),
    ("participant_gap_std", "population standard deviation of positive gaps between consecutive Participant intervals"),
    ("participant_gap_iqr", "IQR of positive gaps between consecutive Participant intervals"),
    ("participant_gap_p90", "90th percentile of positive gaps between consecutive Participant intervals"),
    ("participant_gap_max", "maximum positive gap between consecutive Participant intervals"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2A Androids Corpus preparation audit.")
    parser.add_argument("--android-root", default="", help="Optional Androids-Corpus root containing Androids.conf.")
    parser.add_argument("--run-root", default="outputs/phase2_android/phase2a_preparation_runs")
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def path_relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ignored_path(path: Path) -> bool:
    parts = set(path.parts)
    name = path.name
    return "__MACOSX" in parts or name == ".DS_Store" or name.startswith("._") or path.suffix.lower() in {".zip", ".tar", ".gz", ".rar", ".7z"}


def locate_android_root(root: Path, requested: str) -> Path:
    if requested:
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = root / candidate
        if (candidate / "Androids.conf").exists():
            return candidate
        raise FileNotFoundError(f"Requested Android root does not contain Androids.conf: {candidate}")
    candidates = [path.parent for path in root.rglob("Androids.conf") if not ignored_path(path)]
    if not candidates:
        raise FileNotFoundError("Could not locate Androids.conf under the project workspace.")
    candidates = sorted(candidates, key=lambda path: (len(path.parts), str(path)))
    return candidates[0]


def audio_info(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = int(handle.getframerate())
            channels = int(handle.getnchannels())
            frames = int(handle.getnframes())
            duration = float(frames / sample_rate) if sample_rate else 0.0
        return {"audio_status": "readable", "sample_rate": sample_rate, "num_channels": channels, "duration_sec": duration, "error_message": ""}
    except Exception as exc:
        return {"audio_status": "unreadable", "sample_rate": "", "num_channels": "", "duration_sec": "", "error_message": str(exc)}


FILENAME_RE = re.compile(r"^(?P<num>\d{2})_(?P<condition_char>[CP])(?P<gender>[FMX])(?P<age>\d{2}|XX)_(?P<education>[1-4xX])$")


def parse_recording_id(recording_id: str) -> dict[str, Any]:
    match = FILENAME_RE.match(recording_id)
    if not match:
        return {
            "numeric_id": "",
            "condition_from_filename": "",
            "gender": "",
            "age": "",
            "education": "",
            "participant_id": "",
            "parse_status": "malformed_filename",
            "metadata_warnings": "malformed_filename",
        }
    groups = match.groupdict()
    condition = CONDITION_CHAR_TO_CONDITION[groups["condition_char"]]
    education = groups["education"]
    warnings: list[str] = []
    if groups["gender"] == "X":
        warnings.append("unknown_gender")
    if groups["age"] == "XX":
        warnings.append("unknown_age")
    if education.lower() == "x":
        warnings.append("missing_or_unknown_education")
    return {
        "numeric_id": groups["num"],
        "condition_from_filename": condition,
        "gender": groups["gender"],
        "age": groups["age"] if groups["age"] != "XX" else "",
        "education": "" if education.lower() == "x" else education,
        "participant_id": f"{condition}_{groups['num']}",
        "parse_status": "parsed",
        "metadata_warnings": ";".join(warnings),
    }


def clip_parent_and_order(path: Path) -> tuple[str, str]:
    parent_id = path.parent.name
    stem = path.stem
    prefix = f"{parent_id}_"
    if stem.startswith(prefix):
        return parent_id, stem[len(prefix) :]
    match = re.match(r"^(.*)_(\d+)$", stem)
    return (match.group(1), match.group(2)) if match else (parent_id, "")


def sorted_participants(values: set[str]) -> list[str]:
    def key(pid: str) -> tuple[str, int, str]:
        condition, _, number = pid.partition("_")
        try:
            return (condition, int(number), pid)
        except ValueError:
            return (condition, 10**9, pid)

    return sorted(values, key=key)


def collect_full_recordings(root: Path, project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    recording_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    recording_by_task_id: dict[tuple[str, str], dict[str, Any]] = {}
    for task, path_parts in TASKS.items():
        task_root = root.joinpath(*path_parts)
        for condition in ("HC", "PT"):
            condition_root = task_root / condition
            for path in sorted(condition_root.glob("*.wav")):
                if ignored_path(path):
                    continue
                recording_id = path.stem
                parsed = parse_recording_id(recording_id)
                info = audio_info(path)
                label = CONDITION_TO_LABEL[condition]
                row = {
                    "dataset": PROJECT_DATASET,
                    "participant_id": parsed["participant_id"],
                    "recording_id": recording_id,
                    "task": task,
                    "condition": condition,
                    "label_binary": label,
                    "audio_path": path_relative_to_root(path, project_root),
                    "duration_sec": info["duration_sec"],
                    "sample_rate": info["sample_rate"],
                    "num_channels": info["num_channels"],
                    "fold": "",
                    "filename_gender": parsed["gender"],
                    "filename_age": parsed["age"],
                    "filename_education": parsed["education"],
                    "audio_status": info["audio_status"],
                    "numeric_id": parsed["numeric_id"],
                    "condition_from_filename": parsed["condition_from_filename"],
                    "parse_status": parsed["parse_status"],
                    "metadata_warnings": parsed["metadata_warnings"],
                    "error_message": info["error_message"],
                }
                recording_rows.append(row)
                recording_by_task_id[(task, recording_id)] = row
                inventory_rows.append(
                    {
                        **row,
                        "file_category": "full_recording",
                        "file_path": row["audio_path"],
                        "file_name": path.name,
                        "file_size_bytes": path.stat().st_size,
                    }
                )
    return recording_rows, inventory_rows, recording_by_task_id


def collect_clips(root: Path, project_root: Path, interview_recording_by_id: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    clip_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    clip_count_by_parent: Counter[str] = Counter()
    clip_root = root / "Interview-Task" / "audio_clip"
    for path in sorted(clip_root.rglob("*.wav")):
        if ignored_path(path):
            continue
        parent_id, order_text = clip_parent_and_order(path)
        parent = interview_recording_by_id.get(parent_id)
        parsed = parse_recording_id(parent_id)
        condition = parent["condition"] if parent else parsed["condition_from_filename"]
        label = CONDITION_TO_LABEL.get(condition, "")
        info = audio_info(path)
        try:
            clip_order: Any = int(order_text)
        except ValueError:
            clip_order = order_text
        mapping_status = "mapped_to_single_parent" if parent else "missing_parent_interview"
        row = {
            "dataset": PROJECT_DATASET,
            "participant_id": parsed["participant_id"],
            "clip_id": path.stem,
            "parent_interview_path": parent["audio_path"] if parent else "",
            "clip_audio_path": path_relative_to_root(path, project_root),
            "condition": condition,
            "label_binary": label,
            "duration_sec": info["duration_sec"],
            "sample_rate": info["sample_rate"],
            "num_channels": info["num_channels"],
            "fold": "",
            "clip_order": clip_order,
            "mapping_status": mapping_status,
            "parent_recording_id": parent_id,
            "audio_status": info["audio_status"],
            "error_message": info["error_message"],
        }
        clip_rows.append(row)
        clip_count_by_parent[parent_id] += 1
        inventory_rows.append(
            {
                **row,
                "file_category": "interview_clip",
                "task": "interview_clip",
                "recording_id": parent_id,
                "audio_path": row["clip_audio_path"],
                "file_path": row["clip_audio_path"],
                "file_name": path.name,
                "file_size_bytes": path.stat().st_size,
                "filename_gender": parsed["gender"],
                "filename_age": parsed["age"],
                "filename_education": parsed["education"],
                "numeric_id": parsed["numeric_id"],
                "condition_from_filename": parsed["condition_from_filename"],
                "parse_status": parsed["parse_status"],
                "metadata_warnings": parsed["metadata_warnings"],
            }
        )
    return clip_rows, inventory_rows, dict(clip_count_by_parent)


def parse_folds(root: Path, recording_by_task_id: dict[tuple[str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], str], dict[str, set[str]]]:
    fold_path = root / "fold-lists.csv"
    rows = list(csv.reader(fold_path.open("r", encoding="utf-8-sig", newline="")))
    if len(rows) < 2:
        raise RuntimeError("fold-lists.csv is missing expected header rows.")
    header = rows[1]
    read_cols = [(idx, clean(value)) for idx, value in enumerate(header[:5]) if clean(value).startswith("fold")]
    interview_cols = [(idx, clean(value)) for idx, value in enumerate(header) if idx >= 7 and clean(value).startswith("fold")]
    assignments: list[dict[str, Any]] = []
    fold_by_task_recording: dict[tuple[str, str], str] = {}
    folds_by_participant: dict[str, set[str]] = defaultdict(set)
    for data_row in rows[2:]:
        for task, cols in [("reading", read_cols), ("interview", interview_cols)]:
            for idx, fold in cols:
                entry = clean(data_row[idx] if idx < len(data_row) else "").strip("'\"")
                if not entry:
                    continue
                recording = recording_by_task_id.get((task, entry))
                if recording:
                    status = "resolved"
                    fold_by_task_recording[(task, entry)] = fold
                    folds_by_participant[recording["participant_id"]].add(fold)
                else:
                    status = "listed_recording_missing_from_local_data"
                assignments.append(
                    {
                        "dataset": PROJECT_DATASET,
                        "task": task,
                        "fold": fold,
                        "recording_id": entry,
                        "participant_id": recording["participant_id"] if recording else parse_recording_id(entry)["participant_id"],
                        "condition": recording["condition"] if recording else parse_recording_id(entry)["condition_from_filename"],
                        "label_binary": recording["label_binary"] if recording else "",
                        "assignment_status": status,
                    }
                )
    for key, recording in recording_by_task_id.items():
        task, recording_id = key
        if key not in fold_by_task_recording:
            assignments.append(
                {
                    "dataset": PROJECT_DATASET,
                    "task": task,
                    "fold": "",
                    "recording_id": recording_id,
                    "participant_id": recording["participant_id"],
                    "condition": recording["condition"],
                    "label_binary": recording["label_binary"],
                    "assignment_status": "local_recording_missing_from_fold_file",
                }
            )
    quality_rows: list[dict[str, Any]] = []
    for participant_id, folds in sorted(folds_by_participant.items()):
        if len(folds) > 1:
            quality_rows.append(
                {
                    "severity": "error",
                    "issue_type": "participant_crosses_folds",
                    "participant_id": participant_id,
                    "details": f"Participant appears in folds: {sorted(folds)}",
                }
            )
    for row in assignments:
        if row["assignment_status"] != "resolved":
            quality_rows.append(
                {
                    "severity": "warning" if row["assignment_status"] == "local_recording_missing_from_fold_file" else "error",
                    "issue_type": row["assignment_status"],
                    "participant_id": row["participant_id"],
                    "recording_id": row["recording_id"],
                    "details": f"{row['task']} {row['recording_id']} {row['assignment_status']}",
                }
            )
    if not quality_rows:
        quality_rows.append({"severity": "info", "issue_type": "all_fold_checks_passed", "details": "All fold integrity checks passed."})
    return assignments, quality_rows, fold_by_task_recording, folds_by_participant


def apply_folds(recording_rows: list[dict[str, Any]], clip_rows: list[dict[str, Any]], fold_by_task_recording: dict[tuple[str, str], str]) -> None:
    interview_fold_by_recording = {recording: fold for (task, recording), fold in fold_by_task_recording.items() if task == "interview"}
    for row in recording_rows:
        row["fold"] = fold_by_task_recording.get((row["task"], row["recording_id"]), "")
    for row in clip_rows:
        row["fold"] = interview_fold_by_recording.get(row["parent_recording_id"], "")


def participant_manifest(recording_rows: list[dict[str, Any]], clip_rows: list[dict[str, Any]], folds_by_participant: dict[str, set[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_participant: dict[str, dict[str, Any]] = {}
    warnings: list[dict[str, Any]] = []
    for row in recording_rows:
        pid = row["participant_id"]
        participant = by_participant.setdefault(
            pid,
            {
                "dataset": PROJECT_DATASET,
                "participant_id": pid,
                "numeric_id": row["numeric_id"],
                "condition": row["condition"],
                "label_binary": row["label_binary"],
                "label_source": CONDITION_SOURCE[row["condition"]],
                "gender": row["filename_gender"],
                "age": row["filename_age"],
                "education": row["filename_education"],
                "has_reading_audio": 0,
                "reading_audio_path": "",
                "reading_duration_sec": "",
                "has_interview_audio": 0,
                "interview_audio_path": "",
                "interview_duration_sec": "",
                "num_interview_clips": 0,
                "fold": "",
                "sample_rate_reading": "",
                "sample_rate_interview": "",
                "channels_reading": "",
                "channels_interview": "",
                "data_quality_status": "ok",
            },
        )
        for field, filename_field in [("gender", "filename_gender"), ("age", "filename_age"), ("education", "filename_education")]:
            if participant[field] and row[filename_field] and str(participant[field]) != str(row[filename_field]):
                participant["data_quality_status"] = "metadata_conflict"
                warnings.append({"severity": "warning", "issue_type": f"participant_{field}_conflict", "participant_id": pid, "details": f"{participant[field]} vs {row[filename_field]}"})
        if row["task"] == "reading":
            participant.update(
                {
                    "has_reading_audio": 1,
                    "reading_audio_path": row["audio_path"],
                    "reading_duration_sec": row["duration_sec"],
                    "sample_rate_reading": row["sample_rate"],
                    "channels_reading": row["num_channels"],
                }
            )
        elif row["task"] == "interview":
            participant.update(
                {
                    "has_interview_audio": 1,
                    "interview_audio_path": row["audio_path"],
                    "interview_duration_sec": row["duration_sec"],
                    "sample_rate_interview": row["sample_rate"],
                    "channels_interview": row["num_channels"],
                }
            )
        if row["audio_status"] != "readable":
            participant["data_quality_status"] = "audio_issue"
    clip_counts = Counter(row["participant_id"] for row in clip_rows if row["mapping_status"] == "mapped_to_single_parent")
    for participant_id, count in clip_counts.items():
        by_participant.setdefault(participant_id, {}).update({"num_interview_clips": count})
    for participant_id, participant in by_participant.items():
        folds = folds_by_participant.get(participant_id, set())
        if len(folds) == 1:
            participant["fold"] = next(iter(folds))
        elif len(folds) > 1:
            participant["fold"] = "conflict:" + "|".join(sorted(folds))
            participant["data_quality_status"] = "fold_conflict"
        else:
            participant["fold"] = ""
            participant["data_quality_status"] = "missing_fold"
    return [by_participant[pid] for pid in sorted_participants(set(by_participant))], warnings


def parse_timing(root: Path, project_root: Path, interview_recording_by_id: dict[str, dict[str, Any]], clip_count_by_parent: dict[str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, list[tuple[float, float]]]]:
    path = root / "interview_timedata.csv"
    manifest: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    intervals_by_recording: dict[str, list[tuple[float, float]]] = {}
    rows = list(csv.reader(path.open("r", encoding="utf-8-sig", newline="")))
    seen_intervals: set[tuple[str, float, float]] = set()
    for line_number, row in enumerate(rows, start=1):
        recording_id = clean(row[0]).lstrip("\ufeff") if row else ""
        recording = interview_recording_by_id.get(recording_id)
        parsed = parse_recording_id(recording_id)
        values: list[float] = []
        for value in row[1:]:
            text = clean(value)
            if not text:
                continue
            try:
                values.append(float(text))
            except ValueError:
                quality.append({"severity": "error", "issue_type": "invalid_timing_value", "recording_id": recording_id, "details": f"Line {line_number}: {text}"})
        if len(values) % 2:
            quality.append({"severity": "error", "issue_type": "odd_number_of_timestamps", "recording_id": recording_id, "details": f"Line {line_number} has {len(values)} values."})
        previous_end: float | None = None
        pairs = list(zip(values[0::2], values[1::2]))
        intervals_by_recording[recording_id] = pairs
        duration = float(recording["duration_sec"]) if recording and recording["duration_sec"] != "" else None
        for interval_index, (start, end) in enumerate(pairs, start=1):
            status = "valid"
            if end <= start:
                status = "invalid_start_end"
                quality.append({"severity": "error", "issue_type": "timing_start_not_less_than_end", "recording_id": recording_id, "details": f"{start}-{end}"})
            if duration is not None and (start < 0 or end > duration + 0.05):
                status = "outside_recording_duration"
                quality.append({"severity": "error", "issue_type": "timing_outside_recording_duration", "recording_id": recording_id, "details": f"{start}-{end} duration={duration}"})
            if (recording_id, start, end) in seen_intervals:
                quality.append({"severity": "warning", "issue_type": "duplicate_timing_interval", "recording_id": recording_id, "details": f"{start}-{end}"})
            seen_intervals.add((recording_id, start, end))
            if previous_end is not None and start < previous_end:
                quality.append({"severity": "warning", "issue_type": "overlapping_timing_intervals", "recording_id": recording_id, "details": f"interval {interval_index}"})
            previous_end = max(previous_end, end) if previous_end is not None else end
            manifest.append(
                {
                    "dataset": PROJECT_DATASET,
                    "participant_id": recording["participant_id"] if recording else parsed["participant_id"],
                    "recording_id": recording_id,
                    "interval_index": interval_index,
                    "start_sec": start,
                    "end_sec": end,
                    "duration_sec": end - start,
                    "full_interview_audio_path": recording["audio_path"] if recording else "",
                    "within_recording_duration": bool(duration is not None and start >= 0 and end <= duration + 0.05),
                    "timing_status": status if recording else "unresolved_recording_id",
                }
            )
        if not recording:
            quality.append({"severity": "error", "issue_type": "unresolved_timing_recording_id", "recording_id": recording_id, "details": "No local interview recording."})
        if recording_id in clip_count_by_parent and clip_count_by_parent[recording_id] != len(pairs):
            quality.append({"severity": "warning", "issue_type": "clip_count_timing_interval_count_mismatch", "recording_id": recording_id, "details": f"clips={clip_count_by_parent[recording_id]} intervals={len(pairs)}"})
    timing_ids = set(intervals_by_recording)
    interview_ids = set(interview_recording_by_id)
    for recording_id in sorted(interview_ids - timing_ids):
        quality.append({"severity": "error", "issue_type": "interview_missing_timing_row", "recording_id": recording_id, "details": "No timing row."})
    if not quality:
        quality.append({"severity": "info", "issue_type": "all_timing_checks_passed", "details": "All timing rows passed structural checks."})
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_path": path_relative_to_root(path, project_root),
        "column_names": "No header row; first column is recording identifier, following columns are alternating start/end timestamps.",
        "participant_identifier_format": "recording id nn_XGmm_t, where X is condition character C/P",
        "time_units": "seconds, inferred from README and clip-duration alignment",
        "identifies_interviewer_vs_participant": False,
        "entries_identify": "participant speech turns/clips only; no explicit interviewer rows or speaker column",
        "timestamps_refer_to_full_interview_recordings": True,
        "timing_rows": len(rows),
        "timing_intervals": len(manifest),
        "can_reproduce_participant_speech_turns": True,
        "can_reproduce_phase1_response_latency_features": False,
        "phase1_timing_feature_compatibility": "partial; response-latency features unavailable and interview-duration definition differs",
    }
    return manifest, quality, metadata, intervals_by_recording


def summarize_numeric(values: list[float]) -> dict[str, Any]:
    values = [float(value) for value in values if value != ""]
    if not values:
        return {"count": 0, "min": "", "max": "", "mean": "", "median": ""}
    return {"count": len(values), "min": min(values), "max": max(values), "mean": statistics.fmean(values), "median": statistics.median(values)}


def inventory_summary(participants: list[dict[str, Any]], recording_rows: list[dict[str, Any]], clip_rows: list[dict[str, Any]], inventory_rows: list[dict[str, Any]], data_quality_rows: list[dict[str, Any]]) -> dict[str, Any]:
    participant_ids = {row["participant_id"] for row in participants}
    hc = [row for row in participants if row["condition"] == "HC"]
    pt = [row for row in participants if row["condition"] == "PT"]
    reading = [row for row in recording_rows if row["task"] == "reading"]
    interview = [row for row in recording_rows if row["task"] == "interview"]
    both = [row for row in participants if row["has_reading_audio"] and row["has_interview_audio"]]
    one_task = [row for row in participants if bool(row["has_reading_audio"]) ^ bool(row["has_interview_audio"])]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_full_interview_recordings": len(interview),
        "num_read_speech_recordings": len(reading),
        "num_interview_clips": len(clip_rows),
        "num_unique_participants": len(participant_ids),
        "num_healthy_controls": len(hc),
        "num_depression_patients": len(pt),
        "number_by_task_and_condition": {
            "reading": dict(Counter(row["condition"] for row in reading)),
            "interview": dict(Counter(row["condition"] for row in interview)),
            "interview_clip": dict(Counter(row["condition"] for row in clip_rows)),
        },
        "participants_with_both_tasks": len(both),
        "participants_with_only_one_task": len(one_task),
        "sample_rate_distribution": dict(Counter(str(row["sample_rate"]) for row in inventory_rows if row.get("sample_rate") != "")),
        "channel_count_distribution": dict(Counter(str(row["num_channels"]) for row in inventory_rows if row.get("num_channels") != "")),
        "duration_distribution_all_audio": summarize_numeric([row["duration_sec"] for row in inventory_rows if row.get("duration_sec") != ""]),
        "duration_distribution_reading": summarize_numeric([row["duration_sec"] for row in reading if row.get("duration_sec") != ""]),
        "duration_distribution_full_interview": summarize_numeric([row["duration_sec"] for row in interview if row.get("duration_sec") != ""]),
        "duration_distribution_interview_clips": summarize_numeric([row["duration_sec"] for row in clip_rows if row.get("duration_sec") != ""]),
        "corrupted_or_unreadable_audio_files": sum(1 for row in inventory_rows if row.get("audio_status") != "readable"),
        "data_quality_issue_count": len([row for row in data_quality_rows if row.get("severity") in {"warning", "error"}]),
    }


def fold_summary(assignments: list[dict[str, Any]], clip_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in sorted({row["fold"] for row in assignments if row["fold"]}):
        resolved = [row for row in assignments if row["fold"] == fold and row["assignment_status"] == "resolved"]
        participants = {row["participant_id"] for row in resolved}
        clips = [row for row in clip_rows if row["fold"] == fold]
        rows.append(
            {
                "fold": fold,
                "participants": len(participants),
                "controls": len({row["participant_id"] for row in resolved if row["condition"] == "HC"}),
                "patients": len({row["participant_id"] for row in resolved if row["condition"] == "PT"}),
                "reading_recordings": sum(1 for row in resolved if row["task"] == "reading"),
                "interview_recordings": sum(1 for row in resolved if row["task"] == "interview"),
                "interview_clips": len(clips),
            }
        )
    return rows


def compatibility_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unavailable = {name for name, _ in PHASE1_TIMING_FEATURES if name.startswith("response_latency")}
    approximate = {"interview_duration_sec", "participant_speech_ratio", "participant_turns_per_minute"}
    for feature, definition in PHASE1_TIMING_FEATURES:
        if feature in unavailable:
            classification = "unavailable"
            source = "No interviewer stop/start intervals in Android timing file."
            action = "Do not apply frozen Candidate B directly; feature is unavailable."
            diff = "Android timing has participant turn/clip timestamps only and lacks preceding interviewer intervals."
            direct = False
            approx = False
            not_rep = True
        elif feature in approximate:
            classification = "approximately compatible"
            source = "Full interview audio duration plus participant turn timestamps."
            action = "Treat as definition-shifted; not valid for frozen Candidate B without compatibility resolution."
            diff = "DAIC interview duration uses transcript timeline across speakers; Android lacks interviewer transcript intervals."
            direct = False
            approx = True
            not_rep = False
        else:
            classification = "exactly compatible"
            source = "interview_timedata.csv participant turn/clip start/end pairs."
            action = "Can be derived if Android timing rows are accepted as authoritative participant turns."
            diff = "No material difference for participant-turn duration/gap summary, assuming timing rows are participant turns."
            direct = True
            approx = False
            not_rep = False
        rows.append(
            {
                "phase1_feature": feature,
                "phase1_definition": definition,
                "android_source": source,
                "directly_reproducible": direct,
                "approximately_reproducible": approx,
                "not_reproducible": not_rep,
                "compatibility_classification": classification,
                "definition_difference": diff,
                "recommended_action": action,
            }
        )
    return rows


def wavlm_compatibility(participants: list[dict[str, Any]], recording_rows: list[dict[str, Any]], clip_rows: list[dict[str, Any]], intervals_by_recording: dict[str, list[tuple[float, float]]], timing_quality: list[dict[str, Any]]) -> dict[str, Any]:
    reading_durations = [float(row["reading_duration_sec"]) for row in participants if row["reading_duration_sec"] != ""]
    interview_full_durations = [float(row["interview_duration_sec"]) for row in participants if row["interview_duration_sec"] != ""]
    speech_duration_by_recording = {recording_id: sum(end - start for start, end in intervals) for recording_id, intervals in intervals_by_recording.items()}
    speech_duration_by_participant: dict[str, float] = {}
    for row in recording_rows:
        if row["task"] == "interview":
            speech_duration_by_participant[row["participant_id"]] = speech_duration_by_recording.get(row["recording_id"], 0.0)
    participants_lt_10_interview = [pid for pid, duration in speech_duration_by_participant.items() if duration < 10.0]
    participants_lt_10_reading = [row["participant_id"] for row in participants if row["reading_duration_sec"] != "" and float(row["reading_duration_sec"]) < 10.0]
    timing_errors = [row for row in timing_quality if row.get("severity") == "error"]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_A": "Frozen DAIC WavLM layer_09 participant mean model can be prepared for zero-shot evaluation later, but Phase 2A does not extract embeddings or apply the model.",
        "language_difference": "DAIC-WOZ is English; Androids Corpus recordings are Italian.",
        "task_difference": "DAIC uses spontaneous clinical interview speech; Android includes separate interview and read-speech tasks.",
        "participant_turn_concatenation": "Android interview participant speech can likely be reconstructed from interview_timedata.csv and/or audio_clip files, but this must remain separate from reading speech.",
        "read_speech_task": "Reading recordings are participant speech but are task-shifted and should be evaluated separately.",
        "sample_rate_distribution": dict(Counter(str(row["sample_rate"]) for row in recording_rows if row.get("sample_rate") != "")),
        "channel_format_distribution": dict(Counter(str(row["num_channels"]) for row in recording_rows if row.get("num_channels") != "")),
        "recording_conditions": "Different corpus, language, elicitation protocol, and likely recording setup from DAIC-WOZ.",
        "reading_duration_sec": summarize_numeric(reading_durations),
        "full_interview_duration_sec": summarize_numeric(interview_full_durations),
        "participant_speech_duration_from_timing_sec": summarize_numeric(list(speech_duration_by_participant.values())),
        "ten_second_windows_50_overlap_feasible_reading": len(participants_lt_10_reading) == 0,
        "ten_second_windows_50_overlap_feasible_interview_participant_speech": len(participants_lt_10_interview) == 0,
        "participants_with_less_than_10_sec_reading": participants_lt_10_reading,
        "participants_with_less_than_10_sec_interview_participant_speech": participants_lt_10_interview,
        "participant_only_interview_speech_reconstruction_feasible": len(timing_errors) == 0,
        "participant_only_interview_speech_reconstruction_basis": "interview_timedata.csv start/end pairs and audio_clip turn files",
        "combine_reading_and_interview": False,
        "candidate_A_preprocessing_can_be_reproduced": len(participants_lt_10_reading) == 0 and len(participants_lt_10_interview) == 0 and len(timing_errors) == 0,
        "phase2a_predictions_or_embeddings_created": False,
    }


def protocols_markdown(path: Path, timing_compatible: bool) -> None:
    timing_status = "valid_to_prepare_later" if timing_compatible else "not_valid_without_feature_redefinition"
    text = f"""# Phase 2 Evaluation Protocols

Phase 2A prepared protocols only. It did not extract WavLM embeddings, train models, apply frozen DAIC models, calculate Android prediction metrics, or tune thresholds.

## Protocol A: Zero-Shot DAIC WavLM To Android Interview

Apply frozen DAIC Candidate A without retraining to Android interview participant speech only. Use identical 10-second windows, 50% overlap, WavLM layer 9, temporal mean pooling, participant mean aggregation, and frozen DAIC classifier. Report probability-based metrics. Threshold-based results should be treated cautiously because Android clinical-group labels and prevalence differ from DAIC PHQ-8 labels.

## Protocol B: Zero-Shot DAIC WavLM To Android Reading

Process reading speech separately. Do not claim equivalence to DAIC interview speech. Use as a task-shift robustness experiment.

## Protocol C: Zero-Shot DAIC Timing Model To Android Interview

Status: `{timing_status}`.

The frozen DAIC timing model may be applied only if all 28 timing features can be reproduced with matching definitions, units, ordering, and missing-value behavior. Phase 2A found Android response-latency features unavailable because interviewer turns are not represented in `interview_timedata.csv`; therefore direct Candidate B transfer is invalid unless future work formally redefines features and does not pretend to use the frozen DAIC timing model unchanged.

## Protocol D: Within-Android WavLM Baseline

Use Android's official five-fold participant-independent protocol. Train and evaluate exclusively within Android. Keep reading and interview tasks separate. This is a future baseline and is not implemented in Phase 2A.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    android_root = locate_android_root(project_root, args.android_root)
    timestamp = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (project_root / args.run_root / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    recording_rows, inventory_full, recording_by_task_id = collect_full_recordings(android_root, project_root)
    interview_recording_by_id = {
        recording_id: row for (task, recording_id), row in recording_by_task_id.items() if task == "interview"
    }
    clip_rows, inventory_clips, clip_count_by_parent = collect_clips(android_root, project_root, interview_recording_by_id)
    assignments, fold_quality, fold_by_task_recording, folds_by_participant = parse_folds(android_root, recording_by_task_id)
    apply_folds(recording_rows, clip_rows, fold_by_task_recording)
    participants, participant_quality = participant_manifest(recording_rows, clip_rows, folds_by_participant)
    timing_manifest, timing_quality, timing_metadata, intervals_by_recording = parse_timing(android_root, project_root, interview_recording_by_id, clip_count_by_parent)

    inventory_rows = inventory_full + inventory_clips
    data_quality_rows: list[dict[str, Any]] = []
    for row in inventory_rows:
        if row.get("audio_status") != "readable":
            data_quality_rows.append({"severity": "error", "issue_type": "unreadable_audio", "file_path": row.get("file_path"), "details": row.get("error_message", "")})
        if row.get("parse_status") != "parsed":
            data_quality_rows.append({"severity": "warning", "issue_type": "unexpected_filename_pattern", "file_path": row.get("file_path"), "details": row.get("parse_status")})
        if row.get("metadata_warnings"):
            data_quality_rows.append({"severity": "warning", "issue_type": "filename_metadata_warning", "file_path": row.get("file_path"), "details": row.get("metadata_warnings")})
        if row.get("condition") and row.get("condition_from_filename") and row.get("condition") != row.get("condition_from_filename"):
            data_quality_rows.append({"severity": "error", "issue_type": "condition_conflict", "file_path": row.get("file_path"), "details": f"{row.get('condition')} vs {row.get('condition_from_filename')}"})
    data_quality_rows.extend(participant_quality)
    if not data_quality_rows:
        data_quality_rows.append({"severity": "info", "issue_type": "all_data_quality_checks_passed", "details": "No inventory data-quality issues detected."})

    compatibility_rows = compatibility_table()
    timing_exact_count = sum(1 for row in compatibility_rows if row["compatibility_classification"] == "exactly compatible")
    candidate_b_valid = all(row["compatibility_classification"] == "exactly compatible" for row in compatibility_rows)
    wavlm_report = wavlm_compatibility(participants, recording_rows, clip_rows, intervals_by_recording, timing_quality)

    for row in recording_rows:
        row.pop("numeric_id", None)
        row.pop("condition_from_filename", None)
        row.pop("parse_status", None)
        row.pop("metadata_warnings", None)
        row.pop("error_message", None)

    participant_fields = [
        "dataset",
        "participant_id",
        "numeric_id",
        "condition",
        "label_binary",
        "label_source",
        "gender",
        "age",
        "education",
        "has_reading_audio",
        "reading_audio_path",
        "reading_duration_sec",
        "has_interview_audio",
        "interview_audio_path",
        "interview_duration_sec",
        "num_interview_clips",
        "fold",
        "sample_rate_reading",
        "sample_rate_interview",
        "channels_reading",
        "channels_interview",
        "data_quality_status",
    ]
    recording_fields = [
        "dataset",
        "participant_id",
        "recording_id",
        "task",
        "condition",
        "label_binary",
        "audio_path",
        "duration_sec",
        "sample_rate",
        "num_channels",
        "fold",
        "filename_gender",
        "filename_age",
        "filename_education",
        "audio_status",
    ]
    clip_fields = [
        "dataset",
        "participant_id",
        "clip_id",
        "parent_interview_path",
        "clip_audio_path",
        "condition",
        "label_binary",
        "duration_sec",
        "sample_rate",
        "num_channels",
        "fold",
        "clip_order",
        "mapping_status",
    ]
    write_csv(run_dir / "android_inventory_report.csv", inventory_rows)
    write_json(run_dir / "android_inventory_summary.json", inventory_summary(participants, recording_rows, clip_rows, inventory_rows, data_quality_rows))
    write_csv(run_dir / "android_data_quality_report.csv", data_quality_rows)
    write_csv(run_dir / "android_participant_manifest.csv", participants, participant_fields)
    write_csv(run_dir / "android_recording_manifest.csv", recording_rows, recording_fields)
    write_csv(run_dir / "android_interview_clip_manifest.csv", clip_rows, clip_fields)
    write_csv(run_dir / "android_fold_assignment.csv", assignments)
    write_csv(run_dir / "android_fold_quality_report.csv", fold_quality)
    write_csv(run_dir / "android_fold_summary.csv", fold_summary(assignments, clip_rows))
    write_csv(run_dir / "android_interview_timing_manifest.csv", timing_manifest)
    write_csv(run_dir / "android_interview_timing_quality_report.csv", timing_quality)
    write_json(run_dir / "android_interview_timing_metadata.json", timing_metadata)
    write_csv(run_dir / "daic_android_timing_feature_compatibility.csv", compatibility_rows)
    write_json(run_dir / "daic_android_wavlm_compatibility_report.json", wavlm_report)
    protocols_markdown(run_dir / "PHASE2_EVALUATION_PROTOCOLS.md", candidate_b_valid)

    summary = inventory_summary(participants, recording_rows, clip_rows, inventory_rows, data_quality_rows)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "Phase 2A Android preparation and compatibility audit",
        "android_root": path_relative_to_root(android_root, project_root),
        "phase1_status": "completed_baseline_ready_for_external_validation",
        "phase1_models_modified": False,
        "models_trained": False,
        "predictions_generated": False,
        "wavlm_embeddings_extracted": False,
        "thresholds_tuned": False,
        "labels": {
            "DAIC-WOZ": "PHQ8_Binary derived from questionnaire threshold",
            "Androids-Corpus": "clinical patient-versus-control grouping; HC=0, PT=1",
            "target_equivalence_warning": "Related but not identical clinical targets.",
        },
        "input_hashes": {
            "fold_lists_sha256": sha256_file(android_root / "fold-lists.csv"),
            "interview_timedata_sha256": sha256_file(android_root / "interview_timedata.csv"),
            "androids_conf_sha256": sha256_file(android_root / "Androids.conf"),
            "readme_sha256": sha256_file(android_root / "README.txt"),
        },
        "summary": summary,
        "fold_integrity_status": "passed" if not any(row.get("severity") == "error" for row in fold_quality) else "failed",
        "clip_mapping_status": "passed" if not any(row.get("mapping_status") != "mapped_to_single_parent" for row in clip_rows) else "failed",
        "timing_file_usability": timing_metadata,
        "timing_exactly_compatible_feature_count": timing_exact_count,
        "frozen_candidate_B_direct_transfer_valid": candidate_b_valid,
        "candidate_A_preprocessing_can_be_reproduced": wavlm_report["candidate_A_preprocessing_can_be_reproduced"],
    }
    write_json(run_dir / "phase2a_metadata.json", metadata)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_phase2a_preparation_and_compatibility_audit",
        "summary": summary,
        "fold_quality_issue_count": len([row for row in fold_quality if row.get("severity") in {"warning", "error"}]),
        "timing_quality_issue_count": len([row for row in timing_quality if row.get("severity") in {"warning", "error"}]),
        "timing_exactly_compatible_feature_count": timing_exact_count,
        "timing_feature_total": len(PHASE1_TIMING_FEATURES),
        "frozen_candidate_B_direct_transfer_valid": candidate_b_valid,
        "candidate_A_preprocessing_can_be_reproduced": wavlm_report["candidate_A_preprocessing_can_be_reproduced"],
        "recommended_next_protocol": "Protocol A zero-shot DAIC WavLM to Android interview, after explicit participant-speech reconstruction validation; Protocol B reading as separate task-shift experiment.",
        "stop_condition_confirmation": RANDOM_NOTE,
    }
    write_json(run_dir / "PHASE2A_AUDIT_REPORT.json", report)

    md = [
        "# Phase 2A Androids Corpus Preparation Audit",
        "",
        f"Run directory: `{path_relative_to_root(run_dir, project_root)}`",
        "",
        "## Inventory",
        f"- Valid participant total: `{summary['num_unique_participants']}`",
        f"- Healthy controls: `{summary['num_healthy_controls']}`",
        f"- Depression patients: `{summary['num_depression_patients']}`",
        f"- Reading recordings: `{summary['num_read_speech_recordings']}`",
        f"- Full interview recordings: `{summary['num_full_interview_recordings']}`",
        f"- Interview clips: `{summary['num_interview_clips']}`",
        "",
        "## Labels",
        "DAIC-WOZ uses PHQ8_Binary from questionnaire thresholding. Android uses clinical group status: HC control versus PT depression patient. These are related but not identical targets.",
        "",
        "## Fold And Leakage Audit",
        f"- Fold integrity status: `{metadata['fold_integrity_status']}`",
        "- Participant is the required grouping unit; any participant crossing folds is listed in `android_fold_quality_report.csv`.",
        "- Reading, interview, and clips must remain grouped by canonical participant ID.",
        "",
        "## Timing Compatibility",
        f"- Exactly compatible Phase 1 timing features: `{timing_exact_count}` of `{len(PHASE1_TIMING_FEATURES)}`",
        f"- Frozen Candidate B direct transfer valid: `{candidate_b_valid}`",
        "Android timing contains participant turn/clip timestamps but no interviewer rows, so response-latency features cannot be reproduced directly.",
        "",
        "## WavLM Compatibility",
        f"- Candidate A preprocessing can be reproduced: `{wavlm_report['candidate_A_preprocessing_can_be_reproduced']}`",
        "- Major caveats: Italian language, task shift, corpus/recording-condition shift, and reading versus interview separation.",
        "",
        "## Proposed Next Protocol",
        "Protocol A: zero-shot DAIC WavLM to Android interview after participant-only speech reconstruction validation. Protocol B should keep reading speech separate as a task-shift robustness experiment. Protocol C timing transfer is not valid without feature redefinition.",
        "",
        "No modeling, prediction, threshold tuning, WavLM embedding extraction, or Phase 1 model modification was performed.",
    ]
    (run_dir / "PHASE2A_AUDIT_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("Phase 2A Android preparation complete.")
    print(f"run_dir={path_relative_to_root(run_dir, project_root)}")
    print(f"participants={summary['num_unique_participants']} HC={summary['num_healthy_controls']} PT={summary['num_depression_patients']}")
    print(f"candidate_B_direct_transfer_valid={candidate_b_valid}")
    print(f"candidate_A_preprocessing_can_be_reproduced={wavlm_report['candidate_A_preprocessing_can_be_reproduced']}")
    print("modeling_or_prediction_performed=false")


if __name__ == "__main__":
    main()
