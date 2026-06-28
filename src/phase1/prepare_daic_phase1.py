"""Prepare DAIC-WOZ Phase 1 manifests and label quality reports.

This script intentionally stops before feature extraction or model training.
It creates the data artifacts needed for Tasks 1-6 of the Phase 1 plan:

- participant-level manifest
- window-level manifest over participant-speech-concatenated time
- split summary
- label quality report

Transcript use is limited to timing and speaker segmentation. Transcript text is
not copied into any output and is not used as a model feature.
"""

from __future__ import annotations

import argparse
import csv
import json
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SPLIT_FILES = {
    "train": "train_split_Depression_AVEC2017.csv",
    "dev": "dev_split_Depression_AVEC2017.csv",
    "test": "test_split_Depression_AVEC2017.csv",
    "full_test": "full_test_split.csv",
}


PARTICIPANT_MANIFEST_FIELDS = [
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


WINDOW_MANIFEST_FIELDS = [
    "dataset",
    "participant_id",
    "window_id",
    "audio_path",
    "source_start_sec",
    "source_end_sec",
    "start_sec",
    "end_sec",
    "duration_sec",
    "label_binary",
    "official_split",
    "speaker_source",
    "source_time_mode",
]


SPLIT_SUMMARY_FIELDS = [
    "split",
    "participants",
    "non_depressed",
    "depressed",
    "missing_label",
    "windows",
    "avg_windows_per_participant",
]


LABEL_REPORT_FIELDS = [
    "severity",
    "issue_type",
    "participant_id",
    "file",
    "split",
    "details",
    "phq8_binary",
    "phq8_score",
    "expected_binary",
]


DATA_QUALITY_FIELDS = [
    "severity",
    "issue_type",
    "participant_id",
    "official_split",
    "details",
]


@dataclass(frozen=True)
class AudioInfo:
    duration_sec: float | None
    sample_rate: int | None
    num_channels: int | None


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
        config = json.load(handle)
    return config


def clean_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def participant_sort_key(participant_id: str) -> tuple[int, str]:
    try:
        return (int(participant_id), participant_id)
    except ValueError:
        return (10**9, participant_id)


def read_split_csv(path: Path, split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=2):
            participant_id = clean_value(row.get("Participant_ID"))
            if not participant_id:
                continue
            cleaned = {key: clean_value(value) for key, value in row.items() if key is not None}
            cleaned["Participant_ID"] = participant_id
            cleaned["_source_file"] = path.name
            cleaned["_source_row"] = str(row_index)
            cleaned["_split"] = split
            rows.append(cleaned)
    return rows


def read_all_split_rows(dataset_root: Path) -> dict[str, list[dict[str, str]]]:
    return {
        split: read_split_csv(dataset_root / filename, split)
        for split, filename in SPLIT_FILES.items()
    }


def parse_optional_int(value: str) -> int | None:
    value = clean_value(value)
    if value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def expected_binary_from_score(score: int | None, threshold: int) -> int | None:
    if score is None:
        return None
    return 1 if score >= threshold else 0


def add_label_issue(
    issues: list[dict[str, Any]],
    severity: str,
    issue_type: str,
    participant_id: str,
    file_name: str,
    split: str,
    details: str,
    phq8_binary: int | None = None,
    phq8_score: int | None = None,
    expected_binary: int | None = None,
) -> None:
    issues.append(
        {
            "severity": severity,
            "issue_type": issue_type,
            "participant_id": participant_id,
            "file": file_name,
            "split": split,
            "details": details,
            "phq8_binary": "" if phq8_binary is None else phq8_binary,
            "phq8_score": "" if phq8_score is None else phq8_score,
            "expected_binary": "" if expected_binary is None else expected_binary,
        }
    )


def build_label_quality_report(
    split_rows: dict[str, list[dict[str, str]]],
    threshold: int,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    all_label_rows: list[dict[str, str]] = []

    for rows in split_rows.values():
        all_label_rows.extend(rows)

    by_file_and_id: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in all_label_rows:
        by_file_and_id[(row["_source_file"], row["Participant_ID"])].append(row)

    for (file_name, participant_id), rows in sorted(by_file_and_id.items()):
        if len(rows) > 1:
            row_numbers = ", ".join(row["_source_row"] for row in rows)
            add_label_issue(
                issues,
                "error",
                "duplicate_participant_in_label_file",
                participant_id,
                file_name,
                rows[0]["_split"],
                f"Participant appears {len(rows)} times in this file at rows {row_numbers}.",
            )

    for row in all_label_rows:
        binary = parse_optional_int(row.get("PHQ8_Binary", ""))
        score = parse_optional_int(row.get("PHQ8_Score", ""))
        expected = expected_binary_from_score(score, threshold)
        if binary is not None and expected is not None and binary != expected:
            add_label_issue(
                issues,
                "warning",
                "phq8_binary_score_disagreement",
                row["Participant_ID"],
                row["_source_file"],
                row["_split"],
                f"PHQ8_Binary={binary} disagrees with PHQ8_Score={score} using threshold >= {threshold}.",
                binary,
                score,
                expected,
            )

    label_values_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_label_rows:
        if clean_value(row.get("PHQ8_Binary")) or clean_value(row.get("PHQ8_Score")):
            label_values_by_id[row["Participant_ID"]].append(row)

    for participant_id, rows in sorted(label_values_by_id.items(), key=lambda item: participant_sort_key(item[0])):
        observed = {
            (
                clean_value(row.get("PHQ8_Binary")),
                clean_value(row.get("PHQ8_Score")),
            )
            for row in rows
        }
        if len(observed) > 1:
            sources = "; ".join(
                f"{row['_source_file']}:{row['_source_row']} binary={clean_value(row.get('PHQ8_Binary'))} score={clean_value(row.get('PHQ8_Score'))}"
                for row in rows
            )
            add_label_issue(
                issues,
                "error",
                "conflicting_labels_across_files",
                participant_id,
                "multiple",
                "multiple",
                f"Participant has conflicting label values across files: {sources}.",
            )

    canonical_split_rows = {
        split: rows for split, rows in split_rows.items() if split in {"train", "dev", "test"}
    }
    splits_by_id: dict[str, set[str]] = defaultdict(set)
    files_by_id: dict[str, list[str]] = defaultdict(list)
    for split, rows in canonical_split_rows.items():
        for row in rows:
            participant_id = row["Participant_ID"]
            splits_by_id[participant_id].add(split)
            files_by_id[participant_id].append(f"{row['_source_file']}:{row['_source_row']}")

    for participant_id, splits in sorted(splits_by_id.items(), key=lambda item: participant_sort_key(item[0])):
        if len(splits) > 1:
            add_label_issue(
                issues,
                "error",
                "participant_in_multiple_splits",
                participant_id,
                "multiple",
                ",".join(sorted(splits)),
                f"Participant appears in more than one canonical split: {', '.join(sorted(splits))}. Sources: {'; '.join(files_by_id[participant_id])}.",
            )

    if not issues:
        add_label_issue(
            issues,
            "info",
            "no_label_quality_issues_detected",
            "",
            "",
            "",
            "No duplicate, conflicting, split-overlap, or PHQ8 threshold disagreement issues were detected.",
        )

    return issues


def build_label_lookup(split_rows: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}

    for split in ("train", "dev"):
        for row in split_rows[split]:
            participant_id = row["Participant_ID"]
            lookup[participant_id] = {
                "official_split": split,
                "label_binary": clean_value(row.get("PHQ8_Binary")),
                "phq8_score": clean_value(row.get("PHQ8_Score")),
                "gender": clean_value(row.get("Gender")),
            }

    full_test_labels = {
        row["Participant_ID"]: row
        for row in split_rows["full_test"]
        if clean_value(row.get("PHQ8_Binary")) or clean_value(row.get("PHQ8_Score"))
    }

    for row in split_rows["test"]:
        participant_id = row["Participant_ID"]
        label_row = full_test_labels.get(participant_id, row)
        lookup[participant_id] = {
            "official_split": "test",
            "label_binary": clean_value(label_row.get("PHQ8_Binary")),
            "phq8_score": clean_value(label_row.get("PHQ8_Score")),
            "gender": clean_value(label_row.get("Gender") or row.get("Gender")),
        }

    return lookup


def get_audio_info(path: Path) -> AudioInfo:
    if not path.exists():
        return AudioInfo(None, None, None)
    try:
        with wave.open(str(path), "rb") as audio:
            frames = audio.getnframes()
            sample_rate = audio.getframerate()
            channels = audio.getnchannels()
            duration = frames / sample_rate if sample_rate else None
            return AudioInfo(duration, sample_rate, channels)
    except (wave.Error, EOFError, OSError):
        return AudioInfo(None, None, None)


def read_participant_speech_duration(path: Path, speaker: str) -> float | None:
    if not path.exists():
        return None

    total = 0.0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"start_time", "stop_time", "speaker"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            return None

        for row in reader:
            if clean_value(row.get("speaker")) != speaker:
                continue
            try:
                start = float(clean_value(row.get("start_time")))
                stop = float(clean_value(row.get("stop_time")))
            except ValueError:
                continue
            if stop > start:
                total += stop - start
    return total


def count_windows(duration_sec: float | None, window_sec: float, hop_sec: float) -> int:
    if duration_sec is None or duration_sec < window_sec:
        return 0
    count = 0
    start = 0.0
    epsilon = 1e-9
    while start + window_sec <= duration_sec + epsilon:
        count += 1
        start += hop_sec
    return count


def format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def build_manifests(
    dataset_root: Path,
    label_lookup: dict[str, dict[str, str]],
    window_sec: float,
    overlap: float,
    speaker: str,
    source_time_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    participant_dirs = {
        path.name[:-2]: path
        for path in dataset_root.glob("*_P")
        if path.is_dir() and path.name[:-2].isdigit()
    }
    participant_ids = sorted(
        set(participant_dirs) | set(label_lookup),
        key=participant_sort_key,
    )
    hop_sec = window_sec * (1.0 - overlap)

    participant_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    for participant_id in participant_ids:
        participant_dir = participant_dirs.get(participant_id, dataset_root / f"{participant_id}_P")
        audio_path = participant_dir / f"{participant_id}_AUDIO.wav"
        transcript_path = participant_dir / f"{participant_id}_TRANSCRIPT.csv"
        label_info = label_lookup.get(participant_id, {})

        audio_info = get_audio_info(audio_path)
        speech_duration = read_participant_speech_duration(transcript_path, speaker)
        num_windows = count_windows(speech_duration, window_sec, hop_sec)

        participant_rows.append(
            {
                "dataset": "DAIC-WOZ",
                "participant_id": participant_id,
                "participant_dir": str(participant_dir),
                "audio_path": str(audio_path) if audio_path.exists() else "",
                "transcript_path": str(transcript_path) if transcript_path.exists() else "",
                "label_binary": label_info.get("label_binary", ""),
                "phq8_score": label_info.get("phq8_score", ""),
                "official_split": label_info.get("official_split", "unassigned"),
                "gender": label_info.get("gender", ""),
                "audio_duration_sec": format_float(audio_info.duration_sec),
                "participant_speech_duration_sec": format_float(speech_duration),
                "sample_rate": "" if audio_info.sample_rate is None else audio_info.sample_rate,
                "num_channels": "" if audio_info.num_channels is None else audio_info.num_channels,
                "window_sec": format_float(window_sec, 3),
                "overlap": format_float(overlap, 3),
                "hop_sec": format_float(hop_sec, 3),
                "num_windows": num_windows,
                "has_audio": int(audio_path.exists()),
                "has_transcript": int(transcript_path.exists()),
                "has_label": int(bool(label_info.get("label_binary"))),
                "source_time_mode": source_time_mode,
            }
        )

        for window_index in range(num_windows):
            start = window_index * hop_sec
            end = start + window_sec
            window_rows.append(
                {
                    "dataset": "DAIC-WOZ",
                    "participant_id": participant_id,
                    "window_id": f"{participant_id}_w{window_index:04d}",
                    "audio_path": str(audio_path) if audio_path.exists() else "",
                    "source_start_sec": "",
                    "source_end_sec": "",
                    "start_sec": format_float(start),
                    "end_sec": format_float(end),
                    "duration_sec": format_float(window_sec),
                    "label_binary": label_info.get("label_binary", ""),
                    "official_split": label_info.get("official_split", "unassigned"),
                    "speaker_source": speaker,
                    "source_time_mode": source_time_mode,
                }
            )

    return participant_rows, window_rows


def build_split_summary(participant_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    splits = ["train", "dev", "test", "unassigned"]
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in participant_rows:
        rows_by_split[clean_value(row.get("official_split")) or "unassigned"].append(row)

    summary: list[dict[str, Any]] = []
    for split in splits:
        rows = rows_by_split.get(split, [])
        if not rows and split == "unassigned":
            continue
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
                "missing_label": labels.get("", 0),
                "windows": windows,
                "avg_windows_per_participant": format_float(avg_windows, 3),
            }
        )
    return summary


def build_data_quality_report(participant_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in participant_rows:
        participant_id = clean_value(row.get("participant_id"))
        split = clean_value(row.get("official_split"))
        if clean_value(row.get("has_audio")) != "1":
            issues.append(
                {
                    "severity": "error",
                    "issue_type": "missing_audio",
                    "participant_id": participant_id,
                    "official_split": split,
                    "details": "Expected participant audio file was not found.",
                }
            )
        if clean_value(row.get("has_transcript")) != "1":
            issues.append(
                {
                    "severity": "error",
                    "issue_type": "missing_transcript",
                    "participant_id": participant_id,
                    "official_split": split,
                    "details": "Expected participant transcript file was not found.",
                }
            )
        if clean_value(row.get("has_label")) != "1":
            issues.append(
                {
                    "severity": "error",
                    "issue_type": "missing_label",
                    "participant_id": participant_id,
                    "official_split": split,
                    "details": "Participant has no binary label in the manifest.",
                }
            )
        if clean_value(row.get("num_windows")) == "0":
            issues.append(
                {
                    "severity": "warning",
                    "issue_type": "zero_windows",
                    "participant_id": participant_id,
                    "official_split": split,
                    "details": "No 10-second participant-speech windows were generated.",
                }
            )

    if not issues:
        issues.append(
            {
                "severity": "info",
                "issue_type": "no_data_quality_issues_detected",
                "participant_id": "",
                "official_split": "",
                "details": "No missing file, missing label, or zero-window issues were detected.",
            }
        )
    return issues


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_run_metadata(path: Path, config: dict[str, Any], outputs: dict[str, str]) -> None:
    metadata = {
        "status": "tasks_1_to_6_complete_no_wavlm_extraction",
        "config": config,
        "outputs": outputs,
        "notes": [
            "Transcript use is limited to timing and speaker segmentation.",
            "Transcript text is not used or copied into generated manifests.",
            "First WavLM baseline will use participant-level mean-pooled window embeddings.",
            "Window-level classifier with mean probability aggregation is deferred.",
            "Labels are flagged for manual review only; no labels are automatically corrected.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    dataset_root = Path(config["dataset_root"])
    output_dir = Path(config["output_dir"])
    threshold = int(config["phq8_depression_threshold"])
    window_sec = float(config["window_sec"])
    overlap = float(config["overlap"])
    speaker = str(config["transcript_speaker"])
    source_time_mode = str(config["source_time_mode"])

    split_rows = read_all_split_rows(dataset_root)
    label_quality_rows = build_label_quality_report(split_rows, threshold)
    label_lookup = build_label_lookup(split_rows)
    participant_rows, window_rows = build_manifests(
        dataset_root,
        label_lookup,
        window_sec,
        overlap,
        speaker,
        source_time_mode,
    )
    split_summary_rows = build_split_summary(participant_rows)
    data_quality_rows = build_data_quality_report(participant_rows)

    output_paths = {
        "participant_manifest": str(output_dir / "daic_phase1_participant_manifest.csv"),
        "window_manifest": str(output_dir / "daic_phase1_window_manifest.csv"),
        "split_summary": str(output_dir / "daic_phase1_split_summary.csv"),
        "label_quality_report": str(output_dir / "label_quality_report.csv"),
        "data_quality_report": str(output_dir / "data_quality_report.csv"),
        "run_metadata": str(output_dir / "prepare_daic_phase1_metadata.json"),
    }

    write_csv(Path(output_paths["participant_manifest"]), participant_rows, PARTICIPANT_MANIFEST_FIELDS)
    write_csv(Path(output_paths["window_manifest"]), window_rows, WINDOW_MANIFEST_FIELDS)
    write_csv(Path(output_paths["split_summary"]), split_summary_rows, SPLIT_SUMMARY_FIELDS)
    write_csv(Path(output_paths["label_quality_report"]), label_quality_rows, LABEL_REPORT_FIELDS)
    write_csv(Path(output_paths["data_quality_report"]), data_quality_rows, DATA_QUALITY_FIELDS)
    write_run_metadata(Path(output_paths["run_metadata"]), config, output_paths)

    print("Phase 1 DAIC-WOZ preparation complete.")
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
