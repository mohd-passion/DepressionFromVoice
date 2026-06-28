"""Verify Phase 1 GPU WavLM embeddings, aggregate participants, and run first baselines.

This script intentionally stops after the metadata sanity baseline and the first
participant-level frozen WavLM logistic-regression baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EXPECTED_WINDOW_TOTAL = 17058
EXPECTED_WINDOW_SPLITS = {"train": 9001, "dev": 3314, "test": 4743}
EXPECTED_PARTICIPANT_SPLITS = {"train": 106, "dev": 34, "test": 47}
EXPECTED_CLASS_COUNTS = {
    "train": {"0": 76, "1": 30},
    "dev": {"0": 23, "1": 11},
    "test": {"0": 33, "1": 14},
}
EXPECTED_DIM = 768
EXPECTED_DTYPE = "float32"
EXCLUDED_PARTICIPANTS = {"409", "422"}
RANDOM_SEED = 42
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
CV_SPLITS = 5
CV_REPEATS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase1_daic")
    parser.add_argument("--run-root", default="outputs/phase1_daic/phase1_participant_baseline_runs")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def participant_sort_key(participant_id: str) -> tuple[int, str]:
    try:
        return (int(participant_id), participant_id)
    except ValueError:
        return (10**9, participant_id)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def resolve_path(path_text: str, root: Path) -> Path:
    path = Path(clean(path_text))
    return path if path.is_absolute() else root / path


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(path_or_file := handle, fieldnames=fieldnames, extrasaction="ignore")
        del path_or_file
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def get_package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""


def get_git_commit(root: Path) -> str:
    git = shutil.which("git")
    if not git:
        return ""
    try:
        result = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def make_issue(severity: str, issue_type: str, details: str, **extra: Any) -> dict[str, Any]:
    row = {"severity": severity, "issue_type": issue_type, "details": details}
    row.update(extra)
    return row


def verify_wavlm_outputs(root: Path, output_dir: Path, run_dir: Path) -> dict[str, Any]:
    participant_path = output_dir / "daic_phase1_participant_manifest_clean.csv"
    window_path = output_dir / "daic_phase1_window_manifest_clean.csv"
    embedding_manifest_path = output_dir / "wavlm_embedding_manifest_gpu.csv"
    quality_path = output_dir / "wavlm_embedding_quality_report_gpu.csv"
    extraction_metadata_path = output_dir / "wavlm_extraction_metadata_gpu.json"
    embedding_dir = output_dir / "wavlm_embeddings_gpu" / "final_layer_mean"

    participants = read_csv_dicts(participant_path)
    windows = read_csv_dicts(window_path)
    manifest_rows = read_csv_dicts(embedding_manifest_path)
    quality_rows = read_csv_dicts(quality_path)
    extraction_metadata = json.loads(extraction_metadata_path.read_text(encoding="utf-8"))

    issues: list[dict[str, Any]] = []
    clean_window_ids = {clean(row["window_id"]) for row in windows}
    manifest_window_ids = [clean(row["window_id"]) for row in manifest_rows]
    manifest_window_set = set(manifest_window_ids)
    duplicate_window_count = len(manifest_window_ids) - len(manifest_window_set)
    missing_clean_windows = sorted(clean_window_ids - manifest_window_set)
    unexpected_windows = sorted(manifest_window_set - clean_window_ids)
    status_counts = Counter(clean(row.get("extraction_status")) for row in manifest_rows)
    split_counts = Counter(clean(row.get("official_split")) for row in manifest_rows)
    participant_ids = {clean(row.get("participant_id")) for row in manifest_rows}

    if len(manifest_rows) != EXPECTED_WINDOW_TOTAL:
        issues.append(make_issue("error", "total_embedding_count_mismatch", f"Expected {EXPECTED_WINDOW_TOTAL}, found {len(manifest_rows)}."))
    if split_counts != EXPECTED_WINDOW_SPLITS:
        issues.append(make_issue("error", "split_count_mismatch", f"Expected {EXPECTED_WINDOW_SPLITS}, found {dict(split_counts)}."))
    if status_counts.get("failed", 0) != 0:
        issues.append(make_issue("error", "failed_embeddings_present", f"Failed count is {status_counts.get('failed', 0)}."))
    if duplicate_window_count:
        issues.append(make_issue("error", "duplicate_window_ids", f"Duplicate window ID count is {duplicate_window_count}."))
    if missing_clean_windows:
        issues.append(make_issue("error", "missing_clean_windows", f"Missing clean windows: {len(missing_clean_windows)}.", examples=missing_clean_windows[:10]))
    if unexpected_windows:
        issues.append(make_issue("error", "unexpected_windows", f"Unexpected windows: {len(unexpected_windows)}.", examples=unexpected_windows[:10]))
    excluded_present = sorted(participant_ids & EXCLUDED_PARTICIPANTS)
    if excluded_present:
        issues.append(make_issue("error", "excluded_participants_present", f"Excluded participants present: {excluded_present}."))

    shape_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    nonfinite: list[str] = []
    all_zero: list[str] = []
    missing_files: list[str] = []
    unresolved_paths: list[str] = []
    loaded_count = 0
    norms: list[float] = []

    for row in manifest_rows:
        window_id = clean(row.get("window_id"))
        embedding_path = resolve_path(clean(row.get("embedding_path")), root)
        if not embedding_path.exists():
            missing_files.append(window_id)
            unresolved_paths.append(str(embedding_path))
            continue
        try:
            embedding = np.load(embedding_path)
        except Exception as exc:  # noqa: BLE001 - written into the report.
            issues.append(make_issue("error", "embedding_load_failed", f"{window_id}: {exc}", window_id=window_id))
            continue
        loaded_count += 1
        shape_counts[str(tuple(embedding.shape))] += 1
        dtype_counts[str(embedding.dtype)] += 1
        if embedding.shape != (EXPECTED_DIM,):
            issues.append(make_issue("error", "embedding_dimension_mismatch", f"{window_id}: {embedding.shape}", window_id=window_id))
        if str(embedding.dtype) != EXPECTED_DTYPE:
            issues.append(make_issue("error", "embedding_dtype_mismatch", f"{window_id}: {embedding.dtype}", window_id=window_id))
        if not np.isfinite(embedding).all():
            nonfinite.append(window_id)
        if np.all(embedding == 0):
            all_zero.append(window_id)
        norms.append(float(np.linalg.norm(embedding)))

    if missing_files:
        issues.append(make_issue("error", "missing_embedding_files", f"Missing files: {len(missing_files)}.", examples=missing_files[:10]))
    if nonfinite:
        issues.append(make_issue("error", "nonfinite_embeddings", f"Nonfinite embeddings: {len(nonfinite)}.", examples=nonfinite[:10]))
    if all_zero:
        issues.append(make_issue("error", "all_zero_embeddings", f"All-zero embeddings: {len(all_zero)}.", examples=all_zero[:10]))

    quality_status = [row for row in quality_rows if clean(row.get("severity")) == "error"]
    if quality_status:
        issues.append(make_issue("error", "source_quality_report_has_errors", f"Quality report has {len(quality_status)} error rows."))

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not any(row["severity"] == "error" for row in issues) else "failed",
        "input_paths": {
            "participant_manifest_clean": path_relative_to_root(participant_path, root),
            "window_manifest_clean": path_relative_to_root(window_path, root),
            "embedding_manifest_gpu": path_relative_to_root(embedding_manifest_path, root),
            "embedding_quality_report_gpu": path_relative_to_root(quality_path, root),
            "embedding_metadata_gpu": path_relative_to_root(extraction_metadata_path, root),
            "embedding_dir_gpu": path_relative_to_root(embedding_dir, root),
        },
        "input_hashes": {
            "participant_manifest_clean_sha256": sha256_file(participant_path),
            "window_manifest_clean_sha256": sha256_file(window_path),
            "embedding_manifest_gpu_sha256": sha256_file(embedding_manifest_path),
            "embedding_quality_report_gpu_sha256": sha256_file(quality_path),
            "embedding_metadata_gpu_sha256": sha256_file(extraction_metadata_path),
        },
        "expected": {
            "total_embeddings": EXPECTED_WINDOW_TOTAL,
            "split_counts": EXPECTED_WINDOW_SPLITS,
            "embedding_dimension": EXPECTED_DIM,
            "embedding_dtype": EXPECTED_DTYPE,
        },
        "observed": {
            "total_manifest_rows": len(manifest_rows),
            "loaded_embedding_files": loaded_count,
            "status_counts": dict(status_counts),
            "split_counts": dict(split_counts),
            "duplicate_window_ids": duplicate_window_count,
            "missing_clean_windows": len(missing_clean_windows),
            "unexpected_windows": len(unexpected_windows),
            "participants_409_422_present": excluded_present,
            "shape_counts": dict(shape_counts),
            "dtype_counts": dict(dtype_counts),
            "nonfinite_embeddings": len(nonfinite),
            "all_zero_embeddings": len(all_zero),
            "manifest_paths_unresolved": len(unresolved_paths),
            "embedding_norm_min": min(norms) if norms else None,
            "embedding_norm_max": max(norms) if norms else None,
            "embedding_norm_mean": float(np.mean(norms)) if norms else None,
        },
        "wavlm_extraction_metadata_summary": {
            "device": extraction_metadata.get("device"),
            "gpu_model": extraction_metadata.get("gpu_model"),
            "cuda_version": extraction_metadata.get("cuda_version"),
            "pytorch_version": extraction_metadata.get("pytorch_version"),
            "transformers_version": extraction_metadata.get("transformers_version"),
            "model_config_commit_hash": extraction_metadata.get("model_config_commit_hash"),
            "batch_size": extraction_metadata.get("batch_size"),
            "output_directory": extraction_metadata.get("output_directory"),
        },
        "issues": issues,
    }
    write_json(run_dir / "wavlm_gpu_verification_report.json", report)
    write_csv(
        run_dir / "wavlm_gpu_verification_issues.csv",
        issues or [make_issue("info", "all_verification_checks_passed", "All GPU WavLM verification checks passed.")],
    )
    if report["status"] != "passed":
        raise RuntimeError("GPU WavLM verification failed. See wavlm_gpu_verification_report.json.")
    return report


def test_label_provenance(root: Path, run_dir: Path) -> dict[str, Any]:
    source_path = root / "Data" / "full_test_split.csv"
    avec_test_split_path = root / "Data" / "test_split_Depression_AVEC2017.csv"
    rows = [
        row
        for row in read_csv_dicts(source_path)
        if any(clean(value) for value in row.values())
    ]
    participants = sorted([clean(row["Participant_ID"]) for row in rows], key=participant_sort_key)
    class_counts = Counter(clean(row.get("PHQ8_Binary")) for row in rows)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label_source_file": source_path.name,
        "absolute_source_path": str(source_path.resolve()),
        "project_relative_source_path": path_relative_to_root(source_path, root),
        "source_sha256": sha256_file(source_path),
        "avec_named_test_split_path": path_relative_to_root(avec_test_split_path, root),
        "avec_named_test_split_sha256": sha256_file(avec_test_split_path),
        "participant_ids": participants,
        "num_participants": len(participants),
        "class_counts": {"non_depressed": class_counts.get("0", 0), "depressed": class_counts.get("1", 0)},
        "official_provenance_established": False,
        "test_evaluation_label": "local_labeled_test",
        "test_metrics_provisional": True,
        "source_classification": "locally_available_full_test_label_file",
        "provenance_notes": [
            "Data/test_split_Depression_AVEC2017.csv contains Participant_ID and Gender only.",
            "The PHQ8_Binary and PHQ8_Score fields for the 47 test participants are present in Data/full_test_split.csv.",
            "No accompanying official release documentation was found in this workspace establishing full_test_split.csv as an official AVEC test-label release artifact.",
            "Per protocol, test metrics are labeled local_labeled_test and provisional.",
        ],
        "previous_test_label_use_audit": {
            "manifest_label_population": True,
            "manifest_label_population_note": "src/phase1/prepare_daic_phase1.py uses full_test_split.csv to populate test labels in Phase 1 manifests.",
            "audio_window_preprocessing_decisions": False,
            "feature_selection": False,
            "aggregation_decisions": False,
            "model_selection": False,
            "hyperparameter_selection": False,
            "threshold_tuning": False,
            "model_revision": False,
            "exclusion_decisions": False,
            "exclusion_note": "Clean-manifest exclusions were participant 409 (train label inconsistency) and 422 (dev missing local files), not test participants.",
        },
    }
    write_json(run_dir / "test_label_provenance_report.json", report)
    return report


def aggregate_participant_embeddings(root: Path, output_dir: Path, run_dir: Path) -> dict[str, Any]:
    participant_path = output_dir / "daic_phase1_participant_manifest_clean.csv"
    window_path = output_dir / "daic_phase1_window_manifest_clean.csv"
    embedding_manifest_path = output_dir / "wavlm_embedding_manifest_gpu.csv"
    extraction_metadata_path = output_dir / "wavlm_extraction_metadata_gpu.json"

    participants = {clean(row["participant_id"]): row for row in read_csv_dicts(participant_path)}
    windows = read_csv_dicts(window_path)
    embedding_rows = read_csv_dicts(embedding_manifest_path)
    extraction_metadata = json.loads(extraction_metadata_path.read_text(encoding="utf-8"))

    windows_by_participant = Counter(clean(row["participant_id"]) for row in windows)
    embedding_rows_by_participant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in embedding_rows:
        embedding_rows_by_participant[clean(row["participant_id"])].append(row)

    participant_embedding_dir = run_dir / "participant_embeddings" / "final_layer_mean"
    participant_embedding_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    split_by_participant: dict[str, str] = {}
    total_windows_used = 0

    for participant_id in sorted(participants, key=participant_sort_key):
        participant = participants[participant_id]
        expected_windows = int(clean(participant.get("num_windows")) or "0")
        rows = sorted(embedding_rows_by_participant.get(participant_id, []), key=lambda item: clean(item["window_id"]))
        used_windows = 0
        sum_vector = np.zeros(EXPECTED_DIM, dtype=np.float64)
        status = "success"
        error_messages: list[str] = []
        split = clean(participant.get("official_split"))
        label = clean(participant.get("label_binary"))

        if participant_id in EXCLUDED_PARTICIPANTS:
            status = "failed"
            error_messages.append("excluded_participant_present")
        if len(rows) != expected_windows:
            status = "failed"
            error_messages.append(f"expected_windows={expected_windows} embedding_rows={len(rows)}")
        if windows_by_participant.get(participant_id, 0) != expected_windows:
            status = "failed"
            error_messages.append(f"clean_window_manifest_count={windows_by_participant.get(participant_id, 0)} expected={expected_windows}")

        for row in rows:
            row_split = clean(row.get("official_split"))
            row_label = clean(row.get("label_binary"))
            if row_split != split or row_label != label:
                status = "failed"
                error_messages.append(f"manifest mismatch for {clean(row.get('window_id'))}")
                continue
            if clean(row.get("extraction_status")) != "success":
                status = "failed"
                error_messages.append(f"non_success_window={clean(row.get('window_id'))}")
                continue
            embedding_path = resolve_path(clean(row.get("embedding_path")), root)
            try:
                embedding = np.load(embedding_path)
            except Exception as exc:  # noqa: BLE001 - written into report.
                status = "failed"
                error_messages.append(f"load_failed {clean(row.get('window_id'))}: {exc}")
                continue
            if embedding.shape != (EXPECTED_DIM,) or not np.isfinite(embedding).all() or np.all(embedding == 0):
                status = "failed"
                error_messages.append(f"invalid_window_embedding={clean(row.get('window_id'))}")
                continue
            sum_vector += embedding.astype(np.float64, copy=False)
            used_windows += 1

        output_path = participant_embedding_dir / f"{participant_id}.npy"
        if status == "success" and used_windows == expected_windows and used_windows > 0:
            participant_embedding = (sum_vector / float(used_windows)).astype(np.float32)
            if participant_embedding.shape != (EXPECTED_DIM,) or not np.isfinite(participant_embedding).all() or np.all(participant_embedding == 0):
                status = "failed"
                error_messages.append("invalid_participant_embedding_after_mean")
            else:
                np.save(output_path, participant_embedding)
                total_windows_used += used_windows
        else:
            quality_rows.append(
                make_issue(
                    "error",
                    "participant_aggregation_failed",
                    "; ".join(error_messages),
                    participant_id=participant_id,
                    official_split=split,
                )
            )

        split_by_participant[participant_id] = split
        manifest_rows.append(
            {
                "participant_id": participant_id,
                "official_split": split,
                "label_binary": label,
                "participant_embedding_path": path_relative_to_root(output_path, root),
                "embedding_dimension": EXPECTED_DIM if status == "success" else "",
                "embedding_dtype": EXPECTED_DTYPE if status == "success" else "",
                "num_windows_expected": expected_windows,
                "num_windows_used": used_windows,
                "aggregation_method": "arithmetic_mean_float64_accumulation_window_embeddings_axis0",
                "source_embedding_model": extraction_metadata.get("wavlm_model_name", "microsoft/wavlm-base-plus"),
                "source_embedding_layer": extraction_metadata.get("layer_used", "final_hidden_state"),
                "source_embedding_pooling": extraction_metadata.get("pooling_method", "mean_valid_time_steps"),
                "aggregation_status": status,
                "error_message": "; ".join(error_messages),
            }
        )

    participant_ids = [row["participant_id"] for row in manifest_rows]
    participant_id_set = set(participant_ids)
    split_counts = Counter(row["official_split"] for row in manifest_rows if row["aggregation_status"] == "success")
    class_counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in manifest_rows:
        if row["aggregation_status"] == "success":
            class_counts_by_split[row["official_split"]][row["label_binary"]] += 1

    success_rows = [row for row in manifest_rows if row["aggregation_status"] == "success"]
    duplicate_participants = len(participant_ids) - len(participant_id_set)
    missing_participants = sorted(set(participants) - participant_id_set, key=participant_sort_key)
    excluded_present = sorted(participant_id_set & EXCLUDED_PARTICIPANTS, key=participant_sort_key)

    validation_errors: list[dict[str, Any]] = []
    if len(success_rows) != sum(EXPECTED_PARTICIPANT_SPLITS.values()):
        validation_errors.append(make_issue("error", "participant_total_mismatch", f"Expected 187 successful embeddings, found {len(success_rows)}."))
    if dict(split_counts) != EXPECTED_PARTICIPANT_SPLITS:
        validation_errors.append(make_issue("error", "participant_split_count_mismatch", f"Expected {EXPECTED_PARTICIPANT_SPLITS}, found {dict(split_counts)}."))
    for split, expected_counts in EXPECTED_CLASS_COUNTS.items():
        observed = {key: class_counts_by_split[split].get(key, 0) for key in ("0", "1")}
        if observed != expected_counts:
            validation_errors.append(make_issue("error", "participant_class_count_mismatch", f"{split}: expected {expected_counts}, found {observed}.", official_split=split))
    if duplicate_participants:
        validation_errors.append(make_issue("error", "duplicate_participants", f"Duplicate participant count is {duplicate_participants}."))
    if missing_participants:
        validation_errors.append(make_issue("error", "missing_participants", f"Missing participant count is {len(missing_participants)}.", examples=missing_participants[:10]))
    if excluded_present:
        validation_errors.append(make_issue("error", "excluded_participants_present", f"Excluded participants present: {excluded_present}."))
    if total_windows_used != EXPECTED_WINDOW_TOTAL:
        validation_errors.append(make_issue("error", "total_windows_used_mismatch", f"Expected {EXPECTED_WINDOW_TOTAL}, found {total_windows_used}."))

    loaded_participant_count = 0
    for row in success_rows:
        path = resolve_path(row["participant_embedding_path"], root)
        if not path.exists():
            validation_errors.append(make_issue("error", "missing_participant_embedding_file", f"Missing {path}.", participant_id=row["participant_id"]))
            continue
        embedding = np.load(path)
        loaded_participant_count += 1
        if embedding.shape != (EXPECTED_DIM,):
            validation_errors.append(make_issue("error", "participant_embedding_dimension_mismatch", f"{row['participant_id']}: {embedding.shape}."))
        if str(embedding.dtype) != EXPECTED_DTYPE:
            validation_errors.append(make_issue("error", "participant_embedding_dtype_mismatch", f"{row['participant_id']}: {embedding.dtype}."))
        if not np.isfinite(embedding).all():
            validation_errors.append(make_issue("error", "nonfinite_participant_embedding", f"{row['participant_id']}."))
        if np.all(embedding == 0):
            validation_errors.append(make_issue("error", "all_zero_participant_embedding", f"{row['participant_id']}."))
        if int(row["num_windows_used"]) != int(row["num_windows_expected"]):
            validation_errors.append(make_issue("error", "participant_window_count_mismatch", f"{row['participant_id']}."))

    seen_split_by_participant: defaultdict[str, set[str]] = defaultdict(set)
    for row in embedding_rows:
        seen_split_by_participant[clean(row["participant_id"])].add(clean(row["official_split"]))
    multi_split = sorted([pid for pid, splits in seen_split_by_participant.items() if len(splits) > 1], key=participant_sort_key)
    if multi_split:
        validation_errors.append(make_issue("error", "participant_in_multiple_splits", f"Participants in multiple splits: {multi_split[:10]}."))

    quality_rows.extend(validation_errors)
    if not quality_rows:
        quality_rows.append(make_issue("info", "all_participant_aggregation_checks_passed", "All participant aggregation checks passed."))

    manifest_fields = [
        "participant_id",
        "official_split",
        "label_binary",
        "participant_embedding_path",
        "embedding_dimension",
        "embedding_dtype",
        "num_windows_expected",
        "num_windows_used",
        "aggregation_method",
        "source_embedding_model",
        "source_embedding_layer",
        "source_embedding_pooling",
        "aggregation_status",
        "error_message",
    ]
    write_csv(run_dir / "participant_embedding_manifest.csv", manifest_rows, manifest_fields)
    write_csv(run_dir / "participant_embedding_quality_report.csv", quality_rows)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not validation_errors and all(row["aggregation_status"] == "success" for row in manifest_rows) else "failed",
        "participant_embedding_dir": path_relative_to_root(participant_embedding_dir, root),
        "aggregation_method": "participant arithmetic mean over valid window embeddings; float64 accumulation; saved float32",
        "standardization_during_aggregation": False,
        "participant_counts_by_split": dict(split_counts),
        "class_counts_by_split": {
            split: {"0": class_counts_by_split[split].get("0", 0), "1": class_counts_by_split[split].get("1", 0)}
            for split in ("train", "dev", "test")
        },
        "total_participant_embeddings": loaded_participant_count,
        "total_windows_used": total_windows_used,
        "input_paths": {
            "participant_manifest_clean": path_relative_to_root(participant_path, root),
            "window_manifest_clean": path_relative_to_root(window_path, root),
            "embedding_manifest_gpu": path_relative_to_root(embedding_manifest_path, root),
            "wavlm_extraction_metadata_gpu": path_relative_to_root(extraction_metadata_path, root),
        },
        "input_hashes": {
            "participant_manifest_clean_sha256": sha256_file(participant_path),
            "window_manifest_clean_sha256": sha256_file(window_path),
            "embedding_manifest_gpu_sha256": sha256_file(embedding_manifest_path),
            "wavlm_extraction_metadata_gpu_sha256": sha256_file(extraction_metadata_path),
        },
        "source_wavlm_model": extraction_metadata.get("wavlm_model_name"),
        "source_wavlm_model_commit": extraction_metadata.get("model_config_commit_hash"),
        "source_wavlm_layer": extraction_metadata.get("layer_used"),
        "source_wavlm_pooling": extraction_metadata.get("pooling_method"),
    }
    write_json(run_dir / "participant_aggregation_metadata.json", metadata)
    if metadata["status"] != "passed":
        raise RuntimeError("Participant aggregation validation failed. Model execution stopped.")
    return metadata


def load_model_inputs(root: Path, output_dir: Path, run_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    participant_manifest = pd.read_csv(output_dir / "daic_phase1_participant_manifest_clean.csv", dtype={"participant_id": str, "label_binary": int, "gender": str})
    embedding_manifest = pd.read_csv(run_dir / "participant_embedding_manifest.csv", dtype={"participant_id": str, "label_binary": int})
    merged = participant_manifest.merge(
        embedding_manifest[
            [
                "participant_id",
                "participant_embedding_path",
                "aggregation_status",
                "num_windows_expected",
                "num_windows_used",
            ]
        ],
        on="participant_id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.sort_values("participant_id", key=lambda series: series.map(lambda value: participant_sort_key(str(value))[0])).reset_index(drop=True)
    embeddings: list[np.ndarray] = []
    for row in merged.itertuples(index=False):
        path = resolve_path(row.participant_embedding_path, root)
        embeddings.append(np.load(path).astype(np.float32, copy=False))
    x_wavlm = np.vstack(embeddings)
    y = merged["label_binary"].to_numpy(dtype=int)
    return merged, x_wavlm, y


def metadata_pipeline(C: float) -> Pipeline:
    numeric_features = ["participant_speech_duration_sec", "num_windows"]
    categorical_features = ["gender"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            (
                "gender",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        C=C,
        penalty="l2",
        class_weight="balanced",
        max_iter=10000,
        random_state=RANDOM_SEED,
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


def wavlm_pipeline(C: float) -> Pipeline:
    classifier = LogisticRegression(
        C=C,
        penalty="l2",
        class_weight="balanced",
        max_iter=10000,
        random_state=RANDOM_SEED,
    )
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def positive_prob(model: Any, x_data: Any) -> np.ndarray:
    probabilities = model.predict_proba(x_data)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(probabilities.shape[0], dtype=np.float64)
    return probabilities[:, classes.index(1)].astype(np.float64)


def metric_dict(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "positive_precision": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "positive_recall_sensitivity": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def summarize_metric_rows(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric in ("roc_auc", "balanced_accuracy", "macro_f1"):
        values = np.array([row[metric] for row in metric_rows], dtype=np.float64)
        summary[f"mean_{metric}"] = float(values.mean())
        summary[f"std_{metric}"] = float(values.std(ddof=0))
    return summary


def cv_select_c(model_name: str, x_train: Any, y_train: np.ndarray, factory: Callable[[float], Pipeline]) -> tuple[pd.DataFrame, float]:
    splitter = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_SEED)
    cv_rows: list[dict[str, Any]] = []
    for C in C_GRID:
        fold_metrics: list[dict[str, float]] = []
        for fold_index, (train_index, validation_index) in enumerate(splitter.split(x_train, y_train), start=1):
            model = factory(C)
            if isinstance(x_train, pd.DataFrame):
                x_fold_train = x_train.iloc[train_index]
                x_fold_validation = x_train.iloc[validation_index]
            else:
                x_fold_train = x_train[train_index]
                x_fold_validation = x_train[validation_index]
            y_fold_train = y_train[train_index]
            y_fold_validation = y_train[validation_index]
            model.fit(x_fold_train, y_fold_train)
            probabilities = positive_prob(model, x_fold_validation)
            fold_metrics.append(metric_dict(y_fold_validation, probabilities, 0.5))
        summary = summarize_metric_rows(fold_metrics)
        cv_rows.append(
            {
                "model_name": model_name,
                "C": C,
                "mean_roc_auc": summary["mean_roc_auc"],
                "roc_auc_std": summary["std_roc_auc"],
                "mean_balanced_accuracy_at_0_5": summary["mean_balanced_accuracy"],
                "balanced_accuracy_std_at_0_5": summary["std_balanced_accuracy"],
                "mean_macro_f1_at_0_5": summary["mean_macro_f1"],
                "macro_f1_std_at_0_5": summary["std_macro_f1"],
                "n_splits": CV_SPLITS,
                "n_repeats": CV_REPEATS,
                "random_state": RANDOM_SEED,
            }
        )
    frame = pd.DataFrame(cv_rows)
    selected_row = frame.sort_values(["mean_roc_auc", "roc_auc_std", "C"], ascending=[False, True, True]).iloc[0]
    return frame, float(selected_row["C"])


def dummy_cv_results(x_train: Any, y_train: np.ndarray) -> dict[str, float]:
    splitter = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_SEED)
    metrics: list[dict[str, float]] = []
    for train_index, validation_index in splitter.split(x_train, y_train):
        model = DummyClassifier(strategy="most_frequent")
        if isinstance(x_train, pd.DataFrame):
            x_fold_train = x_train.iloc[train_index]
            x_fold_validation = x_train.iloc[validation_index]
        else:
            x_fold_train = x_train[train_index]
            x_fold_validation = x_train[validation_index]
        model.fit(x_fold_train, y_train[train_index])
        probabilities = positive_prob(model, x_fold_validation)
        metrics.append(metric_dict(y_train[validation_index], probabilities, 0.5))
    return summarize_metric_rows(metrics)


def candidate_thresholds(probabilities: np.ndarray) -> list[float]:
    unique_probabilities = sorted({float(value) for value in probabilities})
    candidates = {0.5}
    candidates.update(unique_probabilities)
    for left, right in zip(unique_probabilities, unique_probabilities[1:]):
        candidates.add((left + right) / 2.0)
    return sorted(candidates)


def tune_threshold(model_name: str, y_dev: np.ndarray, dev_probabilities: np.ndarray) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, Any]] = []
    for threshold in candidate_thresholds(dev_probabilities):
        metrics = metric_dict(y_dev, dev_probabilities, threshold)
        rows.append({"model_name": model_name, **metrics})
    frame = pd.DataFrame(rows)
    selected = frame.assign(
        threshold_distance_to_0_5=(frame["threshold"] - 0.5).abs(),
    ).sort_values(
        ["macro_f1", "balanced_accuracy", "threshold_distance_to_0_5", "threshold"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return frame, float(selected["threshold"])


def prediction_rows(
    model_name: str,
    participants: pd.DataFrame,
    probabilities: np.ndarray,
    tuned_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for participant, probability in zip(participants.itertuples(index=False), probabilities):
        rows.append(
            {
                "participant_id": clean(participant.participant_id),
                "official_split": clean(participant.official_split),
                "label_binary": int(participant.label_binary),
                "predicted_probability": float(probability),
                "prediction_at_0_5": int(probability >= 0.5),
                "prediction_at_tuned_threshold": int(probability >= tuned_threshold),
                "tuned_threshold": tuned_threshold,
                "model_name": model_name,
            }
        )
    return rows


def comparison_rows_for_model(
    model_name: str,
    selected_C: float | None,
    train_cv: dict[str, Any] | None,
    dev_metrics_0_5: dict[str, Any],
    dev_metrics_tuned: dict[str, Any],
    test_metrics_0_5: dict[str, Any],
    test_metrics_tuned: dict[str, Any],
    evaluation_label: str,
    provisional: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if train_cv is not None:
        rows.append(
            {
                "model_name": model_name,
                "stage": "train_repeated_stratified_cv",
                "threshold_type": "fold_default_0_5",
                "C": selected_C if selected_C is not None else "",
                "threshold": 0.5,
                "roc_auc": train_cv.get("mean_roc_auc"),
                "roc_auc_std": train_cv.get("roc_auc_std", train_cv.get("std_roc_auc")),
                "pr_auc": "",
                "balanced_accuracy": train_cv.get("mean_balanced_accuracy_at_0_5", train_cv.get("mean_balanced_accuracy")),
                "macro_f1": train_cv.get("mean_macro_f1_at_0_5", train_cv.get("mean_macro_f1")),
                "positive_precision": "",
                "positive_recall_sensitivity": "",
                "specificity": "",
                "tn": "",
                "fp": "",
                "fn": "",
                "tp": "",
                "evaluation_label": "train_cv",
                "provisional": False,
            }
        )
    for stage, threshold_type, metrics, label, is_provisional in (
        ("dev", "default_0_5", dev_metrics_0_5, "dev", False),
        ("dev", "dev_tuned", dev_metrics_tuned, "dev", False),
        ("test", "default_0_5", test_metrics_0_5, evaluation_label, provisional),
        ("test", "frozen_dev_tuned", test_metrics_tuned, evaluation_label, provisional),
    ):
        rows.append(
            {
                "model_name": model_name,
                "stage": stage,
                "threshold_type": threshold_type,
                "C": selected_C if selected_C is not None else "",
                "threshold": metrics["threshold"],
                "roc_auc": metrics["roc_auc"],
                "roc_auc_std": "",
                "pr_auc": metrics["pr_auc"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "positive_precision": metrics["positive_precision"],
                "positive_recall_sensitivity": metrics["positive_recall_sensitivity"],
                "specificity": metrics["specificity"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "evaluation_label": label,
                "provisional": is_provisional,
            }
        )
    return rows


def run_baselines(root: Path, output_dir: Path, run_dir: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    participant_frame, x_wavlm, y = load_model_inputs(root, output_dir, run_dir)
    participant_frame["participant_speech_duration_sec"] = participant_frame["participant_speech_duration_sec"].astype(float)
    participant_frame["num_windows"] = participant_frame["num_windows"].astype(float)
    participant_frame["gender"] = participant_frame["gender"].astype(str)

    train_mask = participant_frame["official_split"].eq("train").to_numpy()
    dev_mask = participant_frame["official_split"].eq("dev").to_numpy()
    test_mask = participant_frame["official_split"].eq("test").to_numpy()

    metadata_columns = ["participant_speech_duration_sec", "num_windows", "gender"]
    x_meta = participant_frame[metadata_columns]
    x_meta_train = x_meta.loc[train_mask].reset_index(drop=True)
    x_meta_dev = x_meta.loc[dev_mask].reset_index(drop=True)
    x_meta_test = x_meta.loc[test_mask].reset_index(drop=True)
    x_wavlm_train = x_wavlm[train_mask]
    x_wavlm_dev = x_wavlm[dev_mask]
    x_wavlm_test = x_wavlm[test_mask]
    y_train = y[train_mask]
    y_dev = y[dev_mask]
    y_test = y[test_mask]

    train_participants = participant_frame.loc[train_mask].reset_index(drop=True)
    dev_participants = participant_frame.loc[dev_mask].reset_index(drop=True)
    test_participants = participant_frame.loc[test_mask].reset_index(drop=True)

    metadata_correlation_train = float(np.corrcoef(x_meta_train["participant_speech_duration_sec"], x_meta_train["num_windows"])[0, 1])

    metadata_cv, metadata_C = cv_select_c("metadata_logistic_regression", x_meta_train, y_train, metadata_pipeline)
    wavlm_cv, wavlm_C = cv_select_c("wavlm_participant_logistic_regression", x_wavlm_train, y_train, wavlm_pipeline)
    metadata_cv.to_csv(run_dir / "metadata_cv_results.csv", index=False)
    wavlm_cv.to_csv(run_dir / "wavlm_cv_results.csv", index=False)

    metadata_model = metadata_pipeline(metadata_C)
    wavlm_model = wavlm_pipeline(wavlm_C)
    metadata_model.fit(x_meta_train, y_train)
    wavlm_model.fit(x_wavlm_train, y_train)

    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(metadata_model, models_dir / "metadata_logistic_regression_pipeline.joblib")
    joblib.dump(wavlm_model, models_dir / "wavlm_logistic_regression_pipeline.joblib")

    metadata_dev_prob = positive_prob(metadata_model, x_meta_dev)
    metadata_test_prob = positive_prob(metadata_model, x_meta_test)
    wavlm_dev_prob = positive_prob(wavlm_model, x_wavlm_dev)
    wavlm_test_prob = positive_prob(wavlm_model, x_wavlm_test)

    metadata_threshold_search, metadata_threshold = tune_threshold("metadata_logistic_regression", y_dev, metadata_dev_prob)
    wavlm_threshold_search, wavlm_threshold = tune_threshold("wavlm_participant_logistic_regression", y_dev, wavlm_dev_prob)
    pd.concat([metadata_threshold_search, wavlm_threshold_search], ignore_index=True).to_csv(run_dir / "dev_threshold_search.csv", index=False)

    selected_hyperparameters = {
        "selection_rule": {
            "primary": "highest mean train-CV ROC-AUC",
            "tie_break_1": "lowest ROC-AUC standard deviation",
            "tie_break_2": "stronger regularization, lower C",
        },
        "cv_configuration": {
            "splitter": "RepeatedStratifiedKFold",
            "n_splits": CV_SPLITS,
            "n_repeats": CV_REPEATS,
            "random_state": RANDOM_SEED,
        },
        "C_grid": C_GRID,
        "metadata_logistic_regression": {"selected_C": metadata_C},
        "wavlm_participant_logistic_regression": {"selected_C": wavlm_C},
    }
    write_json(run_dir / "selected_hyperparameters.json", selected_hyperparameters)
    selected_thresholds = {
        "threshold_selection_rule": {
            "primary": "maximum dev macro F1",
            "tie_break_1": "highest dev balanced accuracy",
            "tie_break_2": "threshold closest to 0.5",
            "tie_break_3": "lower threshold",
            "candidate_thresholds": "0.5, unique dev predicted probabilities, and midpoints between consecutive sorted unique probabilities",
        },
        "metadata_logistic_regression": {"selected_threshold": metadata_threshold},
        "wavlm_participant_logistic_regression": {"selected_threshold": wavlm_threshold},
    }
    write_json(run_dir / "selected_threshold.json", selected_thresholds)

    metadata_dev_0_5 = metric_dict(y_dev, metadata_dev_prob, 0.5)
    metadata_dev_tuned = metric_dict(y_dev, metadata_dev_prob, metadata_threshold)
    metadata_test_0_5 = metric_dict(y_test, metadata_test_prob, 0.5)
    metadata_test_tuned = metric_dict(y_test, metadata_test_prob, metadata_threshold)
    wavlm_dev_0_5 = metric_dict(y_dev, wavlm_dev_prob, 0.5)
    wavlm_dev_tuned = metric_dict(y_dev, wavlm_dev_prob, wavlm_threshold)
    wavlm_test_0_5 = metric_dict(y_test, wavlm_test_prob, 0.5)
    wavlm_test_tuned = metric_dict(y_test, wavlm_test_prob, wavlm_threshold)

    dummy_model = DummyClassifier(strategy="most_frequent")
    dummy_model.fit(x_meta_train, y_train)
    dummy_dev_prob = positive_prob(dummy_model, x_meta_dev)
    dummy_test_prob = positive_prob(dummy_model, x_meta_test)
    dummy_train_cv = dummy_cv_results(x_meta_train, y_train)
    dummy_dev_0_5 = metric_dict(y_dev, dummy_dev_prob, 0.5)
    dummy_test_0_5 = metric_dict(y_test, dummy_test_prob, 0.5)

    prediction_fields = [
        "participant_id",
        "official_split",
        "label_binary",
        "predicted_probability",
        "prediction_at_0_5",
        "prediction_at_tuned_threshold",
        "tuned_threshold",
        "model_name",
    ]
    metadata_predictions = prediction_rows("metadata_logistic_regression", dev_participants, metadata_dev_prob, metadata_threshold)
    metadata_predictions.extend(prediction_rows("metadata_logistic_regression", test_participants, metadata_test_prob, metadata_threshold))
    wavlm_predictions = prediction_rows("wavlm_participant_logistic_regression", dev_participants, wavlm_dev_prob, wavlm_threshold)
    wavlm_predictions.extend(prediction_rows("wavlm_participant_logistic_regression", test_participants, wavlm_test_prob, wavlm_threshold))
    write_csv(run_dir / "metadata_participant_predictions.csv", metadata_predictions, prediction_fields)
    write_csv(run_dir / "wavlm_participant_predictions.csv", wavlm_predictions, prediction_fields)

    metadata_train_cv_row = metadata_cv.loc[metadata_cv["C"].eq(metadata_C)].iloc[0].to_dict()
    wavlm_train_cv_row = wavlm_cv.loc[wavlm_cv["C"].eq(wavlm_C)].iloc[0].to_dict()
    dummy_train_cv_row = {
        "mean_roc_auc": dummy_train_cv["mean_roc_auc"],
        "roc_auc_std": dummy_train_cv["std_roc_auc"],
        "mean_balanced_accuracy_at_0_5": dummy_train_cv["mean_balanced_accuracy"],
        "mean_macro_f1_at_0_5": dummy_train_cv["mean_macro_f1"],
    }
    evaluation_label = clean(provenance["test_evaluation_label"])
    provisional = bool(provenance["test_metrics_provisional"])
    comparison_rows: list[dict[str, Any]] = []
    comparison_rows.extend(
        comparison_rows_for_model(
            "majority_class_dummy",
            None,
            dummy_train_cv_row,
            dummy_dev_0_5,
            dummy_dev_0_5,
            dummy_test_0_5,
            dummy_test_0_5,
            evaluation_label,
            provisional,
        )
    )
    comparison_rows.extend(
        comparison_rows_for_model(
            "metadata_logistic_regression",
            metadata_C,
            metadata_train_cv_row,
            metadata_dev_0_5,
            metadata_dev_tuned,
            metadata_test_0_5,
            metadata_test_tuned,
            evaluation_label,
            provisional,
        )
    )
    comparison_rows.extend(
        comparison_rows_for_model(
            "wavlm_participant_logistic_regression",
            wavlm_C,
            wavlm_train_cv_row,
            wavlm_dev_0_5,
            wavlm_dev_tuned,
            wavlm_test_0_5,
            wavlm_test_tuned,
            evaluation_label,
            provisional,
        )
    )
    comparison_fields = [
        "model_name",
        "stage",
        "threshold_type",
        "C",
        "threshold",
        "roc_auc",
        "roc_auc_std",
        "pr_auc",
        "balanced_accuracy",
        "macro_f1",
        "positive_precision",
        "positive_recall_sensitivity",
        "specificity",
        "tn",
        "fp",
        "fn",
        "tp",
        "evaluation_label",
        "provisional",
    ]
    write_csv(run_dir / "baseline_comparison.csv", comparison_rows, comparison_fields)

    participant_ids_by_split = {
        split: sorted(participant_frame.loc[participant_frame["official_split"].eq(split), "participant_id"].astype(str).tolist(), key=participant_sort_key)
        for split in ("train", "dev", "test")
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_first_phase1_metadata_and_wavlm_logistic_regression_baselines_only",
        "participant_counts": {split: int(mask.sum()) for split, mask in {"train": train_mask, "dev": dev_mask, "test": test_mask}.items()},
        "metadata_sanity_baseline": {
            "features": ["participant_speech_duration_sec", "num_windows", "gender"],
            "train_only_duration_num_windows_pearson_correlation": metadata_correlation_train,
            "selected_C": metadata_C,
            "selected_threshold": metadata_threshold,
            "train_cv_selected_C": metadata_train_cv_row,
            "dev_metrics_at_0_5": metadata_dev_0_5,
            "dev_metrics_at_selected_threshold": metadata_dev_tuned,
            "test_metrics_at_0_5": metadata_test_0_5,
            "test_metrics_at_selected_threshold": metadata_test_tuned,
        },
        "wavlm_participant_baseline": {
            "features": "participant mean embedding, 768 dimensions",
            "selected_C": wavlm_C,
            "selected_threshold": wavlm_threshold,
            "train_cv_selected_C": wavlm_train_cv_row,
            "dev_metrics_at_0_5": wavlm_dev_0_5,
            "dev_metrics_at_selected_threshold": wavlm_dev_tuned,
            "test_metrics_at_0_5": wavlm_test_0_5,
            "test_metrics_at_selected_threshold": wavlm_test_tuned,
        },
        "dummy_baseline": {
            "strategy": "majority_class_most_frequent_fit_on_train_only",
            "train_cv": dummy_train_cv_row,
            "dev_metrics_at_0_5": dummy_dev_0_5,
            "test_metrics_at_0_5": dummy_test_0_5,
        },
        "test_label_provenance_status": evaluation_label,
        "test_metrics_provisional": provisional,
        "no_additional_model_families_trained": True,
    }
    write_json(run_dir / "evaluation_report.json", report)

    metadata = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "machine_name": platform.node(),
        "git_commit": get_git_commit(root),
        "python_version": sys.version.replace("\n", " "),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": get_package_version("scikit-learn"),
        "input_manifest_paths": {
            "participant_manifest_clean": path_relative_to_root(output_dir / "daic_phase1_participant_manifest_clean.csv", root),
            "window_manifest_clean": path_relative_to_root(output_dir / "daic_phase1_window_manifest_clean.csv", root),
            "participant_embedding_manifest": path_relative_to_root(run_dir / "participant_embedding_manifest.csv", root),
            "embedding_manifest_gpu": path_relative_to_root(output_dir / "wavlm_embedding_manifest_gpu.csv", root),
            "wavlm_extraction_metadata_gpu": path_relative_to_root(output_dir / "wavlm_extraction_metadata_gpu.json", root),
        },
        "input_manifest_hashes": {
            "participant_manifest_clean_sha256": sha256_file(output_dir / "daic_phase1_participant_manifest_clean.csv"),
            "window_manifest_clean_sha256": sha256_file(output_dir / "daic_phase1_window_manifest_clean.csv"),
            "participant_embedding_manifest_sha256": sha256_file(run_dir / "participant_embedding_manifest.csv"),
            "embedding_manifest_gpu_sha256": sha256_file(output_dir / "wavlm_embedding_manifest_gpu.csv"),
            "wavlm_extraction_metadata_gpu_sha256": sha256_file(output_dir / "wavlm_extraction_metadata_gpu.json"),
        },
        "wavlm_extraction_metadata_path": path_relative_to_root(output_dir / "wavlm_extraction_metadata_gpu.json", root),
        "wavlm_model_commit": json.loads((output_dir / "wavlm_extraction_metadata_gpu.json").read_text(encoding="utf-8")).get("model_config_commit_hash"),
        "participant_aggregation_method": "arithmetic mean over participant windows; float64 accumulation; float32 output",
        "random_seed": RANDOM_SEED,
        "cross_validation_configuration": {
            "splitter": "RepeatedStratifiedKFold",
            "n_splits": CV_SPLITS,
            "n_repeats": CV_REPEATS,
            "random_state": RANDOM_SEED,
        },
        "C_search_grid": C_GRID,
        "selected_C": {
            "metadata_logistic_regression": metadata_C,
            "wavlm_participant_logistic_regression": wavlm_C,
        },
        "scaler_configuration": {
            "metadata_numeric": "SimpleImputer(strategy=median) then StandardScaler fit inside CV/final train-only pipeline",
            "metadata_gender": "SimpleImputer(strategy=most_frequent) then OneHotEncoder(handle_unknown=ignore) fit inside CV/final train-only pipeline",
            "wavlm": "StandardScaler fit inside CV/final train-only pipeline",
        },
        "class_weight_configuration": "LogisticRegression(class_weight='balanced')",
        "threshold_selection_rule": selected_thresholds["threshold_selection_rule"],
        "selected_threshold": {
            "metadata_logistic_regression": metadata_threshold,
            "wavlm_participant_logistic_regression": wavlm_threshold,
        },
        "participant_ids_by_split": participant_ids_by_split,
        "test_label_provenance_status": evaluation_label,
        "test_metrics_provisional": provisional,
        "saved_models": {
            "metadata_pipeline": path_relative_to_root(models_dir / "metadata_logistic_regression_pipeline.joblib", root),
            "wavlm_pipeline": path_relative_to_root(models_dir / "wavlm_logistic_regression_pipeline.joblib", root),
        },
    }
    write_json(run_dir / "model_run_metadata.json", metadata)

    report_md = [
        "# Phase 1 Participant Baselines",
        "",
        f"Run directory: `{path_relative_to_root(run_dir, root)}`",
        "",
        "## Provenance",
        f"- Test-label evaluation label: `{evaluation_label}`",
        f"- Test metrics provisional: `{provisional}`",
        "",
        "## Metadata Sanity Baseline",
        f"- Selected C: `{metadata_C}`",
        f"- Dev-selected threshold: `{metadata_threshold:.12g}`",
        f"- Train-only duration/window-count Pearson r: `{metadata_correlation_train:.6f}`",
        f"- Dev tuned macro F1: `{metadata_dev_tuned['macro_f1']:.6f}`, balanced accuracy: `{metadata_dev_tuned['balanced_accuracy']:.6f}`",
        f"- Test tuned macro F1: `{metadata_test_tuned['macro_f1']:.6f}`, balanced accuracy: `{metadata_test_tuned['balanced_accuracy']:.6f}`",
        "",
        "## WavLM Participant Baseline",
        f"- Selected C: `{wavlm_C}`",
        f"- Dev-selected threshold: `{wavlm_threshold:.12g}`",
        f"- Dev tuned macro F1: `{wavlm_dev_tuned['macro_f1']:.6f}`, balanced accuracy: `{wavlm_dev_tuned['balanced_accuracy']:.6f}`",
        f"- Test tuned macro F1: `{wavlm_test_tuned['macro_f1']:.6f}`, balanced accuracy: `{wavlm_test_tuned['balanced_accuracy']:.6f}`",
        "",
        "No MLP, SVM, attention pooling, layer-wise analysis, window-level classifier, WavLM fine-tuning, Android Corpus processing, or app code was run.",
    ]
    (run_dir / "evaluation_report.md").write_text("\n".join(report_md) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    root = Path.cwd()
    output_dir = resolve_path(args.output_dir, root)
    timestamp = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve_path(args.run_root, root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    verification = verify_wavlm_outputs(root, output_dir, run_dir)
    provenance = test_label_provenance(root, run_dir)
    aggregation = aggregate_participant_embeddings(root, output_dir, run_dir)
    evaluation = run_baselines(root, output_dir, run_dir, provenance)

    print("Phase 1 participant baseline stage complete.")
    print(f"run_dir={path_relative_to_root(run_dir, root)}")
    print(f"verification_status={verification['status']}")
    print(f"aggregation_status={aggregation['status']}")
    print(f"test_label_status={provenance['test_evaluation_label']}")
    print(f"metadata_C={evaluation['metadata_sanity_baseline']['selected_C']}")
    print(f"wavlm_C={evaluation['wavlm_participant_baseline']['selected_C']}")
    print(f"wavlm_threshold={evaluation['wavlm_participant_baseline']['selected_threshold']}")


if __name__ == "__main__":
    main()
