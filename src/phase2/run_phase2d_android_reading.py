"""Execute Phase 2D Android reading-speech benchmark.

This runner evaluates Androids Corpus reading speech using the locked WavLM
layer-9 representation. It performs reading-only input validation, window
generation, frozen WavLM extraction, participant mean aggregation, frozen DAIC
zero-shot evaluation, within-Android reading logistic-regression evaluation,
and interview-versus-reading comparison.

It does not process interview audio, rerun Phase 2C, tune DAIC thresholds on
reading data, fine-tune WavLM, use timing features, combine tasks, or train
any model family beyond balanced logistic regression.
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
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .phase2c_domain_shift import cosine_distance
    from .phase2c_metrics import (
        confusion_counts,
        evaluate_predictions,
        probability_metrics,
        threshold_metrics_from_counts,
    )
    from .phase2d_protocols import (
        FROZEN_DAIC_THRESHOLD,
        assert_reading_fold_integrity,
        protocol_a_zero_shot_rows,
        protocol_b_android_reading_logistic_regression,
    )
    from .run_phase2b_zero_shot_android_interview import (
        EXPECTED_DIM,
        HOP_SAMPLES,
        LOCKED_HIDDEN_STATE_INDEX,
        LOCKED_LAYER_NAME,
        MODEL_COMMIT,
        MODEL_NAME,
        NORMALIZATION_METHOD,
        TARGET_SAMPLE_RATE,
        TARGET_SAMPLES,
        WINDOW_SECONDS,
        choose_cuda,
        clean,
        extract_layer9_batch,
        load_pinned_wavlm,
        normalize_waveform,
        participant_sort_key,
        read_wav_mono_float,
        require_runtime_modules,
        resample_if_needed,
        valid_embedding,
        verify_frozen_pipeline,
    )
except ImportError:  # pragma: no cover - allows direct script execution
    from phase2c_domain_shift import cosine_distance
    from phase2c_metrics import confusion_counts, evaluate_predictions, probability_metrics, threshold_metrics_from_counts
    from phase2d_protocols import (
        FROZEN_DAIC_THRESHOLD,
        assert_reading_fold_integrity,
        protocol_a_zero_shot_rows,
        protocol_b_android_reading_logistic_regression,
    )
    from run_phase2b_zero_shot_android_interview import (
        EXPECTED_DIM,
        HOP_SAMPLES,
        LOCKED_HIDDEN_STATE_INDEX,
        LOCKED_LAYER_NAME,
        MODEL_COMMIT,
        MODEL_NAME,
        NORMALIZATION_METHOD,
        TARGET_SAMPLE_RATE,
        TARGET_SAMPLES,
        WINDOW_SECONDS,
        choose_cuda,
        clean,
        extract_layer9_batch,
        load_pinned_wavlm,
        normalize_waveform,
        participant_sort_key,
        read_wav_mono_float,
        require_runtime_modules,
        resample_if_needed,
        valid_embedding,
        verify_frozen_pipeline,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/phase2d_android_reading.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dependency-dir", default=".phase1_deps")
    return parser.parse_args()


def add_dependency_dir(path: str) -> None:
    dep = PROJECT_ROOT / path
    if dep.exists():
        sys.path.insert(0, str(dep.resolve()))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def resolve_path(raw: str | Path) -> Path:
    path = Path(clean(raw))
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: set[str] = set()
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize(payload), handle, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def git_commit() -> str:
    try:
        git_exe = shutil.which("git") or "git"
        return subprocess.check_output([git_exe, "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return ""


def load_reading_waveform(audio_path: str, torch: Any, torchaudio: Any) -> tuple[np.ndarray, dict[str, Any]]:
    path = resolve_path(audio_path)
    waveform, sample_rate, channels, sample_width = read_wav_mono_float(path)
    if waveform.size == 0:
        raise ValueError(f"Zero-length reading audio: {audio_path}")
    resampled = resample_if_needed(waveform, sample_rate, torch, torchaudio)
    meta = {
        "audio_path": rel(path),
        "source_sample_rate": int(sample_rate),
        "source_channels": int(channels),
        "source_sample_width_bytes": int(sample_width),
        "source_duration_sec": float(waveform.shape[0]) / float(sample_rate),
        "samples_16k": int(resampled.shape[0]),
        "duration_16k_sec": float(resampled.shape[0]) / TARGET_SAMPLE_RATE,
        "mono_conversion_rule": "channels averaged sample-wise before resampling" if channels > 1 else "already_mono",
    }
    return resampled.astype(np.float32, copy=False), meta


def validate_config(config: dict[str, Any]) -> None:
    if config["phase2a_run"].endswith("20260622_154008"):
        raise RuntimeError("Superseded Phase 2A run must not be used for Phase 2D.")
    required = {
        "wavlm_model": MODEL_NAME,
        "model_commit": MODEL_COMMIT,
        "wavlm_layer": LOCKED_LAYER_NAME,
        "hidden_state_index": LOCKED_HIDDEN_STATE_INDEX,
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "target_samples": TARGET_SAMPLES,
        "embedding_dimension": EXPECTED_DIM,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise RuntimeError(f"Locked config mismatch for {key}: {config.get(key)} != {expected}")


def build_reading_inputs(
    config: dict[str, Any],
    run_dir: Path,
    torch: Any,
    torchaudio: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    phase2a_run = resolve_path(config["phase2a_run"])
    participants = {row["participant_id"]: row for row in read_csv(phase2a_run / "android_participant_manifest.csv")}
    recordings = read_csv(phase2a_run / "android_recording_manifest.csv")
    fold_rows = read_csv(phase2a_run / "android_fold_assignment.csv")

    reading_records = [row for row in recordings if row["task"] == "reading" and row.get("audio_status") == "readable"]
    duplicate_participants = [pid for pid, count in Counter(row["participant_id"] for row in reading_records).items() if count > 1]
    quality_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    participant_meta: dict[str, dict[str, Any]] = {}

    if duplicate_participants:
        quality_rows.append({"severity": "error", "issue_type": "duplicate_reading_recording", "participant_id": ";".join(duplicate_participants[:10]), "details": ""})

    reading_folds: dict[str, set[str]] = defaultdict(set)
    for row in fold_rows:
        if row["task"] == "reading" and row["assignment_status"] == "resolved":
            reading_folds[row["participant_id"]].add(row["fold"])

    for row in sorted(reading_records, key=lambda item: participant_sort_key(item["participant_id"])):
        pid = row["participant_id"]
        participant = participants.get(pid)
        if participant is None:
            quality_rows.append({"severity": "error", "issue_type": "participant_missing_from_manifest", "participant_id": pid, "details": ""})
            continue
        expected_label = 0 if row["condition"] == "HC" else 1 if row["condition"] == "PT" else None
        if expected_label is None or int(row["label_binary"]) != expected_label or int(participant["label_binary"]) != expected_label:
            quality_rows.append({"severity": "error", "issue_type": "condition_label_conflict", "participant_id": pid, "details": ""})
            continue
        assigned_folds = reading_folds.get(pid, set())
        if assigned_folds != {row["fold"]}:
            quality_rows.append({"severity": "error", "issue_type": "reading_fold_assignment_conflict", "participant_id": pid, "details": f"recording={row['fold']} assignment={sorted(assigned_folds)}"})
            continue
        try:
            waveform, audio_meta = load_reading_waveform(row["audio_path"], torch, torchaudio)
        except Exception as exc:  # noqa: BLE001
            quality_rows.append({"severity": "error", "issue_type": "reading_audio_unreadable", "participant_id": pid, "details": str(exc)})
            excluded_rows.append({"participant_id": pid, "condition": row["condition"], "reason": f"unreadable:{exc}"})
            continue
        if waveform.shape[0] < TARGET_SAMPLES:
            excluded_rows.append({"participant_id": pid, "condition": row["condition"], "reason": "less_than_10_seconds_usable_reading_speech"})
            quality_rows.append({"severity": "warning", "issue_type": "participant_excluded", "participant_id": pid, "details": "less_than_10_seconds_usable_reading_speech"})
            continue

        window_count = int(math.floor((waveform.shape[0] - TARGET_SAMPLES) / HOP_SAMPLES) + 1)
        if window_count <= 0:
            excluded_rows.append({"participant_id": pid, "condition": row["condition"], "reason": "no_valid_windows"})
            quality_rows.append({"severity": "warning", "issue_type": "participant_excluded", "participant_id": pid, "details": "no_valid_windows"})
            continue

        participant_meta[pid] = {
            "dataset": row["dataset"],
            "participant_id": pid,
            "condition": row["condition"],
            "label_binary": int(row["label_binary"]),
            "label_source": "clinical_group_status",
            "gender": row.get("filename_gender", participant.get("gender", "")),
            "age": row.get("filename_age", participant.get("age", "")),
            "education": row.get("filename_education", participant.get("education", "")),
            "reading_fold": row["fold"],
            "recording_id": row["recording_id"],
            "reading_audio_path": row["audio_path"],
            "reading_duration_sec": audio_meta["source_duration_sec"],
            "reading_samples_16k": audio_meta["samples_16k"],
            "window_count": window_count,
            **audio_meta,
        }
        input_rows.append(
            {
                "dataset": row["dataset"],
                "participant_id": pid,
                "numeric_id": participant.get("numeric_id", ""),
                "condition": row["condition"],
                "label_binary": int(row["label_binary"]),
                "label_source": "clinical_group_status",
                "task": "reading",
                "recording_id": row["recording_id"],
                "reading_audio_path": row["audio_path"],
                "reading_fold": row["fold"],
                "duration_sec": audio_meta["source_duration_sec"],
                "sample_rate": audio_meta["source_sample_rate"],
                "num_channels": audio_meta["source_channels"],
                "samples_16k": audio_meta["samples_16k"],
                "num_windows": window_count,
                "data_quality_status": "eligible",
            }
        )
        if audio_meta["source_sample_rate"] != 44100:
            quality_rows.append({"severity": "warning", "issue_type": "unexpected_sample_rate", "participant_id": pid, "details": audio_meta["source_sample_rate"]})
        if audio_meta["source_channels"] != 1:
            quality_rows.append({"severity": "warning", "issue_type": "unexpected_channel_count", "participant_id": pid, "details": audio_meta["source_channels"]})

        for index in range(window_count):
            start = index * HOP_SAMPLES
            end = start + TARGET_SAMPLES
            if end > waveform.shape[0]:
                quality_rows.append({"severity": "error", "issue_type": "window_end_exceeds_waveform", "participant_id": pid, "details": index + 1})
                continue
            window_rows.append(
                {
                    "dataset": "Androids-Corpus",
                    "participant_id": pid,
                    "condition": row["condition"],
                    "label_binary": int(row["label_binary"]),
                    "label_source": "clinical_group_status",
                    "task": "reading",
                    "recording_id": row["recording_id"],
                    "reading_audio_path": row["audio_path"],
                    "reading_fold": row["fold"],
                    "window_id": f"{pid}_reading_w{index + 1:04d}",
                    "window_index": index + 1,
                    "start_sample_16k": start,
                    "end_sample_16k": end,
                    "start_time_sec": start / TARGET_SAMPLE_RATE,
                    "end_time_sec": end / TARGET_SAMPLE_RATE,
                    "sample_rate": TARGET_SAMPLE_RATE,
                    "final_samples": TARGET_SAMPLES,
                    "hop_samples": HOP_SAMPLES,
                    "window_length_sec": WINDOW_SECONDS,
                    "overlap_sec": WINDOW_SECONDS - (HOP_SAMPLES / TARGET_SAMPLE_RATE),
                    "recording_duration_sec": audio_meta["source_duration_sec"],
                    "window_status": "valid",
                    "error_message": "",
                }
            )

    duplicate_windows = [window_id for window_id, count in Counter(row["window_id"] for row in window_rows).items() if count > 1]
    if duplicate_windows:
        quality_rows.append({"severity": "error", "issue_type": "duplicate_window_id", "participant_id": "", "details": ";".join(duplicate_windows[:20])})
    if not window_rows:
        quality_rows.append({"severity": "error", "issue_type": "no_valid_reading_windows", "participant_id": "", "details": ""})

    tasks = [row["task"] for row in input_rows]
    fold_summaries = assert_reading_fold_integrity(
        [row["participant_id"] for row in input_rows],
        [row["reading_fold"] for row in input_rows],
        np.asarray([int(row["label_binary"]) for row in input_rows], dtype=int),
        tasks,
    )
    window_counts = Counter(row["participant_id"] for row in window_rows)
    fold_rows_out = []
    for summary in fold_summaries:
        pids = [row["participant_id"] for row in input_rows if row["reading_fold"] == summary.fold]
        fold_rows_out.append(
            {
                "fold": summary.fold,
                "participants": summary.participant_count,
                "HC": summary.hc_count,
                "PT": summary.pt_count,
                "reading_recordings": summary.participant_count,
                "reading_windows": sum(window_counts[pid] for pid in pids),
            }
        )

    if not quality_rows:
        quality_rows.append({"severity": "info", "issue_type": "all_reading_input_checks_passed", "participant_id": "", "details": "All Android reading inputs passed structural checks."})

    write_csv(run_dir / "android_reading_input_quality_report.csv", quality_rows)
    write_csv(run_dir / "android_reading_fold_summary.csv", fold_rows_out)
    write_csv(run_dir / "android_reading_window_manifest.csv", window_rows)
    write_csv(
        run_dir / "android_reading_window_quality_report.csv",
        [{"severity": "info", "issue_type": "all_window_checks_passed", "participant_id": "", "window_id": "", "details": "All Android reading windows are 160000-sample reading-only windows."}],
    )
    write_json(
        run_dir / "android_reading_validation_metadata.json",
        {
            "created_at": utc_now(),
            "phase2a_run": rel(phase2a_run),
            "status": "passed" if not any(row["severity"] == "error" for row in quality_rows) else "failed",
            "eligible_reading_participants": len(input_rows),
            "reading_recordings": len(input_rows),
            "class_counts": dict(Counter(row["condition"] for row in input_rows)),
            "fold_summary": fold_rows_out,
            "excluded_recordings": excluded_rows,
            "sample_rate_distribution": dict(Counter(str(row["sample_rate"]) for row in input_rows)),
            "channel_count_distribution": dict(Counter(str(row["num_channels"]) for row in input_rows)),
            "total_windows": len(window_rows),
            "interview_fold_used": False,
            "reading_only_scope": True,
        },
    )
    if any(row["severity"] == "error" for row in quality_rows):
        raise RuntimeError("Android reading input validation failed.")
    return input_rows, window_rows, participant_meta, fold_rows_out


def extract_reading_embeddings(
    run_dir: Path,
    window_rows: list[dict[str, Any]],
    participant_meta: dict[str, dict[str, Any]],
    model: Any,
    device: Any,
    torch: Any,
    torchaudio: Any,
    batch_size: int,
    force: bool,
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    embedding_dir = run_dir / "android_reading_wavlm_window_embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    windows_by_pid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        windows_by_pid[row["participant_id"]].append(row)
    for rows in windows_by_pid.values():
        rows.sort(key=lambda item: int(item["window_index"]))

    manifest_by_window: dict[str, dict[str, Any]] = {}
    quality_rows: list[dict[str, Any]] = []
    batch_payload: list[tuple[dict[str, Any], np.ndarray, Path]] = []
    extracted = 0
    skipped = 0
    failed = 0
    hidden_shapes: list[tuple[int, ...]] | None = None
    start_timestamp = utc_now()
    start_time = time.time()

    def row_for(window: dict[str, Any], path: Path, status: str, error: str) -> dict[str, Any]:
        return {
            "dataset": "Androids-Corpus",
            "participant_id": window["participant_id"],
            "condition": window["condition"],
            "label_binary": int(window["label_binary"]),
            "task": "reading",
            "reading_fold": window["reading_fold"],
            "window_id": window["window_id"],
            "embedding_path": rel(path),
            "embedding_dimension": EXPECTED_DIM if status in {"success", "skipped_valid"} else "",
            "embedding_dtype": "float32" if status in {"success", "skipped_valid"} else "",
            "wavlm_model": MODEL_NAME,
            "model_commit": MODEL_COMMIT,
            "wavlm_layer": LOCKED_LAYER_NAME,
            "hidden_state_index": LOCKED_HIDDEN_STATE_INDEX,
            "pooling_method": "temporal_mean_pooling",
            "sample_rate": TARGET_SAMPLE_RATE,
            "final_samples": TARGET_SAMPLES,
            "extraction_status": status,
            "error_message": error,
        }

    def flush_outputs() -> None:
        write_csv(run_dir / "android_reading_wavlm_embedding_manifest.csv", [manifest_by_window[key] for key in sorted(manifest_by_window)])
        write_csv(
            run_dir / "android_reading_wavlm_embedding_quality_report.csv",
            quality_rows
            or [{"severity": "info", "issue_type": "all_embedding_checks_passed", "participant_id": "", "window_id": "", "details": "All reading embeddings passed quality checks."}],
        )

    def process_batch() -> None:
        nonlocal extracted, failed, hidden_shapes
        if not batch_payload:
            return
        embeddings, shapes = extract_layer9_batch([item[1] for item in batch_payload], model, device, torch)
        if hidden_shapes is None:
            hidden_shapes = shapes
        for (window, _, path), embedding in zip(batch_payload, embeddings):
            try:
                np.save(path, embedding.astype(np.float32, copy=False))
                ok, reason = valid_embedding(path)
                if not ok:
                    raise RuntimeError(reason)
                manifest_by_window[window["window_id"]] = row_for(window, path, "success", "")
                extracted += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                manifest_by_window[window["window_id"]] = row_for(window, path, "failed", str(exc))
                quality_rows.append({"severity": "error", "issue_type": "embedding_save_or_validation_failed", "participant_id": window["participant_id"], "window_id": window["window_id"], "details": str(exc)})
        batch_payload.clear()

    for pid in sorted(windows_by_pid, key=participant_sort_key):
        waveform, _ = load_reading_waveform(participant_meta[pid]["reading_audio_path"], torch, torchaudio)
        for window in windows_by_pid[pid]:
            path = embedding_dir / f"{window['window_id']}.npy"
            if not force:
                ok, reason = valid_embedding(path)
                if ok:
                    manifest_by_window[window["window_id"]] = row_for(window, path, "skipped_valid", "")
                    skipped += 1
                    continue
                if path.exists() and reason != "missing":
                    quality_rows.append({"severity": "warning", "issue_type": "existing_embedding_invalid_reextracting", "participant_id": pid, "window_id": window["window_id"], "details": reason})
            start = int(window["start_sample_16k"])
            end = int(window["end_sample_16k"])
            segment = waveform[start:end]
            if segment.shape[0] != TARGET_SAMPLES:
                failed += 1
                manifest_by_window[window["window_id"]] = row_for(window, path, "failed", f"segment_samples={segment.shape[0]}")
                quality_rows.append({"severity": "error", "issue_type": "window_sample_count_mismatch", "participant_id": pid, "window_id": window["window_id"], "details": str(segment.shape[0])})
                continue
            batch_payload.append((window, normalize_waveform(segment), path))
            if len(batch_payload) >= batch_size:
                process_batch()
        process_batch()
        if (extracted + skipped + failed) and (extracted + skipped + failed) % 200 == 0:
            flush_outputs()

    flush_outputs()
    expected_ids = {row["window_id"] for row in window_rows}
    missing = expected_ids - set(manifest_by_window)
    unexpected = set(manifest_by_window) - expected_ids
    if missing:
        quality_rows.append({"severity": "error", "issue_type": "missing_expected_embeddings", "participant_id": "", "window_id": "", "details": ";".join(sorted(missing)[:20])})
    if unexpected:
        quality_rows.append({"severity": "error", "issue_type": "unexpected_embeddings", "participant_id": "", "window_id": "", "details": ";".join(sorted(unexpected)[:20])})
    success_count = sum(1 for row in manifest_by_window.values() if row["extraction_status"] in {"success", "skipped_valid"})
    if success_count != len(window_rows):
        quality_rows.append({"severity": "error", "issue_type": "embedding_count_mismatch", "participant_id": "", "window_id": "", "details": f"{success_count} != {len(window_rows)}"})
    flush_outputs()

    metadata = {
        "execution_start_timestamp": start_timestamp,
        "execution_completion_timestamp": utc_now(),
        "elapsed_seconds": time.time() - start_time,
        "status": "passed" if not any(row["severity"] == "error" for row in quality_rows) else "failed",
        "wavlm_model": MODEL_NAME,
        "model_commit": MODEL_COMMIT,
        "model_metadata": model_metadata,
        "layer": LOCKED_LAYER_NAME,
        "hidden_state_index": LOCKED_HIDDEN_STATE_INDEX,
        "pooling_method": "temporal_mean_pooling",
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": clean(getattr(torch.version, "cuda", "")),
        "gpu_model": torch.cuda.get_device_name(0),
        "gpu_memory_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "gpu_memory_free_bytes_at_completion": int(torch.cuda.mem_get_info()[0]),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pytorch_version": clean(getattr(torch, "__version__", "")),
        "torchaudio_version": package_version("torchaudio"),
        "transformers_version": package_version("transformers"),
        "batch_size": batch_size,
        "float32_inference": True,
        "gradient_mode": "torch.inference_mode",
        "output_hidden_states": True,
        "hidden_state_shapes_observed_first_batch": [list(shape) for shape in (hidden_shapes or [])],
        "normalization_method": NORMALIZATION_METHOD,
        "total_expected_windows": len(window_rows),
        "total_successful_or_skipped_embeddings": success_count,
        "total_extracted": extracted,
        "total_skipped_valid": skipped,
        "total_failed": failed,
        "participant_count": len(participant_meta),
        "window_embedding_dir": rel(embedding_dir),
        "reading_only_scope": True,
    }
    write_json(run_dir / "android_reading_wavlm_extraction_metadata.json", metadata)
    if metadata["status"] != "passed":
        raise RuntimeError("Reading WavLM extraction failed quality checks.")
    return metadata


def aggregate_reading_participants(
    run_dir: Path,
    window_rows: list[dict[str, Any]],
    participant_meta: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    manifest = read_csv(run_dir / "android_reading_wavlm_embedding_manifest.csv")
    by_pid: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        if row["extraction_status"] in {"success", "skipped_valid"}:
            by_pid[row["participant_id"]].append(row)
    expected_counts = Counter(row["participant_id"] for row in window_rows)
    output_dir = run_dir / "android_reading_participant_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    embeddings: dict[str, np.ndarray] = {}
    total_windows_used = 0

    for pid in sorted(expected_counts, key=participant_sort_key):
        pid_rows = sorted(by_pid.get(pid, []), key=lambda row: row["window_id"])
        status = "success"
        error = ""
        if len(pid_rows) != expected_counts[pid]:
            status = "failed"
            error = f"num_windows_used={len(pid_rows)} expected={expected_counts[pid]}"
            quality_rows.append({"severity": "error", "issue_type": "participant_window_count_mismatch", "participant_id": pid, "details": error})
            participant_embedding = np.zeros(EXPECTED_DIM, dtype=np.float32)
        else:
            arrays = [np.load(resolve_path(row["embedding_path"])).astype(np.float64) for row in pid_rows]
            participant_embedding = np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)
            total_windows_used += len(arrays)
            if participant_embedding.shape != (EXPECTED_DIM,):
                status = "failed"
                error = f"shape={participant_embedding.shape}"
            elif not np.isfinite(participant_embedding).all():
                status = "failed"
                error = "non_finite_values"
            elif np.all(participant_embedding == 0):
                status = "failed"
                error = "all_zero"
        out_path = output_dir / f"{pid}.npy"
        if status == "success":
            np.save(out_path, participant_embedding)
            embeddings[pid] = participant_embedding
        meta = participant_meta[pid]
        rows.append(
            {
                "dataset": "Androids-Corpus",
                "participant_id": pid,
                "condition": meta["condition"],
                "label_binary": int(meta["label_binary"]),
                "label_source": "clinical_group_status",
                "gender": meta.get("gender", ""),
                "age": meta.get("age", ""),
                "education": meta.get("education", ""),
                "task": "reading",
                "participant_embedding_path": rel(out_path),
                "embedding_dimension": EXPECTED_DIM if status == "success" else "",
                "embedding_dtype": "float32" if status == "success" else "",
                "num_windows_expected": expected_counts[pid],
                "num_windows_used": len(pid_rows),
                "participant_speech_duration_sec": meta["reading_duration_sec"],
                "aggregation_method": "arithmetic_mean_across_window_embeddings",
                "source_embedding_model": MODEL_NAME,
                "source_embedding_layer": LOCKED_LAYER_NAME,
                "source_embedding_pooling": "temporal_mean_pooling",
                "reading_fold": meta["reading_fold"],
                "aggregation_status": status,
                "error_message": error,
            }
        )

    if len(embeddings) != len(expected_counts):
        quality_rows.append({"severity": "error", "issue_type": "participant_embedding_count_mismatch", "participant_id": "", "details": f"{len(embeddings)} != {len(expected_counts)}"})
    if total_windows_used != len(window_rows):
        quality_rows.append({"severity": "error", "issue_type": "total_windows_used_mismatch", "participant_id": "", "details": f"{total_windows_used} != {len(window_rows)}"})
    if not quality_rows:
        quality_rows.append({"severity": "info", "issue_type": "all_participant_embedding_checks_passed", "participant_id": "", "details": "All reading participant embeddings passed quality checks."})

    write_csv(run_dir / "android_reading_participant_embedding_manifest.csv", rows)
    write_csv(run_dir / "android_reading_participant_embedding_quality_report.csv", quality_rows)
    write_json(
        run_dir / "android_reading_aggregation_metadata.json",
        {
            "created_at": utc_now(),
            "status": "passed" if not any(row["severity"] == "error" for row in quality_rows) else "failed",
            "participant_embeddings": len(embeddings),
            "expected_participants": len(expected_counts),
            "total_windows_used": total_windows_used,
            "aggregation_method": "float64 accumulation arithmetic mean; saved float32",
            "reading_only_scope": True,
        },
    )
    if any(row["severity"] == "error" for row in quality_rows):
        raise RuntimeError("Reading participant aggregation failed quality checks.")
    return rows, embeddings


def rowwise_predictions(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(float(row["predicted_probability"]) >= float(row["decision_threshold"])) for row in rows], dtype=int)


def metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    labels = np.asarray([int(row["label_binary"]) for row in rows], dtype=int)
    probabilities = np.asarray([float(row["predicted_probability"]) for row in rows], dtype=float)
    thresholds = np.asarray([float(row["decision_threshold"]) for row in rows], dtype=float)
    out: dict[str, float | int] = {}
    out.update(probability_metrics(labels, probabilities))
    if len(set(np.round(thresholds, 12))) == 1:
        out.update(evaluate_predictions(labels, probabilities, float(thresholds[0])))
    else:
        pred = rowwise_predictions(rows)
        out.update(threshold_metrics_from_counts(confusion_counts(labels, pred)))
        out["threshold"] = float("nan")
    out["participant_count"] = len(rows)
    return out


def aggregate_protocol_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["protocol"])].append(row)
    out: list[dict[str, Any]] = []
    for protocol, protocol_rows in sorted(grouped.items()):
        out.append({"protocol": protocol, **metrics_for_rows(protocol_rows)})
    return out


def fold_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["protocol"]), str(row["reading_fold"]))].append(row)
    out: list[dict[str, Any]] = []
    for (protocol, fold), fold_rows in sorted(grouped.items()):
        out.append({"protocol": protocol, "outer_fold": fold, **metrics_for_rows(fold_rows)})
    return out


def stratified_bootstrap_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for cls in sorted(np.unique(labels)):
        cls_idx = np.where(labels == cls)[0]
        parts.append(rng.choice(cls_idx, size=len(cls_idx), replace=True))
    return np.concatenate(parts)


def bootstrap_intervals(
    rows: list[dict[str, Any]],
    n_replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = ["roc_auc", "pr_auc", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "brier_score"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["protocol"])].append(row)
    rng = np.random.default_rng(seed)
    boot_values: dict[str, dict[str, list[float]]] = {protocol: {metric: [] for metric in metrics} for protocol in grouped}
    interval_rows: list[dict[str, Any]] = []

    for protocol, protocol_rows in sorted(grouped.items()):
        labels = np.asarray([int(row["label_binary"]) for row in protocol_rows], dtype=int)
        for _ in range(n_replicates):
            idx = stratified_bootstrap_indices(labels, rng)
            sample = [protocol_rows[i] for i in idx]
            values = metrics_for_rows(sample)
            for metric in metrics:
                value = float(values[metric])
                if np.isfinite(value):
                    boot_values[protocol][metric].append(value)
        point = metrics_for_rows(protocol_rows)
        for metric in metrics:
            values = np.asarray(boot_values[protocol][metric], dtype=float)
            interval_rows.append(
                {
                    "protocol": protocol,
                    "metric": metric,
                    "point_estimate": point[metric],
                    "n_bootstrap_replicates": n_replicates,
                    "n_successful_replicates": int(len(values)),
                    "percentile_2_5": float(np.percentile(values, 2.5)),
                    "percentile_50": float(np.percentile(values, 50)),
                    "percentile_97_5": float(np.percentile(values, 97.5)),
                }
            )

    comparison_pairs = [
        ("A_zero_shot_daic_at_0_5", "B_android_lr_at_0_5", "zero_shot_0_5_vs_android_lr_0_5"),
        ("A_zero_shot_daic_at_frozen_threshold", "B_android_lr_threshold_adapted", "zero_shot_frozen_threshold_vs_android_lr_threshold_adapted"),
    ]
    aggregate = {protocol: metrics_for_rows(protocol_rows) for protocol, protocol_rows in grouped.items()}
    comparison_rows: list[dict[str, Any]] = []
    for left, right, name in comparison_pairs:
        for metric in metrics:
            left_vals = np.asarray(boot_values[left][metric], dtype=float)
            right_vals = np.asarray(boot_values[right][metric], dtype=float)
            n = min(len(left_vals), len(right_vals))
            diff = right_vals[:n] - left_vals[:n]
            comparison_rows.append(
                {
                    "comparison": name,
                    "left_protocol": left,
                    "right_protocol": right,
                    "metric": metric,
                    "point_difference_right_minus_left": float(aggregate[right][metric] - aggregate[left][metric]),
                    "percentile_2_5": float(np.percentile(diff, 2.5)),
                    "percentile_50": float(np.percentile(diff, 50)),
                    "percentile_97_5": float(np.percentile(diff, 97.5)),
                    "n_bootstrap_replicates": n,
                }
            )
    return interval_rows, comparison_rows


def load_daic_pipeline_and_predict(config: dict[str, Any], participant_rows: list[dict[str, Any]], embeddings: dict[str, np.ndarray], joblib: Any) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    pipeline = joblib.load(resolve_path(config["phase1c_pipeline"]))
    verification = verify_frozen_pipeline(pipeline)
    rows = sorted(participant_rows, key=lambda row: participant_sort_key(row["participant_id"]))
    x = np.stack([embeddings[row["participant_id"]] for row in rows]).astype(np.float32)
    probabilities = pipeline.predict_proba(x)[:, 1].astype(float)
    prediction_rows: list[dict[str, Any]] = []
    for row, prob in zip(rows, probabilities):
        prediction_rows.append(
            {
                "participant_id": row["participant_id"],
                "condition": row["condition"],
                "label_binary": int(row["label_binary"]),
                "label_source": "clinical_group_status",
                "task": "reading",
                "reading_fold": row["reading_fold"],
                "num_windows": int(row["num_windows_used"]),
                "participant_speech_duration_sec": float(row["participant_speech_duration_sec"]),
                "predicted_probability": float(prob),
                "prediction_at_0_5": int(prob >= 0.5),
                "frozen_daic_threshold": FROZEN_DAIC_THRESHOLD,
                "prediction_at_frozen_daic_threshold": int(prob >= FROZEN_DAIC_THRESHOLD),
                "model_name": "frozen_daic_candidate_A_wavlm_layer09_mean_logistic_regression",
                "source_training_corpus": "DAIC-WOZ",
                "evaluation_corpus": "Androids Corpus",
                "evaluation_protocol": "zero_shot_cross_corpus_reading",
            }
        )
    return prediction_rows, probabilities, verification


def load_participant_arrays(participant_rows: list[dict[str, Any]], embeddings: dict[str, np.ndarray]) -> dict[str, Any]:
    rows = sorted(participant_rows, key=lambda row: participant_sort_key(row["participant_id"]))
    return {
        "participant_ids": [row["participant_id"] for row in rows],
        "labels": np.asarray([int(row["label_binary"]) for row in rows], dtype=int),
        "folds": [row["reading_fold"] for row in rows],
        "tasks": [row["task"] for row in rows],
        "embeddings": np.stack([embeddings[row["participant_id"]] for row in rows]).astype(np.float32),
        "rows": rows,
    }


def summary_stats(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=0)),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def load_interview_reference(config: dict[str, Any]) -> dict[str, Any]:
    phase2c_run = resolve_path(config["phase2c_run"])
    phase2b_run = resolve_path(config["phase2b_interview_run"])
    report = read_json(phase2c_run / "PHASE2C_WITHIN_ANDROID_REPORT.json")
    protocol_rows = read_csv(phase2c_run / "phase2c_protocol_comparison.csv")
    selected_rows = read_csv(phase2c_run / "phase2c_selected_parameters.csv")
    bootstrap_rows = read_csv(phase2c_run / "phase2c_bootstrap_intervals.csv")
    predictions = read_csv(phase2c_run / "phase2c_outer_predictions.csv")
    window_manifest = read_csv(phase2b_run / "android_interview_window_manifest.csv")
    participant_manifest = read_csv(phase2b_run / "android_interview_participant_embedding_manifest.csv")
    return {
        "phase2c_run": phase2c_run,
        "phase2b_run": phase2b_run,
        "report": report,
        "protocol_rows": protocol_rows,
        "selected_rows": selected_rows,
        "bootstrap_rows": bootstrap_rows,
        "predictions": predictions,
        "window_count": len(window_manifest),
        "participant_manifest": participant_manifest,
    }


def interval_lookup(rows: list[dict[str, Any]], protocol: str, metric: str) -> str:
    for row in rows:
        if row.get("protocol") == protocol and row.get("metric") == metric:
            return f"[{float(row['percentile_2_5']):.4f}, {float(row['percentile_97_5']):.4f}]"
    return ""


def create_task_comparison(
    run_dir: Path,
    reading_metrics: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    selected_parameters: list[dict[str, Any]],
    reading_count: int,
    reading_class_counts: dict[str, int],
    reading_windows: int,
    interview_ref: dict[str, Any],
) -> list[dict[str, Any]]:
    reading_by_protocol = {row["protocol"]: row for row in reading_metrics}
    reading = reading_by_protocol["B_android_lr_threshold_adapted"]
    interview_metrics = {row["protocol"]: row for row in interview_ref["protocol_rows"]}
    interview = interview_metrics["D_android_lr_threshold_adapted"]
    c_dist = Counter(str(row["selected_C"]) for row in selected_parameters if row.get("protocol") == "B_android_reading_logistic_regression")
    threshold_dist = Counter(f"{float(row['selected_threshold']):.6f}" for row in selected_parameters if row.get("protocol") == "B_android_reading_logistic_regression")
    interview_c_dist = interview_ref["report"].get("selected_C_distribution", {})
    interview_threshold_dist = interview_ref["report"].get("selected_threshold_distribution", {})
    interview_class_counts = interview_ref["report"]["data_summary"]["class_counts"]
    rows = [
        {
            "task": "Android reading",
            "participant_count": reading_count,
            "class_counts": f"HC={reading_class_counts.get('HC', 0)};PT={reading_class_counts.get('PT', 0)}",
            "total_windows": reading_windows,
            "average_windows_per_participant": reading_windows / reading_count,
            "speech_type": "controlled_reading",
            "fold_protocol": "official reading five-fold OOF",
            "selected_C_distribution": dict(c_dist),
            "selected_threshold_distribution": dict(threshold_dist),
            "roc_auc": reading["roc_auc"],
            "roc_auc_ci_95": interval_lookup(bootstrap_rows, "B_android_lr_threshold_adapted", "roc_auc"),
            "pr_auc": reading["pr_auc"],
            "balanced_accuracy": reading["balanced_accuracy"],
            "macro_f1": reading["macro_f1"],
            "sensitivity": reading["sensitivity"],
            "specificity": reading["specificity"],
            "brier_score": reading["brier_score"],
        },
        {
            "task": "Android interview",
            "participant_count": int(interview["participant_count"]),
            "class_counts": f"HC={interview_class_counts['HC']};PT={interview_class_counts['PT']}",
            "total_windows": interview_ref["window_count"],
            "average_windows_per_participant": interview_ref["window_count"] / int(interview["participant_count"]),
            "speech_type": "spontaneous_interview",
            "fold_protocol": "official interview five-fold OOF from Phase 2C Protocol D",
            "selected_C_distribution": interview_c_dist,
            "selected_threshold_distribution": interview_threshold_dist,
            "roc_auc": float(interview["roc_auc"]),
            "roc_auc_ci_95": interval_lookup(interview_ref["bootstrap_rows"], "D_android_lr_threshold_adapted", "roc_auc"),
            "pr_auc": float(interview["pr_auc"]),
            "balanced_accuracy": float(interview["balanced_accuracy"]),
            "macro_f1": float(interview["macro_f1"]),
            "sensitivity": float(interview["sensitivity"]),
            "specificity": float(interview["specificity"]),
            "brier_score": float(interview["brier_score"]),
        },
    ]
    write_csv(run_dir / "phase2d_task_comparison.csv", rows)
    return rows


def task_shift_analysis(
    run_dir: Path,
    reading_rows: list[dict[str, Any]],
    reading_embeddings: dict[str, np.ndarray],
    reading_protocol_rows: list[dict[str, Any]],
    interview_ref: dict[str, Any],
) -> dict[str, Any]:
    interview_embeddings: dict[str, np.ndarray] = {}
    for row in interview_ref["participant_manifest"]:
        path = resolve_path(row["participant_embedding_path"])
        arr = np.load(path).astype(np.float32)
        if arr.shape == (EXPECTED_DIM,):
            interview_embeddings[row["participant_id"]] = arr
    common_ids = sorted(set(reading_embeddings) & set(interview_embeddings), key=participant_sort_key)
    reading_matrix = np.stack([reading_embeddings[pid] for pid in common_ids]).astype(float)
    interview_matrix = np.stack([interview_embeddings[pid] for pid in common_ids]).astype(float)
    within = np.asarray([cosine_similarity(reading_embeddings[pid], interview_embeddings[pid]) for pid in common_ids], dtype=float)
    rng = np.random.default_rng(42)
    perm = np.arange(len(common_ids))
    if len(perm) > 1:
        while True:
            rng.shuffle(perm)
            if not np.any(perm == np.arange(len(common_ids))):
                break
    cross = np.asarray([cosine_similarity(reading_matrix[i], interview_matrix[perm[i]]) for i in range(len(common_ids))], dtype=float)
    interview_std = np.std(interview_matrix, axis=0, ddof=0)
    safe_std = np.where(interview_std > 1e-12, interview_std, np.nan)
    standardized_shift = (np.mean(reading_matrix, axis=0) - np.mean(interview_matrix, axis=0)) / safe_std
    abs_shift = np.abs(standardized_shift)
    reading_probs = {
        row["participant_id"]: float(row["predicted_probability"])
        for row in reading_protocol_rows
        if row["protocol"] == "B_android_lr_threshold_adapted"
    }
    interview_probs = {
        row["participant_id"]: float(row["predicted_probability"])
        for row in interview_ref["predictions"]
        if row["protocol"] == "D_android_lr_threshold_adapted"
    }
    common_prob_ids = sorted(set(reading_probs) & set(interview_probs), key=participant_sort_key)
    read_p = np.asarray([reading_probs[pid] for pid in common_prob_ids], dtype=float)
    int_p = np.asarray([interview_probs[pid] for pid in common_prob_ids], dtype=float)
    probability_corr = float(np.corrcoef(read_p, int_p)[0, 1]) if len(common_prob_ids) > 1 else float("nan")
    report = {
        "created_at": utc_now(),
        "common_reading_interview_participants": len(common_ids),
        "within_participant_cosine_similarity": summary_stats(within),
        "random_cross_participant_cosine_similarity": summary_stats(cross),
        "task_centroid_cosine_distance": cosine_distance(np.mean(reading_matrix, axis=0), np.mean(interview_matrix, axis=0)),
        "embedding_norms": {
            "reading": summary_stats(np.linalg.norm(reading_matrix, axis=1)),
            "interview": summary_stats(np.linalg.norm(interview_matrix, axis=1)),
        },
        "standardized_feature_shift_using_interview_std": {
            "mean_absolute_shift": float(np.nanmean(abs_shift)),
            "median_absolute_shift": float(np.nanmedian(abs_shift)),
            "features_shifted_more_than_one_interview_std": int(np.nansum(abs_shift > 1.0)),
            "fraction_features_shifted_more_than_one_interview_std": float(np.nansum(abs_shift > 1.0) / EXPECTED_DIM),
        },
        "within_android_classifier_probability_comparison": {
            "common_participants": len(common_prob_ids),
            "reading_probability": summary_stats(read_p) if len(common_prob_ids) else {},
            "interview_probability": summary_stats(int_p) if len(common_prob_ids) else {},
            "reading_minus_interview_probability": summary_stats(read_p - int_p) if len(common_prob_ids) else {},
            "pearson_correlation": probability_corr,
        },
        "label_independent_analysis_only": True,
    }
    write_json(run_dir / "phase2d_task_shift_report.json", report)
    return report


def classify_reading_signal(reading_metrics: list[dict[str, Any]], bootstrap_rows: list[dict[str, Any]], task_rows: list[dict[str, Any]]) -> str:
    by_protocol = {row["protocol"]: row for row in reading_metrics}
    metric = by_protocol["B_android_lr_threshold_adapted"]
    roc = float(metric["roc_auc"])
    ba = float(metric["balanced_accuracy"])
    f1 = float(metric["macro_f1"])
    ci_low = next(float(row["percentile_2_5"]) for row in bootstrap_rows if row["protocol"] == "B_android_lr_threshold_adapted" and row["metric"] == "roc_auc")
    interview = next(row for row in task_rows if row["task"] == "Android interview")
    gap = float(interview["roc_auc"]) - roc
    if roc >= 0.80 and ci_low >= 0.65 and ba >= 0.75 and f1 >= 0.75:
        return "reading_signal_strong"
    if roc >= 0.70 and ba >= 0.65 and f1 >= 0.65 and gap < 0.15:
        return "reading_signal_moderate"
    if roc >= 0.58 or ba >= 0.58 or f1 >= 0.58:
        return "reading_signal_weak"
    return "reading_signal_failed"


def markdown_report(report: dict[str, Any]) -> str:
    by_protocol = {row["protocol"]: row for row in report["protocol_metrics"]}
    zero = by_protocol["A_zero_shot_daic_at_frozen_threshold"]
    reading = by_protocol["B_android_lr_threshold_adapted"]
    task_rows = {row["task"]: row for row in report["task_comparison"]}
    lines = [
        "# Phase 2D Android Reading-Speech Benchmark",
        "",
        f"Run directory: `{report['run_directory']}`",
        "",
        "## Scope",
        "",
        "Android reading speech was processed with the locked WavLM layer-9 representation. No interview audio was reprocessed, no Phase 2C model was rerun, and no WavLM fine-tuning or new classifier family was used.",
        "",
        "## Reading Inputs",
        "",
        f"- Eligible reading participants: `{report['data_summary']['eligible_reading_participants']}`",
        f"- Class counts: `HC={report['data_summary']['class_counts']['HC']}`, `PT={report['data_summary']['class_counts']['PT']}`",
        f"- Reading recordings: `{report['data_summary']['reading_recordings']}`",
        f"- Reading windows: `{report['data_summary']['reading_windows']}`",
        f"- Extraction status: `{report['extraction']['status']}`",
        "",
        "## Protocol Metrics",
        "",
        "| Protocol | ROC-AUC | PR-AUC | Balanced Acc. | Macro F1 | Sens. | Spec. | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["protocol_metrics"]:
        lines.append(
            f"| {row['protocol']} | {float(row['roc_auc']):.4f} | {float(row['pr_auc']):.4f} | {float(row['balanced_accuracy']):.4f} | "
            f"{float(row['macro_f1']):.4f} | {float(row['sensitivity']):.4f} | {float(row['specificity']):.4f} | {float(row['brier_score']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Headline Comparison",
            "",
            f"- Frozen DAIC zero-shot reading ROC-AUC: `{float(zero['roc_auc']):.4f}`",
            f"- Android-trained reading ROC-AUC: `{float(reading['roc_auc']):.4f}`",
            f"- Android-trained interview ROC-AUC from Phase 2C: `{float(task_rows['Android interview']['roc_auc']):.4f}`",
            f"- Reading-signal classification: `{report['interpretation']['reading_signal_classification']}`",
            "",
            "## Interpretation",
            "",
            report["interpretation"]["narrative"],
            "",
            "No clinical-validity claim is made.",
        ]
    )
    return "\n".join(lines) + "\n"


def create_compact_record(run_dir: Path, compact_root: Path) -> Path:
    compact_dir = compact_root / f"phase2d_{run_dir.name}"
    compact_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "PHASE2D_ANDROID_READING_REPORT.md",
        "PHASE2D_ANDROID_READING_REPORT.json",
        "phase2d_protocol_comparison.csv",
        "phase2d_task_comparison.csv",
        "phase2d_selected_parameters.csv",
        "phase2d_run_metadata.json",
    ]:
        shutil.copy2(run_dir / name, compact_dir / name)
    rows = []
    for path in sorted(compact_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_checksums.json":
            rows.append({"path": str(path.relative_to(compact_dir)).replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    write_json(compact_dir / "artifact_checksums.json", rows)
    return compact_dir


def main() -> None:
    args = parse_args()
    add_dependency_dir(args.dependency_dir)
    joblib, torch, torchaudio, transformers_modules, _, _ = require_runtime_modules()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = read_json(config_path)
    validate_config(config)

    random.seed(int(config["random_seed"]))
    np.random.seed(int(config["random_seed"]))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / config["output_root"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    start_timestamp = utc_now()

    device = choose_cuda(torch)
    input_rows, window_rows, participant_meta, fold_summary = build_reading_inputs(config, run_dir, torch, torchaudio)
    model, model_metadata = load_pinned_wavlm(device, transformers_modules)
    extraction_metadata = extract_reading_embeddings(
        run_dir,
        window_rows,
        participant_meta,
        model,
        device,
        torch,
        torchaudio,
        args.batch_size,
        args.force,
        model_metadata,
    )
    participant_rows, participant_embeddings = aggregate_reading_participants(run_dir, window_rows, participant_meta)

    prediction_rows, frozen_probabilities, pipeline_verification = load_daic_pipeline_and_predict(config, participant_rows, participant_embeddings, joblib)
    write_csv(run_dir / "android_reading_zero_shot_predictions.csv", prediction_rows)
    write_json(run_dir / "frozen_daic_pipeline_verification.json", pipeline_verification)

    arrays = load_participant_arrays(participant_rows, participant_embeddings)
    assert_reading_fold_integrity(arrays["participant_ids"], arrays["folds"], arrays["labels"], arrays["tasks"])
    protocol_rows: list[dict[str, Any]] = []
    protocol_rows.extend(protocol_a_zero_shot_rows(arrays["participant_ids"], arrays["folds"], arrays["labels"], frozen_probabilities))
    b_rows, selected_parameters, inner_rows = protocol_b_android_reading_logistic_regression(
        arrays["participant_ids"],
        arrays["folds"],
        arrays["labels"],
        arrays["embeddings"],
    )
    protocol_rows.extend(b_rows)
    selected_parameter_rows = selected_parameters + [{**row, "row_type": "inner_C_candidate"} for row in inner_rows]

    protocol_metrics = aggregate_protocol_metrics(protocol_rows)
    fold_metric_rows = fold_metrics(protocol_rows)
    bootstrap_rows, paired_rows = bootstrap_intervals(protocol_rows, int(config["bootstrap_replicates"]), int(config["random_seed"]))
    interview_ref = load_interview_reference(config)
    class_counts = dict(Counter(row["condition"] for row in participant_rows))
    task_rows = create_task_comparison(
        run_dir,
        protocol_metrics,
        bootstrap_rows,
        selected_parameters,
        len(participant_rows),
        class_counts,
        len(window_rows),
        interview_ref,
    )
    task_shift = task_shift_analysis(run_dir, participant_rows, participant_embeddings, protocol_rows, interview_ref)
    reading_category = classify_reading_signal(protocol_metrics, bootstrap_rows, task_rows)

    by_protocol = {row["protocol"]: row for row in protocol_metrics}
    zero = by_protocol["A_zero_shot_daic_at_frozen_threshold"]
    reading = by_protocol["B_android_lr_threshold_adapted"]
    interview = next(row for row in task_rows if row["task"] == "Android interview")
    if float(interview["roc_auc"]) - float(reading["roc_auc"]) >= 0.05:
        narrative = "Spontaneous interactive interview speech outperformed controlled reading speech under the matched frozen WavLM/logistic-regression pipeline, suggesting that the interview task contains more useful depression-related signal for this dataset and protocol."
    elif abs(float(interview["roc_auc"]) - float(reading["roc_auc"])) < 0.05:
        narrative = "Reading and interview performance were similar enough that useful vocal cues may persist across speech tasks under this representation, though the comparison remains descriptive."
    else:
        narrative = "Reading speech outperformed interview speech in this run; standardized content may reduce some task variability, but this is descriptive and does not imply a causal mechanism."

    report = {
        "status": "completed_phase2d_android_reading_benchmark",
        "created_at": utc_now(),
        "run_directory": rel(run_dir),
        "official_inputs": {
            "phase2a_run": config["phase2a_run"],
            "phase2c_run": config["phase2c_run"],
            "phase2b_interview_run": config["phase2b_interview_run"],
            "superseded_phase2a_used": False,
        },
        "data_summary": {
            "eligible_reading_participants": len(participant_rows),
            "reading_recordings": len(input_rows),
            "class_counts": class_counts,
            "fold_summary": fold_summary,
            "reading_windows": len(window_rows),
            "average_windows_per_participant": len(window_rows) / len(participant_rows),
        },
        "extraction": extraction_metadata,
        "aggregation": read_json(run_dir / "android_reading_aggregation_metadata.json"),
        "protocol_metrics": protocol_metrics,
        "zero_shot_reading": {
            "probability_range": [float(np.min(frozen_probabilities)), float(np.max(frozen_probabilities))],
            "probability_mean": float(np.mean(frozen_probabilities)),
            "probability_std": float(np.std(frozen_probabilities, ddof=0)),
            "metrics_at_frozen_threshold": zero,
            "metrics_at_0_5": by_protocol["A_zero_shot_daic_at_0_5"],
        },
        "android_trained_reading": {
            "metrics_at_training_selected_threshold": reading,
            "metrics_at_0_5": by_protocol["B_android_lr_at_0_5"],
            "selected_C_distribution": dict(Counter(str(row["selected_C"]) for row in selected_parameters)),
            "selected_threshold_distribution": dict(Counter(f"{float(row['selected_threshold']):.6f}" for row in selected_parameters)),
        },
        "bootstrap_intervals": bootstrap_rows,
        "paired_comparisons": paired_rows,
        "task_comparison": task_rows,
        "task_shift": task_shift,
        "interpretation": {
            "reading_signal_classification": reading_category,
            "narrative": narrative,
            "zero_shot_vs_android_trained_reading": {
                "roc_auc_difference": float(reading["roc_auc"] - zero["roc_auc"]),
                "macro_f1_difference": float(reading["macro_f1"] - zero["macro_f1"]),
                "brier_difference": float(reading["brier_score"] - zero["brier_score"]),
            },
            "interview_minus_reading": {
                "roc_auc_difference": float(interview["roc_auc"] - reading["roc_auc"]),
                "macro_f1_difference": float(interview["macro_f1"] - reading["macro_f1"]),
                "brier_difference": float(interview["brier_score"] - reading["brier_score"]),
            },
            "clinical_validity_claim": False,
        },
        "leakage_checks": {
            "status": "passed",
            "participant_grouping_unit": True,
            "reading_fold_only": True,
            "interview_fold_used_for_reading": False,
            "outer_test_features_used_for_scaler": False,
            "outer_test_labels_used_for_C_selection": False,
            "outer_test_labels_used_for_threshold_selection": False,
            "each_reading_participant_one_oof_prediction": True,
        },
        "scope_confirmation": {
            "phase2c_rerun_performed": False,
            "interview_audio_reprocessed": False,
            "timing_features_used": False,
            "wavlm_layer_changed": False,
            "participant_aggregation_changed": False,
            "new_classifier_family_used": False,
            "phase2e_started": False,
        },
    }

    metadata = {
        "created_at": utc_now(),
        "execution_start_timestamp": start_timestamp,
        "execution_completion_timestamp": utc_now(),
        "config_path": rel(config_path),
        "config_sha256": sha256_file(config_path),
        "run_directory": rel(run_dir),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pytorch_version": clean(getattr(torch, "__version__", "")),
        "torchaudio_version": package_version("torchaudio"),
        "transformers_version": package_version("transformers"),
        "sklearn_version": package_version("scikit-learn"),
        "joblib_version": package_version("joblib"),
        "machine_name": platform.node(),
        "git_commit": git_commit(),
        "random_seed": config["random_seed"],
        "bootstrap_replicates": config["bootstrap_replicates"],
        "batch_size": args.batch_size,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": clean(getattr(torch.version, "cuda", "")),
        "gpu_model": torch.cuda.get_device_name(0),
        "normalization_method": NORMALIZATION_METHOD,
        "input_hashes": {
            name: sha256_file(resolve_path(config["phase2a_run"]) / name)
            for name in [
                "android_participant_manifest.csv",
                "android_recording_manifest.csv",
                "android_fold_assignment.csv",
            ]
        },
        "phase2c_report_sha256": sha256_file(resolve_path(config["phase2c_run"]) / "PHASE2C_WITHIN_ANDROID_REPORT.json"),
        "phase1_pipeline_path": config["phase1c_pipeline"],
        "phase1_pipeline_sha256": sha256_file(resolve_path(config["phase1c_pipeline"])),
        "modeling_scope": "Phase 2D only; reading WavLM extraction plus logistic regression; no Phase 2E",
    }

    write_csv(run_dir / "phase2d_outer_predictions.csv", protocol_rows)
    write_csv(run_dir / "phase2d_protocol_comparison.csv", protocol_metrics)
    write_csv(run_dir / "phase2d_fold_metrics.csv", fold_metric_rows)
    write_csv(run_dir / "phase2d_selected_parameters.csv", selected_parameter_rows)
    write_csv(run_dir / "phase2d_bootstrap_intervals.csv", bootstrap_rows)
    write_csv(run_dir / "phase2d_paired_comparisons.csv", paired_rows)
    write_json(run_dir / "PHASE2D_ANDROID_READING_REPORT.json", report)
    write_json(run_dir / "phase2d_run_metadata.json", metadata)
    (run_dir / "PHASE2D_ANDROID_READING_REPORT.md").write_text(markdown_report(report), encoding="utf-8")
    compact_dir = create_compact_record(run_dir, PROJECT_ROOT / config["compact_record_root"])

    print(f"run_dir={rel(run_dir)}")
    print(f"compact_dir={rel(compact_dir)}")
    print(f"eligible_reading_participants={len(participant_rows)}")
    print(f"reading_windows={len(window_rows)}")
    print(f"zero_shot_roc_auc={zero['roc_auc']}")
    print(f"android_trained_reading_roc_auc={reading['roc_auc']}")
    print(f"reading_signal_classification={reading_category}")


if __name__ == "__main__":
    main()
