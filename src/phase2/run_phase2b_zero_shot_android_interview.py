"""Phase 2B frozen zero-shot DAIC WavLM validation on Android interview speech.

This script reconstructs Androids Corpus interview participant speech from the
verified clip files, extracts frozen WavLM layer-9 window embeddings, aggregates
them to participant embeddings, applies the frozen Phase 1 DAIC-WOZ pipeline,
and writes zero-shot external-validation reports.

It does not process reading speech, train on Android, fit preprocessing on
Android, tune thresholds, recalibrate probabilities, change WavLM layers, or
modify Phase 1 artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


MODEL_NAME = "microsoft/wavlm-base-plus"
MODEL_COMMIT = "4c66d4806a428f2e922ccfa1a962776e232d487b"
LOCKED_LAYER_NAME = "layer_09"
LOCKED_HIDDEN_STATE_INDEX = 9
TARGET_SAMPLE_RATE = 16000
WINDOW_SECONDS = 10.0
HOP_SECONDS = 5.0
TARGET_SAMPLES = 160000
HOP_SAMPLES = 80000
EXPECTED_DIM = 768
FROZEN_DAIC_THRESHOLD = 0.555840134621
RANDOM_SEED = 42
BOOTSTRAP_REPLICATES = 5000
NORMALIZATION_METHOD = "float32_dc_offset_removal_peak_normalization_no_dataset_norm_no_augmentation_no_denoising"

PHASE2A_RUN = Path("outputs/phase2_android/phase2a_preparation_runs/20260622_154228")
PHASE1C_PIPELINE = Path("outputs/phase1_daic/phase1c_aggregation_runs/20260622_135800/models/selected_aggregation_pipeline.joblib")
PHASE1C_REPORT = Path("outputs/phase1_daic/phase1c_aggregation_runs/20260622_135800/evaluation_report.json")
PHASE1F_REPORT = Path("outputs/phase1_daic/phase1f_final_evaluation_runs/20260622_151118/PHASE1_FINAL_REPORT.json")
MODEL_CACHE_DIR = Path("outputs/phase1_daic/model_cache/huggingface")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2a-run", default=str(PHASE2A_RUN))
    parser.add_argument("--run-root", default="outputs/phase2_android/phase2b_zero_shot_interview_runs")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dependency-dir", default=".phase1_deps")
    return parser.parse_args()


def add_dependency_dir(path: str) -> None:
    dep = Path(path)
    if dep.exists():
        sys.path.insert(0, str(dep.resolve()))


def require_runtime_modules() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import joblib
        import torch
        import torchaudio
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            balanced_accuracy_score,
            brier_score_loss,
            confusion_matrix,
            f1_score,
            precision_score,
            roc_auc_score,
        )
        from transformers import AutoFeatureExtractor, WavLMModel
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing Phase 2B dependency: torch, torchaudio, transformers, joblib, or scikit-learn.") from exc
    metrics = {
        "average_precision_score": average_precision_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "brier_score_loss": brier_score_loss,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "roc_auc_score": roc_auc_score,
    }
    return joblib, torch, torchaudio, (AutoFeatureExtractor, WavLMModel), metrics, LogisticRegression


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def resolve_path(raw: str, project_root: Path) -> Path:
    path = Path(clean(raw))
    return path if path.is_absolute() else project_root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
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


def participant_sort_key(pid: str) -> tuple[str, int, str]:
    condition, _, numeric = pid.partition("_")
    try:
        number = int(numeric)
    except ValueError:
        number = 10**9
    return condition, number, pid


def read_wav_mono_float(path: Path) -> tuple[np.ndarray, int, int, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)

    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32, copy=False), sample_rate, channels, sample_width


def resample_if_needed(waveform: np.ndarray, sample_rate: int, torch: Any, torchaudio: Any) -> np.ndarray:
    if sample_rate == TARGET_SAMPLE_RATE:
        return waveform.astype(np.float32, copy=False)
    tensor = torch.from_numpy(waveform.astype(np.float32, copy=False)).unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, sample_rate, TARGET_SAMPLE_RATE)
    return resampled.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    normalized = waveform.astype(np.float32, copy=True)
    normalized -= float(np.mean(normalized))
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 0.0:
        normalized /= peak
    return normalized.astype(np.float32, copy=False)


def reconstruct_participant_from_clips(
    participant_id: str,
    clips_by_participant: dict[str, list[dict[str, str]]],
    project_root: Path,
    torch: Any,
    torchaudio: Any,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    clip_rows = sorted(clips_by_participant[participant_id], key=lambda row: int(float(clean(row["clip_order"]))))
    source_rates: list[int] = []
    source_channels: list[int] = []
    raw_parts: list[np.ndarray] = []
    clip_source_meta: list[dict[str, Any]] = []
    source_cursor = 0

    for row in clip_rows:
        path = resolve_path(row["clip_audio_path"], project_root)
        waveform, sample_rate, channels, sample_width = read_wav_mono_float(path)
        if waveform.size == 0:
            raise ValueError(f"Zero-length clip for {participant_id}: {row['clip_id']}")
        raw_parts.append(waveform)
        source_rates.append(sample_rate)
        source_channels.append(channels)
        source_start = source_cursor
        source_end = source_cursor + int(waveform.shape[0])
        source_cursor = source_end
        clip_source_meta.append(
            {
                "clip_id": row["clip_id"],
                "clip_order": int(float(clean(row["clip_order"]))),
                "clip_audio_path": row["clip_audio_path"],
                "source_sample_rate": sample_rate,
                "source_channels": channels,
                "source_sample_width_bytes": sample_width,
                "source_start_sample": source_start,
                "source_end_sample": source_end,
                "source_duration_sec": float(waveform.shape[0]) / float(sample_rate),
            }
        )

    if len(set(source_rates)) == 1:
        source_rate = source_rates[0]
        concatenated = np.concatenate(raw_parts).astype(np.float32, copy=False)
        reconstructed = resample_if_needed(concatenated, source_rate, torch, torchaudio)
        cumulative_sec = 0.0
        for meta in clip_source_meta:
            start_sec = cumulative_sec
            cumulative_sec += float(meta["source_duration_sec"])
            meta["target_start_sample_16k"] = int(round(start_sec * TARGET_SAMPLE_RATE))
            meta["target_end_sample_16k"] = int(round(cumulative_sec * TARGET_SAMPLE_RATE))
        if clip_source_meta:
            clip_source_meta[-1]["target_end_sample_16k"] = int(reconstructed.shape[0])
        reconstruction_basis = "clip_files_concatenated_at_source_rate_then_resampled_once"
    else:
        resampled_parts: list[np.ndarray] = []
        cursor = 0
        for waveform, sample_rate, meta in zip(raw_parts, source_rates, clip_source_meta):
            resampled = resample_if_needed(waveform, sample_rate, torch, torchaudio)
            resampled_parts.append(resampled)
            meta["target_start_sample_16k"] = cursor
            cursor += int(resampled.shape[0])
            meta["target_end_sample_16k"] = cursor
        reconstructed = np.concatenate(resampled_parts).astype(np.float32, copy=False)
        reconstruction_basis = "clip_files_resampled_individually_due_mixed_source_sample_rates"

    for index in range(1, len(clip_source_meta)):
        previous = clip_source_meta[index - 1]
        current = clip_source_meta[index]
        if int(current["target_start_sample_16k"]) < int(previous["target_end_sample_16k"]):
            current["target_start_sample_16k"] = int(previous["target_end_sample_16k"])

    meta = {
        "participant_id": participant_id,
        "num_clips": len(clip_rows),
        "source_sample_rates": sorted(set(source_rates)),
        "source_channels": sorted(set(source_channels)),
        "stereo_clips": [row["clip_id"] for row, channels in zip(clip_rows, source_channels) if channels > 1],
        "mono_conversion_rule": "channels averaged sample-wise to mono before resampling",
        "reconstruction_basis": reconstruction_basis,
        "participant_speech_duration_sec": float(reconstructed.shape[0]) / TARGET_SAMPLE_RATE,
        "participant_speech_samples_16k": int(reconstructed.shape[0]),
    }
    return reconstructed.astype(np.float32, copy=False), clip_source_meta, meta


def build_android_inputs(
    project_root: Path,
    phase2a_run: Path,
    run_dir: Path,
    torch: Any,
    torchaudio: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, str]]]:
    participants = {row["participant_id"]: row for row in read_csv(phase2a_run / "android_participant_manifest.csv")}
    recordings = read_csv(phase2a_run / "android_recording_manifest.csv")
    clips = read_csv(phase2a_run / "android_interview_clip_manifest.csv")
    timing_quality = read_csv(phase2a_run / "android_interview_timing_quality_report.csv")
    wavlm_compat = json.loads((phase2a_run / "daic_android_wavlm_compatibility_report.json").read_text(encoding="utf-8"))

    if not wavlm_compat.get("candidate_A_preprocessing_can_be_reproduced"):
        raise RuntimeError("Phase 2A WavLM compatibility did not approve Candidate A preprocessing reproduction.")
    if any(row.get("severity") == "error" for row in timing_quality):
        raise RuntimeError("Phase 2A timing quality report contains errors.")

    clips_by_participant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in clips:
        clip_status = clean(row.get("audio_status"))
        if row.get("mapping_status") == "mapped_to_single_parent" and clip_status in {"", "readable"}:
            clips_by_participant[row["participant_id"]].append(row)

    interview_recordings = {row["participant_id"]: row for row in recordings if row["task"] == "interview"}
    eligible_participants = [
        pid for pid, row in participants.items() if row["has_interview_audio"] == "1" and pid in clips_by_participant and pid in interview_recordings
    ]

    window_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    participant_meta: dict[str, dict[str, Any]] = {}
    excluded_rows: list[dict[str, Any]] = []

    for pid in sorted(eligible_participants, key=participant_sort_key):
        participant = participants[pid]
        try:
            waveform, clip_meta, meta = reconstruct_participant_from_clips(pid, clips_by_participant, project_root, torch, torchaudio)
        except Exception as exc:  # noqa: BLE001
            excluded_rows.append({"participant_id": pid, "condition": participant["condition"], "reason": f"reconstruction_failed:{exc}"})
            continue

        if waveform.shape[0] < TARGET_SAMPLES:
            excluded_rows.append(
                {
                    "participant_id": pid,
                    "condition": participant["condition"],
                    "reason": "less_than_10_seconds_usable_interview_speech",
                    "participant_speech_duration_sec": meta["participant_speech_duration_sec"],
                }
            )
            continue

        meta["condition"] = participant["condition"]
        meta["label_binary"] = int(participant["label_binary"])
        meta["gender"] = participant.get("gender", "")
        meta["age"] = participant.get("age", "")
        meta["education"] = participant.get("education", "")
        meta["interview_fold"] = interview_recordings[pid].get("fold", "")
        meta["reading_fold_or_conflict"] = participant.get("fold", "")
        participant_meta[pid] = meta

        window_count = int(math.floor((waveform.shape[0] - TARGET_SAMPLES) / HOP_SAMPLES) + 1)
        clip_meta_sorted = sorted(clip_meta, key=lambda item: int(item["target_start_sample_16k"]))
        for index in range(window_count):
            start = index * HOP_SAMPLES
            end = start + TARGET_SAMPLES
            window_id = f"{pid}_interview_w{index + 1:04d}"
            overlapping = [
                item
                for item in clip_meta_sorted
                if int(item["target_start_sample_16k"]) < end and int(item["target_end_sample_16k"]) > start
            ]
            window_rows.append(
                {
                    "dataset": "Androids-Corpus",
                    "participant_id": pid,
                    "condition": participant["condition"],
                    "label_binary": int(participant["label_binary"]),
                    "label_source": "clinical_group_status",
                    "task": "interview",
                    "window_id": window_id,
                    "window_index": index + 1,
                    "start_sample_16k": start,
                    "end_sample_16k": end,
                    "start_time_sec": start / TARGET_SAMPLE_RATE,
                    "end_time_sec": end / TARGET_SAMPLE_RATE,
                    "sample_rate": TARGET_SAMPLE_RATE,
                    "final_samples": TARGET_SAMPLES,
                    "hop_samples": HOP_SAMPLES,
                    "window_length_sec": WINDOW_SECONDS,
                    "overlap_sec": WINDOW_SECONDS - HOP_SECONDS,
                    "num_source_clips": len(overlapping),
                    "source_clip_ids": ";".join(str(item["clip_id"]) for item in overlapping),
                    "participant_speech_duration_sec": meta["participant_speech_duration_sec"],
                    "interview_fold": meta["interview_fold"],
                    "window_status": "valid",
                    "error_message": "",
                }
            )
            if not overlapping:
                quality_rows.append({"severity": "error", "issue_type": "window_without_source_clip", "participant_id": pid, "window_id": window_id, "details": ""})
            for source_index, item in enumerate(overlapping, start=1):
                source_start = max(start, int(item["target_start_sample_16k"]))
                source_end = min(end, int(item["target_end_sample_16k"]))
                mapping_rows.append(
                    {
                        "dataset": "Androids-Corpus",
                        "participant_id": pid,
                        "window_id": window_id,
                        "source_index": source_index,
                        "clip_id": item["clip_id"],
                        "clip_order": item["clip_order"],
                        "clip_audio_path": item["clip_audio_path"],
                        "window_overlap_start_sample": source_start - start,
                        "window_overlap_end_sample": source_end - start,
                        "participant_concat_start_sample_16k": source_start,
                        "participant_concat_end_sample_16k": source_end,
                        "source_clip_start_sample_16k": int(item["target_start_sample_16k"]),
                        "source_clip_end_sample_16k": int(item["target_end_sample_16k"]),
                        "mapping_status": "valid",
                    }
                )

    participant_ids_with_windows = {row["participant_id"] for row in window_rows}
    for pid in sorted(set(eligible_participants) - participant_ids_with_windows, key=participant_sort_key):
        if not any(row["participant_id"] == pid for row in excluded_rows):
            excluded_rows.append({"participant_id": pid, "condition": participants[pid]["condition"], "reason": "no_valid_windows"})

    duplicate_windows = [item for item, count in Counter(row["window_id"] for row in window_rows).items() if count > 1]
    if duplicate_windows:
        for window_id in duplicate_windows:
            quality_rows.append({"severity": "error", "issue_type": "duplicate_window_id", "participant_id": "", "window_id": window_id, "details": ""})
    if excluded_rows:
        for row in excluded_rows:
            quality_rows.append({"severity": "warning", "issue_type": "participant_excluded", "participant_id": row["participant_id"], "window_id": "", "details": row["reason"]})
    if not quality_rows:
        quality_rows.append({"severity": "info", "issue_type": "all_window_checks_passed", "participant_id": "", "window_id": "", "details": "All Android interview windows passed structural checks."})

    write_csv(run_dir / "android_interview_window_manifest.csv", window_rows)
    write_csv(run_dir / "android_interview_window_mapping.csv", mapping_rows)
    write_csv(run_dir / "android_interview_window_quality_report.csv", quality_rows)
    write_csv(run_dir / "android_interview_exclusions.csv", excluded_rows or [{"participant_id": "", "condition": "", "reason": "none"}])

    if any(row["severity"] == "error" for row in quality_rows):
        raise RuntimeError("Android interview window validation failed.")
    if not window_rows:
        raise RuntimeError("No valid Android interview windows were generated.")

    return window_rows, mapping_rows, quality_rows, excluded_rows, participant_meta, clips


def choose_cuda(torch: Any) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 2B WavLM extraction; torch.cuda.is_available() is False.")
    return torch.device("cuda")


def load_pinned_wavlm(device: Any, transformers_modules: tuple[Any, Any]) -> tuple[Any, dict[str, Any]]:
    AutoFeatureExtractor, WavLMModel = transformers_modules
    kwargs = {
        "revision": MODEL_COMMIT,
        "cache_dir": str(MODEL_CACHE_DIR),
        "local_files_only": True,
    }
    AutoFeatureExtractor.from_pretrained(MODEL_NAME, **kwargs)
    model = WavLMModel.from_pretrained(MODEL_NAME, **kwargs)
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata = {
        "model_name": MODEL_NAME,
        "requested_model_commit": MODEL_COMMIT,
        "loaded_model_config_commit_hash": clean(getattr(model.config, "_commit_hash", "")),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "hidden_state_indexing": {
            "hidden_states[0]": "pre-transformer feature-projection state; not used",
            "hidden_states[9]": "transformer encoder layer 09 used for Phase 2B",
        },
    }
    if metadata["num_hidden_layers"] < LOCKED_HIDDEN_STATE_INDEX or metadata["hidden_size"] != EXPECTED_DIM:
        raise RuntimeError(f"Unexpected WavLM config: {metadata}")
    if metadata["loaded_model_config_commit_hash"] and metadata["loaded_model_config_commit_hash"] != MODEL_COMMIT:
        raise RuntimeError(f"Loaded WavLM commit mismatch: {metadata['loaded_model_config_commit_hash']} != {MODEL_COMMIT}")
    return model, metadata


def valid_embedding(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        array = np.load(path)
    except Exception as exc:  # noqa: BLE001
        return False, f"load_failed:{exc}"
    if array.shape != (EXPECTED_DIM,):
        return False, f"shape={array.shape}"
    if array.dtype != np.float32:
        return False, f"dtype={array.dtype}"
    if not np.isfinite(array).all():
        return False, "non_finite"
    if np.all(array == 0):
        return False, "all_zero"
    return True, "ok"


def extract_layer9_batch(waveforms: list[np.ndarray], model: Any, device: Any, torch: Any) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    inputs = torch.from_numpy(np.stack(waveforms).astype(np.float32, copy=False)).to(device)
    with torch.inference_mode():
        outputs = model(inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) <= LOCKED_HIDDEN_STATE_INDEX:
            raise RuntimeError("WavLM did not return the expected hidden states.")
        shapes = [tuple(state.shape) for state in hidden_states]
        layer = hidden_states[LOCKED_HIDDEN_STATE_INDEX]
        pooled = layer.mean(dim=1)
    return pooled.detach().cpu().numpy().astype(np.float32, copy=False), shapes


def extract_window_embeddings(
    project_root: Path,
    run_dir: Path,
    window_rows: list[dict[str, Any]],
    clips_by_participant: dict[str, list[dict[str, str]]],
    participant_meta: dict[str, dict[str, Any]],
    model: Any,
    device: Any,
    torch: Any,
    torchaudio: Any,
    batch_size: int,
    force: bool,
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    embedding_dir = run_dir / "android_interview_wavlm_window_embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    windows_by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        windows_by_participant[row["participant_id"]].append(row)
    for rows in windows_by_participant.values():
        rows.sort(key=lambda row: int(row["window_index"]))

    manifest_by_window: dict[str, dict[str, Any]] = {}
    quality_rows: list[dict[str, Any]] = []
    batch_payload: list[tuple[dict[str, Any], np.ndarray, Path]] = []
    extracted = 0
    skipped = 0
    failed = 0
    hidden_shapes: list[tuple[int, ...]] | None = None
    start_timestamp = utc_now()
    start_time = time.time()

    def manifest_row(window: dict[str, Any], path: Path, status: str, error: str) -> dict[str, Any]:
        return {
            "dataset": "Androids-Corpus",
            "participant_id": window["participant_id"],
            "condition": window["condition"],
            "label_binary": window["label_binary"],
            "window_id": window["window_id"],
            "task": "interview",
            "embedding_path": rel(path, project_root),
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

    def flush_manifest() -> None:
        write_csv(run_dir / "android_interview_wavlm_embedding_manifest.csv", [manifest_by_window[key] for key in sorted(manifest_by_window)])
        write_csv(
            run_dir / "android_interview_wavlm_embedding_quality_report.csv",
            quality_rows
            or [{"severity": "info", "issue_type": "all_embedding_checks_passed", "participant_id": "", "window_id": "", "details": "All embeddings passed quality checks."}],
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
                manifest_by_window[window["window_id"]] = manifest_row(window, path, "success", "")
                extracted += 1
            except Exception as exc:  # noqa: BLE001
                manifest_by_window[window["window_id"]] = manifest_row(window, path, "failed", str(exc))
                quality_rows.append(
                    {"severity": "error", "issue_type": "embedding_save_or_validation_failed", "participant_id": window["participant_id"], "window_id": window["window_id"], "details": str(exc)}
                )
                failed += 1
        batch_payload.clear()

    for participant_id in sorted(windows_by_participant, key=participant_sort_key):
        waveform, _, _ = reconstruct_participant_from_clips(participant_id, clips_by_participant, project_root, torch, torchaudio)
        for window in windows_by_participant[participant_id]:
            path = embedding_dir / f"{window['window_id']}.npy"
            if not force:
                ok, reason = valid_embedding(path)
                if ok:
                    manifest_by_window[window["window_id"]] = manifest_row(window, path, "skipped_valid", "")
                    skipped += 1
                    continue
                if path.exists() and reason != "missing":
                    quality_rows.append(
                        {"severity": "warning", "issue_type": "existing_embedding_invalid_reextracting", "participant_id": participant_id, "window_id": window["window_id"], "details": reason}
                    )
            start = int(window["start_sample_16k"])
            end = int(window["end_sample_16k"])
            segment = waveform[start:end]
            if segment.shape[0] != TARGET_SAMPLES:
                failed += 1
                manifest_by_window[window["window_id"]] = manifest_row(window, path, "failed", f"segment_samples={segment.shape[0]}")
                quality_rows.append({"severity": "error", "issue_type": "window_sample_count_mismatch", "participant_id": participant_id, "window_id": window["window_id"], "details": str(segment.shape[0])})
                continue
            batch_payload.append((window, normalize_waveform(segment), path))
            if len(batch_payload) >= batch_size:
                process_batch()
        process_batch()
        if (extracted + skipped + failed) % 200 == 0:
            flush_manifest()

    flush_manifest()
    missing = {row["window_id"] for row in window_rows} - set(manifest_by_window)
    unexpected = set(manifest_by_window) - {row["window_id"] for row in window_rows}
    if missing:
        quality_rows.append({"severity": "error", "issue_type": "missing_expected_embeddings", "participant_id": "", "window_id": "", "details": ";".join(sorted(missing)[:20])})
    if unexpected:
        quality_rows.append({"severity": "error", "issue_type": "unexpected_embeddings", "participant_id": "", "window_id": "", "details": ";".join(sorted(unexpected)[:20])})
    success_count = sum(1 for row in manifest_by_window.values() if row["extraction_status"] in {"success", "skipped_valid"})
    if success_count != len(window_rows):
        quality_rows.append({"severity": "error", "issue_type": "embedding_count_mismatch", "participant_id": "", "window_id": "", "details": f"{success_count} != {len(window_rows)}"})
    flush_manifest()

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
        "window_embedding_dir": rel(embedding_dir, project_root),
        "no_android_training_or_adaptation": True,
    }
    write_json(run_dir / "android_interview_wavlm_extraction_metadata.json", metadata)
    if metadata["status"] != "passed":
        raise RuntimeError("WavLM extraction failed quality checks.")
    return metadata


def aggregate_participants(project_root: Path, run_dir: Path, window_rows: list[dict[str, Any]], participant_meta: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    manifest = read_csv(run_dir / "android_interview_wavlm_embedding_manifest.csv")
    by_pid: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        if row["extraction_status"] in {"success", "skipped_valid"}:
            by_pid[row["participant_id"]].append(row)

    expected_counts = Counter(row["participant_id"] for row in window_rows)
    output_dir = run_dir / "android_interview_participant_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    embeddings: dict[str, np.ndarray] = {}
    total_windows_used = 0

    for pid in sorted(expected_counts, key=participant_sort_key):
        rows_for_pid = sorted(by_pid.get(pid, []), key=lambda row: row["window_id"])
        error = ""
        status = "success"
        if len(rows_for_pid) != expected_counts[pid]:
            status = "failed"
            error = f"num_windows_used={len(rows_for_pid)} expected={expected_counts[pid]}"
            quality_rows.append({"severity": "error", "issue_type": "participant_window_count_mismatch", "participant_id": pid, "details": error})
            participant_embedding = np.zeros(EXPECTED_DIM, dtype=np.float32)
        else:
            arrays = [np.load(resolve_path(row["embedding_path"], project_root)).astype(np.float64) for row in rows_for_pid]
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
        rows.append(
            {
                "dataset": "Androids-Corpus",
                "participant_id": pid,
                "condition": participant_meta[pid]["condition"],
                "label_binary": participant_meta[pid]["label_binary"],
                "label_source": "clinical_group_status",
                "gender": participant_meta[pid].get("gender", ""),
                "age": participant_meta[pid].get("age", ""),
                "education": participant_meta[pid].get("education", ""),
                "task": "interview",
                "participant_embedding_path": rel(out_path, project_root),
                "embedding_dimension": EXPECTED_DIM if status == "success" else "",
                "embedding_dtype": "float32" if status == "success" else "",
                "num_windows_expected": expected_counts[pid],
                "num_windows_used": len(rows_for_pid),
                "participant_speech_duration_sec": participant_meta[pid]["participant_speech_duration_sec"],
                "aggregation_method": "arithmetic_mean_across_window_embeddings",
                "source_embedding_model": MODEL_NAME,
                "source_embedding_layer": LOCKED_LAYER_NAME,
                "source_embedding_pooling": "temporal_mean_pooling",
                "interview_fold": participant_meta[pid]["interview_fold"],
                "aggregation_status": status,
                "error_message": error,
            }
        )

    if len(embeddings) != len(expected_counts):
        quality_rows.append({"severity": "error", "issue_type": "participant_embedding_count_mismatch", "participant_id": "", "details": f"{len(embeddings)} != {len(expected_counts)}"})
    if total_windows_used != len(window_rows):
        quality_rows.append({"severity": "error", "issue_type": "total_windows_used_mismatch", "participant_id": "", "details": f"{total_windows_used} != {len(window_rows)}"})
    if not quality_rows:
        quality_rows.append({"severity": "info", "issue_type": "all_participant_embedding_checks_passed", "participant_id": "", "details": "All participant embeddings passed quality checks."})

    write_csv(run_dir / "android_interview_participant_embedding_manifest.csv", rows)
    write_csv(run_dir / "android_interview_participant_embedding_quality_report.csv", quality_rows)
    write_json(
        run_dir / "android_interview_aggregation_metadata.json",
        {
            "created_at": utc_now(),
            "status": "passed" if not any(row["severity"] == "error" for row in quality_rows) else "failed",
            "participant_embeddings": len(embeddings),
            "expected_participants": len(expected_counts),
            "total_windows_used": total_windows_used,
            "aggregation_method": "float64 accumulation arithmetic mean; saved float32",
            "no_android_training_or_adaptation": True,
        },
    )
    if any(row["severity"] == "error" for row in quality_rows):
        raise RuntimeError("Participant aggregation failed quality checks.")
    return rows, embeddings


def safe_metric(func: Any, *args: Any, **kwargs: Any) -> float:
    try:
        value = func(*args, **kwargs)
        return float(value)
    except Exception:  # noqa: BLE001
        return float("nan")


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float, metrics: dict[str, Any]) -> dict[str, Any]:
    pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = metrics["confusion_matrix"](y_true, pred, labels=[0, 1]).ravel()
    precision_pos = safe_metric(metrics["precision_score"], y_true, pred, pos_label=1, zero_division=0)
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    npv = float(tn / (tn + fn)) if (tn + fn) else float("nan")
    return {
        "threshold": float(threshold),
        "balanced_accuracy": safe_metric(metrics["balanced_accuracy_score"], y_true, pred),
        "macro_f1": safe_metric(metrics["f1_score"], y_true, pred, average="macro", zero_division=0),
        "positive_class_precision": precision_pos,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "negative_predictive_value": npv,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_count": int(pred.sum()),
        "predicted_positive_proportion": float(np.mean(pred)),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson(rankdata(x), rankdata(y))


def bootstrap_intervals(y: np.ndarray, probabilities: np.ndarray, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(RANDOM_SEED)
    class_indices = {label: np.where(y == label)[0] for label in [0, 1]}
    definitions: list[tuple[str, Any]] = [
        ("roc_auc", lambda yy, pp: metrics["roc_auc_score"](yy, pp)),
        ("pr_auc", lambda yy, pp: metrics["average_precision_score"](yy, pp)),
        ("balanced_accuracy_at_0_5", lambda yy, pp: threshold_metrics(yy, pp, 0.5, metrics)["balanced_accuracy"]),
        ("balanced_accuracy_at_frozen_daic_threshold", lambda yy, pp: threshold_metrics(yy, pp, FROZEN_DAIC_THRESHOLD, metrics)["balanced_accuracy"]),
        ("macro_f1_at_0_5", lambda yy, pp: threshold_metrics(yy, pp, 0.5, metrics)["macro_f1"]),
        ("macro_f1_at_frozen_daic_threshold", lambda yy, pp: threshold_metrics(yy, pp, FROZEN_DAIC_THRESHOLD, metrics)["macro_f1"]),
        ("sensitivity_at_0_5", lambda yy, pp: threshold_metrics(yy, pp, 0.5, metrics)["sensitivity"]),
        ("sensitivity_at_frozen_daic_threshold", lambda yy, pp: threshold_metrics(yy, pp, FROZEN_DAIC_THRESHOLD, metrics)["sensitivity"]),
        ("specificity_at_0_5", lambda yy, pp: threshold_metrics(yy, pp, 0.5, metrics)["specificity"]),
        ("specificity_at_frozen_daic_threshold", lambda yy, pp: threshold_metrics(yy, pp, FROZEN_DAIC_THRESHOLD, metrics)["specificity"]),
    ]
    values: dict[str, list[float]] = {name: [] for name, _ in definitions}
    failures: Counter[str] = Counter()
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in class_indices.values()])
        yy = y[sampled]
        pp = probabilities[sampled]
        for name, func in definitions:
            try:
                value = float(func(yy, pp))
                if math.isfinite(value):
                    values[name].append(value)
                else:
                    failures[name] += 1
            except Exception:  # noqa: BLE001
                failures[name] += 1
    rows: list[dict[str, Any]] = []
    for name, vals in values.items():
        array = np.asarray(vals, dtype=np.float64)
        point = {
            "roc_auc": safe_metric(metrics["roc_auc_score"], y, probabilities),
            "pr_auc": safe_metric(metrics["average_precision_score"], y, probabilities),
            "balanced_accuracy_at_0_5": threshold_metrics(y, probabilities, 0.5, metrics)["balanced_accuracy"],
            "balanced_accuracy_at_frozen_daic_threshold": threshold_metrics(y, probabilities, FROZEN_DAIC_THRESHOLD, metrics)["balanced_accuracy"],
            "macro_f1_at_0_5": threshold_metrics(y, probabilities, 0.5, metrics)["macro_f1"],
            "macro_f1_at_frozen_daic_threshold": threshold_metrics(y, probabilities, FROZEN_DAIC_THRESHOLD, metrics)["macro_f1"],
            "sensitivity_at_0_5": threshold_metrics(y, probabilities, 0.5, metrics)["sensitivity"],
            "sensitivity_at_frozen_daic_threshold": threshold_metrics(y, probabilities, FROZEN_DAIC_THRESHOLD, metrics)["sensitivity"],
            "specificity_at_0_5": threshold_metrics(y, probabilities, 0.5, metrics)["specificity"],
            "specificity_at_frozen_daic_threshold": threshold_metrics(y, probabilities, FROZEN_DAIC_THRESHOLD, metrics)["specificity"],
        }[name]
        rows.append(
            {
                "metric": name,
                "point_estimate": point,
                "n_bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "n_successful_replicates": int(array.size),
                "n_failed_or_undefined_replicates": int(failures[name]),
                "percentile_2_5": float(np.percentile(array, 2.5)) if array.size else "",
                "percentile_50": float(np.percentile(array, 50)) if array.size else "",
                "percentile_97_5": float(np.percentile(array, 97.5)) if array.size else "",
                "bootstrap_type": "stratified_participant_resampling_fixed_predictions_no_refit",
            }
        )
    return rows


def reliability_and_probability_shift(y: np.ndarray, p: np.ndarray, metrics: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    bins = np.linspace(0.0, 1.0, 11)
    hist, edges = np.histogram(p, bins=bins)
    reliability_rows: list[dict[str, Any]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & (p < right if right < 1.0 else p <= right)
        count = int(mask.sum())
        reliability_rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "count": count,
                "mean_predicted_probability": float(np.mean(p[mask])) if count else "",
                "observed_positive_rate": float(np.mean(y[mask])) if count >= 5 else "",
                "status": "reported" if count >= 5 else "too_few_samples",
            }
        )
    write_csv(run_dir / "android_interview_reliability_table.csv", reliability_rows)

    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibration_status = "reported_descriptive_only_not_used_for_recalibration"
    try:
        from sklearn.linear_model import LogisticRegression

        cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)
        cal.fit(logits, y)
        calibration_intercept = float(cal.intercept_[0])
        calibration_slope = float(cal.coef_[0][0])
    except Exception as exc:  # noqa: BLE001
        calibration_intercept = float("nan")
        calibration_slope = float("nan")
        calibration_status = f"not_available:{exc}"

    report = {
        "created_at": utc_now(),
        "probability_min": float(np.min(p)),
        "probability_max": float(np.max(p)),
        "probability_mean": float(np.mean(p)),
        "probability_std": float(np.std(p, ddof=0)),
        "histogram_bins": [float(x) for x in edges],
        "histogram_counts": [int(x) for x in hist],
        "proportion_predicted_positive_at_0_5": float(np.mean(p >= 0.5)),
        "proportion_predicted_positive_at_frozen_daic_threshold": float(np.mean(p >= FROZEN_DAIC_THRESHOLD)),
        "android_class_prevalence": float(np.mean(y)),
        "brier_score": safe_metric(metrics["brier_score_loss"], y, p),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "calibration_status": calibration_status,
        "reliability_table_path": "android_interview_reliability_table.csv",
        "recalibration_performed": False,
        "android_threshold_derived": False,
    }
    write_json(run_dir / "android_interview_probability_shift_report.json", report)
    return report


def duration_correlation_report(prediction_rows: list[dict[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name, group_rows in [("all", prediction_rows), ("HC", [r for r in prediction_rows if r["condition"] == "HC"]), ("PT", [r for r in prediction_rows if r["condition"] == "PT"])]:
        p = np.asarray([float(row["predicted_probability"]) for row in group_rows], dtype=np.float64)
        duration = np.asarray([float(row["participant_speech_duration_sec"]) for row in group_rows], dtype=np.float64)
        windows = np.asarray([float(row["num_windows"]) for row in group_rows], dtype=np.float64)
        for variable, values in [("participant_speech_duration_sec", duration), ("num_windows", windows)]:
            rows.append(
                {
                    "group": group_name,
                    "variable": variable,
                    "n": len(group_rows),
                    "pearson_correlation": pearson(p, values),
                    "spearman_correlation": spearman(p, values),
                    "status": "reported" if len(group_rows) >= 3 else "too_few_samples",
                }
            )
    write_csv(run_dir / "android_interview_duration_correlation_report.csv", rows)
    return rows


def subgroup_audit(prediction_rows: list[dict[str, Any]], metrics: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_gender: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        gender = clean(row.get("gender")) or "unknown"
        by_gender[gender].append(row)
    for gender, group in sorted(by_gender.items()):
        y = np.asarray([int(row["label_binary"]) for row in group], dtype=int)
        p = np.asarray([float(row["predicted_probability"]) for row in group], dtype=np.float64)
        base = {
            "subgroup": "gender",
            "subgroup_value": gender,
            "participant_count": len(group),
            "HC_count": int(np.sum(y == 0)),
            "PT_count": int(np.sum(y == 1)),
            "threshold": FROZEN_DAIC_THRESHOLD,
            "exploratory_only": True,
        }
        if len(set(y.tolist())) == 2:
            tmetrics = threshold_metrics(y, p, FROZEN_DAIC_THRESHOLD, metrics)
            rows.append(
                {
                    **base,
                    "roc_auc": safe_metric(metrics["roc_auc_score"], y, p),
                    "balanced_accuracy": tmetrics["balanced_accuracy"],
                    "sensitivity": tmetrics["sensitivity"],
                    "specificity": tmetrics["specificity"],
                    "metric_status": "reported",
                }
            )
        else:
            rows.append({**base, "roc_auc": "", "balanced_accuracy": "", "sensitivity": "", "specificity": "", "metric_status": "single_class"})
    write_csv(run_dir / "android_interview_subgroup_audit.csv", rows)
    return rows


def verify_frozen_pipeline(pipeline: Any) -> dict[str, Any]:
    named_steps = getattr(pipeline, "named_steps", {})
    scaler = named_steps.get("scaler")
    classifier = named_steps.get("classifier")
    if scaler is None or classifier is None:
        raise RuntimeError("Frozen Phase 1 WavLM pipeline does not have expected scaler/classifier steps.")
    if getattr(pipeline, "n_features_in_", None) != EXPECTED_DIM:
        raise RuntimeError(f"Frozen pipeline expected input dimension mismatch: {getattr(pipeline, 'n_features_in_', None)}")
    if getattr(classifier, "n_features_in_", None) != EXPECTED_DIM:
        raise RuntimeError(f"Frozen classifier expected input dimension mismatch: {getattr(classifier, 'n_features_in_', None)}")
    if abs(float(getattr(classifier, "C", -1.0)) - 0.001) > 1e-12:
        raise RuntimeError("Frozen classifier C does not match Candidate A.")
    if getattr(classifier, "class_weight", None) != "balanced":
        raise RuntimeError("Frozen classifier class_weight does not match Candidate A.")
    if not hasattr(scaler, "mean_") or not hasattr(classifier, "coef_"):
        raise RuntimeError("Frozen pipeline appears unfitted.")
    return {
        "pipeline_path": str(PHASE1C_PIPELINE),
        "pipeline_class": type(pipeline).__name__,
        "steps": list(named_steps.keys()),
        "scaler_fitted": hasattr(scaler, "mean_"),
        "classifier_fitted": hasattr(classifier, "coef_"),
        "input_dimension": int(getattr(pipeline, "n_features_in_", EXPECTED_DIM)),
        "classifier_C": float(classifier.C),
        "classifier_class_weight": classifier.class_weight,
        "classifier_max_iter": int(classifier.max_iter),
        "android_refit_performed": False,
    }


def run_frozen_inference(
    project_root: Path,
    run_dir: Path,
    participant_rows: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
    pipeline: Any,
    pipeline_verification: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    participant_rows = sorted(participant_rows, key=lambda row: participant_sort_key(row["participant_id"]))
    x = np.stack([embeddings[row["participant_id"]] for row in participant_rows]).astype(np.float32)
    y = np.asarray([int(row["label_binary"]) for row in participant_rows], dtype=int)
    probabilities = pipeline.predict_proba(x)[:, 1].astype(np.float64)
    prediction_rows: list[dict[str, Any]] = []
    for row, probability in zip(participant_rows, probabilities):
        prediction_rows.append(
            {
                "participant_id": row["participant_id"],
                "condition": row["condition"],
                "label_binary": int(row["label_binary"]),
                "label_source": "clinical_group_status",
                "gender": row.get("gender", ""),
                "interview_fold": row.get("interview_fold", ""),
                "num_windows": int(row["num_windows_used"]),
                "participant_speech_duration_sec": float(row["participant_speech_duration_sec"]),
                "predicted_probability": float(probability),
                "prediction_at_0_5": int(probability >= 0.5),
                "frozen_daic_threshold": FROZEN_DAIC_THRESHOLD,
                "prediction_at_frozen_daic_threshold": int(probability >= FROZEN_DAIC_THRESHOLD),
                "model_name": "frozen_daic_candidate_A_wavlm_layer09_mean_logistic_regression",
                "source_training_corpus": "DAIC-WOZ",
                "evaluation_corpus": "Androids Corpus",
                "evaluation_protocol": "zero_shot_cross_corpus_interview",
            }
        )
    write_csv(run_dir / "android_interview_zero_shot_predictions.csv", prediction_rows)

    metrics_0_5 = threshold_metrics(y, probabilities, 0.5, metrics)
    metrics_frozen = threshold_metrics(y, probabilities, FROZEN_DAIC_THRESHOLD, metrics)
    primary = {
        "roc_auc": safe_metric(metrics["roc_auc_score"], y, probabilities),
        "pr_auc": safe_metric(metrics["average_precision_score"], y, probabilities),
        "average_predicted_probability_HC": float(np.mean(probabilities[y == 0])),
        "average_predicted_probability_PT": float(np.mean(probabilities[y == 1])),
        "median_predicted_probability_HC": float(np.median(probabilities[y == 0])),
        "median_predicted_probability_PT": float(np.median(probabilities[y == 1])),
        "participant_count": int(len(y)),
        "HC_count": int(np.sum(y == 0)),
        "PT_count": int(np.sum(y == 1)),
    }

    metric_rows: list[dict[str, Any]] = [
        {"metric_group": "threshold_independent", "threshold_type": "none", "threshold": "", **primary},
        {"metric_group": "threshold_dependent", "threshold_type": "default_0_5", **metrics_0_5},
        {"metric_group": "threshold_dependent", "threshold_type": "frozen_daic_dev_threshold", **metrics_frozen},
    ]
    write_csv(run_dir / "android_interview_zero_shot_metrics.csv", metric_rows)
    bootstrap_rows = bootstrap_intervals(y, probabilities, metrics)
    write_csv(run_dir / "android_interview_bootstrap_intervals.csv", bootstrap_rows)
    probability_shift = reliability_and_probability_shift(y, probabilities, metrics, run_dir)
    duration_rows = duration_correlation_report(prediction_rows, run_dir)
    subgroup_rows = subgroup_audit(prediction_rows, metrics, run_dir)
    return prediction_rows, {"primary": primary, "at_0_5": metrics_0_5, "at_frozen_daic_threshold": metrics_frozen}, bootstrap_rows, probability_shift, duration_rows, subgroup_rows


def daic_android_comparison(
    run_dir: Path,
    android_metrics: dict[str, Any],
    android_count: int,
    android_class_counts: dict[str, int],
) -> list[dict[str, Any]]:
    phase1c = json.loads(PHASE1C_REPORT.read_text(encoding="utf-8"))
    phase1f = json.loads(PHASE1F_REPORT.read_text(encoding="utf-8"))
    dev = phase1c["selected_dev_evaluation"]
    local_test = phase1f["frozen_candidates"]["candidate_A_wavlm_free_speech"]["provisional_local_test_metrics"]
    rows = [
        {
            "dataset": "DAIC-WOZ dev",
            "language": "English",
            "label_definition": "PHQ8_Binary questionnaire threshold",
            "task": "clinical interview participant speech",
            "participant_count": 34,
            "class_counts": "non_depressed=23;depressed=11",
            "roc_auc": dev["dev_metrics_at_selected_threshold"]["roc_auc"],
            "pr_auc": dev["dev_metrics_at_selected_threshold"]["pr_auc"],
            "balanced_accuracy_at_applicable_frozen_threshold": dev["dev_metrics_at_selected_threshold"]["balanced_accuracy"],
            "sensitivity": dev["dev_metrics_at_selected_threshold"]["positive_recall_sensitivity"],
            "specificity": dev["dev_metrics_at_selected_threshold"]["specificity"],
            "comparison_note": "Phase 1 dev; threshold selected on DAIC dev during frozen-candidate creation.",
        },
        {
            "dataset": "DAIC-WOZ provisional local test",
            "language": "English",
            "label_definition": "PHQ8_Binary from local labeled test file; official provenance not established",
            "task": "clinical interview participant speech",
            "participant_count": 47,
            "class_counts": "non_depressed=33;depressed=14",
            "roc_auc": local_test["roc_auc"],
            "pr_auc": local_test["pr_auc"],
            "balanced_accuracy_at_applicable_frozen_threshold": local_test["balanced_accuracy"],
            "sensitivity": local_test["positive_recall_sensitivity"],
            "specificity": local_test["specificity"],
            "comparison_note": "Provisional local-labeled test; not directly equivalent to Android.",
        },
        {
            "dataset": "Androids Corpus interview zero-shot",
            "language": "Italian",
            "label_definition": "clinical patient/control grouping; HC=0, PT=1",
            "task": "interview participant speech clips concatenated",
            "participant_count": android_count,
            "class_counts": f"HC={android_class_counts['HC']};PT={android_class_counts['PT']}",
            "roc_auc": android_metrics["primary"]["roc_auc"],
            "pr_auc": android_metrics["primary"]["pr_auc"],
            "balanced_accuracy_at_applicable_frozen_threshold": android_metrics["at_frozen_daic_threshold"]["balanced_accuracy"],
            "sensitivity": android_metrics["at_frozen_daic_threshold"]["sensitivity"],
            "specificity": android_metrics["at_frozen_daic_threshold"]["specificity"],
            "comparison_note": "External cross-corpus zero-shot; labels, language, and task are not directly equivalent.",
        },
    ]
    write_csv(run_dir / "daic_android_zero_shot_comparison.csv", rows)
    return rows


def classify_transfer(metrics_payload: dict[str, Any], probability_shift: dict[str, Any], duration_rows: list[dict[str, Any]], bootstrap_rows: list[dict[str, Any]]) -> str:
    roc = metrics_payload["primary"]["roc_auc"]
    prob_std = probability_shift["probability_std"]
    pred_pos = probability_shift["proportion_predicted_positive_at_frozen_daic_threshold"]
    roc_ci = next((row for row in bootstrap_rows if row["metric"] == "roc_auc"), {})
    low = float(roc_ci.get("percentile_2_5") or float("nan"))
    high = float(roc_ci.get("percentile_97_5") or float("nan"))
    duration_corrs = [abs(float(row["spearman_correlation"])) for row in duration_rows if row["group"] == "all" and str(row["spearman_correlation"]) != "nan"]
    duration_driven = any(value >= 0.7 for value in duration_corrs)
    collapsed = prob_std < 0.02 or pred_pos <= 0.05 or pred_pos >= 0.95
    if roc >= 0.58 and not (low <= 0.5 <= high) and not duration_driven and not collapsed:
        return "evidence_of_useful_zero_shot_transfer"
    if roc <= 0.55 or collapsed or duration_driven:
        return "weak_or_failed_transfer"
    return "inconclusive_weak_transfer"


def main() -> None:
    args = parse_args()
    add_dependency_dir(args.dependency_dir)
    joblib, torch, torchaudio, transformers_modules, metrics, _ = require_runtime_modules()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    project_root = Path.cwd()
    phase2a_run = resolve_path(args.phase2a_run, project_root)
    if phase2a_run.name != "20260622_154228":
        raise RuntimeError(f"Phase 2B must use corrected Phase 2A run 20260622_154228, got {phase2a_run}.")
    timestamp = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (project_root / args.run_root / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    start_timestamp = utc_now()
    device = choose_cuda(torch)

    window_rows, _, window_quality, excluded_rows, participant_meta, clips = build_android_inputs(project_root, phase2a_run, run_dir, torch, torchaudio)
    clips_by_participant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in clips:
        clip_status = clean(row.get("audio_status"))
        if row.get("mapping_status") == "mapped_to_single_parent" and clip_status in {"", "readable"}:
            clips_by_participant[row["participant_id"]].append(row)

    model, model_metadata = load_pinned_wavlm(device, transformers_modules)
    extraction_metadata = extract_window_embeddings(
        project_root,
        run_dir,
        window_rows,
        clips_by_participant,
        participant_meta,
        model,
        device,
        torch,
        torchaudio,
        args.batch_size,
        args.force,
        model_metadata,
    )

    participant_embedding_rows, participant_embeddings = aggregate_participants(project_root, run_dir, window_rows, participant_meta)
    pipeline = joblib.load(PHASE1C_PIPELINE)
    pipeline_verification = verify_frozen_pipeline(pipeline)
    write_json(run_dir / "frozen_daic_pipeline_verification.json", pipeline_verification)

    prediction_rows, metrics_payload, bootstrap_rows, probability_shift, duration_rows, subgroup_rows = run_frozen_inference(
        project_root,
        run_dir,
        participant_embedding_rows,
        participant_embeddings,
        pipeline,
        pipeline_verification,
        metrics,
    )

    class_counts = dict(Counter(row["condition"] for row in prediction_rows))
    comparison_rows = daic_android_comparison(run_dir, metrics_payload, len(prediction_rows), {"HC": class_counts.get("HC", 0), "PT": class_counts.get("PT", 0)})
    transfer_category = classify_transfer(metrics_payload, probability_shift, duration_rows, bootstrap_rows)

    stereo_full_interviews = [
        row
        for row in read_csv(phase2a_run / "android_recording_manifest.csv")
        if row["task"] == "interview" and clean(row.get("channels_interview") or row.get("num_channels")) == "2"
    ]
    input_hashes = {
        name: sha256_file(phase2a_run / name)
        for name in [
            "android_participant_manifest.csv",
            "android_recording_manifest.csv",
            "android_interview_clip_manifest.csv",
            "android_interview_timing_manifest.csv",
            "android_fold_assignment.csv",
            "daic_android_wavlm_compatibility_report.json",
        ]
    }

    report = {
        "created_at": utc_now(),
        "status": "completed_phase2b_zero_shot_android_interview",
        "phase2a_run_used": rel(phase2a_run, project_root),
        "superseded_phase2a_run_used": False,
        "objective": "Frozen zero-shot external validation of DAIC-WOZ Candidate A on Androids Corpus interview speech.",
        "target_definition_warning": {
            "DAIC_training_target": "PHQ8_Binary questionnaire-threshold label",
            "Android_evaluation_target": "clinical patient/control grouping",
            "related_but_not_equivalent": True,
            "not_clinical_validation": True,
        },
        "eligible_interview_participants": len(prediction_rows),
        "class_counts": class_counts,
        "excluded_participants": excluded_rows,
        "total_windows": len(window_rows),
        "embedding_quality_status": extraction_metadata["status"],
        "zero_shot_metrics": metrics_payload,
        "bootstrap_intervals": bootstrap_rows,
        "probability_shift": probability_shift,
        "duration_correlations": duration_rows,
        "subgroup_audit": subgroup_rows,
        "transfer_interpretation": transfer_category,
        "frozen_daic_pipeline": pipeline_verification,
        "comparison": comparison_rows,
        "strict_scope_confirmation": {
            "android_reading_speech_processed": False,
            "android_training_performed": False,
            "android_scaler_or_classifier_refit": False,
            "android_threshold_tuned": False,
            "probability_recalibration_performed": False,
            "wavlm_layer_changed": False,
            "participant_aggregation_changed": False,
            "phase1_model_modified": False,
            "timing_model_run": False,
        },
    }
    write_json(run_dir / "PHASE2B_ZERO_SHOT_INTERVIEW_REPORT.json", report)

    md = [
        "# Phase 2B Zero-Shot Android Interview Report",
        "",
        f"Run directory: `{rel(run_dir, project_root)}`",
        "",
        "## Scope",
        "Frozen DAIC-WOZ Candidate A was applied once to Androids Corpus interview participant speech. No Android training, scaler fitting, threshold tuning, probability recalibration, WavLM layer selection, or reading-speech processing was performed.",
        "",
        "## Participants And Windows",
        f"- Eligible interview participants: `{len(prediction_rows)}`",
        f"- Class counts: `HC={class_counts.get('HC', 0)}`, `PT={class_counts.get('PT', 0)}`",
        f"- Excluded participants: `{len(excluded_rows)}`",
        f"- Valid 10-second windows: `{len(window_rows)}`",
        f"- Embedding quality status: `{extraction_metadata['status']}`",
        "",
        "## Zero-Shot Metrics",
        f"- ROC-AUC: `{metrics_payload['primary']['roc_auc']:.6f}`",
        f"- PR-AUC: `{metrics_payload['primary']['pr_auc']:.6f}`",
        f"- Balanced accuracy at 0.5: `{metrics_payload['at_0_5']['balanced_accuracy']:.6f}`",
        f"- Macro F1 at 0.5: `{metrics_payload['at_0_5']['macro_f1']:.6f}`",
        f"- Balanced accuracy at frozen DAIC threshold `{FROZEN_DAIC_THRESHOLD}`: `{metrics_payload['at_frozen_daic_threshold']['balanced_accuracy']:.6f}`",
        f"- Macro F1 at frozen DAIC threshold: `{metrics_payload['at_frozen_daic_threshold']['macro_f1']:.6f}`",
        "",
        "## Probability Shift",
        f"- Probability range: `{probability_shift['probability_min']:.6f}` to `{probability_shift['probability_max']:.6f}`",
        f"- Probability mean/std: `{probability_shift['probability_mean']:.6f}` / `{probability_shift['probability_std']:.6f}`",
        f"- Predicted-positive proportion at frozen threshold: `{probability_shift['proportion_predicted_positive_at_frozen_daic_threshold']:.6f}`",
        "",
        "## Interpretation",
        f"- Transfer category: `{transfer_category}`",
        "- DAIC labels and Android clinical-group labels are related but not equivalent; this is not clinical validation.",
        "",
        "## Outputs",
        "- Window, embedding, participant, prediction, bootstrap, probability-shift, duration-correlation, subgroup, and comparison artifacts are in this run directory.",
    ]
    (run_dir / "PHASE2B_ZERO_SHOT_INTERVIEW_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    metadata = {
        "created_at": utc_now(),
        "execution_start_timestamp": start_timestamp,
        "execution_completion_timestamp": utc_now(),
        "phase": "Phase 2B frozen zero-shot external validation",
        "phase2a_run": rel(phase2a_run, project_root),
        "input_hashes": input_hashes,
        "phase1_pipeline_path": rel(PHASE1C_PIPELINE, project_root),
        "phase1_pipeline_sha256": sha256_file(PHASE1C_PIPELINE),
        "wavlm_model": MODEL_NAME,
        "model_commit": MODEL_COMMIT,
        "wavlm_layer": LOCKED_LAYER_NAME,
        "window_length_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS,
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "normalization_method": NORMALIZATION_METHOD,
        "participant_aggregation": "arithmetic mean across Android interview window embeddings",
        "frozen_daic_threshold": FROZEN_DAIC_THRESHOLD,
        "batch_size": args.batch_size,
        "random_seed": RANDOM_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "machine_name": platform.node(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pytorch_version": clean(getattr(torch, "__version__", "")),
        "torchaudio_version": package_version("torchaudio"),
        "transformers_version": package_version("transformers"),
        "sklearn_version": package_version("scikit-learn"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": clean(getattr(torch.version, "cuda", "")),
        "gpu_model": torch.cuda.get_device_name(0),
        "stereo_full_interview_cases": stereo_full_interviews,
        "mono_conversion_rule": "channels averaged sample-wise to mono before resampling",
        "phase2b_report": rel(run_dir / "PHASE2B_ZERO_SHOT_INTERVIEW_REPORT.json", project_root),
        "no_android_training_or_adaptation": True,
        "phase1_model_modified": False,
    }
    write_json(run_dir / "phase2b_run_metadata.json", metadata)

    print("Phase 2B zero-shot Android interview validation complete.")
    print(f"run_dir={rel(run_dir, project_root)}")
    print(f"eligible_participants={len(prediction_rows)} HC={class_counts.get('HC', 0)} PT={class_counts.get('PT', 0)}")
    print(f"total_windows={len(window_rows)}")
    print(f"roc_auc={metrics_payload['primary']['roc_auc']}")
    print(f"pr_auc={metrics_payload['primary']['pr_auc']}")
    print(f"transfer_interpretation={transfer_category}")
    print("android_training_or_adaptation_performed=false")


if __name__ == "__main__":
    main()
