"""Extract frozen WavLM embeddings for clean DAIC-WOZ Phase 1 windows.

This script performs embedding extraction only. It does not train a classifier,
aggregate participant embeddings, tune thresholds, or evaluate depression
prediction performance.

Inputs are the clean manifests created after participant 409 and 422 were
excluded. Transcript text is never used. Windows are reconstructed only from
participant source-audio intervals listed in participant_window_mapping.csv.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


MODEL_NAME = "microsoft/wavlm-base-plus"
LAYER_NAME = "final_hidden_state"
POOLING_METHOD = "mean_valid_time_steps"
EMBEDDING_DTYPE = "float32"
TARGET_SAMPLE_RATE = 16000
TARGET_SAMPLES = 160000
RANDOM_SEED = 1337
NORMALIZATION_METHOD = "float32_dc_offset_removal_peak_normalization_no_dataset_norm_no_augmentation_no_denoising"
MAX_ROUNDING_TOLERANCE_SAMPLES = 400
NEAR_ZERO_NORM_THRESHOLD = 1e-8
EXPECTED_TOTAL_WINDOWS = 17058
EXPECTED_SPLIT_COUNTS = {"train": 9001, "dev": 3314, "test": 4743}


MANIFEST_FIELDS = [
    "dataset",
    "participant_id",
    "window_id",
    "official_split",
    "label_binary",
    "embedding_path",
    "wavlm_model",
    "wavlm_layer",
    "pooling_method",
    "embedding_dimension",
    "embedding_dtype",
    "sample_rate",
    "target_samples",
    "reconstructed_samples_before_adjustment",
    "final_samples",
    "num_source_intervals",
    "normalization_method",
    "extraction_status",
    "error_message",
]


QUALITY_FIELDS = [
    "severity",
    "issue_type",
    "participant_id",
    "window_id",
    "official_split",
    "embedding_path",
    "details",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_daic.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--split", choices=["train", "dev", "test"], default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N clean windows after sorting/filtering.")
    parser.add_argument("--partial-ok", action="store_true", help="Allow expected-count validation to use a partial window subset.")
    parser.add_argument("--embedding-output-dir", default=None, help="Directory for .npy embeddings. Defaults to the legacy config output path.")
    parser.add_argument("--manifest-path", default=None, help="Embedding manifest CSV path.")
    parser.add_argument("--quality-path", default=None, help="Embedding quality report CSV path.")
    parser.add_argument("--metadata-path", default=None, help="Extraction metadata JSON path.")
    parser.add_argument("--model-cache-dir", default=None, help="Project-local Hugging Face model cache directory.")
    parser.add_argument("--model-revision", default=None, help="Optional Hugging Face model revision/checkpoint identifier.")
    parser.add_argument(
        "--dependency-dir",
        default=".phase1_deps",
        help="Optional workspace-local dependency directory to prepend to sys.path.",
    )
    return parser.parse_args()


def add_dependency_dir(path: str) -> None:
    dep_path = Path(path)
    if dep_path.exists():
        sys.path.insert(0, str(dep_path.resolve()))


def require_runtime_modules() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torchaudio
        from transformers import AutoFeatureExtractor, WavLMModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing WavLM extraction dependency. Install torch, torchaudio, and transformers "
            "before running extraction."
        ) from exc
    return torch, torchaudio, (AutoFeatureExtractor, WavLMModel)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def participant_sort_key(participant_id: str) -> tuple[int, str]:
    try:
        return (int(participant_id), participant_id)
    except ValueError:
        return (10**9, participant_id)


def window_sort_key(row: dict[str, str]) -> tuple[int, tuple[int, str], str]:
    split_order = {"train": 0, "dev": 1, "test": 2}
    return (
        split_order.get(clean_value(row.get("official_split")), 99),
        participant_sort_key(clean_value(row.get("participant_id"))),
        clean_value(row.get("window_id")),
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(path)


def append_quality_issue(
    issues: list[dict[str, Any]],
    severity: str,
    issue_type: str,
    row: dict[str, Any],
    embedding_path: str,
    details: str,
) -> None:
    issues.append(
        {
            "severity": severity,
            "issue_type": issue_type,
            "participant_id": clean_value(row.get("participant_id")),
            "window_id": clean_value(row.get("window_id")),
            "official_split": clean_value(row.get("official_split")),
            "embedding_path": embedding_path,
            "details": details,
        }
    )


def load_clean_inputs(output_dir: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
    participant_manifest = output_dir / "daic_phase1_participant_manifest_clean.csv"
    window_manifest = output_dir / "daic_phase1_window_manifest_clean.csv"
    mapping_path = output_dir / "participant_window_mapping.csv"

    participants = {row["participant_id"]: row for row in read_csv(participant_manifest)}
    windows = sorted(read_csv(window_manifest), key=window_sort_key)
    mapping_by_window: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(mapping_path):
        mapping_by_window[clean_value(row["window_id"])].append(row)

    for rows in mapping_by_window.values():
        rows.sort(key=lambda item: int(clean_value(item["source_interval_index"])))

    return windows, participants, mapping_by_window


def path_for_display(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def add_existing_path_candidate(candidates: list[Path], raw_path: str, project_root: Path) -> None:
    cleaned = clean_value(raw_path)
    if not cleaned:
        return
    path = Path(cleaned)
    candidates.append(path if path.is_absolute() else project_root / path)


def resolve_participant_audio_paths(
    participants: dict[str, dict[str, str]],
    config: dict[str, Any],
    project_root: Path,
) -> list[tuple[str, str]]:
    dataset_root_value = clean_value(config.get("dataset_root"))
    dataset_root = Path(dataset_root_value) if dataset_root_value else None
    if dataset_root is not None and not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    missing: list[tuple[str, str]] = []
    for participant_id, row in participants.items():
        raw_audio_path = clean_value(row.get("audio_path"))
        raw_participant_dir = clean_value(row.get("participant_dir"))
        audio_name = Path(raw_audio_path).name or f"{participant_id}_AUDIO.wav"
        participant_dir_name = Path(raw_participant_dir).name or f"{participant_id}_P"

        candidates: list[Path] = []
        add_existing_path_candidate(candidates, raw_audio_path, project_root)
        if dataset_root is not None:
            candidates.append(dataset_root / participant_dir_name / audio_name)

        resolved = next((candidate for candidate in candidates if candidate.exists()), None)
        if resolved is None:
            missing.append((participant_id, raw_audio_path))
            continue

        row["_resolved_audio_path"] = str(resolved)
        row["audio_path"] = path_for_display(resolved, project_root)

    return missing


def read_wav_mono_float(path: Path) -> tuple[np.ndarray, int]:
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
    return audio, sample_rate


class AudioCache:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[np.ndarray, int]] = {}

    def get(self, audio_path: Path) -> tuple[np.ndarray, int]:
        key = str(audio_path)
        if key not in self._cache:
            self._cache[key] = read_wav_mono_float(audio_path)
        return self._cache[key]


def resample_if_needed(waveform: np.ndarray, sample_rate: int, target_rate: int, torch: Any, torchaudio: Any) -> np.ndarray:
    if sample_rate == target_rate:
        return waveform.astype(np.float32, copy=False)
    tensor = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, sample_rate, target_rate)
    return resampled.squeeze(0).cpu().numpy().astype(np.float32)


def reconstruct_window(
    window: dict[str, str],
    participant: dict[str, str],
    mappings: list[dict[str, str]],
    audio_cache: AudioCache,
    torch: Any,
    torchaudio: Any,
) -> tuple[np.ndarray, int, list[str]]:
    issues: list[str] = []
    participant_id = clean_value(window["participant_id"])
    window_id = clean_value(window["window_id"])
    audio_path = Path(clean_value(participant.get("_resolved_audio_path") or participant.get("audio_path")))
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio file for participant {participant_id}: {audio_path}")
    if not mappings:
        raise ValueError(f"No source mappings for window {window_id}")

    waveform, sample_rate = audio_cache.get(audio_path)
    fragments: list[np.ndarray] = []
    for mapping in mappings:
        if clean_value(mapping.get("participant_id")) != participant_id:
            raise ValueError(f"Mapping participant mismatch for window {window_id}")
        if clean_value(mapping.get("window_id")) != window_id:
            raise ValueError(f"Mapping window mismatch for window {window_id}")

        source_start = float(clean_value(mapping["source_start_sec"]))
        source_end = float(clean_value(mapping["source_end_sec"]))
        if source_end <= source_start:
            raise ValueError(f"Nonpositive source interval for window {window_id}")

        start_sample = int(round(source_start * sample_rate))
        end_sample = int(round(source_end * sample_rate))
        start_sample = max(0, min(start_sample, len(waveform)))
        end_sample = max(0, min(end_sample, len(waveform)))
        if end_sample <= start_sample:
            raise ValueError(f"Rounded source interval is empty for window {window_id}")
        fragments.append(waveform[start_sample:end_sample])

    reconstructed = np.concatenate(fragments).astype(np.float32, copy=False)
    reconstructed = resample_if_needed(reconstructed, sample_rate, TARGET_SAMPLE_RATE, torch, torchaudio)
    samples_before_adjustment = int(reconstructed.shape[0])
    difference = samples_before_adjustment - TARGET_SAMPLES

    if abs(difference) > MAX_ROUNDING_TOLERANCE_SAMPLES:
        issues.append(
            f"large_duration_mismatch_before_adjustment={samples_before_adjustment}; target={TARGET_SAMPLES}; tolerance={MAX_ROUNDING_TOLERANCE_SAMPLES}"
        )

    if samples_before_adjustment > TARGET_SAMPLES:
        reconstructed = reconstructed[:TARGET_SAMPLES]
    elif samples_before_adjustment < TARGET_SAMPLES:
        reconstructed = np.pad(reconstructed, (0, TARGET_SAMPLES - samples_before_adjustment))

    if reconstructed.shape[0] != TARGET_SAMPLES:
        raise ValueError(f"Final sample count is not {TARGET_SAMPLES} for window {window_id}")

    return reconstructed.astype(np.float32, copy=False), samples_before_adjustment, issues


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    normalized = waveform.astype(np.float32, copy=True)
    normalized -= float(np.mean(normalized))
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 0.0:
        normalized /= peak
    return normalized.astype(np.float32, copy=False)


def choose_device(requested: str, torch: Any) -> Any:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_wavlm_model(
    device: Any,
    transformers_modules: tuple[Any, Any],
    model_cache_dir: Path | None,
    model_revision: str | None,
) -> tuple[Any, Any, int]:
    AutoFeatureExtractor, WavLMModel = transformers_modules
    pretrained_kwargs: dict[str, Any] = {}
    if model_cache_dir is not None:
        pretrained_kwargs["cache_dir"] = str(model_cache_dir)
    if model_revision:
        pretrained_kwargs["revision"] = model_revision

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME, **pretrained_kwargs)
    model = WavLMModel.from_pretrained(MODEL_NAME, **pretrained_kwargs)
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hidden_size = int(model.config.hidden_size)
    return feature_extractor, model, hidden_size


def valid_embedding(path: Path, expected_dim: int | None) -> tuple[bool, str, int | None]:
    if not path.exists():
        return False, "missing", None
    try:
        embedding = np.load(path)
    except Exception as exc:  # noqa: BLE001 - corruption details are useful in report.
        return False, f"load_failed:{exc}", None
    if embedding.ndim != 1:
        return False, f"ndim={embedding.ndim}", None
    if embedding.size == 0:
        return False, "empty", None
    if expected_dim is not None and embedding.shape[0] != expected_dim:
        return False, f"dimension={embedding.shape[0]} expected={expected_dim}", int(embedding.shape[0])
    if not np.isfinite(embedding).all():
        return False, "non_finite_values", int(embedding.shape[0])
    if np.all(embedding == 0):
        return False, "all_zero", int(embedding.shape[0])
    return True, "ok", int(embedding.shape[0])


def extract_batch_embeddings(batch_waveforms: list[np.ndarray], model: Any, device: Any, torch: Any) -> np.ndarray:
    batch_array = np.stack(batch_waveforms, axis=0).astype(np.float32)
    inputs = torch.from_numpy(batch_array).to(device)
    with torch.inference_mode():
        outputs = model(inputs, output_hidden_states=False, return_dict=True)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
    return pooled.detach().cpu().numpy().astype(np.float32)


def make_manifest_row(
    window: dict[str, str],
    embedding_path: Path,
    embedding_dim: int | str,
    reconstructed_samples: int | str,
    status: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "dataset": "DAIC-WOZ",
        "participant_id": clean_value(window.get("participant_id")),
        "window_id": clean_value(window.get("window_id")),
        "official_split": clean_value(window.get("official_split")),
        "label_binary": clean_value(window.get("label_binary")),
        "embedding_path": str(embedding_path),
        "wavlm_model": MODEL_NAME,
        "wavlm_layer": LAYER_NAME,
        "pooling_method": POOLING_METHOD,
        "embedding_dimension": embedding_dim,
        "embedding_dtype": EMBEDDING_DTYPE,
        "sample_rate": TARGET_SAMPLE_RATE,
        "target_samples": TARGET_SAMPLES,
        "reconstructed_samples_before_adjustment": reconstructed_samples,
        "final_samples": TARGET_SAMPLES if status in {"success", "skipped_valid"} else "",
        "num_source_intervals": clean_value(window.get("num_source_intervals")),
        "normalization_method": NORMALIZATION_METHOD,
        "extraction_status": status,
        "error_message": error_message,
    }


def flush_progress(
    manifest_path: Path,
    manifest_rows: dict[str, dict[str, Any]],
    quality_path: Path,
    quality_rows: list[dict[str, Any]],
) -> None:
    rows = sorted(manifest_rows.values(), key=window_sort_key)
    write_csv_atomic(manifest_path, rows, MANIFEST_FIELDS)
    write_csv_atomic(quality_path, quality_rows, QUALITY_FIELDS)


def get_package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def validate_final_outputs(
    windows: list[dict[str, str]],
    manifest_rows: dict[str, dict[str, Any]],
    embedding_dir: Path,
    expected_dim: int | None,
    quality_rows: list[dict[str, Any]],
    enforce_full_expected_counts: bool,
) -> dict[str, Any]:
    clean_window_ids = {clean_value(row["window_id"]) for row in windows}
    manifest_window_ids = set(manifest_rows)
    split_counts = Counter(clean_value(row.get("official_split")) for row in manifest_rows.values())
    successful_rows = [row for row in manifest_rows.values() if row.get("extraction_status") in {"success", "skipped_valid"}]
    failed_rows = [row for row in manifest_rows.values() if row.get("extraction_status") == "failed"]

    if enforce_full_expected_counts and len(clean_window_ids) != EXPECTED_TOTAL_WINDOWS:
        quality_rows.append(
            {
                "severity": "error",
                "issue_type": "unexpected_clean_window_total",
                "participant_id": "",
                "window_id": "",
                "official_split": "",
                "embedding_path": "",
                "details": f"Expected {EXPECTED_TOTAL_WINDOWS} clean windows, found {len(clean_window_ids)}.",
            }
        )

    missing = sorted(clean_window_ids - manifest_window_ids)
    unexpected = sorted(manifest_window_ids - clean_window_ids)
    duplicates = len(manifest_rows) != len(manifest_window_ids)
    if missing:
        for window_id in missing[:100]:
            quality_rows.append(
                {
                    "severity": "error",
                    "issue_type": "missing_manifest_row",
                    "participant_id": "",
                    "window_id": window_id,
                    "official_split": "",
                    "embedding_path": "",
                    "details": "Clean window has no embedding manifest row.",
                }
            )
    if unexpected:
        for window_id in unexpected[:100]:
            quality_rows.append(
                {
                    "severity": "error",
                    "issue_type": "unexpected_manifest_row",
                    "participant_id": "",
                    "window_id": window_id,
                    "official_split": "",
                    "embedding_path": "",
                    "details": "Embedding manifest has a row for an unexpected window.",
                }
            )
    if duplicates:
        quality_rows.append(
            {
                "severity": "error",
                "issue_type": "duplicate_window_id",
                "participant_id": "",
                "window_id": "",
                "official_split": "",
                "embedding_path": "",
                "details": "Duplicate window IDs detected in manifest rows.",
            }
        )

    for excluded_participant in ("409", "422"):
        matches = list(embedding_dir.glob(f"{excluded_participant}_w*.npy"))
        if matches:
            quality_rows.append(
                {
                    "severity": "error",
                    "issue_type": "excluded_participant_embedding_present",
                    "participant_id": excluded_participant,
                    "window_id": "",
                    "official_split": "",
                    "embedding_path": str(matches[0]),
                    "details": f"Found embeddings for excluded participant {excluded_participant}.",
                }
            )

    dimensions: list[int] = []
    norms: list[float] = []
    near_zero = 0
    for row in successful_rows:
        path = Path(clean_value(row.get("embedding_path")))
        ok, reason, dim = valid_embedding(path, expected_dim)
        if not ok:
            quality_rows.append(
                {
                    "severity": "error",
                    "issue_type": "invalid_embedding_file",
                    "participant_id": clean_value(row.get("participant_id")),
                    "window_id": clean_value(row.get("window_id")),
                    "official_split": clean_value(row.get("official_split")),
                    "embedding_path": str(path),
                    "details": reason,
                }
            )
            continue
        embedding = np.load(path)
        dimensions.append(int(embedding.shape[0]))
        norm = float(np.linalg.norm(embedding))
        norms.append(norm)
        if norm < NEAR_ZERO_NORM_THRESHOLD:
            near_zero += 1

    if enforce_full_expected_counts:
        for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
            if split_counts.get(split, 0) != expected_count:
                quality_rows.append(
                    {
                        "severity": "error",
                        "issue_type": "split_count_mismatch",
                        "participant_id": "",
                        "window_id": "",
                        "official_split": split,
                        "embedding_path": "",
                        "details": f"Expected {expected_count} rows for split {split}, found {split_counts.get(split, 0)}.",
                    }
                )

    if quality_rows == []:
        quality_rows.append(
            {
                "severity": "info",
                "issue_type": "all_quality_checks_passed",
                "participant_id": "",
                "window_id": "",
                "official_split": "",
                "embedding_path": "",
                "details": "All WavLM embedding quality checks passed.",
            }
        )

    norm_array = np.array(norms, dtype=np.float64) if norms else np.array([], dtype=np.float64)
    return {
        "total_manifest_rows": len(manifest_rows),
        "total_successful_or_skipped": len(successful_rows),
        "total_failed": len(failed_rows),
        "counts_by_split": dict(split_counts),
        "embedding_dimensions": sorted(set(dimensions)),
        "embedding_norm_min": float(norm_array.min()) if norm_array.size else None,
        "embedding_norm_max": float(norm_array.max()) if norm_array.size else None,
        "embedding_norm_mean": float(norm_array.mean()) if norm_array.size else None,
        "embedding_norm_std": float(norm_array.std()) if norm_array.size else None,
        "near_zero_embeddings": near_zero,
    }


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    config_path: Path,
    config: dict[str, Any],
    device: Any,
    batch_size: int,
    embedding_dir: Path,
    model_cache_dir: Path | None,
    model_config_commit_hash: str,
    start_timestamp: str,
    embedding_dim: int | None,
    extracted: int,
    skipped: int,
    failed: int,
    validation_stats: dict[str, Any],
) -> None:
    torch_module = sys.modules.get("torch")
    cuda_available = bool(torch_module.cuda.is_available()) if torch_module is not None else False
    gpu_model = ""
    gpu_memory_free = None
    gpu_memory_total = None
    if cuda_available and str(device).startswith("cuda"):
        gpu_model = torch_module.cuda.get_device_name(device)
        gpu_memory_free, gpu_memory_total = torch_module.cuda.mem_get_info(device)
    completion_timestamp = datetime.now(timezone.utc).isoformat()
    metadata = {
        "execution_start_timestamp": start_timestamp,
        "execution_completion_timestamp": completion_timestamp,
        "execution_timestamp": completion_timestamp,
        "machine_name": platform.node(),
        "source_code_version_git_commit": get_git_commit(),
        "config_path": str(config_path),
        "clean_manifest_paths": {
            "participant_manifest_clean": str(Path(config["output_dir"]) / "daic_phase1_participant_manifest_clean.csv"),
            "window_manifest_clean": str(Path(config["output_dir"]) / "daic_phase1_window_manifest_clean.csv"),
            "split_summary_clean": str(Path(config["output_dir"]) / "daic_phase1_split_summary_clean.csv"),
        },
        "mapping_path": str(Path(config["output_dir"]) / "participant_window_mapping.csv"),
        "wavlm_model_name": MODEL_NAME,
        "model_checkpoint_identifier": MODEL_NAME,
        "model_revision": args.model_revision or "default",
        "model_config_commit_hash": model_config_commit_hash,
        "model_cache_dir": str(model_cache_dir) if model_cache_dir is not None else "",
        "transformers_version": get_package_version("transformers"),
        "pytorch_version": get_package_version("torch"),
        "torchaudio_version": get_package_version("torchaudio"),
        "device": str(device),
        "cuda_available": cuda_available,
        "cuda_version": getattr(__import__("torch"), "version").cuda if "torch" in sys.modules else "",
        "gpu_model": gpu_model,
        "gpu_memory_free_bytes_at_completion": gpu_memory_free,
        "gpu_memory_total_bytes": gpu_memory_total,
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "random_seed": RANDOM_SEED,
        "layer_used": LAYER_NAME,
        "pooling_method": POOLING_METHOD,
        "waveform_normalization_method": NORMALIZATION_METHOD,
        "sample_rate": TARGET_SAMPLE_RATE,
        "target_sample_count": TARGET_SAMPLES,
        "max_rounding_tolerance_samples": MAX_ROUNDING_TOLERANCE_SAMPLES,
        "embedding_dimension": embedding_dim,
        "embedding_dtype": EMBEDDING_DTYPE,
        "total_expected_windows": EXPECTED_TOTAL_WINDOWS,
        "total_successfully_extracted": extracted,
        "total_skipped_during_resume": skipped,
        "total_failed": failed,
        "counts_by_split": validation_stats.get("counts_by_split", {}),
        "quality_statistics": validation_stats,
        "output_directory": str(embedding_dir),
        "stop_condition": "embedding_extraction_only_no_model_training",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    add_dependency_dir(args.dependency_dir)
    torch, torchaudio, transformers_modules = require_runtime_modules()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    config_path = Path(args.config)
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    project_root = Path.cwd()
    embedding_dir = Path(args.embedding_output_dir) if args.embedding_output_dir else output_dir / "wavlm_embeddings" / "final_layer_mean"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest_path) if args.manifest_path else output_dir / "wavlm_embedding_manifest.csv"
    quality_path = Path(args.quality_path) if args.quality_path else output_dir / "wavlm_embedding_quality_report.csv"
    metadata_path = Path(args.metadata_path) if args.metadata_path else output_dir / "wavlm_extraction_metadata.json"
    model_cache_dir = Path(args.model_cache_dir) if args.model_cache_dir else None
    if model_cache_dir is not None:
        model_cache_dir.mkdir(parents=True, exist_ok=True)

    windows, participants, mapping_by_window = load_clean_inputs(output_dir)
    if args.split:
        windows = [row for row in windows if clean_value(row.get("official_split")) == args.split]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be a positive integer.")
        windows = windows[: args.limit]

    missing_audio = resolve_participant_audio_paths(participants, config, project_root)
    if missing_audio:
        examples = "; ".join(f"{participant_id}: {audio_path}" for participant_id, audio_path in missing_audio[:5])
        raise RuntimeError(f"{len(missing_audio)} participant audio paths could not be resolved. Examples: {examples}")

    device = choose_device(args.device, torch)
    _, model, embedding_dim = load_wavlm_model(device, transformers_modules, model_cache_dir, args.model_revision)
    model_config_commit_hash = clean_value(getattr(model.config, "_commit_hash", ""))

    manifest_rows: dict[str, dict[str, Any]] = {}
    quality_rows: list[dict[str, Any]] = []
    audio_cache = AudioCache()
    batch_payload: list[tuple[dict[str, str], Path, int, list[str], np.ndarray]] = []
    extracted = 0
    skipped = 0
    failed = 0
    processed_since_flush = 0

    def process_batch() -> None:
        nonlocal extracted, processed_since_flush
        if not batch_payload:
            return
        waveforms = [item[4] for item in batch_payload]
        embeddings = extract_batch_embeddings(waveforms, model, device, torch)
        for (window, embedding_path, reconstructed_samples, reconstruction_issues, _), embedding in zip(batch_payload, embeddings):
            if embedding.shape != (embedding_dim,):
                raise RuntimeError(f"Unexpected embedding shape {embedding.shape} for {window['window_id']}")
            np.save(embedding_path, embedding.astype(np.float32, copy=False))
            manifest_rows[clean_value(window["window_id"])] = make_manifest_row(
                window,
                embedding_path,
                embedding_dim,
                reconstructed_samples,
                "success",
                "; ".join(reconstruction_issues),
            )
            if reconstruction_issues:
                append_quality_issue(
                    quality_rows,
                    "warning",
                    "waveform_duration_adjustment_outside_tolerance",
                    window,
                    str(embedding_path),
                    "; ".join(reconstruction_issues),
                )
            extracted += 1
            processed_since_flush += 1
        batch_payload.clear()

    start_timestamp = datetime.now(timezone.utc).isoformat()
    start_time = time.time()
    for index, window in enumerate(windows, start=1):
        window_id = clean_value(window["window_id"])
        participant_id = clean_value(window["participant_id"])
        embedding_path = embedding_dir / f"{window_id}.npy"
        participant = participants.get(participant_id)
        if participant is None:
            manifest_rows[window_id] = make_manifest_row(window, embedding_path, "", "", "failed", "participant_not_found")
            append_quality_issue(quality_rows, "error", "participant_not_found", window, str(embedding_path), "Participant missing from clean manifest.")
            failed += 1
            continue

        if args.resume and not args.force:
            is_valid, reason, existing_dim = valid_embedding(embedding_path, embedding_dim)
            if is_valid:
                manifest_rows[window_id] = make_manifest_row(
                    window,
                    embedding_path,
                    existing_dim or embedding_dim,
                    "",
                    "skipped_valid",
                    "",
                )
                skipped += 1
                processed_since_flush += 1
                if processed_since_flush >= 100:
                    flush_progress(manifest_path, manifest_rows, quality_path, quality_rows)
                    processed_since_flush = 0
                continue
            if embedding_path.exists() and reason != "missing":
                append_quality_issue(quality_rows, "warning", "recomputing_invalid_embedding", window, str(embedding_path), reason)

        try:
            waveform, reconstructed_samples, reconstruction_issues = reconstruct_window(
                window,
                participant,
                mapping_by_window.get(window_id, []),
                audio_cache,
                torch,
                torchaudio,
            )
            waveform = normalize_waveform(waveform)
            batch_payload.append((window, embedding_path, reconstructed_samples, reconstruction_issues, waveform))
            if len(batch_payload) >= args.batch_size:
                process_batch()
        except Exception as exc:  # noqa: BLE001 - continue and report all failures.
            manifest_rows[window_id] = make_manifest_row(window, embedding_path, "", "", "failed", str(exc))
            append_quality_issue(quality_rows, "error", "window_extraction_failed", window, str(embedding_path), str(exc))
            failed += 1
            processed_since_flush += 1

        if processed_since_flush >= 100:
            flush_progress(manifest_path, manifest_rows, quality_path, quality_rows)
            processed_since_flush = 0

        if index % 250 == 0:
            elapsed = time.time() - start_time
            print(
                f"processed={index}/{len(windows)} extracted={extracted} skipped={skipped} failed={failed} elapsed_sec={elapsed:.1f}",
                flush=True,
            )

    try:
        process_batch()
    except Exception as exc:  # noqa: BLE001
        for window, embedding_path, _, _, _ in batch_payload:
            manifest_rows[clean_value(window["window_id"])] = make_manifest_row(window, embedding_path, "", "", "failed", str(exc))
            append_quality_issue(quality_rows, "error", "batch_embedding_failed", window, str(embedding_path), str(exc))
            failed += 1
        batch_payload.clear()

    validation_stats = validate_final_outputs(windows, manifest_rows, embedding_dir, embedding_dim, quality_rows, not args.partial_ok)
    flush_progress(manifest_path, manifest_rows, quality_path, quality_rows)
    write_metadata(
        metadata_path,
        args,
        config_path,
        config,
        device,
        args.batch_size,
        embedding_dir,
        model_cache_dir,
        model_config_commit_hash,
        start_timestamp,
        embedding_dim,
        extracted,
        skipped,
        failed,
        validation_stats,
    )

    print("WavLM embedding extraction finished. No model training was run.")
    print(f"expected_windows={len(windows)} extracted={extracted} skipped={skipped} failed={failed} embedding_dim={embedding_dim}")
    print(f"manifest={manifest_path}")
    print(f"quality_report={quality_path}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
