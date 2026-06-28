"""Phase 1B frozen WavLM layer-wise representation analysis.

This run extracts one 12x768 layer matrix per clean window, aggregates each
participant by arithmetic mean per layer, selects one layer using train-only
nested cross-validation, evaluates that selected layer once on dev, and stops.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DEPENDENCY_DIR = Path(".phase1_deps")
if DEFAULT_DEPENDENCY_DIR.exists():
    sys.path.insert(0, str(DEFAULT_DEPENDENCY_DIR.resolve()))

import joblib
import numpy as np
import pandas as pd
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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from extract_wavlm_embeddings import (
    AudioCache,
    clean_value,
    load_clean_inputs,
    load_config,
    normalize_waveform,
    reconstruct_window,
    require_runtime_modules,
    resolve_participant_audio_paths,
    window_sort_key,
)
from run_participant_baselines import (
    make_issue,
    participant_sort_key,
    path_relative_to_root,
    read_csv_dicts,
    resolve_path,
    sha256_file,
    write_csv,
    write_json,
)


MODEL_NAME = "microsoft/wavlm-base-plus"
MODEL_COMMIT = "4c66d4806a428f2e922ccfa1a962776e232d487b"
EXPECTED_TOTAL_WINDOWS = 17058
EXPECTED_WINDOW_SPLITS = {"train": 9001, "dev": 3314, "test": 4743}
EXPECTED_PARTICIPANT_SPLITS = {"train": 106, "dev": 34, "test": 47}
EXPECTED_NUM_TRANSFORMER_LAYERS = 12
EXPECTED_HIDDEN_DIM = 768
EXPECTED_MATRIX_SHAPE = (EXPECTED_NUM_TRANSFORMER_LAYERS, EXPECTED_HIDDEN_DIM)
EXPECTED_DTYPE = "float32"
TARGET_SAMPLE_RATE = 16000
TARGET_SAMPLES = 160000
EXCLUDED_PARTICIPANTS = {"409", "422"}
RANDOM_SEED = 42
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
OUTER_SPLITS = 5
OUTER_REPEATS = 5
FINAL_C_SPLITS = 5
FINAL_C_REPEATS = 10


WINDOW_MANIFEST_FIELDS = [
    "participant_id",
    "window_id",
    "official_split",
    "label_binary",
    "embedding_path",
    "matrix_shape",
    "num_transformer_layers",
    "hidden_dimension",
    "dtype",
    "model_name",
    "model_commit",
    "extraction_status",
    "error_message",
]


PARTICIPANT_MANIFEST_FIELDS = [
    "participant_id",
    "official_split",
    "label_binary",
    "embedding_path",
    "matrix_shape",
    "num_transformer_layers",
    "hidden_dimension",
    "dtype",
    "num_windows_expected",
    "num_windows_used",
    "aggregation_method",
    "aggregation_status",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_daic.json")
    parser.add_argument("--output-dir", default="outputs/phase1_daic")
    parser.add_argument("--run-root", default="outputs/phase1_daic/phase1b_layerwise_runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dependency-dir", default=".phase1_deps")
    parser.add_argument("--model-cache-dir", default="outputs/phase1_daic/model_cache/huggingface")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def add_dependency_dir(path: str) -> None:
    dep_path = Path(path)
    if dep_path.exists():
        resolved = str(dep_path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


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


def matrix_shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(value) for value in shape)


def write_manifest_and_quality(manifest_path: Path, manifest_rows: dict[str, dict[str, Any]], quality_path: Path, quality_rows: list[dict[str, Any]]) -> None:
    rows = sorted(manifest_rows.values(), key=window_sort_key)
    write_csv(manifest_path, rows, WINDOW_MANIFEST_FIELDS)
    write_csv(quality_path, quality_rows or [make_issue("info", "no_quality_issues_recorded", "No quality issues recorded.")])


def valid_layerwise_matrix(path: Path, expected_shape: tuple[int, int] = EXPECTED_MATRIX_SHAPE) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        matrix = np.load(path)
    except Exception as exc:  # noqa: BLE001
        return False, f"load_failed:{exc}"
    if matrix.shape != expected_shape:
        return False, f"shape={matrix.shape} expected={expected_shape}"
    if str(matrix.dtype) != EXPECTED_DTYPE:
        return False, f"dtype={matrix.dtype} expected={EXPECTED_DTYPE}"
    if not np.isfinite(matrix).all():
        return False, "nonfinite_values"
    if np.any(np.all(matrix == 0, axis=1)):
        return False, "all_zero_layer_vector"
    return True, "ok"


def choose_cuda_device(torch: Any) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 1B extraction, but torch.cuda.is_available() is False.")
    return torch.device("cuda")


def load_pinned_wavlm(device: Any, transformers_modules: tuple[Any, Any], model_cache_dir: Path) -> tuple[Any, int, int, dict[str, Any]]:
    AutoFeatureExtractor, WavLMModel = transformers_modules
    kwargs = {
        "revision": MODEL_COMMIT,
        "cache_dir": str(model_cache_dir),
        "local_files_only": True,
    }
    AutoFeatureExtractor.from_pretrained(MODEL_NAME, **kwargs)
    model = WavLMModel.from_pretrained(MODEL_NAME, **kwargs)
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    num_layers = int(model.config.num_hidden_layers)
    hidden_dim = int(model.config.hidden_size)
    hidden_state_index_map = {
        "hidden_states[0]": "pre-transformer hidden representation after convolutional feature extraction/projection; excluded from primary layer analysis",
    }
    for layer_number in range(1, num_layers + 1):
        hidden_state_index_map[f"hidden_states[{layer_number}]"] = f"transformer encoder layer {layer_number:02d}; saved row {layer_number - 1}"

    metadata = {
        "model_name": MODEL_NAME,
        "requested_model_commit": MODEL_COMMIT,
        "loaded_model_config_commit_hash": clean_value(getattr(model.config, "_commit_hash", "")),
        "num_hidden_layers_from_config": num_layers,
        "hidden_size_from_config": hidden_dim,
        "hidden_state_index_map": hidden_state_index_map,
    }
    if num_layers != EXPECTED_NUM_TRANSFORMER_LAYERS or hidden_dim != EXPECTED_HIDDEN_DIM:
        raise RuntimeError(f"Unexpected WavLM dimensions from config: layers={num_layers}, hidden_dim={hidden_dim}")
    if metadata["loaded_model_config_commit_hash"] and metadata["loaded_model_config_commit_hash"] != MODEL_COMMIT:
        raise RuntimeError(f"Loaded model commit mismatch: {metadata['loaded_model_config_commit_hash']} != {MODEL_COMMIT}")
    return model, num_layers, hidden_dim, metadata


def extract_layerwise_batch(
    waveforms: list[np.ndarray],
    model: Any,
    device: Any,
    torch: Any,
    num_layers: int,
    hidden_dim: int,
) -> tuple[np.ndarray, list[tuple[int, ...]], tuple[int, ...]]:
    batch_array = np.stack(waveforms, axis=0).astype(np.float32)
    inputs = torch.from_numpy(batch_array).to(device)
    with torch.inference_mode():
        outputs = model(inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        hidden_shapes = [tuple(state.shape) for state in hidden_states]
        if len(hidden_states) != num_layers + 1:
            raise RuntimeError(f"Expected {num_layers + 1} hidden states, got {len(hidden_states)}.")
        layer_states = hidden_states[1:]
        for index, state in enumerate(layer_states, start=1):
            if state.shape[0] != len(waveforms) or state.shape[-1] != hidden_dim:
                raise RuntimeError(f"Unexpected hidden state shape at transformer layer {index}: {tuple(state.shape)}")
        pooled_layers = torch.stack([state.mean(dim=1) for state in layer_states], dim=1)
        final_shape = tuple(outputs.last_hidden_state.shape)
    return pooled_layers.detach().cpu().numpy().astype(np.float32), hidden_shapes, final_shape


def input_validation(
    root: Path,
    config: dict[str, Any],
    output_dir: Path,
    windows: list[dict[str, str]],
    participants: dict[str, dict[str, str]],
    mapping_by_window: dict[str, list[dict[str, str]]],
    torch: Any,
    torchaudio: Any,
    run_dir: Path,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    split_counts = Counter(clean_value(row.get("official_split")) for row in windows)
    participant_ids = {clean_value(row.get("participant_id")) for row in windows}
    mapping_window_ids = set(mapping_by_window)
    clean_window_ids = {clean_value(row["window_id"]) for row in windows}

    if len(windows) != EXPECTED_TOTAL_WINDOWS:
        issues.append(make_issue("error", "clean_window_total_mismatch", f"Expected {EXPECTED_TOTAL_WINDOWS}, found {len(windows)}."))
    if dict(split_counts) != EXPECTED_WINDOW_SPLITS:
        issues.append(make_issue("error", "clean_window_split_mismatch", f"Expected {EXPECTED_WINDOW_SPLITS}, found {dict(split_counts)}."))
    excluded = sorted(participant_ids & EXCLUDED_PARTICIPANTS, key=participant_sort_key)
    if excluded:
        issues.append(make_issue("error", "excluded_participants_present", f"Participants present: {excluded}."))
    if clean_window_ids - mapping_window_ids:
        issues.append(make_issue("error", "missing_window_mapping", f"Missing mapping rows for {len(clean_window_ids - mapping_window_ids)} windows."))
    if mapping_window_ids - clean_window_ids:
        issues.append(make_issue("error", "unexpected_window_mapping", f"Unexpected mapping rows for {len(mapping_window_ids - clean_window_ids)} windows."))

    missing_audio = resolve_participant_audio_paths(participants, config, root)
    if missing_audio:
        issues.append(make_issue("error", "missing_audio_paths", f"Could not resolve {len(missing_audio)} participant audio files.", examples=missing_audio[:5]))

    audio_cache = AudioCache()
    reconstructed_count = 0
    duration_adjustment_warnings = 0
    max_source_intervals = 0
    for window in windows:
        participant_id = clean_value(window["participant_id"])
        window_id = clean_value(window["window_id"])
        participant = participants.get(participant_id)
        if participant is None:
            issues.append(make_issue("error", "participant_missing_from_manifest", f"{participant_id}", participant_id=participant_id, window_id=window_id))
            continue
        mappings = mapping_by_window.get(window_id, [])
        max_source_intervals = max(max_source_intervals, len(mappings))
        try:
            waveform, _, reconstruction_issues = reconstruct_window(window, participant, mappings, audio_cache, torch, torchaudio)
        except Exception as exc:  # noqa: BLE001
            issues.append(make_issue("error", "window_reconstruction_failed", f"{window_id}: {exc}", participant_id=participant_id, window_id=window_id))
            continue
        if waveform.shape[0] != TARGET_SAMPLES:
            issues.append(make_issue("error", "reconstructed_sample_count_mismatch", f"{window_id}: {waveform.shape[0]}", participant_id=participant_id, window_id=window_id))
        reconstructed_count += 1
        if reconstruction_issues:
            duration_adjustment_warnings += 1

    del audio_cache
    gc.collect()

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not any(row["severity"] == "error" for row in issues) else "failed",
        "clean_windows": len(windows),
        "split_counts": dict(split_counts),
        "participants_409_422_present": excluded,
        "reconstructed_windows_checked": reconstructed_count,
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "target_samples": TARGET_SAMPLES,
        "all_reconstructed_windows_have_160000_samples": reconstructed_count == len(windows)
        and not any(row["issue_type"] == "reconstructed_sample_count_mismatch" for row in issues),
        "mapping_source": "participant_window_mapping.csv",
        "all_source_intervals_participant_only_basis": "participant_window_mapping.csv was produced from Participant speaker intervals; Phase 1B reads only this mapping and audio, not transcript text.",
        "transcript_text_used": False,
        "max_source_intervals_per_window": max_source_intervals,
        "duration_adjustment_warning_windows": duration_adjustment_warnings,
        "issues": issues,
    }
    write_json(run_dir / "input_validation_report.json", report)
    write_csv(run_dir / "input_validation_issues.csv", issues or [make_issue("info", "all_input_validation_checks_passed", "All input validation checks passed.")])
    if report["status"] != "passed":
        raise RuntimeError("Input validation failed. See input_validation_report.json.")
    return report


def compare_layer12_to_final(layer12: np.ndarray, final_embedding: np.ndarray) -> dict[str, float]:
    difference = layer12.astype(np.float64) - final_embedding.astype(np.float64)
    denominator = float(np.linalg.norm(layer12) * np.linalg.norm(final_embedding))
    cosine = float(np.dot(layer12, final_embedding) / denominator) if denominator else float("nan")
    return {
        "max_absolute_difference": float(np.max(np.abs(difference))),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "cosine_similarity": cosine,
    }


def validation_extraction(
    root: Path,
    output_dir: Path,
    run_dir: Path,
    windows: list[dict[str, str]],
    participants: dict[str, dict[str, str]],
    mapping_by_window: dict[str, list[dict[str, str]]],
    model: Any,
    device: Any,
    torch: Any,
    torchaudio: Any,
    num_layers: int,
    hidden_dim: int,
    batch_size: int,
) -> dict[str, Any]:
    validation_dir = run_dir / "validation_10" / "window_embeddings"
    validation_dir.mkdir(parents=True, exist_ok=True)
    final_manifest = {clean_value(row["window_id"]): row for row in read_csv_dicts(output_dir / "wavlm_embedding_manifest_gpu.csv")}
    audio_cache = AudioCache()
    rows = windows[:10]
    batch_payload: list[tuple[dict[str, str], np.ndarray, Path]] = []
    manifest_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    hidden_state_shapes: list[tuple[int, ...]] | None = None
    final_hidden_state_shape: tuple[int, ...] | None = None
    forward_passes = 0

    def process_batch() -> None:
        nonlocal hidden_state_shapes, final_hidden_state_shape, forward_passes
        if not batch_payload:
            return
        matrices, shapes, final_shape = extract_layerwise_batch(
            [item[1] for item in batch_payload],
            model,
            device,
            torch,
            num_layers,
            hidden_dim,
        )
        forward_passes += 1
        if hidden_state_shapes is None:
            hidden_state_shapes = shapes
            final_hidden_state_shape = final_shape
        for (window, _, output_path), matrix in zip(batch_payload, matrices):
            np.save(output_path, matrix.astype(np.float32, copy=False))
            ok, reason = valid_layerwise_matrix(output_path)
            if not ok:
                raise RuntimeError(f"Validation matrix failed for {window['window_id']}: {reason}")
            final_path = resolve_path(clean_value(final_manifest[clean_value(window["window_id"])]["embedding_path"]), root)
            final_embedding = np.load(final_path)
            comparison = compare_layer12_to_final(matrix[-1], final_embedding)
            equivalence_rows.append({"window_id": clean_value(window["window_id"]), **comparison})
            manifest_rows.append(
                {
                    "participant_id": clean_value(window["participant_id"]),
                    "window_id": clean_value(window["window_id"]),
                    "official_split": clean_value(window["official_split"]),
                    "label_binary": clean_value(window["label_binary"]),
                    "embedding_path": path_relative_to_root(output_path, root),
                    "matrix_shape": matrix_shape_text(matrix.shape),
                    "dtype": str(matrix.dtype),
                    "max_abs_layer12_vs_final": comparison["max_absolute_difference"],
                    "mean_abs_layer12_vs_final": comparison["mean_absolute_difference"],
                    "cosine_layer12_vs_final": comparison["cosine_similarity"],
                }
            )
        batch_payload.clear()

    for window in rows:
        participant = participants[clean_value(window["participant_id"])]
        waveform, _, _ = reconstruct_window(window, participant, mapping_by_window[clean_value(window["window_id"])], audio_cache, torch, torchaudio)
        if waveform.shape[0] != TARGET_SAMPLES:
            raise RuntimeError(f"Validation waveform sample mismatch for {window['window_id']}")
        waveform = normalize_waveform(waveform)
        output_path = validation_dir / f"{clean_value(window['window_id'])}.npy"
        batch_payload.append((window, waveform, output_path))
        if len(batch_payload) >= batch_size:
            process_batch()
    process_batch()

    max_abs = max(row["max_absolute_difference"] for row in equivalence_rows)
    mean_abs = float(np.mean([row["mean_absolute_difference"] for row in equivalence_rows]))
    min_cosine = min(row["cosine_similarity"] for row in equivalence_rows)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "validated_windows": len(rows),
        "cuda_device_used": str(device),
        "batch_size": batch_size,
        "forward_passes": forward_passes,
        "all_layers_from_single_forward_per_batch": True,
        "expected_matrix_shape": matrix_shape_text(EXPECTED_MATRIX_SHAPE),
        "hidden_state_shapes_observed_first_batch": [list(shape) for shape in (hidden_state_shapes or [])],
        "final_hidden_state_shape_observed_first_batch": list(final_hidden_state_shape or []),
        "hidden_state_indexing": {
            "hidden_states[0]": "pre-transformer feature-projection state, excluded from primary analysis",
            "hidden_states[1]..hidden_states[12]": "transformer encoder layers 1..12; saved as rows 0..11",
        },
        "layer12_equivalence": {
            "max_absolute_difference_max_over_windows": max_abs,
            "mean_absolute_difference_mean_over_windows": mean_abs,
            "cosine_similarity_min_over_windows": min_cosine,
            "tolerance_status": "passed" if max_abs <= 1e-5 and min_cosine > 0.999999 else "failed",
        },
        "window_ids": [clean_value(row["window_id"]) for row in rows],
    }
    write_json(run_dir / "validation_10" / "layerwise_validation_report.json", report)
    write_csv(run_dir / "validation_10" / "layerwise_validation_manifest.csv", manifest_rows)
    write_csv(run_dir / "validation_10" / "layer12_equivalence_check.csv", equivalence_rows)
    if report["layer12_equivalence"]["tolerance_status"] != "passed":
        raise RuntimeError("Layer 12 did not match existing final-layer embeddings within tolerance.")
    return report


def full_layerwise_extraction(
    root: Path,
    output_dir: Path,
    run_dir: Path,
    windows: list[dict[str, str]],
    participants: dict[str, dict[str, str]],
    mapping_by_window: dict[str, list[dict[str, str]]],
    model: Any,
    device: Any,
    torch: Any,
    torchaudio: Any,
    num_layers: int,
    hidden_dim: int,
    batch_size: int,
    force: bool,
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    embedding_dir = run_dir / "window_embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "layerwise_window_embedding_manifest.csv"
    quality_path = run_dir / "layerwise_embedding_quality_report.csv"
    metadata_path = run_dir / "layerwise_extraction_metadata.json"
    manifest_rows: dict[str, dict[str, Any]] = {}
    quality_rows: list[dict[str, Any]] = []
    audio_cache = AudioCache()
    batch_payload: list[tuple[dict[str, str], np.ndarray, Path, list[str]]] = []
    extracted = 0
    skipped = 0
    failed = 0
    processed_since_flush = 0
    hidden_state_shapes: list[tuple[int, ...]] | None = None
    final_hidden_state_shape: tuple[int, ...] | None = None
    forward_passes = 0
    start_timestamp = datetime.now(timezone.utc).isoformat()
    start_time = time.time()

    def make_manifest_row(window: dict[str, str], output_path: Path, status: str, error_message: str) -> dict[str, Any]:
        return {
            "participant_id": clean_value(window["participant_id"]),
            "window_id": clean_value(window["window_id"]),
            "official_split": clean_value(window["official_split"]),
            "label_binary": clean_value(window["label_binary"]),
            "embedding_path": path_relative_to_root(output_path, root),
            "matrix_shape": matrix_shape_text(EXPECTED_MATRIX_SHAPE) if status in {"success", "skipped_valid"} else "",
            "num_transformer_layers": num_layers,
            "hidden_dimension": hidden_dim,
            "dtype": EXPECTED_DTYPE if status in {"success", "skipped_valid"} else "",
            "model_name": MODEL_NAME,
            "model_commit": MODEL_COMMIT,
            "extraction_status": status,
            "error_message": error_message,
        }

    def process_batch() -> None:
        nonlocal extracted, processed_since_flush, hidden_state_shapes, final_hidden_state_shape, forward_passes
        if not batch_payload:
            return
        matrices, shapes, final_shape = extract_layerwise_batch(
            [item[1] for item in batch_payload],
            model,
            device,
            torch,
            num_layers,
            hidden_dim,
        )
        forward_passes += 1
        if hidden_state_shapes is None:
            hidden_state_shapes = shapes
            final_hidden_state_shape = final_shape
        for (window, _, output_path, reconstruction_issues), matrix in zip(batch_payload, matrices):
            np.save(output_path, matrix.astype(np.float32, copy=False))
            ok, reason = valid_layerwise_matrix(output_path)
            if not ok:
                raise RuntimeError(f"Saved invalid layerwise file for {clean_value(window['window_id'])}: {reason}")
            error_message = "; ".join(reconstruction_issues)
            manifest_rows[clean_value(window["window_id"])] = make_manifest_row(window, output_path, "success", error_message)
            if reconstruction_issues:
                quality_rows.append(make_issue("warning", "waveform_reconstruction_warning", error_message, window_id=clean_value(window["window_id"])))
            extracted += 1
            processed_since_flush += 1
        batch_payload.clear()

    for index, window in enumerate(windows, start=1):
        window_id = clean_value(window["window_id"])
        output_path = embedding_dir / f"{window_id}.npy"
        if not force:
            ok, reason = valid_layerwise_matrix(output_path)
            if ok:
                manifest_rows[window_id] = make_manifest_row(window, output_path, "skipped_valid", "")
                skipped += 1
                processed_since_flush += 1
                if processed_since_flush >= 100:
                    write_manifest_and_quality(manifest_path, manifest_rows, quality_path, quality_rows)
                    processed_since_flush = 0
                continue
            if output_path.exists() and reason != "missing":
                quality_rows.append(make_issue("warning", "recomputing_invalid_layerwise_file", reason, window_id=window_id, embedding_path=str(output_path)))
        try:
            participant = participants[clean_value(window["participant_id"])]
            waveform, _, reconstruction_issues = reconstruct_window(window, participant, mapping_by_window[window_id], audio_cache, torch, torchaudio)
            if waveform.shape[0] != TARGET_SAMPLES:
                raise RuntimeError(f"final_samples={waveform.shape[0]} expected={TARGET_SAMPLES}")
            waveform = normalize_waveform(waveform)
            batch_payload.append((window, waveform, output_path, reconstruction_issues))
            if len(batch_payload) >= batch_size:
                process_batch()
        except Exception as exc:  # noqa: BLE001
            manifest_rows[window_id] = make_manifest_row(window, output_path, "failed", str(exc))
            quality_rows.append(make_issue("error", "window_layerwise_extraction_failed", str(exc), window_id=window_id, participant_id=clean_value(window["participant_id"])))
            failed += 1
            processed_since_flush += 1

        if processed_since_flush >= 100:
            write_manifest_and_quality(manifest_path, manifest_rows, quality_path, quality_rows)
            processed_since_flush = 0
        if index % 250 == 0:
            elapsed = time.time() - start_time
            print(f"processed={index}/{len(windows)} extracted={extracted} skipped={skipped} failed={failed} elapsed_sec={elapsed:.1f}", flush=True)

    try:
        process_batch()
    except Exception as exc:  # noqa: BLE001
        for window, _, output_path, _ in batch_payload:
            manifest_rows[clean_value(window["window_id"])] = make_manifest_row(window, output_path, "failed", str(exc))
            quality_rows.append(make_issue("error", "batch_layerwise_extraction_failed", str(exc), window_id=clean_value(window["window_id"])))
            failed += 1
        batch_payload.clear()

    clean_window_ids = {clean_value(row["window_id"]) for row in windows}
    manifest_window_ids = set(manifest_rows)
    split_counts = Counter(row["official_split"] for row in manifest_rows.values())
    valid_rows = [row for row in manifest_rows.values() if row["extraction_status"] in {"success", "skipped_valid"}]
    status_counts = Counter(row["extraction_status"] for row in manifest_rows.values())
    for issue_type, ids in (
        ("missing_clean_windows", sorted(clean_window_ids - manifest_window_ids)),
        ("unexpected_windows", sorted(manifest_window_ids - clean_window_ids)),
    ):
        if ids:
            quality_rows.append(make_issue("error", issue_type, f"{len(ids)} windows.", examples=ids[:10]))
    if len(valid_rows) != EXPECTED_TOTAL_WINDOWS:
        quality_rows.append(make_issue("error", "valid_layerwise_file_count_mismatch", f"Expected {EXPECTED_TOTAL_WINDOWS}, found {len(valid_rows)}."))
    if dict(split_counts) != EXPECTED_WINDOW_SPLITS:
        quality_rows.append(make_issue("error", "split_count_mismatch", f"Expected {EXPECTED_WINDOW_SPLITS}, found {dict(split_counts)}."))
    participant_ids = {row["participant_id"] for row in manifest_rows.values()}
    excluded = sorted(participant_ids & EXCLUDED_PARTICIPANTS, key=participant_sort_key)
    if excluded:
        quality_rows.append(make_issue("error", "excluded_participants_present", f"{excluded}."))

    for row in valid_rows:
        ok, reason = valid_layerwise_matrix(resolve_path(row["embedding_path"], root))
        if not ok:
            quality_rows.append(make_issue("error", "invalid_existing_layerwise_file", reason, window_id=row["window_id"], embedding_path=row["embedding_path"]))

    error_count = sum(1 for row in quality_rows if row["severity"] == "error")
    if not quality_rows:
        quality_rows.append(make_issue("info", "all_layerwise_extraction_checks_passed", "All layer-wise extraction checks passed."))
    write_manifest_and_quality(manifest_path, manifest_rows, quality_path, quality_rows)

    torch_module = sys.modules.get("torch")
    gpu_model = torch_module.cuda.get_device_name(device) if torch_module is not None else ""
    free_memory, total_memory = torch_module.cuda.mem_get_info(device) if torch_module is not None else (None, None)
    metadata = {
        "execution_start_timestamp": start_timestamp,
        "execution_completion_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if error_count == 0 and failed == 0 else "failed",
        "model_name": MODEL_NAME,
        "model_commit": MODEL_COMMIT,
        "model_metadata": model_metadata,
        "device": str(device),
        "gpu_model": gpu_model,
        "gpu_memory_free_bytes_at_completion": free_memory,
        "gpu_memory_total_bytes": total_memory,
        "python_version": sys.version.replace("\n", " "),
        "numpy_version": np.__version__,
        "pytorch_version": get_package_version("torch"),
        "transformers_version": get_package_version("transformers"),
        "torchaudio_version": get_package_version("torchaudio"),
        "cuda_version": getattr(torch_module, "version").cuda if torch_module is not None else "",
        "batch_size": batch_size,
        "float32_inference": True,
        "output_hidden_states": True,
        "gradient_mode": "torch.inference_mode",
        "model_eval_mode": True,
        "window_embedding_dir": path_relative_to_root(embedding_dir, root),
        "manifest_path": path_relative_to_root(manifest_path, root),
        "quality_report_path": path_relative_to_root(quality_path, root),
        "hidden_state_shapes_observed_first_batch": [list(shape) for shape in (hidden_state_shapes or [])],
        "final_hidden_state_shape_observed_first_batch": list(final_hidden_state_shape or []),
        "forward_passes": forward_passes,
        "total_expected_windows": EXPECTED_TOTAL_WINDOWS,
        "total_valid_window_files": len(valid_rows),
        "extracted": extracted,
        "skipped_valid": skipped,
        "failed": failed,
        "status_counts": dict(status_counts),
        "split_counts": dict(split_counts),
        "test_evaluation_performed": False,
        "transcript_text_used": False,
        "normalization_policy": "same as final-layer extraction: float32 DC offset removal and peak normalization; no dataset normalization, augmentation, or denoising",
    }
    write_json(metadata_path, metadata)
    if metadata["status"] != "passed":
        raise RuntimeError("Full layer-wise extraction failed validation.")
    return metadata


def aggregate_participant_layers(root: Path, output_dir: Path, run_dir: Path) -> dict[str, Any]:
    participant_rows = {clean_value(row["participant_id"]): row for row in read_csv_dicts(output_dir / "daic_phase1_participant_manifest_clean.csv")}
    manifest_rows = read_csv_dicts(run_dir / "layerwise_window_embedding_manifest.csv")
    by_participant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        by_participant[clean_value(row["participant_id"])].append(row)

    participant_dir = run_dir / "participant_layer_embeddings"
    participant_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    total_windows_used = 0

    for participant_id in sorted(participant_rows, key=participant_sort_key):
        participant = participant_rows[participant_id]
        expected_windows = int(clean_value(participant.get("num_windows")) or "0")
        rows = sorted(by_participant.get(participant_id, []), key=lambda item: clean_value(item["window_id"]))
        accumulator = np.zeros(EXPECTED_MATRIX_SHAPE, dtype=np.float64)
        used = 0
        status = "success"
        errors: list[str] = []
        for row in rows:
            if clean_value(row.get("extraction_status")) not in {"success", "skipped_valid"}:
                status = "failed"
                errors.append(f"non_valid_window={clean_value(row.get('window_id'))}")
                continue
            path = resolve_path(clean_value(row["embedding_path"]), root)
            ok, reason = valid_layerwise_matrix(path)
            if not ok:
                status = "failed"
                errors.append(f"{clean_value(row.get('window_id'))}:{reason}")
                continue
            accumulator += np.load(path).astype(np.float64, copy=False)
            used += 1
        if used != expected_windows or len(rows) != expected_windows:
            status = "failed"
            errors.append(f"expected_windows={expected_windows} manifest_rows={len(rows)} used={used}")
        if participant_id in EXCLUDED_PARTICIPANTS:
            status = "failed"
            errors.append("excluded_participant_present")

        output_path = participant_dir / f"{participant_id}.npy"
        if status == "success" and used > 0:
            participant_matrix = (accumulator / float(used)).astype(np.float32)
            if participant_matrix.shape != EXPECTED_MATRIX_SHAPE or not np.isfinite(participant_matrix).all() or np.any(np.all(participant_matrix == 0, axis=1)):
                status = "failed"
                errors.append("invalid_participant_layer_matrix")
            else:
                np.save(output_path, participant_matrix)
                total_windows_used += used
        if status != "success":
            quality_rows.append(make_issue("error", "participant_layer_aggregation_failed", "; ".join(errors), participant_id=participant_id))
        output_rows.append(
            {
                "participant_id": participant_id,
                "official_split": clean_value(participant["official_split"]),
                "label_binary": clean_value(participant["label_binary"]),
                "embedding_path": path_relative_to_root(output_path, root),
                "matrix_shape": matrix_shape_text(EXPECTED_MATRIX_SHAPE) if status == "success" else "",
                "num_transformer_layers": EXPECTED_NUM_TRANSFORMER_LAYERS,
                "hidden_dimension": EXPECTED_HIDDEN_DIM,
                "dtype": EXPECTED_DTYPE if status == "success" else "",
                "num_windows_expected": expected_windows,
                "num_windows_used": used,
                "aggregation_method": "arithmetic_mean_per_layer_float64_accumulation",
                "aggregation_status": status,
                "error_message": "; ".join(errors),
            }
        )

    success_rows = [row for row in output_rows if row["aggregation_status"] == "success"]
    split_counts = Counter(row["official_split"] for row in success_rows)
    participant_ids = [row["participant_id"] for row in output_rows]
    validation_errors: list[dict[str, Any]] = []
    if len(success_rows) != sum(EXPECTED_PARTICIPANT_SPLITS.values()):
        validation_errors.append(make_issue("error", "participant_count_mismatch", f"Expected 187, found {len(success_rows)}."))
    if dict(split_counts) != EXPECTED_PARTICIPANT_SPLITS:
        validation_errors.append(make_issue("error", "participant_split_count_mismatch", f"Expected {EXPECTED_PARTICIPANT_SPLITS}, found {dict(split_counts)}."))
    if len(participant_ids) != len(set(participant_ids)):
        validation_errors.append(make_issue("error", "duplicate_participants", "Duplicate participant IDs in manifest."))
    excluded = sorted(set(participant_ids) & EXCLUDED_PARTICIPANTS, key=participant_sort_key)
    if excluded:
        validation_errors.append(make_issue("error", "excluded_participants_present", f"{excluded}."))
    if total_windows_used != EXPECTED_TOTAL_WINDOWS:
        validation_errors.append(make_issue("error", "total_windows_used_mismatch", f"Expected {EXPECTED_TOTAL_WINDOWS}, found {total_windows_used}."))

    split_sets: dict[str, set[str]] = defaultdict(set)
    for row in output_rows:
        split_sets[row["participant_id"]].add(row["official_split"])
        if row["aggregation_status"] == "success":
            if int(row["num_windows_used"]) != int(row["num_windows_expected"]):
                validation_errors.append(make_issue("error", "participant_window_count_mismatch", row["participant_id"], participant_id=row["participant_id"]))
            ok, reason = valid_layerwise_matrix(resolve_path(row["embedding_path"], root))
            if not ok:
                validation_errors.append(make_issue("error", "invalid_participant_layer_embedding", reason, participant_id=row["participant_id"]))
    multi_split = sorted([pid for pid, splits in split_sets.items() if len(splits) > 1], key=participant_sort_key)
    if multi_split:
        validation_errors.append(make_issue("error", "participant_leakage_multiple_splits", f"{multi_split[:10]}."))

    quality_rows.extend(validation_errors)
    if not quality_rows:
        quality_rows.append(make_issue("info", "all_participant_layer_aggregation_checks_passed", "All participant layer aggregation checks passed."))

    write_csv(run_dir / "participant_layer_embedding_manifest.csv", output_rows, PARTICIPANT_MANIFEST_FIELDS)
    write_csv(run_dir / "participant_layer_embedding_quality_report.csv", quality_rows)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not validation_errors and len(success_rows) == 187 else "failed",
        "participant_layer_embedding_dir": path_relative_to_root(participant_dir, root),
        "aggregation_method": "arithmetic mean across window matrices per participant; float64 accumulation; float32 output",
        "participant_counts_by_split": dict(split_counts),
        "total_participant_files": len(success_rows),
        "total_windows_used": total_windows_used,
        "test_evaluation_performed": False,
    }
    write_json(run_dir / "participant_layer_aggregation_metadata.json", metadata)
    if metadata["status"] != "passed":
        raise RuntimeError("Participant layer aggregation validation failed.")
    return metadata


def logistic_pipeline(C: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def positive_probabilities(model: Any, x_data: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x_data)
    class_list = list(model.classes_)
    if 1 not in class_list:
        return np.zeros(probabilities.shape[0], dtype=np.float64)
    return probabilities[:, class_list.index(1)].astype(np.float64)


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
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


def select_c_inner(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for C in C_GRID:
        fold_aucs: list[float] = []
        fold_bal_acc: list[float] = []
        fold_macro_f1: list[float] = []
        for train_index, validation_index in splitter.split(x_train, y_train):
            model = logistic_pipeline(C)
            model.fit(x_train[train_index], y_train[train_index])
            probabilities = positive_probabilities(model, x_train[validation_index])
            metrics = threshold_metrics(y_train[validation_index], probabilities, 0.5)
            fold_aucs.append(metrics["roc_auc"])
            fold_bal_acc.append(metrics["balanced_accuracy"])
            fold_macro_f1.append(metrics["macro_f1"])
        rows.append(
            {
                "C": C,
                "mean_roc_auc": float(np.mean(fold_aucs)),
                "roc_auc_std": float(np.std(fold_aucs, ddof=0)),
                "mean_balanced_accuracy_at_0_5": float(np.mean(fold_bal_acc)),
                "mean_macro_f1_at_0_5": float(np.mean(fold_macro_f1)),
            }
        )
    selected = sorted(rows, key=lambda row: (-row["mean_roc_auc"], row["roc_auc_std"], row["C"]))[0]
    return float(selected["C"]), rows


def load_train_dev_layer_inputs(root: Path, output_dir: Path, run_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    participant_manifest = pd.read_csv(
        run_dir / "participant_layer_embedding_manifest.csv",
        usecols=["participant_id", "official_split", "embedding_path", "aggregation_status"],
        dtype={"participant_id": str, "official_split": str, "embedding_path": str, "aggregation_status": str},
    )
    modeling_frame = participant_manifest[participant_manifest["official_split"].isin(["train", "dev"])].copy()
    assert not modeling_frame["official_split"].eq("test").any(), "Model-selection dataframe contains test rows."

    label_rows: list[dict[str, Any]] = []
    with (output_dir / "daic_phase1_participant_manifest_clean.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split = clean_value(row.get("official_split"))
            if split in {"train", "dev"}:
                label_rows.append(
                    {
                        "participant_id": clean_value(row["participant_id"]),
                        "label_binary": int(clean_value(row["label_binary"])),
                    }
                )
    labels = pd.DataFrame(label_rows)
    modeling_frame = modeling_frame.merge(labels, on="participant_id", how="inner", validate="one_to_one")
    modeling_frame = modeling_frame.sort_values("participant_id", key=lambda series: series.map(lambda value: participant_sort_key(str(value))[0])).reset_index(drop=True)
    assert not modeling_frame["official_split"].eq("test").any(), "Labels were loaded for test rows."

    matrices: list[np.ndarray] = []
    for row in modeling_frame.itertuples(index=False):
        matrix = np.load(resolve_path(row.embedding_path, root))
        if matrix.shape != EXPECTED_MATRIX_SHAPE:
            raise RuntimeError(f"Unexpected participant layer shape for {row.participant_id}: {matrix.shape}")
        matrices.append(matrix.astype(np.float32, copy=False))
    x_all = np.stack(matrices, axis=0)
    y_all = modeling_frame["label_binary"].to_numpy(dtype=int)
    train_mask = modeling_frame["official_split"].eq("train").to_numpy()
    dev_mask = modeling_frame["official_split"].eq("dev").to_numpy()
    train_frame = modeling_frame.loc[train_mask].reset_index(drop=True)
    dev_frame = modeling_frame.loc[dev_mask].reset_index(drop=True)
    return train_frame, x_all[train_mask], y_all[train_mask], dev_frame, x_all[dev_mask], y_all[dev_mask]


def nested_layer_evaluation(
    train_frame: pd.DataFrame,
    x_train_layers: np.ndarray,
    y_train: np.ndarray,
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    outer_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    splitter = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=RANDOM_SEED)

    for layer_index in range(EXPECTED_NUM_TRANSFORMER_LAYERS):
        layer_number = layer_index + 1
        layer_name = f"layer_{layer_number:02d}"
        x_layer = x_train_layers[:, layer_index, :]
        layer_metrics: list[dict[str, Any]] = []
        selected_cs: list[float] = []
        for outer_fold_index, (outer_train_index, outer_validation_index) in enumerate(splitter.split(x_layer, y_train), start=1):
            inner_seed = RANDOM_SEED * 1000 + outer_fold_index
            selected_C, inner_rows = select_c_inner(x_layer[outer_train_index], y_train[outer_train_index], inner_seed)
            selected_cs.append(selected_C)
            model = logistic_pipeline(selected_C)
            model.fit(x_layer[outer_train_index], y_train[outer_train_index])
            probabilities = positive_probabilities(model, x_layer[outer_validation_index])
            metrics = threshold_metrics(y_train[outer_validation_index], probabilities, 0.5)
            best_inner = sorted(inner_rows, key=lambda row: (-row["mean_roc_auc"], row["roc_auc_std"], row["C"]))[0]
            row = {
                "layer": layer_name,
                "layer_index": layer_number,
                "outer_fold_index": outer_fold_index,
                "inner_seed": inner_seed,
                "selected_C": selected_C,
                "inner_selected_C_mean_roc_auc": best_inner["mean_roc_auc"],
                "inner_selected_C_roc_auc_std": best_inner["roc_auc_std"],
                **metrics,
                "n_outer_train": int(len(outer_train_index)),
                "n_outer_validation": int(len(outer_validation_index)),
            }
            outer_rows.append(row)
            layer_metrics.append(row)
            for participant_position, probability in zip(outer_validation_index, probabilities):
                participant = train_frame.iloc[int(participant_position)]
                prediction_rows.append(
                    {
                        "layer": layer_name,
                        "layer_index": layer_number,
                        "outer_fold_index": outer_fold_index,
                        "participant_id": participant["participant_id"],
                        "official_split": participant["official_split"],
                        "label_binary": int(y_train[int(participant_position)]),
                        "predicted_probability": float(probability),
                        "prediction_at_0_5": int(probability >= 0.5),
                        "selected_C": selected_C,
                    }
                )
        roc_aucs = np.array([row["roc_auc"] for row in layer_metrics], dtype=np.float64)
        pr_aucs = np.array([row["pr_auc"] for row in layer_metrics], dtype=np.float64)
        bal_accs = np.array([row["balanced_accuracy"] for row in layer_metrics], dtype=np.float64)
        macro_f1s = np.array([row["macro_f1"] for row in layer_metrics], dtype=np.float64)
        selected_c_counts = Counter(selected_cs)
        summary_rows.append(
            {
                "layer": layer_name,
                "layer_index": layer_number,
                "mean_outer_roc_auc": float(roc_aucs.mean()),
                "std_outer_roc_auc": float(roc_aucs.std(ddof=0)),
                "median_outer_roc_auc": float(np.median(roc_aucs)),
                "min_outer_roc_auc": float(roc_aucs.min()),
                "max_outer_roc_auc": float(roc_aucs.max()),
                "mean_outer_pr_auc": float(pr_aucs.mean()),
                "mean_outer_balanced_accuracy_at_0_5": float(bal_accs.mean()),
                "mean_outer_macro_f1_at_0_5": float(macro_f1s.mean()),
                "selected_C_frequency": json.dumps({str(key): selected_c_counts[key] for key in sorted(selected_c_counts)}),
                "n_outer_evaluations": len(layer_metrics),
            }
        )
        print(f"nested_cv_complete layer={layer_name} mean_auc={summary_rows[-1]['mean_outer_roc_auc']:.4f}", flush=True)

    summary = pd.DataFrame(summary_rows)
    outer = pd.DataFrame(outer_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary.to_csv(run_dir / "layerwise_nested_cv_summary.csv", index=False)
    outer.to_csv(run_dir / "layerwise_outer_fold_results.csv", index=False)
    predictions.to_csv(run_dir / "layerwise_outer_predictions.csv", index=False)

    selected = summary.sort_values(
        ["mean_outer_roc_auc", "std_outer_roc_auc", "mean_outer_pr_auc", "layer_index"],
        ascending=[False, True, False, True],
    ).iloc[0].to_dict()
    layer12 = summary.loc[summary["layer_index"].eq(12)].iloc[0].to_dict()
    selected_report = {
        "selected_layer": selected["layer"],
        "selected_layer_index": int(selected["layer_index"]),
        "selection_criterion": [
            "highest mean outer ROC-AUC",
            "lowest outer ROC-AUC standard deviation",
            "highest mean outer PR-AUC",
            "lower numerical layer index if exact tie remains",
        ],
        "mean_outer_roc_auc": float(selected["mean_outer_roc_auc"]),
        "std_outer_roc_auc": float(selected["std_outer_roc_auc"]),
        "mean_outer_pr_auc": float(selected["mean_outer_pr_auc"]),
        "layer12_mean_outer_roc_auc": float(layer12["mean_outer_roc_auc"]),
        "layer12_std_outer_roc_auc": float(layer12["std_outer_roc_auc"]),
        "difference_vs_layer12_mean_outer_roc_auc": float(selected["mean_outer_roc_auc"] - layer12["mean_outer_roc_auc"]),
        "n_outer_evaluations": int(selected["n_outer_evaluations"]),
        "dev_data_used_for_layer_selection": False,
        "test_data_used_for_layer_selection": False,
        "interpretation_note": "A small numerical difference in nested-CV ROC-AUC is not by itself evidence of meaningful superiority.",
    }
    write_json(run_dir / "selected_wavlm_layer.json", selected_report)
    return summary, outer, predictions, selected_report


def final_c_selection(x_train: np.ndarray, y_train: np.ndarray, run_dir: Path) -> tuple[pd.DataFrame, float]:
    splitter = RepeatedStratifiedKFold(n_splits=FINAL_C_SPLITS, n_repeats=FINAL_C_REPEATS, random_state=RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    for C in C_GRID:
        aucs: list[float] = []
        bal_accs: list[float] = []
        macro_f1s: list[float] = []
        for train_index, validation_index in splitter.split(x_train, y_train):
            model = logistic_pipeline(C)
            model.fit(x_train[train_index], y_train[train_index])
            probabilities = positive_probabilities(model, x_train[validation_index])
            metrics = threshold_metrics(y_train[validation_index], probabilities, 0.5)
            aucs.append(metrics["roc_auc"])
            bal_accs.append(metrics["balanced_accuracy"])
            macro_f1s.append(metrics["macro_f1"])
        rows.append(
            {
                "C": C,
                "mean_roc_auc": float(np.mean(aucs)),
                "roc_auc_std": float(np.std(aucs, ddof=0)),
                "mean_balanced_accuracy_at_0_5": float(np.mean(bal_accs)),
                "mean_macro_f1_at_0_5": float(np.mean(macro_f1s)),
                "n_splits": FINAL_C_SPLITS,
                "n_repeats": FINAL_C_REPEATS,
                "random_state": RANDOM_SEED,
            }
        )
    frame = pd.DataFrame(rows)
    selected = frame.sort_values(["mean_roc_auc", "roc_auc_std", "C"], ascending=[False, True, True]).iloc[0]
    frame.to_csv(run_dir / "selected_layer_final_c_search.csv", index=False)
    return frame, float(selected["C"])


def deterministic_thresholds(probabilities: np.ndarray) -> list[float]:
    unique_probabilities = sorted({float(value) for value in probabilities})
    thresholds = {0.5}
    thresholds.update(unique_probabilities)
    for left, right in zip(unique_probabilities, unique_probabilities[1:]):
        thresholds.add((left + right) / 2.0)
    return sorted(thresholds)


def tune_dev_threshold(y_dev: np.ndarray, probabilities: np.ndarray, run_dir: Path) -> tuple[pd.DataFrame, float]:
    rows = [threshold_metrics(y_dev, probabilities, threshold) for threshold in deterministic_thresholds(probabilities)]
    frame = pd.DataFrame(rows)
    selected = frame.assign(distance_to_0_5=(frame["threshold"] - 0.5).abs()).sort_values(
        ["macro_f1", "balanced_accuracy", "distance_to_0_5", "threshold"],
        ascending=[False, False, True, True],
    ).iloc[0]
    frame.to_csv(run_dir / "selected_layer_dev_threshold_search.csv", index=False)
    return frame, float(selected["threshold"])


def evaluate_selected_layer(
    root: Path,
    run_dir: Path,
    train_frame: pd.DataFrame,
    x_train_layers: np.ndarray,
    y_train: np.ndarray,
    dev_frame: pd.DataFrame,
    x_dev_layers: np.ndarray,
    y_dev: np.ndarray,
    selected_report: dict[str, Any],
    nested_summary: pd.DataFrame,
) -> dict[str, Any]:
    selected_layer_index = int(selected_report["selected_layer_index"])
    selected_layer_name = clean_value(selected_report["selected_layer"])
    x_train = x_train_layers[:, selected_layer_index - 1, :]
    x_dev = x_dev_layers[:, selected_layer_index - 1, :]
    final_c_frame, selected_C = final_c_selection(x_train, y_train, run_dir)
    model = logistic_pipeline(selected_C)
    model.fit(x_train, y_train)
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "selected_layer_logistic_regression_pipeline.joblib"
    joblib.dump(model, model_path)

    probabilities = positive_probabilities(model, x_dev)
    threshold_search, selected_threshold = tune_dev_threshold(y_dev, probabilities, run_dir)
    metrics_0_5 = threshold_metrics(y_dev, probabilities, 0.5)
    metrics_tuned = threshold_metrics(y_dev, probabilities, selected_threshold)
    prediction_rows: list[dict[str, Any]] = []
    for participant, probability in zip(dev_frame.itertuples(index=False), probabilities):
        prediction_rows.append(
            {
                "participant_id": clean_value(participant.participant_id),
                "official_split": clean_value(participant.official_split),
                "label_binary": int(participant.label_binary),
                "layer": selected_layer_name,
                "layer_index": selected_layer_index,
                "predicted_probability": float(probability),
                "prediction_at_0_5": int(probability >= 0.5),
                "prediction_at_tuned_threshold": int(probability >= selected_threshold),
                "tuned_threshold": selected_threshold,
                "model_name": "phase1b_selected_wavlm_layer_logistic_regression",
            }
        )
    write_csv(run_dir / "selected_layer_dev_predictions.csv", prediction_rows)

    previous_run = root / "outputs" / "phase1_daic" / "phase1_participant_baseline_runs" / "20260622_114625"
    previous_report = json.loads((previous_run / "evaluation_report.json").read_text(encoding="utf-8"))
    previous_wavlm = previous_report["wavlm_participant_baseline"]
    previous_train = previous_wavlm["train_cv_selected_C"]
    previous_dev_0_5 = previous_wavlm["dev_metrics_at_0_5"]
    previous_dev_tuned = previous_wavlm["dev_metrics_at_selected_threshold"]

    layer12_summary = nested_summary.loc[nested_summary["layer_index"].eq(12)].iloc[0].to_dict()
    selected_summary = nested_summary.loc[nested_summary["layer_index"].eq(selected_layer_index)].iloc[0].to_dict()
    comparison_rows = [
        {
            "row_name": "existing_layer_12_baseline_20260622_114625",
            "protocol": "previous_phase1_repeated_train_cv_not_nested_final_hidden_state",
            "layer": "layer_12",
            "train_roc_auc_mean": previous_train["mean_roc_auc"],
            "train_roc_auc_std": previous_train["roc_auc_std"],
            "dev_roc_auc": previous_dev_0_5["roc_auc"],
            "dev_pr_auc": previous_dev_0_5["pr_auc"],
            "dev_balanced_accuracy_at_0_5": previous_dev_0_5["balanced_accuracy"],
            "dev_macro_f1_at_0_5": previous_dev_0_5["macro_f1"],
            "dev_sensitivity_at_0_5": previous_dev_0_5["positive_recall_sensitivity"],
            "dev_balanced_accuracy_tuned": previous_dev_tuned["balanced_accuracy"],
            "dev_macro_f1_tuned": previous_dev_tuned["macro_f1"],
            "dev_sensitivity_tuned": previous_dev_tuned["positive_recall_sensitivity"],
            "selected_C": previous_wavlm["selected_C"],
            "selected_dev_threshold": previous_wavlm["selected_threshold"],
            "notes": "Existing final-layer baseline; no local-test metrics included.",
        },
        {
            "row_name": "new_layer_12_phase1b_nested_cv",
            "protocol": "phase1b_train_only_nested_cv",
            "layer": "layer_12",
            "train_roc_auc_mean": layer12_summary["mean_outer_roc_auc"],
            "train_roc_auc_std": layer12_summary["std_outer_roc_auc"],
            "dev_roc_auc": metrics_0_5["roc_auc"] if selected_layer_index == 12 else "",
            "dev_pr_auc": metrics_0_5["pr_auc"] if selected_layer_index == 12 else "",
            "dev_balanced_accuracy_at_0_5": metrics_0_5["balanced_accuracy"] if selected_layer_index == 12 else "",
            "dev_macro_f1_at_0_5": metrics_0_5["macro_f1"] if selected_layer_index == 12 else "",
            "dev_sensitivity_at_0_5": metrics_0_5["positive_recall_sensitivity"] if selected_layer_index == 12 else "",
            "dev_balanced_accuracy_tuned": metrics_tuned["balanced_accuracy"] if selected_layer_index == 12 else "",
            "dev_macro_f1_tuned": metrics_tuned["macro_f1"] if selected_layer_index == 12 else "",
            "dev_sensitivity_tuned": metrics_tuned["positive_recall_sensitivity"] if selected_layer_index == 12 else "",
            "selected_C": selected_C if selected_layer_index == 12 else "",
            "selected_dev_threshold": selected_threshold if selected_layer_index == 12 else "",
            "notes": "Dev metrics populated only if layer 12 is the selected layer; Phase 1B performs one dev evaluation of the selected layer only.",
        },
        {
            "row_name": "selected_best_layer_phase1b",
            "protocol": "phase1b_train_nested_cv_then_single_dev_evaluation",
            "layer": selected_layer_name,
            "train_roc_auc_mean": selected_summary["mean_outer_roc_auc"],
            "train_roc_auc_std": selected_summary["std_outer_roc_auc"],
            "dev_roc_auc": metrics_0_5["roc_auc"],
            "dev_pr_auc": metrics_0_5["pr_auc"],
            "dev_balanced_accuracy_at_0_5": metrics_0_5["balanced_accuracy"],
            "dev_macro_f1_at_0_5": metrics_0_5["macro_f1"],
            "dev_sensitivity_at_0_5": metrics_0_5["positive_recall_sensitivity"],
            "dev_balanced_accuracy_tuned": metrics_tuned["balanced_accuracy"],
            "dev_macro_f1_tuned": metrics_tuned["macro_f1"],
            "dev_sensitivity_tuned": metrics_tuned["positive_recall_sensitivity"],
            "selected_C": selected_C,
            "selected_dev_threshold": selected_threshold,
            "notes": "Selected using train-only nested CV; dev evaluated once after layer and C selection.",
        },
    ]
    write_csv(run_dir / "layer12_comparison.csv", comparison_rows)

    selected_c_payload = {
        "selected_layer": selected_layer_name,
        "selected_layer_index": selected_layer_index,
        "selected_C": selected_C,
        "selection_rule": ["highest mean ROC-AUC", "lowest ROC-AUC std", "lower C"],
        "cv_configuration": {
            "splitter": "RepeatedStratifiedKFold",
            "n_splits": FINAL_C_SPLITS,
            "n_repeats": FINAL_C_REPEATS,
            "random_state": RANDOM_SEED,
        },
        "C_grid": C_GRID,
    }
    write_json(run_dir / "selected_layer_final_c.json", selected_c_payload)

    evaluation = {
        "selected_layer": selected_layer_name,
        "selected_layer_index": selected_layer_index,
        "selected_C": selected_C,
        "selected_dev_threshold": selected_threshold,
        "dev_metrics_at_0_5": metrics_0_5,
        "dev_metrics_at_selected_threshold": metrics_tuned,
        "model_path": path_relative_to_root(model_path, root),
        "test_evaluation_performed": False,
        "test_probabilities_generated": False,
        "test_metrics_computed": False,
    }
    return evaluation


def write_reports(
    root: Path,
    output_dir: Path,
    run_dir: Path,
    args: argparse.Namespace,
    input_report: dict[str, Any],
    validation_report: dict[str, Any],
    extraction_metadata: dict[str, Any],
    aggregation_metadata: dict[str, Any],
    selected_report: dict[str, Any],
    evaluation: dict[str, Any],
    nested_summary: pd.DataFrame,
) -> None:
    summary_rows = nested_summary.sort_values("layer_index").to_dict(orient="records")
    layer12 = nested_summary.loc[nested_summary["layer_index"].eq(12)].iloc[0].to_dict()
    best = nested_summary.loc[nested_summary["layer_index"].eq(int(selected_report["selected_layer_index"]))].iloc[0].to_dict()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_phase1b_layerwise_train_only_selection_and_single_dev_evaluation",
        "test_evaluation_performed": False,
        "test_probabilities_generated": False,
        "strict_scope_confirmation": {
            "alternate_aggregation": False,
            "pca": False,
            "mlp": False,
            "svm": False,
            "lightgbm": False,
            "attention_pooling": False,
            "window_level_classifier": False,
            "wavlm_fine_tuning": False,
            "android_corpus": False,
            "app_code": False,
        },
        "layerwise_extraction": {
            "valid_window_files": extraction_metadata["total_valid_window_files"],
            "failed": extraction_metadata["failed"],
            "num_transformer_layers": EXPECTED_NUM_TRANSFORMER_LAYERS,
            "hidden_dimension": EXPECTED_HIDDEN_DIM,
        },
        "layer12_equivalence_check": validation_report["layer12_equivalence"],
        "participant_aggregation": aggregation_metadata,
        "nested_cv_summary_by_layer": summary_rows,
        "selected_layer": selected_report,
        "selected_layer_final_evaluation": evaluation,
        "comparison_with_layer12": {
            "best_layer_mean_outer_roc_auc": best["mean_outer_roc_auc"],
            "layer12_mean_outer_roc_auc": layer12["mean_outer_roc_auc"],
            "difference": float(best["mean_outer_roc_auc"] - layer12["mean_outer_roc_auc"]),
            "note": selected_report["interpretation_note"],
        },
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
        "torch_version": get_package_version("torch"),
        "transformers_version": get_package_version("transformers"),
        "torchaudio_version": get_package_version("torchaudio"),
        "config_path": args.config,
        "input_manifest_paths": {
            "participant_manifest_clean": path_relative_to_root(output_dir / "daic_phase1_participant_manifest_clean.csv", root),
            "window_manifest_clean": path_relative_to_root(output_dir / "daic_phase1_window_manifest_clean.csv", root),
            "participant_window_mapping": path_relative_to_root(output_dir / "participant_window_mapping.csv", root),
            "final_layer_embedding_manifest_gpu": path_relative_to_root(output_dir / "wavlm_embedding_manifest_gpu.csv", root),
            "final_layer_extraction_metadata_gpu": path_relative_to_root(output_dir / "wavlm_extraction_metadata_gpu.json", root),
        },
        "input_hashes": {
            "participant_manifest_clean_sha256": sha256_file(output_dir / "daic_phase1_participant_manifest_clean.csv"),
            "window_manifest_clean_sha256": sha256_file(output_dir / "daic_phase1_window_manifest_clean.csv"),
            "participant_window_mapping_sha256": sha256_file(output_dir / "participant_window_mapping.csv"),
            "final_layer_embedding_manifest_gpu_sha256": sha256_file(output_dir / "wavlm_embedding_manifest_gpu.csv"),
            "final_layer_extraction_metadata_gpu_sha256": sha256_file(output_dir / "wavlm_extraction_metadata_gpu.json"),
        },
        "model_name": MODEL_NAME,
        "model_commit": MODEL_COMMIT,
        "batch_size": args.batch_size,
        "random_seed": RANDOM_SEED,
        "outer_nested_cv": {
            "splitter": "RepeatedStratifiedKFold",
            "n_splits": OUTER_SPLITS,
            "n_repeats": OUTER_REPEATS,
            "random_state": RANDOM_SEED,
        },
        "inner_c_selection": {
            "splitter": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "seed": "RANDOM_SEED * 1000 + outer_fold_index",
            "metric": "ROC-AUC",
        },
        "C_grid": C_GRID,
        "selected_layer": evaluation["selected_layer"],
        "selected_C": evaluation["selected_C"],
        "selected_dev_threshold": evaluation["selected_dev_threshold"],
        "test_evaluation_performed": False,
        "model_selection_dataframe_assertion": "official_split == test absent before labels were loaded",
        "input_validation_status": input_report["status"],
    }
    write_json(run_dir / "run_metadata.json", metadata)

    lines = [
        "# Phase 1B WavLM Layer-Wise Analysis",
        "",
        f"Run directory: `{path_relative_to_root(run_dir, root)}`",
        "",
        "## Extraction",
        f"- Valid layer-wise window files: `{extraction_metadata['total_valid_window_files']}`",
        f"- Extraction failures: `{extraction_metadata['failed']}`",
        f"- Layer-12 max absolute equivalence difference: `{validation_report['layer12_equivalence']['max_absolute_difference_max_over_windows']:.6g}`",
        f"- Layer-12 minimum cosine similarity: `{validation_report['layer12_equivalence']['cosine_similarity_min_over_windows']:.12g}`",
        "",
        "## Train-Only Layer Selection",
        f"- Selected layer: `{evaluation['selected_layer']}`",
        f"- Selected C: `{evaluation['selected_C']}`",
        f"- Best mean outer ROC-AUC: `{best['mean_outer_roc_auc']:.6f}`",
        f"- Layer 12 mean outer ROC-AUC: `{layer12['mean_outer_roc_auc']:.6f}`",
        f"- Difference vs layer 12: `{float(best['mean_outer_roc_auc'] - layer12['mean_outer_roc_auc']):.6f}`",
        "",
        "## Dev Evaluation",
        f"- Dev ROC-AUC: `{evaluation['dev_metrics_at_0_5']['roc_auc']:.6f}`",
        f"- Dev PR-AUC: `{evaluation['dev_metrics_at_0_5']['pr_auc']:.6f}`",
        f"- Dev macro F1 at 0.5: `{evaluation['dev_metrics_at_0_5']['macro_f1']:.6f}`",
        f"- Dev tuned threshold: `{evaluation['selected_dev_threshold']:.12g}`",
        f"- Dev macro F1 at tuned threshold: `{evaluation['dev_metrics_at_selected_threshold']['macro_f1']:.6f}`",
        "",
        "No test probabilities or test metrics were produced.",
    ]
    (run_dir / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    add_dependency_dir(args.dependency_dir)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model._logistic")

    root = Path.cwd()
    output_dir = resolve_path(args.output_dir, root)
    config_path = resolve_path(args.config, root)
    config = load_config(config_path)
    timestamp = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve_path(args.run_root, root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    torch, torchaudio, transformers_modules = require_runtime_modules()
    device = choose_cuda_device(torch)
    windows, participants, mapping_by_window = load_clean_inputs(output_dir)
    input_report = input_validation(root, config, output_dir, windows, participants, mapping_by_window, torch, torchaudio, run_dir)
    model_cache_dir = resolve_path(args.model_cache_dir, root)
    model, num_layers, hidden_dim, model_metadata = load_pinned_wavlm(device, transformers_modules, model_cache_dir)

    validation_report = validation_extraction(
        root,
        output_dir,
        run_dir,
        windows,
        participants,
        mapping_by_window,
        model,
        device,
        torch,
        torchaudio,
        num_layers,
        hidden_dim,
        args.batch_size,
    )
    extraction_metadata = full_layerwise_extraction(
        root,
        output_dir,
        run_dir,
        windows,
        participants,
        mapping_by_window,
        model,
        device,
        torch,
        torchaudio,
        num_layers,
        hidden_dim,
        args.batch_size,
        args.force,
        model_metadata,
    )
    del model
    torch.cuda.empty_cache()
    gc.collect()

    aggregation_metadata = aggregate_participant_layers(root, output_dir, run_dir)
    train_frame, x_train_layers, y_train, dev_frame, x_dev_layers, y_dev = load_train_dev_layer_inputs(root, output_dir, run_dir)
    assert not train_frame["official_split"].eq("test").any()
    assert not dev_frame["official_split"].eq("test").any()
    nested_summary, _, _, selected_report = nested_layer_evaluation(train_frame, x_train_layers, y_train, run_dir)
    evaluation = evaluate_selected_layer(root, run_dir, train_frame, x_train_layers, y_train, dev_frame, x_dev_layers, y_dev, selected_report, nested_summary)
    write_reports(root, output_dir, run_dir, args, input_report, validation_report, extraction_metadata, aggregation_metadata, selected_report, evaluation, nested_summary)

    print("Phase 1B layer-wise analysis complete.")
    print(f"run_dir={path_relative_to_root(run_dir, root)}")
    print(f"window_files={extraction_metadata['total_valid_window_files']} failures={extraction_metadata['failed']}")
    print(f"selected_layer={evaluation['selected_layer']} selected_C={evaluation['selected_C']}")
    print(f"dev_roc_auc={evaluation['dev_metrics_at_0_5']['roc_auc']:.6f}")
    print("test_evaluation_performed=false")


if __name__ == "__main__":
    main()
