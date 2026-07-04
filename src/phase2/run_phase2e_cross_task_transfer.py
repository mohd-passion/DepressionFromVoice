"""Execute Phase 2E Android reading-interview cross-task transfer.

This runner reuses existing Phase 2C/2D participant embeddings only. It creates
one participant-coherent five-fold split over paired Android participants,
evaluates reading-to-interview, interview-to-reading, pooled-task, and matched
within-task logistic-regression protocols, and writes compact reports.

It does not rerun WavLM extraction, preprocess audio, tune WavLM layers, use
timing features, or train any model family beyond balanced logistic regression.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .phase2e_protocols import (
        C_GRID,
        EXPECTED_DIM,
        FOLD_SEED,
        aggregate_protocol_metrics,
        assert_coherent_fold_integrity,
        assert_oof_prediction_completeness,
        assert_paired_identity,
        fit_source_fold_model,
        fold_metric_rows,
        make_participant_coherent_folds,
        metrics_for_rows,
        prediction_rows_for_protocol,
    )
except ImportError:  # pragma: no cover - allows direct script execution
    from phase2e_protocols import (
        C_GRID,
        EXPECTED_DIM,
        FOLD_SEED,
        aggregate_protocol_metrics,
        assert_coherent_fold_integrity,
        assert_oof_prediction_completeness,
        assert_paired_identity,
        fit_source_fold_model,
        fold_metric_rows,
        make_participant_coherent_folds,
        metrics_for_rows,
        prediction_rows_for_protocol,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/phase2e_cross_task_transfer.json"
PROTOCOLS = [
    ("matched_reading_within_task", "reading", "reading", "reading_only"),
    ("matched_interview_within_task", "interview", "interview", "interview_only"),
    ("reading_to_interview", "reading", "interview", "reading_only"),
    ("interview_to_reading", "interview", "reading", "interview_only"),
    ("pooled_to_reading", "pooled", "reading", "pooled"),
    ("pooled_to_interview", "pooled", "interview", "pooled"),
]
PRIMARY_THRESHOLD_VARIANT = "threshold_adapted"
BOOTSTRAP_METRICS = ["roc_auc", "pr_auc", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "brier_score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dependency-dir", default=".phase1_deps")
    return parser.parse_args()


def add_dependency_dir(path: str) -> None:
    dep = PROJECT_ROOT / path
    if dep.exists():
        sys.path.insert(0, str(dep.resolve()))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(p)


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


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
                    fieldnames.append(key)
                    seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
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


def validate_config(config: dict[str, Any]) -> None:
    if config["phase2a_run"].endswith("20260622_154008"):
        raise RuntimeError("Superseded Phase 2A run must not be used for Phase 2E.")
    expected = {
        "wavlm_model": "microsoft/wavlm-base-plus",
        "model_commit": "4c66d4806a428f2e922ccfa1a962776e232d487b",
        "wavlm_layer": "layer_09",
        "participant_aggregation": "mean",
        "embedding_dimension": EXPECTED_DIM,
        "embedding_dtype": "float32",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"Locked config mismatch for {key}: {config.get(key)} != {value}")
    for key, forbidden in config.get("forbidden", {}).items():
        if forbidden is not True:
            raise RuntimeError(f"Forbidden setting must remain true: {key}")


def load_embedding(path_text: str) -> np.ndarray:
    path = resolve_path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Missing participant embedding: {path_text}")
    embedding = np.load(path).astype(np.float32, copy=False)
    if embedding.shape != (EXPECTED_DIM,):
        raise ValueError(f"Bad embedding shape for {path_text}: {embedding.shape}")
    if embedding.dtype != np.float32:
        raise ValueError(f"Bad embedding dtype for {path_text}: {embedding.dtype}")
    if not np.all(np.isfinite(embedding)):
        raise ValueError(f"Embedding contains NaN or infinity: {path_text}")
    if np.all(embedding == 0):
        raise ValueError(f"Embedding is all zero: {path_text}")
    return embedding


def load_paired_data(config: dict[str, Any]) -> dict[str, Any]:
    reading_manifest_path = resolve_path(config["phase2d_run"]) / "android_reading_participant_embedding_manifest.csv"
    interview_manifest_path = resolve_path(config["phase2b_interview_run"]) / "android_interview_participant_embedding_manifest.csv"
    reading_rows = {row["participant_id"]: row for row in read_csv(reading_manifest_path)}
    interview_rows = {row["participant_id"]: row for row in read_csv(interview_manifest_path)}
    common_ids = sorted(set(reading_rows) & set(interview_rows))
    quality_rows: list[dict[str, Any]] = []
    participant_ids: list[str] = []
    labels: list[int] = []
    reading_embeddings: list[np.ndarray] = []
    interview_embeddings: list[np.ndarray] = []
    excluded: list[dict[str, Any]] = []

    for pid in sorted(set(reading_rows) | set(interview_rows)):
        reading = reading_rows.get(pid)
        interview = interview_rows.get(pid)
        status = "eligible"
        error = ""
        if reading is None or interview is None:
            status = "excluded_unpaired"
            error = "missing_reading" if reading is None else "missing_interview"
        elif reading["label_binary"] != interview["label_binary"]:
            status = "excluded_label_conflict"
            error = "reading/interview label mismatch"
        elif reading["condition"] != interview["condition"]:
            status = "excluded_condition_conflict"
            error = "reading/interview condition mismatch"
        elif reading["source_embedding_layer"] != "layer_09" or interview["source_embedding_layer"] != "layer_09":
            status = "excluded_bad_layer"
            error = "unexpected WavLM layer"
        elif int(reading["embedding_dimension"]) != EXPECTED_DIM or int(interview["embedding_dimension"]) != EXPECTED_DIM:
            status = "excluded_bad_dimension"
            error = "unexpected embedding dimension"
        quality = {
            "participant_id": pid,
            "condition": reading["condition"] if reading else interview["condition"],
            "label_binary": reading["label_binary"] if reading else interview["label_binary"],
            "has_reading_embedding": reading is not None,
            "has_interview_embedding": interview is not None,
            "reading_fold": reading.get("reading_fold", "") if reading else "",
            "interview_fold": interview.get("interview_fold", "") if interview else "",
            "status": status,
            "error_message": error,
        }
        if status == "eligible":
            try:
                reading_embedding = load_embedding(reading["participant_embedding_path"])
                interview_embedding = load_embedding(interview["participant_embedding_path"])
            except Exception as exc:
                status = "excluded_embedding_integrity_failure"
                error = str(exc)
                quality["status"] = status
                quality["error_message"] = error
            else:
                participant_ids.append(pid)
                labels.append(int(reading["label_binary"]))
                reading_embeddings.append(reading_embedding)
                interview_embeddings.append(interview_embedding)
        if status != "eligible":
            excluded.append({"participant_id": pid, "status": status, "error_message": error})
        quality_rows.append(quality)

    if len(participant_ids) != len(common_ids):
        # This is allowed only when a common participant failed explicit checks.
        failed_common = sorted(set(common_ids) - set(participant_ids))
        if failed_common:
            raise RuntimeError(f"Common participants failed eligibility checks: {failed_common[:5]}")

    reading_matrix = np.vstack(reading_embeddings).astype(np.float32)
    interview_matrix = np.vstack(interview_embeddings).astype(np.float32)
    label_array = np.asarray(labels, dtype=int)
    assert_paired_identity(participant_ids, label_array, reading_matrix, interview_matrix)
    return {
        "participant_ids": participant_ids,
        "labels": label_array,
        "reading_embeddings": reading_matrix,
        "interview_embeddings": interview_matrix,
        "quality_rows": quality_rows,
        "excluded_rows": excluded,
        "reading_manifest_path": reading_manifest_path,
        "interview_manifest_path": interview_manifest_path,
        "reading_total": len(reading_rows),
        "interview_total": len(interview_rows),
    }


def count_labels(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(int(x) for x in labels)
    return {"HC": int(counts.get(0, 0)), "PT": int(counts.get(1, 0))}


def fold_rows(participant_ids: list[str], labels: np.ndarray, folds: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "participant_id": pid,
            "label_binary": int(label),
            "condition": "HC" if int(label) == 0 else "PT",
            "coherent_fold": fold,
        }
        for pid, label, fold in zip(participant_ids, labels, folds)
    ]


def fold_summary_rows(labels: np.ndarray, folds: list[str]) -> list[dict[str, Any]]:
    y = np.asarray(labels, dtype=int)
    fold_array = np.asarray(folds)
    rows = []
    for fold in sorted(set(folds)):
        mask = fold_array == fold
        rows.append(
            {
                "coherent_fold": fold,
                "participants": int(np.sum(mask)),
                "HC": int(np.sum(y[mask] == 0)),
                "PT": int(np.sum(y[mask] == 1)),
                "outer_training_participants": int(np.sum(~mask)),
                "outer_test_participants": int(np.sum(mask)),
            }
        )
    return rows


def run_protocols(data: dict[str, Any], folds: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], Any]]:
    participant_ids = data["participant_ids"]
    labels = data["labels"]
    reading = data["reading_embeddings"]
    interview = data["interview_embeddings"]
    expected_ids = set(participant_ids)
    source_models: dict[tuple[str, str], Any] = {}
    params: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for outer_fold in sorted(set(folds)):
        for source_model in ["reading_only", "interview_only", "pooled"]:
            model = fit_source_fold_model(source_model, outer_fold, participant_ids, labels, folds, reading, interview)
            source_models[(source_model, outer_fold)] = model
            params.append(
                {
                    "source_model": source_model,
                    "outer_fold": outer_fold,
                    "selected_C": model.selected_c,
                    "selected_threshold": model.selected_threshold,
                    "threshold_macro_f1_train": model.threshold_macro_f1_train,
                    "threshold_balanced_accuracy_train": model.threshold_balanced_accuracy_train,
                    "threshold_candidate_count": model.threshold_candidate_count,
                }
            )
            inner_rows.extend(model.inner_rows)

    for protocol, source_task, target_task, source_model_name in PROTOCOLS:
        for outer_fold in sorted(set(folds)):
            prediction_rows.extend(
                prediction_rows_for_protocol(
                    protocol,
                    source_task,
                    target_task,
                    source_models[(source_model_name, outer_fold)],
                    participant_ids,
                    labels,
                    folds,
                    reading,
                    interview,
                )
            )
        for threshold_variant in ["threshold_0_5", "threshold_adapted"]:
            assert_oof_prediction_completeness(prediction_rows, expected_ids, protocol, threshold_variant)
    return prediction_rows, params, inner_rows, source_models


def selected_parameter_summary(params: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for source_model in sorted({row["source_model"] for row in params}):
        subset = [row for row in params if row["source_model"] == source_model]
        c_counts = Counter(str(row["selected_C"]) for row in subset)
        threshold_values = [float(row["selected_threshold"]) for row in subset]
        rows.append(
            {
                "source_model": source_model,
                "outer_folds": len(subset),
                "selected_C_distribution": json.dumps(dict(sorted(c_counts.items()))),
                "selected_threshold_min": float(np.min(threshold_values)),
                "selected_threshold_max": float(np.max(threshold_values)),
                "selected_threshold_mean": float(np.mean(threshold_values)),
                "selected_threshold_std": float(np.std(threshold_values, ddof=0)),
            }
        )
    return rows


def group_prediction_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["protocol"]), str(row["threshold_variant"])), []).append(row)
    return grouped


def _bootstrap_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    out: list[int] = []
    for label in sorted(set(y.tolist())):
        idx = np.where(y == label)[0]
        out.extend(rng.choice(idx, size=len(idx), replace=True).astype(int).tolist())
    return np.asarray(out, dtype=int)


def metric_values_for_sample(rows: list[dict[str, Any]], indices: np.ndarray) -> dict[str, float]:
    sampled = [rows[int(idx)] for idx in indices]
    metrics = metrics_for_rows(sampled)
    return {metric: float(metrics[metric]) for metric in BOOTSTRAP_METRICS}


def bootstrap_intervals(rows: list[dict[str, Any]], replicates: int, seed: int) -> list[dict[str, Any]]:
    grouped = group_prediction_rows(rows)
    out: list[dict[str, Any]] = []
    for (protocol, threshold_variant), group_rows in sorted(grouped.items()):
        labels = np.asarray([int(row["label_binary"]) for row in group_rows], dtype=int)
        observed = metrics_for_rows(group_rows)
        rng = np.random.default_rng(seed + abs(hash((protocol, threshold_variant))) % 100000)
        values = {metric: [] for metric in BOOTSTRAP_METRICS}
        for _ in range(replicates):
            idx = _bootstrap_indices(labels, rng)
            sample_metrics = metric_values_for_sample(group_rows, idx)
            for metric, value in sample_metrics.items():
                values[metric].append(value)
        for metric in BOOTSTRAP_METRICS:
            arr = np.asarray(values[metric], dtype=float)
            out.append(
                {
                    "protocol": protocol,
                    "threshold_variant": threshold_variant,
                    "metric": metric,
                    "observed": float(observed[metric]),
                    "bootstrap_replicates": replicates,
                    "p2_5": float(np.percentile(arr, 2.5)),
                    "median": float(np.percentile(arr, 50.0)),
                    "p97_5": float(np.percentile(arr, 97.5)),
                }
            )
    return out


def paired_comparison_rows(rows: list[dict[str, Any]], replicates: int, seed: int) -> list[dict[str, Any]]:
    pairs = [
        ("reading_to_interview_vs_matched_interview", "reading_to_interview", "matched_interview_within_task"),
        ("interview_to_reading_vs_matched_reading", "interview_to_reading", "matched_reading_within_task"),
        ("pooled_to_interview_vs_reading_to_interview", "pooled_to_interview", "reading_to_interview"),
        ("pooled_to_reading_vs_interview_to_reading", "pooled_to_reading", "interview_to_reading"),
        ("reading_to_interview_vs_interview_to_reading", "reading_to_interview", "interview_to_reading"),
    ]
    grouped = group_prediction_rows(rows)
    out: list[dict[str, Any]] = []
    for threshold_variant in ["threshold_0_5", "threshold_adapted"]:
        for comparison, protocol_a, protocol_b in pairs:
            rows_a = sorted(grouped[(protocol_a, threshold_variant)], key=lambda row: row["participant_id"])
            rows_b = sorted(grouped[(protocol_b, threshold_variant)], key=lambda row: row["participant_id"])
            ids_a = [row["participant_id"] for row in rows_a]
            ids_b = [row["participant_id"] for row in rows_b]
            if ids_a != ids_b:
                raise ValueError(f"Paired comparison ID mismatch for {comparison}/{threshold_variant}")
            labels = np.asarray([int(row["label_binary"]) for row in rows_a], dtype=int)
            metrics_a = metrics_for_rows(rows_a)
            metrics_b = metrics_for_rows(rows_b)
            rng = np.random.default_rng(seed + abs(hash((comparison, threshold_variant))) % 100000)
            sampled_diffs = {metric: [] for metric in BOOTSTRAP_METRICS}
            for _ in range(replicates):
                idx = _bootstrap_indices(labels, rng)
                sample_a = metric_values_for_sample(rows_a, idx)
                sample_b = metric_values_for_sample(rows_b, idx)
                for metric in BOOTSTRAP_METRICS:
                    sampled_diffs[metric].append(float(sample_a[metric]) - float(sample_b[metric]))
            for metric in BOOTSTRAP_METRICS:
                arr = np.asarray(sampled_diffs[metric], dtype=float)
                out.append(
                    {
                        "comparison": comparison,
                        "protocol_a": protocol_a,
                        "protocol_b": protocol_b,
                        "threshold_variant": threshold_variant,
                        "metric": metric,
                        "observed_difference_a_minus_b": float(metrics_a[metric]) - float(metrics_b[metric]),
                        "bootstrap_replicates": replicates,
                        "p2_5": float(np.percentile(arr, 2.5)),
                        "median": float(np.percentile(arr, 50.0)),
                        "p97_5": float(np.percentile(arr, 97.5)),
                    }
                )
    return out


def protocol_metric_lookup(metric_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["protocol"], row["threshold_variant"]): row for row in metric_rows}


def transfer_gap_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = protocol_metric_lookup(metric_rows)
    comparisons = [
        ("reading_to_interview_gap", "reading_to_interview", "matched_interview_within_task"),
        ("interview_to_reading_gap", "interview_to_reading", "matched_reading_within_task"),
        ("pooled_to_interview_gain", "pooled_to_interview", "reading_to_interview"),
        ("pooled_to_reading_gain", "pooled_to_reading", "interview_to_reading"),
    ]
    rows = []
    for threshold_variant in ["threshold_0_5", "threshold_adapted"]:
        for comparison, protocol_a, protocol_b in comparisons:
            a = lookup[(protocol_a, threshold_variant)]
            b = lookup[(protocol_b, threshold_variant)]
            for metric in BOOTSTRAP_METRICS:
                rows.append(
                    {
                        "comparison": comparison,
                        "protocol_a": protocol_a,
                        "protocol_b": protocol_b,
                        "threshold_variant": threshold_variant,
                        "metric": metric,
                        "value_a": a[metric],
                        "value_b": b[metric],
                        "difference_a_minus_b": float(a[metric]) - float(b[metric]),
                    }
                )
    return rows


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def classifier_boundary_rows(
    source_models: dict[tuple[str, str], Any],
    folds: list[str],
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    fold_array = np.asarray(folds)
    reading = data["reading_embeddings"]
    interview = data["interview_embeddings"]
    for fold in sorted(set(folds)):
        reading_model = source_models[("reading_only", fold)]
        interview_model = source_models[("interview_only", fold)]
        read_clf = reading_model.pipeline.named_steps["classifier"]
        int_clf = interview_model.pipeline.named_steps["classifier"]
        read_scaler = reading_model.pipeline.named_steps["scaler"]
        int_scaler = interview_model.pipeline.named_steps["scaler"]
        read_coef = read_clf.coef_[0]
        int_coef = int_clf.coef_[0]
        read_abs = np.abs(read_coef)
        int_abs = np.abs(int_coef)
        top_k = 50
        read_top = set(np.argsort(read_abs)[-top_k:].astype(int).tolist())
        int_top = set(np.argsort(int_abs)[-top_k:].astype(int).tolist())
        test_mask = fold_array == fold
        read_score_on_reading = reading_model.pipeline.predict_proba(reading[test_mask])[:, 1]
        interview_score_on_interview = interview_model.pipeline.predict_proba(interview[test_mask])[:, 1]
        read_score_on_interview = reading_model.pipeline.predict_proba(interview[test_mask])[:, 1]
        interview_score_on_reading = interview_model.pipeline.predict_proba(reading[test_mask])[:, 1]
        rows.append(
            {
                "outer_fold": fold,
                "reading_selected_C": reading_model.selected_c,
                "interview_selected_C": interview_model.selected_c,
                "reading_coefficient_norm": float(np.linalg.norm(read_coef)),
                "interview_coefficient_norm": float(np.linalg.norm(int_coef)),
                "coefficient_cosine_similarity_standardized_space": cosine_similarity(read_coef, int_coef),
                "coefficient_sign_agreement": float(np.mean(np.sign(read_coef) == np.sign(int_coef))),
                "top_abs_weight_feature_overlap_at_50": len(read_top & int_top),
                "top_abs_weight_feature_jaccard_at_50": float(len(read_top & int_top) / len(read_top | int_top)),
                "reading_intercept": float(read_clf.intercept_[0]),
                "interview_intercept": float(int_clf.intercept_[0]),
                "scaler_mean_l2_shift": float(np.linalg.norm(read_scaler.mean_ - int_scaler.mean_)),
                "scaler_mean_abs_shift": float(np.mean(np.abs(read_scaler.mean_ - int_scaler.mean_))),
                "scaler_variance_l2_shift": float(np.linalg.norm(read_scaler.var_ - int_scaler.var_)),
                "scaler_variance_abs_shift": float(np.mean(np.abs(read_scaler.var_ - int_scaler.var_))),
                "score_correlation_matched_source_tasks": safe_corr(read_score_on_reading, interview_score_on_interview),
                "score_correlation_cross_targets": safe_corr(read_score_on_interview, interview_score_on_reading),
                "reading_probability_mean_on_reading": float(np.mean(read_score_on_reading)),
                "interview_probability_mean_on_interview": float(np.mean(interview_score_on_interview)),
                "reading_probability_mean_on_interview": float(np.mean(read_score_on_interview)),
                "interview_probability_mean_on_reading": float(np.mean(interview_score_on_reading)),
            }
        )
    return rows


def summarize_boundary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "coefficient_cosine_similarity_standardized_space",
        "coefficient_sign_agreement",
        "top_abs_weight_feature_jaccard_at_50",
        "scaler_mean_l2_shift",
        "scaler_variance_l2_shift",
        "score_correlation_matched_source_tasks",
        "score_correlation_cross_targets",
    ]
    out: dict[str, Any] = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows if row[field] == row[field]], dtype=float)
        out[field] = {
            "mean": float(np.mean(values)) if len(values) else None,
            "std": float(np.std(values, ddof=0)) if len(values) else None,
            "min": float(np.min(values)) if len(values) else None,
            "max": float(np.max(values)) if len(values) else None,
        }
    return out


def classify_interpretation(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = protocol_metric_lookup(metric_rows)
    variant = PRIMARY_THRESHOLD_VARIANT
    r2i = lookup[("reading_to_interview", variant)]["roc_auc"]
    i2r = lookup[("interview_to_reading", variant)]["roc_auc"]
    matched_i = lookup[("matched_interview_within_task", variant)]["roc_auc"]
    matched_r = lookup[("matched_reading_within_task", variant)]["roc_auc"]
    pooled_i = lookup[("pooled_to_interview", variant)]["roc_auc"]
    pooled_r = lookup[("pooled_to_reading", variant)]["roc_auc"]
    gap_i = float(r2i) - float(matched_i)
    gap_r = float(i2r) - float(matched_r)
    pooled_gain_i = float(pooled_i) - float(r2i)
    pooled_gain_r = float(pooled_r) - float(i2r)
    if gap_i >= -0.05 and gap_r >= -0.05:
        category = "strong_bidirectional_task_transfer"
    elif (gap_i >= -0.05 and gap_r < -0.10) or (gap_r >= -0.05 and gap_i < -0.10):
        category = "asymmetric_task_transfer"
    elif gap_i <= -0.10 and gap_r <= -0.10 and matched_i >= 0.75 and matched_r >= 0.75:
        category = "task_specific_classifier_boundaries"
    elif pooled_gain_i > 0.02 and pooled_gain_r > 0.02:
        category = "pooled_training_advantage"
    elif gap_i < -0.05 or gap_r < -0.05:
        category = "weak_task_transfer"
    else:
        category = "mixed_cross_task_transfer"
    return {
        "category": category,
        "reading_to_interview_auc": r2i,
        "interview_to_reading_auc": i2r,
        "matched_interview_auc": matched_i,
        "matched_reading_auc": matched_r,
        "pooled_to_interview_auc": pooled_i,
        "pooled_to_reading_auc": pooled_r,
        "reading_to_interview_auc_gap_vs_matched_interview": gap_i,
        "interview_to_reading_auc_gap_vs_matched_reading": gap_r,
        "pooled_to_interview_auc_gain_vs_reading_to_interview": pooled_gain_i,
        "pooled_to_reading_auc_gain_vs_interview_to_reading": pooled_gain_r,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 2E Cross-Task Transfer Report",
        "",
        f"Status: `{report['status']}`",
        f"Run directory: `{report['run_directory']}`",
        "",
        "## Paired Participants",
        "",
        f"- Eligible paired participants: {report['data_summary']['paired_participants']}",
        f"- Class counts: HC={report['data_summary']['class_counts']['HC']}, PT={report['data_summary']['class_counts']['PT']}",
        f"- Reading-only excluded: {report['data_summary']['reading_only_participants']}",
        f"- Interview-only excluded: {report['data_summary']['interview_only_participants']}",
        "",
        "## Protocol Metrics",
        "",
        "| Protocol | Threshold | ROC-AUC | PR-AUC | BA | Macro F1 | Sens | Spec | Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["protocol_metrics"]:
        lines.append(
            "| {protocol} | {threshold_variant} | {roc_auc:.4f} | {pr_auc:.4f} | {balanced_accuracy:.4f} | {macro_f1:.4f} | {sensitivity:.4f} | {specificity:.4f} | {brier_score:.4f} |".format(
                **row
            )
        )
    interp = report["interpretation"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Category: `{interp['category']}`",
            f"- Reading-to-interview ROC-AUC gap versus matched interview: {interp['reading_to_interview_auc_gap_vs_matched_interview']:.4f}",
            f"- Interview-to-reading ROC-AUC gap versus matched reading: {interp['interview_to_reading_auc_gap_vs_matched_reading']:.4f}",
            f"- Pooled-to-interview ROC-AUC gain versus reading-to-interview: {interp['pooled_to_interview_auc_gain_vs_reading_to_interview']:.4f}",
            f"- Pooled-to-reading ROC-AUC gain versus interview-to-reading: {interp['pooled_to_reading_auc_gain_vs_interview_to_reading']:.4f}",
            "",
            "## Safeguards",
            "",
            f"- Leakage status: `{report['leakage_checks']['status']}`",
            "- No WavLM extraction, audio preprocessing, WavLM fine-tuning, task fusion, or new model family was performed.",
            "- Participant IDs and participant-level predictions remain only in ignored local outputs.",
        ]
    )
    return "\n".join(lines) + "\n"


def checksum_report(run_dir: Path) -> dict[str, str]:
    checksums = {}
    for path in sorted(run_dir.glob("*")):
        if path.is_file() and path.name != "artifact_checksums.json":
            checksums[rel(path)] = sha256_file(path)
    return checksums


def copy_compact_record(run_dir: Path, config: dict[str, Any]) -> Path:
    compact_root = resolve_path(config["compact_record_root"])
    compact_dir = compact_root / f"phase2e_{run_dir.name}"
    compact_dir.mkdir(parents=True, exist_ok=True)
    safe_files = [
        "PHASE2E_CROSS_TASK_REPORT.md",
        "PHASE2E_CROSS_TASK_REPORT.json",
        "phase2e_protocol_comparison.csv",
        "phase2e_transfer_gap_analysis.csv",
        "phase2e_selected_parameters_summary.csv",
        "phase2e_classifier_boundary_analysis.csv",
        "phase2e_run_metadata.json",
        "artifact_checksums.json",
    ]
    for name in safe_files:
        shutil.copy2(run_dir / name, compact_dir / name)
    return compact_dir


def run_phase2e(config_path: Path, run_id: str = "") -> Path:
    config = read_json(config_path)
    validate_config(config)
    start = utc_now()
    run_stamp = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve_path(config["output_root"]) / run_stamp
    run_dir.mkdir(parents=True, exist_ok=False)

    data = load_paired_data(config)
    folds = make_participant_coherent_folds(data["labels"], seed=int(config["fold_seed"]))
    fold_summaries = assert_coherent_fold_integrity(data["participant_ids"], data["labels"], folds)

    quality_rows = data["quality_rows"]
    write_csv(run_dir / "phase2e_paired_participant_quality_report.csv", quality_rows)
    participant_summary = {
        "status": "passed",
        "reading_participants": data["reading_total"],
        "interview_participants": data["interview_total"],
        "paired_participants": len(data["participant_ids"]),
        "class_counts": count_labels(data["labels"]),
        "reading_only_participants": data["reading_total"] - len(data["participant_ids"]),
        "interview_only_participants": data["interview_total"] - len(data["participant_ids"]),
        "excluded_count": len(data["excluded_rows"]),
        "embedding_dimension": EXPECTED_DIM,
        "all_vectors_finite_nonzero": True,
    }
    write_json(run_dir / "phase2e_paired_participant_summary.json", participant_summary)
    write_csv(run_dir / "phase2e_participant_coherent_folds.csv", fold_rows(data["participant_ids"], data["labels"], folds))
    fold_summary = fold_summary_rows(data["labels"], folds)
    write_csv(run_dir / "phase2e_fold_summary.csv", fold_summary)
    fold_metadata = {
        "method": "deterministic stratified round-robin assignment after per-class seeded shuffle",
        "seed": FOLD_SEED,
        "participant_count": len(data["participant_ids"]),
        "class_counts": count_labels(data["labels"]),
    }
    write_json(run_dir / "phase2e_fold_metadata.json", fold_metadata)
    fold_metadata["fold_manifest_sha256"] = sha256_file(run_dir / "phase2e_participant_coherent_folds.csv")
    write_json(run_dir / "phase2e_fold_metadata.json", fold_metadata)

    prediction_rows, selected_params, inner_rows, source_models = run_protocols(data, folds)
    metric_rows = aggregate_protocol_metrics(prediction_rows)
    per_fold_metrics = fold_metric_rows(prediction_rows)
    selected_summary = selected_parameter_summary(selected_params)
    boundary_rows = classifier_boundary_rows(source_models, folds, data)
    boundary_summary = summarize_boundary(boundary_rows)
    transfer_gaps = transfer_gap_rows(metric_rows)
    bootstrap_rows = bootstrap_intervals(prediction_rows, int(config["bootstrap_replicates"]), int(config["random_seed"]))
    paired_rows = paired_comparison_rows(prediction_rows, int(config["bootstrap_replicates"]), int(config["random_seed"]))
    interpretation = classify_interpretation(metric_rows)

    write_csv(run_dir / "phase2e_outer_predictions.csv", prediction_rows)
    write_csv(run_dir / "phase2e_selected_parameters.csv", selected_params)
    write_csv(run_dir / "phase2e_selected_parameters_summary.csv", selected_summary)
    write_csv(run_dir / "phase2e_inner_c_selection.csv", inner_rows)
    write_csv(run_dir / "phase2e_protocol_comparison.csv", metric_rows)
    write_csv(run_dir / "phase2e_fold_metrics.csv", per_fold_metrics)
    write_csv(run_dir / "phase2e_bootstrap_intervals.csv", bootstrap_rows)
    write_csv(run_dir / "phase2e_paired_comparisons.csv", paired_rows)
    write_csv(run_dir / "phase2e_transfer_gap_analysis.csv", transfer_gaps)
    write_csv(run_dir / "phase2e_classifier_boundary_analysis.csv", boundary_rows)

    leakage_report = {
        "status": "passed",
        "participant_grouping_unit": "participant_id",
        "participant_coherent_folds": True,
        "outer_test_participants_excluded_from_all_training_tasks": True,
        "inner_validation_participants_excluded_from_inner_training_tasks": True,
        "reading_to_interview_uses_reading_training_only": True,
        "interview_to_reading_uses_interview_training_only": True,
        "pooled_training_uses_both_tasks_for_training_participants_only": True,
        "oof_prediction_completeness": "one prediction per participant per protocol and threshold variant",
    }
    write_json(run_dir / "phase2e_leakage_report.json", leakage_report)
    quality_report = {
        "status": "passed",
        "paired_participant_summary": participant_summary,
        "fold_summary": fold_summary,
        "source_embedding_manifests": {
            "reading": rel(data["reading_manifest_path"]),
            "interview": rel(data["interview_manifest_path"]),
        },
    }
    write_json(run_dir / "phase2e_quality_report.json", quality_report)

    report = {
        "status": "completed_phase2e_cross_task_transfer",
        "created_at": utc_now(),
        "run_directory": rel(run_dir),
        "official_inputs": {
            "phase2a_run": config["phase2a_run"],
            "phase2c_run": config["phase2c_run"],
            "phase2d_run": config["phase2d_run"],
            "phase2b_interview_run": config["phase2b_interview_run"],
            "superseded_phase2a_used": False,
        },
        "data_summary": participant_summary,
        "fold_summary": fold_summary,
        "protocol_metrics": metric_rows,
        "frozen_reference_metrics": config["frozen_references"],
        "selected_parameter_summary": selected_summary,
        "transfer_gaps": transfer_gaps,
        "classifier_boundary_summary": boundary_summary,
        "interpretation": interpretation,
        "leakage_checks": leakage_report,
        "bootstrap": {
            "replicates": int(config["bootstrap_replicates"]),
            "intervals_path": rel(run_dir / "phase2e_bootstrap_intervals.csv"),
            "paired_comparisons_path": rel(run_dir / "phase2e_paired_comparisons.csv"),
        },
        "prohibited_changes_confirmed": {
            "new_wavlm_extraction": False,
            "audio_preprocessing": False,
            "wavlm_fine_tuning": False,
            "new_model_family": False,
            "task_feature_fusion": False,
        },
    }
    write_json(run_dir / "PHASE2E_CROSS_TASK_REPORT.json", report)
    (run_dir / "PHASE2E_CROSS_TASK_REPORT.md").write_text(markdown_report(report), encoding="utf-8")

    metadata = {
        "created_at": utc_now(),
        "execution_start_timestamp": start,
        "execution_completion_timestamp": utc_now(),
        "config_path": rel(config_path),
        "config_sha256": sha256_file(config_path),
        "run_directory": rel(run_dir),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "sklearn_version": package_version("scikit-learn"),
        "joblib_version": package_version("joblib"),
        "machine_name": platform.node(),
        "git_commit": git_commit(),
        "random_seed": int(config["random_seed"]),
        "fold_seed": int(config["fold_seed"]),
        "bootstrap_replicates": int(config["bootstrap_replicates"]),
        "c_grid": C_GRID,
        "input_hashes": {
            "config": sha256_file(config_path),
            "reading_manifest": sha256_file(data["reading_manifest_path"]),
            "interview_manifest": sha256_file(data["interview_manifest_path"]),
            "phase2c_report": sha256_file(resolve_path(config["phase2c_run"]) / "PHASE2C_WITHIN_ANDROID_REPORT.json"),
            "phase2d_report": sha256_file(resolve_path(config["phase2d_run"]) / "PHASE2D_ANDROID_READING_REPORT.json"),
        },
        "modeling_scope": "Phase 2E only; existing participant embeddings; balanced logistic regression; no WavLM extraction",
    }
    write_json(run_dir / "phase2e_run_metadata.json", metadata)
    write_json(run_dir / "artifact_checksums.json", checksum_report(run_dir))
    compact_dir = copy_compact_record(run_dir, config)
    report["compact_record_directory"] = rel(compact_dir)
    write_json(run_dir / "PHASE2E_CROSS_TASK_REPORT.json", report)
    write_json(run_dir / "artifact_checksums.json", checksum_report(run_dir))
    shutil.copy2(run_dir / "PHASE2E_CROSS_TASK_REPORT.json", compact_dir / "PHASE2E_CROSS_TASK_REPORT.json")
    shutil.copy2(run_dir / "artifact_checksums.json", compact_dir / "artifact_checksums.json")
    return run_dir


def main() -> None:
    args = parse_args()
    add_dependency_dir(args.dependency_dir)
    run_dir = run_phase2e(resolve_path(args.config), args.run_id)
    print(f"phase2e_run_dir={rel(run_dir)}")


if __name__ == "__main__":
    main()
