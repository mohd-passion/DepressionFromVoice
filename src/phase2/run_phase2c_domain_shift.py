"""Execute Phase 2C DAIC-to-Android domain-shift diagnosis.

This runner uses existing Phase 2B Android interview participant embeddings and
zero-shot probabilities. It does not rerun WavLM, process reading speech, use
timing features, fine-tune models, or alter frozen Phase 1 artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
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
    from .phase2c_domain_shift import embedding_shift_report, summary_rows as domain_shift_summary_rows
    from .phase2c_metrics import (
        calibration_slope_intercept,
        expected_calibration_error,
        probability_metrics,
        threshold_metrics_from_counts,
    )
    from .phase2c_protocols import (
        FROZEN_DAIC_THRESHOLD,
        aggregate_protocol_metrics,
        assert_fold_integrity,
        protocol_a_rows,
        protocol_b_threshold_adaptation,
        protocol_c_calibration,
        protocol_d_android_logistic_regression,
    )
except ImportError:  # pragma: no cover - allows direct script execution
    from phase2c_domain_shift import embedding_shift_report, summary_rows as domain_shift_summary_rows
    from phase2c_metrics import (
        calibration_slope_intercept,
        expected_calibration_error,
        probability_metrics,
        threshold_metrics_from_counts,
    )
    from phase2c_protocols import (
        FROZEN_DAIC_THRESHOLD,
        aggregate_protocol_metrics,
        assert_fold_integrity,
        protocol_a_rows,
        protocol_b_threshold_adaptation,
        protocol_c_calibration,
        protocol_d_android_logistic_regression,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/phase2c_android_domain_shift.json"


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


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


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


def load_android_data(config: dict[str, Any]) -> dict[str, Any]:
    phase2b = PROJECT_ROOT / config["phase2b_run"]
    manifest = read_csv(phase2b / "android_interview_participant_embedding_manifest.csv")
    predictions = {row["participant_id"]: row for row in read_csv(phase2b / "android_interview_zero_shot_predictions.csv")}
    if len(manifest) != 116:
        raise ValueError(f"Expected 116 Android interview participants; found {len(manifest)}")
    rows = sorted(manifest, key=lambda row: (row["condition"], row["participant_id"]))
    participant_ids: list[str] = []
    labels: list[int] = []
    folds: list[str] = []
    embeddings: list[np.ndarray] = []
    frozen_probs: list[float] = []
    for row in rows:
        pid = row["participant_id"]
        if row["task"] != "interview":
            raise ValueError(f"Non-interview row in participant manifest: {pid}")
        if pid not in predictions:
            raise ValueError(f"Missing Phase 2B zero-shot prediction for {pid}")
        if row["source_embedding_layer"] != config["wavlm_layer"]:
            raise ValueError(f"Unexpected WavLM layer for {pid}: {row['source_embedding_layer']}")
        if int(row["embedding_dimension"]) != int(config["embedding_dimension"]):
            raise ValueError(f"Unexpected embedding dimension for {pid}: {row['embedding_dimension']}")
        path = PROJECT_ROOT / row["participant_embedding_path"]
        embedding = np.load(path).astype(np.float32)
        if embedding.shape != (int(config["embedding_dimension"]),):
            raise ValueError(f"Bad embedding shape for {pid}: {embedding.shape}")
        if not np.all(np.isfinite(embedding)) or np.all(embedding == 0):
            raise ValueError(f"Invalid embedding values for {pid}")
        participant_ids.append(pid)
        labels.append(int(row["label_binary"]))
        folds.append(row["interview_fold"])
        embeddings.append(embedding)
        frozen_probs.append(float(predictions[pid]["predicted_probability"]))
    label_counts = Counter(labels)
    if label_counts[0] != 52 or label_counts[1] != 64:
        raise ValueError(f"Expected Android class counts HC=52/PT=64, found {label_counts}")
    return {
        "participant_ids": participant_ids,
        "labels": np.asarray(labels, dtype=int),
        "folds": folds,
        "embeddings": np.vstack(embeddings).astype(np.float32),
        "frozen_probabilities": np.asarray(frozen_probs, dtype=float),
        "manifest_rows": rows,
    }


def load_daic_train_embeddings(config: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    phase1c = PROJECT_ROOT / config["phase1c_run"]
    rows = [
        row
        for row in read_csv(phase1c / "aggregation_manifest.csv")
        if row["official_split"] == "train" and row["aggregation_method"] == "mean"
    ]
    rows = sorted(rows, key=lambda row: int(row["participant_id"]))
    if len(rows) != 106:
        raise ValueError(f"Expected 106 DAIC train participants; found {len(rows)}")
    ids = [row["participant_id"] for row in rows]
    embeddings = []
    for row in rows:
        arr = np.load(PROJECT_ROOT / row["representation_path"]).astype(np.float32)
        if arr.shape != (768,):
            raise ValueError(f"Bad DAIC embedding shape for {row['participant_id']}: {arr.shape}")
        embeddings.append(arr)
    return ids, np.vstack(embeddings).astype(np.float32)


def rowwise_threshold_predictions(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(float(row["predicted_probability"]) >= float(row["decision_threshold"])) for row in rows], dtype=int)


def metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    labels = np.asarray([int(row["label_binary"]) for row in rows], dtype=int)
    probs = np.asarray([float(row["predicted_probability"]) for row in rows], dtype=float)
    preds = rowwise_threshold_predictions(rows)
    out: dict[str, float | int] = {}
    out.update(probability_metrics(labels, probs))
    out.update(threshold_metrics_from_counts({
        "tn": int(np.sum((labels == 0) & (preds == 0))),
        "fp": int(np.sum((labels == 0) & (preds == 1))),
        "fn": int(np.sum((labels == 1) & (preds == 0))),
        "tp": int(np.sum((labels == 1) & (preds == 1))),
    }))
    out["participant_count"] = len(rows)
    return out


def fold_metrics(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in all_rows:
        by_key.setdefault((str(row["protocol"]), str(row["interview_fold"])), []).append(row)
    for (protocol, fold), rows in sorted(by_key.items()):
        metrics = metrics_for_rows(rows)
        out.append({"protocol": protocol, "outer_fold": fold, **metrics})
    return out


def calibration_report_rows(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_protocol: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_protocol.setdefault(str(row["protocol"]), []).append(row)
    for protocol, rows in sorted(by_protocol.items()):
        labels = np.asarray([int(row["label_binary"]) for row in rows], dtype=int)
        probs = np.asarray([float(row["predicted_probability"]) for row in rows], dtype=float)
        thresholds = np.asarray([float(row["decision_threshold"]) for row in rows], dtype=float)
        preds = probs >= thresholds
        ece, bins = expected_calibration_error(labels, probs)
        summary = {
            "row_type": "summary",
            "protocol": protocol,
            "probability_min": float(np.min(probs)),
            "probability_max": float(np.max(probs)),
            "probability_mean": float(np.mean(probs)),
            "probability_std": float(np.std(probs, ddof=0)),
            "predicted_positive_rate": float(np.mean(preds)),
            "brier_score": float(np.mean((probs - labels) ** 2)),
            "expected_calibration_error": ece,
        }
        summary.update(calibration_slope_intercept(labels, probs))
        out.append(summary)
        for bin_row in bins:
            out.append({"row_type": "fixed_bin", "protocol": protocol, **bin_row})
    return out


def stratified_bootstrap_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    parts = []
    for cls in sorted(np.unique(y)):
        cls_idx = np.where(y == cls)[0]
        parts.append(rng.choice(cls_idx, size=len(cls_idx), replace=True))
    return np.concatenate(parts)


def bootstrap_intervals(
    all_rows: list[dict[str, Any]],
    n_replicates: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(random_seed)
    by_protocol: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_protocol.setdefault(str(row["protocol"]), []).append(row)
    metrics = ["roc_auc", "pr_auc", "balanced_accuracy", "macro_f1", "sensitivity", "specificity", "brier_score"]
    interval_rows: list[dict[str, Any]] = []
    boot_values: dict[str, dict[str, list[float]]] = {p: {m: [] for m in metrics} for p in by_protocol}
    for protocol, rows in sorted(by_protocol.items()):
        labels = np.asarray([int(row["label_binary"]) for row in rows], dtype=int)
        for _ in range(n_replicates):
            idx = stratified_bootstrap_indices(labels, rng)
            sample = [rows[i] for i in idx]
            vals = metrics_for_rows(sample)
            for metric in metrics:
                value = float(vals[metric])
                if np.isfinite(value):
                    boot_values[protocol][metric].append(value)
        point = metrics_for_rows(rows)
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
        ("A_frozen_daic_at_frozen_threshold", "B_threshold_only", "A_vs_B"),
        ("A_frozen_daic_at_frozen_threshold", "C_frozen_score_calibration", "A_vs_C"),
        ("A_frozen_daic_at_frozen_threshold", "D_android_lr_threshold_adapted", "A_vs_D"),
    ]
    aggregate = {protocol: metrics_for_rows(rows) for protocol, rows in by_protocol.items()}
    best_frozen = max(
        ["B_threshold_only", "C_frozen_score_calibration"],
        key=lambda name: (aggregate[name]["macro_f1"], aggregate[name]["balanced_accuracy"]),
    )
    comparison_pairs.append((best_frozen, "D_android_lr_threshold_adapted", f"{best_frozen}_vs_D"))
    comparison_rows: list[dict[str, Any]] = []
    for left, right, label in comparison_pairs:
        for metric in metrics:
            left_vals = np.asarray(boot_values[left][metric], dtype=float)
            right_vals = np.asarray(boot_values[right][metric], dtype=float)
            n = min(len(left_vals), len(right_vals))
            diff = right_vals[:n] - left_vals[:n]
            comparison_rows.append(
                {
                    "comparison": label,
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


def interpretation(protocol_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    by_protocol = {row["protocol"]: row for row in protocol_metrics}
    a = by_protocol["A_frozen_daic_at_frozen_threshold"]
    b = by_protocol["B_threshold_only"]
    c = by_protocol["C_frozen_score_calibration"]
    d = by_protocol["D_android_lr_threshold_adapted"]
    best_frozen = b if (b["macro_f1"], b["balanced_accuracy"]) >= (c["macro_f1"], c["balanced_accuracy"]) else c
    ranking_gain_d = float(d["roc_auc"] - a["roc_auc"])
    classification_gain_frozen = float(best_frozen["macro_f1"] - a["macro_f1"])
    classification_gain_d = float(d["macro_f1"] - a["macro_f1"])
    if ranking_gain_d >= 0.05:
        dominant = "classifier_boundary_shift"
    elif classification_gain_frozen >= 0.10 and abs(float(best_frozen["roc_auc"] - a["roc_auc"])) < 0.03:
        dominant = "threshold_or_calibration_shift"
    elif float(d["roc_auc"]) < 0.60:
        dominant = "representation_limitation"
    else:
        dominant = "mixed_threshold_calibration_and_boundary_shift"
    return {
        "dominant_failure_source": dominant,
        "best_adapted_frozen_score_protocol": best_frozen["protocol"],
        "android_lr_roc_auc_gain_over_frozen": ranking_gain_d,
        "best_frozen_score_macro_f1_gain_over_frozen": classification_gain_frozen,
        "android_lr_macro_f1_gain_over_frozen": classification_gain_d,
        "clinical_validity_claim": False,
    }


def markdown_report(report: dict[str, Any]) -> str:
    metrics = report["protocol_metrics"]
    lines = [
        "# Phase 2C DAIC-to-Android Domain-Shift Diagnosis",
        "",
        f"Run directory: `{report['run_directory']}`",
        "",
        "## Integrity",
        "",
        f"- Participants: {report['data_summary']['participants']} ({report['data_summary']['class_counts']})",
        f"- Fold leakage status: {report['leakage_checks']['status']}",
        f"- WavLM rerun performed: {report['scope_confirmation']['wavlm_rerun_performed']}",
        f"- Reading speech processed: {report['scope_confirmation']['reading_speech_processed']}",
        "",
        "## Protocol Metrics",
        "",
        "| Protocol | ROC-AUC | PR-AUC | Balanced Acc. | Macro F1 | Sens. | Spec. | Brier | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        threshold = row.get("threshold")
        threshold_text = "" if threshold is None else f"{threshold:.6f}" if isinstance(threshold, float) and np.isfinite(threshold) else "fold-specific"
        lines.append(
            f"| {row['protocol']} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['balanced_accuracy']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['sensitivity']:.4f} | {row['specificity']:.4f} | {row['brier_score']:.4f} | {threshold_text} |"
        )
    lines.extend(
        [
            "",
            "## Domain-Shift Finding",
            "",
            f"Dominant failure source: `{report['interpretation']['dominant_failure_source']}`.",
            "",
            report["interpretation"]["narrative"],
            "",
            "No clinical-validity claim is made.",
        ]
    )
    return "\n".join(lines) + "\n"


def create_compact_record(run_dir: Path, compact_root: Path) -> Path:
    compact_dir = compact_root / f"phase2c_{run_dir.name}"
    compact_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "PHASE2C_WITHIN_ANDROID_REPORT.md",
        "PHASE2C_WITHIN_ANDROID_REPORT.json",
        "phase2c_protocol_comparison.csv",
        "phase2c_selected_parameters.csv",
        "phase2c_run_metadata.json",
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
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = read_json(config_path)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / config["output_root"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    android = load_android_data(config)
    fold_summaries = assert_fold_integrity(android["participant_ids"], android["folds"], android["labels"])

    all_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    method_score_rows: list[dict[str, Any]] = []

    all_rows.extend(protocol_a_rows(android["participant_ids"], android["folds"], android["labels"], android["frozen_probabilities"]))
    b_rows, b_params = protocol_b_threshold_adaptation(android["participant_ids"], android["folds"], android["labels"], android["frozen_probabilities"])
    all_rows.extend(b_rows)
    parameter_rows.extend(b_params)
    c_rows, c_params, c_scores = protocol_c_calibration(android["participant_ids"], android["folds"], android["labels"], android["frozen_probabilities"])
    all_rows.extend(c_rows)
    parameter_rows.extend(c_params)
    method_score_rows.extend(c_scores)
    d_rows, d_params, d_inner = protocol_d_android_logistic_regression(android["participant_ids"], android["folds"], android["labels"], android["embeddings"])
    all_rows.extend(d_rows)
    parameter_rows.extend(d_params)
    for row in d_inner:
        row["row_type"] = "D_inner_C_candidate"
        method_score_rows.append(row)

    protocol_metrics = aggregate_protocol_metrics(all_rows)
    fold_metric_rows = fold_metrics(all_rows)
    calibration_rows = calibration_report_rows(all_rows)
    bootstrap_rows, paired_rows = bootstrap_intervals(all_rows, int(config["bootstrap_replicates"]), int(config["random_seed"]))

    joblib = __import__("joblib")
    pipeline = joblib.load(PROJECT_ROOT / config["phase1c_pipeline"])
    daic_ids, daic_train_embeddings = load_daic_train_embeddings(config)
    daic_scores = pipeline.predict_proba(daic_train_embeddings)[:, 1]
    android_scores = android["frozen_probabilities"]
    domain_report = embedding_shift_report(
        daic_train_embeddings,
        android["embeddings"],
        daic_scores,
        android_scores,
        daic_ids,
        android["participant_ids"],
    )
    domain_summary = domain_shift_summary_rows(domain_report)

    interp = interpretation(protocol_metrics)
    interp["narrative"] = (
        "Phase 2C separates the failure modes by keeping the frozen DAIC scores fixed for threshold and calibration "
        "adaptation, then comparing them with a within-Android logistic-regression boundary on the same frozen layer-9 "
        "participant embeddings. The conclusion is descriptive research evidence only, not a clinical claim."
    )

    selected_c_distribution = Counter(str(row["selected_C"]) for row in parameter_rows if row["protocol"] == "D_android_trained_logistic_regression")
    selected_threshold_distribution = Counter(
        f"{float(row['selected_threshold']):.6f}" for row in parameter_rows if row["protocol"] in {"B_threshold_only", "D_android_trained_logistic_regression"}
    )
    calibration_method_distribution = Counter(str(row["calibration_method"]) for row in parameter_rows if row["protocol"] == "C_frozen_score_calibration")

    report = {
        "status": "completed_phase2c_domain_shift_diagnosis",
        "created_at": utc_now(),
        "run_directory": rel(run_dir),
        "planned_registry_id": config["planned_registry_id"],
        "official_inputs": {
            "phase2a_run": config["phase2a_run"],
            "phase2b_run": config["phase2b_run"],
            "superseded_phase2a_used": False,
        },
        "data_summary": {
            "participants": len(android["participant_ids"]),
            "class_counts": {"HC": int(np.sum(android["labels"] == 0)), "PT": int(np.sum(android["labels"] == 1))},
            "folds": [summary.__dict__ for summary in fold_summaries],
        },
        "protocol_metrics": protocol_metrics,
        "selected_C_distribution": dict(selected_c_distribution),
        "selected_threshold_distribution": dict(selected_threshold_distribution),
        "calibration_method_distribution": dict(calibration_method_distribution),
        "bootstrap_intervals": bootstrap_rows,
        "paired_bootstrap_comparisons": paired_rows,
        "domain_shift_summary": {
            "centroid_cosine_distance": domain_report["centroid_cosine_distance"],
            "mean_absolute_standardized_shift": domain_report["per_feature_shift"]["mean_absolute_standardized_shift"],
            "features_shifted_more_than_one_daic_std": domain_report["per_feature_shift"]["features_shifted_more_than_one_daic_std"],
            "mmd2": domain_report["maximum_mean_discrepancy"]["mmd2"],
        },
        "interpretation": interp,
        "leakage_checks": {
            "status": "passed",
            "participant_grouping_unit": True,
            "each_participant_one_outer_test_fold": True,
            "outer_test_labels_used_for_threshold_selection": False,
            "outer_test_labels_used_for_calibration": False,
            "outer_test_labels_used_for_C_selection": False,
            "reading_task_fold_assignment_used": False,
            "android_data_modified_protocol_A": False,
        },
        "scope_confirmation": {
            "wavlm_rerun_performed": False,
            "audio_windows_regenerated": False,
            "reading_speech_processed": False,
            "timing_features_used": False,
            "wavlm_layer_changed": False,
            "participant_aggregation_changed": False,
            "model_binaries_saved": False,
        },
    }
    metadata = {
        "created_at": utc_now(),
        "config_path": rel(config_path),
        "config_sha256": sha256_file(config_path),
        "run_directory": rel(run_dir),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "sklearn_version": package_version("scikit-learn"),
        "joblib_version": package_version("joblib"),
        "machine_name": platform.node(),
        "git_commit": git_commit(),
        "random_seed": config["random_seed"],
        "bootstrap_replicates": config["bootstrap_replicates"],
        "modeling_scope": "Phase 2C only; no WavLM extraction, no reading speech, no fine tuning",
    }

    write_csv(run_dir / "phase2c_outer_predictions.csv", all_rows)
    write_csv(run_dir / "phase2c_protocol_comparison.csv", protocol_metrics)
    write_csv(run_dir / "phase2c_fold_metrics.csv", fold_metric_rows)
    write_csv(run_dir / "phase2c_selected_parameters.csv", parameter_rows + method_score_rows)
    write_csv(run_dir / "phase2c_bootstrap_intervals.csv", bootstrap_rows)
    write_csv(run_dir / "phase2c_paired_comparisons.csv", paired_rows)
    write_csv(run_dir / "phase2c_calibration_report.csv", calibration_rows)
    write_csv(run_dir / "daic_android_embedding_shift_summary.csv", domain_summary)
    write_json(run_dir / "daic_android_domain_shift_report.json", domain_report)
    write_json(run_dir / "PHASE2C_WITHIN_ANDROID_REPORT.json", report)
    write_json(run_dir / "phase2c_run_metadata.json", metadata)
    (run_dir / "PHASE2C_WITHIN_ANDROID_REPORT.md").write_text(markdown_report(report), encoding="utf-8")
    compact_dir = create_compact_record(run_dir, PROJECT_ROOT / config["compact_record_root"])
    print(f"run_dir={rel(run_dir)}")
    print(f"compact_dir={rel(compact_dir)}")
    print(f"dominant_failure_source={interp['dominant_failure_source']}")


if __name__ == "__main__":
    main()
