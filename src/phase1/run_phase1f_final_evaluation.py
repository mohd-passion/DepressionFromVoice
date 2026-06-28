"""Phase 1F frozen final DAIC-WOZ evaluation and closure.

This script performs no model selection, no hyperparameter tuning, and no
model refitting. It loads the two frozen train-only pipelines and evaluates
them once on the provisional local-labeled test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DEPENDENCY_DIR = Path(".phase1_deps")
if DEFAULT_DEPENDENCY_DIR.exists():
    sys.path.insert(0, str(DEFAULT_DEPENDENCY_DIR.resolve()))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from run_participant_baselines import participant_sort_key, path_relative_to_root, resolve_path, sha256_file, write_csv, write_json
from run_phase1d_timing_fusion_analysis import TIMING_FEATURES, WAVLM_DIM, clean, csv_cell, positive_probabilities


PHASE1_BASELINE_RUN_ID = "20260622_114625"
PHASE1B_RUN_ID = "20260622_123725"
PHASE1C_RUN_ID = "20260622_135800"
PHASE1D_RUN_ID = "20260622_142132"
PHASE1E_RUN_ID = "20260622_145240"
MODEL_NAME = "microsoft/wavlm-base-plus"
MODEL_COMMIT = "4c66d4806a428f2e922ccfa1a962776e232d487b"
LOCKED_LAYER = "layer_09"
LOCKED_AGGREGATION = "mean"
EXCLUDED_PARTICIPANTS = {"409", "422"}
EXPECTED_TEST_COUNT = 47
EXPECTED_TEST_CLASS_COUNTS = {0: 33, 1: 14}
WAVLM_THRESHOLD = 0.5558401346206665
TIMING_THRESHOLD = 0.2927276896337141
BOOTSTRAP_REPLICATES = 5000
RANDOM_SEED = 42
EVALUATION_STATUS = "provisional_local_labeled_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1F frozen final DAIC-WOZ evaluation.")
    parser.add_argument("--output-dir", default="outputs/phase1_daic")
    parser.add_argument("--baseline-run-dir", default=f"outputs/phase1_daic/phase1_participant_baseline_runs/{PHASE1_BASELINE_RUN_ID}")
    parser.add_argument("--phase1b-run-dir", default=f"outputs/phase1_daic/phase1b_layerwise_runs/{PHASE1B_RUN_ID}")
    parser.add_argument("--phase1c-run-dir", default=f"outputs/phase1_daic/phase1c_aggregation_runs/{PHASE1C_RUN_ID}")
    parser.add_argument("--phase1d-run-dir", default=f"outputs/phase1_daic/phase1d_timing_fusion_runs/{PHASE1D_RUN_ID}")
    parser.add_argument("--phase1e-run-dir", default=f"outputs/phase1_daic/phase1e_late_fusion_runs/{PHASE1E_RUN_ID}")
    parser.add_argument("--run-root", default="outputs/phase1_daic/phase1f_final_evaluation_runs")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dependency-dir", default=str(DEFAULT_DEPENDENCY_DIR))
    return parser.parse_args()


def add_dependency_dir(path_text: str) -> None:
    path = Path(path_text)
    if path.exists():
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def get_package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""


def get_git_commit(root: Path) -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True, timeout=10)
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_sort_int"] = result["participant_id"].astype(str).map(lambda value: participant_sort_key(value)[0])
    result["_sort_text"] = result["participant_id"].astype(str)
    return result.sort_values(["_sort_int", "_sort_text"]).drop(columns=["_sort_int", "_sort_text"]).reset_index(drop=True)


def read_full_test_labels(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            participant_id = clean(row.get("Participant_ID"))
            label_text = clean(row.get("PHQ8_Binary"))
            gender = clean(row.get("Gender"))
            if not participant_id or not label_text:
                continue
            rows.append({"participant_id": participant_id, "label_binary": int(float(label_text)), "gender": gender})
    return sort_frame(pd.DataFrame(rows))


def read_named_test_split(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [{"participant_id": clean(row.get("Participant_ID")), "gender": clean(row.get("Gender"))} for row in reader if clean(row.get("Participant_ID"))]
    frame = sort_frame(pd.DataFrame(rows))
    frame.attrs["fieldnames"] = fieldnames
    return frame


def read_manifest_test_metadata(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with (output_dir / "daic_phase1_participant_manifest_clean.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = {name: column_index for column_index, name in enumerate(header)}
        for row in reader:
            if csv_cell(row, index["official_split"]) == "test":
                rows.append(
                    {
                        "participant_id": csv_cell(row, index["participant_id"]),
                        "official_split": "test",
                        "gender": csv_cell(row, index["gender"]),
                    }
                )
    return sort_frame(pd.DataFrame(rows))


def npv_score(tn: int, fn: int) -> float:
    return float(tn / (tn + fn)) if (tn + fn) else float("nan")


def metric_dict(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "positive_precision": float(precision_score(y_true, predictions, zero_division=0)),
        "positive_recall_sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "negative_predictive_value": npv_score(int(tn), int(fn)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def bootstrap_intervals(y_true: np.ndarray, probabilities: np.ndarray, threshold: float, model_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(RANDOM_SEED)
    pos_indices = np.flatnonzero(y_true == 1)
    neg_indices = np.flatnonzero(y_true == 0)
    replicates: dict[str, list[float]] = {key: [] for key in ["roc_auc", "pr_auc", "balanced_accuracy", "macro_f1", "sensitivity", "specificity"]}
    point_metrics = metric_dict(y_true, probabilities, threshold)
    point_estimates = {
        "roc_auc": point_metrics["roc_auc"],
        "pr_auc": point_metrics["pr_auc"],
        "balanced_accuracy": point_metrics["balanced_accuracy"],
        "macro_f1": point_metrics["macro_f1"],
        "sensitivity": point_metrics["positive_recall_sensitivity"],
        "specificity": point_metrics["specificity"],
    }
    failed = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            [
                rng.choice(neg_indices, size=len(neg_indices), replace=True),
                rng.choice(pos_indices, size=len(pos_indices), replace=True),
            ]
        )
        try:
            metrics = metric_dict(y_true[sampled], probabilities[sampled], threshold)
        except ValueError:
            failed += 1
            continue
        replicates["roc_auc"].append(metrics["roc_auc"])
        replicates["pr_auc"].append(metrics["pr_auc"])
        replicates["balanced_accuracy"].append(metrics["balanced_accuracy"])
        replicates["macro_f1"].append(metrics["macro_f1"])
        replicates["sensitivity"].append(metrics["positive_recall_sensitivity"])
        replicates["specificity"].append(metrics["specificity"])
    rows: list[dict[str, Any]] = []
    for metric_name, values in replicates.items():
        array = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "model_name": model_name,
                "metric": metric_name,
                "n_bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "n_successful_replicates": int(len(array)),
                "n_failed_or_undefined_replicates": failed,
                "point_estimate": float(point_estimates[metric_name]),
                "percentile_2_5": float(np.percentile(array, 2.5)) if len(array) else "",
                "percentile_50": float(np.percentile(array, 50)) if len(array) else "",
                "percentile_97_5": float(np.percentile(array, 97.5)) if len(array) else "",
                "bootstrap_type": "stratified_participant_resampling_fixed_predictions_no_refit",
            }
        )
    return rows, {"failed_replicates": failed}


def paired_bootstrap(
    y_true: np.ndarray,
    wavlm_prob: np.ndarray,
    timing_prob: np.ndarray,
    wavlm_threshold: float,
    timing_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(RANDOM_SEED)
    pos_indices = np.flatnonzero(y_true == 1)
    neg_indices = np.flatnonzero(y_true == 0)
    rows_by_metric = {"roc_auc_difference_wavlm_minus_timing": [], "balanced_accuracy_difference_wavlm_minus_timing": [], "macro_f1_difference_wavlm_minus_timing": []}
    wavlm_point = metric_dict(y_true, wavlm_prob, wavlm_threshold)
    timing_point = metric_dict(y_true, timing_prob, timing_threshold)
    point_estimates = {
        "roc_auc_difference_wavlm_minus_timing": wavlm_point["roc_auc"] - timing_point["roc_auc"],
        "balanced_accuracy_difference_wavlm_minus_timing": wavlm_point["balanced_accuracy"] - timing_point["balanced_accuracy"],
        "macro_f1_difference_wavlm_minus_timing": wavlm_point["macro_f1"] - timing_point["macro_f1"],
    }
    failed = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            [
                rng.choice(neg_indices, size=len(neg_indices), replace=True),
                rng.choice(pos_indices, size=len(pos_indices), replace=True),
            ]
        )
        try:
            wavlm_metrics = metric_dict(y_true[sampled], wavlm_prob[sampled], wavlm_threshold)
            timing_metrics = metric_dict(y_true[sampled], timing_prob[sampled], timing_threshold)
        except ValueError:
            failed += 1
            continue
        rows_by_metric["roc_auc_difference_wavlm_minus_timing"].append(wavlm_metrics["roc_auc"] - timing_metrics["roc_auc"])
        rows_by_metric["balanced_accuracy_difference_wavlm_minus_timing"].append(wavlm_metrics["balanced_accuracy"] - timing_metrics["balanced_accuracy"])
        rows_by_metric["macro_f1_difference_wavlm_minus_timing"].append(wavlm_metrics["macro_f1"] - timing_metrics["macro_f1"])
    rows: list[dict[str, Any]] = []
    for metric_name, values in rows_by_metric.items():
        array = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "comparison": "candidate_A_wavlm_minus_candidate_B_timing",
                "metric": metric_name,
                "n_bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "n_successful_replicates": int(len(array)),
                "n_failed_or_undefined_replicates": failed,
                "point_estimate": float(point_estimates[metric_name]),
                "percentile_2_5": float(np.percentile(array, 2.5)) if len(array) else "",
                "percentile_50": float(np.percentile(array, 50)) if len(array) else "",
                "percentile_97_5": float(np.percentile(array, 97.5)) if len(array) else "",
            }
        )
    return rows, {"failed_replicates": failed}


def hash_model_pipeline(path: Path) -> str:
    return sha256_file(path)


def verify_pipeline(name: str, pipeline: Any, expected_c: float) -> dict[str, Any]:
    classifier = pipeline.named_steps.get("classifier")
    scaler_present = "scaler" in pipeline.named_steps or "preprocessor" in pipeline.named_steps
    c_value = float(classifier.C) if classifier is not None and hasattr(classifier, "C") else float("nan")
    return {
        "model_name": name,
        "classifier_class": classifier.__class__.__name__ if classifier is not None else "",
        "C": c_value,
        "expected_C": expected_c,
        "C_matches_expected": bool(abs(c_value - expected_c) < 1e-12),
        "class_weight": classifier.class_weight if classifier is not None else "",
        "max_iter": int(classifier.max_iter) if classifier is not None and hasattr(classifier, "max_iter") else "",
        "scaler_or_preprocessor_present": scaler_present,
    }


def load_frozen_inputs(root: Path, output_dir: Path, phase1c_run_dir: Path, phase1d_run_dir: Path, baseline_run_dir: Path, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    provenance_path = baseline_run_dir / "test_label_provenance_report.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    full_test_path = resolve_path("Data/full_test_split.csv", root)
    named_test_path = resolve_path("Data/test_split_Depression_AVEC2017.csv", root)
    labels = read_full_test_labels(full_test_path)
    named = read_named_test_split(named_test_path)
    manifest_test = read_manifest_test_metadata(output_dir)
    phase1c_manifest = pd.read_csv(phase1c_run_dir / "aggregation_manifest.csv", dtype={"participant_id": str})
    wavlm_manifest = phase1c_manifest[
        phase1c_manifest["aggregation_method"].eq(LOCKED_AGGREGATION)
        & phase1c_manifest["aggregation_status"].eq("success")
        & phase1c_manifest["official_split"].eq("test")
    ].copy()
    timing = pd.read_csv(phase1d_run_dir / "speech_timing_features.csv", dtype={"participant_id": str})
    timing_test = timing[timing["official_split"].eq("test")].copy()

    issues: list[dict[str, Any]] = []
    if set(named.attrs.get("fieldnames", [])) != {"Participant_ID", "Gender"}:
        issues.append({"severity": "error", "issue_type": "avec_named_test_split_columns", "details": f"Unexpected columns: {named.attrs.get('fieldnames', [])}"})
    if len(labels) != EXPECTED_TEST_COUNT:
        issues.append({"severity": "error", "issue_type": "test_label_count_mismatch", "details": f"Expected {EXPECTED_TEST_COUNT}, found {len(labels)}."})
    class_counts = labels["label_binary"].value_counts().to_dict()
    if class_counts != EXPECTED_TEST_CLASS_COUNTS:
        issues.append({"severity": "error", "issue_type": "test_class_counts_mismatch", "details": f"Expected {EXPECTED_TEST_CLASS_COUNTS}, found {class_counts}."})
    for name, frame in [("labels", labels), ("named_test_split", named), ("manifest_test", manifest_test), ("wavlm_manifest", wavlm_manifest), ("timing_test", timing_test)]:
        if frame["participant_id"].duplicated().any():
            issues.append({"severity": "error", "issue_type": f"duplicate_{name}_participants", "details": f"Duplicate participant IDs in {name}."})
        excluded = sorted(set(frame["participant_id"]) & EXCLUDED_PARTICIPANTS, key=participant_sort_key)
        if excluded:
            issues.append({"severity": "error", "issue_type": f"excluded_{name}_participants", "details": f"Excluded participants present in {name}: {excluded}."})
    id_sets = [set(frame["participant_id"]) for frame in [labels, named, manifest_test, wavlm_manifest, timing_test]]
    if any(id_set != id_sets[0] for id_set in id_sets[1:]):
        issues.append({"severity": "error", "issue_type": "test_participant_alignment_mismatch", "details": "Test participant IDs do not align across labels, named split, manifest, WavLM, and timing tables."})
    if any(frame["participant_id"].isin(EXCLUDED_PARTICIPANTS).any() for frame in [labels, named, manifest_test, wavlm_manifest, timing_test]):
        issues.append({"severity": "error", "issue_type": "excluded_participant_present", "details": "409 or 422 present in test inputs."})

    merged = sort_frame(
        labels.merge(named[["participant_id", "gender"]], on="participant_id", how="inner", suffixes=("_full", "_named"), validate="one_to_one")
        .merge(manifest_test[["participant_id", "official_split"]], on="participant_id", how="inner", validate="one_to_one")
        .merge(wavlm_manifest[["participant_id", "representation_path"]], on="participant_id", how="inner", validate="one_to_one")
        .merge(timing_test[["participant_id", *TIMING_FEATURES]], on="participant_id", how="inner", validate="one_to_one")
    )
    merged["gender"] = merged["gender_named"].where(merged["gender_named"].astype(str).str.len() > 0, merged["gender_full"])

    wavlm_vectors: list[np.ndarray] = []
    missing_wavlm: list[str] = []
    for row in merged.itertuples(index=False):
        path = resolve_path(row.representation_path, root)
        if not path.exists():
            missing_wavlm.append(row.participant_id)
            continue
        vector = np.load(path).astype(np.float32, copy=False)
        if vector.shape != (WAVLM_DIM,) or not np.isfinite(vector).all() or np.all(vector == 0):
            missing_wavlm.append(row.participant_id)
            continue
        wavlm_vectors.append(vector)
    if missing_wavlm:
        issues.append({"severity": "error", "issue_type": "missing_or_invalid_wavlm_embeddings", "details": ";".join(missing_wavlm)})
    timing_numeric = merged[TIMING_FEATURES].apply(pd.to_numeric, errors="coerce")
    if np.isinf(timing_numeric.to_numpy(dtype=np.float64)).any():
        issues.append({"severity": "error", "issue_type": "infinite_timing_features", "details": "Timing feature table contains infinite values."})
    if list(timing_numeric.columns) != TIMING_FEATURES:
        issues.append({"severity": "error", "issue_type": "timing_feature_order_mismatch", "details": "Timing feature order mismatch."})

    validation_rows = issues or [{"severity": "info", "issue_type": "all_phase1f_input_checks_passed", "details": "All Phase 1F reproducibility checks passed before prediction."}]
    write_csv(run_dir / "phase1f_reproducibility_verification.csv", validation_rows)
    if issues:
        raise RuntimeError("Phase 1F reproducibility verification failed.")

    metadata = {
        "provenance_report_path": path_relative_to_root(provenance_path, root),
        "provenance": provenance,
        "test_label_source": "Data/full_test_split.csv",
        "named_avec_test_split": "Data/test_split_Depression_AVEC2017.csv",
        "named_avec_test_split_contains_phq_labels": False,
        "evaluation_status": EVALUATION_STATUS,
        "test_counts": {"participants": int(len(merged)), "non_depressed": int((merged["label_binary"] == 0).sum()), "depressed": int((merged["label_binary"] == 1).sum())},
        "local_test_set_used_for_phase1b_to_phase1e_selection": False,
        "earlier_phase1a_final_layer_baseline_test_result_previously_inspected": True,
        "untouched_research_test_set": False,
        "provisional_external_replication_required": True,
    }
    return merged[["participant_id", "label_binary", "gender", "official_split"]], pd.DataFrame(np.vstack(wavlm_vectors), columns=[f"wavlm_{i:03d}" for i in range(WAVLM_DIM)]), timing_numeric.reset_index(drop=True), metadata


def write_predictions(path: Path, test_meta: pd.DataFrame, probabilities: np.ndarray, threshold: float, model_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for participant, probability in zip(test_meta.itertuples(index=False), probabilities):
        rows.append(
            {
                "participant_id": participant.participant_id,
                "label_binary": int(participant.label_binary),
                "predicted_probability": float(probability),
                "prediction_at_0_5": int(probability >= 0.5),
                "frozen_dev_threshold": threshold,
                "prediction_at_frozen_dev_threshold": int(probability >= threshold),
                "model_name": model_name,
                "evaluation_status": EVALUATION_STATUS,
            }
        )
    write_csv(path, rows)
    return pd.DataFrame(rows)


def metrics_rows(model_name: str, y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold_type, current_threshold in [("default_0_5", 0.5), ("frozen_dev_threshold", threshold)]:
        metrics = metric_dict(y_true, probabilities, current_threshold)
        row = {
            "model_name": model_name,
            "threshold_type": threshold_type,
            "threshold": current_threshold,
            "evaluation_status": EVALUATION_STATUS,
            **metrics,
        }
        rows.append(row)
    return rows


def subgroup_rows(test_meta: pd.DataFrame, predictions: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    meta = test_meta[["participant_id", "label_binary", "gender"]].copy()
    for model_name, pred in predictions.items():
        frame = meta.merge(pred[["participant_id", "predicted_probability", "frozen_dev_threshold"]], on="participant_id", how="inner", validate="one_to_one")
        for gender, group in frame.groupby("gender"):
            label_counts = group["label_binary"].value_counts().to_dict()
            row = {
                "model_name": model_name,
                "subgroup": "gender",
                "subgroup_value": gender,
                "participant_count": int(len(group)),
                "non_depressed_count": int(label_counts.get(0, 0)),
                "depressed_count": int(label_counts.get(1, 0)),
                "threshold": float(group["frozen_dev_threshold"].iloc[0]),
                "exploratory": True,
            }
            if len(label_counts) < 2:
                row.update({"sensitivity": "", "specificity": "", "balanced_accuracy": "", "macro_f1": "", "metric_status": "not_reportable_single_class_subgroup"})
            else:
                metrics = metric_dict(group["label_binary"].to_numpy(dtype=int), group["predicted_probability"].to_numpy(dtype=float), float(group["frozen_dev_threshold"].iloc[0]))
                row.update(
                    {
                        "sensitivity": metrics["positive_recall_sensitivity"],
                        "specificity": metrics["specificity"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "metric_status": "reported",
                    }
                )
            rows.append(row)
    return rows


def paired_comparison(test_meta: pd.DataFrame, wavlm_pred: pd.DataFrame, timing_pred: pd.DataFrame, y_true: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    wavlm = wavlm_pred.sort_values("participant_id", key=lambda series: series.map(lambda value: participant_sort_key(str(value))[0])).reset_index(drop=True)
    timing = timing_pred.sort_values("participant_id", key=lambda series: series.map(lambda value: participant_sort_key(str(value))[0])).reset_index(drop=True)
    wavlm_prob = wavlm["predicted_probability"].to_numpy(dtype=float)
    timing_prob = timing["predicted_probability"].to_numpy(dtype=float)
    wavlm_frozen = wavlm["prediction_at_frozen_dev_threshold"].to_numpy(dtype=int)
    timing_frozen = timing["prediction_at_frozen_dev_threshold"].to_numpy(dtype=int)
    correct_wavlm = wavlm_frozen == y_true
    correct_timing = timing_frozen == y_true
    both_correct = test_meta.loc[correct_wavlm & correct_timing, "participant_id"].astype(str).tolist()
    both_incorrect = test_meta.loc[(~correct_wavlm) & (~correct_timing), "participant_id"].astype(str).tolist()
    wavlm_only = test_meta.loc[correct_wavlm & (~correct_timing), "participant_id"].astype(str).tolist()
    timing_only = test_meta.loc[(~correct_wavlm) & correct_timing, "participant_id"].astype(str).tolist()
    wavlm_metrics = metric_dict(y_true, wavlm_prob, WAVLM_THRESHOLD)
    timing_metrics = metric_dict(y_true, timing_prob, TIMING_THRESHOLD)
    point = {
        "roc_auc_difference_wavlm_minus_timing": wavlm_metrics["roc_auc"] - timing_metrics["roc_auc"],
        "pr_auc_difference_wavlm_minus_timing": wavlm_metrics["pr_auc"] - timing_metrics["pr_auc"],
        "balanced_accuracy_difference_wavlm_minus_timing": wavlm_metrics["balanced_accuracy"] - timing_metrics["balanced_accuracy"],
        "macro_f1_difference_wavlm_minus_timing": wavlm_metrics["macro_f1"] - timing_metrics["macro_f1"],
        "probability_pearson_correlation": float(pd.Series(wavlm_prob).corr(pd.Series(timing_prob), method="pearson")),
        "probability_spearman_correlation": float(pd.Series(wavlm_prob).corr(pd.Series(timing_prob), method="spearman")),
        "both_correct_count": len(both_correct),
        "both_incorrect_count": len(both_incorrect),
        "wavlm_only_correct_count": len(wavlm_only),
        "timing_only_correct_count": len(timing_only),
        "both_correct_participant_ids": ";".join(both_correct),
        "both_incorrect_participant_ids": ";".join(both_incorrect),
        "wavlm_only_correct_participant_ids": ";".join(wavlm_only),
        "timing_only_correct_participant_ids": ";".join(timing_only),
        "comparison_status": "descriptive_only_no_model_selection_or_modification",
    }
    bootstrap_rows, bootstrap_meta = paired_bootstrap(y_true, wavlm_prob, timing_prob, WAVLM_THRESHOLD, TIMING_THRESHOLD)
    rows = [{"comparison": "candidate_A_wavlm_vs_candidate_B_timing", "metric": key, "value": value} for key, value in point.items()]
    rows.extend(bootstrap_rows)
    return rows, {**point, "paired_bootstrap": bootstrap_meta}


def final_comparison_rows(
    baseline_run_dir: Path,
    phase1c_run_dir: Path,
    phase1d_run_dir: Path,
    phase1e_run_dir: Path,
    final_metrics: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline = pd.read_csv(baseline_run_dir / "baseline_comparison.csv")
    aggregation = pd.read_csv(phase1c_run_dir / "aggregation_comparison.csv")
    timing = pd.read_csv(phase1d_run_dir / "timing_fusion_comparison.csv")
    late = pd.read_csv(phase1e_run_dir / "late_fusion_comparison.csv")

    def add_row(name: str, status: str, protocol: str, train_auc: Any = "", train_auc_std: Any = "", dev_auc: Any = "", dev_pr: Any = "", test_auc: Any = "", test_pr: Any = "", test_bal: Any = "", test_f1: Any = "", notes: str = "") -> None:
        rows.append(
            {
                "model_or_stage": name,
                "status": status,
                "evaluation_protocol_note": protocol,
                "train_nested_or_cv_roc_auc": train_auc,
                "train_roc_auc_std": train_auc_std,
                "dev_roc_auc": dev_auc,
                "dev_pr_auc": dev_pr,
                "provisional_local_test_roc_auc": test_auc,
                "provisional_local_test_pr_auc": test_pr,
                "provisional_local_test_balanced_accuracy_frozen_threshold": test_bal,
                "provisional_local_test_macro_f1_frozen_threshold": test_f1,
                "notes": notes,
            }
        )

    metadata_train = baseline[(baseline["model_name"].eq("metadata_logistic_regression")) & (baseline["stage"].eq("train_repeated_stratified_cv"))].iloc[0]
    metadata_dev = baseline[(baseline["model_name"].eq("metadata_logistic_regression")) & (baseline["stage"].eq("dev")) & (baseline["threshold_type"].eq("default_0_5"))].iloc[0]
    metadata_test = baseline[(baseline["model_name"].eq("metadata_logistic_regression")) & (baseline["stage"].eq("test")) & (baseline["threshold_type"].eq("frozen_dev_tuned"))].iloc[0]
    add_row("metadata baseline", "not_final_candidate", "Phase 1A repeated train CV plus dev/test from earlier baseline", metadata_train["roc_auc"], metadata_train["roc_auc_std"], metadata_dev["roc_auc"], metadata_dev["pr_auc"], metadata_test["roc_auc"], metadata_test["pr_auc"], metadata_test["balanced_accuracy"], metadata_test["macro_f1"], "Earlier local test result was inspected before later phases.")

    final_layer_train = baseline[(baseline["model_name"].eq("wavlm_participant_logistic_regression")) & (baseline["stage"].eq("train_repeated_stratified_cv"))].iloc[0]
    final_layer_dev = baseline[(baseline["model_name"].eq("wavlm_participant_logistic_regression")) & (baseline["stage"].eq("dev")) & (baseline["threshold_type"].eq("default_0_5"))].iloc[0]
    final_layer_test = baseline[(baseline["model_name"].eq("wavlm_participant_logistic_regression")) & (baseline["stage"].eq("test")) & (baseline["threshold_type"].eq("frozen_dev_tuned"))].iloc[0]
    add_row("final-layer WavLM baseline", "not_final_candidate", "Phase 1A final-layer baseline protocol", final_layer_train["roc_auc"], final_layer_train["roc_auc_std"], final_layer_dev["roc_auc"], final_layer_dev["pr_auc"], final_layer_test["roc_auc"], final_layer_test["pr_auc"], final_layer_test["balanced_accuracy"], final_layer_test["macro_f1"], "Earlier local test result was inspected before later phases.")

    phase1c_mean = aggregation[aggregation["row_name"].eq("phase1c_mean")].iloc[0]
    wavlm_final = final_metrics[(final_metrics["model_name"].eq("candidate_A_wavlm_free_speech")) & (final_metrics["threshold_type"].eq("frozen_dev_threshold"))].iloc[0]
    add_row("selected layer-9 WavLM baseline", "final_frozen_candidate_A", "Phase 1C train nested CV/dev; Phase 1F provisional local test", phase1c_mean["train_nested_cv_mean_roc_auc"], phase1c_mean["train_nested_cv_std_roc_auc"], phase1c_mean["dev_roc_auc"], phase1c_mean["dev_pr_auc"], wavlm_final["roc_auc"], wavlm_final["pr_auc"], wavlm_final["balanced_accuracy"], wavlm_final["macro_f1"], "Free-speech-compatible research candidate.")

    for row_name, label in [("phase1c_median", "median aggregation"), ("phase1c_mean_std_pca", "mean-plus-standard-deviation aggregation")]:
        row = aggregation[aggregation["row_name"].eq(row_name)].iloc[0]
        add_row(label, "not_advanced", "Phase 1C train nested CV only", row["train_nested_cv_mean_roc_auc"], row["train_nested_cv_std_roc_auc"], "", "", "", "", "", "", "Not evaluated on dev/test as selected candidate.")

    phase1d_timing = timing[timing["model_name"].eq("timing_only")].iloc[0]
    timing_final = final_metrics[(final_metrics["model_name"].eq("candidate_B_timing_guided_interview")) & (final_metrics["threshold_type"].eq("frozen_dev_threshold"))].iloc[0]
    add_row("timing-only", "final_frozen_candidate_B", "Phase 1D/1E train nested CV/dev; Phase 1F provisional local test", phase1d_timing["train_nested_cv_mean_roc_auc"], phase1d_timing["train_nested_cv_std_roc_auc"], phase1d_timing["dev_roc_auc"], phase1d_timing["dev_pr_auc"], timing_final["roc_auc"], timing_final["pr_auc"], timing_final["balanced_accuracy"], timing_final["macro_f1"], "Guided-interview-dependent research candidate.")

    early = timing[timing["model_name"].eq("early_fusion")].iloc[0]
    add_row("early feature fusion", "not_advanced", "Phase 1D train nested CV only", early["train_nested_cv_mean_roc_auc"], early["train_nested_cv_std_roc_auc"], "", "", "", "", "", "", "Underperformed; not repeated in Phase 1E.")

    for row_name, label in [("phase1e_weighted_late_fusion", "weighted late fusion"), ("phase1e_logistic_stacking", "logistic stacking")]:
        row = late[late["row_name"].eq(row_name)].iloc[0]
        add_row(label, "not_advanced", "Phase 1E train nested CV only", row["train_nested_cv_mean_roc_auc"], row["train_nested_cv_std_roc_auc"], "", "", "", "", "", "", "No post-test fusion performed.")

    add_row("final frozen Candidate A", "final_frozen_candidate_A", "Same as selected layer-9 WavLM baseline; Phase 1F final provisional local test", phase1c_mean["train_nested_cv_mean_roc_auc"], phase1c_mean["train_nested_cv_std_roc_auc"], phase1c_mean["dev_roc_auc"], phase1c_mean["dev_pr_auc"], wavlm_final["roc_auc"], wavlm_final["pr_auc"], wavlm_final["balanced_accuracy"], wavlm_final["macro_f1"], "Not deployment-ready; external validation required.")
    add_row("final frozen Candidate B", "final_frozen_candidate_B", "Same as timing-only candidate; Phase 1F final provisional local test", phase1d_timing["train_nested_cv_mean_roc_auc"], phase1d_timing["train_nested_cv_std_roc_auc"], phase1d_timing["dev_roc_auc"], phase1d_timing["dev_pr_auc"], timing_final["roc_auc"], timing_final["pr_auc"], timing_final["balanced_accuracy"], timing_final["macro_f1"], "Not deployment-ready; external validation required.")
    return rows


def write_model_card(path: Path, title: str, candidate: str, intended_use: str, prohibited_use: str, input_requirements: str, features: str, classifier: str, threshold: float, dev_metrics: dict[str, Any], test_metrics: dict[str, Any], deployment_note: str) -> None:
    content = f"""# {title}

## Intended Use
{intended_use}

## Prohibited Use
{prohibited_use}

## Training Dataset
DAIC-WOZ Phase 1 clean split, train participants only for fitted preprocessing and classifier training.

## Participant Counts
Train: 106; dev: 34; provisional local-labeled test: 47. Depressed participants in provisional local test: 14.

## Label Definition
Binary PHQ-8 depression label as available in local DAIC-WOZ metadata. Test labels are from `Data/full_test_split.csv` and are treated as provisional local labels.

## Excluded Participants
Participants 409 and 422 are excluded from the clean Phase 1 manifests.

## Input Requirements
{input_requirements}

## Preprocessing
The fitted preprocessing object is frozen and loaded from the saved train-only pipeline. No test participant was used for fitting preprocessing.

## Features
{features}

## Fitted Classifier
{classifier}

## Frozen Threshold
{threshold:.12f}

## Evaluation Protocol
Train/dev model development occurred before Phase 1F. Phase 1F performs one provisional local-labeled test evaluation with no refitting, retuning, model selection, or threshold changes.

## Dev Results
ROC-AUC: {dev_metrics['roc_auc']:.6f}; PR-AUC: {dev_metrics['pr_auc']:.6f}; balanced accuracy at frozen threshold: {dev_metrics['balanced_accuracy']:.6f}; macro F1 at frozen threshold: {dev_metrics['macro_f1']:.6f}.

## Provisional Local-Test Results
ROC-AUC: {test_metrics['roc_auc']:.6f}; PR-AUC: {test_metrics['pr_auc']:.6f}; balanced accuracy at frozen threshold: {test_metrics['balanced_accuracy']:.6f}; macro F1 at frozen threshold: {test_metrics['macro_f1']:.6f}.

## Known Limitations
Small provisional local test sample; labels are not established here as an official AVEC challenge test release; prior Phase 1A local test metrics were already known; results are exploratory.

## Demographic And Language Limitations
DAIC-WOZ is limited in size and population coverage. Performance may vary by gender, accent, language, recording condition, and interview protocol.

## Data Leakage Safeguards
Phase 1B-1E model selection did not use the local test split. Phase 1F did not modify any model after viewing test results.

## Privacy Considerations
Inputs may contain sensitive health-related speech or interview timing information. Store and process only under appropriate consent, access control, and data minimization practices.

## Clinical-Safety Warning
This model is not clinically validated, not a diagnostic device, and must not be used to diagnose, triage, or deny care.

## External-Validation Requirement
External validation on an independent cohort is required before any deployment-oriented claim.

## Deployment Note
{deployment_note}
"""
    path.write_text(content, encoding="utf-8")


def write_final_reports(
    root: Path,
    output_dir: Path,
    run_dir: Path,
    dirs: dict[str, Path],
    provenance_metadata: dict[str, Any],
    verification: dict[str, Any],
    final_metrics: pd.DataFrame,
    bootstrap_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    paired_meta: dict[str, Any],
    subgroup_rows_data: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    pipeline_hashes: dict[str, str],
) -> dict[str, Any]:
    phase1c_eval = json.loads((dirs["phase1c"] / "evaluation_report.json").read_text(encoding="utf-8"))
    phase1e_eval = json.loads((dirs["phase1e"] / "evaluation_report.json").read_text(encoding="utf-8"))
    wavlm_dev = phase1c_eval["selected_dev_evaluation"]["dev_metrics_at_selected_threshold"]
    timing_dev = phase1e_eval["selected_dev_evaluation"]["dev_metrics_at_selected_threshold"]
    wavlm_test = final_metrics[(final_metrics["model_name"].eq("candidate_A_wavlm_free_speech")) & (final_metrics["threshold_type"].eq("frozen_dev_threshold"))].iloc[0].to_dict()
    timing_test = final_metrics[(final_metrics["model_name"].eq("candidate_B_timing_guided_interview")) & (final_metrics["threshold_type"].eq("frozen_dev_threshold"))].iloc[0].to_dict()

    write_model_card(
        run_dir / "MODEL_CARD_WAVLM_FREE_SPEECH.md",
        "Model Card: WavLM Free-Speech Research Candidate",
        "candidate_A",
        "Research screening experiments using participant speech represented as frozen WavLM layer-9 mean embeddings.",
        "Clinical diagnosis, automated care decisions, employment/insurance decisions, or use as a validated mental-health device.",
        "Participant-only speech processed into 10-second windows using the Phase 1 preprocessing design.",
        "768-dimensional participant embedding from WavLM layer_09, mean pooled over time and mean aggregated across windows.",
        "StandardScaler + LogisticRegression(class_weight='balanced', C=0.001, max_iter>=10000).",
        WAVLM_THRESHOLD,
        wavlm_dev,
        wavlm_test,
        "Free-speech compatible in the narrow sense that interviewer timing features are not required; still research-only and dependent on the Phase 1 windowing pipeline.",
    )
    write_model_card(
        run_dir / "MODEL_CARD_TIMING_GUIDED_INTERVIEW.md",
        "Model Card: Timing Guided-Interview Research Candidate",
        "candidate_B",
        "Research screening experiments using standardized interview timing features.",
        "Arbitrary isolated voice-clip screening, clinical diagnosis, automated care decisions, or any deployment without a guided interview protocol.",
        "DAIC-style transcript timestamps and speaker labels from a guided interview with prompts/interviewer turns.",
        "Validated 28 speech-timing features from Phase 1D/1E; no lexical or transcript-text features.",
        "SimpleImputer + StandardScaler + LogisticRegression(class_weight='balanced', C=100.0, max_iter>=10000).",
        TIMING_THRESHOLD,
        timing_dev,
        timing_test,
        "Guided-interview dependent; cannot be applied correctly to arbitrary isolated voice recordings.",
    )

    closure_classification = "completed_baseline_ready_for_external_validation"
    warnings = []
    if verification.get("quality_control_status") != "passed":
        closure_classification = "failed_quality_control"
    if pipeline_hashes.get("git_commit", "") == "":
        warnings.append("Git commit unavailable because git executable was not available or repository metadata could not be read.")

    final_report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase1_status": closure_classification,
        "quality_control_status": verification.get("quality_control_status", "passed"),
        "evaluation_status": EVALUATION_STATUS,
        "test_provenance": provenance_metadata,
        "frozen_candidates": {
            "candidate_A_wavlm_free_speech": {
                "model": MODEL_NAME,
                "model_commit": MODEL_COMMIT,
                "layer": LOCKED_LAYER,
                "aggregation": LOCKED_AGGREGATION,
                "classifier": "StandardScaler + LogisticRegression(C=0.001, class_weight=balanced)",
                "frozen_dev_threshold": WAVLM_THRESHOLD,
                "provisional_local_test_metrics": wavlm_test,
            },
            "candidate_B_timing_guided_interview": {
                "features": "Phase 1D validated 28 speech-timing features",
                "classifier": "SimpleImputer + StandardScaler + LogisticRegression(C=100.0, class_weight=balanced)",
                "frozen_dev_threshold": TIMING_THRESHOLD,
                "provisional_local_test_metrics": timing_test,
            },
        },
        "bootstrap_intervals": bootstrap_rows,
        "paired_model_comparison": paired_meta,
        "subgroup_audit": subgroup_rows_data,
        "consolidated_comparison": comparison_rows,
        "deployment_interpretation": {
            "free_speech_candidate": "Candidate A accepts participant speech without interviewer timing but remains a research screening model, not clinically validated and not diagnostic.",
            "guided_interview_candidate": "Candidate B requires standardized prompts/interviewer interaction and cannot be applied to arbitrary isolated voice recordings.",
        },
        "no_post_test_model_modification": True,
        "external_validation_required": True,
        "warnings": warnings,
    }
    write_json(run_dir / "PHASE1_FINAL_REPORT.json", final_report)

    md_lines = [
        "# Phase 1 Final Report",
        "",
        f"Run directory: `{path_relative_to_root(run_dir, root)}`",
        "",
        "## Closure Decision",
        f"Phase 1 status: `{closure_classification}`. This means the baselines are frozen and documented for external validation, not deployment-ready or clinically validated.",
        "",
        "## Test Provenance",
        "`Data/test_split_Depression_AVEC2017.csv` contains participant IDs and gender only. Local test labels came from `Data/full_test_split.csv`, so Phase 1F is labeled `provisional_local_labeled_test`, not an official AVEC challenge result. The earlier Phase 1A final-layer baseline test result was previously inspected; Phase 1B-1E did not computationally use the local test split for model selection, but this is not a completely untouched research test set.",
        "",
        "## Data Preparation And Exclusions",
        "Phase 1 used clean DAIC-WOZ manifests with participant-only speech reconstruction and excluded participants 409 and 422. Quality checks for embeddings, timing features, and participant alignment passed before Phase 1F prediction.",
        "",
        "## WavLM Extraction And Representation Findings",
        "Frozen WavLM Base Plus embeddings used model commit `4c66d4806a428f2e922ccfa1a962776e232d487b`. Layer-wise analysis selected transformer `layer_09` using train-only nested CV; aggregation analysis retained arithmetic mean across windows.",
        "",
        "## Timing And Fusion Findings",
        "Speech-timing features were compact and guided-interview dependent. Early feature fusion and Phase 1E late-fusion/stacking did not outperform the stronger individual candidate under the train-only selection rules.",
        "",
        "## Final Provisional Local-Test Results",
        f"Candidate A WavLM at frozen dev threshold: ROC-AUC `{wavlm_test['roc_auc']:.6f}`, PR-AUC `{wavlm_test['pr_auc']:.6f}`, balanced accuracy `{wavlm_test['balanced_accuracy']:.6f}`, macro F1 `{wavlm_test['macro_f1']:.6f}`.",
        f"Candidate B timing at frozen dev threshold: ROC-AUC `{timing_test['roc_auc']:.6f}`, PR-AUC `{timing_test['pr_auc']:.6f}`, balanced accuracy `{timing_test['balanced_accuracy']:.6f}`, macro F1 `{timing_test['macro_f1']:.6f}`.",
        "",
        "## Paired Comparison",
        f"WavLM minus timing ROC-AUC difference: `{paired_meta['roc_auc_difference_wavlm_minus_timing']:.6f}`. PR-AUC difference: `{paired_meta['pr_auc_difference_wavlm_minus_timing']:.6f}`. Balanced-accuracy difference at frozen thresholds: `{paired_meta['balanced_accuracy_difference_wavlm_minus_timing']:.6f}`. Probability Pearson correlation: `{paired_meta['probability_pearson_correlation']:.6f}`.",
        "",
        "## Subgroup Audit",
        "Gender subgroup metrics are exploratory, use frozen thresholds only, and suppress participant identities in this narrative. Small subgroup sizes mean these findings are not reliable fairness or clinical-validity evidence.",
        "",
        "## Application Implications",
        "Candidate A is the free-speech-compatible research candidate, but it still depends on Phase 1 preprocessing and 10-second participant speech windows. Candidate B requires a guided interview or standardized prompts; it cannot be applied correctly to arbitrary isolated voice recordings. Neither model is diagnostic or clinically validated.",
        "",
        "## External Validation Requirement",
        "External validation on an independent cohort is required before any deployment-oriented claim. No model, threshold, feature, or participant exclusion was changed after Phase 1F test results.",
    ]
    (run_dir / "PHASE1_FINAL_REPORT.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return final_report


def main() -> None:
    args = parse_args()
    add_dependency_dir(args.dependency_dir)
    root = Path.cwd()
    output_dir = resolve_path(args.output_dir, root)
    dirs = {
        "baseline": resolve_path(args.baseline_run_dir, root),
        "phase1b": resolve_path(args.phase1b_run_dir, root),
        "phase1c": resolve_path(args.phase1c_run_dir, root),
        "phase1d": resolve_path(args.phase1d_run_dir, root),
        "phase1e": resolve_path(args.phase1e_run_dir, root),
    }
    timestamp = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve_path(args.run_root, root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    test_meta, wavlm_test, timing_test, provenance_metadata = load_frozen_inputs(root, output_dir, dirs["phase1c"], dirs["phase1d"], dirs["baseline"], run_dir)
    y_test = test_meta["label_binary"].to_numpy(dtype=int)

    wavlm_model_path = dirs["phase1c"] / "models" / "selected_aggregation_pipeline.joblib"
    timing_model_path = dirs["phase1e"] / "models" / "timing_base_pipeline.joblib"
    wavlm_model = joblib.load(wavlm_model_path)
    timing_model = joblib.load(timing_model_path)
    pipeline_checks = [
        verify_pipeline("candidate_A_wavlm_free_speech", wavlm_model, 0.001),
        verify_pipeline("candidate_B_timing_guided_interview", timing_model, 100.0),
    ]
    if not all(row["C_matches_expected"] and row["max_iter"] >= 10000 for row in pipeline_checks):
        write_csv(run_dir / "phase1f_pipeline_verification.csv", pipeline_checks)
        raise RuntimeError("Frozen pipeline verification failed.")
    write_csv(run_dir / "phase1f_pipeline_verification.csv", pipeline_checks)

    wavlm_prob = positive_probabilities(wavlm_model, wavlm_test)
    timing_prob = positive_probabilities(timing_model, timing_test)
    wavlm_predictions = write_predictions(run_dir / "wavlm_final_test_predictions.csv", test_meta, wavlm_prob, WAVLM_THRESHOLD, "candidate_A_wavlm_free_speech")
    timing_predictions = write_predictions(run_dir / "timing_final_test_predictions.csv", test_meta, timing_prob, TIMING_THRESHOLD, "candidate_B_timing_guided_interview")

    metric_rows = metrics_rows("candidate_A_wavlm_free_speech", y_test, wavlm_prob, WAVLM_THRESHOLD)
    metric_rows.extend(metrics_rows("candidate_B_timing_guided_interview", y_test, timing_prob, TIMING_THRESHOLD))
    write_csv(run_dir / "phase1_final_test_metrics.csv", metric_rows)
    final_metrics = pd.DataFrame(metric_rows)

    bootstrap_rows: list[dict[str, Any]] = []
    wavlm_bootstrap, wavlm_bootstrap_meta = bootstrap_intervals(y_test, wavlm_prob, WAVLM_THRESHOLD, "candidate_A_wavlm_free_speech")
    timing_bootstrap, timing_bootstrap_meta = bootstrap_intervals(y_test, timing_prob, TIMING_THRESHOLD, "candidate_B_timing_guided_interview")
    bootstrap_rows.extend(wavlm_bootstrap)
    bootstrap_rows.extend(timing_bootstrap)
    write_csv(run_dir / "phase1_bootstrap_intervals.csv", bootstrap_rows)

    paired_rows, paired_meta = paired_comparison(test_meta, wavlm_predictions, timing_predictions, y_test)
    write_csv(run_dir / "phase1_paired_model_comparison.csv", paired_rows)
    subgroup_data = subgroup_rows(test_meta, {"candidate_A_wavlm_free_speech": wavlm_predictions, "candidate_B_timing_guided_interview": timing_predictions})
    write_csv(run_dir / "phase1_subgroup_audit.csv", subgroup_data)
    comparison_rows = final_comparison_rows(dirs["baseline"], dirs["phase1c"], dirs["phase1d"], dirs["phase1e"], final_metrics)
    write_csv(run_dir / "phase1_final_comparison.csv", comparison_rows)

    hashes = {
        "wavlm_pipeline_sha256": hash_model_pipeline(wavlm_model_path),
        "timing_pipeline_sha256": hash_model_pipeline(timing_model_path),
        "participant_embedding_manifest_sha256": sha256_file(dirs["phase1c"] / "aggregation_manifest.csv"),
        "timing_feature_table_sha256": sha256_file(dirs["phase1d"] / "speech_timing_features.csv"),
        "test_label_source_sha256": sha256_file(resolve_path("Data/full_test_split.csv", root)),
        "test_named_split_sha256": sha256_file(resolve_path("Data/test_split_Depression_AVEC2017.csv", root)),
        "phase1c_evaluation_report_sha256": sha256_file(dirs["phase1c"] / "evaluation_report.json"),
        "phase1e_evaluation_report_sha256": sha256_file(dirs["phase1e"] / "evaluation_report.json"),
        "git_commit": get_git_commit(root),
    }
    write_json(run_dir / "phase1f_hashes.json", hashes)

    verification = {
        "quality_control_status": "passed",
        "test_participants": int(len(test_meta)),
        "test_class_counts": {str(key): int(value) for key, value in Counter(y_test).items()},
        "pipeline_checks": pipeline_checks,
        "frozen_thresholds": {"candidate_A_wavlm_free_speech": WAVLM_THRESHOLD, "candidate_B_timing_guided_interview": TIMING_THRESHOLD},
        "no_test_participant_used_for_fitting": True,
        "no_post_test_model_modification": True,
    }
    final_report = write_final_reports(
        root,
        output_dir,
        run_dir,
        dirs,
        provenance_metadata,
        verification,
        final_metrics,
        bootstrap_rows,
        paired_rows,
        paired_meta,
        subgroup_data,
        comparison_rows,
        hashes,
    )

    metadata = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "machine_name": platform.node(),
        "python_version": sys.version.replace("\n", " "),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": get_package_version("scikit-learn"),
        "phase1_status": final_report["phase1_status"],
        "evaluation_status": EVALUATION_STATUS,
        "test_evaluation_performed": True,
        "test_probabilities_generated": True,
        "model_selection_performed": False,
        "hyperparameter_tuning_performed": False,
        "threshold_tuning_performed": False,
        "post_test_model_modification": False,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "random_seed": RANDOM_SEED,
        "input_hashes": hashes,
        "warnings": final_report["warnings"],
    }
    write_json(run_dir / "phase1f_run_metadata.json", metadata)

    print("Phase 1F final evaluation complete.")
    print(f"run_dir={path_relative_to_root(run_dir, root)}")
    print(f"phase1_status={final_report['phase1_status']}")
    print(f"wavlm_test_auc={final_metrics[(final_metrics['model_name'].eq('candidate_A_wavlm_free_speech')) & (final_metrics['threshold_type'].eq('frozen_dev_threshold'))].iloc[0]['roc_auc']:.6f}")
    print(f"timing_test_auc={final_metrics[(final_metrics['model_name'].eq('candidate_B_timing_guided_interview')) & (final_metrics['threshold_type'].eq('frozen_dev_threshold'))].iloc[0]['roc_auc']:.6f}")
    print("no_post_test_model_modification=true")


if __name__ == "__main__":
    main()
