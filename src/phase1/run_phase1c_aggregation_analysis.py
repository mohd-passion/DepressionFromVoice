"""Phase 1C participant-level aggregation analysis for locked WavLM layer 9.

This script does not run WavLM extraction and does not evaluate the test set.
It uses the Phase 1B layer-wise window embeddings as fixed inputs.
"""

from __future__ import annotations

import argparse
import csv
import gc
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
from sklearn.decomposition import PCA
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


PHASE1B_RUN_ID = "20260622_123725"
MODEL_NAME = "microsoft/wavlm-base-plus"
MODEL_COMMIT = "4c66d4806a428f2e922ccfa1a962776e232d487b"
LOCKED_LAYER_INDEX = 9
LOCKED_LAYER_NAME = "layer_09"
EXPECTED_WINDOWS = 17058
EXPECTED_WINDOW_SPLITS = {"train": 9001, "dev": 3314, "test": 4743}
EXPECTED_PARTICIPANT_SPLITS = {"train": 106, "dev": 34, "test": 47}
EXPECTED_LAYER_DIM = 768
EXPECTED_MEAN_STD_DIM = 1536
EXPECTED_LAYERWISE_SHAPE = (12, 768)
EXCLUDED_PARTICIPANTS = {"409", "422"}
RANDOM_SEED = 42
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
PCA_GRID: list[int | float] = [16, 32, 64, 0.95]
OUTER_SPLITS = 5
OUTER_REPEATS = 5
FINAL_SPLITS = 5
FINAL_REPEATS = 10
BOOTSTRAP_REPLICATES = 5000
COMPLEXITY_ORDER = {"mean": 1, "median": 2, "mean_std_pca": 3}


AGGREGATION_MANIFEST_FIELDS = [
    "participant_id",
    "official_split",
    "aggregation_method",
    "representation_path",
    "feature_dimension",
    "dtype",
    "num_windows_expected",
    "num_windows_used",
    "aggregation_status",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/phase1_daic")
    parser.add_argument("--phase1b-run-dir", default=f"outputs/phase1_daic/phase1b_layerwise_runs/{PHASE1B_RUN_ID}")
    parser.add_argument("--run-root", default="outputs/phase1_daic/phase1c_aggregation_runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dependency-dir", default=".phase1_deps")
    return parser.parse_args()


def add_dependency_dir(path: str) -> None:
    dep_path = Path(path)
    if dep_path.exists():
        resolved = str(dep_path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


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


def method_feature_dimension(method: str) -> int:
    return EXPECTED_MEAN_STD_DIM if method == "mean_std_pca" else EXPECTED_LAYER_DIM


def pca_label(config: int | float | None) -> str:
    if config is None:
        return ""
    if isinstance(config, float):
        return f"variance_{config:g}"
    return str(config)


def positive_probabilities(model: Any, x_data: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x_data)
    class_list = list(model.classes_)
    if 1 not in class_list:
        return np.zeros(probabilities.shape[0], dtype=np.float64)
    return probabilities[:, class_list.index(1)].astype(np.float64)


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


def build_pipeline(method: str, C: float, pca_config: int | float | None = None) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if method == "mean_std_pca":
        if pca_config is None:
            raise ValueError("mean_std_pca requires a PCA configuration.")
        steps.append(("pca", PCA(n_components=pca_config, whiten=False, svd_solver="full")))
    steps.append(
        (
            "classifier",
            LogisticRegression(
                C=C,
                class_weight="balanced",
                max_iter=10000,
                random_state=RANDOM_SEED,
            ),
        )
    )
    return Pipeline(steps)


def valid_pca_config(config: int | float | None, n_samples: int, n_features: int) -> bool:
    if config is None:
        return True
    if isinstance(config, float):
        return 0.0 < config < 1.0
    return config <= min(n_samples, n_features)


def parameter_grid(method: str, n_samples: int, n_features: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    if method == "mean_std_pca":
        for pca_config in PCA_GRID:
            for C in C_GRID:
                if valid_pca_config(pca_config, n_samples, n_features):
                    candidates.append({"C": C, "pca_config": pca_config})
                else:
                    skipped.append({"C": C, "pca_config": pca_label(pca_config), "reason": f"pca_config_exceeds_min_samples_features={min(n_samples, n_features)}"})
    else:
        candidates = [{"C": C, "pca_config": None} for C in C_GRID]
    return candidates, skipped


def parameter_sort_key(method: str, row: dict[str, Any]) -> tuple[float, float, int, float]:
    pca_config = row.get("pca_config")
    if method == "mean_std_pca":
        if isinstance(pca_config, float):
            pca_complexity = 10**9
        else:
            pca_complexity = int(pca_config)
    else:
        pca_complexity = 0
    return (-float(row["mean_roc_auc"]), float(row["roc_auc_std"]), pca_complexity, float(row["C"]))


def select_inner_parameters(method: str, x_train: np.ndarray, y_train: np.ndarray, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates, skipped = parameter_grid(method, len(y_train), x_train.shape[1])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows: list[dict[str, Any]] = []
    for params in candidates:
        aucs: list[float] = []
        pr_aucs: list[float] = []
        bal_accs: list[float] = []
        macro_f1s: list[float] = []
        for train_index, validation_index in splitter.split(x_train, y_train):
            fold_candidates, fold_skipped = parameter_grid(method, len(train_index), x_train.shape[1])
            if params not in fold_candidates:
                skipped.append({**params, "pca_config": pca_label(params.get("pca_config")), "reason": "skipped_in_inner_fold_due_to_component_limit"})
                continue
            model = build_pipeline(method, params["C"], params.get("pca_config"))
            model.fit(x_train[train_index], y_train[train_index])
            probabilities = positive_probabilities(model, x_train[validation_index])
            metrics = metric_dict(y_train[validation_index], probabilities, 0.5)
            aucs.append(metrics["roc_auc"])
            pr_aucs.append(metrics["pr_auc"])
            bal_accs.append(metrics["balanced_accuracy"])
            macro_f1s.append(metrics["macro_f1"])
        if not aucs:
            continue
        rows.append(
            {
                "C": params["C"],
                "pca_config": params.get("pca_config"),
                "pca_config_label": pca_label(params.get("pca_config")),
                "mean_roc_auc": float(np.mean(aucs)),
                "roc_auc_std": float(np.std(aucs, ddof=0)),
                "mean_pr_auc": float(np.mean(pr_aucs)),
                "mean_balanced_accuracy_at_0_5": float(np.mean(bal_accs)),
                "mean_macro_f1_at_0_5": float(np.mean(macro_f1s)),
            }
        )
    if not rows:
        raise RuntimeError(f"No valid inner candidates for method {method}.")
    selected = sorted(rows, key=lambda row: parameter_sort_key(method, row))[0]
    return selected, rows, skipped


def deterministic_thresholds(probabilities: np.ndarray) -> list[float]:
    unique_probabilities = sorted({float(value) for value in probabilities})
    thresholds = {0.5}
    thresholds.update(unique_probabilities)
    for left, right in zip(unique_probabilities, unique_probabilities[1:]):
        thresholds.add((left + right) / 2.0)
    return sorted(thresholds)


def tune_threshold(y_dev: np.ndarray, probabilities: np.ndarray) -> tuple[pd.DataFrame, float]:
    rows = [metric_dict(y_dev, probabilities, threshold) for threshold in deterministic_thresholds(probabilities)]
    frame = pd.DataFrame(rows)
    selected = frame.assign(distance_to_0_5=(frame["threshold"] - 0.5).abs()).sort_values(
        ["macro_f1", "balanced_accuracy", "distance_to_0_5", "threshold"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return frame, float(selected["threshold"])


def layer_selection_stability(phase1b_run_dir: Path, run_dir: Path) -> dict[str, Any]:
    outer = pd.read_csv(phase1b_run_dir / "layerwise_outer_fold_results.csv")
    predictions = pd.read_csv(phase1b_run_dir / "layerwise_outer_predictions.csv", dtype={"participant_id": str})
    target_layers = ["layer_01", "layer_02", "layer_08", "layer_12"]
    layer9 = outer[outer["layer"].eq(LOCKED_LAYER_NAME)][["outer_fold_index", "roc_auc"]].rename(columns={"roc_auc": "layer09_roc_auc"})
    stability_rows: list[dict[str, Any]] = []
    for target in target_layers:
        target_frame = outer[outer["layer"].eq(target)][["outer_fold_index", "roc_auc"]].rename(columns={"roc_auc": "target_roc_auc"})
        paired = layer9.merge(target_frame, on="outer_fold_index", how="inner")
        differences = paired["layer09_roc_auc"] - paired["target_roc_auc"]
        layer9_wins = int((differences > 0).sum())
        target_wins = int((differences < 0).sum())
        ties = int((differences == 0).sum())
        stability_rows.append(
            {
                "comparison": f"{LOCKED_LAYER_NAME}_minus_{target}",
                "target_layer": target,
                "mean_paired_outer_fold_roc_auc_difference": float(differences.mean()),
                "median_paired_difference": float(differences.median()),
                "min_paired_difference": float(differences.min()),
                "max_paired_difference": float(differences.max()),
                "layer09_outer_folds_won": layer9_wins,
                "target_outer_folds_won": target_wins,
                "tied_outer_folds": ties,
                "layer09_outer_folds_won_percent": float(layer9_wins / len(paired) * 100.0),
                "target_outer_folds_won_percent": float(target_wins / len(paired) * 100.0),
                "num_paired_outer_folds": int(len(paired)),
            }
        )

    rank_rows: list[dict[str, Any]] = []
    for outer_fold_index, fold in outer.groupby("outer_fold_index"):
        repeat_index = int((int(outer_fold_index) - 1) // 5 + 1)
        fold_number = int((int(outer_fold_index) - 1) % 5 + 1)
        ranked = fold.sort_values(["roc_auc", "layer_index"], ascending=[False, True]).copy()
        ranked["rank"] = range(1, len(ranked) + 1)
        for row in ranked.itertuples(index=False):
            rank_rows.append(
                {
                    "outer_fold_index": int(outer_fold_index),
                    "repeat_index": repeat_index,
                    "fold_number": fold_number,
                    "layer": row.layer,
                    "layer_index": int(row.layer_index),
                    "roc_auc": float(row.roc_auc),
                    "rank": int(row.rank),
                }
            )
    rank_frame = pd.DataFrame(rank_rows)
    rank_frame.to_csv(run_dir / "layer_rank_by_outer_fold.csv", index=False)
    rank_summary = rank_frame.groupby(["layer", "layer_index"], as_index=False).agg(mean_rank=("rank", "mean"), rank_std=("rank", "std"))
    rank_summary["rank_std"] = rank_summary["rank_std"].fillna(0.0)

    repeat_winners: Counter[str] = Counter()
    for repeat_index, repeat_rows in outer.assign(repeat_index=((outer["outer_fold_index"] - 1) // 5 + 1)).groupby("repeat_index"):
        means = repeat_rows.groupby("layer")["roc_auc"].mean()
        max_mean = means.max()
        for layer, value in means.items():
            if value == max_mean:
                repeat_winners[layer] += 1

    layer_rows: list[dict[str, Any]] = []
    for row in rank_summary.itertuples(index=False):
        layer_rows.append(
            {
                "comparison": "rank_summary_all_layers",
                "target_layer": row.layer,
                "mean_paired_outer_fold_roc_auc_difference": "",
                "median_paired_difference": "",
                "min_paired_difference": "",
                "max_paired_difference": "",
                "layer09_outer_folds_won": "",
                "target_outer_folds_won": "",
                "tied_outer_folds": "",
                "layer09_outer_folds_won_percent": "",
                "target_outer_folds_won_percent": "",
                "num_paired_outer_folds": "",
                "mean_rank": float(row.mean_rank),
                "rank_std": float(row.rank_std),
                "repeat_top_mean_auc_count": int(repeat_winners[row.layer]),
            }
        )
    write_csv(run_dir / "layer_selection_stability.csv", stability_rows + layer_rows)

    averaged = (
        predictions.groupby(["layer", "participant_id", "label_binary"], as_index=False)["predicted_probability"]
        .mean()
        .rename(columns={"predicted_probability": "mean_oof_probability"})
    )
    pivot = averaged.pivot(index=["participant_id", "label_binary"], columns="layer", values="mean_oof_probability").reset_index()
    y = pivot["label_binary"].to_numpy(dtype=int)
    prob9 = pivot[LOCKED_LAYER_NAME].to_numpy(dtype=float)
    prob1 = pivot["layer_01"].to_numpy(dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    pos_indices = np.flatnonzero(y == 1)
    neg_indices = np.flatnonzero(y == 0)
    bootstrap_rows: list[dict[str, Any]] = []
    for replicate in range(1, BOOTSTRAP_REPLICATES + 1):
        sampled = np.concatenate(
            [
                rng.choice(neg_indices, size=len(neg_indices), replace=True),
                rng.choice(pos_indices, size=len(pos_indices), replace=True),
            ]
        )
        auc9 = float(roc_auc_score(y[sampled], prob9[sampled]))
        auc1 = float(roc_auc_score(y[sampled], prob1[sampled]))
        bootstrap_rows.append(
            {
                "replicate": replicate,
                "layer09_roc_auc": auc9,
                "layer01_roc_auc": auc1,
                "roc_auc_difference_layer09_minus_layer01": auc9 - auc1,
            }
        )
    write_csv(run_dir / "layer09_vs_layer01_bootstrap.csv", bootstrap_rows)
    differences = np.array([row["roc_auc_difference_layer09_minus_layer01"] for row in bootstrap_rows], dtype=float)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_descriptive_stability_audit",
        "locked_layer_for_phase1c": LOCKED_LAYER_NAME,
        "layer_locked_regardless_of_audit_outcome": True,
        "paired_comparisons": stability_rows,
        "rank_summary": layer_rows,
        "repeat_top_mean_auc_counts": dict(repeat_winners),
        "bootstrap": {
            "description": "Exploratory participant bootstrap over averaged train out-of-fold probabilities; repeated CV folds are dependent, so this is not a definitive significance test.",
            "replicates": BOOTSTRAP_REPLICATES,
            "resampling": "stratified by label",
            "comparison": "layer_09 minus layer_01 ROC-AUC",
            "median_difference": float(np.median(differences)),
            "percentile_95_interval": [float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))],
        },
    }
    write_json(run_dir / "layer_selection_stability_report.json", report)
    return report


def validate_and_aggregate_layer9(root: Path, output_dir: Path, phase1b_run_dir: Path, run_dir: Path) -> dict[str, Any]:
    participant_rows: dict[str, dict[str, str]] = {}
    with (output_dir / "daic_phase1_participant_manifest_clean.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            participant_id = clean(row["participant_id"])
            participant_rows[participant_id] = {
                "participant_id": participant_id,
                "official_split": clean(row["official_split"]),
                "num_windows": clean(row["num_windows"]),
            }
    window_rows = read_csv_dicts(phase1b_run_dir / "layerwise_window_embedding_manifest.csv")
    issues: list[dict[str, Any]] = []
    split_counts = Counter(clean(row["official_split"]) for row in window_rows)
    participant_ids = {clean(row["participant_id"]) for row in window_rows}
    window_ids = [clean(row["window_id"]) for row in window_rows]
    duplicate_windows = len(window_ids) - len(set(window_ids))
    if len(window_rows) != EXPECTED_WINDOWS:
        issues.append(make_issue("error", "window_count_mismatch", f"Expected {EXPECTED_WINDOWS}, found {len(window_rows)}."))
    if dict(split_counts) != EXPECTED_WINDOW_SPLITS:
        issues.append(make_issue("error", "window_split_count_mismatch", f"Expected {EXPECTED_WINDOW_SPLITS}, found {dict(split_counts)}."))
    if duplicate_windows:
        issues.append(make_issue("error", "duplicate_windows", f"Duplicate window count {duplicate_windows}."))
    excluded = sorted(participant_ids & EXCLUDED_PARTICIPANTS, key=participant_sort_key)
    if excluded:
        issues.append(make_issue("error", "excluded_participants_present", f"{excluded}."))

    vectors_by_participant: dict[str, list[np.ndarray]] = defaultdict(list)
    split_by_participant: dict[str, set[str]] = defaultdict(set)
    window_link_counts: Counter[str] = Counter()
    missing_files = 0
    nonfinite = 0
    all_zero = 0
    bad_shape = 0
    bad_status = 0
    start_time = time.time()
    for index, row in enumerate(window_rows, start=1):
        participant_id = clean(row["participant_id"])
        split = clean(row["official_split"])
        split_by_participant[participant_id].add(split)
        window_link_counts[clean(row["window_id"])] += 1
        if clean(row.get("extraction_status")) not in {"success", "skipped_valid"}:
            bad_status += 1
            continue
        path = resolve_path(clean(row["embedding_path"]), root)
        if not path.exists():
            missing_files += 1
            continue
        matrix = np.load(path)
        if matrix.shape != EXPECTED_LAYERWISE_SHAPE:
            bad_shape += 1
            continue
        vector = matrix[LOCKED_LAYER_INDEX - 1].astype(np.float32, copy=False)
        if vector.shape != (EXPECTED_LAYER_DIM,):
            bad_shape += 1
            continue
        if not np.isfinite(vector).all():
            nonfinite += 1
            continue
        if np.all(vector == 0):
            all_zero += 1
            continue
        vectors_by_participant[participant_id].append(vector.copy())
        if index % 2500 == 0:
            print(f"aggregation_load_progress={index}/{len(window_rows)} elapsed_sec={time.time() - start_time:.1f}", flush=True)

    if missing_files:
        issues.append(make_issue("error", "missing_window_embedding_files", f"Missing files: {missing_files}."))
    if bad_status:
        issues.append(make_issue("error", "bad_window_status", f"Non-success statuses: {bad_status}."))
    if bad_shape:
        issues.append(make_issue("error", "bad_layer_matrix_shape", f"Bad shape rows: {bad_shape}."))
    if nonfinite:
        issues.append(make_issue("error", "nonfinite_layer9_vectors", f"Nonfinite layer-9 vectors: {nonfinite}."))
    if all_zero:
        issues.append(make_issue("error", "all_zero_layer9_vectors", f"All-zero layer-9 vectors: {all_zero}."))
    if any(count != 1 for count in window_link_counts.values()):
        issues.append(make_issue("error", "window_not_linked_exactly_once", "At least one window did not have exactly one participant/split link."))
    leakage = sorted([pid for pid, splits in split_by_participant.items() if len(splits) > 1], key=participant_sort_key)
    if leakage:
        issues.append(make_issue("error", "participant_leakage_multiple_splits", f"{leakage[:10]}."))

    representation_root = run_dir / "participant_representations"
    manifest_rows: list[dict[str, Any]] = []
    total_windows_accounted: dict[str, int] = {"mean": 0, "median": 0, "mean_std_pca": 0}
    for participant_id in sorted(participant_rows, key=participant_sort_key):
        participant = participant_rows[participant_id]
        expected_windows = int(clean(participant.get("num_windows")) or "0")
        vectors = vectors_by_participant.get(participant_id, [])
        split = clean(participant["official_split"])
        if len(vectors) != expected_windows:
            issues.append(make_issue("error", "participant_window_count_mismatch", f"{participant_id}: expected {expected_windows}, found {len(vectors)}.", participant_id=participant_id))
            status = "failed"
            error = f"expected_windows={expected_windows} found={len(vectors)}"
        elif participant_id in EXCLUDED_PARTICIPANTS:
            status = "failed"
            error = "excluded_participant_present"
        else:
            status = "success"
            error = ""

        if status == "success":
            matrix = np.vstack(vectors).astype(np.float64, copy=False)
            representations = {
                "mean": matrix.mean(axis=0).astype(np.float32),
                "median": np.median(matrix, axis=0).astype(np.float32),
                "mean_std_pca": np.concatenate([matrix.mean(axis=0), matrix.std(axis=0, ddof=0)]).astype(np.float32),
            }
        else:
            representations = {
                "mean": np.array([], dtype=np.float32),
                "median": np.array([], dtype=np.float32),
                "mean_std_pca": np.array([], dtype=np.float32),
            }

        for method, representation in representations.items():
            method_dir = representation_root / method
            method_dir.mkdir(parents=True, exist_ok=True)
            output_path = method_dir / f"{participant_id}.npy"
            method_status = status
            method_error = error
            expected_dim = method_feature_dimension(method)
            if method_status == "success":
                if representation.shape != (expected_dim,) or not np.isfinite(representation).all() or np.all(representation == 0):
                    method_status = "failed"
                    method_error = "invalid_participant_representation"
                    issues.append(make_issue("error", "invalid_participant_representation", f"{participant_id} {method}", participant_id=participant_id, aggregation_method=method))
                else:
                    np.save(output_path, representation)
                    total_windows_accounted[method] += len(vectors)
            manifest_rows.append(
                {
                    "participant_id": participant_id,
                    "official_split": split,
                    "aggregation_method": method,
                    "representation_path": path_relative_to_root(output_path, root),
                    "feature_dimension": expected_dim if method_status == "success" else "",
                    "dtype": "float32" if method_status == "success" else "",
                    "num_windows_expected": expected_windows,
                    "num_windows_used": len(vectors),
                    "aggregation_status": method_status,
                    "error_message": method_error,
                }
            )

    for method in ("mean", "median", "mean_std_pca"):
        rows = [row for row in manifest_rows if row["aggregation_method"] == method]
        success_rows = [row for row in rows if row["aggregation_status"] == "success"]
        split_count = Counter(row["official_split"] for row in success_rows)
        participant_count = len({row["participant_id"] for row in rows})
        if len(success_rows) != 187:
            issues.append(make_issue("error", "method_participant_count_mismatch", f"{method}: expected 187, found {len(success_rows)}.", aggregation_method=method))
        if dict(split_count) != EXPECTED_PARTICIPANT_SPLITS:
            issues.append(make_issue("error", "method_split_count_mismatch", f"{method}: expected {EXPECTED_PARTICIPANT_SPLITS}, found {dict(split_count)}.", aggregation_method=method))
        if participant_count != 187:
            issues.append(make_issue("error", "method_duplicate_or_missing_participants", f"{method}: participant count {participant_count}.", aggregation_method=method))
        if total_windows_accounted[method] != EXPECTED_WINDOWS:
            issues.append(make_issue("error", "method_total_windows_accounted_mismatch", f"{method}: expected {EXPECTED_WINDOWS}, found {total_windows_accounted[method]}.", aggregation_method=method))

    quality_rows = issues or [make_issue("info", "all_aggregation_checks_passed", "All Phase 1C aggregation checks passed.")]
    write_csv(run_dir / "aggregation_manifest.csv", manifest_rows, AGGREGATION_MANIFEST_FIELDS)
    write_csv(run_dir / "aggregation_quality_report.csv", quality_rows)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not any(row["severity"] == "error" for row in issues) else "failed",
        "locked_layer": LOCKED_LAYER_NAME,
        "source_phase1b_run": str(phase1b_run_dir),
        "methods": {
            "mean": {"dimension": EXPECTED_LAYER_DIM, "pca_used": False},
            "median": {"dimension": EXPECTED_LAYER_DIM, "pca_used": False},
            "mean_std_pca": {"dimension": EXPECTED_MEAN_STD_DIM, "pca_used_in_model_pipeline": True},
        },
        "window_counts_by_split": dict(split_counts),
        "participant_counts_by_split": EXPECTED_PARTICIPANT_SPLITS,
        "total_windows_accounted_by_method": total_windows_accounted,
        "test_evaluation_performed": False,
        "test_labels_loaded_for_modeling": False,
    }
    write_json(run_dir / "aggregation_metadata.json", metadata)
    if metadata["status"] != "passed":
        raise RuntimeError("Phase 1C aggregation validation failed.")
    gc.collect()
    return metadata


def load_method_inputs(root: Path, output_dir: Path, run_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(run_dir / "aggregation_manifest.csv", dtype={"participant_id": str})
    modeling_frame = manifest[manifest["official_split"].isin(["train", "dev"])].copy()
    assert not modeling_frame["official_split"].eq("test").any(), "Modeling dataframe contains test participants before labels are loaded."

    label_rows: list[dict[str, Any]] = []
    with (output_dir / "daic_phase1_participant_manifest_clean.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split = clean(row.get("official_split"))
            if split in {"train", "dev"}:
                label_rows.append({"participant_id": clean(row["participant_id"]), "label_binary": int(clean(row["label_binary"]))})
    labels = pd.DataFrame(label_rows)

    method_train: dict[str, np.ndarray] = {}
    method_dev: dict[str, np.ndarray] = {}
    base_frame: pd.DataFrame | None = None
    train_labels: np.ndarray | None = None
    dev_labels: np.ndarray | None = None
    train_frame: pd.DataFrame | None = None
    dev_frame: pd.DataFrame | None = None

    for method in ("mean", "median", "mean_std_pca"):
        frame = modeling_frame[modeling_frame["aggregation_method"].eq(method)].merge(labels, on="participant_id", how="inner", validate="one_to_one")
        frame = frame.sort_values("participant_id", key=lambda series: series.map(lambda value: participant_sort_key(str(value))[0])).reset_index(drop=True)
        assert not frame["official_split"].eq("test").any(), "Labels were loaded for test participants."
        matrices: list[np.ndarray] = []
        for row in frame.itertuples(index=False):
            representation = np.load(resolve_path(row.representation_path, root)).astype(np.float32, copy=False)
            expected_dim = method_feature_dimension(method)
            if representation.shape != (expected_dim,):
                raise RuntimeError(f"Unexpected shape for {method} participant {row.participant_id}: {representation.shape}")
            matrices.append(representation)
        x_all = np.vstack(matrices)
        y_all = frame["label_binary"].to_numpy(dtype=int)
        train_mask = frame["official_split"].eq("train").to_numpy()
        dev_mask = frame["official_split"].eq("dev").to_numpy()
        method_train[method] = x_all[train_mask]
        method_dev[method] = x_all[dev_mask]
        if base_frame is None:
            base_frame = frame
            train_labels = y_all[train_mask]
            dev_labels = y_all[dev_mask]
            train_frame = frame.loc[train_mask].reset_index(drop=True)
            dev_frame = frame.loc[dev_mask].reset_index(drop=True)
        else:
            if not np.array_equal(train_labels, y_all[train_mask]) or not np.array_equal(dev_labels, y_all[dev_mask]):
                raise RuntimeError("Label alignment mismatch across aggregation methods.")
    assert train_labels is not None and dev_labels is not None and train_frame is not None and dev_frame is not None
    return base_frame, method_train, method_dev, train_labels, dev_labels, train_frame, dev_frame


def nested_cv_aggregation(method_train: dict[str, np.ndarray], y_train: np.ndarray, train_frame: pd.DataFrame, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selected_parameter_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    splitter = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=RANDOM_SEED)
    for method, x_train in method_train.items():
        method_metrics: list[dict[str, Any]] = []
        for outer_fold_index, (outer_train_index, outer_validation_index) in enumerate(splitter.split(x_train, y_train), start=1):
            inner_seed = RANDOM_SEED * 1000 + outer_fold_index
            selected, inner_rows, skipped = select_inner_parameters(method, x_train[outer_train_index], y_train[outer_train_index], inner_seed)
            skipped_rows.extend([{**row, "aggregation_method": method, "outer_fold_index": outer_fold_index, "inner_seed": inner_seed} for row in skipped])
            selected_parameter_rows.append(
                {
                    "aggregation_method": method,
                    "outer_fold_index": outer_fold_index,
                    "inner_seed": inner_seed,
                    "selected_C": selected["C"],
                    "selected_pca_config": selected["pca_config_label"],
                    "inner_mean_roc_auc": selected["mean_roc_auc"],
                    "inner_roc_auc_std": selected["roc_auc_std"],
                }
            )
            model = build_pipeline(method, selected["C"], selected.get("pca_config"))
            model.fit(x_train[outer_train_index], y_train[outer_train_index])
            probabilities = positive_probabilities(model, x_train[outer_validation_index])
            metrics = metric_dict(y_train[outer_validation_index], probabilities, 0.5)
            row = {
                "aggregation_method": method,
                "outer_fold_index": outer_fold_index,
                "repeat_index": int((outer_fold_index - 1) // 5 + 1),
                "fold_number": int((outer_fold_index - 1) % 5 + 1),
                "selected_C": selected["C"],
                "selected_pca_config": selected["pca_config_label"],
                **metrics,
                "n_outer_train": int(len(outer_train_index)),
                "n_outer_validation": int(len(outer_validation_index)),
            }
            outer_rows.append(row)
            method_metrics.append(row)
            for participant_position, probability in zip(outer_validation_index, probabilities):
                participant = train_frame.iloc[int(participant_position)]
                prediction_rows.append(
                    {
                        "aggregation_method": method,
                        "outer_fold_index": outer_fold_index,
                        "participant_id": participant["participant_id"],
                        "official_split": participant["official_split"],
                        "label_binary": int(y_train[int(participant_position)]),
                        "predicted_probability": float(probability),
                        "prediction_at_0_5": int(probability >= 0.5),
                        "selected_C": selected["C"],
                        "selected_pca_config": selected["pca_config_label"],
                    }
                )
        print(f"aggregation_nested_cv_complete method={method} mean_auc={np.mean([row['roc_auc'] for row in method_metrics]):.4f}", flush=True)

    summary_rows: list[dict[str, Any]] = []
    for method, rows in pd.DataFrame(outer_rows).groupby("aggregation_method"):
        aucs = rows["roc_auc"].to_numpy(dtype=float)
        pr_aucs = rows["pr_auc"].to_numpy(dtype=float)
        bal_accs = rows["balanced_accuracy"].to_numpy(dtype=float)
        macro_f1s = rows["macro_f1"].to_numpy(dtype=float)
        c_counts = Counter(rows["selected_C"].astype(str))
        pca_counts = Counter(rows["selected_pca_config"].astype(str)) if method == "mean_std_pca" else Counter()
        summary_rows.append(
            {
                "aggregation_method": method,
                "complexity_order": COMPLEXITY_ORDER[method],
                "representation_dimension": method_feature_dimension(method),
                "pca_used": method == "mean_std_pca",
                "mean_outer_roc_auc": float(aucs.mean()),
                "std_outer_roc_auc": float(aucs.std(ddof=0)),
                "median_outer_roc_auc": float(np.median(aucs)),
                "min_outer_roc_auc": float(aucs.min()),
                "max_outer_roc_auc": float(aucs.max()),
                "mean_outer_pr_auc": float(pr_aucs.mean()),
                "mean_outer_balanced_accuracy_at_0_5": float(bal_accs.mean()),
                "mean_outer_macro_f1_at_0_5": float(macro_f1s.mean()),
                "selected_C_frequency": json.dumps({key: c_counts[key] for key in sorted(c_counts)}),
                "selected_PCA_frequency": json.dumps({key: pca_counts[key] for key in sorted(pca_counts)}) if pca_counts else "",
                "n_outer_evaluations": int(len(rows)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("complexity_order")
    outer = pd.DataFrame(outer_rows)
    predictions = pd.DataFrame(prediction_rows)
    selected_parameters = pd.DataFrame(selected_parameter_rows)
    summary.to_csv(run_dir / "aggregation_nested_cv_summary.csv", index=False)
    outer.to_csv(run_dir / "aggregation_outer_fold_results.csv", index=False)
    predictions.to_csv(run_dir / "aggregation_outer_predictions.csv", index=False)
    selected_parameters.to_csv(run_dir / "aggregation_selected_parameters_by_fold.csv", index=False)
    if skipped_rows:
        write_csv(run_dir / "aggregation_skipped_inner_candidates.csv", skipped_rows)
    else:
        write_csv(run_dir / "aggregation_skipped_inner_candidates.csv", [make_issue("info", "no_inner_candidates_skipped", "No inner candidates were skipped.")])
    return summary, outer, predictions, selected_parameters


def select_aggregation(summary: pd.DataFrame, run_dir: Path) -> dict[str, Any]:
    sorted_summary = summary.sort_values(
        ["mean_outer_roc_auc", "std_outer_roc_auc", "mean_outer_pr_auc", "complexity_order"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    nominal_best = sorted_summary.iloc[0].to_dict()
    mean_row = summary[summary["aggregation_method"].eq("mean")].iloc[0].to_dict()
    improvement_over_mean = float(nominal_best["mean_outer_roc_auc"] - mean_row["mean_outer_roc_auc"])
    variability_reduction = float(mean_row["std_outer_roc_auc"] - nominal_best["std_outer_roc_auc"])
    selected = nominal_best
    practical_rule_applied = False
    practical_rule = "If the best method improves mean ROC-AUC over mean aggregation by less than 0.01, select mean unless the more complex method also reduces ROC-AUC std by at least 0.01."
    if nominal_best["aggregation_method"] != "mean" and improvement_over_mean < 0.01 and variability_reduction < 0.01:
        selected = mean_row
        practical_rule_applied = True
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_method": selected["aggregation_method"],
        "nominal_best_method": nominal_best["aggregation_method"],
        "selection_criteria": [
            "highest mean outer ROC-AUC",
            "lowest ROC-AUC standard deviation",
            "highest mean PR-AUC",
            "simpler representation if effectively tied",
            practical_rule,
        ],
        "improvement_over_mean_roc_auc": improvement_over_mean,
        "std_reduction_vs_mean": variability_reduction,
        "practical_negligibility_rule_applied": practical_rule_applied,
        "selected_method_summary": selected,
        "mean_method_summary": mean_row,
        "selected_pipeline_structure": "StandardScaler + LogisticRegression"
        if selected["aggregation_method"] in {"mean", "median"}
        else "StandardScaler + PCA + LogisticRegression",
        "test_data_used_for_selection": False,
    }
    write_json(run_dir / "selected_aggregation_method.json", report)
    return report


def final_hyperparameter_selection(method: str, x_train: np.ndarray, y_train: np.ndarray, run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    splitter = RepeatedStratifiedKFold(n_splits=FINAL_SPLITS, n_repeats=FINAL_REPEATS, random_state=RANDOM_SEED)
    candidates, skipped = parameter_grid(method, len(y_train), x_train.shape[1])
    rows: list[dict[str, Any]] = []
    for params in candidates:
        aucs: list[float] = []
        pr_aucs: list[float] = []
        bal_accs: list[float] = []
        macro_f1s: list[float] = []
        for train_index, validation_index in splitter.split(x_train, y_train):
            fold_candidates, _ = parameter_grid(method, len(train_index), x_train.shape[1])
            if params not in fold_candidates:
                skipped.append({**params, "pca_config": pca_label(params.get("pca_config")), "reason": "skipped_in_final_cv_fold_due_to_component_limit"})
                continue
            model = build_pipeline(method, params["C"], params.get("pca_config"))
            model.fit(x_train[train_index], y_train[train_index])
            probabilities = positive_probabilities(model, x_train[validation_index])
            metrics = metric_dict(y_train[validation_index], probabilities, 0.5)
            aucs.append(metrics["roc_auc"])
            pr_aucs.append(metrics["pr_auc"])
            bal_accs.append(metrics["balanced_accuracy"])
            macro_f1s.append(metrics["macro_f1"])
        if not aucs:
            continue
        rows.append(
            {
                "aggregation_method": method,
                "C": params["C"],
                "pca_config": pca_label(params.get("pca_config")),
                "mean_roc_auc": float(np.mean(aucs)),
                "roc_auc_std": float(np.std(aucs, ddof=0)),
                "mean_pr_auc": float(np.mean(pr_aucs)),
                "mean_balanced_accuracy_at_0_5": float(np.mean(bal_accs)),
                "mean_macro_f1_at_0_5": float(np.mean(macro_f1s)),
                "n_splits": FINAL_SPLITS,
                "n_repeats": FINAL_REPEATS,
                "random_state": RANDOM_SEED,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No valid final hyperparameter candidates for method {method}.")
    selected = frame.assign(
        pca_sort=frame["pca_config"].map(lambda value: 0 if value == "" else (10**9 if str(value).startswith("variance") else int(value)))
    ).sort_values(["mean_roc_auc", "roc_auc_std", "pca_sort", "C"], ascending=[False, True, True, True]).iloc[0]
    frame.drop(columns=["pca_sort"], errors="ignore").to_csv(run_dir / "final_hyperparameter_search.csv", index=False)
    if skipped:
        write_csv(run_dir / "final_hyperparameter_skipped_candidates.csv", skipped)
    selected_params = {
        "aggregation_method": method,
        "selected_C": float(selected["C"]),
        "selected_pca_config": clean(selected["pca_config"]),
        "selection_rule": ["highest mean ROC-AUC", "lowest ROC-AUC std", "fewer PCA components", "lower C"],
    }
    write_json(run_dir / "final_selected_hyperparameters.json", selected_params)
    return frame, selected_params


def evaluate_selected(
    root: Path,
    run_dir: Path,
    selected_report: dict[str, Any],
    method_train: dict[str, np.ndarray],
    method_dev: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_dev: np.ndarray,
    dev_frame: pd.DataFrame,
) -> dict[str, Any]:
    method = selected_report["selected_method"]
    final_search, selected_params = final_hyperparameter_selection(method, method_train[method], y_train, run_dir)
    pca_config_text = selected_params["selected_pca_config"]
    pca_config: int | float | None
    if not pca_config_text:
        pca_config = None
    elif pca_config_text.startswith("variance_"):
        pca_config = float(pca_config_text.replace("variance_", ""))
    else:
        pca_config = int(pca_config_text)
    model = build_pipeline(method, selected_params["selected_C"], pca_config)
    model.fit(method_train[method], y_train)
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "selected_aggregation_pipeline.joblib"
    joblib.dump(model, model_path)
    probabilities = positive_probabilities(model, method_dev[method])
    threshold_search, selected_threshold = tune_threshold(y_dev, probabilities)
    threshold_search.to_csv(run_dir / "dev_threshold_search.csv", index=False)
    metrics_0_5 = metric_dict(y_dev, probabilities, 0.5)
    metrics_tuned = metric_dict(y_dev, probabilities, selected_threshold)
    prediction_rows: list[dict[str, Any]] = []
    for participant, probability in zip(dev_frame.itertuples(index=False), probabilities):
        prediction_rows.append(
            {
                "participant_id": participant.participant_id,
                "official_split": participant.official_split,
                "label_binary": int(participant.label_binary),
                "aggregation_method": method,
                "predicted_probability": float(probability),
                "prediction_at_0_5": int(probability >= 0.5),
                "prediction_at_tuned_threshold": int(probability >= selected_threshold),
                "tuned_threshold": selected_threshold,
                "model_name": "phase1c_selected_aggregation_logistic_regression",
            }
        )
    write_csv(run_dir / "selected_aggregation_dev_predictions.csv", prediction_rows)
    result = {
        "selected_method": method,
        "selected_C": selected_params["selected_C"],
        "selected_pca_config": selected_params["selected_pca_config"],
        "selected_dev_threshold": selected_threshold,
        "dev_metrics_at_0_5": metrics_0_5,
        "dev_metrics_at_selected_threshold": metrics_tuned,
        "model_path": path_relative_to_root(model_path, root),
        "test_evaluation_performed": False,
    }
    return result


def write_comparison_and_reports(
    root: Path,
    output_dir: Path,
    phase1b_run_dir: Path,
    run_dir: Path,
    stability_report: dict[str, Any],
    aggregation_metadata: dict[str, Any],
    summary: pd.DataFrame,
    selected_report: dict[str, Any],
    evaluation: dict[str, Any],
) -> None:
    phase1b_eval = json.loads((phase1b_run_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    phase1b_selected = phase1b_eval["selected_layer_final_evaluation"]
    phase1b_final_c = pd.read_csv(phase1b_run_dir / "selected_layer_final_c_search.csv")
    phase1b_final_c_selected = phase1b_final_c[phase1b_final_c["C"].eq(float(phase1b_selected["selected_C"]))].iloc[0]
    comparison_rows: list[dict[str, Any]] = []
    mean_summary = summary[summary["aggregation_method"].eq("mean")].iloc[0].to_dict()
    comparison_rows.append(
        {
            "row_name": "phase1b_locked_layer9_mean_baseline",
            "aggregation_method": "mean",
            "representation_dimension": mean_summary["representation_dimension"],
            "pca_used": mean_summary["pca_used"],
            "computational_complexity": COMPLEXITY_ORDER["mean"],
            "train_nested_cv_mean_roc_auc": phase1b_final_c_selected["mean_roc_auc"],
            "train_nested_cv_std_roc_auc": phase1b_final_c_selected["roc_auc_std"],
            "train_nested_cv_mean_pr_auc": "",
            "train_nested_cv_mean_balanced_accuracy": phase1b_final_c_selected["mean_balanced_accuracy_at_0_5"],
            "train_nested_cv_mean_macro_f1": phase1b_final_c_selected["mean_macro_f1_at_0_5"],
            "dev_roc_auc": phase1b_selected["dev_metrics_at_0_5"]["roc_auc"],
            "dev_pr_auc": phase1b_selected["dev_metrics_at_0_5"]["pr_auc"],
            "dev_balanced_accuracy_at_0_5": phase1b_selected["dev_metrics_at_0_5"]["balanced_accuracy"],
            "dev_macro_f1_at_0_5": phase1b_selected["dev_metrics_at_0_5"]["macro_f1"],
            "dev_sensitivity_at_0_5": phase1b_selected["dev_metrics_at_0_5"]["positive_recall_sensitivity"],
            "dev_balanced_accuracy_tuned": phase1b_selected["dev_metrics_at_selected_threshold"]["balanced_accuracy"],
            "dev_macro_f1_tuned": phase1b_selected["dev_metrics_at_selected_threshold"]["macro_f1"],
            "dev_sensitivity_tuned": phase1b_selected["dev_metrics_at_selected_threshold"]["positive_recall_sensitivity"],
            "selected_C": phase1b_selected["selected_C"],
            "selected_pca_config": "",
            "selected_dev_threshold": phase1b_selected["selected_dev_threshold"],
            "notes": "Existing Phase 1B locked layer-9 mean baseline; no test metrics included.",
        }
    )
    for row in summary.to_dict(orient="records"):
        method = row["aggregation_method"]
        is_selected = method == evaluation["selected_method"]
        comparison_rows.append(
            {
                "row_name": f"phase1c_{method}",
                "aggregation_method": method,
                "representation_dimension": row["representation_dimension"],
                "pca_used": row["pca_used"],
                "computational_complexity": COMPLEXITY_ORDER[method],
                "train_nested_cv_mean_roc_auc": row["mean_outer_roc_auc"],
                "train_nested_cv_std_roc_auc": row["std_outer_roc_auc"],
                "train_nested_cv_mean_pr_auc": row["mean_outer_pr_auc"],
                "train_nested_cv_mean_balanced_accuracy": row["mean_outer_balanced_accuracy_at_0_5"],
                "train_nested_cv_mean_macro_f1": row["mean_outer_macro_f1_at_0_5"],
                "dev_roc_auc": evaluation["dev_metrics_at_0_5"]["roc_auc"] if is_selected else "",
                "dev_pr_auc": evaluation["dev_metrics_at_0_5"]["pr_auc"] if is_selected else "",
                "dev_balanced_accuracy_at_0_5": evaluation["dev_metrics_at_0_5"]["balanced_accuracy"] if is_selected else "",
                "dev_macro_f1_at_0_5": evaluation["dev_metrics_at_0_5"]["macro_f1"] if is_selected else "",
                "dev_sensitivity_at_0_5": evaluation["dev_metrics_at_0_5"]["positive_recall_sensitivity"] if is_selected else "",
                "dev_balanced_accuracy_tuned": evaluation["dev_metrics_at_selected_threshold"]["balanced_accuracy"] if is_selected else "",
                "dev_macro_f1_tuned": evaluation["dev_metrics_at_selected_threshold"]["macro_f1"] if is_selected else "",
                "dev_sensitivity_tuned": evaluation["dev_metrics_at_selected_threshold"]["positive_recall_sensitivity"] if is_selected else "",
                "selected_C": evaluation["selected_C"] if is_selected else "",
                "selected_pca_config": evaluation["selected_pca_config"] if is_selected else "",
                "selected_dev_threshold": evaluation["selected_dev_threshold"] if is_selected else "",
                "notes": "Selected and evaluated once on dev." if is_selected else "Train-only nested-CV result only; not evaluated on dev.",
            }
        )
    write_csv(run_dir / "aggregation_comparison.csv", comparison_rows)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_phase1c_aggregation_analysis",
        "locked_layer": LOCKED_LAYER_NAME,
        "test_evaluation_performed": False,
        "test_probabilities_generated": False,
        "strict_scope_confirmation": {
            "wavlm_extraction_rerun": False,
            "test_evaluation": False,
            "layer_reselection": False,
            "multi_layer_fusion": False,
            "window_level_classification": False,
            "attention_pooling": False,
            "mlp": False,
            "svm": False,
            "lightgbm": False,
            "wavlm_fine_tuning": False,
            "android_corpus": False,
            "app_code": False,
        },
        "layer_selection_stability": stability_report,
        "aggregation_quality": aggregation_metadata,
        "nested_cv_summary": summary.to_dict(orient="records"),
        "selected_aggregation": selected_report,
        "selected_dev_evaluation": evaluation,
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
        "source_phase1b_run": path_relative_to_root(phase1b_run_dir, root),
        "input_paths": {
            "phase1b_layerwise_window_manifest": path_relative_to_root(phase1b_run_dir / "layerwise_window_embedding_manifest.csv", root),
            "phase1b_outer_fold_results": path_relative_to_root(phase1b_run_dir / "layerwise_outer_fold_results.csv", root),
            "phase1b_outer_predictions": path_relative_to_root(phase1b_run_dir / "layerwise_outer_predictions.csv", root),
            "participant_manifest_clean": path_relative_to_root(output_dir / "daic_phase1_participant_manifest_clean.csv", root),
        },
        "input_hashes": {
            "phase1b_layerwise_window_manifest_sha256": sha256_file(phase1b_run_dir / "layerwise_window_embedding_manifest.csv"),
            "phase1b_outer_fold_results_sha256": sha256_file(phase1b_run_dir / "layerwise_outer_fold_results.csv"),
            "phase1b_outer_predictions_sha256": sha256_file(phase1b_run_dir / "layerwise_outer_predictions.csv"),
            "participant_manifest_clean_sha256": sha256_file(output_dir / "daic_phase1_participant_manifest_clean.csv"),
        },
        "model_name": MODEL_NAME,
        "model_commit": MODEL_COMMIT,
        "locked_layer": LOCKED_LAYER_NAME,
        "random_seed": RANDOM_SEED,
        "outer_nested_cv": {
            "splitter": "RepeatedStratifiedKFold",
            "n_splits": OUTER_SPLITS,
            "n_repeats": OUTER_REPEATS,
            "random_state": RANDOM_SEED,
        },
        "inner_cv": {
            "splitter": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "seed": "RANDOM_SEED * 1000 + outer_fold_index",
            "metric": "ROC-AUC",
        },
        "C_grid": C_GRID,
        "PCA_grid": [str(value) for value in PCA_GRID],
        "selected_aggregation_method": evaluation["selected_method"],
        "selected_C": evaluation["selected_C"],
        "selected_pca_config": evaluation["selected_pca_config"],
        "selected_dev_threshold": evaluation["selected_dev_threshold"],
        "test_evaluation_performed": False,
        "test_labels_loaded_for_modeling": False,
    }
    write_json(run_dir / "run_metadata.json", metadata)

    lines = [
        "# Phase 1C Aggregation Analysis",
        "",
        f"Run directory: `{path_relative_to_root(run_dir, root)}`",
        "",
        "## Layer Stability",
        f"- Layer 9 vs layer 1 mean paired ROC-AUC difference: `{stability_report['paired_comparisons'][0]['mean_paired_outer_fold_roc_auc_difference']:.6f}`",
        f"- Layer 9 vs layer 1 bootstrap median difference: `{stability_report['bootstrap']['median_difference']:.6f}`",
        f"- Bootstrap 95% interval: `{stability_report['bootstrap']['percentile_95_interval'][0]:.6f}` to `{stability_report['bootstrap']['percentile_95_interval'][1]:.6f}`",
        "",
        "## Aggregation Selection",
        f"- Selected method: `{evaluation['selected_method']}`",
        f"- Selected C: `{evaluation['selected_C']}`",
        f"- Selected PCA config: `{evaluation['selected_pca_config'] or 'none'}`",
        f"- Dev ROC-AUC: `{evaluation['dev_metrics_at_0_5']['roc_auc']:.6f}`",
        f"- Dev PR-AUC: `{evaluation['dev_metrics_at_0_5']['pr_auc']:.6f}`",
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
    phase1b_run_dir = resolve_path(args.phase1b_run_dir, root)
    timestamp = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve_path(args.run_root, root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    stability_report = layer_selection_stability(phase1b_run_dir, run_dir)
    aggregation_metadata = validate_and_aggregate_layer9(root, output_dir, phase1b_run_dir, run_dir)
    _, method_train, method_dev, y_train, y_dev, train_frame, dev_frame = load_method_inputs(root, output_dir, run_dir)
    assert all(frame_split != "test" for frame_split in train_frame["official_split"])
    assert all(frame_split != "test" for frame_split in dev_frame["official_split"])
    summary, _, _, _ = nested_cv_aggregation(method_train, y_train, train_frame, run_dir)
    selected_report = select_aggregation(summary, run_dir)
    evaluation = evaluate_selected(root, run_dir, selected_report, method_train, method_dev, y_train, y_dev, dev_frame)
    write_comparison_and_reports(root, output_dir, phase1b_run_dir, run_dir, stability_report, aggregation_metadata, summary, selected_report, evaluation)

    print("Phase 1C aggregation analysis complete.")
    print(f"run_dir={path_relative_to_root(run_dir, root)}")
    print(f"selected_method={evaluation['selected_method']} selected_C={evaluation['selected_C']} selected_pca={evaluation['selected_pca_config'] or 'none'}")
    print(f"dev_roc_auc={evaluation['dev_metrics_at_0_5']['roc_auc']:.6f}")
    print("test_evaluation_performed=false")


if __name__ == "__main__":
    main()
