"""Phase 1E leakage-safe probability-level late fusion.

Compares timing-only, WavLM-only, weighted probability late fusion, and
two-input logistic stacking using only official train participants for model
selection. Dev labels are loaded only after the Phase 1E method and train-only
parameters are frozen. No test probabilities or metrics are produced.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import random
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_participant_baselines import (
    make_issue,
    participant_sort_key,
    path_relative_to_root,
    resolve_path,
    sha256_file,
    write_csv,
    write_json,
)
from run_phase1d_timing_fusion_analysis import (
    TIMING_FEATURES,
    WAVLM_DIM,
    build_pipeline,
    clean,
    csv_cell,
    metric_dict,
    positive_probabilities,
    tune_threshold,
)


PHASE1C_RUN_ID = "20260622_135800"
PHASE1D_RUN_ID = "20260622_142132"
MODEL_NAME = "microsoft/wavlm-base-plus"
MODEL_COMMIT = "4c66d4806a428f2e922ccfa1a962776e232d487b"
LOCKED_LAYER = "layer_09"
LOCKED_AGGREGATION = "mean"
EXPECTED_SPLITS = {"train": 106, "dev": 34, "test": 47}
EXCLUDED_PARTICIPANTS = {"409", "422"}
RANDOM_SEED = 42
BASE_C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
STACKER_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
WEIGHT_GRID = [round(index / 10.0, 1) for index in range(11)]
OUTER_SPLITS = 5
OUTER_REPEATS = 5
FINAL_SPLITS = 5
FINAL_REPEATS = 10
MODEL_ORDER = ["timing_only", "wavlm_only", "weighted_late_fusion", "logistic_stacking"]
COMPLEXITY_ORDER = {"timing_only": 1, "wavlm_only": 2, "weighted_late_fusion": 3, "logistic_stacking": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1E probability-level late fusion.")
    parser.add_argument("--output-dir", default="outputs/phase1_daic")
    parser.add_argument("--phase1c-run-dir", default=f"outputs/phase1_daic/phase1c_aggregation_runs/{PHASE1C_RUN_ID}")
    parser.add_argument("--phase1d-run-dir", default=f"outputs/phase1_daic/phase1d_timing_fusion_runs/{PHASE1D_RUN_ID}")
    parser.add_argument("--run-root", default="outputs/phase1_daic/phase1e_late_fusion_runs")
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
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def sort_participant_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_participant_sort_int"] = result["participant_id"].astype(str).map(lambda value: participant_sort_key(value)[0])
    result["_participant_sort_text"] = result["participant_id"].astype(str)
    result = result.sort_values(["_participant_sort_int", "_participant_sort_text"]).drop(columns=["_participant_sort_int", "_participant_sort_text"])
    return result.reset_index(drop=True)


def build_meta_pipeline(C: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=C, class_weight="balanced", max_iter=10000, random_state=RANDOM_SEED)),
        ]
    )


def transform_meta_probabilities(probabilities: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return probabilities
    if transform == "log_odds":
        clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        return np.log(clipped / (1.0 - clipped))
    raise ValueError(f"Unknown meta transform: {transform}")


def make_meta_frame(p_timing: np.ndarray, p_wavlm: np.ndarray, transform: str) -> pd.DataFrame:
    matrix = np.column_stack([p_timing, p_wavlm]).astype(np.float64)
    transformed = transform_meta_probabilities(matrix, transform)
    return pd.DataFrame(transformed, columns=["timing_probability", "wavlm_probability"])


def selected_positive_probabilities(model: Pipeline, x_data: pd.DataFrame) -> np.ndarray:
    classes = list(model.named_steps["classifier"].classes_)
    return model.predict_proba(x_data)[:, classes.index(1)]


def select_base_c(model_name: str, x_data: pd.DataFrame, y_data: np.ndarray, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows: list[dict[str, Any]] = []
    for C in BASE_C_GRID:
        aucs: list[float] = []
        pr_aucs: list[float] = []
        for train_index, validation_index in splitter.split(x_data, y_data):
            model = build_pipeline(model_name, C, TIMING_FEATURES, [f"wavlm_{index:03d}" for index in range(WAVLM_DIM)])
            model.fit(x_data.iloc[train_index], y_data[train_index])
            probabilities = positive_probabilities(model, x_data.iloc[validation_index])
            aucs.append(float(roc_auc_score(y_data[validation_index], probabilities)))
            pr_aucs.append(float(average_precision_score(y_data[validation_index], probabilities)))
        rows.append(
            {
                "model_name": model_name,
                "C": C,
                "mean_roc_auc": float(np.mean(aucs)),
                "roc_auc_std": float(np.std(aucs, ddof=0)),
                "mean_pr_auc": float(np.mean(pr_aucs)),
                "seed": seed,
            }
        )
    selected = sorted(rows, key=lambda row: (-row["mean_roc_auc"], row["roc_auc_std"], row["C"]))[0]
    return selected, rows


def select_base_c_repeated(model_name: str, x_data: pd.DataFrame, y_data: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    splitter = RepeatedStratifiedKFold(n_splits=FINAL_SPLITS, n_repeats=FINAL_REPEATS, random_state=RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    for C in BASE_C_GRID:
        aucs: list[float] = []
        pr_aucs: list[float] = []
        bal_accs: list[float] = []
        macro_f1s: list[float] = []
        for train_index, validation_index in splitter.split(x_data, y_data):
            model = build_pipeline(model_name, C, TIMING_FEATURES, [f"wavlm_{index:03d}" for index in range(WAVLM_DIM)])
            model.fit(x_data.iloc[train_index], y_data[train_index])
            probabilities = positive_probabilities(model, x_data.iloc[validation_index])
            metrics = metric_dict(y_data[validation_index], probabilities, 0.5)
            aucs.append(metrics["roc_auc"])
            pr_aucs.append(metrics["pr_auc"])
            bal_accs.append(metrics["balanced_accuracy"])
            macro_f1s.append(metrics["macro_f1"])
        rows.append(
            {
                "component": model_name,
                "C": C,
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
    selected = sorted(rows, key=lambda row: (-row["mean_roc_auc"], row["roc_auc_std"], row["C"]))[0]
    return selected, rows


def fit_base_predict(model_name: str, C: float, x_train: pd.DataFrame, y_train: np.ndarray, x_validation: pd.DataFrame) -> tuple[Pipeline, np.ndarray]:
    model = build_pipeline(model_name, C, TIMING_FEATURES, [f"wavlm_{index:03d}" for index in range(WAVLM_DIM)])
    model.fit(x_train, y_train)
    return model, positive_probabilities(model, x_validation)


def generate_inner_oof_base_probabilities(
    timing_train: pd.DataFrame,
    wavlm_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    timing_oof = np.full(len(y_train), np.nan, dtype=np.float64)
    wavlm_oof = np.full(len(y_train), np.nan, dtype=np.float64)
    selected_rows: list[dict[str, Any]] = []
    for inner_fold_index, (inner_train_index, inner_validation_index) in enumerate(splitter.split(timing_train, y_train), start=1):
        inner_seed = seed * 100 + inner_fold_index
        timing_selected, _ = select_base_c("timing_only", timing_train.iloc[inner_train_index], y_train[inner_train_index], inner_seed)
        wavlm_selected, _ = select_base_c("wavlm_only", wavlm_train.iloc[inner_train_index], y_train[inner_train_index], inner_seed)
        _, timing_probabilities = fit_base_predict(
            "timing_only",
            timing_selected["C"],
            timing_train.iloc[inner_train_index],
            y_train[inner_train_index],
            timing_train.iloc[inner_validation_index],
        )
        _, wavlm_probabilities = fit_base_predict(
            "wavlm_only",
            wavlm_selected["C"],
            wavlm_train.iloc[inner_train_index],
            y_train[inner_train_index],
            wavlm_train.iloc[inner_validation_index],
        )
        timing_oof[inner_validation_index] = timing_probabilities
        wavlm_oof[inner_validation_index] = wavlm_probabilities
        selected_rows.append(
            {
                "inner_fold_index": inner_fold_index,
                "inner_seed": inner_seed,
                "timing_inner_selected_C": timing_selected["C"],
                "wavlm_inner_selected_C": wavlm_selected["C"],
            }
        )
    if np.isnan(timing_oof).any() or np.isnan(wavlm_oof).any():
        raise RuntimeError("Inner OOF base probabilities were not generated for every outer-training participant.")
    return timing_oof, wavlm_oof, selected_rows


def select_weight(p_timing: np.ndarray, p_wavlm: np.ndarray, y_train: np.ndarray, outer_fold_index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for weight in WEIGHT_GRID:
        fused = weight * p_timing + (1.0 - weight) * p_wavlm
        rows.append(
            {
                "outer_fold_index": outer_fold_index,
                "weight_timing": weight,
                "roc_auc": float(roc_auc_score(y_train, fused)),
                "pr_auc": float(average_precision_score(y_train, fused)),
                "distance_to_0_5": abs(weight - 0.5),
            }
        )
    selected = sorted(rows, key=lambda row: (-row["roc_auc"], -row["pr_auc"], row["distance_to_0_5"], row["weight_timing"]))[0]
    selected = {key: value for key, value in selected.items() if key != "distance_to_0_5"}
    return selected, rows


def select_stacker(
    p_timing: np.ndarray,
    p_wavlm: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    outer_fold_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], Pipeline]:
    rows: list[dict[str, Any]] = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 777)
    for transform in ("raw", "log_odds"):
        meta = make_meta_frame(p_timing, p_wavlm, transform)
        for C in STACKER_C_GRID:
            aucs: list[float] = []
            pr_aucs: list[float] = []
            for train_index, validation_index in splitter.split(meta, y_train):
                model = build_meta_pipeline(C)
                model.fit(meta.iloc[train_index], y_train[train_index])
                probabilities = selected_positive_probabilities(model, meta.iloc[validation_index])
                aucs.append(float(roc_auc_score(y_train[validation_index], probabilities)))
                pr_aucs.append(float(average_precision_score(y_train[validation_index], probabilities)))
            rows.append(
                {
                    "outer_fold_index": outer_fold_index,
                    "transform": transform,
                    "stacker_C": C,
                    "mean_roc_auc": float(np.mean(aucs)),
                    "roc_auc_std": float(np.std(aucs, ddof=0)),
                    "mean_pr_auc": float(np.mean(pr_aucs)),
                    "seed": seed + 777,
                }
            )
    transform_order = {"raw": 0, "log_odds": 1}
    selected = sorted(
        rows,
        key=lambda row: (-row["mean_roc_auc"], row["roc_auc_std"], -row["mean_pr_auc"], transform_order[row["transform"]], row["stacker_C"]),
    )[0]
    final_meta = make_meta_frame(p_timing, p_wavlm, selected["transform"])
    final_model = build_meta_pipeline(selected["stacker_C"])
    final_model.fit(final_meta, y_train)
    return selected, rows, final_model


def stacker_coefficients(model: Pipeline, outer_fold_index: int, transform: str, stacker_C: float) -> dict[str, Any]:
    classifier = model.named_steps["classifier"]
    coef = classifier.coef_[0]
    return {
        "outer_fold_index": outer_fold_index,
        "transform": transform,
        "stacker_C": stacker_C,
        "timing_coefficient": float(coef[0]),
        "wavlm_coefficient": float(coef[1]),
        "intercept": float(classifier.intercept_[0]),
    }


def read_train_labels_only(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with (output_dir / "daic_phase1_participant_manifest_clean.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = {name: column_index for column_index, name in enumerate(header)}
        required = ["participant_id", "official_split", "label_binary", "has_label"]
        if not all(name in index for name in required):
            raise RuntimeError(f"Participant manifest missing required columns: {required}")
        for row in reader:
            split = csv_cell(row, index["official_split"])
            if split == "train":
                if csv_cell(row, index["has_label"]) != "1" or csv_cell(row, index["label_binary"]) == "":
                    raise RuntimeError(f"Missing train label for participant {csv_cell(row, index['participant_id'])}")
                rows.append({"participant_id": csv_cell(row, index["participant_id"]), "label_binary": int(csv_cell(row, index["label_binary"]))})
    return pd.DataFrame(rows)


def read_split_labels(output_dir: Path, split_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with (output_dir / "daic_phase1_participant_manifest_clean.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = {name: column_index for column_index, name in enumerate(header)}
        required = ["participant_id", "official_split", "label_binary", "has_label"]
        if not all(name in index for name in required):
            raise RuntimeError(f"Participant manifest missing required columns: {required}")
        for row in reader:
            split = csv_cell(row, index["official_split"])
            if split == split_name:
                if csv_cell(row, index["has_label"]) != "1" or csv_cell(row, index["label_binary"]) == "":
                    raise RuntimeError(f"Missing {split_name} label for participant {csv_cell(row, index['participant_id'])}")
                rows.append({"participant_id": csv_cell(row, index["participant_id"]), "label_binary": int(csv_cell(row, index["label_binary"]))})
    return pd.DataFrame(rows)


def validate_inputs(root: Path, output_dir: Path, phase1c_run_dir: Path, phase1d_run_dir: Path, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    timing = pd.read_csv(phase1d_run_dir / "speech_timing_features.csv", dtype={"participant_id": str})
    phase1c_manifest = pd.read_csv(phase1c_run_dir / "aggregation_manifest.csv", dtype={"participant_id": str})
    wavlm_manifest = phase1c_manifest[
        phase1c_manifest["aggregation_method"].eq(LOCKED_AGGREGATION)
        & phase1c_manifest["aggregation_status"].eq("success")
    ].copy()
    issues: list[dict[str, Any]] = []
    split_counts = timing["official_split"].value_counts().to_dict()
    if split_counts != EXPECTED_SPLITS:
        issues.append(make_issue("error", "timing_split_count_mismatch", f"Expected {EXPECTED_SPLITS}, found {split_counts}."))
    if wavlm_manifest["official_split"].value_counts().to_dict() != EXPECTED_SPLITS:
        issues.append(make_issue("error", "wavlm_split_count_mismatch", f"Expected {EXPECTED_SPLITS}, found {wavlm_manifest['official_split'].value_counts().to_dict()}."))
    if timing["participant_id"].duplicated().any() or wavlm_manifest["participant_id"].duplicated().any():
        issues.append(make_issue("error", "duplicate_participants", "Duplicate participant IDs in timing or WavLM table."))
    excluded = sorted((set(timing["participant_id"]) | set(wavlm_manifest["participant_id"])) & EXCLUDED_PARTICIPANTS, key=participant_sort_key)
    if excluded:
        issues.append(make_issue("error", "excluded_participants_present", f"Excluded participants present: {excluded}."))
    if set(timing["participant_id"]) != set(wavlm_manifest["participant_id"]):
        issues.append(make_issue("error", "participant_alignment_mismatch", "Timing and WavLM participant IDs do not align exactly."))

    has_label_missing: list[str] = []
    with (output_dir / "daic_phase1_participant_manifest_clean.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = {name: column_index for column_index, name in enumerate(header)}
        for row in reader:
            split = csv_cell(row, index["official_split"])
            if split in {"train", "dev"} and csv_cell(row, index["has_label"]) != "1":
                has_label_missing.append(csv_cell(row, index["participant_id"]))
    if has_label_missing:
        issues.append(make_issue("error", "missing_train_or_dev_labels", f"Missing train/dev labels: {has_label_missing}."))

    wavlm_vectors: dict[str, np.ndarray] = {}
    bad_wavlm: list[str] = []
    for row in wavlm_manifest.itertuples(index=False):
        path = resolve_path(row.representation_path, root)
        if not path.exists():
            bad_wavlm.append(str(row.participant_id))
            continue
        vector = np.load(path).astype(np.float32, copy=False)
        if vector.shape != (WAVLM_DIM,) or not np.isfinite(vector).all() or np.all(vector == 0):
            bad_wavlm.append(str(row.participant_id))
            continue
        wavlm_vectors[str(row.participant_id)] = vector
    if bad_wavlm:
        issues.append(make_issue("error", "invalid_wavlm_representations", f"Invalid WavLM representations: {bad_wavlm[:10]}."))

    timing_numeric = timing[TIMING_FEATURES].apply(pd.to_numeric, errors="coerce")
    if np.isinf(timing_numeric.to_numpy(dtype=np.float64)).any():
        issues.append(make_issue("error", "infinite_timing_features", "Timing features contain infinite values."))

    quality_rows = issues or [make_issue("info", "all_phase1e_input_checks_passed", "All Phase 1E input checks passed.")]
    write_csv(run_dir / "phase1e_input_validation_report.csv", quality_rows)
    if issues:
        raise RuntimeError("Phase 1E input validation failed.")

    merged = sort_participant_frame(
        timing.merge(
            wavlm_manifest[["participant_id", "official_split", "representation_path"]],
            on=["participant_id", "official_split"],
            how="inner",
            validate="one_to_one",
        )
    )
    labels = read_train_labels_only(output_dir)
    train_meta = sort_participant_frame(
        merged[merged["official_split"].eq("train")]
        .merge(labels, on="participant_id", how="inner", validate="one_to_one")
        .reset_index(drop=True)
    )
    assert train_meta["official_split"].eq("train").all(), "Model-selection rows are not all official train rows."
    if len(train_meta) != EXPECTED_SPLITS["train"]:
        raise RuntimeError(f"Expected 106 train participants, found {len(train_meta)}.")

    timing_train = train_meta[TIMING_FEATURES].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    wavlm_train = pd.DataFrame(
        np.vstack([wavlm_vectors[str(participant_id)] for participant_id in train_meta["participant_id"]]),
        columns=[f"wavlm_{index:03d}" for index in range(WAVLM_DIM)],
    )
    metadata = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "participant_counts_by_split": split_counts,
        "timing_source_path": path_relative_to_root(phase1d_run_dir / "speech_timing_features.csv", root),
        "wavlm_source_manifest": path_relative_to_root(phase1c_run_dir / "aggregation_manifest.csv", root),
        "timing_feature_count": len(TIMING_FEATURES),
        "wavlm_dimension": WAVLM_DIM,
        "test_evaluation_performed": False,
        "test_labels_loaded_for_modeling": False,
    }
    write_json(run_dir / "phase1e_input_validation_metadata.json", metadata)
    return train_meta[["participant_id", "official_split", "label_binary"]], timing_train, wavlm_train, merged, pd.DataFrame.from_dict(wavlm_vectors, orient="index"), metadata


def load_dev_after_selection(root: Path, output_dir: Path, merged_nonlabel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    dev_labels = read_split_labels(output_dir, "dev")
    dev_meta = sort_participant_frame(
        merged_nonlabel[merged_nonlabel["official_split"].eq("dev")]
        .merge(dev_labels, on="participant_id", how="inner", validate="one_to_one")
        .reset_index(drop=True)
    )
    assert dev_meta["official_split"].eq("dev").all(), "Dev evaluation rows are not all official dev rows."
    timing_dev = dev_meta[TIMING_FEATURES].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    wavlm_vectors: list[np.ndarray] = []
    for row in dev_meta.itertuples(index=False):
        vector = np.load(resolve_path(row.representation_path, root)).astype(np.float32, copy=False)
        wavlm_vectors.append(vector)
    wavlm_dev = pd.DataFrame(np.vstack(wavlm_vectors), columns=[f"wavlm_{index:03d}" for index in range(WAVLM_DIM)])
    y_dev = dev_meta["label_binary"].to_numpy(dtype=int)
    return dev_meta[["participant_id", "official_split", "label_binary"]], timing_dev, wavlm_dev, y_dev


def run_nested_outer_cv(
    train_meta: pd.DataFrame,
    timing_train: pd.DataFrame,
    wavlm_train: pd.DataFrame,
    y_train: np.ndarray,
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    stacker_rows: list[dict[str, Any]] = []
    weight_search_rows: list[dict[str, Any]] = []
    stacker_search_rows: list[dict[str, Any]] = []
    inner_base_rows: list[dict[str, Any]] = []
    splitter = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=RANDOM_SEED)
    for outer_fold_index, (outer_train_index, outer_validation_index) in enumerate(splitter.split(timing_train, y_train), start=1):
        outer_seed = RANDOM_SEED * 1000 + outer_fold_index
        repeat_index = int((outer_fold_index - 1) // OUTER_SPLITS + 1)
        fold_number = int((outer_fold_index - 1) % OUTER_SPLITS + 1)
        y_outer_train = y_train[outer_train_index]
        y_outer_validation = y_train[outer_validation_index]
        timing_selected, _ = select_base_c("timing_only", timing_train.iloc[outer_train_index], y_outer_train, outer_seed)
        wavlm_selected, _ = select_base_c("wavlm_only", wavlm_train.iloc[outer_train_index], y_outer_train, outer_seed)
        _, timing_validation_prob = fit_base_predict(
            "timing_only",
            timing_selected["C"],
            timing_train.iloc[outer_train_index],
            y_outer_train,
            timing_train.iloc[outer_validation_index],
        )
        _, wavlm_validation_prob = fit_base_predict(
            "wavlm_only",
            wavlm_selected["C"],
            wavlm_train.iloc[outer_train_index],
            y_outer_train,
            wavlm_train.iloc[outer_validation_index],
        )

        timing_oof, wavlm_oof, inner_selected = generate_inner_oof_base_probabilities(
            timing_train.iloc[outer_train_index].reset_index(drop=True),
            wavlm_train.iloc[outer_train_index].reset_index(drop=True),
            y_outer_train,
            outer_seed,
        )
        for row in inner_selected:
            inner_base_rows.append({**row, "outer_fold_index": outer_fold_index, "repeat_index": repeat_index, "fold_number": fold_number})
        selected_weight, weight_rows = select_weight(timing_oof, wavlm_oof, y_outer_train, outer_fold_index)
        weight_search_rows.extend(weight_rows)
        selected_stacker, stacker_search, stacker_model = select_stacker(timing_oof, wavlm_oof, y_outer_train, outer_seed, outer_fold_index)
        stacker_search_rows.extend(stacker_search)
        stacker_rows.append(stacker_coefficients(stacker_model, outer_fold_index, selected_stacker["transform"], selected_stacker["stacker_C"]))

        weighted_validation_prob = selected_weight["weight_timing"] * timing_validation_prob + (1.0 - selected_weight["weight_timing"]) * wavlm_validation_prob
        stacker_meta_validation = make_meta_frame(timing_validation_prob, wavlm_validation_prob, selected_stacker["transform"])
        stacker_validation_prob = selected_positive_probabilities(stacker_model, stacker_meta_validation)
        method_probabilities = {
            "timing_only": timing_validation_prob,
            "wavlm_only": wavlm_validation_prob,
            "weighted_late_fusion": weighted_validation_prob,
            "logistic_stacking": stacker_validation_prob,
        }
        selected_rows.append(
            {
                "outer_fold_index": outer_fold_index,
                "repeat_index": repeat_index,
                "fold_number": fold_number,
                "timing_selected_C": timing_selected["C"],
                "wavlm_selected_C": wavlm_selected["C"],
                "selected_weight_timing": selected_weight["weight_timing"],
                "selected_weight_roc_auc": selected_weight["roc_auc"],
                "selected_weight_pr_auc": selected_weight["pr_auc"],
                "selected_stacker_transform": selected_stacker["transform"],
                "selected_stacker_C": selected_stacker["stacker_C"],
                "selected_stacker_inner_mean_roc_auc": selected_stacker["mean_roc_auc"],
                "selected_stacker_inner_roc_auc_std": selected_stacker["roc_auc_std"],
            }
        )
        for method_name, probabilities in method_probabilities.items():
            metrics = metric_dict(y_outer_validation, probabilities, 0.5)
            outer_rows.append(
                {
                    "model_name": method_name,
                    "outer_fold_index": outer_fold_index,
                    "repeat_index": repeat_index,
                    "fold_number": fold_number,
                    "timing_selected_C": timing_selected["C"],
                    "wavlm_selected_C": wavlm_selected["C"],
                    "selected_weight_timing": selected_weight["weight_timing"] if method_name == "weighted_late_fusion" else "",
                    "selected_stacker_transform": selected_stacker["transform"] if method_name == "logistic_stacking" else "",
                    "selected_stacker_C": selected_stacker["stacker_C"] if method_name == "logistic_stacking" else "",
                    **metrics,
                    "n_outer_train": int(len(outer_train_index)),
                    "n_outer_validation": int(len(outer_validation_index)),
                }
            )
            for participant_position, probability in zip(outer_validation_index, probabilities):
                participant = train_meta.iloc[int(participant_position)]
                prediction_rows.append(
                    {
                        "model_name": method_name,
                        "outer_fold_index": outer_fold_index,
                        "repeat_index": repeat_index,
                        "fold_number": fold_number,
                        "participant_id": participant["participant_id"],
                        "official_split": participant["official_split"],
                        "label_binary": int(y_train[int(participant_position)]),
                        "predicted_probability": float(probability),
                        "prediction_at_0_5": int(probability >= 0.5),
                        "timing_selected_C": timing_selected["C"],
                        "wavlm_selected_C": wavlm_selected["C"],
                        "selected_weight_timing": selected_weight["weight_timing"] if method_name == "weighted_late_fusion" else "",
                        "selected_stacker_transform": selected_stacker["transform"] if method_name == "logistic_stacking" else "",
                        "selected_stacker_C": selected_stacker["stacker_C"] if method_name == "logistic_stacking" else "",
                    }
                )
        if outer_fold_index % 5 == 0:
            print(f"late_fusion_outer_progress={outer_fold_index}/25", flush=True)

    outer = pd.DataFrame(outer_rows)
    predictions = pd.DataFrame(prediction_rows)
    selected = pd.DataFrame(selected_rows)
    stacker = pd.DataFrame(stacker_rows)
    summary_rows: list[dict[str, Any]] = []
    for method_name in MODEL_ORDER:
        rows = outer[outer["model_name"].eq(method_name)]
        aucs = rows["roc_auc"].to_numpy(dtype=float)
        pr_aucs = rows["pr_auc"].to_numpy(dtype=float)
        bal_accs = rows["balanced_accuracy"].to_numpy(dtype=float)
        macro_f1s = rows["macro_f1"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "model_name": method_name,
                "complexity_order": COMPLEXITY_ORDER[method_name],
                "representation_dimension": 28
                if method_name == "timing_only"
                else (WAVLM_DIM if method_name == "wavlm_only" else 2),
                "mean_outer_roc_auc": float(aucs.mean()),
                "std_outer_roc_auc": float(aucs.std(ddof=0)),
                "median_outer_roc_auc": float(np.median(aucs)),
                "min_outer_roc_auc": float(aucs.min()),
                "max_outer_roc_auc": float(aucs.max()),
                "mean_outer_pr_auc": float(pr_aucs.mean()),
                "mean_outer_balanced_accuracy_at_0_5": float(bal_accs.mean()),
                "mean_outer_macro_f1_at_0_5": float(macro_f1s.mean()),
                "selected_timing_C_frequency": json.dumps(dict(sorted(Counter(rows["timing_selected_C"].astype(str)).items()))),
                "selected_wavlm_C_frequency": json.dumps(dict(sorted(Counter(rows["wavlm_selected_C"].astype(str)).items()))),
                "selected_weight_frequency": json.dumps(dict(sorted(Counter(rows["selected_weight_timing"].astype(str)).items()))) if method_name == "weighted_late_fusion" else "",
                "selected_stacker_transform_frequency": json.dumps(dict(sorted(Counter(rows["selected_stacker_transform"].astype(str)).items()))) if method_name == "logistic_stacking" else "",
                "selected_stacker_C_frequency": json.dumps(dict(sorted(Counter(rows["selected_stacker_C"].astype(str)).items()))) if method_name == "logistic_stacking" else "",
                "n_outer_evaluations": int(len(rows)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("complexity_order")
    summary.to_csv(run_dir / "late_fusion_nested_cv_summary.csv", index=False)
    outer.to_csv(run_dir / "late_fusion_outer_fold_results.csv", index=False)
    predictions.to_csv(run_dir / "late_fusion_outer_predictions.csv", index=False)
    selected.to_csv(run_dir / "late_fusion_selected_parameters_by_fold.csv", index=False)
    stacker.to_csv(run_dir / "stacker_coefficients_by_fold.csv", index=False)
    pd.DataFrame(weight_search_rows).to_csv(run_dir / "weighted_fusion_weight_search_by_fold.csv", index=False)
    pd.DataFrame(stacker_search_rows).to_csv(run_dir / "stacker_config_search_by_fold.csv", index=False)
    pd.DataFrame(inner_base_rows).to_csv(run_dir / "inner_base_model_selected_c_by_fold.csv", index=False)
    return summary, outer, predictions, selected, stacker


def write_stability_report(selected: pd.DataFrame, stacker: pd.DataFrame, run_dir: Path) -> dict[str, Any]:
    weights = selected["selected_weight_timing"].to_numpy(dtype=float)
    weight_counts = Counter(selected["selected_weight_timing"].astype(str))
    timing_coef = stacker["timing_coefficient"].to_numpy(dtype=float)
    wavlm_coef = stacker["wavlm_coefficient"].to_numpy(dtype=float)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weighted_fusion": {
            "selected_weight_frequency": dict(sorted(weight_counts.items())),
            "mean_selected_timing_weight": float(np.mean(weights)),
            "std_selected_timing_weight": float(np.std(weights, ddof=0)),
            "percent_weight_0_wavlm_only": float(np.mean(weights == 0.0) * 100.0),
            "percent_weight_1_timing_only": float(np.mean(weights == 1.0) * 100.0),
            "percent_true_mixed_weight": float(np.mean((weights > 0.0) & (weights < 1.0)) * 100.0),
            "interpretation_rule": "If fusion frequently chooses weight 0 or 1, evidence for complementarity is weak.",
        },
        "logistic_stacking": {
            "timing_coefficient_mean": float(np.mean(timing_coef)),
            "timing_coefficient_std": float(np.std(timing_coef, ddof=0)),
            "timing_coefficient_min": float(np.min(timing_coef)),
            "timing_coefficient_max": float(np.max(timing_coef)),
            "wavlm_coefficient_mean": float(np.mean(wavlm_coef)),
            "wavlm_coefficient_std": float(np.std(wavlm_coef, ddof=0)),
            "wavlm_coefficient_min": float(np.min(wavlm_coef)),
            "wavlm_coefficient_max": float(np.max(wavlm_coef)),
            "percent_timing_coefficient_positive": float(np.mean(timing_coef > 0.0) * 100.0),
            "percent_wavlm_coefficient_positive": float(np.mean(wavlm_coef > 0.0) * 100.0),
            "timing_sign_consistent": bool(np.all(timing_coef > 0.0) or np.all(timing_coef < 0.0)),
            "wavlm_sign_consistent": bool(np.all(wavlm_coef > 0.0) or np.all(wavlm_coef < 0.0)),
            "interpretation_rule": "If stacker coefficient signs are unstable, do not claim a reliable fusion effect.",
        },
    }
    write_json(run_dir / "fusion_stability_report.json", report)
    stability_rows = [
        {"section": "weighted_fusion", "metric": key, "value": value}
        for key, value in report["weighted_fusion"].items()
        if key != "selected_weight_frequency"
    ]
    stability_rows.extend(
        {"section": "weighted_fusion", "metric": f"weight_frequency_{key}", "value": value}
        for key, value in report["weighted_fusion"]["selected_weight_frequency"].items()
    )
    stability_rows.extend(
        {"section": "logistic_stacking", "metric": key, "value": value}
        for key, value in report["logistic_stacking"].items()
    )
    write_csv(run_dir / "fusion_stability_report.csv", stability_rows)
    return report


def select_phase1e_model(summary: pd.DataFrame, run_dir: Path) -> dict[str, Any]:
    sorted_summary = summary.sort_values(
        ["mean_outer_roc_auc", "std_outer_roc_auc", "mean_outer_pr_auc", "complexity_order"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    nominal_best = sorted_summary.iloc[0].to_dict()
    timing = summary[summary["model_name"].eq("timing_only")].iloc[0].to_dict()
    wavlm = summary[summary["model_name"].eq("wavlm_only")].iloc[0].to_dict()
    better_individual = timing if timing["mean_outer_roc_auc"] >= wavlm["mean_outer_roc_auc"] else wavlm
    selected = nominal_best
    practical_rule_applied = False
    fusion_models = {"weighted_late_fusion", "logistic_stacking"}
    if nominal_best["model_name"] in fusion_models:
        improvement = float(nominal_best["mean_outer_roc_auc"] - better_individual["mean_outer_roc_auc"])
        std_reduction = float(better_individual["std_outer_roc_auc"] - nominal_best["std_outer_roc_auc"])
        if improvement < 0.01 and std_reduction < 0.01:
            selected = better_individual
            practical_rule_applied = True
    else:
        improvement = float(summary[summary["model_name"].isin(fusion_models)]["mean_outer_roc_auc"].max() - better_individual["mean_outer_roc_auc"])
        std_reduction = 0.0
    timing_matches_wavlm = abs(float(timing["mean_outer_roc_auc"] - wavlm["mean_outer_roc_auc"])) < 0.01
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected["model_name"],
        "nominal_best_model": nominal_best["model_name"],
        "selection_criteria": [
            "highest mean outer ROC-AUC",
            "lowest ROC-AUC standard deviation",
            "highest mean outer PR-AUC",
            "simpler method if effectively tied",
            "fusion must improve over the better individual model by at least 0.01 unless it materially lowers variability",
        ],
        "complexity_order": COMPLEXITY_ORDER,
        "better_individual_model": better_individual["model_name"],
        "best_fusion_improvement_over_better_individual": improvement,
        "fusion_std_reduction_vs_better_individual": std_reduction,
        "fusion_practical_improvement_rule_applied": practical_rule_applied,
        "timing_matches_wavlm_within_0_01": timing_matches_wavlm,
        "timing_confounding_warning": "Timing-only matches WavLM within 0.01 mean ROC-AUC; possible dataset-timing confounding should be considered."
        if timing_matches_wavlm
        else "",
        "selected_model_summary": selected,
        "test_data_used_for_selection": False,
    }
    write_json(run_dir / "selected_phase1e_model.json", report)
    return report


def final_repeated_oof_base_probabilities(
    timing_train: pd.DataFrame,
    wavlm_train: pd.DataFrame,
    y_train: np.ndarray,
    train_meta: pd.DataFrame,
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    splitter = RepeatedStratifiedKFold(n_splits=FINAL_SPLITS, n_repeats=FINAL_REPEATS, random_state=RANDOM_SEED)
    timing_sum = np.zeros(len(y_train), dtype=np.float64)
    wavlm_sum = np.zeros(len(y_train), dtype=np.float64)
    counts = np.zeros(len(y_train), dtype=np.int32)
    selected_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for fold_index, (train_index, validation_index) in enumerate(splitter.split(timing_train, y_train), start=1):
        seed = RANDOM_SEED * 10000 + fold_index
        timing_selected, _ = select_base_c("timing_only", timing_train.iloc[train_index], y_train[train_index], seed)
        wavlm_selected, _ = select_base_c("wavlm_only", wavlm_train.iloc[train_index], y_train[train_index], seed)
        _, timing_prob = fit_base_predict("timing_only", timing_selected["C"], timing_train.iloc[train_index], y_train[train_index], timing_train.iloc[validation_index])
        _, wavlm_prob = fit_base_predict("wavlm_only", wavlm_selected["C"], wavlm_train.iloc[train_index], y_train[train_index], wavlm_train.iloc[validation_index])
        timing_sum[validation_index] += timing_prob
        wavlm_sum[validation_index] += wavlm_prob
        counts[validation_index] += 1
        selected_rows.append(
            {
                "fold_index": fold_index,
                "repeat_index": int((fold_index - 1) // FINAL_SPLITS + 1),
                "fold_number": int((fold_index - 1) % FINAL_SPLITS + 1),
                "timing_selected_C": timing_selected["C"],
                "wavlm_selected_C": wavlm_selected["C"],
            }
        )
        for participant_position, timing_probability, wavlm_probability in zip(validation_index, timing_prob, wavlm_prob):
            participant = train_meta.iloc[int(participant_position)]
            prediction_rows.append(
                {
                    "fold_index": fold_index,
                    "participant_id": participant["participant_id"],
                    "label_binary": int(y_train[int(participant_position)]),
                    "timing_probability": float(timing_probability),
                    "wavlm_probability": float(wavlm_probability),
                    "timing_selected_C": timing_selected["C"],
                    "wavlm_selected_C": wavlm_selected["C"],
                }
            )
    if np.any(counts == 0):
        raise RuntimeError("Final repeated OOF probabilities missing at least one participant.")
    write_csv(run_dir / "final_train_oof_base_predictions_by_fold.csv", prediction_rows)
    write_csv(run_dir / "final_train_oof_base_selected_c_by_fold.csv", selected_rows)
    return timing_sum / counts, wavlm_sum / counts, selected_rows


def final_train_selection_and_fit(
    root: Path,
    run_dir: Path,
    selected_report: dict[str, Any],
    train_meta: pd.DataFrame,
    timing_train: pd.DataFrame,
    wavlm_train: pd.DataFrame,
    y_train: np.ndarray,
) -> dict[str, Any]:
    selected_model = selected_report["selected_model"]
    search_rows: list[dict[str, Any]] = []
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"selected_model": selected_model}
    if selected_model in {"timing_only", "weighted_late_fusion", "logistic_stacking"}:
        timing_selected, timing_rows = select_base_c_repeated("timing_only", timing_train, y_train)
        search_rows.extend(timing_rows)
        result["timing_selected_C"] = timing_selected["C"]
        timing_model, _ = fit_base_predict("timing_only", timing_selected["C"], timing_train, y_train, timing_train)
        joblib.dump(timing_model, model_dir / "timing_base_pipeline.joblib")
        result["timing_model_path"] = path_relative_to_root(model_dir / "timing_base_pipeline.joblib", root)
    if selected_model in {"wavlm_only", "weighted_late_fusion", "logistic_stacking"}:
        wavlm_selected, wavlm_rows = select_base_c_repeated("wavlm_only", wavlm_train, y_train)
        search_rows.extend(wavlm_rows)
        result["wavlm_selected_C"] = wavlm_selected["C"]
        wavlm_model, _ = fit_base_predict("wavlm_only", wavlm_selected["C"], wavlm_train, y_train, wavlm_train)
        joblib.dump(wavlm_model, model_dir / "wavlm_base_pipeline.joblib")
        result["wavlm_model_path"] = path_relative_to_root(model_dir / "wavlm_base_pipeline.joblib", root)

    if selected_model == "weighted_late_fusion":
        timing_oof, wavlm_oof, _ = final_repeated_oof_base_probabilities(timing_train, wavlm_train, y_train, train_meta, run_dir)
        weight_selected, weight_rows = select_weight(timing_oof, wavlm_oof, y_train, outer_fold_index=0)
        for row in weight_rows:
            search_rows.append({"component": "weighted_late_fusion", **row})
        result["selected_weight_timing"] = weight_selected["weight_timing"]
        train_fused = weight_selected["weight_timing"] * timing_oof + (1.0 - weight_selected["weight_timing"]) * wavlm_oof
        write_csv(
            run_dir / "final_train_oof_selected_model_predictions.csv",
            [
                {
                    "participant_id": participant.participant_id,
                    "label_binary": int(participant.label_binary),
                    "timing_oof_probability": float(pt),
                    "wavlm_oof_probability": float(pw),
                    "selected_model_oof_probability": float(pf),
                }
                for participant, pt, pw, pf in zip(train_meta.itertuples(index=False), timing_oof, wavlm_oof, train_fused)
            ],
        )
        write_json(model_dir / "weighted_fusion_component.json", {"weight_timing": weight_selected["weight_timing"]})
        result["fusion_component_path"] = path_relative_to_root(model_dir / "weighted_fusion_component.json", root)
    elif selected_model == "logistic_stacking":
        timing_oof, wavlm_oof, _ = final_repeated_oof_base_probabilities(timing_train, wavlm_train, y_train, train_meta, run_dir)
        stacker_selected, stacker_rows, stacker_model = select_stacker(timing_oof, wavlm_oof, y_train, RANDOM_SEED, outer_fold_index=0)
        for row in stacker_rows:
            search_rows.append({"component": "logistic_stacking", **row})
        joblib.dump(stacker_model, model_dir / "stacker_pipeline.joblib")
        result["stacker_transform"] = stacker_selected["transform"]
        result["stacker_C"] = stacker_selected["stacker_C"]
        result["stacker_model_path"] = path_relative_to_root(model_dir / "stacker_pipeline.joblib", root)
        meta = make_meta_frame(timing_oof, wavlm_oof, stacker_selected["transform"])
        stacked = selected_positive_probabilities(stacker_model, meta)
        write_csv(
            run_dir / "final_train_oof_selected_model_predictions.csv",
            [
                {
                    "participant_id": participant.participant_id,
                    "label_binary": int(participant.label_binary),
                    "timing_oof_probability": float(pt),
                    "wavlm_oof_probability": float(pw),
                    "selected_model_oof_probability": float(ps),
                }
                for participant, pt, pw, ps in zip(train_meta.itertuples(index=False), timing_oof, wavlm_oof, stacked)
            ],
        )
    else:
        model_name = selected_model
        selected_C = result["timing_selected_C"] if model_name == "timing_only" else result["wavlm_selected_C"]
        selected_rows = [row for row in search_rows if row.get("component") == model_name and row.get("C") == selected_C]
        result["selected_C"] = selected_C
        result["selected_cv_mean_roc_auc"] = selected_rows[0]["mean_roc_auc"] if selected_rows else ""
    write_csv(run_dir / "final_train_parameter_search.csv", search_rows)
    write_json(run_dir / "final_selected_parameters.json", result)
    return result


def evaluate_dev(
    root: Path,
    run_dir: Path,
    final_params: dict[str, Any],
    dev_meta: pd.DataFrame,
    timing_dev: pd.DataFrame,
    wavlm_dev: pd.DataFrame,
    y_dev: np.ndarray,
) -> dict[str, Any]:
    model_dir = run_dir / "models"
    selected_model = final_params["selected_model"]
    if selected_model == "timing_only":
        model = joblib.load(model_dir / "timing_base_pipeline.joblib")
        probabilities = positive_probabilities(model, timing_dev)
        selected_hyperparameters = {"timing_C": final_params["timing_selected_C"]}
    elif selected_model == "wavlm_only":
        model = joblib.load(model_dir / "wavlm_base_pipeline.joblib")
        probabilities = positive_probabilities(model, wavlm_dev)
        selected_hyperparameters = {"wavlm_C": final_params["wavlm_selected_C"]}
    elif selected_model == "weighted_late_fusion":
        timing_model = joblib.load(model_dir / "timing_base_pipeline.joblib")
        wavlm_model = joblib.load(model_dir / "wavlm_base_pipeline.joblib")
        component = json.loads((model_dir / "weighted_fusion_component.json").read_text(encoding="utf-8"))
        timing_prob = positive_probabilities(timing_model, timing_dev)
        wavlm_prob = positive_probabilities(wavlm_model, wavlm_dev)
        probabilities = component["weight_timing"] * timing_prob + (1.0 - component["weight_timing"]) * wavlm_prob
        selected_hyperparameters = {
            "timing_C": final_params["timing_selected_C"],
            "wavlm_C": final_params["wavlm_selected_C"],
            "weight_timing": component["weight_timing"],
        }
    elif selected_model == "logistic_stacking":
        timing_model = joblib.load(model_dir / "timing_base_pipeline.joblib")
        wavlm_model = joblib.load(model_dir / "wavlm_base_pipeline.joblib")
        stacker_model = joblib.load(model_dir / "stacker_pipeline.joblib")
        timing_prob = positive_probabilities(timing_model, timing_dev)
        wavlm_prob = positive_probabilities(wavlm_model, wavlm_dev)
        meta = make_meta_frame(timing_prob, wavlm_prob, final_params["stacker_transform"])
        probabilities = selected_positive_probabilities(stacker_model, meta)
        selected_hyperparameters = {
            "timing_C": final_params["timing_selected_C"],
            "wavlm_C": final_params["wavlm_selected_C"],
            "stacker_C": final_params["stacker_C"],
            "stacker_transform": final_params["stacker_transform"],
        }
    else:
        raise ValueError(f"Unknown selected model {selected_model}")

    threshold_search, selected_threshold = tune_threshold(y_dev, probabilities)
    threshold_search.to_csv(run_dir / "dev_threshold_search.csv", index=False)
    metrics_0_5 = metric_dict(y_dev, probabilities, 0.5)
    metrics_tuned = metric_dict(y_dev, probabilities, selected_threshold)
    prediction_rows = []
    for participant, probability in zip(dev_meta.itertuples(index=False), probabilities):
        prediction_rows.append(
            {
                "participant_id": participant.participant_id,
                "official_split": participant.official_split,
                "label_binary": int(participant.label_binary),
                "predicted_probability": float(probability),
                "prediction_at_0_5": int(probability >= 0.5),
                "prediction_at_tuned_threshold": int(probability >= selected_threshold),
                "tuned_threshold": selected_threshold,
                "model_name": selected_model,
            }
        )
    write_csv(run_dir / "selected_phase1e_dev_predictions.csv", prediction_rows)
    return {
        "selected_model": selected_model,
        "selected_hyperparameters": selected_hyperparameters,
        "selected_dev_threshold": selected_threshold,
        "dev_metrics_at_0_5": metrics_0_5,
        "dev_metrics_at_selected_threshold": metrics_tuned,
        "test_evaluation_performed": False,
    }


def write_reports(
    root: Path,
    output_dir: Path,
    phase1c_run_dir: Path,
    phase1d_run_dir: Path,
    run_dir: Path,
    input_metadata: dict[str, Any],
    summary: pd.DataFrame,
    selected_report: dict[str, Any],
    stability_report: dict[str, Any],
    final_params: dict[str, Any],
    dev_evaluation: dict[str, Any],
) -> None:
    phase1c_summary = pd.read_csv(phase1c_run_dir / "aggregation_nested_cv_summary.csv")
    phase1d_summary = pd.read_csv(phase1d_run_dir / "timing_fusion_nested_cv_summary.csv")
    phase1c_eval = json.loads((phase1c_run_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    phase1d_eval = json.loads((phase1d_run_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    timing_phase1d = phase1d_summary[phase1d_summary["model_name"].eq("timing_only")].iloc[0]
    wavlm_phase1c = phase1c_summary[phase1c_summary["aggregation_method"].eq("mean")].iloc[0]
    early_phase1d = phase1d_summary[phase1d_summary["model_name"].eq("early_fusion")].iloc[0]
    timing_phase1e = summary[summary["model_name"].eq("timing_only")].iloc[0]
    wavlm_phase1e = summary[summary["model_name"].eq("wavlm_only")].iloc[0]
    reproduction = {
        "timing_phase1d_vs_phase1e_mean_auc_abs_diff": float(abs(timing_phase1d["mean_outer_roc_auc"] - timing_phase1e["mean_outer_roc_auc"])),
        "wavlm_phase1c_vs_phase1e_mean_auc_abs_diff": float(abs(wavlm_phase1c["mean_outer_roc_auc"] - wavlm_phase1e["mean_outer_roc_auc"])),
        "status": "passed"
        if abs(timing_phase1d["mean_outer_roc_auc"] - timing_phase1e["mean_outer_roc_auc"]) <= 1e-12
        and abs(wavlm_phase1c["mean_outer_roc_auc"] - wavlm_phase1e["mean_outer_roc_auc"]) <= 1e-12
        else "warning",
    }

    comparison_rows: list[dict[str, Any]] = [
        {
            "row_name": "phase1d_timing_only",
            "model_name": "timing_only",
            "source": "Phase 1D",
            "representation_dimension": 28,
            "fusion_complexity": "guided-interview dependent timing model",
            "train_nested_cv_mean_roc_auc": timing_phase1d["mean_outer_roc_auc"],
            "train_nested_cv_std_roc_auc": timing_phase1d["std_outer_roc_auc"],
            "train_nested_cv_mean_pr_auc": timing_phase1d["mean_outer_pr_auc"],
            "dev_roc_auc": phase1d_eval["selected_dev_evaluation"]["dev_metrics_at_0_5"]["roc_auc"],
            "dev_pr_auc": phase1d_eval["selected_dev_evaluation"]["dev_metrics_at_0_5"]["pr_auc"],
            "selected_hyperparameters": json.dumps({"C": phase1d_eval["selected_dev_evaluation"]["selected_C"]}),
            "notes": "Previous Phase 1D selected timing-only dev result.",
        },
        {
            "row_name": "phase1c_wavlm_only",
            "model_name": "wavlm_only",
            "source": "Phase 1C",
            "representation_dimension": WAVLM_DIM,
            "fusion_complexity": "free-speech compatible WavLM model",
            "train_nested_cv_mean_roc_auc": wavlm_phase1c["mean_outer_roc_auc"],
            "train_nested_cv_std_roc_auc": wavlm_phase1c["std_outer_roc_auc"],
            "train_nested_cv_mean_pr_auc": wavlm_phase1c["mean_outer_pr_auc"],
            "dev_roc_auc": phase1c_eval["selected_dev_evaluation"]["dev_metrics_at_0_5"]["roc_auc"],
            "dev_pr_auc": phase1c_eval["selected_dev_evaluation"]["dev_metrics_at_0_5"]["pr_auc"],
            "selected_hyperparameters": json.dumps({"C": phase1c_eval["selected_dev_evaluation"]["selected_C"]}),
            "notes": "Previous Phase 1C locked layer-9 mean dev result.",
        },
        {
            "row_name": "phase1d_early_feature_fusion",
            "model_name": "early_feature_fusion",
            "source": "Phase 1D",
            "representation_dimension": int(early_phase1d["representation_dimension"]),
            "fusion_complexity": "early feature concatenation; not repeated in Phase 1E",
            "train_nested_cv_mean_roc_auc": early_phase1d["mean_outer_roc_auc"],
            "train_nested_cv_std_roc_auc": early_phase1d["std_outer_roc_auc"],
            "train_nested_cv_mean_pr_auc": early_phase1d["mean_outer_pr_auc"],
            "dev_roc_auc": "",
            "dev_pr_auc": "",
            "selected_hyperparameters": "",
            "notes": "Train-only result; not selected/evaluated on dev in Phase 1D.",
        },
    ]
    for method in ["weighted_late_fusion", "logistic_stacking"]:
        row = summary[summary["model_name"].eq(method)].iloc[0]
        is_selected = dev_evaluation["selected_model"] == method
        comparison_rows.append(
            {
                "row_name": f"phase1e_{method}",
                "model_name": method,
                "source": "Phase 1E",
                "representation_dimension": int(row["representation_dimension"]),
                "fusion_complexity": "probability-level late fusion",
                "train_nested_cv_mean_roc_auc": row["mean_outer_roc_auc"],
                "train_nested_cv_std_roc_auc": row["std_outer_roc_auc"],
                "train_nested_cv_mean_pr_auc": row["mean_outer_pr_auc"],
                "dev_roc_auc": dev_evaluation["dev_metrics_at_0_5"]["roc_auc"] if is_selected else "",
                "dev_pr_auc": dev_evaluation["dev_metrics_at_0_5"]["pr_auc"] if is_selected else "",
                "selected_hyperparameters": json.dumps(dev_evaluation["selected_hyperparameters"]) if is_selected else "",
                "notes": "Selected Phase 1E model evaluated once on dev." if is_selected else "Train-only nested-CV result only; not evaluated on dev.",
            }
        )
    if dev_evaluation["selected_model"] in {"timing_only", "wavlm_only"}:
        row = summary[summary["model_name"].eq(dev_evaluation["selected_model"])].iloc[0]
        comparison_rows.append(
            {
                "row_name": "selected_phase1e_method",
                "model_name": dev_evaluation["selected_model"],
                "source": "Phase 1E",
                "representation_dimension": int(row["representation_dimension"]),
                "fusion_complexity": "individual base model selected under Phase 1E rule",
                "train_nested_cv_mean_roc_auc": row["mean_outer_roc_auc"],
                "train_nested_cv_std_roc_auc": row["std_outer_roc_auc"],
                "train_nested_cv_mean_pr_auc": row["mean_outer_pr_auc"],
                "dev_roc_auc": dev_evaluation["dev_metrics_at_0_5"]["roc_auc"],
                "dev_pr_auc": dev_evaluation["dev_metrics_at_0_5"]["pr_auc"],
                "selected_hyperparameters": json.dumps(dev_evaluation["selected_hyperparameters"]),
                "notes": "Selected Phase 1E model evaluated once on dev.",
            }
        )
    write_csv(run_dir / "late_fusion_comparison.csv", comparison_rows)

    deployment_note = {
        "free_speech_compatible": ["wavlm_only"],
        "guided_interview_dependent": ["timing_only", "weighted_late_fusion", "logistic_stacking"],
        "note": "Timing-only and timing/WavLM fusion models require response-latency or interaction-timing features from prompts/interviewer turns; an app using timing features would need standardized voice prompts or a short guided interview and should not be described as operating from an arbitrary voice clip.",
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_phase1e_late_fusion_train_only_selection_and_single_dev_evaluation",
        "locked_inputs": {
            "timing_features": "Phase 1D validated 28-feature participant timing table",
            "wavlm_model": MODEL_NAME,
            "wavlm_model_commit": MODEL_COMMIT,
            "wavlm_layer": LOCKED_LAYER,
            "wavlm_aggregation": LOCKED_AGGREGATION,
        },
        "test_evaluation_performed": False,
        "test_probabilities_generated": False,
        "strict_scope_confirmation": {
            "test_evaluation": False,
            "early_feature_concatenation": False,
            "additional_timing_feature_engineering": False,
            "manual_feature_removal": False,
            "wavlm_layer_reselection": False,
            "alternate_participant_aggregation": False,
            "mlp": False,
            "svm": False,
            "lightgbm": False,
            "attention_pooling": False,
            "window_level_classification": False,
            "wavlm_fine_tuning": False,
            "android_corpus": False,
            "app_code": False,
        },
        "input_validation": input_metadata,
        "base_model_reproduction": reproduction,
        "nested_cv_summary": summary.to_dict(orient="records"),
        "fusion_stability": stability_report,
        "selected_model_report": selected_report,
        "final_selected_parameters": final_params,
        "selected_dev_evaluation": dev_evaluation,
        "deployment_interpretation": deployment_note,
    }
    write_json(run_dir / "evaluation_report.json", report)

    git_commit = get_git_commit(root)
    warnings_list = []
    if not git_commit:
        warnings_list.append("Git commit unavailable because git executable was not available or repository metadata could not be read.")
    if reproduction["status"] != "passed":
        warnings_list.append("Base-model reproduction check did not exactly match prior train nested-CV results.")
    metadata = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "machine_name": platform.node(),
        "git_commit": git_commit,
        "warnings": warnings_list,
        "python_version": sys.version.replace("\n", " "),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": get_package_version("scikit-learn"),
        "source_phase1c_run": path_relative_to_root(phase1c_run_dir, root),
        "source_phase1d_run": path_relative_to_root(phase1d_run_dir, root),
        "input_paths": {
            "participant_manifest_clean": path_relative_to_root(output_dir / "daic_phase1_participant_manifest_clean.csv", root),
            "phase1c_aggregation_manifest": path_relative_to_root(phase1c_run_dir / "aggregation_manifest.csv", root),
            "phase1d_timing_features": path_relative_to_root(phase1d_run_dir / "speech_timing_features.csv", root),
        },
        "input_hashes": {
            "participant_manifest_clean_sha256": sha256_file(output_dir / "daic_phase1_participant_manifest_clean.csv"),
            "phase1c_aggregation_manifest_sha256": sha256_file(phase1c_run_dir / "aggregation_manifest.csv"),
            "phase1d_timing_features_sha256": sha256_file(phase1d_run_dir / "speech_timing_features.csv"),
        },
        "base_C_grid": BASE_C_GRID,
        "weight_grid": WEIGHT_GRID,
        "stacker_C_grid": STACKER_C_GRID,
        "outer_nested_cv": {
            "splitter": "RepeatedStratifiedKFold",
            "n_splits": OUTER_SPLITS,
            "n_repeats": OUTER_REPEATS,
            "random_state": RANDOM_SEED,
        },
        "final_train_selection_cv": {
            "splitter": "RepeatedStratifiedKFold",
            "n_splits": FINAL_SPLITS,
            "n_repeats": FINAL_REPEATS,
            "random_state": RANDOM_SEED,
        },
        "selected_model": dev_evaluation["selected_model"],
        "selected_hyperparameters": dev_evaluation["selected_hyperparameters"],
        "selected_dev_threshold": dev_evaluation["selected_dev_threshold"],
        "test_evaluation_performed": False,
        "test_labels_loaded_for_modeling": False,
    }
    write_json(run_dir / "run_metadata.json", metadata)

    lines = [
        "# Phase 1E Late Fusion Analysis",
        "",
        f"Run directory: `{path_relative_to_root(run_dir, root)}`",
        "",
        "## Base Reproduction",
        f"- Status: `{reproduction['status']}`",
        f"- Timing train AUC diff: `{reproduction['timing_phase1d_vs_phase1e_mean_auc_abs_diff']:.12f}`",
        f"- WavLM train AUC diff: `{reproduction['wavlm_phase1c_vs_phase1e_mean_auc_abs_diff']:.12f}`",
        "",
        "## Train Outer CV",
    ]
    for row in summary.sort_values("complexity_order").itertuples(index=False):
        lines.append(f"- `{row.model_name}` ROC-AUC: `{row.mean_outer_roc_auc:.6f} +/- {row.std_outer_roc_auc:.6f}`; PR-AUC: `{row.mean_outer_pr_auc:.6f}`")
    lines.extend(
        [
            "",
            "## Fusion Stability",
            f"- Mixed weighted-fusion folds: `{stability_report['weighted_fusion']['percent_true_mixed_weight']:.1f}%`",
            f"- Timing stacker coefficient positive: `{stability_report['logistic_stacking']['percent_timing_coefficient_positive']:.1f}%`",
            f"- WavLM stacker coefficient positive: `{stability_report['logistic_stacking']['percent_wavlm_coefficient_positive']:.1f}%`",
            "",
            "## Selected Model",
            f"- Selected model: `{dev_evaluation['selected_model']}`",
            f"- Dev ROC-AUC: `{dev_evaluation['dev_metrics_at_0_5']['roc_auc']:.6f}`",
            f"- Dev PR-AUC: `{dev_evaluation['dev_metrics_at_0_5']['pr_auc']:.6f}`",
            f"- Dev macro F1 at tuned threshold: `{dev_evaluation['dev_metrics_at_selected_threshold']['macro_f1']:.6f}`",
            "",
            "## Deployment Note",
            deployment_note["note"],
            "",
            "No test probabilities or test metrics were produced.",
        ]
    )
    (run_dir / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    add_dependency_dir(args.dependency_dir)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model._logistic")
    root = Path.cwd()
    output_dir = resolve_path(args.output_dir, root)
    phase1c_run_dir = resolve_path(args.phase1c_run_dir, root)
    phase1d_run_dir = resolve_path(args.phase1d_run_dir, root)
    timestamp = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve_path(args.run_root, root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    train_meta, timing_train, wavlm_train, merged_nonlabel, _, input_metadata = validate_inputs(root, output_dir, phase1c_run_dir, phase1d_run_dir, run_dir)
    y_train = train_meta["label_binary"].to_numpy(dtype=int)
    assert train_meta["official_split"].eq("train").all(), "Model-selection dataframe contains non-train rows."
    summary, _, _, selected_by_fold, stacker_by_fold = run_nested_outer_cv(train_meta, timing_train, wavlm_train, y_train, run_dir)
    stability_report = write_stability_report(selected_by_fold, stacker_by_fold, run_dir)
    selected_report = select_phase1e_model(summary, run_dir)
    final_params = final_train_selection_and_fit(root, run_dir, selected_report, train_meta, timing_train, wavlm_train, y_train)

    dev_meta, timing_dev, wavlm_dev, y_dev = load_dev_after_selection(root, output_dir, merged_nonlabel)
    dev_evaluation = evaluate_dev(root, run_dir, final_params, dev_meta, timing_dev, wavlm_dev, y_dev)
    write_reports(root, output_dir, phase1c_run_dir, phase1d_run_dir, run_dir, input_metadata, summary, selected_report, stability_report, final_params, dev_evaluation)

    print("Phase 1E late fusion analysis complete.")
    print(f"run_dir={path_relative_to_root(run_dir, root)}")
    print(f"selected_model={dev_evaluation['selected_model']} hyperparameters={json.dumps(dev_evaluation['selected_hyperparameters'], sort_keys=True)}")
    print(f"dev_roc_auc={dev_evaluation['dev_metrics_at_0_5']['roc_auc']:.6f}")
    print(f"test_evaluation_performed=false elapsed_sec={time.time() - start:.1f}")


if __name__ == "__main__":
    main()
