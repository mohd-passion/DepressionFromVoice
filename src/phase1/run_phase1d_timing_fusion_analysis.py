"""Phase 1D speech-timing feature analysis and fusion.

This script does not run WavLM extraction and does not evaluate the test set.
It uses locked WavLM layer-9 mean participant representations from Phase 1C.
Transcript text is never read as a feature source; only start_time, stop_time,
and speaker are used for timing features.
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
from sklearn.compose import ColumnTransformer
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


PHASE1C_RUN_ID = "20260622_135800"
MODEL_NAME = "microsoft/wavlm-base-plus"
MODEL_COMMIT = "4c66d4806a428f2e922ccfa1a962776e232d487b"
LOCKED_LAYER = "layer_09"
LOCKED_AGGREGATION = "mean"
EXPECTED_PARTICIPANTS = 187
EXPECTED_SPLITS = {"train": 106, "dev": 34, "test": 47}
EXCLUDED_PARTICIPANTS = {"409", "422"}
WAVLM_DIM = 768
RANDOM_SEED = 42
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
OUTER_SPLITS = 5
OUTER_REPEATS = 5
FINAL_SPLITS = 5
FINAL_REPEATS = 10
MODEL_ORDER = ["timing_only", "wavlm_only", "early_fusion"]
COMPLEXITY_ORDER = {"timing_only": 1, "wavlm_only": 2, "early_fusion": 3}
PHASE1C_REPRO_TOLERANCE = 1e-12

TIMING_FEATURES = [
    "interview_duration_sec",
    "participant_speech_duration_sec",
    "participant_speech_ratio",
    "participant_turn_count",
    "participant_turns_per_minute",
    "turn_duration_mean",
    "turn_duration_median",
    "turn_duration_std",
    "turn_duration_iqr",
    "turn_duration_p10",
    "turn_duration_p90",
    "turn_duration_min",
    "turn_duration_max",
    "response_latency_count",
    "response_latency_mean",
    "response_latency_median",
    "response_latency_std",
    "response_latency_iqr",
    "response_latency_p10",
    "response_latency_p90",
    "response_latency_max",
    "participant_gap_count",
    "participant_gap_mean",
    "participant_gap_median",
    "participant_gap_std",
    "participant_gap_iqr",
    "participant_gap_p90",
    "participant_gap_max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1D timing and WavLM fusion analysis.")
    parser.add_argument("--output-dir", default="outputs/phase1_daic")
    parser.add_argument("--phase1c-run-dir", default=f"outputs/phase1_daic/phase1c_aggregation_runs/{PHASE1C_RUN_ID}")
    parser.add_argument("--run-root", default="outputs/phase1_daic/phase1d_timing_fusion_runs")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dependency-dir", default=str(DEFAULT_DEPENDENCY_DIR))
    return parser.parse_args()


def add_dependency_dir(path_text: str) -> None:
    path = Path(path_text)
    if path.exists():
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def csv_cell(row: list[str], index: int) -> str:
    return clean(row[index]) if 0 <= index < len(row) else ""


def maybe_float(value: Any) -> float:
    text = clean(value)
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


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


def resolve_transcript_path(raw_path: str, participant_id: str, root: Path) -> Path:
    raw = clean(raw_path)
    transcript_name = Path(raw).name or f"{participant_id}_TRANSCRIPT.csv"
    participant_dir_name = Path(raw).parent.name or f"{participant_id}_P"
    candidates = []
    if raw:
        raw_path_obj = Path(raw)
        candidates.append(raw_path_obj if raw_path_obj.is_absolute() else root / raw_path_obj)
    candidates.append(root / "Data" / participant_dir_name / transcript_name)
    candidates.append(root / "Data" / f"{participant_id}_P" / transcript_name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def vector_stats(values: list[float], prefix: str, stats: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    if not values:
        for name in stats:
            result[f"{prefix}_{name}"] = float("nan")
        return result
    array = np.asarray(values, dtype=np.float64)
    for name in stats:
        if name == "mean":
            result[f"{prefix}_{name}"] = float(np.mean(array))
        elif name == "median":
            result[f"{prefix}_{name}"] = float(np.median(array))
        elif name == "std":
            result[f"{prefix}_{name}"] = float(np.std(array, ddof=0))
        elif name == "iqr":
            result[f"{prefix}_{name}"] = float(np.percentile(array, 75) - np.percentile(array, 25))
        elif name == "p10":
            result[f"{prefix}_{name}"] = float(np.percentile(array, 10))
        elif name == "p90":
            result[f"{prefix}_{name}"] = float(np.percentile(array, 90))
        elif name == "min":
            result[f"{prefix}_{name}"] = float(np.min(array))
        elif name == "max":
            result[f"{prefix}_{name}"] = float(np.max(array))
        else:
            raise ValueError(f"Unknown stat {name}")
    return result


def read_timing_intervals(transcript_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counters: Counter[str] = Counter()
    intervals: list[dict[str, Any]] = []
    if not transcript_path.exists():
        counters["missing_transcript"] += 1
        return intervals, dict(counters)
    with transcript_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"start_time", "stop_time", "speaker"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            counters["invalid_transcript_header"] += 1
            return intervals, dict(counters)
        for row_number, row in enumerate(reader, start=2):
            speaker = clean(row.get("speaker"))
            try:
                start = float(clean(row.get("start_time")))
                stop = float(clean(row.get("stop_time")))
            except ValueError:
                counters["invalid_time_rows"] += 1
                continue
            if not np.isfinite(start) or not np.isfinite(stop):
                counters["invalid_time_rows"] += 1
                continue
            if stop <= start:
                counters["nonpositive_duration_rows"] += 1
                continue
            intervals.append({"row_number": row_number, "start": start, "stop": stop, "speaker": speaker})
    intervals.sort(key=lambda row: (row["start"], row["stop"], row["row_number"]))
    return intervals, dict(counters)


def is_interviewer_speaker(speaker: str) -> bool:
    return clean(speaker).lower() in {"ellie", "interviewer"}


def timing_features_for_participant(participant_id: str, transcript_path: Path) -> tuple[dict[str, Any], dict[str, int]]:
    intervals, counters = read_timing_intervals(transcript_path)
    features: dict[str, Any] = {feature: float("nan") for feature in TIMING_FEATURES}
    if not intervals:
        features["participant_turn_count"] = 0
        features["response_latency_count"] = 0
        features["participant_gap_count"] = 0
        return features, counters

    min_start = min(row["start"] for row in intervals)
    max_stop = max(row["stop"] for row in intervals)
    interview_duration = max_stop - min_start
    participant_intervals = [row for row in intervals if clean(row["speaker"]) == "Participant"]
    participant_durations = [row["stop"] - row["start"] for row in participant_intervals]
    participant_speech_duration = float(np.sum(participant_durations, dtype=np.float64)) if participant_durations else 0.0
    features["interview_duration_sec"] = float(interview_duration)
    features["participant_speech_duration_sec"] = participant_speech_duration
    features["participant_speech_ratio"] = (
        participant_speech_duration / interview_duration if interview_duration > 0 else float("nan")
    )
    features["participant_turn_count"] = int(len(participant_intervals))
    features["participant_turns_per_minute"] = (
        len(participant_intervals) / (interview_duration / 60.0) if interview_duration > 0 else float("nan")
    )
    features.update(
        vector_stats(
            participant_durations,
            "turn_duration",
            ["mean", "median", "std", "iqr", "p10", "p90", "min", "max"],
        )
    )

    latencies: list[float] = []
    for previous, current in zip(intervals, intervals[1:]):
        if not is_interviewer_speaker(previous["speaker"]) or clean(current["speaker"]) != "Participant":
            continue
        latency = current["start"] - previous["stop"]
        if latency < 0:
            counters["negative_or_overlapping_response_latency_rows"] = counters.get("negative_or_overlapping_response_latency_rows", 0) + 1
            continue
        latencies.append(float(latency))
    features["response_latency_count"] = int(len(latencies))
    features.update(
        vector_stats(
            latencies,
            "response_latency",
            ["mean", "median", "std", "iqr", "p10", "p90", "max"],
        )
    )

    gaps: list[float] = []
    for previous, current in zip(participant_intervals, participant_intervals[1:]):
        gap = current["start"] - previous["stop"]
        if gap > 0:
            gaps.append(float(gap))
        elif gap < 0:
            counters["overlapping_participant_turn_rows"] = counters.get("overlapping_participant_turn_rows", 0) + 1
    features["participant_gap_count"] = int(len(gaps))
    features.update(
        vector_stats(
            gaps,
            "participant_gap",
            ["mean", "median", "std", "iqr", "p90", "max"],
        )
    )

    return features, counters


def finite_or_blank(value: Any) -> Any:
    if isinstance(value, float) and np.isnan(value):
        return ""
    return value


def metric_dict(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "positive_precision": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "positive_recall_sensitivity": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def positive_probabilities(model: Pipeline, x_data: pd.DataFrame) -> np.ndarray:
    classes = list(model.named_steps["classifier"].classes_)
    return model.predict_proba(x_data)[:, classes.index(1)]


def deterministic_thresholds(probabilities: np.ndarray) -> list[float]:
    unique = sorted({float(value) for value in probabilities})
    thresholds = {0.5, *unique}
    for left, right in zip(unique, unique[1:]):
        thresholds.add((left + right) / 2.0)
    return sorted(thresholds)


def tune_threshold(y_dev: np.ndarray, probabilities: np.ndarray) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, Any]] = []
    for threshold in deterministic_thresholds(probabilities):
        metrics = metric_dict(y_dev, probabilities, threshold)
        rows.append(metrics)
    frame = pd.DataFrame(rows)
    frame["distance_to_0_5"] = (frame["threshold"] - 0.5).abs()
    selected = frame.sort_values(
        ["macro_f1", "balanced_accuracy", "distance_to_0_5", "threshold"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return frame.drop(columns=["distance_to_0_5"]), float(selected["threshold"])


def build_pipeline(model_name: str, C: float, timing_columns: list[str], wavlm_columns: list[str]) -> Pipeline:
    classifier = LogisticRegression(C=C, class_weight="balanced", max_iter=10000, random_state=RANDOM_SEED)
    if model_name == "timing_only":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )
    if model_name == "wavlm_only":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )
    if model_name == "early_fusion":
        preprocessor = ColumnTransformer(
            [
                ("timing", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), timing_columns),
                ("wavlm", StandardScaler(), wavlm_columns),
            ],
            remainder="drop",
        )
        return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    raise ValueError(f"Unknown model {model_name}")


def select_inner_c(model_name: str, x_train: pd.DataFrame, y_train: np.ndarray, timing_columns: list[str], wavlm_columns: list[str], seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows: list[dict[str, Any]] = []
    for C in C_GRID:
        aucs: list[float] = []
        for inner_train_index, inner_validation_index in splitter.split(x_train, y_train):
            model = build_pipeline(model_name, C, timing_columns, wavlm_columns)
            model.fit(x_train.iloc[inner_train_index], y_train[inner_train_index])
            probabilities = positive_probabilities(model, x_train.iloc[inner_validation_index])
            aucs.append(float(roc_auc_score(y_train[inner_validation_index], probabilities)))
        rows.append(
            {
                "model_name": model_name,
                "C": C,
                "mean_roc_auc": float(np.mean(aucs)),
                "roc_auc_std": float(np.std(aucs, ddof=0)),
                "n_inner_folds": 5,
                "inner_seed": seed,
            }
        )
    selected = sorted(rows, key=lambda row: (-row["mean_roc_auc"], row["roc_auc_std"], row["C"]))[0]
    return selected, rows


def generate_timing_features(root: Path, output_dir: Path, run_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    participants = []
    with (output_dir / "daic_phase1_participant_manifest_clean.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        column_index = {name: index for index, name in enumerate(header)}
        required = ["participant_id", "official_split", "transcript_path"]
        if not all(name in column_index for name in required):
            raise RuntimeError(f"Participant manifest missing required non-label columns: {required}")
        for row in reader:
            participant_id = csv_cell(row, column_index["participant_id"])
            participants.append(
                {
                    "participant_id": participant_id,
                    "official_split": csv_cell(row, column_index["official_split"]),
                    "transcript_path": csv_cell(row, column_index["transcript_path"]),
                }
            )
    participants = sorted(participants, key=lambda row: participant_sort_key(row["participant_id"]))
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    transcript_issue_counter: Counter[str] = Counter()
    source_paths: dict[str, str] = {}
    for participant in participants:
        participant_id = participant["participant_id"]
        transcript_path = resolve_transcript_path(participant["transcript_path"], participant_id, root)
        source_paths[participant_id] = path_relative_to_root(transcript_path, root)
        features, counters = timing_features_for_participant(participant_id, transcript_path)
        transcript_issue_counter.update(counters)
        row = {
            "participant_id": participant_id,
            "official_split": participant["official_split"],
            **{feature: finite_or_blank(features[feature]) for feature in TIMING_FEATURES},
        }
        rows.append(row)
        for issue_type, count in counters.items():
            if count:
                issues.append(
                    make_issue(
                        "warning",
                        f"transcript_{issue_type}",
                        f"{count} rows for participant {participant_id}.",
                        participant_id=participant_id,
                    )
                )

    feature_frame = pd.DataFrame(rows)
    feature_frame.to_csv(run_dir / "speech_timing_features.csv", index=False)

    quality_rows: list[dict[str, Any]] = []
    participant_count = len(feature_frame)
    split_counts = feature_frame["official_split"].value_counts().to_dict()
    duplicates = int(feature_frame["participant_id"].duplicated().sum())
    excluded = sorted(set(feature_frame["participant_id"]) & EXCLUDED_PARTICIPANTS, key=participant_sort_key)
    quality_errors: list[dict[str, Any]] = []
    if participant_count != EXPECTED_PARTICIPANTS:
        quality_errors.append(make_issue("error", "participant_count_mismatch", f"Expected {EXPECTED_PARTICIPANTS}, found {participant_count}."))
    if split_counts != EXPECTED_SPLITS:
        quality_errors.append(make_issue("error", "split_count_mismatch", f"Expected {EXPECTED_SPLITS}, found {split_counts}."))
    if duplicates:
        quality_errors.append(make_issue("error", "duplicate_participants", f"Duplicate participant rows: {duplicates}."))
    if excluded:
        quality_errors.append(make_issue("error", "excluded_participants_present", f"Excluded participants present: {excluded}."))

    numeric = feature_frame[TIMING_FEATURES].apply(pd.to_numeric, errors="coerce")
    if (numeric["interview_duration_sec"].dropna() < 0).any() or (numeric["participant_speech_duration_sec"].dropna() < 0).any():
        quality_errors.append(make_issue("error", "negative_duration_feature", "Negative timing duration feature found."))
    if ((numeric["participant_speech_ratio"].dropna() < 0) | (numeric["participant_speech_ratio"].dropna() > 1)).any():
        quality_errors.append(make_issue("error", "speech_ratio_out_of_range", "Participant speech ratio outside [0, 1]."))
    for feature in TIMING_FEATURES:
        values = numeric[feature]
        missing_count = int(values.isna().sum())
        quality_rows.append(
            {
                "section": "missingness",
                "feature": feature,
                "missing_count": missing_count,
                "missing_fraction": float(missing_count / len(values)),
                "min": "",
                "max": "",
                "mean": "",
                "std": "",
                "feature_a": "",
                "feature_b": "",
                "pearson": "",
                "spearman": "",
                "issue_type": "",
                "severity": "info",
                "details": "",
            }
        )
        quality_rows.append(
            {
                "section": "statistics",
                "feature": feature,
                "missing_count": missing_count,
                "missing_fraction": float(missing_count / len(values)),
                "min": float(values.min(skipna=True)) if values.notna().any() else "",
                "max": float(values.max(skipna=True)) if values.notna().any() else "",
                "mean": float(values.mean(skipna=True)) if values.notna().any() else "",
                "std": float(values.std(skipna=True, ddof=0)) if values.notna().any() else "",
                "feature_a": "",
                "feature_b": "",
                "pearson": "",
                "spearman": "",
                "issue_type": "",
                "severity": "info",
                "details": "",
            }
        )

    train_numeric = numeric[feature_frame["official_split"].eq("train")].copy()
    pearson = train_numeric.corr(method="pearson", min_periods=3)
    spearman = train_numeric.corr(method="spearman", min_periods=3)
    correlation_rows: list[dict[str, Any]] = []
    high_correlation_pairs: list[dict[str, Any]] = []
    for left_index, feature_a in enumerate(TIMING_FEATURES):
        for feature_b in TIMING_FEATURES[left_index + 1 :]:
            pearson_value = pearson.loc[feature_a, feature_b]
            spearman_value = spearman.loc[feature_a, feature_b]
            row = {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "pearson": float(pearson_value) if pd.notna(pearson_value) else "",
                "spearman": float(spearman_value) if pd.notna(spearman_value) else "",
                "abs_pearson_gt_0_95": bool(pd.notna(pearson_value) and abs(float(pearson_value)) > 0.95),
                "abs_spearman_gt_0_95": bool(pd.notna(spearman_value) and abs(float(spearman_value)) > 0.95),
            }
            correlation_rows.append(row)
            if row["abs_pearson_gt_0_95"] or row["abs_spearman_gt_0_95"]:
                high_correlation_pairs.append(row)
                quality_rows.append(
                    {
                        "section": "high_train_correlation",
                        "feature": "",
                        "missing_count": "",
                        "missing_fraction": "",
                        "min": "",
                        "max": "",
                        "mean": "",
                        "std": "",
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "pearson": row["pearson"],
                        "spearman": row["spearman"],
                        "issue_type": "absolute_train_correlation_gt_0_95",
                        "severity": "warning",
                        "details": "High train-only feature correlation; feature retained for predefined analysis.",
                    }
                )
    write_csv(run_dir / "speech_timing_train_correlations.csv", correlation_rows)

    near_constant_features: list[str] = []
    for feature in TIMING_FEATURES:
        values = train_numeric[feature].dropna()
        if values.empty or values.nunique() <= 1:
            near_constant_features.append(feature)
            quality_rows.append(
                {
                    "section": "constant_or_near_constant",
                    "feature": feature,
                    "missing_count": int(train_numeric[feature].isna().sum()),
                    "missing_fraction": float(train_numeric[feature].isna().mean()),
                    "min": float(values.min()) if not values.empty else "",
                    "max": float(values.max()) if not values.empty else "",
                    "mean": float(values.mean()) if not values.empty else "",
                    "std": float(values.std(ddof=0)) if not values.empty else "",
                    "feature_a": "",
                    "feature_b": "",
                    "pearson": "",
                    "spearman": "",
                    "issue_type": "constant_or_near_constant_train_feature",
                    "severity": "warning",
                    "details": "Feature retained; no dev/test statistics used for removal.",
                }
            )

    quality_rows.extend(quality_errors)
    if not quality_errors:
        quality_rows.append(make_issue("info", "all_timing_feature_checks_passed", "All mandatory Phase 1D timing feature checks passed."))
    quality_rows.extend(issues)
    write_csv(run_dir / "speech_timing_quality_report.csv", quality_rows)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not quality_errors else "failed",
        "feature_source": "DAIC-WOZ transcript start_time, stop_time, and speaker only; transcript text/value column not used.",
        "participant_gap_interpretation": "Participant between-turn gaps may contain interviewer speech or silence and are not pure-silence estimates.",
        "feature_definitions": {
            "interview_duration_sec": "maximum valid stop_time minus minimum valid start_time",
            "participant_speech_duration_sec": "sum of valid Participant interval durations",
            "participant_speech_ratio": "participant_speech_duration_sec divided by interview_duration_sec",
            "participant_turn_features": "summary statistics over valid Participant intervals with stop_time > start_time",
            "response_latency_features": "non-negative gap from a valid Ellie/interviewer interval to the immediately following Participant interval",
            "participant_gap_features": "positive gaps between consecutive Participant intervals; may include interviewer speech or silence",
        },
        "participant_count": participant_count,
        "split_counts": split_counts,
        "excluded_participants_present": excluded,
        "transcript_issue_counts": dict(transcript_issue_counter),
        "high_train_correlation_pair_count": len(high_correlation_pairs),
        "near_constant_features": near_constant_features,
        "source_transcript_paths_by_participant": source_paths,
        "test_evaluation_performed": False,
        "test_labels_loaded_for_modeling": False,
    }
    write_json(run_dir / "speech_timing_metadata.json", metadata)
    if quality_errors:
        raise RuntimeError("Phase 1D timing feature quality validation failed.")
    return feature_frame, high_correlation_pairs, metadata


def load_model_inputs(root: Path, output_dir: Path, phase1c_run_dir: Path, run_dir: Path, timing_frame: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    timing_model_frame = timing_frame[timing_frame["official_split"].isin(["train", "dev"])].copy()
    assert not timing_model_frame["official_split"].eq("test").any(), "Timing modeling frame contains test participants before labels are loaded."

    phase1c_manifest = pd.read_csv(phase1c_run_dir / "aggregation_manifest.csv", dtype={"participant_id": str})
    wavlm_manifest = phase1c_manifest[
        phase1c_manifest["aggregation_method"].eq(LOCKED_AGGREGATION)
        & phase1c_manifest["aggregation_status"].eq("success")
        & phase1c_manifest["official_split"].isin(["train", "dev"])
    ].copy()
    assert not wavlm_manifest["official_split"].eq("test").any(), "WavLM modeling frame contains test participants before labels are loaded."

    label_rows: list[dict[str, Any]] = []
    with (output_dir / "daic_phase1_participant_manifest_clean.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        column_index = {name: index for index, name in enumerate(header)}
        required = ["participant_id", "official_split", "label_binary"]
        if not all(name in column_index for name in required):
            raise RuntimeError(f"Participant manifest missing required train/dev label columns: {required}")
        for row in reader:
            split = csv_cell(row, column_index["official_split"])
            if split in {"train", "dev"}:
                label_rows.append({"participant_id": csv_cell(row, column_index["participant_id"]), "label_binary": int(csv_cell(row, column_index["label_binary"]))})
    labels = pd.DataFrame(label_rows)

    frame = (
        timing_model_frame.merge(labels, on="participant_id", how="inner", validate="one_to_one")
        .merge(
            wavlm_manifest[["participant_id", "representation_path"]],
            on="participant_id",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("participant_id", key=lambda series: series.map(lambda value: participant_sort_key(str(value))[0]))
        .reset_index(drop=True)
    )
    assert not frame["official_split"].eq("test").any(), "Labels were loaded for test participants."
    if frame["official_split"].value_counts().to_dict() != {"train": 106, "dev": 34}:
        raise RuntimeError(f"Unexpected train/dev split counts: {frame['official_split'].value_counts().to_dict()}")

    timing_columns = TIMING_FEATURES.copy()
    wavlm_columns = [f"wavlm_{index:03d}" for index in range(WAVLM_DIM)]
    wavlm_vectors: list[np.ndarray] = []
    for row in frame.itertuples(index=False):
        vector = np.load(resolve_path(row.representation_path, root)).astype(np.float32, copy=False)
        if vector.shape != (WAVLM_DIM,) or not np.isfinite(vector).all() or np.all(vector == 0):
            raise RuntimeError(f"Invalid WavLM representation for participant {row.participant_id}: {vector.shape}")
        wavlm_vectors.append(vector)
    wavlm_frame = pd.DataFrame(np.vstack(wavlm_vectors), columns=wavlm_columns)
    timing_numeric = frame[timing_columns].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    combined = pd.concat(
        [
            frame[["participant_id", "official_split", "label_binary"]].reset_index(drop=True),
            timing_numeric,
            wavlm_frame,
        ],
        axis=1,
    )
    train_mask = combined["official_split"].eq("train").to_numpy()
    dev_mask = combined["official_split"].eq("dev").to_numpy()
    train_frame = combined.loc[train_mask, ["participant_id", "official_split", "label_binary"]].reset_index(drop=True)
    dev_frame = combined.loc[dev_mask, ["participant_id", "official_split", "label_binary"]].reset_index(drop=True)
    y_train = combined.loc[train_mask, "label_binary"].to_numpy(dtype=int)
    y_dev = combined.loc[dev_mask, "label_binary"].to_numpy(dtype=int)
    x_by_model = {
        "timing_only": combined.loc[:, timing_columns],
        "wavlm_only": combined.loc[:, wavlm_columns],
        "early_fusion": combined.loc[:, timing_columns + wavlm_columns],
    }
    model_train = {model: data.loc[train_mask].reset_index(drop=True) for model, data in x_by_model.items()}
    model_dev = {model: data.loc[dev_mask].reset_index(drop=True) for model, data in x_by_model.items()}
    assert all(split != "test" for split in train_frame["official_split"])
    assert all(split != "test" for split in dev_frame["official_split"])
    return {**{f"train_{k}": v for k, v in model_train.items()}, **{f"dev_{k}": v for k, v in model_dev.items()}}, y_train, y_dev, train_frame, dev_frame, timing_columns, wavlm_columns


def nested_cv_models(
    model_data: dict[str, pd.DataFrame],
    y_train: np.ndarray,
    train_frame: pd.DataFrame,
    timing_columns: list[str],
    wavlm_columns: list[str],
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    inner_rows_all: list[dict[str, Any]] = []
    splitter = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=RANDOM_SEED)
    for model_name in MODEL_ORDER:
        x_train = model_data[f"train_{model_name}"]
        model_metrics: list[dict[str, Any]] = []
        for outer_fold_index, (outer_train_index, outer_validation_index) in enumerate(splitter.split(x_train, y_train), start=1):
            inner_seed = RANDOM_SEED * 1000 + outer_fold_index
            selected, inner_rows = select_inner_c(model_name, x_train.iloc[outer_train_index], y_train[outer_train_index], timing_columns, wavlm_columns, inner_seed)
            inner_rows_all.extend([{**row, "outer_fold_index": outer_fold_index} for row in inner_rows])
            selected_rows.append(
                {
                    "model_name": model_name,
                    "outer_fold_index": outer_fold_index,
                    "repeat_index": int((outer_fold_index - 1) // OUTER_SPLITS + 1),
                    "fold_number": int((outer_fold_index - 1) % OUTER_SPLITS + 1),
                    "selected_C": selected["C"],
                    "inner_mean_roc_auc": selected["mean_roc_auc"],
                    "inner_roc_auc_std": selected["roc_auc_std"],
                    "inner_seed": inner_seed,
                }
            )
            model = build_pipeline(model_name, selected["C"], timing_columns, wavlm_columns)
            model.fit(x_train.iloc[outer_train_index], y_train[outer_train_index])
            probabilities = positive_probabilities(model, x_train.iloc[outer_validation_index])
            metrics = metric_dict(y_train[outer_validation_index], probabilities, 0.5)
            row = {
                "model_name": model_name,
                "outer_fold_index": outer_fold_index,
                "repeat_index": int((outer_fold_index - 1) // OUTER_SPLITS + 1),
                "fold_number": int((outer_fold_index - 1) % OUTER_SPLITS + 1),
                "selected_C": selected["C"],
                **metrics,
                "n_outer_train": int(len(outer_train_index)),
                "n_outer_validation": int(len(outer_validation_index)),
            }
            outer_rows.append(row)
            model_metrics.append(row)
            for participant_position, probability in zip(outer_validation_index, probabilities):
                participant = train_frame.iloc[int(participant_position)]
                prediction_rows.append(
                    {
                        "model_name": model_name,
                        "outer_fold_index": outer_fold_index,
                        "repeat_index": int((outer_fold_index - 1) // OUTER_SPLITS + 1),
                        "participant_id": participant["participant_id"],
                        "official_split": participant["official_split"],
                        "label_binary": int(y_train[int(participant_position)]),
                        "predicted_probability": float(probability),
                        "prediction_at_0_5": int(probability >= 0.5),
                        "selected_C": selected["C"],
                    }
                )
        print(f"phase1d_nested_cv_complete model={model_name} mean_auc={np.mean([row['roc_auc'] for row in model_metrics]):.4f}", flush=True)

    summary_rows: list[dict[str, Any]] = []
    for model_name, rows in pd.DataFrame(outer_rows).groupby("model_name"):
        aucs = rows["roc_auc"].to_numpy(dtype=float)
        pr_aucs = rows["pr_auc"].to_numpy(dtype=float)
        bal_accs = rows["balanced_accuracy"].to_numpy(dtype=float)
        macro_f1s = rows["macro_f1"].to_numpy(dtype=float)
        c_counts = Counter(rows["selected_C"].astype(str))
        summary_rows.append(
            {
                "model_name": model_name,
                "complexity_order": COMPLEXITY_ORDER[model_name],
                "representation_dimension": len(timing_columns) if model_name == "timing_only" else (WAVLM_DIM if model_name == "wavlm_only" else len(timing_columns) + WAVLM_DIM),
                "mean_outer_roc_auc": float(aucs.mean()),
                "std_outer_roc_auc": float(aucs.std(ddof=0)),
                "median_outer_roc_auc": float(np.median(aucs)),
                "min_outer_roc_auc": float(aucs.min()),
                "max_outer_roc_auc": float(aucs.max()),
                "mean_outer_pr_auc": float(pr_aucs.mean()),
                "mean_outer_balanced_accuracy_at_0_5": float(bal_accs.mean()),
                "mean_outer_macro_f1_at_0_5": float(macro_f1s.mean()),
                "selected_C_frequency": json.dumps({key: c_counts[key] for key in sorted(c_counts)}),
                "n_outer_evaluations": int(len(rows)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("complexity_order")
    outer = pd.DataFrame(outer_rows)
    predictions = pd.DataFrame(prediction_rows)
    selected = pd.DataFrame(selected_rows)
    inner = pd.DataFrame(inner_rows_all)
    summary.to_csv(run_dir / "timing_fusion_nested_cv_summary.csv", index=False)
    outer.to_csv(run_dir / "timing_fusion_outer_fold_results.csv", index=False)
    predictions.to_csv(run_dir / "timing_fusion_outer_predictions.csv", index=False)
    selected.to_csv(run_dir / "timing_fusion_selected_parameters.csv", index=False)
    inner.to_csv(run_dir / "timing_fusion_inner_cv_results.csv", index=False)
    return summary, outer, predictions, selected


def fusion_contribution_audit(predictions: pd.DataFrame, run_dir: Path) -> dict[str, Any]:
    averaged = (
        predictions.groupby(["model_name", "participant_id", "label_binary"], as_index=False)["predicted_probability"]
        .mean()
        .rename(columns={"predicted_probability": "mean_oof_probability"})
    )
    pivot = averaged.pivot(index=["participant_id", "label_binary"], columns="model_name", values="mean_oof_probability").reset_index()
    timing_prob = pivot["timing_only"].to_numpy(dtype=float)
    wavlm_prob = pivot["wavlm_only"].to_numpy(dtype=float)
    fusion_prob = pivot["early_fusion"].to_numpy(dtype=float)
    y = pivot["label_binary"].to_numpy(dtype=int)
    wavlm_pred = (wavlm_prob >= 0.5).astype(int)
    fusion_pred = (fusion_prob >= 0.5).astype(int)
    corrected = pivot.loc[(wavlm_pred != y) & (fusion_pred == y), "participant_id"].astype(str).tolist()
    made_incorrect = pivot.loc[(wavlm_pred == y) & (fusion_pred != y), "participant_id"].astype(str).tolist()
    audit_rows = []
    for row in pivot.itertuples(index=False):
        audit_rows.append(
            {
                "participant_id": row.participant_id,
                "label_binary": int(row.label_binary),
                "timing_only_mean_oof_probability": float(row.timing_only),
                "wavlm_only_mean_oof_probability": float(row.wavlm_only),
                "early_fusion_mean_oof_probability": float(row.early_fusion),
                "wavlm_prediction_at_0_5": int(row.wavlm_only >= 0.5),
                "fusion_prediction_at_0_5": int(row.early_fusion >= 0.5),
                "fusion_corrected_wavlm": int((row.wavlm_only >= 0.5) != row.label_binary and (row.early_fusion >= 0.5) == row.label_binary),
                "fusion_made_incorrect": int((row.wavlm_only >= 0.5) == row.label_binary and (row.early_fusion >= 0.5) != row.label_binary),
            }
        )
    write_csv(run_dir / "fusion_contribution_audit.csv", audit_rows)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "basis": "train participant-level averaged out-of-fold probabilities only",
        "timing_wavlm_probability_pearson": float(pd.Series(timing_prob).corr(pd.Series(wavlm_prob), method="pearson")),
        "timing_wavlm_probability_spearman": float(pd.Series(timing_prob).corr(pd.Series(wavlm_prob), method="spearman")),
        "wavlm_fusion_probability_pearson": float(pd.Series(wavlm_prob).corr(pd.Series(fusion_prob), method="pearson")),
        "wavlm_fusion_probability_spearman": float(pd.Series(wavlm_prob).corr(pd.Series(fusion_prob), method="spearman")),
        "participants_corrected_by_fusion_relative_to_wavlm_at_0_5": corrected,
        "participants_made_incorrect_by_fusion_relative_to_wavlm_at_0_5": made_incorrect,
        "descriptive_only": True,
    }
    write_json(run_dir / "fusion_contribution_audit.json", report)
    return report


def select_model(summary: pd.DataFrame, run_dir: Path) -> dict[str, Any]:
    sorted_summary = summary.sort_values(
        ["mean_outer_roc_auc", "std_outer_roc_auc", "mean_outer_pr_auc", "complexity_order"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    nominal_best = sorted_summary.iloc[0].to_dict()
    wavlm = summary[summary["model_name"].eq("wavlm_only")].iloc[0].to_dict()
    timing = summary[summary["model_name"].eq("timing_only")].iloc[0].to_dict()
    fusion = summary[summary["model_name"].eq("early_fusion")].iloc[0].to_dict()
    selected = nominal_best
    practical_rule_applied = False
    fusion_improvement = float(fusion["mean_outer_roc_auc"] - wavlm["mean_outer_roc_auc"])
    fusion_std_reduction = float(wavlm["std_outer_roc_auc"] - fusion["std_outer_roc_auc"])
    if nominal_best["model_name"] == "early_fusion" and fusion_improvement < 0.01 and fusion_std_reduction < 0.01:
        selected = wavlm
        practical_rule_applied = True
    timing_matches_wavlm = abs(float(timing["mean_outer_roc_auc"] - wavlm["mean_outer_roc_auc"])) < 0.01
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected["model_name"],
        "nominal_best_model": nominal_best["model_name"],
        "selection_criteria": [
            "highest mean outer ROC-AUC",
            "lowest ROC-AUC standard deviation",
            "highest mean outer PR-AUC",
            "simpler model if effectively tied",
            "if fusion improves over WavLM-only by less than 0.01, retain WavLM-only unless fusion materially reduces variability",
        ],
        "complexity_order": COMPLEXITY_ORDER,
        "fusion_improvement_over_wavlm_mean_roc_auc": fusion_improvement,
        "fusion_std_reduction_vs_wavlm": fusion_std_reduction,
        "fusion_practical_improvement_rule_applied": practical_rule_applied,
        "timing_matches_wavlm_within_0_01": timing_matches_wavlm,
        "timing_confounding_warning": "Timing-only matches WavLM within 0.01 mean ROC-AUC; possible dataset-timing confounding should be considered."
        if timing_matches_wavlm
        else "",
        "selected_model_summary": selected,
        "test_data_used_for_selection": False,
    }
    write_json(run_dir / "selected_phase1d_model.json", report)
    return report


def final_c_selection(model_name: str, x_train: pd.DataFrame, y_train: np.ndarray, timing_columns: list[str], wavlm_columns: list[str], run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    splitter = RepeatedStratifiedKFold(n_splits=FINAL_SPLITS, n_repeats=FINAL_REPEATS, random_state=RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    for C in C_GRID:
        aucs: list[float] = []
        pr_aucs: list[float] = []
        bal_accs: list[float] = []
        macro_f1s: list[float] = []
        for train_index, validation_index in splitter.split(x_train, y_train):
            model = build_pipeline(model_name, C, timing_columns, wavlm_columns)
            model.fit(x_train.iloc[train_index], y_train[train_index])
            probabilities = positive_probabilities(model, x_train.iloc[validation_index])
            metrics = metric_dict(y_train[validation_index], probabilities, 0.5)
            aucs.append(metrics["roc_auc"])
            pr_aucs.append(metrics["pr_auc"])
            bal_accs.append(metrics["balanced_accuracy"])
            macro_f1s.append(metrics["macro_f1"])
        rows.append(
            {
                "model_name": model_name,
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
    frame = pd.DataFrame(rows)
    selected = frame.sort_values(["mean_roc_auc", "roc_auc_std", "C"], ascending=[False, True, True]).iloc[0]
    frame.to_csv(run_dir / "final_c_search.csv", index=False)
    selected_params = {
        "model_name": model_name,
        "selected_C": float(selected["C"]),
        "selection_rule": ["highest mean ROC-AUC", "lowest ROC-AUC std", "lower C"],
    }
    write_json(run_dir / "final_selected_hyperparameters.json", selected_params)
    return frame, selected_params


def evaluate_selected_model(
    root: Path,
    run_dir: Path,
    selected_report: dict[str, Any],
    model_data: dict[str, pd.DataFrame],
    y_train: np.ndarray,
    y_dev: np.ndarray,
    dev_frame: pd.DataFrame,
    timing_columns: list[str],
    wavlm_columns: list[str],
) -> dict[str, Any]:
    model_name = selected_report["selected_model"]
    _, selected_params = final_c_selection(model_name, model_data[f"train_{model_name}"], y_train, timing_columns, wavlm_columns, run_dir)
    model = build_pipeline(model_name, selected_params["selected_C"], timing_columns, wavlm_columns)
    model.fit(model_data[f"train_{model_name}"], y_train)
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "selected_phase1d_pipeline.joblib"
    joblib.dump(model, model_path)
    probabilities = positive_probabilities(model, model_data[f"dev_{model_name}"])
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
                "predicted_probability": float(probability),
                "prediction_at_0_5": int(probability >= 0.5),
                "prediction_at_tuned_threshold": int(probability >= selected_threshold),
                "tuned_threshold": selected_threshold,
                "model_name": model_name,
            }
        )
    write_csv(run_dir / "selected_phase1d_dev_predictions.csv", prediction_rows)
    return {
        "selected_model": model_name,
        "selected_C": selected_params["selected_C"],
        "selected_dev_threshold": selected_threshold,
        "dev_metrics_at_0_5": metrics_0_5,
        "dev_metrics_at_selected_threshold": metrics_tuned,
        "model_path": path_relative_to_root(model_path, root),
        "test_evaluation_performed": False,
    }


def write_reports(
    root: Path,
    output_dir: Path,
    phase1c_run_dir: Path,
    run_dir: Path,
    timing_metadata: dict[str, Any],
    high_correlations: list[dict[str, Any]],
    summary: pd.DataFrame,
    selected_report: dict[str, Any],
    fusion_audit: dict[str, Any],
    evaluation: dict[str, Any],
) -> None:
    phase1c_summary = pd.read_csv(phase1c_run_dir / "aggregation_nested_cv_summary.csv")
    phase1c_mean = phase1c_summary[phase1c_summary["aggregation_method"].eq("mean")].iloc[0]
    wavlm_summary = summary[summary["model_name"].eq("wavlm_only")].iloc[0]
    reproduction = {
        "phase1c_mean_outer_roc_auc": float(phase1c_mean["mean_outer_roc_auc"]),
        "phase1d_wavlm_outer_roc_auc": float(wavlm_summary["mean_outer_roc_auc"]),
        "absolute_difference": float(abs(phase1c_mean["mean_outer_roc_auc"] - wavlm_summary["mean_outer_roc_auc"])),
        "tolerance": PHASE1C_REPRO_TOLERANCE,
        "status": "passed"
        if abs(phase1c_mean["mean_outer_roc_auc"] - wavlm_summary["mean_outer_roc_auc"]) <= PHASE1C_REPRO_TOLERANCE
        else "warning",
    }

    comparison_rows: list[dict[str, Any]] = []
    for row in summary.to_dict(orient="records"):
        is_selected = row["model_name"] == evaluation["selected_model"]
        comparison_rows.append(
            {
                "model_name": row["model_name"],
                "complexity_order": row["complexity_order"],
                "representation_dimension": row["representation_dimension"],
                "uses_timing_features": row["model_name"] in {"timing_only", "early_fusion"},
                "uses_wavlm_layer9_mean": row["model_name"] in {"wavlm_only", "early_fusion"},
                "train_nested_cv_mean_roc_auc": row["mean_outer_roc_auc"],
                "train_nested_cv_std_roc_auc": row["std_outer_roc_auc"],
                "train_nested_cv_mean_pr_auc": row["mean_outer_pr_auc"],
                "train_nested_cv_mean_balanced_accuracy": row["mean_outer_balanced_accuracy_at_0_5"],
                "train_nested_cv_mean_macro_f1": row["mean_outer_macro_f1_at_0_5"],
                "selected_C": evaluation["selected_C"] if is_selected else "",
                "selected_dev_threshold": evaluation["selected_dev_threshold"] if is_selected else "",
                "dev_roc_auc": evaluation["dev_metrics_at_0_5"]["roc_auc"] if is_selected else "",
                "dev_pr_auc": evaluation["dev_metrics_at_0_5"]["pr_auc"] if is_selected else "",
                "dev_balanced_accuracy_at_0_5": evaluation["dev_metrics_at_0_5"]["balanced_accuracy"] if is_selected else "",
                "dev_macro_f1_at_0_5": evaluation["dev_metrics_at_0_5"]["macro_f1"] if is_selected else "",
                "dev_balanced_accuracy_tuned": evaluation["dev_metrics_at_selected_threshold"]["balanced_accuracy"] if is_selected else "",
                "dev_macro_f1_tuned": evaluation["dev_metrics_at_selected_threshold"]["macro_f1"] if is_selected else "",
                "notes": "Selected model evaluated once on dev." if is_selected else "Train-only nested-CV result only; not evaluated on dev in Phase 1D.",
            }
        )
    write_csv(run_dir / "timing_fusion_comparison.csv", comparison_rows)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_phase1d_timing_fusion_analysis",
        "locked_wavlm": {
            "model_name": MODEL_NAME,
            "model_commit": MODEL_COMMIT,
            "layer": LOCKED_LAYER,
            "participant_aggregation": LOCKED_AGGREGATION,
        },
        "test_evaluation_performed": False,
        "test_probabilities_generated": False,
        "strict_scope_confirmation": {
            "wavlm_extraction_rerun": False,
            "test_evaluation": False,
            "wavlm_layer_reselection": False,
            "alternate_aggregation": False,
            "multi_layer_fusion": False,
            "attention_pooling": False,
            "window_level_classification": False,
            "mlp": False,
            "svm": False,
            "lightgbm": False,
            "wavlm_fine_tuning": False,
            "android_corpus": False,
            "app_code": False,
        },
        "timing_feature_quality": timing_metadata,
        "high_train_correlation_pairs": high_correlations,
        "phase1c_wavlm_reproduction_check": reproduction,
        "nested_cv_summary": summary.to_dict(orient="records"),
        "fusion_contribution_audit": fusion_audit,
        "selected_model_report": selected_report,
        "selected_dev_evaluation": evaluation,
    }
    write_json(run_dir / "evaluation_report.json", report)

    git_commit = get_git_commit(root)
    warnings_list = []
    if not git_commit:
        warnings_list.append("Git commit unavailable because git executable was not available or repository metadata could not be read.")
    if reproduction["status"] != "passed":
        warnings_list.append("Phase 1D WavLM-only nested CV did not exactly reproduce Phase 1C mean aggregation nested CV.")
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
        "input_paths": {
            "participant_manifest_clean": path_relative_to_root(output_dir / "daic_phase1_participant_manifest_clean.csv", root),
            "phase1c_aggregation_manifest": path_relative_to_root(phase1c_run_dir / "aggregation_manifest.csv", root),
            "phase1c_nested_cv_summary": path_relative_to_root(phase1c_run_dir / "aggregation_nested_cv_summary.csv", root),
        },
        "input_hashes": {
            "participant_manifest_clean_sha256": sha256_file(output_dir / "daic_phase1_participant_manifest_clean.csv"),
            "phase1c_aggregation_manifest_sha256": sha256_file(phase1c_run_dir / "aggregation_manifest.csv"),
            "phase1c_nested_cv_summary_sha256": sha256_file(phase1c_run_dir / "aggregation_nested_cv_summary.csv"),
        },
        "timing_feature_names": TIMING_FEATURES,
        "model_grid": C_GRID,
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
        "selected_model": evaluation["selected_model"],
        "selected_C": evaluation["selected_C"],
        "selected_dev_threshold": evaluation["selected_dev_threshold"],
        "test_evaluation_performed": False,
        "test_labels_loaded_for_modeling": False,
    }
    write_json(run_dir / "run_metadata.json", metadata)

    lines = [
        "# Phase 1D Timing Fusion Analysis",
        "",
        f"Run directory: `{path_relative_to_root(run_dir, root)}`",
        "",
        "## Timing Features",
        f"- Quality status: `{timing_metadata['status']}`",
        f"- High train-correlation pairs: `{len(high_correlations)}`",
        "",
        "## Train Nested CV",
    ]
    for row in summary.sort_values("complexity_order").itertuples(index=False):
        lines.append(f"- `{row.model_name}` ROC-AUC: `{row.mean_outer_roc_auc:.6f} +/- {row.std_outer_roc_auc:.6f}`; PR-AUC: `{row.mean_outer_pr_auc:.6f}`")
    lines.extend(
        [
            "",
            "## Selected Model",
            f"- Selected model: `{evaluation['selected_model']}`",
            f"- Selected C: `{evaluation['selected_C']}`",
            f"- Dev ROC-AUC: `{evaluation['dev_metrics_at_0_5']['roc_auc']:.6f}`",
            f"- Dev PR-AUC: `{evaluation['dev_metrics_at_0_5']['pr_auc']:.6f}`",
            f"- Dev macro F1 at tuned threshold: `{evaluation['dev_metrics_at_selected_threshold']['macro_f1']:.6f}`",
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
    timestamp = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve_path(args.run_root, root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    timing_frame, high_correlations, timing_metadata = generate_timing_features(root, output_dir, run_dir)
    model_data, y_train, y_dev, train_frame, dev_frame, timing_columns, wavlm_columns = load_model_inputs(root, output_dir, phase1c_run_dir, run_dir, timing_frame)
    assert not train_frame["official_split"].eq("test").any()
    assert not dev_frame["official_split"].eq("test").any()
    summary, _, predictions, _ = nested_cv_models(model_data, y_train, train_frame, timing_columns, wavlm_columns, run_dir)
    fusion_audit = fusion_contribution_audit(predictions, run_dir)
    selected_report = select_model(summary, run_dir)
    evaluation = evaluate_selected_model(root, run_dir, selected_report, model_data, y_train, y_dev, dev_frame, timing_columns, wavlm_columns)
    write_reports(root, output_dir, phase1c_run_dir, run_dir, timing_metadata, high_correlations, summary, selected_report, fusion_audit, evaluation)

    print("Phase 1D timing fusion analysis complete.")
    print(f"run_dir={path_relative_to_root(run_dir, root)}")
    print(f"selected_model={evaluation['selected_model']} selected_C={evaluation['selected_C']}")
    print(f"dev_roc_auc={evaluation['dev_metrics_at_0_5']['roc_auc']:.6f}")
    print(f"test_evaluation_performed=false elapsed_sec={time.time() - start:.1f}")


if __name__ == "__main__":
    main()
