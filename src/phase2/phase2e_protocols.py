"""Participant-safe protocol helpers for Phase 2E cross-task transfer."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from .phase2c_metrics import confusion_counts, probability_metrics, select_threshold, threshold_metrics_from_counts
except ImportError:  # pragma: no cover - supports direct script execution
    from phase2c_metrics import confusion_counts, probability_metrics, select_threshold, threshold_metrics_from_counts


C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
FOLD_SEED = 20260704
N_OUTER_FOLDS = 5
EXPECTED_DIM = 768


@dataclass(frozen=True)
class CoherentFoldSummary:
    fold: str
    participant_count: int
    hc_count: int
    pt_count: int
    outer_training_count: int
    outer_test_count: int


@dataclass(frozen=True)
class SourceFoldModel:
    source_model: str
    outer_fold: str
    selected_c: float
    selected_threshold: float
    threshold_macro_f1_train: float
    threshold_balanced_accuracy_train: float
    threshold_candidate_count: int
    pipeline: Any
    inner_rows: list[dict[str, Any]]


def assert_paired_identity(
    participant_ids: list[str],
    labels: np.ndarray,
    reading_embeddings: np.ndarray,
    interview_embeddings: np.ndarray,
) -> None:
    if len(participant_ids) != len(set(participant_ids)):
        duplicates = [pid for pid, count in Counter(participant_ids).items() if count > 1]
        raise ValueError(f"Duplicate paired participant IDs detected: {duplicates[:5]}")
    if len(participant_ids) == 0:
        raise ValueError("No paired participants are available.")
    y = np.asarray(labels, dtype=int)
    if len(y) != len(participant_ids):
        raise ValueError("Participant and label counts differ.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Paired labels must contain both binary classes.")
    for name, matrix in [("reading", reading_embeddings), ("interview", interview_embeddings)]:
        x = np.asarray(matrix)
        if x.shape != (len(participant_ids), EXPECTED_DIM):
            raise ValueError(f"{name} embedding matrix has shape {x.shape}, expected {(len(participant_ids), EXPECTED_DIM)}.")
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{name} embedding matrix contains NaN or infinite values.")
        zero_rows = np.where(np.all(x == 0, axis=1))[0]
        if len(zero_rows):
            raise ValueError(f"{name} embedding matrix contains all-zero participant vectors.")


def make_participant_coherent_folds(labels: np.ndarray, seed: int = FOLD_SEED) -> list[str]:
    y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("Cannot create stratified folds without both classes.")
    folds = [""] * len(y)
    rng = np.random.default_rng(int(seed))
    for label in sorted(np.unique(y).tolist()):
        indices = np.where(y == label)[0].astype(int)
        shuffled = rng.permutation(indices)
        for position, idx in enumerate(shuffled):
            folds[int(idx)] = f"fold{(position % N_OUTER_FOLDS) + 1}"
    if any(not fold for fold in folds):
        raise ValueError("Fold assignment is incomplete.")
    return folds


def assert_coherent_fold_integrity(participant_ids: list[str], labels: np.ndarray, folds: list[str]) -> list[CoherentFoldSummary]:
    if len(participant_ids) != len(folds):
        raise ValueError("Participant and fold counts differ.")
    if len(participant_ids) != len(set(participant_ids)):
        raise ValueError("Participant IDs are not unique in coherent folds.")
    fold_values = sorted(set(folds))
    if fold_values != [f"fold{i}" for i in range(1, N_OUTER_FOLDS + 1)]:
        raise ValueError(f"Expected coherent folds fold1..fold5, found {fold_values}.")
    y = np.asarray(labels, dtype=int)
    fold_array = np.asarray(folds)
    summaries: list[CoherentFoldSummary] = []
    for fold in fold_values:
        test_mask = fold_array == fold
        train_mask = ~test_mask
        hc = int(np.sum(y[test_mask] == 0))
        pt = int(np.sum(y[test_mask] == 1))
        if hc == 0 or pt == 0:
            raise ValueError(f"{fold} does not contain both classes.")
        summaries.append(
            CoherentFoldSummary(
                fold=fold,
                participant_count=int(np.sum(test_mask)),
                hc_count=hc,
                pt_count=pt,
                outer_training_count=int(np.sum(train_mask)),
                outer_test_count=int(np.sum(test_mask)),
            )
        )
    if sum(row.outer_test_count for row in summaries) != len(participant_ids):
        raise ValueError("Every participant must be assigned to exactly one outer fold.")
    return summaries


def assert_no_participant_overlap(train_ids: set[str], test_ids: set[str], context: str) -> None:
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"{context} participant leakage detected: {sorted(overlap)[:5]}")


def make_logistic_pipeline(c_value: float) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    penalty="l2",
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _build_source_training_matrix(
    source_model: str,
    reading_embeddings: np.ndarray,
    interview_embeddings: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if source_model == "reading_only":
        return (
            np.asarray(reading_embeddings[train_mask], dtype=float),
            np.asarray(labels[train_mask], dtype=int),
            np.asarray(folds[train_mask]),
            ["reading"] * int(np.sum(train_mask)),
        )
    if source_model == "interview_only":
        return (
            np.asarray(interview_embeddings[train_mask], dtype=float),
            np.asarray(labels[train_mask], dtype=int),
            np.asarray(folds[train_mask]),
            ["interview"] * int(np.sum(train_mask)),
        )
    if source_model == "pooled":
        return (
            np.vstack([reading_embeddings[train_mask], interview_embeddings[train_mask]]).astype(float),
            np.concatenate([labels[train_mask], labels[train_mask]]).astype(int),
            np.concatenate([folds[train_mask], folds[train_mask]]),
            ["reading"] * int(np.sum(train_mask)) + ["interview"] * int(np.sum(train_mask)),
        )
    raise ValueError(f"Unknown source model: {source_model}")


def select_c_leave_one_fold_out(x_train: np.ndarray, y_train: np.ndarray, train_folds: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows: list[dict[str, Any]] = []
    best_c: float | None = None
    best_key: tuple[float, float, float] | None = None
    for c_value in C_GRID:
        aucs: list[float] = []
        pr_aucs: list[float] = []
        for validation_fold in sorted(np.unique(train_folds)):
            inner_validation = train_folds == validation_fold
            inner_train = ~inner_validation
            if len(np.unique(y_train[inner_validation])) != 2:
                raise ValueError(f"Inner validation fold {validation_fold} does not contain both classes.")
            model = make_logistic_pipeline(c_value)
            model.fit(x_train[inner_train], y_train[inner_train])
            prob = model.predict_proba(x_train[inner_validation])[:, 1]
            aucs.append(float(roc_auc_score(y_train[inner_validation], prob)))
            pr_aucs.append(float(average_precision_score(y_train[inner_validation], prob)))
        row = {
            "C": float(c_value),
            "mean_inner_roc_auc": float(np.mean(aucs)),
            "std_inner_roc_auc": float(np.std(aucs, ddof=0)),
            "mean_inner_pr_auc": float(np.mean(pr_aucs)),
        }
        rows.append(row)
        key = (row["mean_inner_roc_auc"], -row["std_inner_roc_auc"], -float(c_value))
        if best_key is None or key > best_key:
            best_key = key
            best_c = float(c_value)
    if best_c is None:
        raise ValueError("No C candidate was selected.")
    return best_c, rows


def inner_oof_probabilities(x_train: np.ndarray, y_train: np.ndarray, train_folds: np.ndarray, c_value: float) -> np.ndarray:
    oof = np.full(len(y_train), np.nan, dtype=float)
    for validation_fold in sorted(np.unique(train_folds)):
        inner_validation = train_folds == validation_fold
        inner_train = ~inner_validation
        model = make_logistic_pipeline(c_value)
        model.fit(x_train[inner_train], y_train[inner_train])
        oof[inner_validation] = model.predict_proba(x_train[inner_validation])[:, 1]
    if np.any(~np.isfinite(oof)):
        raise ValueError("Inner out-of-fold probabilities are incomplete.")
    return oof


def fit_source_fold_model(
    source_model: str,
    outer_fold: str,
    participant_ids: list[str],
    labels: np.ndarray,
    folds: list[str],
    reading_embeddings: np.ndarray,
    interview_embeddings: np.ndarray,
) -> SourceFoldModel:
    y = np.asarray(labels, dtype=int)
    fold_array = np.asarray(folds)
    train_mask = fold_array != outer_fold
    test_mask = fold_array == outer_fold
    train_ids = {participant_ids[idx] for idx in np.where(train_mask)[0]}
    test_ids = {participant_ids[idx] for idx in np.where(test_mask)[0]}
    assert_no_participant_overlap(train_ids, test_ids, f"{source_model} {outer_fold} outer")
    x_train, y_train, train_folds, train_tasks = _build_source_training_matrix(
        source_model, reading_embeddings, interview_embeddings, y, fold_array, train_mask
    )
    if source_model == "reading_only" and any(task != "reading" for task in train_tasks):
        raise ValueError("Reading-only source model received non-reading samples.")
    if source_model == "interview_only" and any(task != "interview" for task in train_tasks):
        raise ValueError("Interview-only source model received non-interview samples.")
    selected_c, c_rows = select_c_leave_one_fold_out(x_train, y_train, train_folds)
    inner_oof = inner_oof_probabilities(x_train, y_train, train_folds, selected_c)
    threshold_selection = select_threshold(y_train, inner_oof)
    model = make_logistic_pipeline(selected_c)
    model.fit(x_train, y_train)
    inner_rows = []
    for c_row in c_rows:
        row = dict(c_row)
        row.update({"source_model": source_model, "outer_fold": outer_fold})
        inner_rows.append(row)
    return SourceFoldModel(
        source_model=source_model,
        outer_fold=outer_fold,
        selected_c=float(selected_c),
        selected_threshold=float(threshold_selection.threshold),
        threshold_macro_f1_train=float(threshold_selection.macro_f1),
        threshold_balanced_accuracy_train=float(threshold_selection.balanced_accuracy),
        threshold_candidate_count=int(threshold_selection.candidate_count),
        pipeline=model,
        inner_rows=inner_rows,
    )


def target_embeddings(target_task: str, reading_embeddings: np.ndarray, interview_embeddings: np.ndarray) -> np.ndarray:
    if target_task == "reading":
        return reading_embeddings
    if target_task == "interview":
        return interview_embeddings
    raise ValueError(f"Unknown target task: {target_task}")


def prediction_rows_for_protocol(
    protocol: str,
    source_task: str,
    target_task: str,
    model: SourceFoldModel,
    participant_ids: list[str],
    labels: np.ndarray,
    folds: list[str],
    reading_embeddings: np.ndarray,
    interview_embeddings: np.ndarray,
) -> list[dict[str, Any]]:
    fold_array = np.asarray(folds)
    test_mask = fold_array == model.outer_fold
    x_target = target_embeddings(target_task, reading_embeddings, interview_embeddings)
    probabilities = model.pipeline.predict_proba(x_target[test_mask])[:, 1]
    rows: list[dict[str, Any]] = []
    for idx, prob in zip(np.where(test_mask)[0], probabilities):
        for threshold_variant, threshold, source in [
            ("threshold_0_5", 0.5, "default_0_5"),
            ("threshold_adapted", model.selected_threshold, "outer_training_inner_oof_macro_f1"),
        ]:
            rows.append(
                {
                    "participant_id": participant_ids[idx],
                    "coherent_fold": model.outer_fold,
                    "label_binary": int(labels[idx]),
                    "protocol": protocol,
                    "source_task": source_task,
                    "target_task": target_task,
                    "threshold_variant": threshold_variant,
                    "threshold_source": source,
                    "predicted_probability": float(prob),
                    "decision_threshold": float(threshold),
                    "prediction": int(float(prob) >= float(threshold)),
                    "selected_C": float(model.selected_c),
                }
            )
    return rows


def assert_oof_prediction_completeness(rows: list[dict[str, Any]], expected_ids: set[str], protocol: str, threshold_variant: str) -> None:
    ids = [
        str(row["participant_id"])
        for row in rows
        if row.get("protocol") == protocol and row.get("threshold_variant") == threshold_variant
    ]
    if set(ids) != expected_ids:
        missing = sorted(expected_ids - set(ids))
        extra = sorted(set(ids) - expected_ids)
        raise ValueError(f"{protocol}/{threshold_variant} OOF mismatch. missing={missing[:5]} extra={extra[:5]}")
    duplicates = [pid for pid, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"{protocol}/{threshold_variant} duplicate OOF predictions: {duplicates[:5]}")


def _metrics_for_arrays(labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    out.update(probability_metrics(labels, probabilities))
    predictions = (np.asarray(probabilities, dtype=float) >= np.asarray(thresholds, dtype=float)).astype(int)
    out.update(threshold_metrics_from_counts(confusion_counts(labels, predictions)))
    if len(set(np.round(thresholds.astype(float), 12))) == 1:
        out["threshold"] = float(thresholds[0])
    else:
        out["threshold"] = float("nan")
    return out


def metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    labels = np.asarray([int(row["label_binary"]) for row in rows], dtype=int)
    probabilities = np.asarray([float(row["predicted_probability"]) for row in rows], dtype=float)
    thresholds = np.asarray([float(row["decision_threshold"]) for row in rows], dtype=float)
    return _metrics_for_arrays(labels, probabilities, thresholds)


def aggregate_protocol_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["protocol"]), str(row["threshold_variant"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (protocol, threshold_variant), group_rows in sorted(grouped.items()):
        metrics = metrics_for_rows(group_rows)
        first = group_rows[0]
        out.append(
            {
                "protocol": protocol,
                "source_task": first["source_task"],
                "target_task": first["target_task"],
                "threshold_variant": threshold_variant,
                **metrics,
                "participant_count": len(group_rows),
            }
        )
    return out


def fold_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["protocol"]), str(row["threshold_variant"]), str(row["coherent_fold"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (protocol, threshold_variant, fold), group_rows in sorted(grouped.items()):
        metrics = metrics_for_rows(group_rows)
        first = group_rows[0]
        out.append(
            {
                "protocol": protocol,
                "source_task": first["source_task"],
                "target_task": first["target_task"],
                "threshold_variant": threshold_variant,
                "outer_fold": fold,
                **metrics,
                "participant_count": len(group_rows),
            }
        )
    return out
