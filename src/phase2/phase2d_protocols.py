"""Protocol helpers for Phase 2D Android reading-speech evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from .phase2c_metrics import select_threshold
except ImportError:  # pragma: no cover - allows direct script execution
    from phase2c_metrics import select_threshold


FROZEN_DAIC_THRESHOLD = 0.555840134621
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


@dataclass(frozen=True)
class ReadingFoldSummary:
    fold: str
    participant_count: int
    hc_count: int
    pt_count: int
    outer_training_count: int
    outer_test_count: int


def assert_reading_fold_integrity(
    participant_ids: list[str],
    folds: list[str],
    labels: np.ndarray,
    tasks: list[str] | None = None,
) -> list[ReadingFoldSummary]:
    if len(participant_ids) != len(set(participant_ids)):
        duplicates = [pid for pid, count in Counter(participant_ids).items() if count > 1]
        raise ValueError(f"Duplicate reading participants detected: {duplicates[:5]}")
    if len(participant_ids) != len(folds) or len(participant_ids) != len(labels):
        raise ValueError("Participant, fold and label arrays have inconsistent lengths.")
    if tasks is not None:
        bad_tasks = sorted({task for task in tasks if task != "reading"})
        if bad_tasks:
            raise ValueError(f"Non-reading tasks detected in reading protocol: {bad_tasks}")
    fold_values = sorted(set(folds))
    if len(fold_values) != 5:
        raise ValueError(f"Expected exactly five reading folds; found {fold_values}")
    y = np.asarray(labels, dtype=int)
    fold_array = np.asarray(folds)
    summaries: list[ReadingFoldSummary] = []
    for fold in fold_values:
        test_mask = fold_array == fold
        train_mask = ~test_mask
        summaries.append(
            ReadingFoldSummary(
                fold=fold,
                participant_count=int(np.sum(test_mask)),
                hc_count=int(np.sum(y[test_mask] == 0)),
                pt_count=int(np.sum(y[test_mask] == 1)),
                outer_training_count=int(np.sum(train_mask)),
                outer_test_count=int(np.sum(test_mask)),
            )
        )
        if summaries[-1].hc_count == 0 or summaries[-1].pt_count == 0:
            raise ValueError(f"Reading fold {fold} does not contain both classes.")
    if sum(item.outer_test_count for item in summaries) != len(participant_ids):
        raise ValueError("Every reading participant must appear in exactly one outer test fold.")
    return summaries


def assert_unique_oof_predictions(rows: list[dict[str, Any]], expected_ids: set[str], protocol: str) -> None:
    ids = [str(row["participant_id"]) for row in rows if row.get("protocol") == protocol]
    if set(ids) != expected_ids:
        missing = sorted(expected_ids - set(ids))
        extra = sorted(set(ids) - expected_ids)
        raise ValueError(f"{protocol} OOF ID mismatch. missing={missing[:5]} extra={extra[:5]}")
    duplicates = [pid for pid, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"{protocol} duplicate OOF predictions: {duplicates[:5]}")


def base_prediction_row(
    participant_id: str,
    fold: str,
    label: int,
    probability: float,
    protocol: str,
    threshold: float,
    threshold_source: str,
    selected_c: float | None = None,
) -> dict[str, Any]:
    return {
        "participant_id": participant_id,
        "reading_fold": fold,
        "label_binary": int(label),
        "protocol": protocol,
        "predicted_probability": float(probability),
        "decision_threshold": float(threshold),
        "threshold_source": threshold_source,
        "prediction": int(float(probability) >= float(threshold)),
        "selected_C": selected_c,
    }


def protocol_a_zero_shot_rows(
    participant_ids: list[str],
    folds: list[str],
    labels: np.ndarray,
    frozen_probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pid, fold, label, prob in zip(participant_ids, folds, labels, frozen_probabilities):
        rows.append(base_prediction_row(pid, fold, int(label), float(prob), "A_zero_shot_daic_at_0_5", 0.5, "default_0_5"))
        rows.append(
            base_prediction_row(
                pid,
                fold,
                int(label),
                float(prob),
                "A_zero_shot_daic_at_frozen_threshold",
                FROZEN_DAIC_THRESHOLD,
                "frozen_daic_dev_threshold",
            )
        )
    return rows


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


def select_c_leave_one_fold_out(x_train: np.ndarray, y_train: np.ndarray, train_folds: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows: list[dict[str, Any]] = []
    best_c: float | None = None
    best_key: tuple[float, float, float] | None = None
    for c_value in C_GRID:
        aucs: list[float] = []
        pr_aucs: list[float] = []
        for validation_fold in sorted(np.unique(train_folds)):
            inner_train = train_folds != validation_fold
            inner_validation = train_folds == validation_fold
            if len(np.unique(y_train[inner_validation])) < 2:
                raise ValueError(f"Inner reading validation fold {validation_fold} has one class.")
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
        raise ValueError("No C candidate was selected for reading protocol.")
    return best_c, rows


def inner_oof_probabilities(x_train: np.ndarray, y_train: np.ndarray, train_folds: np.ndarray, c_value: float) -> np.ndarray:
    oof = np.full(len(y_train), np.nan, dtype=float)
    for validation_fold in sorted(np.unique(train_folds)):
        inner_train = train_folds != validation_fold
        inner_validation = train_folds == validation_fold
        model = make_logistic_pipeline(c_value)
        model.fit(x_train[inner_train], y_train[inner_train])
        oof[inner_validation] = model.predict_proba(x_train[inner_validation])[:, 1]
    if np.any(~np.isfinite(oof)):
        raise ValueError("Reading inner OOF probabilities are incomplete.")
    return oof


def protocol_b_android_reading_logistic_regression(
    participant_ids: list[str],
    folds: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    y = np.asarray(labels, dtype=int)
    x = np.asarray(embeddings, dtype=float)
    fold_array = np.asarray(folds)
    rows: list[dict[str, Any]] = []
    params: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []

    for fold in sorted(set(folds)):
        train_mask = fold_array != fold
        test_mask = fold_array == fold
        selected_c, c_rows = select_c_leave_one_fold_out(x[train_mask], y[train_mask], fold_array[train_mask])
        for c_row in c_rows:
            row = dict(c_row)
            row.update({"protocol": "B_android_reading_logistic_regression", "outer_fold": fold})
            inner_rows.append(row)

        inner_oof = inner_oof_probabilities(x[train_mask], y[train_mask], fold_array[train_mask], selected_c)
        threshold_selection = select_threshold(y[train_mask], inner_oof)
        model = make_logistic_pipeline(selected_c)
        model.fit(x[train_mask], y[train_mask])
        test_prob = model.predict_proba(x[test_mask])[:, 1]
        params.append(
            {
                "protocol": "B_android_reading_logistic_regression",
                "outer_fold": fold,
                "selected_C": selected_c,
                "selected_threshold": threshold_selection.threshold,
                "threshold_macro_f1_train": threshold_selection.macro_f1,
                "threshold_balanced_accuracy_train": threshold_selection.balanced_accuracy,
                "threshold_candidate_count": threshold_selection.candidate_count,
            }
        )
        for idx, prob in zip(np.where(test_mask)[0], test_prob):
            rows.append(
                base_prediction_row(
                    participant_ids[idx],
                    folds[idx],
                    int(y[idx]),
                    float(prob),
                    "B_android_lr_at_0_5",
                    0.5,
                    "default_0_5",
                    selected_c=selected_c,
                )
            )
            rows.append(
                base_prediction_row(
                    participant_ids[idx],
                    folds[idx],
                    int(y[idx]),
                    float(prob),
                    "B_android_lr_threshold_adapted",
                    threshold_selection.threshold,
                    "outer_training_inner_oof_macro_f1",
                    selected_c=selected_c,
                )
            )

    expected_ids = set(participant_ids)
    assert_unique_oof_predictions(rows, expected_ids, "B_android_lr_at_0_5")
    assert_unique_oof_predictions(rows, expected_ids, "B_android_lr_threshold_adapted")
    return rows, params, inner_rows
