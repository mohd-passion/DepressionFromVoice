"""Protocol implementations for Phase 2C DAIC-to-Android diagnosis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from .phase2c_calibration import select_calibration_method
    from .phase2c_metrics import evaluate_predictions, select_threshold, threshold_metrics
except ImportError:  # pragma: no cover - allows direct script execution
    from phase2c_calibration import select_calibration_method
    from phase2c_metrics import evaluate_predictions, select_threshold, threshold_metrics


FROZEN_DAIC_THRESHOLD = 0.555840134621
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


@dataclass(frozen=True)
class FoldSummary:
    fold: str
    participant_count: int
    hc_count: int
    pt_count: int
    outer_training_count: int
    outer_test_count: int


def assert_fold_integrity(participant_ids: list[str], folds: list[str], labels: np.ndarray) -> list[FoldSummary]:
    if len(participant_ids) != len(set(participant_ids)):
        duplicates = [pid for pid, count in Counter(participant_ids).items() if count > 1]
        raise ValueError(f"Duplicate participants detected: {duplicates[:5]}")
    if len(participant_ids) != len(folds) or len(participant_ids) != len(labels):
        raise ValueError("Participant, fold and label arrays have inconsistent lengths.")
    fold_values = sorted(set(folds))
    if len(fold_values) != 5:
        raise ValueError(f"Expected exactly five interview folds; found {fold_values}")
    summaries: list[FoldSummary] = []
    y = np.asarray(labels, dtype=int)
    fold_array = np.asarray(folds)
    for fold in fold_values:
        test_mask = fold_array == fold
        train_mask = ~test_mask
        if np.any(test_mask & train_mask):
            raise ValueError(f"Fold leakage detected for {fold}")
        summaries.append(
            FoldSummary(
                fold=fold,
                participant_count=int(np.sum(test_mask)),
                hc_count=int(np.sum(y[test_mask] == 0)),
                pt_count=int(np.sum(y[test_mask] == 1)),
                outer_training_count=int(np.sum(train_mask)),
                outer_test_count=int(np.sum(test_mask)),
            )
        )
    if sum(item.outer_test_count for item in summaries) != len(participant_ids):
        raise ValueError("Every participant must appear in exactly one outer test fold.")
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


def _base_prediction_row(
    participant_id: str,
    fold: str,
    label: int,
    probability: float,
    protocol: str,
    threshold: float,
    threshold_source: str,
    selected_c: float | None = None,
    calibration_method: str = "",
) -> dict[str, Any]:
    return {
        "participant_id": participant_id,
        "interview_fold": fold,
        "label_binary": int(label),
        "protocol": protocol,
        "predicted_probability": float(probability),
        "decision_threshold": float(threshold),
        "threshold_source": threshold_source,
        "prediction": int(float(probability) >= float(threshold)),
        "selected_C": selected_c,
        "calibration_method": calibration_method,
    }


def protocol_a_rows(
    participant_ids: list[str],
    folds: list[str],
    labels: np.ndarray,
    frozen_probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pid, fold, label, prob in zip(participant_ids, folds, labels, frozen_probabilities):
        rows.append(_base_prediction_row(pid, fold, int(label), float(prob), "A_frozen_daic_at_0_5", 0.5, "default_0_5"))
        rows.append(
            _base_prediction_row(
                pid,
                fold,
                int(label),
                float(prob),
                "A_frozen_daic_at_frozen_threshold",
                FROZEN_DAIC_THRESHOLD,
                "frozen_daic_dev_threshold",
            )
        )
    return rows


def protocol_b_threshold_adaptation(
    participant_ids: list[str],
    folds: list[str],
    labels: np.ndarray,
    frozen_probabilities: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(frozen_probabilities, dtype=float)
    fold_array = np.asarray(folds)
    rows: list[dict[str, Any]] = []
    params: list[dict[str, Any]] = []
    for fold in sorted(set(folds)):
        train_mask = fold_array != fold
        test_mask = fold_array == fold
        selection = select_threshold(y[train_mask], p[train_mask], extra_thresholds=[FROZEN_DAIC_THRESHOLD])
        params.append(
            {
                "protocol": "B_threshold_only",
                "outer_fold": fold,
                "selected_threshold": selection.threshold,
                "threshold_macro_f1_train": selection.macro_f1,
                "threshold_balanced_accuracy_train": selection.balanced_accuracy,
                "threshold_candidate_count": selection.candidate_count,
                "selected_C": "",
                "calibration_method": "",
            }
        )
        for idx in np.where(test_mask)[0]:
            rows.append(
                _base_prediction_row(
                    participant_ids[idx],
                    folds[idx],
                    int(y[idx]),
                    float(p[idx]),
                    "B_threshold_only",
                    selection.threshold,
                    "outer_training_fold_macro_f1",
                )
            )
    assert_unique_oof_predictions(rows, set(participant_ids), "B_threshold_only")
    return rows, params


def protocol_c_calibration(
    participant_ids: list[str],
    folds: list[str],
    labels: np.ndarray,
    frozen_probabilities: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(frozen_probabilities, dtype=float)
    fold_array = np.asarray(folds)
    rows: list[dict[str, Any]] = []
    params: list[dict[str, Any]] = []
    method_scores: list[dict[str, Any]] = []
    for fold in sorted(set(folds)):
        train_mask = fold_array != fold
        test_mask = fold_array == fold
        result = select_calibration_method(fold_array[train_mask], y[train_mask], p[train_mask])
        selected = dict(result["selected"])
        calibrator = result["calibrator"]
        method = str(selected["method"])
        for candidate in result["rows"]:
            candidate_row = dict(candidate)
            candidate_row.update({"protocol": "C_frozen_score_calibration", "outer_fold": fold})
            method_scores.append(candidate_row)
        params.append(
            {
                "protocol": "C_frozen_score_calibration",
                "outer_fold": fold,
                "selected_threshold": 0.5,
                "selected_C": "",
                "calibration_method": method,
                "inner_mean_brier": selected["mean_brier"],
                "inner_std_brier": selected["std_brier"],
                "inner_mean_macro_f1_at_0_5": selected["mean_macro_f1_at_0_5"],
            }
        )
        calibrated = np.asarray(calibrator.predict(p[test_mask]), dtype=float)
        for idx, prob in zip(np.where(test_mask)[0], calibrated):
            rows.append(
                _base_prediction_row(
                    participant_ids[idx],
                    folds[idx],
                    int(y[idx]),
                    float(prob),
                    "C_frozen_score_calibration",
                    0.5,
                    "calibrated_probability_default_0_5",
                    calibration_method=method,
                )
            )
    assert_unique_oof_predictions(rows, set(participant_ids), "C_frozen_score_calibration")
    return rows, params, method_scores


def _make_pipeline(c_value: float) -> Any:
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


def _select_c_with_inner_folds(x_train: np.ndarray, y_train: np.ndarray, train_folds: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
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
            model = _make_pipeline(c_value)
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


def _inner_oof_probabilities(x_train: np.ndarray, y_train: np.ndarray, train_folds: np.ndarray, c_value: float) -> np.ndarray:
    oof = np.full(len(y_train), np.nan, dtype=float)
    for validation_fold in sorted(np.unique(train_folds)):
        inner_train = train_folds != validation_fold
        inner_validation = train_folds == validation_fold
        model = _make_pipeline(c_value)
        model.fit(x_train[inner_train], y_train[inner_train])
        oof[inner_validation] = model.predict_proba(x_train[inner_validation])[:, 1]
    if np.any(~np.isfinite(oof)):
        raise ValueError("Inner OOF probabilities are incomplete.")
    return oof


def protocol_d_android_logistic_regression(
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
        selected_c, c_rows = _select_c_with_inner_folds(x[train_mask], y[train_mask], fold_array[train_mask])
        for c_row in c_rows:
            row = dict(c_row)
            row.update({"protocol": "D_android_trained_logistic_regression", "outer_fold": fold})
            inner_rows.append(row)
        inner_oof = _inner_oof_probabilities(x[train_mask], y[train_mask], fold_array[train_mask], selected_c)
        threshold = select_threshold(y[train_mask], inner_oof).threshold
        model = _make_pipeline(selected_c)
        model.fit(x[train_mask], y[train_mask])
        test_prob = model.predict_proba(x[test_mask])[:, 1]
        params.append(
            {
                "protocol": "D_android_trained_logistic_regression",
                "outer_fold": fold,
                "selected_threshold": threshold,
                "selected_C": selected_c,
                "calibration_method": "",
            }
        )
        for idx, prob in zip(np.where(test_mask)[0], test_prob):
            rows.append(
                _base_prediction_row(
                    participant_ids[idx],
                    folds[idx],
                    int(y[idx]),
                    float(prob),
                    "D_android_lr_at_0_5",
                    0.5,
                    "default_0_5",
                    selected_c=selected_c,
                )
            )
            rows.append(
                _base_prediction_row(
                    participant_ids[idx],
                    folds[idx],
                    int(y[idx]),
                    float(prob),
                    "D_android_lr_threshold_adapted",
                    threshold,
                    "outer_training_inner_oof_macro_f1",
                    selected_c=selected_c,
                )
            )
    assert_unique_oof_predictions(rows, set(participant_ids), "D_android_lr_at_0_5")
    assert_unique_oof_predictions(rows, set(participant_ids), "D_android_lr_threshold_adapted")
    return rows, params, inner_rows


def aggregate_protocol_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics_rows: list[dict[str, Any]] = []
    by_protocol: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_protocol.setdefault(str(row["protocol"]), []).append(row)
    for protocol, protocol_rows in sorted(by_protocol.items()):
        labels = np.asarray([int(row["label_binary"]) for row in protocol_rows], dtype=int)
        probabilities = np.asarray([float(row["predicted_probability"]) for row in protocol_rows], dtype=float)
        thresholds = np.asarray([float(row["decision_threshold"]) for row in protocol_rows], dtype=float)
        if len(set(np.round(thresholds, 12))) == 1:
            metrics = evaluate_predictions(labels, probabilities, float(thresholds[0]))
        else:
            predictions = (probabilities >= thresholds).astype(int)
            threshold_proxy = float("nan")
            metrics = {}
            try:
                from .phase2c_metrics import probability_metrics, threshold_metrics_from_counts, confusion_counts
            except ImportError:  # pragma: no cover
                from phase2c_metrics import probability_metrics, threshold_metrics_from_counts, confusion_counts

            metrics.update(probability_metrics(labels, probabilities))
            metrics.update(threshold_metrics_from_counts(confusion_counts(labels, predictions)))
            metrics["threshold"] = threshold_proxy
        metrics_rows.append({"protocol": protocol, **metrics, "participant_count": len(protocol_rows)})
    return metrics_rows
