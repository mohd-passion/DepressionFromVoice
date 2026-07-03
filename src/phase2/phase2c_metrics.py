"""Metric and threshold helpers for Phase 2C domain-shift analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    macro_f1: float
    balanced_accuracy: float
    candidate_count: int


def clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), EPS, 1.0 - EPS)


def logit(probabilities: np.ndarray) -> np.ndarray:
    p = clip_probabilities(probabilities)
    return np.log(p / (1.0 - p))


def threshold_candidates(probabilities: Iterable[float], extra: Iterable[float] = ()) -> list[float]:
    values = sorted({float(p) for p in probabilities if np.isfinite(float(p))})
    candidates = {0.5}
    for value in extra:
        if np.isfinite(float(value)):
            candidates.add(float(value))
    candidates.update(values)
    for left, right in zip(values, values[1:]):
        candidates.add((left + right) / 2.0)
    return sorted(candidates)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    return {
        "tn": int(np.sum((y == 0) & (pred == 0))),
        "fp": int(np.sum((y == 0) & (pred == 1))),
        "fn": int(np.sum((y == 1) & (pred == 0))),
        "tp": int(np.sum((y == 1) & (pred == 1))),
    }


def threshold_metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    tp = counts["tp"]
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    negative_predictive_value = tn / (tn + fn) if (tn + fn) else 0.0
    f1_pos = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0
    precision_neg = tn / (tn + fn) if (tn + fn) else 0.0
    recall_neg = specificity
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if (precision_neg + recall_neg) else 0.0
    macro_f1 = (f1_pos + f1_neg) / 2.0
    balanced_accuracy = (sensitivity + specificity) / 2.0
    out: dict[str, float | int] = {
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "positive_class_precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "negative_predictive_value": negative_predictive_value,
        "predicted_positive_rate": (tp + fp) / max(tn + fp + fn + tp, 1),
    }
    out.update(counts)
    return out


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (np.asarray(probabilities, dtype=float) >= float(threshold)).astype(int)
    return threshold_metrics_from_counts(confusion_counts(y_true, pred))


def select_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    extra_thresholds: Iterable[float] = (),
) -> ThresholdSelection:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    candidates = threshold_candidates(p, extra_thresholds)
    best: ThresholdSelection | None = None
    for threshold in candidates:
        metrics = threshold_metrics(y, p, threshold)
        current = ThresholdSelection(
            threshold=float(threshold),
            macro_f1=float(metrics["macro_f1"]),
            balanced_accuracy=float(metrics["balanced_accuracy"]),
            candidate_count=len(candidates),
        )
        if best is None:
            best = current
            continue
        key = (
            current.macro_f1,
            current.balanced_accuracy,
            -abs(current.threshold - 0.5),
            -current.threshold,
        )
        best_key = (
            best.macro_f1,
            best.balanced_accuracy,
            -abs(best.threshold - 0.5),
            -best.threshold,
        )
        if key > best_key:
            best = current
    if best is None:
        raise ValueError("No threshold candidates were available.")
    return best


def require_sklearn_metrics() -> dict[str, Any]:
    try:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is required for Phase 2C metric evaluation.") from exc
    return {
        "average_precision_score": average_precision_score,
        "brier_score_loss": brier_score_loss,
        "roc_auc_score": roc_auc_score,
    }


def probability_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    metrics = require_sklearn_metrics()
    y = np.asarray(y_true, dtype=int)
    p = clip_probabilities(probabilities)
    out = {
        "roc_auc": float(metrics["roc_auc_score"](y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "pr_auc": float(metrics["average_precision_score"](y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "brier_score": float(metrics["brier_score_loss"](y, p)),
        "probability_min": float(np.min(p)),
        "probability_max": float(np.max(p)),
        "probability_mean": float(np.mean(p)),
        "probability_std": float(np.std(p, ddof=0)),
    }
    return out


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    out.update(probability_metrics(y_true, probabilities))
    out.update(threshold_metrics(y_true, probabilities, threshold))
    out["threshold"] = float(threshold)
    return out


def calibration_slope_intercept(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float | str]:
    try:
        from sklearn.linear_model import LogisticRegression
    except ModuleNotFoundError:
        return {"calibration_intercept": float("nan"), "calibration_slope": float("nan"), "calibration_fit_status": "sklearn_unavailable"}
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return {"calibration_intercept": float("nan"), "calibration_slope": float("nan"), "calibration_fit_status": "single_class"}
    x = logit(probabilities).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)
    model.fit(x, y)
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0, 0]),
        "calibration_fit_status": "success",
    }


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, float | int]]]:
    y = np.asarray(y_true, dtype=int)
    p = clip_probabilities(probabilities)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    ece = 0.0
    n = len(y)
    for idx in range(n_bins):
        left = bins[idx]
        right = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        count = int(np.sum(mask))
        if count:
            mean_pred = float(np.mean(p[mask]))
            observed = float(np.mean(y[mask]))
            abs_gap = abs(mean_pred - observed)
            ece += (count / n) * abs_gap
        else:
            mean_pred = float("nan")
            observed = float("nan")
            abs_gap = float("nan")
        rows.append(
            {
                "bin_index": idx,
                "bin_left": float(left),
                "bin_right": float(right),
                "count": count,
                "mean_predicted_probability": mean_pred,
                "observed_positive_rate": observed,
                "absolute_gap": abs_gap,
            }
        )
    return float(ece), rows


def assert_metric_range(name: str, value: float) -> None:
    if not np.isfinite(value) or not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} is outside [0, 1]: {value}")
